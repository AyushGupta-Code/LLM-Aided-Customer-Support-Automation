"""
Train five text classification models on manually labeled tweets (no Gemini).

The script loads `data/manual_labels.csv` as ground truth, preprocesses the
text, splits into train/test sets, and trains the following TF-IDF pipelines:
  1) Logistic Regression (multinomial)
  2) Linear SVM (LinearSVC)
  3) SGDClassifier (logistic loss)
  4) SGDClassifier (hinge/SVM-style)
  5) Multinomial Naive Bayes

Results are printed and saved to `results_summary.csv`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Ensure imports work regardless of CWD (e.g., running from src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, evaluation, preprocessing


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data loading & preprocessing
# -----------------------------------------------------------------------------
def load_manual_labels(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load manually labeled tweets and apply preprocessing.
    """
    data_path = Path(path) if path else config.MANUAL_LABELS_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"Manual labels not found at {data_path}. "
            "Ensure data/manual_labels.csv is present."
        )

    df = pd.read_csv(data_path)
    required_cols = {"text", "intent", "severity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in manual_labels.csv: {missing}")

    df = df.dropna(subset=["text", "intent", "severity"]).copy()
    df["severity"] = df["severity"].astype(int)

    df = preprocessing.preprocess_dataframe(df, text_column="text")
    df["text"] = df["text_processed"]
    df = df.drop(columns=["text_processed"])

    # Filter empty/whitespace-only texts post-preprocessing
    df = df[df["text"].astype(str).str.strip().str.len() > 0].copy()
    logger.info("Loaded %d labeled tweets from %s", len(df), data_path)
    return df


def _safe_train_test_split(
    X,
    y,
    label_name: str,
    test_size: float,
    random_state: int,
):
    """
    Stratify when possible, otherwise fall back gracefully.
    """
    y_series = pd.Series(y)
    class_counts = y_series.value_counts()
    n_classes = len(class_counts)
    n_samples = len(y)

    can_stratify = class_counts.min() >= 2
    n_test = int(test_size * n_samples) if isinstance(test_size, float) else int(test_size)

    stratify = None
    if can_stratify and n_test >= n_classes:
        stratify = y
    else:
        logger.warning(
            "[%s] Not stratifying (classes=%s, min_count=%s, n_test=%s)",
            label_name,
            n_classes,
            class_counts.min(),
            n_test,
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def _tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )


def build_model_registry() -> Dict[str, Pipeline]:
    """
    Create the 5 TF-IDF model pipelines.
    """
    return {
        "log_reg": Pipeline(
            [
                ("tfidf", _tfidf_vectorizer()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
                        multi_class="multinomial",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "linear_svc": Pipeline(
            [
                ("tfidf", _tfidf_vectorizer()),
                ("clf", LinearSVC(random_state=config.RANDOM_STATE)),
            ]
        ),
        "sgd_log": Pipeline(
            [
                ("tfidf", _tfidf_vectorizer()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        max_iter=2000,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "sgd_hinge": Pipeline(
            [
                ("tfidf", _tfidf_vectorizer()),
                (
                    "clf",
                    SGDClassifier(
                        loss="hinge",
                        max_iter=2000,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "multinomial_nb": Pipeline(
            [
                ("tfidf", _tfidf_vectorizer()),
                ("clf", MultinomialNB(alpha=0.5)),
            ]
        ),
    }


def train_and_evaluate_models(
    labeled_df: pd.DataFrame, target_col: str
) -> Dict[str, Dict[str, float]]:
    """
    Train/evaluate all registered models for a given target column.
    """
    X = labeled_df["text"].values
    y = labeled_df[target_col].values

    X_train, X_test, y_train, y_test = _safe_train_test_split(
        X,
        y,
        label_name=target_col,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    results: Dict[str, Dict[str, float]] = {}
    registry = build_model_registry()
    for name, pipeline in registry.items():
        logger.info("Training %s model: %s", target_col, name)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)
            except Exception as exc:  # pragma: no cover - best-effort only
                logger.warning("No probabilities for %s: %s", name, exc)

        evaluator = evaluation.ModelEvaluator(task_type=f"{target_col} ({name})")
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)
        results[name] = metrics
        evaluator.print_metrics_summary()

    return results


def save_results(intent_results: Dict[str, Dict[str, float]], severity_results: Dict[str, Dict[str, float]]) -> Path:
    """
    Persist a combined metrics table to CSV for quick inspection.
    """
    flat_records = []
    for model_name, metrics in intent_results.items():
        flat_records.append({"task": "intent", "model": model_name, **metrics})
    for model_name, metrics in severity_results.items():
        flat_records.append({"task": "severity", "model": model_name, **metrics})

    df = pd.DataFrame(flat_records)
    out_path = config.PROJECT_ROOT / "results_summary.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved summary metrics to %s", out_path)
    return out_path


def main():
    labeled_df = load_manual_labels()
    logger.info("Starting training on manual labels (no Gemini).")

    intent_results = train_and_evaluate_models(labeled_df, target_col="intent")
    severity_results = train_and_evaluate_models(
        labeled_df, target_col="severity"
    )

    save_results(intent_results, severity_results)


if __name__ == "__main__":
    main()
