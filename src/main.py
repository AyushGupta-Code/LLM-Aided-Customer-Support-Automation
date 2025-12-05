"""
Train five text classification models on manually labeled tweets (no Gemini).

The script loads `data/manual_labels.csv` as ground truth, preprocesses the
text, splits into train/test sets, and trains the following models:
  1) Logistic Regression (multinomial, TF-IDF)
  2) SGDClassifier (logistic loss, TF-IDF)
  3) Zero-shot classifier (NLI model)
  4) RNN + Bidirectional LSTM
  5) Multinomial Naive Bayes (TF-IDF)

Results are printed and saved to `results_summary.csv`.

After training:
  - It selects the best overall model (by f1_macro averaged over intent + severity).
  - Randomly picks one tweet from the dataset.
  - Predicts intent & severity for that tweet with the best model.
  - If Gemini is available, generates a reply to that tweet.
  - Prints: the tweet being answered, the answer, and the model used.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import sys
import os
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Ensure imports work regardless of CWD (e.g., running from src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, evaluation, preprocessing

# -----------------------------------------------------------------------------
# Environment configuration (GPU-friendly)
# -----------------------------------------------------------------------------
# These only affect Hugging Face Transformers, keeping them on PyTorch.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")

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

    Keeps both:
      - text_raw: original tweet text
      - text: preprocessed text for modeling
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

    # Preserve original text
    df["text_raw"] = df["text"].astype(str)

    # Preprocess for modeling
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
# Models (classical + RNN)
# -----------------------------------------------------------------------------
def _tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )


class ZeroShotModel:
    """
    Zero-shot classifier using a Hugging Face NLI model (e.g., bart-large-mnli).

    Uses PyTorch and will run on GPU (device=0) if available, otherwise CPU.
    """

    def __init__(self, task_type: str):
        # Keep Transformers on PyTorch backend (not TensorFlow).
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("USE_TF", "0")

        from transformers import pipeline

        # Try to use GPU if torch + CUDA are available
        try:
            import torch

            device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device = -1

        self.task_type = task_type
        self.model_name = config.ZERO_SHOT_MODEL_NAME
        self.pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            framework="pt",
            device=device,  # 0 for GPU if available, else -1 for CPU
            batch_size=getattr(config, "ZERO_SHOT_BATCH_SIZE", 8),
            truncation=True,
            max_length=getattr(config, "ZERO_SHOT_MAX_LENGTH", 128),
        )
        self.label_map: Dict[str, object] = {}
        self.candidate_labels: list[str] = []

        logger.info(
            "Initialized ZeroShotModel for task '%s' using device=%s",
            self.task_type,
            device,
        )

    def fit(self, X, y):
        unique_labels = sorted(set(y))
        self.label_map = {str(lbl): lbl for lbl in unique_labels}
        self.candidate_labels = list(self.label_map.keys())

    def predict(self, X):
        results = self.pipeline(
            list(X),
            candidate_labels=self.candidate_labels,
            multi_label=False,
        )
        preds = []
        for res in results:
            label_str = res["labels"][0]
            preds.append(self.label_map.get(label_str, label_str))
        return np.array(preds)

    def predict_proba(self, X):
        results = self.pipeline(
            list(X),
            candidate_labels=self.candidate_labels,
            multi_label=False,
        )
        prob_list = []
        for res in results:
            score_by_label = {
                lbl: score for lbl, score in zip(res["labels"], res["scores"])
            }
            probs = [score_by_label.get(lbl, 0.0) for lbl in self.candidate_labels]
            prob_list.append(probs)
        return np.array(prob_list)


class RnnLstmTextClassifier:
    """
    Thin wrapper around Keras model + tokenizer + label encoder
    to expose a scikit-like predict / predict_proba API for new texts.
    """

    def __init__(self, tokenizer, label_encoder, model, max_seq_len: int, num_classes: int):
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.model = model
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

        from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
        self._pad_sequences = pad_sequences

    def _texts_to_seq(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        return self._pad_sequences(
            seqs,
            maxlen=self.max_seq_len,
            padding="post",
            truncating="post",
        )

    def predict_proba(self, X):
        X_seq = self._texts_to_seq(list(X))
        proba = self.model.predict(X_seq, verbose=0)
        if self.num_classes == 2:
            scores = proba.flatten()
            proba_full = np.vstack([1 - scores, scores]).T
            return proba_full
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.num_classes == 2:
            scores = proba[:, 1]
            y_enc = (scores >= 0.5).astype(int)
        else:
            y_enc = proba.argmax(axis=1)
        labels = self.label_encoder.inverse_transform(y_enc)
        return np.array(labels)


def build_model_registry() -> Dict[str, Any]:
    """
    Create the 5 model definitions:
      - 3 TF-IDF + linear models
      - 1 Zero-shot classifier (NLI)
      - 1 RNN+LSTM classifier (trained separately)
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
        "zero_shot": ZeroShotModel(task_type="intent"),
        "rnn_lstm": None,  # placeholder, trained separately
        "multinomial_nb": Pipeline(
            [
                ("tfidf", _tfidf_vectorizer()),
                ("clf", MultinomialNB(alpha=0.5)),
            ]
        ),
    }


def _train_rnn_lstm(
    X_train,
    y_train,
    X_test,
    y_test,
    task_name: str,
) -> Tuple[Dict[str, float], RnnLstmTextClassifier]:
    """
    Train a simple RNN+Bidirectional LSTM classifier.

    Will use GPU if available (no manual disabling).
    Returns metrics and a RnnLstmTextClassifier object.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for the RNN+LSTM model. "
            "Install it via `pip install tensorflow[and-cuda]`."
        ) from exc

    # Try to enable GPU + memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Using GPU(s) for RNN+LSTM: %s", gpus)
        except Exception as e:  # pragma: no cover - best-effort only
            logger.warning("Could not set memory growth on GPUs: %s", e)
    else:
        logger.warning("No GPU detected by TensorFlow, running RNN+LSTM on CPU.")

    # Tokenize
    tokenizer = Tokenizer(num_words=config.RNN_MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_train),
        maxlen=config.MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_test),
        maxlen=config.MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(list(y_train) + list(y_test))
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    num_classes = len(set(y_train_enc))
    activation = "sigmoid" if num_classes == 2 else "softmax"
    loss = "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"

    model = models.Sequential(
        [
            layers.Embedding(
                input_dim=config.RNN_MAX_VOCAB_SIZE,
                output_dim=config.EMBEDDING_DIM,
                input_length=config.MAX_SEQUENCE_LENGTH,
            ),
            layers.Bidirectional(layers.LSTM(config.LSTM_UNITS, return_sequences=True)),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Bidirectional(layers.LSTM(config.LSTM_UNITS)),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(64, activation="relu"),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(num_classes if num_classes > 2 else 1, activation=activation),
        ]
    )

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train_seq,
        y_train_enc,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=0,
    )

    # Predictions for metrics
    proba = model.predict(X_test_seq, verbose=0)
    if num_classes == 2:
        scores = proba.flatten()
        y_pred_enc = (scores >= 0.5).astype(int)
        y_proba = np.vstack([1 - scores, scores]).T
    else:
        y_pred_enc = proba.argmax(axis=1)
        y_proba = proba

    y_pred = label_encoder.inverse_transform(y_pred_enc)

    evaluator = evaluation.ModelEvaluator(task_type=f"{task_name} (rnn_lstm)")
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)
    evaluator.print_metrics_summary()

    clf = RnnLstmTextClassifier(
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        model=model,
        max_seq_len=config.MAX_SEQUENCE_LENGTH,
        num_classes=num_classes,
    )
    return metrics, clf


def train_and_evaluate_models(
    labeled_df: pd.DataFrame,
    target_col: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Train/evaluate all registered models for a given target column.

    Returns:
        results:  model_name -> metrics dict
        models:   model_name -> fitted model object (Pipeline, ZeroShotModel, RnnLstmTextClassifier)
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
    models: Dict[str, Any] = {}

    registry = build_model_registry()
    for name, pipeline_obj in registry.items():
        logger.info("Training %s model: %s", target_col, name)

        if name == "rnn_lstm":
            metrics, clf = _train_rnn_lstm(X_train, y_train, X_test, y_test, target_col)
            results[name] = metrics
            models[name] = clf
            continue

        if name == "zero_shot":
            zs_model = ZeroShotModel(task_type=target_col)
            zs_model.fit(X_train, y_train)
            y_pred = zs_model.predict(X_test)
            y_proba = zs_model.predict_proba(X_test)
            evaluator = evaluation.ModelEvaluator(task_type=f"{target_col} ({name})")
            metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)
            results[name] = metrics
            models[name] = zs_model
            evaluator.print_metrics_summary()
            continue

        # Classical models
        pipeline_obj.fit(X_train, y_train)
        y_pred = pipeline_obj.predict(X_test)

        y_proba = None
        if hasattr(pipeline_obj, "predict_proba"):
            try:
                y_proba = pipeline_obj.predict_proba(X_test)
            except Exception as exc:  # pragma: no cover - best-effort only
                logger.warning("No probabilities for %s: %s", name, exc)

        evaluator = evaluation.ModelEvaluator(task_type=f"{target_col} ({name})")
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)
        results[name] = metrics
        models[name] = pipeline_obj
        evaluator.print_metrics_summary()

    return results, models


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


# -----------------------------------------------------------------------------
# Best model selection + Gemini answering
# -----------------------------------------------------------------------------
def select_best_overall_model(
    intent_results: Dict[str, Dict[str, float]],
    severity_results: Dict[str, Dict[str, float]],
    metric: str = "f1_macro",
) -> str:
    """
    Pick the single best model across the 5 by averaging a given metric
    (e.g., f1_macro) over intent + severity tasks.
    """
    scores = {}
    for model_name in intent_results.keys():
        intent_score = intent_results.get(model_name, {}).get(metric, 0.0)
        severity_score = severity_results.get(model_name, {}).get(metric, 0.0)
        scores[model_name] = (intent_score + severity_score) / 2.0

    best_model = max(scores, key=scores.get)
    logger.info(
        "Best overall model by %s: %s (score=%.4f)",
        metric,
        best_model,
        scores[best_model],
    )
    return best_model


def get_gemini_model():
    """
    Initialize Gemini client if possible.

    Returns a configured GenerativeModel, or None if:
      - API key is missing, or
      - google-generativeai is not installed.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning(
            "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
            "in your environment. Skipping answer generation."
        )
        return None

    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        logger.warning(
            "google-generativeai package not installed. "
            "Install it via `pip install google-generativeai` to enable Gemini answers."
        )
        return None

    genai.configure(api_key=api_key)
    model_name = getattr(config, "GEMINI_MODEL_NAME", "gemini-1.5-pro")
    model = genai.GenerativeModel(model_name)
    logger.info("Initialized Gemini model '%s' for answer generation.", model_name)
    return model


def generate_gemini_answer(gemini_model, tweet: str, intent: str, severity: int) -> str:
    """
    Use a configured Gemini model to generate a customer support reply.
    """
    sev_text = {
        1: "low (non-urgent, general question, casual feedback)",
        2: "medium (inconvenient, but service basically works)",
        3: "high (blocking/critical, user is very frustrated)",
    }.get(int(severity), "unknown")

    prompt = f"""
You are a helpful and empathetic customer support agent.

User tweet:
"{tweet}"

Predicted intent: {intent}
Predicted severity: {severity} - {sev_text}

Write a concise, friendly, and professional reply to this tweet:
- Acknowledge the issue or feedback.
- Be empathetic.
- If it's high severity, show urgency.
- Do NOT mention that an AI model or intent classifier was used.
- Keep it within 2–3 sentences max.
"""

    response = gemini_model.generate_content(prompt)
    text = getattr(response, "text", None)
    if text is None and hasattr(response, "candidates") and response.candidates:
        text = response.candidates[0].content.parts[0].text
    if text is None:
        text = ""
    return text.strip()


def answer_random_tweet(
    labeled_df: pd.DataFrame,
    intent_results: Dict[str, Dict[str, float]],
    severity_results: Dict[str, Dict[str, float]],
    intent_models: Dict[str, Any],
    severity_models: Dict[str, Any],
):
    """
    Randomly pick one tweet from the dataset, classify it with the best model,
    generate a Gemini response (if possible), and print:
      Tweet : ...
      Answer: ...
      Model : ...
    """
    gemini_model = get_gemini_model()
    if gemini_model is None:
        return  # already logged reason; nothing else to do

    best_model_name = select_best_overall_model(intent_results, severity_results, metric="f1_macro")

    intent_model = intent_models[best_model_name]
    severity_model = severity_models[best_model_name]

    # Randomly sample one row
    idx = random.randint(0, len(labeled_df) - 1)
    row = labeled_df.iloc[idx]

    tweet_raw = row.get("text_raw", row["text"])
    tweet_processed = row["text"]

    intent_pred = intent_model.predict([tweet_processed])[0]
    severity_pred = severity_model.predict([tweet_processed])[0]
    severity_pred_int = int(severity_pred)

    answer = generate_gemini_answer(gemini_model, tweet_raw, intent_pred, severity_pred_int)

    # Exactly the 3 fields requested
    print(f"Tweet : {tweet_raw}")
    print(f"Answer: {answer}")
    print(f"Model : {best_model_name}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    labeled_df = load_manual_labels()
    logger.info("Starting training on manual labels (no Gemini).")

    intent_results, intent_models = train_and_evaluate_models(
        labeled_df, target_col="intent"
    )
    severity_results, severity_models = train_and_evaluate_models(
        labeled_df, target_col="severity"
    )

    save_results(intent_results, severity_results)

    # Randomly choose a tweet from the dataset and answer it (if Gemini is available)
    answer_random_tweet(
        labeled_df=labeled_df,
        intent_results=intent_results,
        severity_results=severity_results,
        intent_models=intent_models,
        severity_models=severity_models,
    )


if __name__ == "__main__":
    main()
