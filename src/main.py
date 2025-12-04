"""
Main entry point for the LLM-Aided Customer Support Automation system.
Implements the complete pipeline as per EM-538 Machine Learning Project Proposal.

This version also trains a multitask RNN+BiLSTM on data/manual_labels.csv
and contrasts its performance with the baseline models.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")

import time
import logging
import pandas as pd
import numpy as np
import sys
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model

# Ensure the package is importable when running as a script (python src/main.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src import preprocessing
from src import llm_integration
from src import models
from src import evaluation
from src import visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ========= RNN+BiLSTM hyperparameters (manual_labels.csv) =========
RNN_MAX_VOCAB_SIZE = 20000
RNN_MAX_SEQUENCE_LENGTH = 50
RNN_EMBED_DIM = 128
RNN_LSTM_UNITS = 128
RNN_DENSE_UNITS = 64
RNN_BATCH_SIZE = 32
RNN_EPOCHS = 10
RNN_VAL_SPLIT = 0.2
RNN_RANDOM_SEED = 42
RNN_MODEL_NAME = "rnn_lstm_multitask_manual"


# ============================================================
# Core dataset + baseline / LSTM / BERT pipeline
# ============================================================

def load_customer_support_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Twitter Customer Support (twcs) dataset and keep only
    inbound customer tweets (inbound == True).
    """
    data_path = Path(path) if path else config.DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df[df["inbound"] == True].copy()
    df = df.dropna(subset=["text"])

    # Filter out empty texts
    df = df[df["text"].astype(str).str.strip().str.len() > 0].copy()

    logger.info(f"Loaded {len(df)} customer tweets from {data_path}")
    return df


def create_labeled_subset(
    df: pd.DataFrame,
    llm: llm_integration.LLMIntegration,
    n_samples: int,
) -> pd.DataFrame:
    """
    Label a subset of data using Gemini LLM.
    """
    sampled = df.sample(min(n_samples, len(df)), random_state=config.RANDOM_STATE)
    labeled_rows = []

    logger.info(
        f"Labeling {len(sampled)} samples with Gemini (this uses your API key)..."
    )

    for i, (_, row) in enumerate(sampled.iterrows(), start=1):
        text = str(row["text"])
        ex = llm.call_gemini_for_labels(text)

        if ex == "QUOTA_EXCEEDED":
            logger.warning("Hit Gemini quota; stopping further labeling.")
            break

        if isinstance(ex, llm_integration.LabeledExample):
            labeled_rows.append(
                {
                    "text": ex.text,
                    "intent": ex.intent,
                    "severity": ex.severity,
                }
            )

        if i % 5 == 0:
            logger.info(
                f"  Processed {i}/{len(sampled)} rows... (current labeled: {len(labeled_rows)})"
            )

        time.sleep(config.API_DELAY)

    labeled_df = pd.DataFrame(labeled_rows)
    logger.info(f"Successfully labeled {len(labeled_df)} examples.")
    return labeled_df


def preprocess_labeled_data(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to labeled data."""
    logger.info("Preprocessing labeled data...")
    processed_df = preprocessing.preprocess_dataframe(labeled_df, text_column="text")
    # Use processed text for training
    processed_df["text"] = processed_df["text_processed"]
    processed_df = processed_df.drop(columns=["text_processed"])

    # Filter out empty texts after preprocessing
    initial_count = len(processed_df)
    processed_df = processed_df[
        processed_df["text"].astype(str).str.strip().str.len() > 0
    ].copy()
    if len(processed_df) < initial_count:
        logger.warning(
            f"Filtered out {initial_count - len(processed_df)} empty texts after preprocessing"
        )

    return processed_df


def _safe_train_test_split(
    X,
    y,
    label_name: str,
    test_size: float,
    random_state: int,
):
    """
    Wrapper around train_test_split that uses stratify when possible,
    and gracefully falls back to non-stratified split when the dataset
    is too small (e.g., test_size < number of classes).
    """
    y_series = pd.Series(y)
    class_counts = y_series.value_counts()
    n_classes = len(class_counts)
    n_samples = len(y)

    # Decide whether we *can* stratify at all
    can_stratify = class_counts.min() >= 2

    # Compute how many samples will go into the test set
    if isinstance(test_size, float):
        n_test = int(np.floor(test_size * n_samples))
    else:
        n_test = int(test_size)

    stratify = None
    if can_stratify:
        if n_test >= n_classes:
            stratify = y
        else:
            logger.warning(
                f"[{label_name}] test_size={n_test} is smaller than number of classes "
                f"({n_classes}); falling back to non-stratified split."
            )
    else:
        logger.warning(
            f"[{label_name}] Some classes have fewer than 2 samples; "
            "falling back to non-stratified split."
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def train_models(
    labeled_df: pd.DataFrame,
    model_type: str = "baseline",
    tune_hyperparameters: bool = False,
) -> Tuple[
    Optional[models.BaselineModel],
    Optional[models.BaselineModel],
    Optional[models.LSTMModel],
    Optional[models.LSTMModel],
    Optional[models.BERTModel],
    Optional[models.BERTModel],
    Dict[str, Any],
]:
    """
    Train models for intent and severity classification.

    Args:
        labeled_df: Labeled DataFrame
        model_type: Type of model to train ("baseline", "lstm", "bert", or "all")
        tune_hyperparameters: Whether to perform hyperparameter tuning

    Returns:
        Tuple of (intent_model, severity_model) for each model type, plus test_data dict
    """
    # Prepare data
    X = labeled_df["text"].values
    y_intent = labeled_df["intent"].values
    y_severity = labeled_df["severity"].values

    # === Safely split for intent and severity ===
    X_train_intent, X_test_intent, y_train_intent, y_test_intent = _safe_train_test_split(
        X,
        y_intent,
        label_name="intent",
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    (
        X_train_severity,
        X_test_severity,
        y_train_severity,
        y_test_severity,
    ) = _safe_train_test_split(
        X,
        y_severity,
        label_name="severity",
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    intent_model_baseline = None
    severity_model_baseline = None
    intent_model_lstm = None
    severity_model_lstm = None
    intent_model_bert = None
    severity_model_bert = None

    # Train baseline model
    if model_type in ["baseline", "all"]:
        print("\n" + "=" * 60)
        print("Training Baseline Models (Logistic Regression + TF-IDF)")
        print("=" * 60)

        intent_model_baseline = models.BaselineModel(task_type="intent")
        intent_model_baseline.train(
            X_train_intent,
            y_train_intent,
            X_test_intent,
            y_test_intent,
            tune_hyperparameters=tune_hyperparameters,
        )

        severity_model_baseline = models.BaselineModel(task_type="severity")
        severity_model_baseline.train(
            X_train_severity,
            y_train_severity,
            X_test_severity,
            y_test_severity,
            tune_hyperparameters=tune_hyperparameters,
        )

    # Train LSTM model (single-task Keras LSTM from models.py, if desired)
    if model_type in ["lstm", "all"] and models.TF_AVAILABLE:
        print("\n" + "=" * 60)
        print("Training LSTM Models")
        print("=" * 60)

        try:
            from sklearn.metrics import classification_report

            # Split training data into train/val for LSTM (to avoid data leakage)
            intent_train_val_split = train_test_split(
                X_train_intent,
                y_train_intent,
                test_size=config.VALIDATION_SPLIT,
                random_state=config.RANDOM_STATE,
                stratify=(
                    y_train_intent
                    if (pd.Series(y_train_intent).value_counts().min() >= 2)
                    else None
                ),
            )
            (
                X_train_intent_lstm,
                X_val_intent_lstm,
                y_train_intent_lstm,
                y_val_intent_lstm,
            ) = intent_train_val_split

            intent_model_lstm = models.LSTMModel(task_type="intent")
            intent_model_lstm.build_tokenizer(X_train_intent_lstm)
            (
                X_train_intent_seq,
                y_train_intent_seq,
            ) = intent_model_lstm.prepare_data(
                X_train_intent_lstm, y_train_intent_lstm
            )
            X_val_intent_seq, y_val_intent_seq = intent_model_lstm.prepare_data(
                X_val_intent_lstm, y_val_intent_lstm
            )
            X_test_intent_seq, _ = intent_model_lstm.prepare_data(X_test_intent)
            intent_model_lstm.train(
                X_train_intent_seq,
                y_train_intent_seq,
                X_val_intent_seq,
                y_val_intent_seq,
                use_early_stopping=True,
            )

            # Evaluate LSTM intent model
            y_pred_intent_lstm = intent_model_lstm.predict(X_test_intent_seq)
            print(f"\nIntent LSTM Model Results:")
            print(classification_report(y_test_intent, y_pred_intent_lstm))

            # Severity LSTM
            severity_train_val_split = train_test_split(
                X_train_severity,
                y_train_severity,
                test_size=config.VALIDATION_SPLIT,
                random_state=config.RANDOM_STATE,
                stratify=(
                    y_train_severity
                    if (pd.Series(y_train_severity).value_counts().min() >= 2)
                    else None
                ),
            )
            (
                X_train_severity_lstm,
                X_val_severity_lstm,
                y_train_severity_lstm,
                y_val_severity_lstm,
            ) = severity_train_val_split

            severity_model_lstm = models.LSTMModel(task_type="severity")
            severity_model_lstm.build_tokenizer(X_train_severity_lstm)
            (
                X_train_severity_seq,
                y_train_severity_seq,
            ) = severity_model_lstm.prepare_data(
                X_train_severity_lstm, y_train_severity_lstm
            )
            X_val_severity_seq, y_val_severity_seq = severity_model_lstm.prepare_data(
                X_val_severity_lstm, y_val_severity_lstm
            )
            X_test_severity_seq, _ = severity_model_lstm.prepare_data(X_test_severity)
            severity_model_lstm.train(
                X_train_severity_seq,
                y_train_severity_seq,
                X_val_severity_seq,
                y_val_severity_seq,
                use_early_stopping=True,
            )

            # Evaluate LSTM severity model
            y_pred_severity_lstm = severity_model_lstm.predict(X_test_severity_seq)
            print(f"\nSeverity LSTM Model Results:")
            print(classification_report(y_test_severity, y_pred_severity_lstm))
        except Exception as e:
            logger.error(f"Error training LSTM models: {e}", exc_info=True)

    # Train BERT model
    if model_type in ["bert", "all"] and models.TRANSFORMERS_AVAILABLE:
        print("\n" + "=" * 60)
        print("Training BERT/DistilBERT Models")
        print("=" * 60)

        try:
            from sklearn.metrics import classification_report

            # Split training data into train/val for BERT
            intent_train_val_split = train_test_split(
                X_train_intent,
                y_train_intent,
                test_size=config.VALIDATION_SPLIT,
                random_state=config.RANDOM_STATE,
                stratify=(
                    y_train_intent
                    if (pd.Series(y_train_intent).value_counts().min() >= 2)
                    else None
                ),
            )
            (
                X_train_intent_bert,
                X_val_intent_bert,
                y_train_intent_bert,
                y_val_intent_bert,
            ) = intent_train_val_split

            intent_model_bert = models.BERTModel(task_type="intent")
            intent_model_bert.train(
                X_train_intent_bert.tolist(),
                y_train_intent_bert.tolist(),
                X_val_intent_bert.tolist(),
                y_val_intent_bert.tolist(),
            )

            # Evaluate BERT intent model
            y_pred_intent_bert = intent_model_bert.predict(X_test_intent.tolist())
            print(f"\nIntent BERT Model Results:")
            print(classification_report(y_test_intent, y_pred_intent_bert))

            severity_train_val_split = train_test_split(
                X_train_severity,
                y_train_severity,
                test_size=config.VALIDATION_SPLIT,
                random_state=config.RANDOM_STATE,
                stratify=(
                    y_train_severity
                    if (pd.Series(y_train_severity).value_counts().min() >= 2)
                    else None
                ),
            )
            (
                X_train_severity_bert,
                X_val_severity_bert,
                y_train_severity_bert,
                y_val_severity_bert,
            ) = severity_train_val_split

            severity_model_bert = models.BERTModel(task_type="severity")
            severity_model_bert.train(
                X_train_severity_bert.tolist(),
                y_train_severity_bert.tolist(),
                X_val_severity_bert.tolist(),
                y_val_severity_bert.tolist(),
            )

            # Evaluate BERT severity model
            y_pred_severity_bert = severity_model_bert.predict(
                X_test_severity.tolist()
            )
            print(f"\nSeverity BERT Model Results:")
            print(classification_report(y_test_severity, y_pred_severity_bert))
        except Exception as e:
            logger.error(f"Error training BERT models: {e}", exc_info=True)

    # Return models and test data
    test_data = {
        "X_test_intent": X_test_intent,
        "X_test_severity": X_test_severity,
        "y_test_intent": y_test_intent,
        "y_test_severity": y_test_severity,
    }

    return (
        intent_model_baseline,
        severity_model_baseline,
        intent_model_lstm,
        severity_model_lstm,
        intent_model_bert,
        severity_model_bert,
        test_data,
    )


def handle_new_tweet(
    text: str,
    intent_model,
    severity_model,
    llm: llm_integration.LLMIntegration,
    generate_explanation: bool = True,
) -> dict:
    """
    Handle a new tweet: predict intent/severity, generate reply and explanation.

    Returns:
        Dictionary with predictions, reply, and explanation
    """
    print("\n" + "=" * 60)
    print("NEW TWEET")
    print("=" * 60)
    print(f"Tweet: {text}")

    if intent_model is None or severity_model is None:
        print("\nModels not fully trained, skipping prediction.")
        return {}

    # Preprocess text
    processed_text = preprocessing.preprocess_text(text)

    # Validate processed text
    if not processed_text or len(processed_text.strip()) == 0:
        logger.warning("Text became empty after preprocessing")
        return {
            "text": text,
            "predicted_intent": "other",
            "predicted_severity": 1,
            "reply": "I apologize, but I couldn't process your message. Could you please rephrase?",
            "explanation": "The input text was empty after preprocessing.",
        }

    # Predict
    if isinstance(intent_model, models.BaselineModel):
        intent = intent_model.predict([processed_text])[0]
        severity = int(severity_model.predict([processed_text])[0])
    elif isinstance(intent_model, models.LSTMModel):
        X_seq = intent_model.prepare_data([processed_text])
        intent_pred = intent_model.predict(X_seq)
        intent = intent_pred[0] if isinstance(intent_pred, list) else intent_pred[0]
        X_seq_sev = severity_model.prepare_data([processed_text])
        severity_pred = severity_model.predict(X_seq_sev)
        severity = int(
            severity_pred[0] if isinstance(severity_pred, list) else severity_pred[0]
        )
    elif isinstance(intent_model, models.BERTModel):
        intent = intent_model.predict([processed_text])[0]
        severity = int(severity_model.predict([processed_text])[0])
    else:
        print("Unknown model type")
        return {}

    print(f"\nPredicted Intent: {intent}")
    print(f"Predicted Severity: {severity}")

    # Generate reply
    reply = llm.generate_support_reply(text, intent, severity)
    print(f"\nGenerated Reply:\n{reply}")

    # Generate explanation
    explanation = None
    if generate_explanation:
        explanation = llm.generate_explanation(text, intent, severity)
        print(f"\nExplanation:\n{explanation}")

    return {
        "text": text,
        "predicted_intent": intent,
        "predicted_severity": severity,
        "reply": reply,
        "explanation": explanation,
    }


# ============================================================
# Multitask RNN+BiLSTM on manual_labels.csv
# ============================================================

def _load_manual_labels(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"manual_labels.csv not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"text", "intent", "severity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"manual_labels.csv missing required columns: {missing}")

    df = df.dropna(subset=["text", "intent", "severity"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["intent"] = df["intent"].astype(str).str.strip()
    df["severity"] = df["severity"].astype(str).str.strip()

    # Drop empty texts
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("After filtering, manual_labels.csv has no non-empty rows.")

    return df


def _preprocess_manual_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess manual_labels.csv so we end up with exactly one cleaned
    'text' column plus 'intent' and 'severity'.
    """
    processed_df = preprocessing.preprocess_dataframe(df, text_column="text")

    # Overwrite 'text' with the processed version and drop helper column
    processed_df["text"] = processed_df["text_processed"]
    processed_df = processed_df.drop(columns=["text_processed"])

    # Keep only what we need
    processed_df = processed_df[["text", "intent", "severity"]].copy()

    # Clean types and strip
    processed_df["text"] = processed_df["text"].astype(str)
    processed_df["intent"] = processed_df["intent"].astype(str).str.strip()
    processed_df["severity"] = processed_df["severity"].astype(str).str.strip()

    # Drop empty texts
    processed_df = processed_df[
        processed_df["text"].str.strip().str.len() > 0
    ].reset_index(drop=True)

    if len(processed_df) == 0:
        raise ValueError(
            "After preprocessing, manual_labels.csv has no non-empty rows."
        )

    return processed_df


def _build_rnn_label_mappings(intents, severities):
    intent_labels = sorted(sorted(set(intents)))
    severity_labels = sorted(sorted(set(severities)))

    intent2id = {lbl: i for i, lbl in enumerate(intent_labels)}
    id2intent = {i: lbl for lbl, i in intent2id.items()}

    severity2id = {lbl: i for i, lbl in enumerate(severity_labels)}
    id2severity = {i: lbl for lbl, i in severity2id.items()}

    return intent2id, id2intent, severity2id, id2severity


def _encode_rnn_labels(intents, severities, intent2id, severity2id):
    y_intent = np.array([intent2id[x] for x in intents], dtype="int32")
    y_severity = np.array([severity2id[x] for x in severities], dtype="int32")
    return y_intent, y_severity


def _tokenize_rnn_texts(train_texts, val_texts):
    tokenizer = Tokenizer(num_words=RNN_MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    def _to_padded(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            seqs,
            maxlen=RNN_MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post",
        )

    X_train = _to_padded(train_texts)
    X_val = _to_padded(val_texts)
    return tokenizer, X_train, X_val


def _build_multitask_rnn_model(
    vocab_size: int, num_intents: int, num_severities: int
) -> Model:
    inputs = layers.Input(shape=(RNN_MAX_SEQUENCE_LENGTH,), name="input_ids")

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=RNN_EMBED_DIM,
        input_length=RNN_MAX_SEQUENCE_LENGTH,
        mask_zero=True,
        name="embedding",
    )(inputs)

    x = layers.Bidirectional(
        layers.LSTM(RNN_LSTM_UNITS, return_sequences=False),
        name="bilstm",
    )(x)

    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(RNN_DENSE_UNITS, activation="relu", name="dense")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    intent_output = layers.Dense(
        num_intents, activation="softmax", name="intent_output"
    )(x)
    severity_output = layers.Dense(
        num_severities, activation="softmax", name="severity_output"
    )(x)

    model = Model(
        inputs=inputs,
        outputs={"intent_output": intent_output, "severity_output": severity_output},
        name=RNN_MODEL_NAME,
    )

    model.compile(
        optimizer="adam",
        loss={
            "intent_output": "sparse_categorical_crossentropy",
            "severity_output": "sparse_categorical_crossentropy",
        },
        loss_weights={
            "intent_output": 1.0,
            "severity_output": 0.5,
        },
        metrics={
            "intent_output": "accuracy",
            "severity_output": "accuracy",
        },
    )
    return model


def train_rnn_lstm_on_manual_labels() -> Dict[str, float]:
    """
    Train multitask RNN+BiLSTM on data/manual_labels.csv and return validation metrics.

    Returns:
        eval_metrics: dict with keys:
          - total_loss
          - intent_loss
          - severity_loss
          - intent_accuracy
          - severity_accuracy
    """
    np.random.seed(RNN_RANDOM_SEED)
    tf.random.set_seed(RNN_RANDOM_SEED)

    manual_csv = PROJECT_ROOT / "data" / "manual_labels.csv"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{RNN_MODEL_NAME}.keras"
    meta_path = models_dir / f"{RNN_MODEL_NAME}_meta.json"

    logger.info("Loading manual_labels.csv for RNN+LSTM training...")
    df = _load_manual_labels(manual_csv)
    df = _preprocess_manual_text(df)

    # Robustly extract Series even if duplicate column names somehow exist
    text_col = df["text"]
    if isinstance(text_col, pd.DataFrame):
        text_col = text_col.iloc[:, 0]

    intent_col = df["intent"]
    if isinstance(intent_col, pd.DataFrame):
        intent_col = intent_col.iloc[:, 0]

    severity_col = df["severity"]
    if isinstance(severity_col, pd.DataFrame):
        severity_col = severity_col.iloc[:, 0]

    texts = text_col.astype(str).tolist()
    intents = intent_col.astype(str).tolist()
    severities = severity_col.astype(str).tolist()

    intent2id, id2intent, severity2id, id2severity = _build_rnn_label_mappings(
        intents, severities
    )
    num_intents = len(intent2id)
    num_severities = len(severity2id)

    y_intent, y_severity = _encode_rnn_labels(
        intents, severities, intent2id, severity2id
    )

    # Train/val split (note: texts is a list, y_* are np arrays)
    (
        X_train,
        X_val,
        y_intent_train,
        y_intent_val,
        y_severity_train,
        y_severity_val,
    ) = train_test_split(
        texts,
        y_intent,
        y_severity,
        test_size=RNN_VAL_SPLIT,
        random_state=RNN_RANDOM_SEED,
        stratify=y_intent,
    )

    tokenizer, X_train_ids, X_val_ids = _tokenize_rnn_texts(X_train, X_val)
    vocab_size = min(RNN_MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)

    model = _build_multitask_rnn_model(
        vocab_size=vocab_size,
        num_intents=num_intents,
        num_severities=num_severities,
    )

    logger.info("Training multitask RNN+BiLSTM on manual labels...")
    history = model.fit(
        X_train_ids,
        {
            "intent_output": y_intent_train,
            "severity_output": y_severity_train,
        },
        validation_data=(
            X_val_ids,
            {
                "intent_output": y_intent_val,
                "severity_output": y_severity_val,
            },
        ),
        batch_size=RNN_BATCH_SIZE,
        epochs=RNN_EPOCHS,
        verbose=1,
    )

    eval_results = model.evaluate(
        X_val_ids,
        {"intent_output": y_intent_val, "severity_output": y_severity_val},
        verbose=0,
    )

    eval_metrics = {
        "total_loss": float(eval_results[0]),
        "intent_loss": float(eval_results[1]),
        "severity_loss": float(eval_results[2]),
        "intent_accuracy": float(eval_results[3]),
        "severity_accuracy": float(eval_results[4]),
    }

    # Save model + meta
    logger.info("Saving RNN+LSTM model and metadata...")
    model.save(model_path)

    meta = {
        "tokenizer_config": tokenizer.to_json(),
        "intent2id": intent2id,
        "id2intent": {str(k): v for k, v in id2intent.items()},
        "severity2id": severity2id,
        "id2severity": {str(k): v for k, v in id2severity.items()},
        "max_sequence_length": RNN_MAX_SEQUENCE_LENGTH,
        "max_vocab_size": RNN_MAX_VOCAB_SIZE,
        "eval_metrics": eval_metrics,
        "model_name": RNN_MODEL_NAME,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, indent=2)

    logger.info("RNN+LSTM training complete. Validation metrics: %s", eval_metrics)
    return eval_metrics


# ============================================================
# main()
# ============================================================

def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("LLM-Aided Customer Support Automation System")
    logger.info("EM-538 Machine Learning Project")
    logger.info("=" * 60)

    # Initialize LLM integration
    llm = llm_integration.LLMIntegration()

    # Load data
    logger.info("[Step 1] Loading dataset...")
    df = load_customer_support_data()

    # Create labeled subset
    logger.info("[Step 2] Creating labeled subset with Gemini...")
    labeled_df = create_labeled_subset(df, llm, config.N_LABEL_SAMPLES)

    if labeled_df.empty:
        logger.error("No labeled examples were created. Exiting.")
        return

    logger.info("\nLabel distribution (intent):")
    logger.info(f"\n{labeled_df['intent'].value_counts()}")
    logger.info("\nLabel distribution (severity):")
    logger.info(f"\n{labeled_df['severity'].value_counts()}")

    # Data augmentation for underrepresented classes
    logger.info("[Step 3] Data augmentation for underrepresented classes...")
    labeled_df = llm.augment_dataset(labeled_df, min_samples_per_class=5)
    logger.info(f"After augmentation: {len(labeled_df)} samples")

    # Preprocess data
    logger.info("[Step 4] Preprocessing data...")
    labeled_df = preprocess_labeled_data(labeled_df)

    # Train models (currently only baseline; flip to "all" if you want LSTM/BERT too)
    logger.info("[Step 5] Training models...")
    (
        intent_baseline,
        severity_baseline,
        intent_lstm,
        severity_lstm,
        intent_bert,
        severity_bert,
        test_data,
    ) = train_models(
        labeled_df,
        model_type="baseline",
        tune_hyperparameters=False,
    )

    # Evaluate models (using the test sets from training)
    logger.info("[Step 6] Evaluating models...")

    # Initialize visualization generator
    viz_gen = visualizations.VisualizationGenerator()

    # Plot data distribution
    logger.info("Generating data distribution plots...")
    viz_gen.plot_data_distribution(labeled_df, save=True)

    model_results: Dict[str, Dict[str, float]] = {}

    if intent_baseline is not None and severity_baseline is not None:
        X_test_intent = test_data["X_test_intent"]
        X_test_severity = test_data["X_test_severity"]
        y_test_intent = test_data["y_test_intent"]
        y_test_severity = test_data["y_test_severity"]

        X_test_intent_processed = [
            preprocessing.preprocess_text(t) for t in X_test_intent
        ]
        X_test_severity_processed = [
            preprocessing.preprocess_text(t) for t in X_test_severity
        ]

        y_pred_intent = intent_baseline.predict(X_test_intent_processed)
        y_pred_severity = severity_baseline.predict(X_test_severity_processed)

        # Get prediction probabilities for ROC curves
        try:
            y_proba_intent = intent_baseline.predict_proba(X_test_intent_processed)
            y_proba_severity = severity_baseline.predict_proba(
                X_test_severity_processed
            )
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {e}")
            y_proba_intent = None
            y_proba_severity = None

        intent_metrics, severity_metrics = evaluation.evaluate_all_models(
            y_test_intent,
            y_pred_intent,
            y_test_severity,
            y_pred_severity,
            y_proba_intent,
            y_proba_severity,
        )

        # Confusion matrices
        logger.info("Generating evaluation visualizations...")
        intent_labels = sorted(set(y_test_intent) | set(y_pred_intent))
        severity_labels = sorted(set(y_test_severity) | set(y_pred_severity))

        viz_gen.plot_confusion_matrix(
            y_test_intent,
            y_pred_intent,
            labels=intent_labels,
            task_type="intent",
            save=True,
        )
        viz_gen.plot_confusion_matrix(
            y_test_severity,
            y_pred_severity,
            labels=severity_labels,
            task_type="severity",
            save=True,
        )

        # ROC curves – guard against shape mismatches
        try:
            if (
                y_proba_intent is not None
                and y_proba_intent.shape[0] == len(y_test_intent)
            ):
                viz_gen.plot_roc_curve(
                    y_test_intent,
                    y_proba_intent,
                    labels=intent_labels,
                    task_type="intent",
                    save=True,
                )
            if (
                y_proba_severity is not None
                and y_proba_severity.shape[0] == len(y_test_severity)
            ):
                viz_gen.plot_roc_curve(
                    y_test_severity,
                    y_proba_severity,
                    labels=severity_labels,
                    task_type="severity",
                    save=True,
                )
        except Exception as e:
            logger.warning(f"Could not generate ROC curves: {e}")

        # Error analysis
        viz_gen.plot_error_analysis(
            y_test_intent,
            y_pred_intent,
            texts=X_test_intent.tolist()
            if hasattr(X_test_intent, "tolist")
            else list(X_test_intent),
            labels=intent_labels,
            task_type="intent",
            save=True,
        )
        viz_gen.plot_error_analysis(
            y_test_severity,
            y_pred_severity,
            labels=severity_labels,
            task_type="severity",
            save=True,
        )

        # Seed comparison dict with Baseline metrics
        model_results["Baseline Intent"] = intent_metrics
        model_results["Baseline Severity"] = severity_metrics

    # === Train + include multitask RNN-LSTM on manual_labels.csv ===
    try:
        rnn_eval_metrics = train_rnn_lstm_on_manual_labels()
        rnn_intent_metrics = {
            "accuracy": rnn_eval_metrics.get("intent_accuracy", np.nan),
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "f1_macro": np.nan,
        }
        rnn_severity_metrics = {
            "accuracy": rnn_eval_metrics.get("severity_accuracy", np.nan),
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "f1_macro": np.nan,
        }
        model_results["RNN-LSTM Intent (manual)"] = rnn_intent_metrics
        model_results["RNN-LSTM Severity (manual)"] = rnn_severity_metrics
    except Exception as e:
        logger.error(
            "Error training/evaluating RNN+LSTM on manual labels: %s", e, exc_info=True
        )

    # === COMPARISONS ===
    if len(model_results) > 1:
        # High-level bar charts across all models
        viz_gen.plot_model_comparison(
            model_results,
            metric="accuracy",
            title="Model Accuracy Comparison",
            save=True,
        )
        viz_gen.plot_metrics_comparison(
            model_results,
            metrics=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
            title="Comprehensive Model Metrics Comparison",
            save=True,
        )

        # Focused Baseline vs RNN-LSTM accuracy for Intent & Severity
        viz_gen.plot_baseline_vs_rnn_accuracy(
            model_results,
            save=True,
            filename="baseline_vs_rnn_accuracy.png",
        )

    # Generate results table (if anything was collected)
    if model_results:
        results_table = viz_gen.generate_results_table(
            model_results, save_path="results_summary.csv"
        )
        print("\nResults Summary Table:")
        try:
            print(results_table.to_string())
        except Exception:
            print(results_table)

    # Save baseline models
    model_dir = config.PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if "intent_baseline" in locals() and intent_baseline:
        intent_baseline.save(str(model_dir / "intent_baseline.joblib"))
        logger.info("Saved intent baseline model")
    if "severity_baseline" in locals() and severity_baseline:
        severity_baseline.save(str(model_dir / "severity_baseline.joblib"))
        logger.info("Saved severity baseline model")

    # Test pipeline on sample tweets (baseline path)
    logger.info("[Step 7] Testing pipeline on sample tweets...")
    quality_rater = evaluation.HumanQualityRater()

    for _ in range(3):
        row = df.sample(1).iloc[0]
        result = handle_new_tweet(
            str(row["text"]),
            intent_baseline,
            severity_baseline,
            llm,
            generate_explanation=True,
        )
        if result:
            logger.info("-" * 60)

    logger.info("=" * 60)
    logger.info("Pipeline execution complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
