"""
Configuration module for the customer support automation project.
"""
import os
from pathlib import Path

# Project root directory (one level above /src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Data paths (using relative paths)
DATA_PATH = DATA_DIR / "twcs.csv"
MANUAL_LABELS_PATH = DATA_DIR / "manual_labels.csv"

# Gemini configuration (legacy; optional for any future experiments)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

# Manual labeling configuration
N_LABEL_SAMPLES = 50  # No longer used for Gemini; kept for backward compatibility

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Baseline model (Logistic Regression + TF-IDF) parameters
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
LOGISTIC_REGRESSION_MAX_ITER = 200

# Zero-shot model
ZERO_SHOT_MODEL_NAME = "typeform/distilbert-base-uncased-mnli"  # lighter than bart-large-mnli
ZERO_SHOT_ENABLED = False  # Disabled by default (slow); enable via CLI/env if needed
ZERO_SHOT_BATCH_SIZE = 16
ZERO_SHOT_MAX_LENGTH = 128

# LSTM/GRU parameters
RNN_MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2

# BERT/DistilBERT parameters
BERT_MODEL_NAME = "distilbert-base-uncased"  # Using DistilBERT for faster training
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 3
BERT_LEARNING_RATE = 2e-5

# Hyperparameter tuning
GRID_SEARCH_CV = 3  # Cross-validation folds for grid search
EARLY_STOPPING_PATIENCE = 3

# Data augmentation
AUGMENTATION_SAMPLES_PER_CLASS = 5  # Number of synthetic samples per underrepresented class

# Evaluation
EVAL_METRICS = ['accuracy', 'precision', 'recall', 'f1']
HUMAN_RATING_SCALE = (1, 5)  # 1-5 scale for human quality ratings

# API rate limiting
API_DELAY = 0.1  # Delay between API calls in seconds
