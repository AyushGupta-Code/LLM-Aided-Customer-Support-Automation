"""
Machine learning models module.
Implements baseline (Logistic Regression), LSTM/GRU, and BERT/DistilBERT models.
"""
import os
# Prevent Transformers from loading TensorFlow/Keras integration (Keras 3 incompatible).
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM/GRU models will be skipped.")

# Avoid pulling TensorFlow/Keras inside transformers (unsupported with Keras 3).
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from transformers import DistilBertForSequenceClassification
    from datasets import Dataset
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers library not available. BERT models will be skipped.")

from . import config


class BaselineModel:
    """Baseline model: Logistic Regression + TF-IDF."""
    
    def __init__(self, task_type: str = "intent"):
        self.task_type = task_type
        self.pipeline = None
        self.is_trained = False
    
    def build_pipeline(self, solver: str = "lbfgs"):
        """Build the baseline pipeline."""
        if self.task_type == "severity":
            clf = LogisticRegression(
                max_iter=config.LOGISTIC_REGRESSION_MAX_ITER, 
                multi_class="multinomial",
                solver=solver
            )
        else:
            clf = LogisticRegression(
                max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
                solver=solver
            )
        
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=config.TFIDF_MAX_FEATURES,
                ngram_range=config.TFIDF_NGRAM_RANGE
            )),
            ("clf", clf)
        ])
    
    def train(self, X_train, y_train, X_test=None, y_test=None, tune_hyperparameters: bool = False):
        """Train the baseline model."""
        if self.pipeline is None:
            self.build_pipeline()
        
        if tune_hyperparameters:
            # Grid search for hyperparameter tuning
            # Note: L1 penalty requires solver='liblinear' or 'saga'
            # We'll use separate grids for L1 and L2
            param_grids = [
                {
                    'clf__C': [0.1, 1.0, 10.0],
                    'clf__penalty': ['l2'],
                    'clf__solver': ['lbfgs', 'liblinear'],
                    'tfidf__max_features': [10000, 20000],
                },
                {
                    'clf__C': [0.1, 1.0, 10.0],
                    'clf__penalty': ['l1'],
                    'clf__solver': ['liblinear'],
                    'tfidf__max_features': [10000, 20000],
                }
            ]
            
            best_score = -1
            best_estimator = None
            best_params = None
            
            for param_grid in param_grids:
                grid_search = GridSearchCV(
                    self.pipeline,
                    param_grid,
                    cv=config.GRID_SEARCH_CV,
                    scoring='f1_macro',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_estimator = grid_search.best_estimator_
                    best_params = grid_search.best_params_
            
            self.pipeline = best_estimator
            print(f"Best parameters: {best_params}")
            print(f"Best cross-validation score: {best_score:.4f}")
        else:
            self.pipeline.fit(X_train, y_train)
        
        self.is_trained = True
        
        if X_test is not None and y_test is not None:
            y_pred = self.pipeline.predict(X_test)
            print(f"\n{self.task_type.upper()} Baseline Model Results:")
            print(classification_report(y_test, y_pred))
            print("Confusion matrix:")
            print(confusion_matrix(y_test, y_pred))
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        return self.pipeline.predict_proba(X)
    
    def save(self, filepath: str):
        """Save the model."""
        if self.pipeline is not None:
            joblib.dump(self.pipeline, filepath)
    
    def load(self, filepath: str):
        """Load the model."""
        self.pipeline = joblib.load(filepath)
        self.is_trained = True


class LSTMModel:
    """LSTM model for sequence classification."""
    
    def __init__(self, task_type: str = "intent", vocab_size: int = 10000):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models.")
        self.task_type = task_type
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.model = None
        self.num_classes = None
        self.is_trained = False
    
    def build_tokenizer(self, texts):
        """Build tokenizer from training texts."""
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
    
    def prepare_data(self, texts, labels=None):
        """Prepare data for LSTM."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not built. Call build_tokenizer first.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH)
        
        if labels is not None:
            # Convert labels to numeric if needed
            if isinstance(labels[0], str):
                unique_labels = sorted(set(labels))
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                y = np.array([label_to_idx[label] for label in labels])
                self.label_to_idx = label_to_idx
                self.idx_to_label = {idx: label for label, idx in label_to_idx.items()}
                self.num_classes = len(unique_labels)
            else:
                y = np.array(labels)
                self.num_classes = len(set(labels))
            return X, y
        
        return X
    
    def build_model(self):
        """Build the LSTM model."""
        if self.num_classes is None:
            raise ValueError("Number of classes not determined. Prepare data first.")
        
        self.model = Sequential([
            Embedding(self.vocab_size, config.EMBEDDING_DIM, input_length=config.MAX_SEQUENCE_LENGTH),
            Bidirectional(LSTM(config.LSTM_UNITS, return_sequences=True)),
            Dropout(config.DROPOUT_RATE),
            Bidirectional(LSTM(config.LSTM_UNITS)),
            Dropout(config.DROPOUT_RATE),
            Dense(64, activation='relu'),
            Dropout(config.DROPOUT_RATE),
            Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, use_early_stopping: bool = True):
        """Train the LSTM model."""
        if self.model is None:
            self.build_model()
        
        callbacks = []
        if use_early_stopping and X_val is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        predictions = self.model.predict(X)
        if self.num_classes > 2:
            predicted_indices = np.argmax(predictions, axis=1)
            if hasattr(self, 'idx_to_label'):
                return [self.idx_to_label[idx] for idx in predicted_indices]
            return predicted_indices
        else:
            return (predictions > 0.5).astype(int).flatten()


class BERTModel:
    """BERT/DistilBERT model for classification."""
    
    def __init__(self, task_type: str = "intent", model_name: str = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for BERT models.")
        self.task_type = task_type
        self.model_name = model_name or config.BERT_MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.label_to_id = None
        self.id_to_label = None
        self.is_trained = False
    
    def prepare_data(self, texts, labels):
        """Prepare data for BERT."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Convert labels to numeric if needed
        if isinstance(labels[0], str):
            unique_labels = sorted(set(labels))
            self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
            numeric_labels = [self.label_to_id[label] for label in labels]
        else:
            numeric_labels = labels
            unique_labels = sorted(set(labels))
            self.id_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=config.BERT_MAX_LENGTH,
            return_tensors="pt"
        )
        
        return encodings, numeric_labels
    
    def build_model(self, num_labels: int):
        """Build the BERT model."""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate: float = None):
        """Train the BERT model."""
        lr = learning_rate or config.BERT_LEARNING_RATE
        
        # Prepare data
        train_encodings, train_labels = self.prepare_data(X_train, y_train)
        num_labels = len(set(train_labels))
        
        if self.model is None:
            self.build_model(num_labels)
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'].tolist(),
            'attention_mask': train_encodings['attention_mask'].tolist(),
            'labels': train_labels
        })
        
        if X_val is not None and y_val is not None:
            val_encodings, val_labels = self.prepare_data(X_val, y_val)
            val_dataset = Dataset.from_dict({
                'input_ids': val_encodings['input_ids'].tolist(),
                'attention_mask': val_encodings['attention_mask'].tolist(),
                'labels': val_labels
            })
        else:
            val_dataset = None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./results_{self.task_type}',
            num_train_epochs=config.BERT_EPOCHS,
            per_device_train_batch_size=config.BERT_BATCH_SIZE,
            per_device_eval_batch_size=config.BERT_BATCH_SIZE,
            learning_rate=lr,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'./logs_{self.task_type}',
            logging_steps=10,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
        )
        
        # Custom data collator
        from transformers import DataCollatorWithPadding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        self.is_trained = True
    
    def predict(self, texts):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=config.BERT_MAX_LENGTH,
            return_tensors="pt"
        )
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_ids = torch.argmax(predictions, dim=-1)
        
        if hasattr(self, 'id_to_label'):
            return [self.id_to_label[idx.item()] for idx in predicted_ids]
        return predicted_ids.numpy()
