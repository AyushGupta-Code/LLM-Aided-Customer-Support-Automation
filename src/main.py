"""
Main entry point for the LLM-Aided Customer Support Automation system.
Implements the complete pipeline as per EM-538 Machine Learning Project Proposal.
"""
import os
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def create_labeled_subset(df: pd.DataFrame, llm: llm_integration.LLMIntegration, 
                          n_samples: int) -> pd.DataFrame:
    """
    Label a subset of data using Gemini LLM.
    """
    sampled = df.sample(min(n_samples, len(df)), random_state=config.RANDOM_STATE)
    labeled_rows = []
    
    logger.info(f"Labeling {len(sampled)} samples with Gemini (this uses your API key)...")
    
    for i, (_, row) in enumerate(sampled.iterrows(), start=1):
        text = str(row["text"])
        ex = llm.call_gemini_for_labels(text)
        
        if ex == "QUOTA_EXCEEDED":
            logger.warning("Hit Gemini quota; stopping further labeling.")
            break
        
        if isinstance(ex, llm_integration.LabeledExample):
            labeled_rows.append({
                "text": ex.text,
                "intent": ex.intent,
                "severity": ex.severity
            })
        
        if i % 5 == 0:
            logger.info(f"  Processed {i}/{len(sampled)} rows... (current labeled: {len(labeled_rows)})")
        
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
    processed_df = processed_df[processed_df["text"].astype(str).str.strip().str.len() > 0].copy()
    if len(processed_df) < initial_count:
        logger.warning(f"Filtered out {initial_count - len(processed_df)} empty texts after preprocessing")
    
    return processed_df


def train_models(labeled_df: pd.DataFrame, model_type: str = "baseline",
                 tune_hyperparameters: bool = False) -> Tuple[Optional[models.BaselineModel], 
                                                              Optional[models.BaselineModel],
                                                              Optional[models.LSTMModel],
                                                              Optional[models.LSTMModel],
                                                              Optional[models.BERTModel],
                                                              Optional[models.BERTModel],
                                                              Dict[str, Any]]:
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
    
    # Split data
    intent_counts = pd.Series(y_intent).value_counts()
    severity_counts = pd.Series(y_severity).value_counts()
    
    use_stratify_intent = intent_counts.min() >= 2
    use_stratify_severity = severity_counts.min() >= 2
    
    if use_stratify_intent:
        X_train_intent, X_test_intent, y_train_intent, y_test_intent = train_test_split(
            X, y_intent, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, 
            stratify=y_intent
        )
    else:
        X_train_intent, X_test_intent, y_train_intent, y_test_intent = train_test_split(
            X, y_intent, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
    
    if use_stratify_severity:
        X_train_severity, X_test_severity, y_train_severity, y_test_severity = train_test_split(
            X, y_severity, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
            stratify=y_severity
        )
    else:
        X_train_severity, X_test_severity, y_train_severity, y_test_severity = train_test_split(
            X, y_severity, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
    
    intent_model_baseline = None
    severity_model_baseline = None
    intent_model_lstm = None
    severity_model_lstm = None
    intent_model_bert = None
    severity_model_bert = None
    
    # Train baseline model
    if model_type in ["baseline", "all"]:
        print("\n" + "="*60)
        print("Training Baseline Models (Logistic Regression + TF-IDF)")
        print("="*60)
        
        intent_model_baseline = models.BaselineModel(task_type="intent")
        intent_model_baseline.train(
            X_train_intent, y_train_intent, 
            X_test_intent, y_test_intent,
            tune_hyperparameters=tune_hyperparameters
        )
        
        severity_model_baseline = models.BaselineModel(task_type="severity")
        severity_model_baseline.train(
            X_train_severity, y_train_severity,
            X_test_severity, y_test_severity,
            tune_hyperparameters=tune_hyperparameters
        )
    
    # Train LSTM model
    if model_type in ["lstm", "all"] and models.TF_AVAILABLE:
        print("\n" + "="*60)
        print("Training LSTM Models")
        print("="*60)
        
        try:
            # Split training data into train/val for LSTM (to avoid data leakage)
            intent_train_val_split = train_test_split(
                X_train_intent, y_train_intent, 
                test_size=config.VALIDATION_SPLIT, 
                random_state=config.RANDOM_STATE,
                stratify=y_train_intent if use_stratify_intent else None
            )
            X_train_intent_lstm, X_val_intent_lstm, y_train_intent_lstm, y_val_intent_lstm = intent_train_val_split
            
            intent_model_lstm = models.LSTMModel(task_type="intent")
            intent_model_lstm.build_tokenizer(X_train_intent_lstm)
            X_train_intent_seq, y_train_intent_seq = intent_model_lstm.prepare_data(X_train_intent_lstm, y_train_intent_lstm)
            X_val_intent_seq, y_val_intent_seq = intent_model_lstm.prepare_data(X_val_intent_lstm, y_val_intent_lstm)
            X_test_intent_seq, _ = intent_model_lstm.prepare_data(X_test_intent)
            intent_model_lstm.train(X_train_intent_seq, y_train_intent_seq, X_val_intent_seq, y_val_intent_seq, use_early_stopping=True)
            
            # Evaluate LSTM intent model
            y_pred_intent_lstm = intent_model_lstm.predict(X_test_intent_seq)
            print(f"\nIntent LSTM Model Results:")
            from sklearn.metrics import classification_report
            print(classification_report(y_test_intent, y_pred_intent_lstm))
            
            # Same for severity
            severity_train_val_split = train_test_split(
                X_train_severity, y_train_severity,
                test_size=config.VALIDATION_SPLIT,
                random_state=config.RANDOM_STATE,
                stratify=y_train_severity if use_stratify_severity else None
            )
            X_train_severity_lstm, X_val_severity_lstm, y_train_severity_lstm, y_val_severity_lstm = severity_train_val_split
            
            severity_model_lstm = models.LSTMModel(task_type="severity")
            severity_model_lstm.build_tokenizer(X_train_severity_lstm)
            X_train_severity_seq, y_train_severity_seq = severity_model_lstm.prepare_data(X_train_severity_lstm, y_train_severity_lstm)
            X_val_severity_seq, y_val_severity_seq = severity_model_lstm.prepare_data(X_val_severity_lstm, y_val_severity_lstm)
            X_test_severity_seq, _ = severity_model_lstm.prepare_data(X_test_severity)
            severity_model_lstm.train(X_train_severity_seq, y_train_severity_seq, X_val_severity_seq, y_val_severity_seq, use_early_stopping=True)
            
            # Evaluate LSTM severity model
            y_pred_severity_lstm = severity_model_lstm.predict(X_test_severity_seq)
            print(f"\nSeverity LSTM Model Results:")
            print(classification_report(y_test_severity, y_pred_severity_lstm))
        except Exception as e:
            logger.error(f"Error training LSTM models: {e}", exc_info=True)
    
    # Train BERT model
    if model_type in ["bert", "all"] and models.TRANSFORMERS_AVAILABLE:
        print("\n" + "="*60)
        print("Training BERT/DistilBERT Models")
        print("="*60)
        
        try:
            # Split training data into train/val for BERT
            intent_train_val_split = train_test_split(
                X_train_intent, y_train_intent,
                test_size=config.VALIDATION_SPLIT,
                random_state=config.RANDOM_STATE,
                stratify=y_train_intent if use_stratify_intent else None
            )
            X_train_intent_bert, X_val_intent_bert, y_train_intent_bert, y_val_intent_bert = intent_train_val_split
            
            intent_model_bert = models.BERTModel(task_type="intent")
            intent_model_bert.train(
                X_train_intent_bert.tolist(), y_train_intent_bert.tolist(),
                X_val_intent_bert.tolist(), y_val_intent_bert.tolist()
            )
            
            # Evaluate BERT intent model
            y_pred_intent_bert = intent_model_bert.predict(X_test_intent.tolist())
            print(f"\nIntent BERT Model Results:")
            from sklearn.metrics import classification_report
            print(classification_report(y_test_intent, y_pred_intent_bert))
            
            severity_train_val_split = train_test_split(
                X_train_severity, y_train_severity,
                test_size=config.VALIDATION_SPLIT,
                random_state=config.RANDOM_STATE,
                stratify=y_train_severity if use_stratify_severity else None
            )
            X_train_severity_bert, X_val_severity_bert, y_train_severity_bert, y_val_severity_bert = severity_train_val_split
            
            severity_model_bert = models.BERTModel(task_type="severity")
            severity_model_bert.train(
                X_train_severity_bert.tolist(), y_train_severity_bert.tolist(),
                X_val_severity_bert.tolist(), y_val_severity_bert.tolist()
            )
            
            # Evaluate BERT severity model
            y_pred_severity_bert = severity_model_bert.predict(X_test_severity.tolist())
            print(f"\nSeverity BERT Model Results:")
            print(classification_report(y_test_severity, y_pred_severity_bert))
        except Exception as e:
            logger.error(f"Error training BERT models: {e}", exc_info=True)
    
    # Return models and test data
    test_data = {
        'X_test_intent': X_test_intent,
        'X_test_severity': X_test_severity,
        'y_test_intent': y_test_intent,
        'y_test_severity': y_test_severity
    }
    
    return (intent_model_baseline, severity_model_baseline,
            intent_model_lstm, severity_model_lstm,
            intent_model_bert, severity_model_bert,
            test_data)


def handle_new_tweet(text: str, intent_model, severity_model, 
                    llm: llm_integration.LLMIntegration, 
                    generate_explanation: bool = True) -> dict:
    """
    Handle a new tweet: predict intent/severity, generate reply and explanation.
    
    Returns:
        Dictionary with predictions, reply, and explanation
    """
    print("\n" + "="*60)
    print("NEW TWEET")
    print("="*60)
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
            'text': text,
            'predicted_intent': 'other',
            'predicted_severity': 1,
            'reply': "I apologize, but I couldn't process your message. Could you please rephrase?",
            'explanation': 'The input text was empty after preprocessing.'
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
        severity = int(severity_pred[0] if isinstance(severity_pred, list) else severity_pred[0])
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
        'text': text,
        'predicted_intent': intent,
        'predicted_severity': severity,
        'reply': reply,
        'explanation': explanation
    }


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("LLM-Aided Customer Support Automation System")
    logger.info("EM-538 Machine Learning Project")
    logger.info("="*60)
    
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
    
    # Train models
    logger.info("[Step 5] Training models...")
    (intent_baseline, severity_baseline,
     intent_lstm, severity_lstm,
     intent_bert, severity_bert,
     test_data) = train_models(
        labeled_df, 
        model_type="baseline",  # Start with baseline, can be changed to "all"
        tune_hyperparameters=False  # Set to True for hyperparameter tuning
    )
    
    # Evaluate models (using the test sets from training)
    logger.info("[Step 6] Evaluating models...")
    
    # Initialize visualization generator
    viz_gen = visualizations.VisualizationGenerator()
    
    # Plot data distribution
    logger.info("Generating data distribution plots...")
    viz_gen.plot_data_distribution(labeled_df, save=True)
    
    if intent_baseline is not None:
        X_test_intent = test_data['X_test_intent']
        X_test_severity = test_data['X_test_severity']
        y_test_intent = test_data['y_test_intent']
        y_test_severity = test_data['y_test_severity']
        
        X_test_intent_processed = [preprocessing.preprocess_text(t) for t in X_test_intent]
        y_pred_intent = intent_baseline.predict(X_test_intent_processed)
        y_pred_severity = severity_baseline.predict([preprocessing.preprocess_text(t) for t in X_test_severity])
        
        # Get prediction probabilities for ROC curves
        try:
            y_proba_intent = intent_baseline.predict_proba(X_test_intent_processed)
            y_proba_severity = severity_baseline.predict_proba([preprocessing.preprocess_text(t) for t in X_test_severity])
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {e}")
            y_proba_intent = None
            y_proba_severity = None
        
        intent_metrics, severity_metrics = evaluation.evaluate_all_models(
            y_test_intent, y_pred_intent,
            y_test_severity, y_pred_severity,
            y_proba_intent, y_proba_severity
        )
        
        # Generate visualizations
        logger.info("Generating evaluation visualizations...")
        
        # Confusion matrices
        intent_labels = sorted(set(y_test_intent) | set(y_pred_intent))
        severity_labels = sorted(set(y_test_severity) | set(y_pred_severity))
        
        viz_gen.plot_confusion_matrix(
            y_test_intent, y_pred_intent, 
            labels=intent_labels,
            task_type="intent",
            save=True
        )
        viz_gen.plot_confusion_matrix(
            y_test_severity, y_pred_severity,
            labels=severity_labels,
            task_type="severity",
            save=True
        )
        
        # ROC curves
        try:
            viz_gen.plot_roc_curve(
                y_test_intent, y_proba_intent,
                labels=intent_labels,
                task_type="intent",
                save=True
            )
            viz_gen.plot_roc_curve(
                y_test_severity, y_proba_severity,
                labels=severity_labels,
                task_type="severity",
                save=True
            )
        except Exception as e:
            logger.warning(f"Could not generate ROC curves: {e}")
        
        # Error analysis
        viz_gen.plot_error_analysis(
            y_test_intent, y_pred_intent,
            texts=X_test_intent.tolist() if hasattr(X_test_intent, 'tolist') else list(X_test_intent),
            labels=intent_labels,
            task_type="intent",
            save=True
        )
        viz_gen.plot_error_analysis(
            y_test_severity, y_pred_severity,
            labels=severity_labels,
            task_type="severity",
            save=True
        )
        
        # Model comparison (if multiple models trained)
        model_results = {
            'Baseline Intent': intent_metrics,
            'Baseline Severity': severity_metrics
        }
        
        # Add other models if available
        if intent_lstm is not None:
            try:
                X_test_intent_seq = intent_lstm.prepare_data(X_test_intent)
                y_pred_intent_lstm = intent_lstm.predict(X_test_intent_seq)
                lstm_evaluator = evaluation.ModelEvaluator("intent")
                lstm_metrics = lstm_evaluator.calculate_metrics(y_test_intent, y_pred_intent_lstm)
                model_results['LSTM Intent'] = lstm_metrics
            except:
                pass
        
        if intent_bert is not None:
            try:
                y_pred_intent_bert = intent_bert.predict(X_test_intent.tolist())
                bert_evaluator = evaluation.ModelEvaluator("intent")
                bert_metrics = bert_evaluator.calculate_metrics(y_test_intent, y_pred_intent_bert)
                model_results['BERT Intent'] = bert_metrics
            except:
                pass
        
        # Generate comparison plots
        if len(model_results) > 1:
            viz_gen.plot_model_comparison(
                model_results,
                metric='accuracy',
                title='Model Accuracy Comparison',
                save=True
            )
            viz_gen.plot_metrics_comparison(
                model_results,
                metrics=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                title='Comprehensive Model Metrics Comparison',
                save=True
            )
        
        # Generate results table
        results_table = viz_gen.generate_results_table(
            model_results,
            save_path='results_summary.csv'
        )
        print("\nResults Summary Table:")
        print(results_table.to_string())
        
        # Save models
        model_dir = config.PROJECT_ROOT / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        if intent_baseline:
            intent_baseline.save(str(model_dir / "intent_baseline.joblib"))
            logger.info("Saved intent baseline model")
        if severity_baseline:
            severity_baseline.save(str(model_dir / "severity_baseline.joblib"))
            logger.info("Saved severity baseline model")
    
    # Test pipeline on sample tweets
    logger.info("[Step 7] Testing pipeline on sample tweets...")
    quality_rater = evaluation.HumanQualityRater()
    
    for i in range(3):
        row = df.sample(1).iloc[0]
        result = handle_new_tweet(
            str(row["text"]), 
            intent_baseline, 
            severity_baseline,
            llm,
            generate_explanation=True
        )
        
        if result:
            # Example: You would collect human ratings here
            # For demonstration, we'll skip actual rating collection
            logger.info("-"*60)
    
    logger.info("="*60)
    logger.info("Pipeline execution complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
