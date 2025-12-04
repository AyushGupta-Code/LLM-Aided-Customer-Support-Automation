"""
Evaluation module for comprehensive model assessment.
Includes metrics calculation and human quality rating system.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
from . import config


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics."""

    def __init__(self, task_type: str = "intent"):
        self.task_type = task_type
        self.metrics = {}

    def calculate_metrics(
        self,
        y_true,
        y_pred,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional, for ROC-AUC)

        Returns:
            Dictionary of metrics
        """
        metrics: Dict[str, float] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        # --- Fixed binary metrics block ---
        # If this is a binary classification problem, compute "binary" metrics
        # with a valid positive label (works for string or numeric labels).
        unique_labels = list(sorted(set(y_true)))
        if len(unique_labels) == 2:
            pos_label = unique_labels[1]  # choose a valid label as "positive"
            metrics["precision_binary"] = precision_score(
                y_true, y_pred, pos_label=pos_label, zero_division=0
            )
            metrics["recall_binary"] = recall_score(
                y_true, y_pred, pos_label=pos_label, zero_division=0
            )
            metrics["f1_binary"] = f1_score(
                y_true, y_pred, pos_label=pos_label, zero_division=0
            )

        # ROC-AUC if probabilities provided
        if y_proba is not None:
            try:
                if len(unique_labels) == 2:
                    # Binary ROC-AUC
                    if y_proba.ndim == 1:
                        # single column of scores
                        scores = y_proba
                    else:
                        # assume proba for each class; take column of pos_label
                        # map labels to column indices
                        label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
                        pos_idx = label_to_idx[unique_labels[1]]
                        scores = y_proba[:, pos_idx]
                    metrics["roc_auc"] = roc_auc_score(y_true, scores)
                else:
                    # Multi-class ROC-AUC (macro)
                    metrics["roc_auc_macro"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="macro"
                    )
            except Exception as e:
                print(f"Warning: Could not calculate ROC-AUC: {e}")

        self.metrics = metrics
        return metrics

    def print_report(self, y_true, y_pred):
        """Print detailed classification report."""
        print(f"\n{'=' * 60}")
        print(f"{self.task_type.upper()} Classification Report")
        print(f"{'=' * 60}")
        print(classification_report(y_true, y_pred, zero_division=0))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print(f"{'=' * 60}\n")

    def print_metrics_summary(self):
        """Print summary of calculated metrics."""
        if not self.metrics:
            print("No metrics calculated yet.")
            return

        print(f"\n{self.task_type.upper()} Model Metrics Summary:")
        print("-" * 40)
        for metric_name, value in self.metrics.items():
            if isinstance(value, float):
                print(f"{metric_name:25s}: {value:.4f}")
        print("-" * 40)

    def check_accuracy_threshold(self, threshold: float = 0.8) -> bool:
        """
        Check if accuracy meets the project requirement (80%).

        Args:
            threshold: Minimum accuracy threshold (default 0.8)

        Returns:
            True if accuracy >= threshold
        """
        if "accuracy" not in self.metrics:
            return False
        return self.metrics["accuracy"] >= threshold


class HumanQualityRater:
    """
    System for human quality ratings (1-5 scale) as per project requirements.
    This is a framework for collecting human evaluations.
    """

    def __init__(self):
        self.ratings = []

    def rate_response(
        self,
        original_tweet: str,
        predicted_intent: str,
        predicted_severity: int,
        generated_reply: str,
        rating: int,
        rater_id: Optional[str] = None,
    ) -> Dict:
        """
        Record a human quality rating for a generated response.

        Args:
            original_tweet: Original customer tweet
            predicted_intent: Predicted intent category
            predicted_severity: Predicted severity (0-3)
            generated_reply: LLM-generated support reply
            rating: Quality rating (1-5 scale)
            rater_id: Optional identifier for the rater

        Returns:
            Dictionary with rating information
        """
        if not (config.HUMAN_RATING_SCALE[0] <= rating <= config.HUMAN_RATING_SCALE[1]):
            raise ValueError(
                f"Rating must be between {config.HUMAN_RATING_SCALE[0]} "
                f"and {config.HUMAN_RATING_SCALE[1]}"
            )

        rating_record = {
            "original_tweet": original_tweet,
            "predicted_intent": predicted_intent,
            "predicted_severity": predicted_severity,
            "generated_reply": generated_reply,
            "rating": rating,
            "rater_id": rater_id,
        }

        self.ratings.append(rating_record)
        return rating_record

    def get_average_rating(self) -> float:
        """Calculate average quality rating."""
        if not self.ratings:
            return 0.0
        return float(np.mean([r["rating"] for r in self.ratings]))

    def get_rating_distribution(self) -> Dict[int, int]:
        """Get distribution of ratings."""
        distribution: Dict[int, int] = {}
        for rating in range(
            config.HUMAN_RATING_SCALE[0], config.HUMAN_RATING_SCALE[1] + 1
        ):
            distribution[rating] = sum(
                1 for r in self.ratings if r["rating"] == rating
            )
        return distribution

    def print_rating_summary(self):
        """Print summary of human quality ratings."""
        if not self.ratings:
            print("No ratings collected yet.")
            return

        print("\nHuman Quality Rating Summary:")
        print("-" * 40)
        print(f"Total ratings: {len(self.ratings)}")
        print(f"Average rating: {self.get_average_rating():.2f}")
        print("\nRating distribution:")
        distribution = self.get_rating_distribution()
        for rating, count in sorted(distribution.items()):
            pct = (count / len(self.ratings) * 100.0) if self.ratings else 0.0
            print(f"  {rating}: {count} ({pct:.1f}%)")
        print("-" * 40)

    def save_ratings(self, filepath: str):
        """Save ratings to CSV file."""
        df = pd.DataFrame(self.ratings)
        df.to_csv(filepath, index=False)
        print(f"Ratings saved to {filepath}")

    def load_ratings(self, filepath: str):
        """Load ratings from CSV file."""
        df = pd.read_csv(filepath)
        self.ratings = df.to_dict("records")
        print(f"Loaded {len(self.ratings)} ratings from {filepath}")


def evaluate_all_models(
    y_true_intent,
    y_pred_intent,
    y_true_severity,
    y_pred_severity,
    y_proba_intent=None,
    y_proba_severity=None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate both intent and severity models.

    Returns:
        Tuple of (intent_metrics, severity_metrics)
    """
    intent_evaluator = ModelEvaluator("intent")
    severity_evaluator = ModelEvaluator("severity")

    intent_metrics = intent_evaluator.calculate_metrics(
        y_true_intent, y_pred_intent, y_proba_intent
    )
    severity_metrics = severity_evaluator.calculate_metrics(
        y_true_severity, y_pred_severity, y_proba_severity
    )

    intent_evaluator.print_report(y_true_intent, y_pred_intent)
    severity_evaluator.print_report(y_true_severity, y_pred_severity)

    intent_evaluator.print_metrics_summary()
    severity_evaluator.print_metrics_summary()

    # Check if accuracy meets requirement
    if intent_evaluator.check_accuracy_threshold(0.8):
        print("\n✓ Intent classifier meets 80% accuracy requirement!")
    else:
        print(
            f"\n⚠ Intent classifier accuracy "
            f"({intent_evaluator.metrics['accuracy']:.2%}) below 80% requirement."
        )

    return intent_metrics, severity_metrics
