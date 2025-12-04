"""
Visualization module for generating charts, plots, and analysis figures.
Required for presentation deliverables per grading rubric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
)
from . import config

# Set style for better-looking plots
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    try:
        plt.style.use("seaborn-darkgrid")
    except OSError:
        plt.style.use("default")
sns.set_palette("husl")


class VisualizationGenerator:
    """Generate visualizations for model evaluation and comparison."""

    def __init__(self, output_dir: str | Path = None):
        base_dir = Path(output_dir) if output_dir else config.PROJECT_ROOT / "figures"
        self.output_dir = base_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core evaluation plots
    # ------------------------------------------------------------------
    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        labels: Optional[List] = None,
        title: str = "Confusion Matrix",
        task_type: str = "intent",
        normalize: Optional[str] = None,
        save: bool = True,
    ) -> None:
        """
        Plot confusion matrix with heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names (optional)
            title: Plot title
            task_type: Type of task (intent/severity)
            normalize: None, 'true', 'pred', or 'all' (sklearn style)
            save: Whether to save the figure
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        plt.figure(figsize=(10, 8))

        if normalize is None:
            fmt = "d"
            cbar_label = "Count"
        else:
            fmt = ".2f"
            cbar_label = "Proportion"

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": cbar_label},
        )
        norm_str = "" if normalize is None else f" (normalized: {normalize})"
        plt.title(
            f"{title} - {task_type.upper()}{norm_str}",
            fontsize=14,
            fontweight="bold",
        )
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()

        if save:
            suffix = f"_{normalize}" if normalize else ""
            filename = self.output_dir / f"confusion_matrix_{task_type}{suffix}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved confusion matrix to {filename}")
            plt.close()
        else:
            plt.show()

    def plot_roc_curve(
        self,
        y_true,
        y_proba,
        labels: Optional[List] = None,
        title: str = "ROC Curve",
        task_type: str = "intent",
        save: bool = True,
    ) -> None:
        """
        Plot ROC curve for binary or multi-class classification.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            labels: Label names (optional)
            title: Plot title
            task_type: Type of task
            save: Whether to save the figure
        """
        n_classes = len(set(y_true))

        # ------------------ Binary case ------------------
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(
                y_true,
                y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten(),
            )
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(
                fpr,
                tpr,
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f})",
            )
            plt.plot([0, 1], [0, 1], lw=2, linestyle="--", label="Random")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title(f"{title} - {task_type.upper()}", fontsize=14, fontweight="bold")
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save:
                filename = self.output_dir / f"roc_curve_{task_type}.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Saved ROC curve to {filename}")
                plt.close()
            else:
                plt.show()

        # ------------------ Multi-class case ------------------
        else:
            from sklearn.preprocessing import label_binarize
            from itertools import cycle

            classes_sorted = sorted(set(y_true))
            y_bin = label_binarize(y_true, classes=classes_sorted)
            n_classes = y_bin.shape[1]

            fpr = {}
            tpr = {}
            roc_auc = {}

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Micro-average ROC curve
            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_bin.ravel(), y_proba.ravel()
            )
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            plt.figure(figsize=(10, 8))
            colors = cycle(
                ["aqua", "darkorange", "cornflowerblue", "red", "green"]
            )

            for i, color in zip(range(n_classes), colors):
                label_name = (
                    labels[i] if labels and i < len(labels) else f"Class {i}"
                )
                plt.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=2,
                    label=f"{label_name} (AUC = {roc_auc[i]:.2f})",
                )

            plt.plot(
                fpr["micro"],
                tpr["micro"],
                linestyle="--",
                lw=2,
                label=f"Micro-average (AUC = {roc_auc['micro']:.2f})",
            )
            plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title(f"{title} - {task_type.upper()}", fontsize=14, fontweight="bold")
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save:
                filename = self.output_dir / f"roc_curve_{task_type}.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Saved ROC curve to {filename}")
                plt.close()
            else:
                plt.show()

    # ------------------------------------------------------------------
    # New: per-class metrics bar chart (precision / recall / F1)
    # ------------------------------------------------------------------
    def plot_per_class_metrics(
        self,
        y_true,
        y_pred,
        task_type: str = "intent",
        labels: Optional[List[str]] = None,
        save: bool = True,
    ) -> None:
        """
        Plot per-class precision, recall, and F1 scores as a grouped bar chart.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            task_type: 'intent' or 'severity'
            labels: Optional explicit label order
            save: Whether to save the figure
        """
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).T

        # Drop overall rows
        per_class = report_df.iloc[:-3][["precision", "recall", "f1-score"]]

        if labels is not None:
            # Ensure we preserve the desired order and only keep those labels
            per_class = per_class.reindex(labels)

        plt.figure(figsize=(10, 6))
        per_class.plot(kind="bar")
        plt.title(
            f"Per-class Precision / Recall / F1 - {task_type.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Score")
        plt.ylim(0, 1.0)
        plt.legend(title="")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save:
            filename = self.output_dir / f"per_class_metrics_{task_type}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved per-class metrics plot to {filename}")
            plt.close()
        else:
            plt.show()

    # ------------------------------------------------------------------
    # New: training curves for RNN (loss & accuracy vs epochs)
    # ------------------------------------------------------------------
    def plot_training_curves(
        self,
        history: Any,
        task_name: str = "RNN-LSTM",
        save: bool = True,
    ) -> None:
        """
        Plot training & validation loss / accuracy curves.

        Args:
            history: Keras History object, dict, or DataFrame with keys/cols:
                     'loss', 'val_loss', 'accuracy', 'val_accuracy'
            task_name: Name to show in plot titles
            save: Whether to save figures
        """
        # Normalize to a dict of lists
        if hasattr(history, "history"):  # Keras History
            h = history.history
        elif isinstance(history, pd.DataFrame):
            h = {col: history[col].tolist() for col in history.columns}
        elif isinstance(history, dict):
            h = history
        else:
            raise ValueError("Unsupported history type")

        epochs = range(1, len(h.get("loss", [])) + 1)

        # Loss
        if "loss" in h and "val_loss" in h:
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, h["loss"], label="Train loss")
            plt.plot(epochs, h["val_loss"], label="Val loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{task_name} - Loss Curves", fontsize=14, fontweight="bold")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save:
                filename = self.output_dir / f"{task_name.lower().replace(' ', '_')}_loss_curves.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Saved training loss curves to {filename}")
                plt.close()
            else:
                plt.show()

        # Accuracy
        if "accuracy" in h and "val_accuracy" in h:
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, h["accuracy"], label="Train accuracy")
            plt.plot(epochs, h["val_accuracy"], label="Val accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{task_name} - Accuracy Curves", fontsize=14, fontweight="bold")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save:
                filename = self.output_dir / f"{task_name.lower().replace(' ', '_')}_accuracy_curves.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Saved training accuracy curves to {filename}")
                plt.close()
            else:
                plt.show()

    # ------------------------------------------------------------------
    # Model-level comparisons
    # ------------------------------------------------------------------
    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        metric: str = "accuracy",
        title: str = "Model Comparison",
        save: bool = True,
    ) -> None:
        """
        Plot bar chart comparing models across a single metric.

        Args:
            model_results: Dict mapping model names to metric dictionaries
            metric: Metric to compare (accuracy, f1_macro, etc.)
            title: Plot title
            save: Whether to save the figure
        """
        model_names = list(model_results.keys())
        metric_values = [model_results[name].get(metric, 0.0) for name in model_names]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            model_names,
            metric_values,
            color=sns.color_palette("husl", len(model_names)),
        )
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.title(
            f"{title} - {metric.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )
        ymax = max(metric_values) if metric_values else 0.0
        plt.ylim([0, max(1.0, ymax * 1.1)])

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save:
            filename = self.output_dir / f"model_comparison_{metric}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved model comparison to {filename}")
            plt.close()
        else:
            plt.show()

    def plot_metrics_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        title: str = "Model Metrics Comparison",
        save: bool = True,
    ) -> None:
        """
        Plot multiple metrics comparison across models.

        Args:
            model_results: Dict mapping model names to metric dictionaries
            metrics: List of metrics to compare
            title: Plot title
            save: Whether to save the figure
        """
        if metrics is None:
            metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

        model_names = list(model_results.keys())
        x = np.arange(len(model_names))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, metric in enumerate(metrics):
            values = [model_results[name].get(metric, 0.0) for name in model_names]
            offset = (i - len(metrics) / 2) * width + width / 2
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=metric.replace("_", " ").title(),
            )

            # Add value labels
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_ylabel("Score", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0, 1.1])

        plt.tight_layout()

        if save:
            filename = self.output_dir / "metrics_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved metrics comparison to {filename}")
            plt.close()
        else:
            plt.show()

    def plot_baseline_vs_rnn_accuracy(
        self,
        model_results: Dict[str, Dict[str, float]],
        save: bool = True,
        filename: str = "baseline_vs_rnn_accuracy.png",
    ) -> None:
        """
        Focused accuracy comparison between Baseline and RNN-LSTM models
        for both Intent and Severity.

        Expects the following keys (if present) in model_results:
          - "Baseline Intent"
          - "Baseline Severity"
          - "RNN-LSTM Intent (manual)"
          - "RNN-LSTM Severity (manual)"
        """
        tasks = ["Intent", "Severity"]

        baseline_intent_acc = model_results.get("Baseline Intent", {}).get(
            "accuracy", np.nan
        )
        baseline_severity_acc = model_results.get("Baseline Severity", {}).get(
            "accuracy", np.nan
        )
        rnn_intent_acc = model_results.get("RNN-LSTM Intent (manual)", {}).get(
            "accuracy", np.nan
        )
        rnn_severity_acc = model_results.get("RNN-LSTM Severity (manual)", {}).get(
            "accuracy", np.nan
        )

        baseline_vals = [baseline_intent_acc, baseline_severity_acc]
        rnn_vals = [rnn_intent_acc, rnn_severity_acc]

        x = np.arange(len(tasks))
        width = 0.35

        plt.figure(figsize=(6, 4))
        bars_baseline = plt.bar(
            x - width / 2, baseline_vals, width, label="Baseline"
        )
        bars_rnn = plt.bar(
            x + width / 2, rnn_vals, width, label="RNN-LSTM (manual)"
        )

        plt.ylabel("Accuracy")
        plt.ylim(0, 1.0)
        plt.xticks(x, tasks)
        plt.title("Baseline vs RNN-LSTM Accuracy (Intent & Severity)")
        plt.legend()

        # Add value labels
        for bar_group in (bars_baseline, bars_rnn):
            for bar in bar_group:
                height = bar.get_height()
                if not np.isnan(height):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.02,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        plt.tight_layout()

        if save:
            out_path = self.output_dir / filename
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Saved baseline vs RNN accuracy plot to {out_path}")
            plt.close()
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Error analysis & distributions
    # ------------------------------------------------------------------
    def plot_error_analysis(
        self,
        y_true,
        y_pred,
        texts: Optional[List[str]] = None,
        labels: Optional[List] = None,
        task_type: str = "intent",
        top_n: int = 10,
        save: bool = True,
    ) -> None:
        """
        Analyze and visualize misclassification patterns.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            texts: Original texts (optional, for detailed analysis)
            labels: Label names
            task_type: Type of task
            top_n: Number of top misclassifications to show
            save: Whether to save the figure
        """
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Collect misclassifications
        misclassifications = []
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                if i != j and cm[i, j] > 0:
                    misclassifications.append(
                        {
                            "true": true_label,
                            "predicted": pred_label,
                            "count": cm[i, j],
                        }
                    )

        misclassifications.sort(key=lambda x: x["count"], reverse=True)
        top_misclass = misclassifications[:top_n]

        if not top_misclass:
            print("No misclassifications found.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Confusion matrix heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
            ax=ax1,
        )
        ax1.set_title(
            f"Confusion Matrix - {task_type.upper()}",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_ylabel("True Label", fontsize=10)
        ax1.set_xlabel("Predicted Label", fontsize=10)

        # Top misclassifications bar chart
        true_labels = [m["true"] for m in top_misclass]
        pred_labels = [m["predicted"] for m in top_misclass]
        counts = [m["count"] for m in top_misclass]

        misclass_labels = [f"{t} → {p}" for t, p in zip(true_labels, pred_labels)]

        bars = ax2.barh(misclass_labels, counts, color="coral")
        ax2.set_xlabel("Count", fontsize=10)
        ax2.set_title(
            f"Top {top_n} Misclassifications",
            fontsize=12,
            fontweight="bold",
        )
        ax2.grid(axis="x", alpha=0.3)

        for bar, count in zip(bars, counts):
            ax2.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        if save:
            filename = self.output_dir / f"error_analysis_{task_type}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved error analysis to {filename}")
            plt.close()
        else:
            plt.show()

        # Print summary to console
        print(f"\nError Analysis Summary - {task_type.upper()}:")
        print("-" * 60)
        for m in top_misclass:
            print(f"  {m['true']} → {m['predicted']}: {m['count']} cases")

    def generate_results_table(
        self,
        model_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate a comprehensive results table.

        Args:
            model_results: Dict mapping model names to metric dictionaries
            save_path: Optional path to save CSV

        Returns:
            DataFrame with results
        """
        df = pd.DataFrame(model_results).T
        df = df.round(4)

        if save_path:
            df.to_csv(save_path)
            print(f"Saved results table to {save_path}")

        return df

    def plot_data_distribution(
        self,
        labeled_df: pd.DataFrame,
        save: bool = True,
    ) -> None:
        """
        Plot distribution of labels in the dataset.

        Args:
            labeled_df: DataFrame with 'intent' and 'severity' columns
            save: Whether to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Intent distribution
        intent_counts = labeled_df["intent"].value_counts()
        ax1.bar(intent_counts.index, intent_counts.values, color="steelblue")
        ax1.set_xlabel("Intent Category", fontsize=11)
        ax1.set_ylabel("Count", fontsize=11)
        ax1.set_title("Intent Distribution", fontsize=12, fontweight="bold")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(axis="y", alpha=0.3)

        for i, v in enumerate(intent_counts.values):
            ax1.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=9)

        # Severity distribution
        severity_counts = labeled_df["severity"].value_counts().sort_index()
        ax2.bar(
            severity_counts.index.astype(str),
            severity_counts.values,
            color="coral",
        )
        ax2.set_xlabel("Severity Level", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("Severity Distribution", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        for i, v in enumerate(severity_counts.values):
            ax2.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=9)

        plt.tight_layout()

        if save:
            filename = self.output_dir / "data_distribution.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved data distribution plot to {filename}")
            plt.close()
        else:
            plt.show()

    # ------------------------------------------------------------------
    # New: Intent × Severity heatmap
    # ------------------------------------------------------------------
    def plot_intent_severity_heatmap(
        self,
        labeled_df: pd.DataFrame,
        normalize: bool = True,
        save: bool = True,
    ) -> None:
        """
        Plot heatmap of severity distribution within each intent.

        Args:
            labeled_df: DataFrame with 'intent' and 'severity' columns
            normalize: If True, normalize counts by row (intent)
            save: Whether to save the figure
        """
        crosstab = pd.crosstab(
            labeled_df["intent"], labeled_df["severity"], normalize="index" if normalize else False
        )

        plt.figure(figsize=(8, 5))
        sns.heatmap(
            crosstab,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="YlGnBu",
            cbar_kws={"label": "Proportion" if normalize else "Count"},
        )
        plt.title("Severity Distribution Within Each Intent", fontsize=14, fontweight="bold")
        plt.ylabel("Intent")
        plt.xlabel("Severity")
        plt.tight_layout()

        if save:
            filename = self.output_dir / "intent_severity_heatmap.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved intent-severity heatmap to {filename}")
            plt.close()
        else:
            plt.show()

    # ------------------------------------------------------------------
    # New: Agreement-by-class bar chart (e.g., RNN vs Gemini/manual)
    # ------------------------------------------------------------------
    def plot_agreement_by_label(
        self,
        y_true,
        y_pred,
        task_type: str = "intent",
        labels: Optional[List[str]] = None,
        save: bool = True,
    ) -> None:
        """
        Plot agreement rate (accuracy) for each true class.

        This is effectively recall, but framed as "agreement rate"
        which is nicer to explain in a presentation.

        Args:
            y_true: Ground truth labels (e.g., from manual_labels.csv)
            y_pred: Model predictions
            task_type: 'intent' or 'severity'
            labels: Optional explicit label order
            save: Whether to save the figure
        """
        df = pd.DataFrame({"true": y_true, "pred": y_pred})
        df["agree"] = (df["true"] == df["pred"]).astype(float)

        if labels is None:
            labels = sorted(df["true"].unique())

        agreement = (
            df.groupby("true")["agree"].mean()
            .reindex(labels)
            .reset_index()
            .rename(columns={"true": "label", "agree": "agreement"})
        )

        plt.figure(figsize=(8, 4))
        sns.barplot(x="label", y="agreement", data=agreement)
        plt.ylim(0, 1.0)
        plt.ylabel("Agreement Rate")
        plt.xlabel(f"True {task_type.title()}")
        plt.title(f"Agreement by {task_type.title()} Class", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save:
            filename = self.output_dir / f"agreement_by_{task_type}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved agreement-by-class plot to {filename}")
            plt.close()
        else:
            plt.show()
