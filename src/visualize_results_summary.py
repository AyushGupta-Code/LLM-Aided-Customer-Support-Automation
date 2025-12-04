"""
Visualize model performance stored in `results_summary.csv`.

Creates horizontal bar charts of key metrics (accuracy, precision_macro,
recall_macro, f1_macro) for each task (intent, severity) and writes PNGs
into `figures/`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ensure imports work regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config


FIGURES_DIR = config.PROJECT_ROOT / "figures"
RESULTS_PATH = config.PROJECT_ROOT / "results_summary.csv"
DEFAULT_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

# Style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
sns.set_palette("colorblind")


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"results_summary.csv not found at {path}. Run `python src/main.py` first."
        )
    df = pd.read_csv(path)
    required = {"task", "model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def _plot_task_metrics(
    df: pd.DataFrame, task: str, metrics: Iterable[str], out_path: Path
) -> None:
    """Plot selected metrics for a single task."""
    subset = df[df["task"] == task].copy()
    if subset.empty:
        raise ValueError(f"No rows found for task '{task}' in results_summary.csv")

    # Order models by f1_macro (desc) if present, otherwise accuracy
    sort_key = "f1_macro" if "f1_macro" in subset.columns else "accuracy"
    model_order = subset.sort_values(sort_key, ascending=False)["model"]

    plot_cols = [m for m in metrics if m in subset.columns]
    melted = subset[["model", *plot_cols]].melt(
        id_vars="model", value_vars=plot_cols, var_name="metric", value_name="score"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=melted,
        x="score",
        y="model",
        hue="metric",
        orient="h",
        order=model_order,
        ax=ax,
    )
    ax.set_title(f"{task.title()} Model Metrics")
    ax.set_xlabel("Score")
    ax.set_ylabel("Model")
    ax.set_xlim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def generate_plots(metrics: List[str]) -> None:
    df = load_results()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tasks = sorted(df["task"].unique())
    for task in tasks:
        out_path = FIGURES_DIR / f"results_{task}_metrics.png"
        _plot_task_metrics(df, task, metrics, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model metrics from results_summary.csv"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to plot (columns in results_summary.csv).",
    )
    args = parser.parse_args()
    generate_plots(args.metrics)


if __name__ == "__main__":
    main()
