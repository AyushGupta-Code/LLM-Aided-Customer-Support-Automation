"""
Lightweight, reproducible visualizations for the raw `data/twcs.csv` dataset.

Reads the 500MB+ CSV in chunks to avoid memory blow-ups and writes a few
high-level figures into `figures/` for quick EDA:
  - inbound vs outbound volume
  - monthly tweet volume (customer vs. support)
  - top support handles by outbound replies
  - inbound tweet length distribution
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "twcs.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Style setup
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
sns.set_palette("colorblind")


def _twcs_chunks(chunk_size: int = 200_000) -> Iterable[pd.DataFrame]:
    """
    Stream the TWCS CSV in manageable chunks.
    Only pulls the columns we need to keep memory low.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"TWCS dataset not found at {DATA_PATH}. "
            "Place twcs.csv under the data/ directory."
        )

    return pd.read_csv(
        DATA_PATH,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
        ],
        parse_dates=["created_at"],
        date_format="%a %b %d %H:%M:%S %z %Y",
        true_values=["True", "TRUE", "true"],
        false_values=["False", "FALSE", "false"],
        dtype={
            "tweet_id": "Int64",
            "author_id": "string",
            "text": "string",
        },
        chunksize=chunk_size,
    )


def _aggregate_twcs(
    chunk_iter: Iterable[pd.DataFrame],
) -> Tuple[Counter, Counter, Counter, Counter, Counter]:
    """
    Aggregate counts needed for the plots without storing the full dataset.

    Returns:
        inbound_counter: bool -> count
        month_inbound: month string -> count
        month_outbound: month string -> count
        inbound_length_bins: bin label -> count
        top_support_handles: handle -> count
    """
    inbound_counter: Counter = Counter()
    month_inbound: Counter = Counter()
    month_outbound: Counter = Counter()
    inbound_length_bins: Counter = Counter()
    top_support_handles: Counter = Counter()

    # Precompute length bins for repeatable ordering
    length_bins = [
        0,
        40,
        80,
        120,
        160,
        200,
        260,
        320,
        400,
        600,
        800,
        1000,
        math.inf,
    ]
    length_labels = []
    for i in range(len(length_bins) - 1):
        upper = length_bins[i + 1]
        if math.isinf(upper):
            label = f"{int(length_bins[i])}+"
        else:
            label = f"{int(length_bins[i])}-{int(upper - 1)}"
        length_labels.append(label)

    for chunk in chunk_iter:
        chunk = chunk.copy()
        chunk["inbound"] = chunk["inbound"].fillna(False).astype(bool)
        chunk["author_id"] = chunk["author_id"].fillna("unknown")
        chunk["text"] = chunk["text"].fillna("")
        # Remove timezone info to keep monthly grouping simple
        chunk["created_at"] = chunk["created_at"].dt.tz_localize(None)

        inbound_counter.update(chunk["inbound"].tolist())

        # Monthly volume split by inbound/outbound
        month_period = chunk["created_at"].dt.to_period("M")
        inbound_mask = chunk["inbound"]

        month_inbound.update(
            month_period[inbound_mask].dropna().astype(str).tolist()
        )
        month_outbound.update(
            month_period[~inbound_mask].dropna().astype(str).tolist()
        )

        # Top support handles (outbound tweets are from companies)
        top_support_handles.update(
            chunk.loc[~inbound_mask, "author_id"].dropna().tolist()
        )

        # Inbound tweet length distribution (customers)
        inbound_lengths = chunk.loc[inbound_mask, "text"].str.len()
        binned_lengths = pd.cut(
            inbound_lengths,
            bins=length_bins,
            labels=length_labels,
            right=False,
            include_lowest=True,
        )
        inbound_length_bins.update(binned_lengths.dropna().tolist())

    return (
        inbound_counter,
        month_inbound,
        month_outbound,
        inbound_length_bins,
        top_support_handles,
    )


def _plot_inbound_outbound_split(counts: Counter, out_dir: Path) -> Path:
    df = pd.DataFrame(
        {
            "type": ["Customer (inbound)", "Support (outbound)"],
            "count": [counts.get(True, 0), counts.get(False, 0)],
        }
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="type", y="count", ax=ax)
    ax.set_ylabel("Tweets")
    ax.set_xlabel("")
    ax.set_title("Inbound vs Outbound Tweets")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out_path = out_dir / "twcs_inbound_outbound_split.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_monthly_volume(
    inbound: Counter, outbound: Counter, out_dir: Path
) -> Path:
    months = sorted(set(inbound.keys()) | set(outbound.keys()))
    df = pd.DataFrame(
        {
            "month": pd.to_datetime(months),
            "Customer (inbound)": [inbound.get(m, 0) for m in months],
            "Support (outbound)": [outbound.get(m, 0) for m in months],
        }
    ).sort_values("month")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["month"], df["Customer (inbound)"], label="Customer (inbound)")
    ax.plot(df["month"], df["Support (outbound)"], label="Support (outbound)")
    ax.set_ylabel("Tweets per Month")
    ax.set_xlabel("Month")
    ax.set_title("Monthly Volume: TWCS")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    out_path = out_dir / "twcs_monthly_volume.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_top_support_handles(handles: Counter, out_dir: Path, top_n: int = 12) -> Path:
    common = handles.most_common(top_n)
    if not common:
        raise ValueError("No outbound/support tweets found to plot.")

    df = pd.DataFrame(common, columns=["author_id", "count"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="count", y="author_id", orient="h", ax=ax)
    ax.set_xlabel("Outbound Tweets")
    ax.set_ylabel("Support Handle")
    ax.set_title(f"Top {top_n} Support Handles by Activity")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=9)
    plt.tight_layout()

    out_path = out_dir / "twcs_top_support_handles.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_inbound_length_distribution(length_bins: Counter, out_dir: Path) -> Path:
    if not length_bins:
        raise ValueError("No inbound tweets found to plot length distribution.")

    labels = list(length_bins.keys())
    # Preserve the numeric order of bins using the lower bound in the label
    def _bin_start(label: str) -> float:
        if "+" in label:
            return float(label.replace("+", ""))
        start = label.split("-")[0]
        return float(start)

    labels = sorted(labels, key=_bin_start)
    counts = [length_bins[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=labels, y=counts, ax=ax)
    ax.set_xlabel("Inbound Tweet Length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Customer Tweet Length Distribution")
    plt.xticks(rotation=45, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=8)
    plt.tight_layout()

    out_path = out_dir / "twcs_inbound_length_distribution.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_twcs_figures(chunk_size: int = 200_000) -> Dict[str, Path]:
    """
    Run the aggregation + plotting pipeline and return saved figure paths.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    aggregates = _aggregate_twcs(_twcs_chunks(chunk_size=chunk_size))
    inbound_counts, month_in, month_out, length_bins, support_handles = aggregates

    paths = {
        "inbound_outbound_split": _plot_inbound_outbound_split(
            inbound_counts, FIGURES_DIR
        ),
        "monthly_volume": _plot_monthly_volume(month_in, month_out, FIGURES_DIR),
        "top_support_handles": _plot_top_support_handles(
            support_handles, FIGURES_DIR
        ),
        "inbound_length_distribution": _plot_inbound_length_distribution(
            length_bins, FIGURES_DIR
        ),
    }
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TWCS dataset visualizations into figures/."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Rows per chunk to stream from twcs.csv (default: 200000).",
    )
    args = parser.parse_args()

    paths = generate_twcs_figures(chunk_size=args.chunk_size)
    print("Saved TWCS dataset visualizations:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")
