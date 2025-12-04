import csv
import json
import random
import sys
import time
from pathlib import Path

from transformers import pipeline


# ========= CONFIG =========

# Path to TWCS file
TWCSDATA = Path("data/twcs.csv")

# Output file with labels (existing + new)
OUTPUT = Path("data/manual_labels.csv")

# How many *new* tweets to label per run
N_SAMPLE = 500  # you can bump this once it works

# Random seed for reproducibility
SEED = 42

# Optional small sleep between calls (can be 0, model is local)
SLEEP_BETWEEN_CALLS = 0.0


INTENT_LABELS = [
    "technical_issue",
    "account_issue",
    "billing_issue",
    "shipping_issue",
    "general_query",
    "feedback",
    "other",
]

SEVERITY_LABELS = ["1", "2", "3"]  # we'll cast to int later

SEVERITY_SCALE_TEXT = """
1 = low (non-urgent, general question, casual feedback)
2 = medium (inconvenient, but app/service basically works)
3 = high (blocking/critical, cannot use service, strong frustration)
"""


# ========= LOCAL MODELS =========
# One model is enough; we'll reuse it for both tasks.

print("Loading local zero-shot classification model (first time may download weights)...")
zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    framework="pt",   # <--- force PyTorch, avoid TensorFlow/Keras
    device=-1,        # CPU; set to 0 if you have a CUDA GPU
)


# ========= HELPERS =========

def load_existing_labels(path: Path):
    """Load existing manual_labels.csv if it exists."""
    existing_rows = []
    existing_texts = set()

    if not path.exists():
        return existing_rows, existing_texts

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "")
            existing_rows.append(row)
            existing_texts.add(text)

    print(f"Loaded {len(existing_rows)} existing labelled tweets.")
    return existing_rows, existing_texts


def load_all_twcs_rows(path: Path):
    """Load TWCS and return all rows with text."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = [row for row in reader if row.get("text")]

    if not all_rows:
        raise RuntimeError(f"No rows with a 'text' field found in {path}")

    print(f"Found {len(all_rows)} total tweets in TWCS.")
    return all_rows


def classify_intent(text: str) -> str:
    """Use local zero-shot classifier to pick one intent label."""
    result = zero_shot(
        text,
        candidate_labels=INTENT_LABELS,
        multi_label=False,
    )
    # result["labels"] is sorted by score desc
    if not result or "labels" not in result or not result["labels"]:
        return "other"
    intent = result["labels"][0]
    if intent not in INTENT_LABELS:
        intent = "other"
    return intent


def classify_severity(text: str) -> int:
    """Use local zero-shot classifier to pick severity 1/2/3."""
    # Give the model a bit of guidance so it knows we're talking about severity/urgency.
    result = zero_shot(
        text + " " + SEVERITY_SCALE_TEXT,
        candidate_labels=SEVERITY_LABELS,
        multi_label=False,
    )
    if not result or "labels" not in result or not result["labels"]:
        return 2

    label = result["labels"][0]
    try:
        sev = int(label)
    except ValueError:
        sev = 2

    if sev not in (1, 2, 3):
        sev = 2
    return sev


def call_local_label(text: str):
    """Wrapper that returns (intent, severity) using local models."""
    intent = classify_intent(text)
    severity = classify_severity(text)
    return intent, severity


# ========= MAIN =========

def main():
    if not TWCSDATA.exists():
        raise FileNotFoundError(f"Cannot find TWCS file at: {TWCSDATA}")

    # 1) Load existing manual_labels.csv
    existing_rows, existing_texts = load_existing_labels(OUTPUT)

    # 2) Load all TWCS tweets
    all_twcs_rows = load_all_twcs_rows(TWCSDATA)

    # 3) Filter out tweets that already exist in manual_labels.csv
    candidate_rows = [
        r for r in all_twcs_rows
        if r["text"] not in existing_texts
    ]

    if not candidate_rows:
        print("No new unique tweets found to label. You're fully up to date!")
        return

    random.seed(SEED)

    # We only need up to N_SAMPLE new tweets this run
    n_new = min(N_SAMPLE, len(candidate_rows))
    sampled_new_rows = random.sample(candidate_rows, n_new)

    print(f"Found {len(candidate_rows)} unique tweets not in manual_labels.csv.")
    print(f"Sampling {n_new} of them to label locally with BART (zero-shot).")

    new_labelled_rows = []

    for i, row in enumerate(sampled_new_rows, start=1):
        text = row["text"]

        try:
            intent, severity = call_local_label(text)
        except Exception as e:
            print(f"[ERROR] Local model failed at new row {i}: {e}", file=sys.stderr)
            intent, severity = "other", 2

        new_labelled_rows.append(
            {
                "text": text,
                "intent": intent,
                "severity": severity,
            }
        )

        if i % 20 == 0:
            print(f"Labelled {i}/{n_new} new tweets...")

        if SLEEP_BETWEEN_CALLS > 0:
            time.sleep(SLEEP_BETWEEN_CALLS)

    # 4) Rewrite manual_labels.csv with existing + new unique rows
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f_out:
        fieldnames = ["text", "intent", "severity"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        # Existing labels stay as-is
        for row in existing_rows:
            writer.writerow(
                {
                    "text": row.get("text", ""),
                    "intent": row.get("intent", ""),
                    "severity": row.get("severity", ""),
                }
            )

        # Then append new labelled rows
        for row in new_labelled_rows:
            writer.writerow(row)

    print(f"Done. manual_labels.csv now has {len(existing_rows) + len(new_labelled_rows)} rows.")
    print(f"  - Existing rows kept: {len(existing_rows)}")
    print(f"  - New unique rows added this run: {len(new_labelled_rows)}")


if __name__ == "__main__":
    main()
