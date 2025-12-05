import csv
import json
import random
import sys
import time
from pathlib import Path
from collections import Counter

from transformers import pipeline
import torch  # for GPU detection

# ========= CONFIG =========

# Path to TWCS file
TWCSDATA = Path("data/twcs.csv")

# Output file with labels (existing + new)
OUTPUT = Path("data/manual_labels.csv")

# How many *new* tweets to actually ADD per run (upper bound)
N_SAMPLE = 10000  # you can bump this once it works

# Random seed for reproducibility
SEED = 42

# Optional small sleep between calls (can be 0, model is local)
SLEEP_BETWEEN_CALLS = 0.0

# Minimum desired examples per intent (soft target)
MIN_PER_INTENT = 200

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


# ========= LOCAL MODEL (GPU-AWARE) =========
# One model is enough; we'll reuse it for both tasks.

print("Loading local zero-shot classification model (first time may download weights)...")

# Use GPU if available, else CPU
DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Using device={DEVICE} for zero-shot model "
      f"({'GPU' if DEVICE >= 0 else 'CPU'})")

zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    framework="pt",   # force PyTorch, avoid TensorFlow/Keras
    device=DEVICE,    # 0 = first CUDA GPU, -1 = CPU
)


# ========= HELPERS =========

def load_existing_labels(path: Path):
    """Load existing manual_labels.csv if it exists."""
    existing_rows = []
    existing_texts = set()

    if not path.exists():
        print("No existing manual_labels.csv found; starting from scratch.")
        return existing_rows, existing_texts

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "")
            existing_rows.append(row)
            existing_texts.add(text)

    print(f"Loaded {len(existing_rows)} existing labelled tweets.")
    return existing_rows, existing_texts


def compute_intent_counts(existing_rows):
    """Count how many examples we have per intent in existing labels."""
    intents = [
        row.get("intent", "")
        for row in existing_rows
        if row.get("intent")
    ]
    counts = Counter(intents)
    # Ensure all known labels appear, even if 0
    for lbl in INTENT_LABELS:
        counts.setdefault(lbl, 0)
    return counts


def compute_needed_per_intent(intent_counts):
    """
    Decide how many new examples to add per intent to improve balance.

    Strategy:
      - Find the current max count across intents.
      - Target per-intent count = max(current_max, MIN_PER_INTENT).
      - For each intent, we want up to (target - current_count) more examples.
    """
    if intent_counts:
        current_max = max(intent_counts.values())
    else:
        current_max = 0

    target_per_intent = max(current_max, MIN_PER_INTENT)

    needed = {
        intent: max(target_per_intent - intent_counts.get(intent, 0), 0)
        for intent in INTENT_LABELS
    }

    print("\nCurrent intent counts:")
    for intent in INTENT_LABELS:
        print(f"  {intent:15s} = {intent_counts.get(intent, 0)}")

    print(f"\nTarget per intent: {target_per_intent}")
    print("New examples needed per intent (upper bound):")
    for intent in INTENT_LABELS:
        print(f"  {intent:15s} -> {needed[intent]}")

    total_needed = sum(needed.values())
    print(f"Total extra examples desired (cap before N_SAMPLE): {total_needed}\n")
    return needed, total_needed


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

    # 2) Check existing balance
    intent_counts = compute_intent_counts(existing_rows)
    needed_per_intent, total_needed = compute_needed_per_intent(intent_counts)

    if total_needed == 0:
        print("Dataset is already balanced up to the current target; nothing to add.")
        return

    # 3) Load all TWCS tweets
    all_twcs_rows = load_all_twcs_rows(TWCSDATA)

    # 4) Filter out tweets that already exist in manual_labels.csv
    candidate_rows = [
        r for r in all_twcs_rows
        if r["text"] not in existing_texts
    ]

    if not candidate_rows:
        print("No new unique tweets found to label. You're fully up to date!")
        return

    random.seed(SEED)
    random.shuffle(candidate_rows)

    # We only need up to N_SAMPLE *and* up to the total_needed to balance
    max_new_labels = min(N_SAMPLE, total_needed, len(candidate_rows))

    print(f"Found {len(candidate_rows)} unique tweets not in manual_labels.csv.")
    print(f"Will attempt to add up to {max_new_labels} new tweets, "
          f"prioritizing under-represented intents.")

    new_labelled_rows = []
    processed = 0

    for row in candidate_rows:
        if len(new_labelled_rows) >= max_new_labels:
            break

        text = row["text"]

        try:
            intent, severity = call_local_label(text)
        except Exception as e:
            print(f"[ERROR] Local model failed at new row {processed}: {e}", file=sys.stderr)
            intent, severity = "other", 2

        # Only accept if that intent still needs more examples
        if needed_per_intent.get(intent, 0) > 0:
            new_labelled_rows.append(
                {
                    "text": text,
                    "intent": intent,
                    "severity": severity,
                }
            )
            needed_per_intent[intent] -= 1

            if len(new_labelled_rows) % 20 == 0:
                print(f"Accepted {len(new_labelled_rows)}/{max_new_labels} new tweets...")

        processed += 1

        if SLEEP_BETWEEN_CALLS > 0:
            time.sleep(SLEEP_BETWEEN_CALLS)

    if not new_labelled_rows:
        print("No suitable new tweets were found to improve class balance.")
        return

    # 5) Rewrite manual_labels.csv with existing + new unique rows
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

    new_total = len(existing_rows) + len(new_labelled_rows)
    print(f"\nDone. manual_labels.csv now has {new_total} rows.")
    print(f"  - Existing rows kept: {len(existing_rows)}")
    print(f"  - New unique rows added this run: {len(new_labelled_rows)}")

    # Show new intent counts after augmentation
    final_counts = compute_intent_counts(existing_rows + new_labelled_rows)
    print("\nFinal intent counts after augmentation:")
    for intent in INTENT_LABELS:
        print(f"  {intent:15s} = {final_counts[intent]}")


if __name__ == "__main__":
    main()
