# Customer Support Intent/Severity Classification

Classifies customer tweets by intent and severity (0–3) using the manually labeled ground-truth dataset (no Gemini/LLM labeling). Five models are trained: Logistic Regression, Linear SVM, SGD (logistic), RNN + Bidirectional LSTM, and Multinomial Naive Bayes.

## How to Run
Prereqs: Python 3.10+, a virtual environment, and the dataset files `data/manual_labels.csv` (ground truth) plus `data/twcs.csv` if you want the optional TWCS visuals (download it manually from Kaggle: Customer Support on Twitter).

1) Create and activate a virtual env, then install deps:
   - Conda:
     ```
     conda create -n cs-support-llm python=3.10 -y
     conda activate cs-support-llm
     ```
   - venv:
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   Then install:
   ```
   pip install -r requirements.txt
   ```
2) Place `manual_labels.csv` in `data/` (already committed for convenience). If you also want to regenerate TWCS visuals, place `twcs.csv` in `data/`.
3) Train/evaluate all five models on manual labels:
   ```
   python src/main.py
   ```
   Metrics for each model/task are printed and saved to `results_summary.csv`.

Visualize the saved metrics (accuracy/precision/recall/F1 for each model and task):
```
python src/visualize_results_summary.py
```
PNG plots land in `figures/`.

### Generate TWCS dataset visuals
Quick EDA plots for the raw `twcs.csv` live in `figures/` (monthly volume, inbound/outbound split, top support handles, inbound tweet lengths):
```
python src/twcs_visualization.py
```
Pass `--chunk-size` if you want to override the default 200k rows per batch.
