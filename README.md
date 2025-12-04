# LLM-Aided Customer Support Automation

Classifies customer tweets by intent and severity (0–3) and drafts brief, empathetic replies using Gemini. Models include a fast TF‑IDF + Logistic Regression baseline, with optional LSTM and DistilBERT tiers.

## How to Run
Prereqs: Python 3.10+, a virtual environment, Gemini API key, and the dataset file `data/twcs.csv` (download it manually from Kaggle: Customer Support on Twitter).

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
2) Create `.env` in the repo root with:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
3) Place `twcs.csv` in `data/` (no download commands here).
4) From the repo root, run:
   ```
   python src/main.py
   ```

The script labels a subset with Gemini (size set in `src/config.py`), preprocesses, trains, evaluates, saves figures to `figures/`, and stores trained models in `models/`.

### Generate TWCS dataset visuals
Quick EDA plots for the raw `twcs.csv` live in `figures/` (monthly volume, inbound/outbound split, top support handles, inbound tweet lengths):
```
python src/twcs_visualization.py
```
Pass `--chunk-size` if you want to override the default 200k rows per batch.
