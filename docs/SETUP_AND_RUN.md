# Setup and Run Guide

This document provides step-by-step instructions for setting up and running the LLM-Aided Customer Support Automation system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Understanding the Output](#understanding-the-output)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **Memory**: At least 4GB RAM (8GB+ recommended for deep learning models)
- **Storage**: At least 2GB free space for dependencies and models
- **Internet Connection**: Required for downloading dependencies and API calls

### Required Accounts

1. **Google Gemini API Key**: 
   - Sign up at [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create an API key for Gemini
   - Note: Free tier has rate limits

2. **Kaggle Account** (for dataset):
   - Sign up at [Kaggle](https://www.kaggle.com/)
   - Download the dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

---

## Installation

### Step 1: Clone or Download the Project

If you have the project in a repository:
```bash
git clone <repository-url>
cd em_final_proj
```

Or if you have the project folder, navigate to it:
```bash
cd /path/to/em_final_proj
```

### Step 2: Create a Virtual Environment (Recommended)

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
```

**Note**: This may take several minutes as it installs:
- Machine learning libraries (scikit-learn, TensorFlow, PyTorch)
- NLP libraries (transformers)
- Visualization libraries (matplotlib, seaborn)
- Other dependencies

**Troubleshooting Installation:**

If you encounter issues with TensorFlow or PyTorch:
- **TensorFlow**: May require specific versions for your system. Check [TensorFlow installation guide](https://www.tensorflow.org/install)
- **PyTorch**: Visit [PyTorch website](https://pytorch.org/get-started/locally/) for system-specific installation

If you don't need deep learning models (LSTM/BERT), you can skip TensorFlow/PyTorch:
```bash
pip install pandas scikit-learn python-dotenv google-genai numpy matplotlib seaborn joblib nltk regex
```

### Step 4: Download the Dataset

1. Download `twcs.csv` from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
2. Place it in the `data/` directory:
   ```bash
   mkdir -p data
   # Move twcs.csv to data/twcs.csv
   ```

**Alternative**: If you already have the dataset elsewhere, you can modify the path in `config.py`:
```python
DATA_PATH = Path("/your/path/to/twcs.csv")
```

---

## Configuration

### Step 1: Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
touch .env
```

Add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

**Important**: 
- Never commit the `.env` file to version control
- Replace `your_api_key_here` with your actual API key
- The `.env` file is already in `.gitignore`

### Step 2: Configure System Parameters (Optional)

Edit `config.py` to customize settings:

**Key Configuration Options:**

```python
# Number of samples to label (affects API usage and training time)
N_LABEL_SAMPLES = 50  # Increase for better model performance

# Model selection
BERT_MODEL_NAME = "distilbert-base-uncased"  # Can use "bert-base-uncased" for better accuracy

# Training parameters
EPOCHS = 10  # For LSTM models
BERT_EPOCHS = 3  # For BERT models

# API rate limiting
API_DELAY = 0.1  # Seconds between API calls
```

**For First-Time Users**: Default settings are fine. You can adjust later based on your needs.

---

## Running the System

### Basic Run

The simplest way to run the complete pipeline:

```bash
python main.py
```

This will:
1. Load the dataset
2. Label samples using Gemini API
3. Augment data for underrepresented classes
4. Preprocess the data
5. Train baseline models (Logistic Regression)
6. Evaluate models
7. Generate visualizations
8. Test on sample tweets

**Expected Runtime**: 
- With 50 samples: ~5-10 minutes (depends on API response time)
- Most time is spent on API calls for labeling

### Model Selection

To train different model types, edit `main.py` line ~410:

```python
# Train only baseline models (fastest, default)
model_type="baseline"

# Train LSTM models (requires TensorFlow)
model_type="lstm"

# Train BERT models (requires transformers)
model_type="bert"

# Train all models (slowest, most comprehensive)
model_type="all"
```

### Hyperparameter Tuning

To enable hyperparameter tuning, edit `main.py` line ~411:

```python
tune_hyperparameters=True  # Set to True for grid search
```

**Note**: This significantly increases training time but may improve performance.

---

## Understanding the Output

### Console Output

The system provides detailed logging:

```
[Step 1] Loading dataset...
Loaded 1000 customer tweets from data/twcs.csv

[Step 2] Creating labeled subset with Gemini...
Labeling 50 samples with Gemini...
  Processed 5/50 rows... (current labeled: 4)
  Processed 10/50 rows... (current labeled: 9)
  ...

[Step 3] Data augmentation...
[Step 4] Preprocessing data...
[Step 5] Training models...
[Step 6] Evaluating models...
[Step 7] Testing pipeline on sample tweets...
```

### Generated Files

After running, you'll find:

**Models** (in `models/` directory):
- `intent_baseline.joblib` - Intent classification model
- `severity_baseline.joblib` - Severity classification model

**Visualizations** (in `figures/` directory):
- `data_distribution.png` - Label distribution plots
- `confusion_matrix_intent.png` - Intent confusion matrix
- `confusion_matrix_severity.png` - Severity confusion matrix
- `roc_curve_intent.png` - ROC curve for intent
- `roc_curve_severity.png` - ROC curve for severity
- `error_analysis_intent.png` - Error analysis for intent
- `error_analysis_severity.png` - Error analysis for severity
- `model_comparison_accuracy.png` - Model comparison chart
- `metrics_comparison.png` - Comprehensive metrics comparison

**Results**:
- `results_summary.csv` - Detailed metrics table

### Evaluation Metrics

The system reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: Macro and weighted averages
- **Recall**: Macro and weighted averages
- **F1-Score**: Macro and weighted averages
- **ROC-AUC**: Area under ROC curve (if probabilities available)
- **Confusion Matrix**: Per-class performance

**Target**: Intent classifier should achieve ≥80% accuracy (per project requirements)

---

## Troubleshooting

### Common Issues

#### 1. API Key Error

**Error**: `ValueError: Please set GEMINI_API_KEY in your .env file.`

**Solution**:
- Check that `.env` file exists in project root
- Verify API key is correctly formatted: `GEMINI_API_KEY=your_key_here`
- Ensure no extra spaces around the `=`

#### 2. Dataset Not Found

**Error**: `FileNotFoundError: Data file not found`

**Solution**:
- Verify `data/twcs.csv` exists
- Check path in `config.py` matches your file location
- Ensure file name is exactly `twcs.csv` (case-sensitive)

#### 3. Import Errors

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- For TensorFlow/PyTorch issues, see installation troubleshooting above

#### 4. API Quota Exceeded

**Error**: `Hit Gemini quota; stopping further labeling.`

**Solution**:
- Reduce `N_LABEL_SAMPLES` in `config.py`
- Increase `API_DELAY` to slow down API calls
- Wait and try again later (free tier has daily limits)
- Consider upgrading API tier

#### 5. Out of Memory

**Error**: Memory errors during model training

**Solution**:
- Reduce `N_LABEL_SAMPLES` (fewer training samples)
- Use only baseline models (skip LSTM/BERT)
- Reduce `TFIDF_MAX_FEATURES` in `config.py`
- Close other applications

#### 6. Empty Results

**Error**: `No labeled examples were created`

**Solution**:
- Check API key is valid
- Verify internet connection
- Check API quota hasn't been exceeded
- Review API response in logs

#### 7. Visualization Errors

**Error**: Issues generating plots

**Solution**:
- Ensure matplotlib and seaborn are installed
- For headless servers, set: `export MPLBACKEND=Agg`
- Check disk space for saving figures

### Getting Help

If issues persist:
1. Check logs for detailed error messages
2. Review `IMPROVEMENTS.md` for known issues
3. Verify all prerequisites are met
4. Ensure you're using Python 3.8+

---

## Advanced Usage

### Running Specific Components

#### Test API Connection

Create a simple test script:

```python
# test_api.py
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Say hi",
)
print(response.text)
```

Run: `python test_api.py`

#### Load and Use Trained Models

```python
import joblib
from models import BaselineModel

# Load models
intent_model = BaselineModel("intent")
intent_model.load("models/intent_baseline.joblib")

# Make predictions
prediction = intent_model.predict(["My account is locked"])
print(prediction)
```

#### Generate Visualizations Only

```python
from visualizations import VisualizationGenerator
import pandas as pd
from sklearn.metrics import confusion_matrix

# Load your results
y_true = [...]  # Your true labels
y_pred = [...]  # Your predictions

# Generate visualizations
viz = VisualizationGenerator()
viz.plot_confusion_matrix(y_true, y_pred, save=True)
```

### Custom Configuration

#### Adjust Training Parameters

Edit `config.py`:

```python
# For faster training (lower quality)
N_LABEL_SAMPLES = 20
EPOCHS = 5

# For better quality (slower)
N_LABEL_SAMPLES = 100
EPOCHS = 20
BERT_EPOCHS = 5
```

#### Custom Preprocessing

Modify `preprocessing.py` to add custom text cleaning:

```python
def custom_preprocess(text: str) -> str:
    # Your custom preprocessing
    text = text.lower()
    # ... more steps
    return text
```

### Batch Processing

To process multiple tweets:

```python
from main import handle_new_tweet, load_customer_support_data
import llm_integration
import models

# Load models
intent_model = models.BaselineModel("intent")
intent_model.load("models/intent_baseline.joblib")
# ... load severity model

# Process batch
llm = llm_integration.LLMIntegration()
tweets = ["Tweet 1", "Tweet 2", "Tweet 3"]

for tweet in tweets:
    result = handle_new_tweet(tweet, intent_model, severity_model, llm)
    print(result)
```

---

## Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded and placed in `data/twcs.csv`
- [ ] `.env` file created with `GEMINI_API_KEY`
- [ ] Run `python main.py`
- [ ] Check `figures/` directory for visualizations
- [ ] Review `results_summary.csv` for metrics

---

## Next Steps

After successful setup:

1. **Review Results**: Check generated visualizations and metrics
2. **Experiment**: Try different model types and parameters
3. **Present**: Use visualizations for your presentation
4. **Improve**: Adjust parameters based on results

For more information:
- See `README.md` for project overview
- See `IMPROVEMENTS.md` for recent changes
- See `GRADING_RUBRIC_CHANGES.md` for rubric compliance

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review error messages in logs
3. Verify configuration matches examples
4. Ensure all prerequisites are met

Good luck with your project! 🚀

