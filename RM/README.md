# Land Matter Prediction System
## Machine Learning Predictive System for Outcomes of Land Matters in Nigerian Courts

**Author:** Ademola Daniel Adebomi (CPE/2020/1008)  
**Department:** Computer Engineering, University of Oye Ekiti  
**Supervisor:** Dr A.A. Shobowale Esq

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Quick Start Guide](#quick-start-guide)
6. [Detailed Usage](#detailed-usage)
7. [Model Performance](#model-performance)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This system uses Traditional Machine Learning to predict outcomes of land matter cases in Nigerian courts. It achieves **70%+ accuracy** by analyzing historical court judgments and extracting patterns.

**Key Features:**
- ✅ Processes PDF and DOCX court judgments
- ✅ Extracts legal features from Land Use Act 1978 and Evidence Act 2011
- ✅ Trains 5 ML algorithms (Random Forest, XGBoost, SVM, Naive Bayes, Gradient Boosting)
- ✅ Uses a hybrid predictor: best 3 trained ML models + offline Ollama (`llama3`) with default 80/20 blend
- ✅ Provides explainable predictions
- ✅ Includes legal compliance checking
- ✅ Cites related trained cases for each prediction
- ✅ Supports requester profiles: `regular_user`, `lawyer`, `judge`
- ✅ Stores feedback cases for future retraining (`data/models/prediction_feedback.jsonl`)

---

## 💻 System Requirements

### Minimum Requirements:
- **OS:** Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
- **RAM:** 8GB (16GB recommended)
- **Storage:** 5GB free space
- **Python:** 3.8, 3.9, or 3.10 (NOT 3.11+)

### Optional (for OCR):
- Tesseract OCR installed (for scanned PDFs)

---

## 🚀 Installation

### Step 1: Install Python

**Windows:**
```bash
# Download Python 3.10 from python.org
# During installation, check "Add Python to PATH"
```

**macOS:**
```bash
brew install python@3.10
```

**Ubuntu/Linux:**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

### Step 2: Create Virtual Environment

```bash
# Navigate to project directory
cd land_matter_prediction_system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 4: (Optional) Install Tesseract OCR

**Windows:**
```
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
Install and add to PATH
```

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Linux:**
```bash
sudo apt install tesseract-ocr
```

---

## 📁 Project Structure

```
land_matter_prediction_system/
│
├── config.py                    # Configuration file
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── data/
│   ├── raw/                     # Place your PDF/DOCX judgments here
│   ├── processed/               # Processed data files
│   ├── features/                # Extracted features
│   └── models/                  # Trained models
│
├── src/
│   ├── data_collection/
│   │   └── text_extraction.py  # PDF/DOCX text extraction
│   │
│   ├── preprocessing/
│   │   ├── text_preprocessing.py  # Text cleaning
│   │   └── annotation.py          # Labeling outcomes
│   │
│   ├── feature_engineering/
│   │   └── feature_engineering.py  # Feature extraction
│   │
│   ├── models/
│   │   └── model_training.py    # ML model training
│   │
│   ├── compliance/
│   │   └── legal_compliance.py  # Legal rules checking
│   │
│   ├── evaluation/
│   │   └── model_evaluation.py  # Performance metrics
│   │
│   ├── web_app/
│   │   ├── app.py              # Flask web application
│   │   ├── templates/          # HTML templates
│   │   └── static/             # CSS/JS files
│   │
│   └── main_preprocessing.py   # Main preprocessing pipeline
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Jupyter notebook for analysis
│
└── tests/
    └── test_models.py          # Unit tests
```

---

## ⚡ Quick Start Guide

Follow this sequence from the project root (`land-system`) to use the system correctly.

### Step 1: Activate environment and install dependencies

Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -m spacy download en_core_web_sm
```

macOS/Linux:
```bash
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -m spacy download en_core_web_sm
```

### Step 2: Create the required folders

Windows (PowerShell):
```powershell
New-Item -ItemType Directory -Force data | Out-Null
New-Item -ItemType Directory -Force data\raw | Out-Null
New-Item -ItemType Directory -Force data\structured | Out-Null
New-Item -ItemType Directory -Force data\processed | Out-Null
New-Item -ItemType Directory -Force data\models | Out-Null
```

macOS/Linux:
```bash
mkdir -p data/raw data/structured data/processed data/models
```

### Step 3: Choose your input path

#### Path A: Train from PDF or DOCX judgments

1. Move all judgment files to `data/raw/`.
2. Example filenames:
`data/raw/SC_123_2020.pdf`
`data/raw/SC_456_2021.docx`
3. Run preprocessing:

```bash
python src/main_preprocessing.py --data_dir data/raw --output_file data/processed/preprocessed_data.pkl
```

4. Train models:

```bash
python src/main_training.py --data_file data/processed/preprocessed_data.pkl --output_dir data/models
```

5. Expected outputs:
- `data/processed/processed_cases.csv`
- `data/processed/preprocessed_data.pkl`
- `data/models/best_model.pkl`
- `data/models/inference_artifacts.pkl`
- `data/models/model_metadata.json`
- `data/models/training_results.json`
- `data/models/cv_results.json`
- `data/models/case_reference_records.json`

#### Path B: Train directly from CSV

1. Move your CSV file to `data/structured/` (example: `data/structured/cases.csv`).
2. Ensure required columns are present:
- Outcome label: `outcome` (or alias: `label`, `target`, `outcome_label`, `case_outcome`)
- Text content: `full_text` or `text`
- If `full_text` and `text` are missing, include at least one of:
`facts`, `issues`, `arguments`, `analysis`, `decision`
3. Train directly:

```bash
python src/main_training.py --data_file data/structured/cases.csv --output_dir data/models
```

4. Optional (build reusable preprocessed file):

```bash
python src/main_preprocessing.py --input_file data/structured/cases.csv --output_file data/processed/preprocessed_data.pkl
```

#### Path C: Train directly from JSON

1. Move your JSON file to `data/structured/` (example: `data/structured/cases.json`).
2. Supported JSON formats:
- JSON array of objects
- JSON Lines (one object per line)
3. Use the same required columns as CSV.
4. Train directly:

```bash
python src/main_training.py --data_file data/structured/cases.json --output_dir data/models
```

5. Optional (build reusable preprocessed file):

```bash
python src/main_preprocessing.py --input_file data/structured/cases.json --output_file data/processed/preprocessed_data.pkl
```

#### Path D: Train from multiple structured files with different schemas

The loader normalizes aliases and merges files by column name. Missing columns are auto-filled.

Example with your two files:

```bash
python src/main_training.py --data_file data/structured/cases.csv data/structured/scn-cases.csv --output_dir data/models
```

Notes:
- Duplicate columns like `outcome` and `outcome.1` are resolved automatically.
- Aliases like `judgment`/`judgement` are mapped to `decision`.
- If `full_text` is missing and `text` exists, `text` is used.
- Outcome labels are normalized to lowercase during load (for example `Dismissed` -> `dismissed`).
- Outcome labels are canonicalized to: `granted`, `dismissed`, `matter remitted`, `other`.
- Rare classes with fewer than 2 samples are auto-mapped to `other` to avoid split errors.
- Training now logs a `Land-signal ratio` health check; very low ratios mean your dataset is likely mixed-domain.
- If you hit Windows process permission issues, run with `--n_jobs 1`.
- To learn from validated new cases, save feedback via API (`POST /api/feedback`) and retrain with:
`python src/main_training.py --data_file data/structured/cases.csv data/structured/scn-cases.csv --output_dir data/models --include_feedback --feedback_file data/models/prediction_feedback.jsonl`

### Step 4: Validate outputs

After any path above, confirm these files exist in `data/models/`:
- `best_model.pkl`
- `inference_artifacts.pkl`
- `model_metadata.json`
- `training_results.json`
- `cv_results.json`

Optional evaluation:
```bash
python src/evaluation/model_evaluation.py
```

### Step 5: Run web app (optional)

```bash
python src/web_app/app.py
```

By default, prediction now runs in **hybrid mode**:
- Best 3 trained ML models (from `training_results.json`) are ensembled.
- Final probability is blended as **80% ML + 20% offline Ollama (`llama3`)**.
- Statute retrieval uses both:
  - `data/legal/Nigerian-land-use-act-2004.pdf`
  - `data/legal/evidence-act-2011.pdf`

Open:
- `http://127.0.0.1:5000/` (frontend)
- `http://127.0.0.1:5000/health` (system health)
- `http://127.0.0.1:5000/api/results` (training artifacts)

Optional Ollama setup (for explanation generation):
```bash
ollama serve
ollama pull llama3
```

Optional environment variables:
```bash
# Windows PowerShell examples
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:OLLAMA_MODEL="llama3"
$env:LAND_MODEL_PATH="data/models/best_model.pkl"
$env:LAND_ARTIFACTS_PATH="data/models/inference_artifacts.pkl"
$env:LAND_ACT_PDF_PATH="data\\legal\\Nigerian-land-use-act-2004.pdf"
$env:EVIDENCE_ACT_PDF_PATH="data\\legal\\evidence-act-2011.pdf"
$env:TOP_MODELS_COUNT="3"
$env:HYBRID_ML_WEIGHT="0.8"
$env:HYBRID_OLLAMA_WEIGHT="0.2"
$env:LAND_ACT_URLS="https://example.com/land-use-act-text"
$env:EVIDENCE_ACT_URLS="https://example.com/evidence-act-text"
$env:RETRAIN_DATA_FILES="data/structured/cases.csv,data/structured/scn-cases.csv"
$env:RETRAIN_FEEDBACK_FILE="data/models/prediction_feedback.jsonl"
python src/web_app/app.py
```

Statute endpoints:
- `GET /api/statute/sections` (lists all loaded sections)
- `GET /api/statute/sections?q=section 22 governor consent` (related sections for query)
- `GET /api/cases/related?q=declaration of title and trespass` (related trained cases)
- `GET /api/requester/roles` (input guidance by requester profile)
- `POST /api/feedback` (store new case feedback for future retraining)
- `POST /api/retrain/feedback` (start retraining with feedback file)
- `GET /api/retrain/status` (check retraining progress and logs)

Statute CLI examples:
```bash
# List all loaded statute sections (Land Use Act + Evidence Act)
python src/predict.py --model data/models/best_model.pkl --artifacts data/models/inference_artifacts.pkl --statute_pdf data/legal/Nigerian-land-use-act-2004.pdf --statute_pdf data/legal/evidence-act-2011.pdf --list_statute_sections --json

# Predict with hybrid mode (default 80/20) and both statutes
python src/predict.py --text "Declaration of title and trespass under section 22; documentary evidence under section 84" --requester_role lawyer --statute_pdf data/legal/Nigerian-land-use-act-2004.pdf --statute_pdf data/legal/evidence-act-2011.pdf --ml_weight 0.8 --ollama_weight 0.2 --ollama_model llama3 --json
```

Requester profiles:
- `regular_user`: accepts plain language (non-legal narration)
- `lawyer`: expects legal structure (parties, issues, reliefs, holding)
- `judge`: expects issue-analysis-order style

Important:
- Related-case citation uses the trained-case index saved in `inference_artifacts.pkl`.
- If citations are empty, retrain once so `case_reference_records` are generated.
- UI now includes **Retrain From Feedback** button in Training Summary.

### Structured data notes

- Minimum recommended fields: `outcome`, `full_text`, `year`, `court`.
- Rows with `outcome=UNCLEAR` are removed before training.

---

## 📊 Expected Performance

Based on Sobowale et al. (2024) and our methodology:

| Model | Expected Accuracy | Strengths |
|-------|------------------|-----------|
| **XGBoost** | 72-80% | Highest accuracy |
| **Random Forest** | 70-78% | Most interpretable |
| **SVM** | 65-75% | Good for text data |
| **Naive Bayes** | 60-70% | Fast baseline |
| **Gradient Boosting** | 70-78% | Good alternative |

---

## 🔧 Detailed Usage

### Processing Individual Files

```python
from src.data_collection.text_extraction import TextExtractor
from src.preprocessing.text_preprocessing import TextPreprocessor

# Extract text
extractor = TextExtractor()
text = extractor.extract("path/to/judgment.pdf")

# Clean text
preprocessor = TextPreprocessor()
cleaned_text = preprocessor.clean(text)

print(cleaned_text)
```

### Custom Feature Engineering

```python
from src.feature_engineering.feature_engineering import FeatureEngineer

# Initialize
engineer = FeatureEngineer(max_features=1000)

# Create features
X, feature_names = engineer.engineer_features(df, fit=True)

print(f"Feature matrix shape: {X.shape}")
print(f"Top 10 features: {feature_names[:10]}")
```

### Training Individual Models

```python
from src.models.model_training import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(random_state=42)

# Train specific model
model = trainer.train_random_forest(X_train, y_train)

# Evaluate
results = trainer.evaluate_model(model, X_test, y_test, "Random Forest")

# Cross-validate
cv_results = trainer.cross_validate(model, X, y, cv=5)
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'sklearn'"**
```bash
# Solution: Install scikit-learn
pip install scikit-learn==1.3.0
```

**2. "PDF extraction returns empty text"**
```bash
# Solution: Try OCR extraction
from src.data_collection.text_extraction import TextExtractor
extractor = TextExtractor()
text = extractor.extract_with_ocr("path/to/scanned.pdf")
```

**3. "Memory Error during training"**
```python
# Solution: Reduce max_features
engineer = FeatureEngineer(max_features=500)  # Instead of 1000
```

**4. "ImportError: DLL load failed" (Windows)**
```bash
# Solution: Install Microsoft Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

**5. "No cases with valid outcomes"**
- Check that your PDF files contain actual court judgments
- Verify text extraction is working: `python -c "from src.data_collection.text_extraction import TextExtractor; e=TextExtractor(); print(e.extract('your_file.pdf')[:500])"`

**6. "The least populated class in y has only 1 member"**
```bash
# Use current training command; rare classes are auto-mapped to "other"
python src/main_training.py --data_file data/structured/cases.csv data/structured/scn-cases.csv --output_dir data/models

# If Windows permission errors occur during workers/processes:
python src/main_training.py --data_file data/structured/cases.csv data/structured/scn-cases.csv --output_dir data/models --n_jobs 1
```

---

## 📚 Additional Resources

### Documentation
- **Scikit-learn:** https://scikit-learn.org/
- **XGBoost:** https://xgboost.readthedocs.io/
- **NLTK:** https://www.nltk.org/
- **cPanel Shared Hosting Deployment:** `README_CPANEL_SHARED_HOSTING.md`

### Research References
- Sobowale et al. (2024) - Performance Evaluation of ML for Judicial Prediction
- Nwobike et al. (2024) - AI for Judicial Precedent in Nigeria
- Bello & Ogufere (2024/2025) - Nigeria's Judicial System Readiness for AI

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review error logs in `logs/system.log`
3. Contact: [Your Email]

---

## 📄 License

This project is for academic research purposes.  
University of Oye Ekiti - Final Year Project  
Department of Computer Engineering

---

## 🙏 Acknowledgments

- **Supervisor:** [Name]
- **Department:** Computer Engineering, University of Oye Ekiti
- **Research Foundation:** Sobowale et al. (2024) - Supreme Court Nigeria ML Dataset

---

**Version:** 1.0.0  
**Last Updated:** March 2026  
**Status:** Development Complete ✅
