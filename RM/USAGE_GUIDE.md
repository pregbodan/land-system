# COMPLETE USAGE GUIDE
## Land Matter Prediction System

---

## 📖 Table of Contents
1. [Installation & Setup](#installation--setup)
2. [Data Preparation](#data-preparation)
3. [Running the Complete Pipeline](#running-the-complete-pipeline)
4. [Making Predictions](#making-predictions)
5. [Python API Usage](#python-api-usage)
6. [Command Line Usage](#command-line-usage)
7. [Example Workflows](#example-workflows)

---

## 1. Installation & Setup

### Step 1: Extract the Code
```bash
# Extract the archive
tar -xzf land_matter_prediction_system_code.tar.gz
cd land_matter_prediction_system
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 4: Verify Installation
```bash
python -c "import sklearn, xgboost, nltk, pandas; print('✓ All packages installed successfully!')"
```

---

## 2. Data Preparation

### Option A: Using Provided Sample Data

If sample data is provided:
```bash
# Sample data should be in data/raw/
ls data/raw/

# Expected output:
# SC_123_2020.pdf
# SC_456_2021.pdf
# ...
```

### Option B: Adding Your Own Data

```bash
# Create data directory
mkdir -p data/raw

# Copy your PDF/DOCX judgment files
cp /path/to/your/judgments/*.pdf data/raw/
cp /path/to/your/judgments/*.docx data/raw/

# Verify files
ls -lh data/raw/
```

**Requirements for judgment files:**
- ✅ PDF or DOCX format
- ✅ Actual court judgments (not case summaries)
- ✅ Minimum 100 cases (500+ recommended)
- ✅ Land-related matters
- ✅ Complete judgments with decisions

---

## 3. Running the Complete Pipeline

### Step 1: Preprocessing

```bash
# Run preprocessing on all files in data/raw/
python src/main_preprocessing.py \
    --data_dir data/raw \
    --output_file data/processed/preprocessed_data.pkl
```

**What happens:**
1. Extracts text from PDF/DOCX files
2. Cleans and standardizes text
3. Parses into sections (facts, issues, decision)
4. Labels outcomes (success/failure)
5. Categorizes dispute types
6. Extracts features (TF-IDF + legal features)
7. Splits into train/test sets

**Output files:**
- `data/processed/processed_cases.csv` - Human-readable intermediate results
- `data/processed/preprocessed_data.pkl` - ML-ready dataset

**Expected time:** 2-5 minutes per 100 cases

### Step 2: Training Models

```bash
# Train all ML models
python src/main_training.py \
    --data_file data/processed/preprocessed_data.pkl \
    --output_dir data/models \
    --cv_folds 5
```

**What happens:**
1. Loads preprocessed data
2. Trains 5 models:
   - Random Forest
   - XGBoost
   - SVM
   - Naive Bayes
   - Gradient Boosting
3. Evaluates each model
4. Performs cross-validation
5. Saves all models and results

**Output files:**
- `data/models/random_forest.pkl`
- `data/models/xgboost.pkl`
- `data/models/svm.pkl`
- `data/models/naive_bayes.pkl`
- `data/models/gradient_boosting.pkl`
- `data/models/best_model.pkl` - Best performing model
- `data/models/training_results.json` - Performance metrics
- `data/models/model_metadata.json` - Model info
- `data/models/cv_results.json` - Cross-validation results

**Expected time:** 5-15 minutes depending on data size

### Step 3: View Results

```bash
# View training results
cat data/models/training_results.json

# View best model info
cat data/models/model_metadata.json
```

---

## 4. Making Predictions

### Predict from File

```bash
# Predict outcome for a new judgment
python src/predict.py \
    --file path/to/new_judgment.pdf \
    --model data/models/best_model.pkl \
    --data data/processed/preprocessed_data.pkl
```

**Example output:**
```
============================================================
PREDICTION RESULT
============================================================

🎯 PREDICTED OUTCOME: PLAINTIFF/APPELLANT LIKELY TO SUCCEED
   Confidence: 78.5%

📊 PROBABILITIES:
   PLAINTIFF_FAILURE: 21.5%
   PLAINTIFF_SUCCESS: 78.5%

📋 DISPUTE CATEGORIES:
   • Ownership Dispute
   • Governor's Consent

⚖️  LEGAL FACTORS DETECTED:
   ✓ Certificate of Occupancy mentioned
   ✓ Governor's Consent mentioned
   ✓ Land Use Act 1978 referenced
   ✓ Survey evidence present

============================================================
⚠️  DISCLAIMER:
This is a predictive model based on historical patterns.
It should be used as a decision support tool, not a
definitive answer. Always consult with legal professionals.
============================================================
```

---

## 5. Python API Usage

### Example 1: Complete Pipeline in Python

```python
# Import pipeline
from src.main_preprocessing import PreprocessingPipeline
from src.models.model_training import ModelTrainer

# Configuration
config = {
    'test_size': 0.2,
    'random_state': 42,
    'tfidf_max_features': 1000
}

# Run preprocessing
pipeline = PreprocessingPipeline(config)
ml_dataset = pipeline.run('data/raw', 'data/processed/preprocessed_data.pkl')

# Train models
trainer = ModelTrainer(random_state=42)
results = trainer.train_all_models(
    ml_dataset['X_train'],
    ml_dataset['y_train'],
    ml_dataset['X_test'],
    ml_dataset['y_test']
)

# Get best model
best_name, best_model, best_acc = trainer.get_best_model()
print(f"Best model: {best_name} ({best_acc:.2%} accuracy)")
```

### Example 2: Making Predictions

```python
from src.predict import LandMatterPredictor

# Initialize predictor
predictor = LandMatterPredictor(
    model_path='data/models/best_model.pkl',
    preprocessed_data_path='data/processed/preprocessed_data.pkl'
)

# Predict from file
result = predictor.predict_from_file('new_judgment.pdf')

# Print result
print(result['predicted_outcome'])
print(f"Confidence: {result['confidence']:.2%}")
print(f"Categories: {', '.join(result['categories'])}")

# Get explanation
explanation = predictor.explain_prediction(result)
print(explanation)
```

### Example 3: Processing Individual Components

```python
# Text extraction
from src.data_collection.text_extraction import TextExtractor

extractor = TextExtractor()
text = extractor.extract('judgment.pdf')
print(f"Extracted {len(text)} characters")

# Text cleaning
from src.preprocessing.text_preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned_text = preprocessor.clean(text)
print(f"Cleaned text: {cleaned_text[:200]}...")

# Parse sections
from src.preprocessing.text_preprocessing import JudgmentParser

parser = JudgmentParser()
sections = parser.parse_sections(cleaned_text)
print(f"Facts: {sections['facts'][:200]}...")
print(f"Decision: {sections['decision'][:200]}...")

# Annotation
from src.preprocessing.annotation import CaseAnnotator

annotator = CaseAnnotator()
annotations = annotator.annotate_case(cleaned_text, sections)
print(f"Outcome: {annotations['outcome']}")
print(f"Categories: {annotations['categories']}")
```

### Example 4: Custom Model Training

```python
from src.models.model_training import ModelTrainer
import pickle

# Load data
with open('data/processed/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# Initialize trainer
trainer = ModelTrainer(random_state=42)

# Train specific model with custom parameters
model = trainer.train_random_forest(
    X_train, y_train,
    n_estimators=200,  # More trees
    max_depth=20,      # Deeper trees
    min_samples_split=5
)

# Evaluate
results = trainer.evaluate_model(model, X_test, y_test, "Custom RF")

# Cross-validate
from scipy.sparse import vstack
import numpy as np

X_full = vstack([X_train, X_test])
y_full = np.concatenate([y_train, y_test])

cv_results = trainer.cross_validate(model, X_full, y_full, cv=10)
```

---

## 6. Command Line Usage

### All Available Commands

```bash
# 1. Preprocessing
python src/main_preprocessing.py \
    --data_dir data/raw \
    --output_file data/processed/preprocessed_data.pkl

# 2. Training
python src/main_training.py \
    --data_file data/processed/preprocessed_data.pkl \
    --output_dir data/models \
    --cv_folds 5

# 3. Prediction
python src/predict.py \
    --file path/to/judgment.pdf \
    --model data/models/best_model.pkl \
    --data data/processed/preprocessed_data.pkl
```

### Get Help

```bash
# Preprocessing help
python src/main_preprocessing.py --help

# Training help
python src/main_training.py --help

# Prediction help
python src/predict.py --help
```

---

## 7. Example Workflows

### Workflow 1: Quick Start (Minimum Steps)

```bash
# 1. Place judgment files in data/raw/
# 2. Run preprocessing
python src/main_preprocessing.py

# 3. Train models
python src/main_training.py

# 4. Make prediction
python src/predict.py --file new_case.pdf

# Done! ✅
```

### Workflow 2: Research & Experimentation

```python
# research_notebook.py

import pandas as pd
import pickle
from src.main_preprocessing import PreprocessingPipeline
from src.models.model_training import ModelTrainer

# 1. Load and explore data
pipeline = PreprocessingPipeline()
df = pipeline.process_directory('data/raw')

print(f"Total cases: {len(df)}")
print(f"Outcomes:\n{df['outcome'].value_counts()}")
print(f"Categories:\n{df['primary_category'].value_counts()}")

# 2. Prepare dataset
ml_dataset = pipeline.prepare_ml_dataset(df)

# 3. Train multiple configurations
configs = [
    {'n_estimators': 50},
    {'n_estimators': 100},
    {'n_estimators': 200},
]

results_comparison = []

for i, config in enumerate(configs):
    trainer = ModelTrainer()
    model = trainer.train_random_forest(
        ml_dataset['X_train'],
        ml_dataset['y_train'],
        **config
    )
    
    metrics = trainer.evaluate_model(
        model,
        ml_dataset['X_test'],
        ml_dataset['y_test'],
        f"RF_config_{i}"
    )
    
    results_comparison.append({
        'config': config,
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score']
    })

# 4. Find best configuration
best = max(results_comparison, key=lambda x: x['accuracy'])
print(f"Best config: {best}")
```

### Workflow 3: Production Deployment

```python
# production_predict.py

from src.predict import LandMatterPredictor
import sys

def predict_case(file_path):
    """
    Production-ready prediction function
    """
    try:
        # Initialize predictor
        predictor = LandMatterPredictor(
            model_path='data/models/best_model.pkl',
            preprocessed_data_path='data/processed/preprocessed_data.pkl'
        )
        
        # Make prediction
        result = predictor.predict_from_file(file_path)
        
        # Format output
        output = {
            'success': True,
            'prediction': result['predicted_outcome'],
            'confidence': f"{result['confidence']*100:.1f}%",
            'categories': result['categories'],
            'warnings': []
        }
        
        # Add warnings for low confidence
        if result['confidence'] < 0.6:
            output['warnings'].append('Low confidence prediction - review carefully')
        
        return output
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python production_predict.py <file_path>")
        sys.exit(1)
    
    result = predict_case(sys.argv[1])
    print(result)
```

---

## 📝 Notes

1. **First Time Setup:** Takes ~30 minutes (installation + downloading data)
2. **Preprocessing:** ~2-5 minutes per 100 cases
3. **Training:** ~5-15 minutes for all 5 models
4. **Prediction:** ~5-10 seconds per case

5. **Minimum Dataset:** 100 cases (500+ recommended)
6. **Expected Accuracy:** 70-80% (matches Sobowale et al. 2024)

7. **GPU:** Not required (runs on CPU)
8. **RAM:** 8GB minimum (16GB recommended)

---

**For more information, see README.md**
