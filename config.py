"""
Configuration file for Land Matter Prediction System
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
FEATURES_DIR = DATA_DIR / 'features'
MODELS_DIR = DATA_DIR / 'models'

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
}

# Feature engineering parameters
FEATURE_CONFIG = {
    'tfidf_max_features': 1000,
    'tfidf_min_df': 2,
    'tfidf_max_df': 0.8,
    'tfidf_ngram_range': (1, 2),
}

# Legal compliance rules
LAND_USE_ACT_SECTIONS = {
    'section_22': "Governor's Consent required for land transfers",
    'section_28': "Revocation of rights of occupancy",
}

EVIDENCE_ACT_SECTIONS = {
    'section_84': "Documentary evidence must be properly authenticated",
}

# Land dispute categories
LAND_DISPUTE_CATEGORIES = [
    'OWNERSHIP_DISPUTE',
    'INHERITANCE',
    'BOUNDARY',
    'TRESPASS',
    'GOVERNORS_CONSENT',
    'REVOCATION',
    'COMPENSATION',
    'LAND_USE_VIOLATION',
    'OTHER_LAND_MATTER'
]

# Outcome labels
OUTCOME_LABELS = {
    0: 'PLAINTIFF_FAILURE',
    1: 'PLAINTIFF_SUCCESS'
}

# File paths
PREPROCESSED_DATA_FILE = PROCESSED_DATA_DIR / 'preprocessed_land_cases.pkl'
TRAINED_MODELS_DIR = MODELS_DIR / 'trained_models'
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / 'system.log'

print(f"Configuration loaded. Project root: {PROJECT_ROOT}")
