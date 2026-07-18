"""
Main training script
Loads preprocessed data, trains all models, and saves results
"""
import pickle
import argparse
import logging
import json
from pathlib import Path
import sys
import re
from typing import Any, Dict, List
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.model_training import ModelTrainer
from main_preprocessing import PreprocessingPipeline
from preprocessing.structured_data import load_and_merge_structured_cases, canonicalize_outcome_label

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


LAND_SIGNAL_REGEX = re.compile(
    r"\b(?:land|title to land|declaration of title|certificate of occupancy|right of occupancy|"
    r"survey plan|boundary|trespass|customary tenure|governor's consent|land use act)\b"
)


def _first_non_empty(record: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = record.get(key, "")
        text = str(value or "").strip()
        if text:
            return text
    return ""


def build_case_reference_records(df):
    """
    Build compact retrieval records from training dataframe for citation support.
    """
    if df is None or len(df) == 0:
        return []

    records = []
    title_candidates = [
        "case_title",
        "case_name",
        "title",
        "suit_title",
        "matter_title",
    ]
    court_candidates = ["court", "court_name", "tribunal"]

    for idx, row in df.reset_index(drop=True).iterrows():
        row_dict = row.to_dict()
        full_text = str(row_dict.get("full_text", "") or "")
        facts = str(row_dict.get("facts", "") or "")
        decision = str(row_dict.get("decision", "") or "")
        retrieval_text = " ".join(
            part for part in [facts, str(row_dict.get("issues", "") or ""), decision, full_text] if part
        ).strip()
        retrieval_text = retrieval_text[:4500]
        if not retrieval_text:
            continue

        case_title = _first_non_empty(row_dict, title_candidates)
        if not case_title:
            case_title = f"Trained Case {idx + 1}"

        records.append(
            {
                "case_id": f"case_{idx + 1}",
                "case_title": case_title,
                "outcome": str(row_dict.get("outcome", "") or ""),
                "court": _first_non_empty(row_dict, court_candidates),
                "year": str(row_dict.get("year", "") or ""),
                "source_file": str(row_dict.get("source_file", "") or ""),
                "facts": facts[:1200],
                "decision": decision[:1200],
                "retrieval_text": retrieval_text,
            }
        )

    return records


def log_domain_health(df):
    """
    Log quick dataset-domain health diagnostics for land-case training.
    """
    if 'full_text' not in df.columns:
        logger.warning("Domain health check skipped: 'full_text' column not found.")
        return

    text_series = df['full_text'].astype(str).str.lower()
    if len(text_series) == 0:
        logger.warning("Domain health check skipped: empty dataset.")
        return

    land_hit_ratio = text_series.str.contains(LAND_SIGNAL_REGEX).mean()
    logger.info(f"Land-signal ratio in training text: {land_hit_ratio:.2%}")
    if land_hit_ratio < 0.20:
        logger.warning(
            "Low land-signal ratio detected. Dataset may contain non-land matters; "
            "model outcomes can be unreliable for pure land-case predictions."
        )


def load_feedback_cases(feedback_file: Path):
    """
    Load user feedback JSONL and convert to training rows.
    Only rows with an actual_outcome are used for learning.
    """
    if not feedback_file.exists():
        return pd.DataFrame()

    rows = []
    with open(feedback_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            actual_outcome = canonicalize_outcome_label(payload.get("actual_outcome", ""))
            case_text = str(payload.get("case_text", "") or "").strip()
            if not actual_outcome or not case_text:
                continue

            rows.append(
                {
                    "full_text": case_text,
                    "facts": "",
                    "issues": "",
                    "arguments": "",
                    "analysis": "",
                    "decision": "",
                    "outcome": actual_outcome,
                    "source_file": str(feedback_file),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ML models for land matter prediction')
    parser.add_argument(
        '--data_file',
        type=str,
        nargs='+',
        default=['../data/processed/preprocessed_data.pkl'],
        help='Path(s) to input data file(s): preprocessed .pkl or structured .csv/.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data/models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Parallel workers for sklearn operations (use 1 for maximum compatibility)'
    )
    parser.add_argument(
        '--include_feedback',
        action='store_true',
        help='Include labeled prediction feedback JSONL in training data'
    )
    parser.add_argument(
        '--feedback_file',
        type=str,
        default='data/models/prediction_feedback.jsonl',
        help='Path to feedback JSONL file used when --include_feedback is set'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("LAND MATTER PREDICTION SYSTEM - MODEL TRAINING")
    logger.info("="*70)
    
    # Load preprocessed or structured data
    input_files = [Path(p) for p in args.data_file]
    logger.info(f"Loading data from: {[str(p) for p in input_files]}")

    missing = [str(p) for p in input_files if not p.exists()]
    if missing:
        logger.error(f"Data file(s) not found: {missing}")
        logger.error("Please run preprocessing first: python src/main_preprocessing.py")
        sys.exit(1)

    suffixes = {p.suffix.lower() for p in input_files}
    structured_suffixes = {".csv", ".json"}

    if suffixes.issubset(structured_suffixes):
        logger.info("Detected structured data input. Running feature engineering...")
        df = load_and_merge_structured_cases(input_files)
        if args.include_feedback:
            feedback_df = load_feedback_cases(Path(args.feedback_file))
            if not feedback_df.empty:
                df = pd.concat([df, feedback_df], ignore_index=True, sort=False)
                logger.info(
                    "Included %s feedback case(s) from %s",
                    len(feedback_df),
                    args.feedback_file,
                )
            else:
                logger.info("No usable feedback cases found in %s", args.feedback_file)
        log_domain_health(df)
        config = {
            'test_size': 0.2,
            'random_state': 42,
            'tfidf_max_features': 1000,
            'tfidf_min_df': 2,
            'tfidf_max_df': 0.8,
            'tfidf_ngram_range': (1, 2),
            'min_class_count': 2,
            'rare_class_strategy': 'other',
            'rare_class_label': 'other',
        }
        pipeline = PreprocessingPipeline(config)
        ml_dataset = pipeline.prepare_ml_dataset(df)
        if ml_dataset is None:
            logger.error("Failed to prepare ML dataset from structured data.")
            sys.exit(1)
    elif suffixes == {".pkl"} and len(input_files) == 1:
        data_path = input_files[0]
        try:
            with open(data_path, 'rb') as f:
                ml_dataset = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
            logger.error("Please run preprocessing first: python src/main_preprocessing.py")
            sys.exit(1)
    else:
        logger.error(
            "Unsupported input combination. Use either:\n"
            "1) A single .pkl file, or\n"
            "2) One or more structured .csv/.json files."
        )
        sys.exit(1)
    
    X_train = ml_dataset['X_train']
    X_test = ml_dataset['X_test']
    y_train = ml_dataset['y_train']
    y_test = ml_dataset['y_test']
    label_encoder = ml_dataset['label_encoder']
    feature_engineer = ml_dataset['feature_engineer']
    feature_names = ml_dataset.get('feature_names', [])
    
    logger.info(f"Data loaded successfully!")
    logger.info(f"  Train samples: {X_train.shape[0]}")
    logger.info(f"  Test samples: {X_test.shape[0]}")
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  Classes: {label_encoder.classes_}")
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42, n_jobs=args.n_jobs)
    
    # Train all models
    logger.info("\n" + "="*70)
    logger.info("TRAINING ALL MODELS")
    logger.info("="*70 + "\n")
    
    all_results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Save all models
    logger.info("\n" + "="*70)
    logger.info("SAVING MODELS")
    logger.info("="*70)
    
    for model_name in trainer.models.keys():
        model_filename = f"{model_name}.pkl"
        model_path = output_dir / model_filename
        trainer.save_model(model_name.replace('_', ' ').title(), model_path)
    
    # Save results
    results_path = output_dir / 'training_results.json'
    trainer.save_results(results_path)

    # Save inference artifacts used by predict/web app
    case_reference_records = build_case_reference_records(ml_dataset.get("df_metadata"))
    inference_artifacts = {
        'feature_engineer': feature_engineer,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'case_reference_records': case_reference_records,
    }
    inference_artifacts_path = output_dir / 'inference_artifacts.pkl'
    with open(inference_artifacts_path, 'wb') as f:
        pickle.dump(inference_artifacts, f)
    logger.info(f"Inference artifacts saved to {inference_artifacts_path}")
    if case_reference_records:
        case_reference_path = output_dir / "case_reference_records.json"
        with open(case_reference_path, "w", encoding="utf-8") as f:
            json.dump(case_reference_records, f, indent=2)
        logger.info(f"Case reference records saved to {case_reference_path}")
    
    # Get and save best model
    best_name, best_model, best_acc = trainer.get_best_model()
    if best_model:
        best_model_path = output_dir / 'best_model.pkl'
        import joblib
        joblib.dump(best_model, best_model_path)
        logger.info(f"Best model ({best_name}) saved to {best_model_path}")
        
        # Save model metadata
        metadata = {
            'best_model_name': best_name,
            'best_accuracy': float(best_acc),
            'label_encoder_classes': label_encoder.classes_.tolist(),
            'feature_count': X_train.shape[1],
            'inference_artifacts_path': str(inference_artifacts_path),
            'case_reference_records_count': len(case_reference_records),
        }
        
        metadata_path = output_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Model metadata saved to {metadata_path}")
    
    # Print final summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE - RESULTS SUMMARY")
    logger.info("="*70)
    
    # Sort results by accuracy
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    logger.info(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    logger.info("-" * 70)
    
    for model_name, metrics in sorted_results:
        logger.info(
            f"{model_name:<20} "
            f"{metrics['accuracy']:<12.4f} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1_score']:<12.4f}"
        )
    
    logger.info("\n" + "="*70)
    logger.info(f"✓ Best Model: {best_name}")
    logger.info(f"✓ Best Accuracy: {best_acc:.2%}")
    logger.info(f"✓ Models saved to: {output_dir}")
    logger.info(f"✓ Results saved to: {results_path}")
    logger.info("="*70)
    
    # Optional: Cross-validation on best model
    if best_model:
        logger.info("\n" + "="*70)
        logger.info("CROSS-VALIDATION ON BEST MODEL")
        logger.info("="*70)
        
        # Combine train and test for CV
        from scipy.sparse import vstack
        import numpy as np
        
        X_full = vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])
        
        try:
            cv_results = trainer.cross_validate(
                best_model,
                X_full,
                y_full,
                cv=args.cv_folds,
                model_name=best_name
            )
            
            # Save CV results
            cv_results_path = output_dir / 'cv_results.json'
            with open(cv_results_path, 'w') as f:
                json.dump(cv_results, f, indent=2)
            
            logger.info(f"Cross-validation results saved to {cv_results_path}")
        except Exception as exc:
            logger.error(f"Cross-validation failed: {exc}")
            logger.info("Continuing without CV results.")
    
    logger.info("\n✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!\n")


if __name__ == "__main__":
    main()
