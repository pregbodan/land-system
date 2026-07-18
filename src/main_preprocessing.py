"""
Main preprocessing pipeline
Orchestrates text extraction, cleaning, parsing, annotation, and feature engineering
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data_collection.text_extraction import TextExtractor
from preprocessing.text_preprocessing import TextPreprocessor, JudgmentParser
from preprocessing.annotation import CaseAnnotator
from preprocessing.structured_data import load_structured_cases
from feature_engineering.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Complete preprocessing pipeline for land matter judgments"""
    
    def __init__(self, config=None):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dict (optional)
        """
        self.config = config or {}
        
        # Initialize components
        self.text_extractor = TextExtractor()
        self.text_preprocessor = TextPreprocessor()
        self.judgment_parser = JudgmentParser()
        self.case_annotator = CaseAnnotator()
        self.feature_engineer = FeatureEngineer(
            max_features=self.config.get('tfidf_max_features', 1000),
            min_df=self.config.get('tfidf_min_df', 2),
            max_df=self.config.get('tfidf_max_df', 0.8),
            ngram_range=self.config.get('tfidf_ngram_range', (1, 2))
        )
        
        logger.info("Preprocessing pipeline initialized")
    
    def process_single_file(self, file_path):
        """
        Process a single judgment file
        
        Args:
            file_path: Path to judgment file
            
        Returns:
            dict: Processed case data
        """
        try:
            # Step 1: Extract text
            text = self.text_extractor.extract(file_path)
            is_valid, message = self.text_extractor.validate_extraction(text)
            
            if not is_valid:
                logger.warning(f"Extraction validation failed for {file_path}: {message}")
                return None
            
            # Step 2: Clean text
            cleaned_text = self.text_preprocessor.clean(text)
            
            # Step 3: Parse sections
            sections = self.judgment_parser.parse_sections(cleaned_text)
            
            # Step 4: Extract metadata
            metadata = self.judgment_parser.extract_metadata(cleaned_text)
            
            # Step 5: Annotate
            annotations = self.case_annotator.annotate_case(cleaned_text, sections)
            
            # Combine all data
            case_data = {
                'file_name': str(file_path),
                'full_text': cleaned_text,
                'facts': sections.get('facts', ''),
                'issues': sections.get('issues', ''),
                'arguments': sections.get('arguments', ''),
                'analysis': sections.get('analysis', ''),
                'decision': sections.get('decision', ''),
                **metadata,
                **annotations
            }
            
            return case_data
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return None
    
    def process_directory(self, data_dir):
        """
        Process all judgment files in a directory
        
        Args:
            data_dir: Directory containing judgment files
            
        Returns:
            DataFrame: Processed cases
        """
        data_dir = Path(data_dir)
        
        # Get all PDF and DOCX files
        pdf_files = list(data_dir.glob('**/*.pdf'))
        docx_files = list(data_dir.glob('**/*.docx'))
        all_files = pdf_files + docx_files
        
        logger.info(f"Found {len(all_files)} judgment files ({len(pdf_files)} PDF, {len(docx_files)} DOCX)")
        
        if len(all_files) == 0:
            logger.warning(f"No judgment files found in {data_dir}")
            return pd.DataFrame()
        
        # Process each file
        processed_cases = []
        
        for file_path in tqdm(all_files, desc="Processing judgments"):
            case_data = self.process_single_file(file_path)
            if case_data:
                processed_cases.append(case_data)
        
        # Create DataFrame
        df = pd.DataFrame(processed_cases)
        
        logger.info(f"Successfully processed {len(df)}/{len(all_files)} cases")
        
        # Log outcome distribution
        if 'outcome' in df.columns:
            logger.info(f"Outcome distribution:\n{df['outcome'].value_counts()}")
        
        # Log category distribution
        if 'primary_category' in df.columns:
            logger.info(f"Category distribution:\n{df['primary_category'].value_counts()}")
        
        return df
    
    def prepare_ml_dataset(self, df):
        """
        Prepare dataset for machine learning
        
        Args:
            df: DataFrame with processed cases
            
        Returns:
            dict: ML-ready dataset
        """
        logger.info("Preparing ML dataset...")
        
        if 'outcome' not in df.columns:
            logger.error("Missing 'outcome' column in dataset")
            return None

        # Normalize and clean outcomes
        outcome_series = (
            df['outcome']
            .where(df['outcome'].notna(), '')
            .astype(str)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
            .str.lower()
        )
        invalid_tokens = {'', 'nan', 'none', 'null', '<na>', 'n/a', 'na'}
        before_clean = len(df)
        df = df[~outcome_series.isin(invalid_tokens)].copy()
        df['outcome'] = outcome_series.loc[df.index]
        after_invalid = len(df)

        # Filter out unclear outcomes
        before_unclear = after_invalid
        df = df[df['outcome'].str.upper() != 'UNCLEAR'].copy()
        logger.info(f"Removed {before_clean - after_invalid} invalid/empty outcomes")
        logger.info(f"Removed {before_unclear - len(df)} cases with unclear outcomes")

        if len(df) == 0:
            logger.error("No cases remaining after filtering")
            return None

        # Handle classes with too few samples for stratified split
        min_class_count = int(self.config.get('min_class_count', 2))
        rare_class_strategy = self.config.get('rare_class_strategy', 'other')
        rare_class_label = self.config.get('rare_class_label', 'other')

        outcome_counts = df['outcome'].value_counts()
        rare_classes = outcome_counts[outcome_counts < min_class_count].index.tolist()
        if rare_classes:
            logger.warning(
                "Found %s rare outcome classes (<%s samples): %s",
                len(rare_classes),
                min_class_count,
                rare_classes[:10],
            )
            if rare_class_strategy == 'drop':
                df = df[~df['outcome'].isin(rare_classes)].copy()
                logger.info("Dropped %s rare-class rows", int(outcome_counts[outcome_counts < min_class_count].sum()))
            else:
                df.loc[df['outcome'].isin(rare_classes), 'outcome'] = rare_class_label
                logger.info(
                    "Mapped %s rare classes to '%s'",
                    len(rare_classes),
                    rare_class_label,
                )

        if df['outcome'].nunique() < 2:
            logger.error("Need at least 2 outcome classes after filtering")
            return None
        
        # Engineer features
        X, feature_names = self.feature_engineer.engineer_features(df, fit=True)
        
        # Encode outcome labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['outcome'])
        
        logger.info(f"Class distribution: {dict(zip(label_encoder.classes_, np.bincount(y)))}")
        
        # Split into train and test
        from sklearn.model_selection import train_test_split
        
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)

        # Stratify when possible; fall back to non-stratified split if sklearn rejects split constraints
        stratify_y = y if np.bincount(y).min() >= 2 else None
        if stratify_y is None:
            logger.warning("Disabling stratification because at least one class has <2 samples.")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_y
            )
        except ValueError as exc:
            logger.warning(f"Stratified split failed ({exc}); retrying without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=None
            )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Prepare dataset dict
        ml_dataset = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'label_encoder': label_encoder,
            'feature_engineer': self.feature_engineer,
            'df_metadata': df
        }
        
        return ml_dataset
    
    def run(self, data_dir, output_file, input_file=None):
        """
        Run complete preprocessing pipeline
        
        Args:
            data_dir: Directory containing raw judgment files
            output_file: Path to save preprocessed data
            
        Returns:
            dict: ML-ready dataset
        """
        logger.info("="*60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Load/Process data
        if input_file:
            logger.info(f"Loading structured data from {input_file}...")
            df = load_structured_cases(input_file)
            logger.info(f"Loaded {len(df)} structured cases")
        else:
            df = self.process_directory(data_dir)
        
        if len(df) == 0:
            logger.error("No cases were successfully processed")
            return None
        
        # Save intermediate results
        intermediate_file = Path(output_file).parent / 'processed_cases.csv'
        df.to_csv(intermediate_file, index=False)
        logger.info(f"Saved processed cases to {intermediate_file}")
        
        # Step 2: Prepare ML dataset
        ml_dataset = self.prepare_ml_dataset(df)
        
        if ml_dataset is None:
            return None
        
        # Step 3: Save ML-ready dataset
        with open(output_file, 'wb') as f:
            pickle.dump(ml_dataset, f)
        
        logger.info(f"Saved ML-ready dataset to {output_file}")
        
        # Print summary
        logger.info("="*60)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total cases processed: {len(df)}")
        logger.info(f"Feature matrix shape: {ml_dataset['X_train'].shape}")
        logger.info(f"Number of features: {len(ml_dataset['feature_names'])}")
        logger.info(f"Train samples: {len(ml_dataset['y_train'])}")
        logger.info(f"Test samples: {len(ml_dataset['y_test'])}")
        logger.info("="*60)
        
        return ml_dataset


def main():
    """Main function for running preprocessing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess land matter judgments')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw',
        help='Directory containing raw judgment files'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help='Path to structured CSV/JSON file (skips raw file extraction)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/processed/preprocessed_data.pkl',
        help='Output file for preprocessed data'
    )
    
    args = parser.parse_args()
    
    # Configuration
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
    
    # Run pipeline
    pipeline = PreprocessingPipeline(config)
    ml_dataset = pipeline.run(args.data_dir, args.output_file, input_file=args.input_file)
    
    if ml_dataset:
        print("\n✓ Preprocessing completed successfully!")
        print(f"✓ ML-ready dataset saved to: {args.output_file}")
    else:
        print("\n✗ Preprocessing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
