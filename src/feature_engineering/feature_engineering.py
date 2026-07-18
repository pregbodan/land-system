"""
Feature engineering module for creating ML features
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class FeatureEngineer:
    """Create ML features from processed judgments"""
    
    def __init__(self, max_features=1000, min_df=2, max_df=0.8, ngram_range=(1, 2)):
        """
        Initialize feature engineer
        
        Args:
            max_features: Maximum number of TF-IDF features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: N-gram range for TF-IDF
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        # Initialize vectorizer
        stop_words = set(stopwords.words('english'))
        # Add legal stop words that don't help discrimination
        legal_stopwords = {'court', 'case', 'appellant', 'respondent'}
        stop_words.update(legal_stopwords)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=list(stop_words),
            ngram_range=ngram_range,
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.feature_names = []
    
    def create_tfidf_features(self, texts, fit=True):
        """
        Create TF-IDF features from texts
        
        Args:
            texts: List of text strings
            fit: Whether to fit the vectorizer (True for train, False for test)
            
        Returns:
            scipy.sparse matrix: TF-IDF features
        """
        logger.info(f"Creating TF-IDF features from {len(texts)} texts...")
        
        if fit:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info(f"Fitted TF-IDF vectorizer with {len(self.tfidf_vectorizer.get_feature_names_out())} features")
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix
    
    def extract_structural_features(self, df):
        """
        Extract structural features from judgments
        
        Args:
            df: DataFrame with text columns
            
        Returns:
            DataFrame: Structural features
        """
        logger.info("Extracting structural features...")
        
        structural_features = pd.DataFrame()
        
        # Text length features
        structural_features['total_length'] = df['full_text'].str.len().fillna(0)
        structural_features['facts_length'] = df['facts'].str.len().fillna(0)
        structural_features['issues_length'] = df['issues'].str.len().fillna(0)
        structural_features['decision_length'] = df['decision'].str.len().fillna(0)
        
        # Count features
        structural_features['num_paragraphs'] = df['full_text'].str.count('\n\n').fillna(0)
        structural_features['num_sentences'] = df['full_text'].str.count(r'[.!?]').fillna(0)
        structural_features['num_words'] = df['full_text'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        # Average lengths (avoid division by zero)
        structural_features['avg_sentence_length'] = np.where(
            structural_features['num_sentences'] > 0,
            structural_features['num_words'] / structural_features['num_sentences'],
            0
        )
        
        # Citation features
        structural_features['num_case_citations'] = df['full_text'].str.count(r'\[20\d{2}\]').fillna(0)
        structural_features['num_statutory_refs'] = df['full_text'].str.count(r'Section\s+\d+').fillna(0)
        
        # Ratio features
        structural_features['facts_ratio'] = np.where(
            structural_features['total_length'] > 0,
            structural_features['facts_length'] / structural_features['total_length'],
            0
        )
        
        structural_features['decision_ratio'] = np.where(
            structural_features['total_length'] > 0,
            structural_features['decision_length'] / structural_features['total_length'],
            0
        )
        
        logger.info(f"Created {len(structural_features.columns)} structural features")
        
        return structural_features
    
    def extract_metadata_features(self, df):
        """
        Extract metadata features
        
        Args:
            df: DataFrame with metadata columns
            
        Returns:
            DataFrame: Metadata features
        """
        logger.info("Extracting metadata features...")
        
        metadata_features = pd.DataFrame()
        
        # Court level (categorical -> numerical)
        court_mapping = {
            'SUPREME COURT': 3,
            'COURT OF APPEAL': 2,
            'HIGH COURT': 1
        }
        
        if 'court' in df.columns:
            metadata_features['court_level'] = df['court'].map(court_mapping).fillna(1)
        else:
            metadata_features['court_level'] = 1
        
        # Year
        if 'year' in df.columns:
            metadata_features['year'] = df['year'].fillna(df['year'].median() if len(df) > 0 else 2020)
        else:
            metadata_features['year'] = 2020
        
        # Normalize year (subtract minimum year to make it relative)
        if len(metadata_features) > 0:
            min_year = metadata_features['year'].min()
            metadata_features['years_since_start'] = metadata_features['year'] - min_year
        
        logger.info(f"Created {len(metadata_features.columns)} metadata features")
        
        return metadata_features
    
    def combine_features(self, df, tfidf_matrix):
        """
        Combine all features into single matrix
        
        Args:
            df: DataFrame with all data
            tfidf_matrix: TF-IDF feature matrix
            
        Returns:
            tuple: (combined_features, feature_names)
        """
        logger.info("Combining all features...")
        
        # Get structural features
        structural_features = self.extract_structural_features(df)
        
        # Get metadata features
        metadata_features = self.extract_metadata_features(df)
        
        # Get legal features (already in df)
        legal_feature_cols = [
            'mentions_land_use_act', 'has_certificate_of_occupancy',
            'has_governors_consent', 'mentions_section_22', 'mentions_section_28',
            'mentions_evidence_act', 'has_documentary_evidence',
            'has_survey_evidence', 'mentions_section_84',
            'customary_tenure', 'statutory_tenure',
            'has_witness_testimony', 'has_expert_evidence',
            'involves_inheritance', 'has_boundary_issue',
            'mentions_trespass', 'involves_fraud', 'has_oral_evidence'
        ]
        
        # Filter to only existing columns
        existing_legal_cols = [col for col in legal_feature_cols if col in df.columns]
        if existing_legal_cols:
            legal_features = df[existing_legal_cols].fillna(0).astype(int)
        else:
            legal_features = pd.DataFrame()
        
        # Combine non-TF-IDF features
        other_features = pd.concat([
            structural_features,
            metadata_features,
            legal_features
        ], axis=1)
        
        # Fill any remaining NaN values
        other_features = other_features.fillna(0)
        
        # Convert to sparse matrix
        other_features_sparse = csr_matrix(other_features.values)
        
        # Combine with TF-IDF
        combined_features = hstack([tfidf_matrix, other_features_sparse])
        
        # Store feature names
        tfidf_names = list(self.tfidf_vectorizer.get_feature_names_out())
        self.feature_names = tfidf_names + list(other_features.columns)
        
        logger.info(f"Combined feature matrix shape: {combined_features.shape}")
        logger.info(f"Total features: {len(self.feature_names)}")
        
        return combined_features, self.feature_names
    
    def engineer_features(self, df, fit=True):
        """
        Complete feature engineering pipeline
        
        Args:
            df: DataFrame with processed judgment data
            fit: Whether to fit transformers (True for train, False for test)
            
        Returns:
            tuple: (feature_matrix, feature_names)
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Create TF-IDF features
        texts = df['full_text'].fillna('').astype(str).tolist()
        tfidf_matrix = self.create_tfidf_features(texts, fit=fit)
        
        # Combine all features
        combined_features, feature_names = self.combine_features(df, tfidf_matrix)
        
        logger.info("Feature engineering complete!")
        
        return combined_features, feature_names


if __name__ == "__main__":
    # Example usage
    print("Feature engineering module loaded successfully.")
    print("\nExample usage:")
    print("""
    from feature_engineering import FeatureEngineer
    
    # Initialize
    engineer = FeatureEngineer(max_features=1000)
    
    # Engineer features
    X, feature_names = engineer.engineer_features(df, fit=True)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    """)
