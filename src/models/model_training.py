"""
Model training module
Implements Random Forest, XGBoost, SVM, Naive Bayes, and optional Neural Network
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)
import xgboost as xgb
import joblib
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate ML models for land matter prediction"""
    
    def __init__(self, random_state=42, n_jobs=1):
        """
        Initialize model trainer
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Parallel workers for sklearn operations
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.results = {}
        
    def train_random_forest(self, X_train, y_train, **kwargs):
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            Trained model
        """
        logger.info("Training Random Forest...")
        
        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'class_weight': 'balanced'  # Handle imbalanced classes
        }
        params.update(kwargs)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        logger.info("✓ Random Forest training complete")
        
        return model
    
    def train_xgboost(self, X_train, y_train, **kwargs):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for XGBClassifier
            
        Returns:
            Trained model
        """
        logger.info("Training XGBoost...")
        
        n_classes = len(np.unique(y_train))

        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }

        if n_classes > 2:
            params.update({
                'objective': 'multi:softprob',
                'num_class': n_classes,
                'eval_metric': 'mlogloss',
            })
        else:
            params.update({
                'objective': 'binary:logistic',
                'scale_pos_weight': 1,
                'eval_metric': 'logloss',
            })

        params.update(kwargs)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        logger.info("✓ XGBoost training complete")
        
        return model
    
    def train_svm(self, X_train, y_train, **kwargs):
        """
        Train SVM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for SVC
            
        Returns:
            Trained model
        """
        logger.info("Training SVM...")
        
        # Default parameters
        params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True,  # Enable probability estimates
            'random_state': self.random_state,
            'class_weight': 'balanced'
        }
        params.update(kwargs)
        
        model = SVC(**params)
        model.fit(X_train, y_train)
        
        self.models['svm'] = model
        logger.info("✓ SVM training complete")
        
        return model
    
    def train_naive_bayes(self, X_train, y_train, **kwargs):
        """
        Train Naive Bayes model
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for MultinomialNB
            
        Returns:
            Trained model
        """
        logger.info("Training Naive Bayes...")
        
        # Default parameters
        params = {
            'alpha': 1.0  # Laplace smoothing
        }
        params.update(kwargs)
        
        model = MultinomialNB(**params)
        model.fit(X_train, y_train)
        
        self.models['naive_bayes'] = model
        logger.info("✓ Naive Bayes training complete")
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train, **kwargs):
        """
        Train Gradient Boosting model (sklearn implementation)
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters
            
        Returns:
            Trained model
        """
        logger.info("Training Gradient Boosting (sklearn)...")
        
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': self.random_state
        }
        params.update(kwargs)
        
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = model
        logger.info("✓ Gradient Boosting training complete")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        n_classes = len(np.unique(y_test))
        average = 'binary' if n_classes == 2 else 'weighted'
        
        # Probabilities (if available)
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average=average, zero_division=0),
        }
        
        # AUC-ROC (only if probabilities available)
        if y_pred_proba is not None:
            try:
                if n_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(
                        y_test,
                        y_pred_proba,
                        multi_class='ovr',
                        average='weighted'
                    )
            except Exception:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        # Log results
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc'] is not None:
            logger.info(f"  AUC-ROC:   {metrics['roc_auc']:.4f}")
        
        self.results[model_name] = metrics
        
        return metrics
    
    def cross_validate(self, model, X, y, cv=5, model_name="Model"):
        """
        Perform cross-validation
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Labels
            cv: Number of folds
            model_name: Name of the model
            
        Returns:
            dict: Cross-validation scores
        """
        logger.info(f"Cross-validating {model_name} with {cv} folds...")
        n_classes = len(np.unique(y))
        class_counts = np.bincount(y)
        min_class_count = int(class_counts.min()) if class_counts.size > 0 else 0
        effective_cv = min(cv, min_class_count) if min_class_count > 0 else cv

        if effective_cv < 2:
            logger.warning(
                "Skipping cross-validation for %s because minimum class count is %s.",
                model_name,
                min_class_count,
            )
            return {
                'cv_folds_used': 0,
                'accuracy_mean': None,
                'accuracy_std': None,
                'precision_mean': None,
                'precision_std': None,
                'recall_mean': None,
                'recall_std': None,
                'f1_mean': None,
                'f1_std': None,
                'message': 'Cross-validation skipped due to insufficient class frequency.',
            }

        if effective_cv != cv:
            logger.warning(
                "Reducing CV folds from %s to %s due to rare classes (min class count=%s).",
                cv,
                effective_cv,
                min_class_count,
            )

        cv_splitter = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=self.random_state)
        average = 'binary' if n_classes == 2 else 'weighted'
        precision_scoring = make_scorer(precision_score, average=average, zero_division=0)
        recall_scoring = make_scorer(recall_score, average=average, zero_division=0)
        f1_scoring = make_scorer(f1_score, average=average, zero_division=0)

        # Accuracy
        accuracy_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring='accuracy', n_jobs=self.n_jobs)
        
        # Precision
        precision_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=precision_scoring, n_jobs=self.n_jobs)
        
        # Recall
        recall_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=recall_scoring, n_jobs=self.n_jobs)
        
        # F1
        f1_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=f1_scoring, n_jobs=self.n_jobs)
        
        cv_results = {
            'cv_folds_used': effective_cv,
            'accuracy_mean': accuracy_scores.mean(),
            'accuracy_std': accuracy_scores.std(),
            'precision_mean': precision_scores.mean(),
            'precision_std': precision_scores.std(),
            'recall_mean': recall_scores.mean(),
            'recall_std': recall_scores.std(),
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
        }
        
        logger.info(f"{model_name} CV Results:")
        logger.info(f"  Accuracy:  {cv_results['accuracy_mean']:.4f} (+/- {cv_results['accuracy_std']:.4f})")
        logger.info(f"  Precision: {cv_results['precision_mean']:.4f} (+/- {cv_results['precision_std']:.4f})")
        logger.info(f"  Recall:    {cv_results['recall_mean']:.4f} (+/- {cv_results['recall_std']:.4f})")
        logger.info(f"  F1-Score:  {cv_results['f1_mean']:.4f} (+/- {cv_results['f1_std']:.4f})")
        
        return cv_results
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate all models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Results for all models
        """
        logger.info("="*60)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*60)
        
        all_results = {}
        
        # 1. Random Forest
        try:
            rf_model = self.train_random_forest(X_train, y_train)
            rf_results = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
            all_results['Random Forest'] = rf_results
        except Exception as e:
            logger.error(f"Random Forest failed: {e}")
        
        # 2. XGBoost
        try:
            xgb_model = self.train_xgboost(X_train, y_train)
            xgb_results = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
            all_results['XGBoost'] = xgb_results
        except Exception as e:
            logger.error(f"XGBoost failed: {e}")
        
        # 3. SVM
        try:
            svm_model = self.train_svm(X_train, y_train)
            svm_results = self.evaluate_model(svm_model, X_test, y_test, "SVM")
            all_results['SVM'] = svm_results
        except Exception as e:
            logger.error(f"SVM failed: {e}")
        
        # 4. Naive Bayes
        try:
            nb_model = self.train_naive_bayes(X_train, y_train)
            nb_results = self.evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
            all_results['Naive Bayes'] = nb_results
        except Exception as e:
            logger.error(f"Naive Bayes failed: {e}")
        
        # 5. Gradient Boosting (optional - sklearn version)
        try:
            gb_model = self.train_gradient_boosting(X_train, y_train)
            gb_results = self.evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
            all_results['Gradient Boosting'] = gb_results
        except Exception as e:
            logger.error(f"Gradient Boosting failed: {e}")
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        
        return all_results
    
    def get_best_model(self):
        """
        Get the best performing model based on accuracy
        
        Returns:
            tuple: (model_name, model, accuracy)
        """
        if not self.results:
            logger.warning("No results available. Train models first.")
            return None, None, None
        
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        best_accuracy = self.results[best_model_name]['accuracy']
        best_model = self.models.get(best_model_name.lower().replace(' ', '_'))
        
        logger.info(f"Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return best_model_name, best_model, best_accuracy
    
    def save_model(self, model_name, output_path):
        """
        Save a trained model
        
        Args:
            model_name: Name of the model to save
            output_path: Path to save the model
        """
        model_key = model_name.lower().replace(' ', '_')
        
        if model_key not in self.models:
            logger.error(f"Model '{model_name}' not found")
            return
        
        model = self.models[model_key]
        joblib.dump(model, output_path)
        logger.info(f"Saved {model_name} to {output_path}")
    
    def save_results(self, output_path):
        """
        Save evaluation results to JSON
        
        Args:
            output_path: Path to save results
        """
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    print("Model training module loaded successfully.")
    print("\nExample usage:")
    print("""
    from model_training import ModelTrainer
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Get best model
    best_name, best_model, best_acc = trainer.get_best_model()
    print(f"Best model: {best_name} ({best_acc:.2%} accuracy)")
    
    # Save best model
    trainer.save_model(best_name, f'models/{best_name}.pkl')
    """)
