"""
Model Training Module for Gaming Player Churn Prediction

This module provides comprehensive model training, evaluation, and comparison
for player churn prediction using multiple machine learning algorithms.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, learning_curve, validation_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeaturePreprocessor:
    """Handle feature preprocessing and selection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.preprocessing_pipeline = {}
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, 
                     feature_selection_method: str = 'kbest', 
                     k_features: int = 20) -> Tuple[np.ndarray, List[str]]:
        """Fit preprocessing pipeline and transform features"""
        logger.info(f"Preprocessing {X.shape[1]} features for {len(X)} samples")
        
        # Handle missing values
        X_cleaned = self._handle_missing_values(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_cleaned)
        
        # Feature selection
        if feature_selection_method == 'kbest':
            self.feature_selector = SelectKBest(score_func=f_classif, k=k_features)
        elif feature_selection_method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            self.feature_selector = RFE(estimator, n_features_to_select=k_features)
        else:
            raise ValueError(f"Unknown feature selection method: {feature_selection_method}")
        
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        # Get selected feature names
        if hasattr(self.feature_selector, 'get_support'):
            feature_mask = self.feature_selector.get_support()
            self.selected_features = X.columns[feature_mask].tolist()
        else:
            self.selected_features = X.columns.tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features")
        
        # Store preprocessing info
        self.preprocessing_pipeline = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'feature_selection_method': feature_selection_method
        }
        
        return X_selected, self.selected_features
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted pipeline"""
        if self.scaler is None or self.feature_selector is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        X_cleaned = self._handle_missing_values(X)
        X_scaled = self.scaler.transform(X_cleaned)
        X_selected = self.feature_selector.transform(X_scaled)
        
        return X_selected
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        X_cleaned = X.copy()
        
        # Fill numeric columns with median
        numeric_columns = X_cleaned.select_dtypes(include=[np.number]).columns
        X_cleaned[numeric_columns] = X_cleaned[numeric_columns].fillna(
            X_cleaned[numeric_columns].median()
        )
        
        # Fill categorical columns with mode
        categorical_columns = X_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if len(X_cleaned[col].mode()) > 0:
                X_cleaned[col] = X_cleaned[col].fillna(X_cleaned[col].mode().iloc[0])
            else:
                X_cleaned[col] = X_cleaned[col].fillna('Unknown')
        
        return X_cleaned

class ModelTrainer:
    """Train and evaluate multiple churn prediction models"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.model_configs = {}
        self.results = {}
        self.best_model = None
        self.preprocessor = FeaturePreprocessor()
        
    def prepare_models(self) -> Dict[str, Any]:
        """Initialize model configurations"""
        
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'param_grid': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'param_grid': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            
            'neural_network': {
                'model': MLPClassifier(random_state=self.random_state, max_iter=500),
                'param_grid': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        logger.info(f"Prepared {len(self.model_configs)} model configurations")
        return self.model_configs
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2, cv_folds: int = 5) -> Dict[str, Dict]:
        """Train all models with hyperparameter tuning"""
        logger.info("Starting comprehensive model training pipeline")
        
        # Prepare models
        self.prepare_models()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        # Preprocess features
        X_train_processed, feature_names = self.preprocessor.fit_transform(X_train, y_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        logger.info(f"Training on {X_train_processed.shape[0]} samples, testing on {X_test_processed.shape[0]}")
        
        # Train each model
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Hyperparameter tuning with cross-validation
                grid_search = GridSearchCV(
                    config['model'], 
                    config['param_grid'],
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train_processed, y_train)
                
                # Store best model
                self.models[model_name] = grid_search.best_estimator_
                
                # Evaluate model
                model_results = self._evaluate_model(
                    grid_search.best_estimator_, 
                    X_test_processed, 
                    y_test,
                    model_name
                )
                
                # Add training metrics
                model_results.update({
                    'best_params': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'cv_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
                })
                
                self.results[model_name] = model_results
                
                logger.info(f"✅ {model_name} - ROC-AUC: {model_results['roc_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                continue
        
        # Identify best model
        self._identify_best_model()
        
        # Store test data for later use
        self.X_test = X_test_processed
        self.y_test = y_test
        self.feature_names = feature_names
        
        logger.info("Model training completed successfully")
        return self.results
    
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                       model_name: str) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
        
        # Additional metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics.update({
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        })
        
        return metrics
    
    def _identify_best_model(self):
        """Identify the best performing model based on ROC-AUC"""
        if not self.results:
            return
        
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['roc_auc'])
        self.best_model = {
            'name': best_model_name,
            'model': self.models[best_model_name],
            'metrics': self.results[best_model_name]
        }
        
        logger.info(f"🏆 Best model: {best_model_name} (ROC-AUC: {self.best_model['metrics']['roc_auc']:.4f})")
    
    def generate_model_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive model comparison report"""
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'ROC-AUC': results['roc_auc'],
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'Specificity': results['specificity'],
                'CV Score': results.get('best_cv_score', 0),
                'CV Std': results.get('cv_std', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        
        return comparison_df
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Extract feature importance from tree-based models"""
        importance_data = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[name] = model.feature_importances_
        
        return importance_data
    
    def save_models(self, output_dir: str = "models/"):
        """Save all trained models and metadata"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_file = output_path / f"{model_name}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {model_name} to {model_file}")
        
        # Save preprocessor
        preprocessor_file = output_path / "preprocessor.pkl"
        with open(preprocessor_file, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # Save model metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_results': self.results,
            'best_model': {
                'name': self.best_model['name'] if self.best_model else None,
                'metrics': self.best_model['metrics'] if self.best_model else None
            },
            'feature_names': self.feature_names,
            'model_version': '1.0',
            'trainer_config': {
                'random_state': self.random_state,
                'models_trained': list(self.models.keys())
            }
        }
        
        metadata_file = output_path / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model artifacts saved to {output_path}")
        
        return output_path

class HyperparameterTuner:
    """Advanced hyperparameter tuning with multiple strategies"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def tune_model(self, model, param_distributions: Dict, 
                   X_train: np.ndarray, y_train: np.ndarray,
                   method: str = 'grid', n_iter: int = 100) -> Dict:
        """Tune hyperparameters using specified method"""
        
        if method == 'grid':
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(
                model, param_distributions, cv=5, scoring='roc_auc',
                n_jobs=-1, random_state=self.random_state
            )
        elif method == 'random':
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                model, param_distributions, n_iter=n_iter, cv=5,
                scoring='roc_auc', n_jobs=-1, random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown tuning method: {method}")
        
        search.fit(X_train, y_train)
        
        return {
            'best_estimator': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }

class ModelEvaluator:
    """Advanced model evaluation and interpretation"""
    
    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
        self.models = trainer.models
        self.results = trainer.results
        
    def generate_business_metrics(self) -> Dict[str, Any]:
        """Generate business-relevant evaluation metrics"""
        
        if not self.trainer.best_model:
            return {}
        
        best_model = self.trainer.best_model['model']
        y_pred = best_model.predict(self.trainer.X_test)
        y_pred_proba = best_model.predict_proba(self.trainer.X_test)[:, 1]
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(self.trainer.y_test, y_pred).ravel()
        
        # Business metrics
        total_players = len(self.trainer.y_test)
        actual_churners = sum(self.trainer.y_test)
        predicted_churners = sum(y_pred)
        
        # Cost-benefit analysis (example values)
        cost_per_intervention = 10  # Cost to target a player
        revenue_per_retained_player = 50  # Revenue from retaining a player
        
        business_metrics = {
            'total_players_evaluated': total_players,
            'actual_churn_rate': actual_churners / total_players,
            'predicted_churn_rate': predicted_churners / total_players,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'players_correctly_identified_as_churn': int(tp),
            'players_incorrectly_flagged_as_churn': int(fp),
            'potential_revenue_saved': int(tp) * revenue_per_retained_player,
            'intervention_cost': int(predicted_churners) * cost_per_intervention,
            'net_benefit': (int(tp) * revenue_per_retained_player) - (int(predicted_churners) * cost_per_intervention),
            'precision_business': f"Out of {predicted_churners} flagged players, {tp} will actually churn",
            'recall_business': f"Model catches {tp} out of {actual_churners} actual churners"
        }
        
        return business_metrics
    
    def threshold_optimization(self, metric: str = 'f1') -> Dict[str, float]:
        """Optimize decision threshold for business objectives"""
        
        if not self.trainer.best_model:
            return {}
        
        best_model = self.trainer.best_model['model']
        y_pred_proba = best_model.predict_proba(self.trainer.X_test)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(self.trainer.y_test, y_pred_thresh)
            elif metric == 'precision':
                score = precision_score(self.trainer.y_test, y_pred_thresh)
            elif metric == 'recall':
                score = recall_score(self.trainer.y_test, y_pred_thresh)
            else:
                score = accuracy_score(self.trainer.y_test, y_pred_thresh)
                
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        return {
            'optimal_threshold': optimal_threshold,
            'optimal_score': optimal_score,
            'metric_optimized': metric,
            'threshold_scores': dict(zip(thresholds, scores))
        }

def main():
    """Main function for CLI usage"""
    from ..data.data_collector import SyntheticDataGenerator
    from ..features.feature_engineering import FeatureEngineer
    
    logger.info("Starting model training pipeline...")
    
    # Generate sample data
    data_generator = SyntheticDataGenerator()
    player_data = data_generator.generate_player_data(5000)
    
    # Feature engineering
    feature_engineer = FeatureEngineer(player_data)
    engineered_data = feature_engineer.create_all_features()
    
    # Prepare features and target
    X = engineered_data.drop(['churned', 'player_id'], axis=1)
    y = engineered_data['churned']
    
    logger.info(f"Dataset shape: {X.shape}, Churn rate: {y.mean():.3f}")
    
    # Train models
    trainer = ModelTrainer(random_state=42)
    results = trainer.train_all_models(X, y)
    
    # Generate comparison report
    comparison_df = trainer.generate_model_comparison_report()
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save models
    model_path = trainer.save_models()
    
    # Business evaluation
    evaluator = ModelEvaluator(trainer)
    business_metrics = evaluator.generate_business_metrics()
    threshold_optimization = evaluator.threshold_optimization()
    
    print(f"\nBusiness Metrics:")
    for key, value in business_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
    
    print(f"\nTraining completed successfully!")
    print(f"Best model: {trainer.best_model['name']}")
    print(f"ROC-AUC: {trainer.best_model['metrics']['roc_auc']:.4f}")
    print(f"Models saved to: {model_path}")

if __name__ == "__main__":
    main()