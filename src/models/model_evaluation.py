"""
Model Evaluation Module for Gaming Player Churn Prediction

This module provides comprehensive model evaluation capabilities including
validation, performance analysis, business metrics, and model comparison.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, brier_score_loss, log_loss
)
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, 
    learning_curve, validation_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Comprehensive model validation with multiple validation strategies
    
    Provides rigorous validation including cross-validation, temporal validation,
    and stability testing for production readiness assessment.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize model validator"""
        self.random_state = random_state
        self.validation_results = {}
        
    def validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                      validation_strategy: str = 'comprehensive') -> Dict[str, Any]:
        """
        Comprehensive model validation
        
        Args:
            model: Trained model to validate
            X: Feature matrix
            y: Target variable
            validation_strategy: Validation approach ('basic', 'comprehensive', 'production')
            
        Returns:
            Dictionary containing validation results
        """
        
        logger.info(f"Starting {validation_strategy} model validation")
        
        results = {
            'validation_strategy': validation_strategy,
            'validation_date': datetime.now().isoformat(),
            'dataset_size': len(X),
            'feature_count': X.shape[1],
            'class_distribution': y.value_counts().to_dict()
        }
        
        if validation_strategy == 'basic':
            results.update(self._basic_validation(model, X, y))
        elif validation_strategy == 'comprehensive':
            results.update(self._comprehensive_validation(model, X, y))
        elif validation_strategy == 'production':
            results.update(self._production_validation(model, X, y))
        else:
            raise ValueError(f"Unknown validation strategy: {validation_strategy}")
        
        self.validation_results = results
        logger.info("Model validation completed")
        
        return results
    
    def _basic_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Basic cross-validation"""
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Cross-validation scores
        cv_scores = cross_validate(
            model, X, y, cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            return_train_score=True
        )
        
        results = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            test_scores = cv_scores[f'test_{metric}']
            train_scores = cv_scores[f'train_{metric}']
            
            results[f'{metric}_cv_mean'] = np.mean(test_scores)
            results[f'{metric}_cv_std'] = np.std(test_scores)
            results[f'{metric}_train_mean'] = np.mean(train_scores)
            results[f'{metric}_overfitting'] = np.mean(train_scores) - np.mean(test_scores)
        
        return results
    
    def _comprehensive_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Comprehensive validation with multiple techniques"""
        
        results = {}
        
        # Basic cross-validation
        results.update(self._basic_validation(model, X, y))
        
        # Learning curves
        results.update(self._learning_curve_analysis(model, X, y))
        
        # Feature stability
        results.update(self._feature_stability_analysis(model, X, y))
        
        # Model calibration
        results.update(self._calibration_analysis(model, X, y))
        
        # Robustness testing
        results.update(self._robustness_testing(model, X, y))
        
        return results
    
    def _production_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Production-ready validation with temporal and stability tests"""
        
        results = self._comprehensive_validation(model, X, y)
        
        # Temporal validation (if date columns exist)
        results.update(self._temporal_validation(model, X, y))
        
        # Performance stability over time
        results.update(self._stability_over_time(model, X, y))
        
        # Prediction consistency
        results.update(self._prediction_consistency(model, X))
        
        # Resource usage analysis
        results.update(self._resource_usage_analysis(model, X))
        
        return results
    
    def _learning_curve_analysis(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze learning curves to detect overfitting/underfitting"""
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes,
            cv=5, scoring='roc_auc', random_state=self.random_state
        )
        
        # Calculate learning curve metrics
        final_train_score = np.mean(train_scores[-1])
        final_val_score = np.mean(val_scores[-1])
        learning_gap = final_train_score - final_val_score
        
        # Detect convergence
        val_score_trend = np.polyfit(range(len(val_scores)), np.mean(val_scores, axis=1), 1)[0]
        
        return {
            'learning_curve_final_train_score': final_train_score,
            'learning_curve_final_val_score': final_val_score,
            'learning_curve_gap': learning_gap,
            'learning_curve_converged': abs(val_score_trend) < 0.01,
            'learning_curve_trend': val_score_trend
        }
    
    def _feature_stability_analysis(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze feature importance stability across CV folds"""
        
        if not hasattr(model, 'feature_importances_'):
            return {'feature_stability_available': False}
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        importance_variations = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            
            # Clone and fit model on fold
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train_fold, y_train_fold)
            
            if hasattr(fold_model, 'feature_importances_'):
                importance_variations.append(fold_model.feature_importances_)
        
        if importance_variations:
            importance_matrix = np.array(importance_variations)
            importance_std = np.std(importance_matrix, axis=0)
            importance_stability = 1 - np.mean(importance_std)
            
            return {
                'feature_stability_available': True,
                'feature_importance_stability': importance_stability,
                'most_stable_features': np.argsort(importance_std)[:5].tolist(),
                'least_stable_features': np.argsort(importance_std)[-5:].tolist()
            }
        
        return {'feature_stability_available': False}
    
    def _calibration_analysis(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze model calibration"""
        
        if not hasattr(model, 'predict_proba'):
            return {'calibration_available': False}
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        all_y_true = []
        all_y_prob = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Clone and fit model on fold
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train_fold, y_train_fold)
            
            y_prob = fold_model.predict_proba(X_val_fold)[:, 1]
            
            all_y_true.extend(y_val_fold)
            all_y_prob.extend(y_prob)
        
        # Calculate calibration metrics
        fraction_positives, mean_predicted_value = calibration_curve(
            all_y_true, all_y_prob, n_bins=10
        )
        
        # Brier score (lower is better)
        brier_score = brier_score_loss(all_y_true, all_y_prob)
        
        # Calibration error
        calibration_error = np.mean(np.abs(fraction_positives - mean_predicted_value))
        
        return {
            'calibration_available': True,
            'brier_score': brier_score,
            'calibration_error': calibration_error,
            'well_calibrated': calibration_error < 0.1
        }
    
    def _robustness_testing(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Test model robustness to data perturbations"""
        
        baseline_score = cross_val_score(model, X, y, cv=3, scoring='roc_auc').mean()
        
        robustness_results = {'baseline_score': baseline_score}
        
        # Test with noise injection
        noise_levels = [0.01, 0.05, 0.1]
        for noise_level in noise_levels:
            X_noisy = X.copy()
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            noise = np.random.normal(0, noise_level * X_noisy[numeric_cols].std(), X_noisy[numeric_cols].shape)
            X_noisy[numeric_cols] += noise
            
            noisy_score = cross_val_score(model, X_noisy, y, cv=3, scoring='roc_auc').mean()
            score_drop = baseline_score - noisy_score
            
            robustness_results[f'noise_{noise_level}_score'] = noisy_score
            robustness_results[f'noise_{noise_level}_drop'] = score_drop
        
        # Overall robustness assessment
        max_score_drop = max([robustness_results[k] for k in robustness_results.keys() if k.endswith('_drop')])
        robustness_results['robust_to_noise'] = max_score_drop < 0.05
        
        return robustness_results
    
    def _temporal_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Temporal validation for time-aware evaluation"""
        
        # Check if temporal features exist
        date_columns = [col for col in X.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if not date_columns:
            return {'temporal_validation_available': False}
        
        # Simple temporal split (last 20% as test)
        split_idx = int(len(X) * 0.8)
        X_train_temp = X.iloc[:split_idx]
        y_train_temp = y.iloc[:split_idx]
        X_test_temp = X.iloc[split_idx:]
        y_test_temp = y.iloc[split_idx:]
        
        # Train on earlier data, test on later data
        from sklearn.base import clone
        temp_model = clone(model)
        temp_model.fit(X_train_temp, y_train_temp)
        
        if hasattr(temp_model, 'predict_proba'):
            y_pred_proba = temp_model.predict_proba(X_test_temp)[:, 1]
            temporal_auc = roc_auc_score(y_test_temp, y_pred_proba)
        else:
            temporal_auc = None
        
        y_pred = temp_model.predict(X_test_temp)
        temporal_accuracy = accuracy_score(y_test_temp, y_pred)
        
        return {
            'temporal_validation_available': True,
            'temporal_accuracy': temporal_accuracy,
            'temporal_auc': temporal_auc,
            'temporal_split_ratio': 0.8
        }
    
    def _stability_over_time(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Test model stability over different time periods"""
        
        # Simulate temporal stability by random subsampling
        n_samples = len(X)
        time_windows = [0.7, 0.8, 0.9, 1.0]
        
        stability_scores = []
        
        for window in time_windows:
            sample_size = int(n_samples * window)
            
            # Random sample to simulate different time periods
            sample_idx = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            
            score = cross_val_score(model, X_sample, y_sample, cv=3, scoring='roc_auc').mean()
            stability_scores.append(score)
        
        score_variance = np.var(stability_scores)
        score_stability = 1 - score_variance
        
        return {
            'stability_scores': stability_scores,
            'stability_variance': score_variance,
            'stability_metric': score_stability,
            'stable_over_time': score_variance < 0.01
        }
    
    def _prediction_consistency(self, model: Any, X: pd.DataFrame) -> Dict[str, Any]:
        """Test prediction consistency"""
        
        # Test prediction consistency on same data
        if hasattr(model, 'predict_proba'):
            pred1 = model.predict_proba(X)[:, 1]
            pred2 = model.predict_proba(X)[:, 1]
            consistency = np.corrcoef(pred1, pred2)[0, 1]
        else:
            pred1 = model.predict(X)
            pred2 = model.predict(X)
            consistency = np.mean(pred1 == pred2)
        
        return {
            'prediction_consistency': consistency,
            'predictions_consistent': consistency > 0.99
        }
    
    def _resource_usage_analysis(self, model: Any, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze computational resource usage"""
        
        import time
        import psutil
        import os
        
        # Memory usage
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Prediction timing
        start_time = time.time()
        _ = model.predict(X.head(100))  # Test on small sample
        prediction_time = (time.time() - start_time) * 1000  # milliseconds
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        return {
            'prediction_time_100_samples_ms': prediction_time,
            'prediction_time_per_sample_ms': prediction_time / 100,
            'memory_usage_mb': memory_usage,
            'suitable_for_realtime': prediction_time < 100  # < 100ms for 100 samples
        }
    
    def generate_validation_report(self) -> str:
        """Generate human-readable validation report"""
        
        if not self.validation_results:
            return "No validation results available. Run validate_model() first."
        
        report = []
        report.append("=" * 60)
        report.append("MODEL VALIDATION REPORT")
        report.append("=" * 60)
        
        results = self.validation_results
        
        # Basic metrics
        report.append("\n📊 BASIC PERFORMANCE METRICS:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if f'{metric}_cv_mean' in results:
                mean_val = results[f'{metric}_cv_mean']
                std_val = results[f'{metric}_cv_std']
                report.append(f"  {metric.upper()}: {mean_val:.4f} (±{std_val:.4f})")
        
        # Overfitting assessment
        report.append("\n🔍 OVERFITTING ANALYSIS:")
        for metric in ['accuracy', 'roc_auc']:
            if f'{metric}_overfitting' in results:
                overfitting = results[f'{metric}_overfitting']
                status = "⚠️ High" if overfitting > 0.1 else "✅ Low"
                report.append(f"  {metric.upper()} overfitting: {overfitting:.4f} ({status})")
        
        # Model quality assessment
        report.append("\n🎯 MODEL QUALITY:")
        
        if 'well_calibrated' in results:
            calibration = "✅ Well calibrated" if results['well_calibrated'] else "⚠️ Poorly calibrated"
            report.append(f"  Calibration: {calibration}")
        
        if 'robust_to_noise' in results:
            robustness = "✅ Robust" if results['robust_to_noise'] else "⚠️ Sensitive to noise"
            report.append(f"  Robustness: {robustness}")
        
        if 'stable_over_time' in results:
            stability = "✅ Stable" if results['stable_over_time'] else "⚠️ Unstable"
            report.append(f"  Temporal stability: {stability}")
        
        # Production readiness
        report.append("\n🚀 PRODUCTION READINESS:")
        
        if 'suitable_for_realtime' in results:
            realtime = "✅ Suitable" if results['suitable_for_realtime'] else "⚠️ Too slow"
            report.append(f"  Real-time prediction: {realtime}")
        
        if 'predictions_consistent' in results:
            consistency = "✅ Consistent" if results['predictions_consistent'] else "⚠️ Inconsistent"
            report.append(f"  Prediction consistency: {consistency}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


class PerformanceAnalyzer:
    """
    Advanced performance analysis with detailed metrics and visualizations
    
    Provides in-depth analysis of model performance including error analysis,
    threshold optimization, and performance breakdowns by data segments.
    """
    
    def __init__(self):
        """Initialize performance analyzer"""
        self.analysis_results = {}
        
    def analyze_model_performance(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                                X_train: pd.DataFrame = None, y_train: pd.Series = None) -> Dict[str, Any]:
        """
        Comprehensive performance analysis
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            X_train: Training features (optional)
            y_train: Training labels (optional)
            
        Returns:
            Dictionary containing detailed performance analysis
        """
        
        logger.info("Starting comprehensive performance analysis")
        
        results = {
            'analysis_date': datetime.now().isoformat(),
            'test_set_size': len(X_test),
            'class_distribution': y_test.value_counts().to_dict()
        }
        
        # Basic performance metrics
        results.update(self._calculate_basic_metrics(model, X_test, y_test))
        
        # Advanced metrics
        results.update(self._calculate_advanced_metrics(model, X_test, y_test))
        
        # Error analysis
        results.update(self._error_analysis(model, X_test, y_test))
        
        # Threshold analysis
        results.update(self._threshold_analysis(model, X_test, y_test))
        
        # Performance by segments
        results.update(self._segment_analysis(model, X_test, y_test))
        
        # Training vs test comparison (if training data provided)
        if X_train is not None and y_train is not None:
            results.update(self._train_test_comparison(model, X_train, y_train, X_test, y_test))
        
        self.analysis_results = results
        logger.info("Performance analysis completed")
        
        return results
    
    def _calculate_basic_metrics(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Calculate basic classification metrics"""
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        if y_pred_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'avg_precision': average_precision_score(y_test, y_pred_proba),
                'log_loss': log_loss(y_test, y_pred_proba),
                'brier_score': brier_score_loss(y_test, y_pred_proba)
            })
        
        return metrics
    
    def _calculate_advanced_metrics(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Calculate advanced performance metrics"""
        
        if not hasattr(model, 'predict_proba'):
            return {}
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # ROC curve analysis
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        
        # Find optimal ROC threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = roc_thresholds[optimal_idx]
        
        # Precision-Recall curve analysis
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Find optimal PR threshold (F1-score)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = pr_thresholds[optimal_f1_idx]
        
        return {
            'optimal_roc_threshold': optimal_threshold,
            'optimal_f1_threshold': optimal_f1_threshold,
            'max_f1_score': f1_scores[optimal_f1_idx],
            'roc_curve_data': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'pr_curve_data': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        }
    
    def _error_analysis(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Analyze prediction errors"""
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Identify error types
        false_positives = (y_test == 0) & (y_pred == 1)
        false_negatives = (y_test == 1) & (y_pred == 0)
        
        error_analysis = {
            'false_positive_rate': np.mean(false_positives),
            'false_negative_rate': np.mean(false_negatives),
            'false_positive_count': int(np.sum(false_positives)),
            'false_negative_count': int(np.sum(false_negatives))
        }
        
        # Confidence analysis for errors
        if y_pred_proba is not None:
            fp_confidences = y_pred_proba[false_positives]
            fn_confidences = y_pred_proba[false_negatives]
            
            error_analysis.update({
                'fp_avg_confidence': np.mean(fp_confidences) if len(fp_confidences) > 0 else 0,
                'fn_avg_confidence': np.mean(fn_confidences) if len(fn_confidences) > 0 else 0,
                'high_confidence_errors': np.sum((fp_confidences > 0.8) | (fn_confidences < 0.2)) if len(fp_confidences) > 0 or len(fn_confidences) > 0 else 0
            })
        
        return error_analysis
    
    def _threshold_analysis(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Analyze performance across different thresholds"""
        
        if not hasattr(model, 'predict_proba'):
            return {}
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        thresholds = np.arange(0.1, 1.0, 0.05)
        
        threshold_results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            if np.sum(y_pred_thresh) == 0 or np.sum(y_pred_thresh) == len(y_pred_thresh):
                # Skip thresholds that result in all 0s or all 1s
                continue
            
            result = {
                'threshold': threshold,
                'accuracy': accuracy_score(y_test, y_pred_thresh),
                'precision': precision_score(y_test, y_pred_thresh, zero_division=0),
                'recall': recall_score(y_test, y_pred_thresh, zero_division=0),
                'f1_score': f1_score(y_test, y_pred_thresh, zero_division=0)
            }
            
            threshold_results.append(result)
        
        # Find best thresholds for different metrics
        best_thresholds = {}
        if threshold_results:
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                best_idx = np.argmax([r[metric] for r in threshold_results])
                best_thresholds[f'best_{metric}_threshold'] = threshold_results[best_idx]['threshold']
                best_thresholds[f'best_{metric}_value'] = threshold_results[best_idx][metric]
        
        return {
            'threshold_analysis': threshold_results,
            'best_thresholds': best_thresholds
        }
    
    def _segment_analysis(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Analyze performance across different data segments"""
        
        segment_results = {}
        
        # Analyze by feature ranges (for numeric features)
        numeric_features = X_test.select_dtypes(include=[np.number]).columns[:5]  # Limit to 5 features
        
        for feature in numeric_features:
            try:
                # Create quintiles
                quintiles = pd.qcut(X_test[feature], q=5, duplicates='drop', labels=False)
                
                for quintile in range(quintiles.max() + 1):
                    mask = quintiles == quintile
                    if np.sum(mask) < 10:  # Skip small segments
                        continue
                    
                    X_segment = X_test[mask]
                    y_segment = y_test[mask]
                    
                    y_pred_segment = model.predict(X_segment)
                    segment_accuracy = accuracy_score(y_segment, y_pred_segment)
                    
                    segment_key = f'{feature}_quintile_{quintile}'
                    segment_results[segment_key] = {
                        'size': int(np.sum(mask)),
                        'accuracy': segment_accuracy,
                        'churn_rate': np.mean(y_segment)
                    }
                    
            except Exception as e:
                logger.warning(f"Could not analyze segment for {feature}: {e}")
                continue
        
        return {'segment_analysis': segment_results}
    
    def _train_test_comparison(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Compare performance between training and test sets"""
        
        # Training set performance
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Test set performance
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        comparison = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'accuracy_drop': train_accuracy - test_accuracy,
            'overfitting_detected': (train_accuracy - test_accuracy) > 0.1
        }
        
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)
            
            comparison.update({
                'train_auc': train_auc,
                'test_auc': test_auc,
                'auc_drop': train_auc - test_auc
            })
        
        return comparison
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        
        if not self.analysis_results:
            return "No analysis results available. Run analyze_model_performance() first."
        
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)
        
        results = self.analysis_results
        
        # Basic performance
        report.append("\n📊 CLASSIFICATION PERFORMANCE:")
        report.append(f"  Accuracy: {results.get('accuracy', 0):.4f}")
        report.append(f"  Precision: {results.get('precision', 0):.4f}")
        report.append(f"  Recall: {results.get('recall', 0):.4f}")
        report.append(f"  F1-Score: {results.get('f1_score', 0):.4f}")
        if 'roc_auc' in results:
            report.append(f"  ROC-AUC: {results['roc_auc']:.4f}")
        
        # Error analysis
        report.append("\n❌ ERROR ANALYSIS:")
        report.append(f"  False Positives: {results.get('false_positive_count', 0)}")
        report.append(f"  False Negatives: {results.get('false_negative_count', 0)}")
        report.append(f"  FP Rate: {results.get('false_positive_rate', 0):.4f}")
        report.append(f"  FN Rate: {results.get('false_negative_rate', 0):.4f}")
        
        # Threshold recommendations
        if 'best_thresholds' in results:
            report.append("\n🎯 OPTIMAL THRESHOLDS:")
            best = results['best_thresholds']
            for metric in ['f1_score', 'accuracy']:
                if f'best_{metric}_threshold' in best:
                    threshold = best[f'best_{metric}_threshold']
                    value = best[f'best_{metric}_value']
                    report.append(f"  Best {metric}: {threshold:.3f} (score: {value:.4f})")
        
        # Train vs test comparison
        if 'overfitting_detected' in results:
            overfitting = "⚠️ Yes" if results['overfitting_detected'] else "✅ No"
            report.append(f"\n🔍 OVERFITTING DETECTED: {overfitting}")
            report.append(f"  Accuracy Drop: {results.get('accuracy_drop', 0):.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


class BusinessMetricsCalculator:
    """
    Calculator for business-relevant metrics and ROI analysis
    
    Translates model performance into business impact metrics
    including cost-benefit analysis and revenue projections.
    """
    
    def __init__(self, business_parameters: Dict[str, float] = None):
        """
        Initialize business metrics calculator
        
        Args:
            business_parameters: Dictionary of business-specific parameters
        """
        self.business_parameters = business_parameters or {
            'cost_per_intervention': 10.0,
            'revenue_per_retained_player': 50.0,
            'intervention_success_rate': 0.25,
            'average_player_lifetime_months': 12,
            'monthly_revenue_per_player': 15.0,
            'acquisition_cost_per_player': 25.0
        }
        
    def calculate_business_metrics(self, y_true: pd.Series, y_pred: pd.Series, 
                                 y_pred_proba: pd.Series = None,
                                 player_segments: pd.Series = None) -> Dict[str, Any]:
        """
        Calculate comprehensive business metrics
        
        Args:
            y_true: Actual churn labels
            y_pred: Predicted churn labels
            y_pred_proba: Predicted churn probabilities
            player_segments: Player value segments (optional)
            
        Returns:
            Dictionary containing business metrics
        """
        
        logger.info("Calculating business impact metrics")
        
        # Basic confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        results = {
            'calculation_date': datetime.now().isoformat(),
            'total_players': len(y_true),
            'actual_churners': int(np.sum(y_true)),
            'predicted_churners': int(np.sum(y_pred)),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        # Cost-benefit analysis
        results.update(self._calculate_cost_benefit(tp, fp, tn, fn))
        
        # Revenue impact
        results.update(self._calculate_revenue_impact(tp, fp, tn, fn))
        
        # ROI analysis
        results.update(self._calculate_roi_analysis(tp, fp, tn, fn))
        
        # Intervention efficiency
        results.update(self._calculate_intervention_efficiency(tp, fp, tn, fn))
        
        # Player value analysis
        if player_segments is not None:
            results.update(self._calculate_segment_value(y_true, y_pred, player_segments))
        
        # Threshold optimization for business metrics
        if y_pred_proba is not None:
            results.update(self._optimize_business_threshold(y_true, y_pred_proba))
        
        return results
    
    def _calculate_cost_benefit(self, tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
        """Calculate cost-benefit metrics"""
        
        params = self.business_parameters
        
        # Intervention costs
        total_interventions = tp + fp
        intervention_cost = total_interventions * params['cost_per_intervention']
        
        # Benefits from successful interventions
        successful_interventions = tp * params['intervention_success_rate']
        intervention_revenue = successful_interventions * params['revenue_per_retained_player']
        
        # Costs from missed opportunities (false negatives)
        missed_opportunity_cost = fn * params['revenue_per_retained_player']
        
        # Net benefit
        net_benefit = intervention_revenue - intervention_cost
        
        return {
            'total_intervention_cost': intervention_cost,
            'intervention_revenue': intervention_revenue,
            'missed_opportunity_cost': missed_opportunity_cost,
            'net_benefit': net_benefit,
            'cost_per_true_positive': intervention_cost / tp if tp > 0 else float('inf'),
            'revenue_per_intervention': intervention_revenue / total_interventions if total_interventions > 0 else 0
        }
    
    def _calculate_revenue_impact(self, tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
        """Calculate revenue impact metrics"""
        
        params = self.business_parameters
        
        # Revenue from retained players
        retained_players = tp * params['intervention_success_rate']
        monthly_revenue_saved = retained_players * params['monthly_revenue_per_player']
        annual_revenue_saved = monthly_revenue_saved * 12
        
        # Lifetime value impact
        lifetime_revenue_saved = retained_players * params['monthly_revenue_per_player'] * params['average_player_lifetime_months']
        
        # Revenue lost from missed churners
        revenue_lost_from_fn = fn * params['monthly_revenue_per_player'] * params['average_player_lifetime_months']
        
        return {
            'players_successfully_retained': retained_players,
            'monthly_revenue_saved': monthly_revenue_saved,
            'annual_revenue_saved': annual_revenue_saved,
            'lifetime_revenue_saved': lifetime_revenue_saved,
            'revenue_lost_from_missed_churners': revenue_lost_from_fn,
            'net_revenue_impact': lifetime_revenue_saved - revenue_lost_from_fn
        }
    
    def _calculate_roi_analysis(self, tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
        """Calculate ROI and related metrics"""
        
        params = self.business_parameters
        
        # Investment
        total_investment = (tp + fp) * params['cost_per_intervention']
        
        # Returns
        successful_retentions = tp * params['intervention_success_rate']
        total_returns = successful_retentions * params['revenue_per_retained_player']
        
        # ROI calculation
        roi = (total_returns - total_investment) / total_investment if total_investment > 0 else 0
        
        # Payback period (months)
        monthly_returns = successful_retentions * params['monthly_revenue_per_player']
        payback_months = total_investment / monthly_returns if monthly_returns > 0 else float('inf')
        
        return {
            'total_investment': total_investment,
            'total_returns': total_returns,
            'roi_ratio': roi,
            'roi_percentage': roi * 100,
            'payback_period_months': payback_months,
            'profitable': roi > 0,
            'break_even_retention_rate': total_investment / (tp * params['revenue_per_retained_player']) if tp > 0 else float('inf')
        }
    
    def _calculate_intervention_efficiency(self, tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
        """Calculate intervention efficiency metrics"""
        
        total_interventions = tp + fp
        
        return {
            'intervention_precision': tp / total_interventions if total_interventions > 0 else 0,
            'intervention_recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'intervention_efficiency': tp / total_interventions if total_interventions > 0 else 0,
            'unnecessary_interventions': fp,
            'missed_interventions': fn,
            'intervention_accuracy': (tp + tn) / (tp + tn + fp + fn)
        }
    
    def _calculate_segment_value(self, y_true: pd.Series, y_pred: pd.Series, 
                                player_segments: pd.Series) -> Dict[str, Any]:
        """Calculate business metrics by player value segments"""
        
        segment_metrics = {}
        
        for segment in player_segments.unique():
            mask = player_segments == segment
            
            if np.sum(mask) < 10:  # Skip small segments
                continue
            
            segment_y_true = y_true[mask]
            segment_y_pred = y_pred[mask]
            
            # Calculate confusion matrix for segment
            tn, fp, fn, tp = confusion_matrix(segment_y_true, segment_y_pred).ravel()
            
            # Segment-specific value multipliers
            value_multipliers = {
                'VIP': 5.0,
                'High': 3.0,
                'Medium': 1.5,
                'Low': 1.0
            }
            
            multiplier = value_multipliers.get(segment, 1.0)
            
            # Calculate segment ROI with value multiplier
            segment_investment = (tp + fp) * self.business_parameters['cost_per_intervention']
            segment_returns = tp * self.business_parameters['intervention_success_rate'] * \
                            self.business_parameters['revenue_per_retained_player'] * multiplier
            
            segment_roi = (segment_returns - segment_investment) / segment_investment if segment_investment > 0 else 0
            
            segment_metrics[f'segment_{segment}'] = {
                'players': int(np.sum(mask)),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'roi': segment_roi,
                'value_multiplier': multiplier,
                'segment_returns': segment_returns
            }
        
        return {'segment_analysis': segment_metrics}
    
    def _optimize_business_threshold(self, y_true: pd.Series, y_pred_proba: pd.Series) -> Dict[str, Any]:
        """Optimize threshold for maximum business value"""
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        threshold_results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
            
            # Calculate business metrics for this threshold
            total_cost = (tp + fp) * self.business_parameters['cost_per_intervention']
            total_benefit = tp * self.business_parameters['intervention_success_rate'] * \
                          self.business_parameters['revenue_per_retained_player']
            
            net_value = total_benefit - total_cost
            roi = (total_benefit - total_cost) / total_cost if total_cost > 0 else 0
            
            threshold_results.append({
                'threshold': threshold,
                'net_value': net_value,
                'roi': roi,
                'total_cost': total_cost,
                'total_benefit': total_benefit,
                'interventions': tp + fp
            })
        
        # Find optimal thresholds
        best_net_value_idx = np.argmax([r['net_value'] for r in threshold_results])
        best_roi_idx = np.argmax([r['roi'] for r in threshold_results])
        
        return {
            'threshold_optimization': threshold_results,
            'optimal_threshold_net_value': threshold_results[best_net_value_idx]['threshold'],
            'optimal_threshold_roi': threshold_results[best_roi_idx]['threshold'],
            'max_net_value': threshold_results[best_net_value_idx]['net_value'],
            'max_roi': threshold_results[best_roi_idx]['roi']
        }
    
    def generate_business_report(self, business_metrics: Dict[str, Any]) -> str:
        """Generate business-focused report"""
        
        report = []
        report.append("=" * 60)
        report.append("BUSINESS IMPACT ANALYSIS")
        report.append("=" * 60)
        
        # Executive Summary
        report.append("\n💼 EXECUTIVE SUMMARY:")
        report.append(f"  Total Players Analyzed: {business_metrics.get('total_players', 0):,}")
        report.append(f"  Players at Risk: {business_metrics.get('predicted_churners', 0):,}")
        report.append(f"  ROI: {business_metrics.get('roi_percentage', 0):.1f}%")
        report.append(f"  Net Benefit: ${business_metrics.get('net_benefit', 0):,.2f}")
        
        # Financial Impact
        report.append("\n💰 FINANCIAL IMPACT:")
        report.append(f"  Intervention Investment: ${business_metrics.get('total_intervention_cost', 0):,.2f}")
        report.append(f"  Revenue from Retention: ${business_metrics.get('intervention_revenue', 0):,.2f}")
        report.append(f"  Annual Revenue Saved: ${business_metrics.get('annual_revenue_saved', 0):,.2f}")
        
        # Efficiency Metrics
        report.append("\n⚡ INTERVENTION EFFICIENCY:")
        report.append(f"  Precision: {business_metrics.get('intervention_precision', 0):.3f}")
        report.append(f"  Players Successfully Retained: {business_metrics.get('players_successfully_retained', 0):.0f}")
        report.append(f"  Unnecessary Interventions: {business_metrics.get('unnecessary_interventions', 0)}")
        
        # Recommendations
        report.append("\n🎯 RECOMMENDATIONS:")
        if business_metrics.get('profitable', False):
            report.append("  ✅ Model is profitable - proceed with deployment")
        else:
            report.append("  ⚠️ Model needs optimization before deployment")
        
        if 'optimal_threshold_net_value' in business_metrics:
            threshold = business_metrics['optimal_threshold_net_value']
            report.append(f"  📊 Use threshold {threshold:.3f} for maximum business value")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


class ModelComparator:
    """
    Compare multiple models across various metrics and business criteria
    
    Provides comprehensive comparison including statistical significance testing
    and business impact analysis for model selection.
    """
    
    def __init__(self):
        """Initialize model comparator"""
        self.comparison_results = {}
        
    def compare_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series,
                      business_parameters: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Compare multiple models comprehensively
        
        Args:
            models: Dictionary of {model_name: model_object}
            X_test: Test features
            y_test: Test labels
            business_parameters: Business-specific parameters
            
        Returns:
            Comprehensive comparison results
        """
        
        logger.info(f"Comparing {len(models)} models")
        
        results = {
            'comparison_date': datetime.now().isoformat(),
            'models_compared': list(models.keys()),
            'test_set_size': len(X_test),
            'model_performances': {},
            'rankings': {},
            'recommendations': {}
        }
        
        # Evaluate each model
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}")
            
            try:
                # Performance analysis
                analyzer = PerformanceAnalyzer()
                performance = analyzer.analyze_model_performance(model, X_test, y_test)
                
                # Business metrics
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                calculator = BusinessMetricsCalculator(business_parameters)
                business_metrics = calculator.calculate_business_metrics(y_test, y_pred, y_pred_proba)
                
                results['model_performances'][model_name] = {
                    'performance_metrics': performance,
                    'business_metrics': business_metrics
                }
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        # Generate rankings
        results['rankings'] = self._generate_rankings(results['model_performances'])
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results['model_performances'], results['rankings'])
        
        # Statistical significance testing
        results['significance_tests'] = self._statistical_significance_testing(models, X_test, y_test)
        
        self.comparison_results = results
        
        return results
    
    def _generate_rankings(self, model_performances: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate rankings for different criteria"""
        
        rankings = {}
        
        # Performance-based rankings
        criteria = {
            'accuracy': lambda x: x['performance_metrics'].get('accuracy', 0),
            'roc_auc': lambda x: x['performance_metrics'].get('roc_auc', 0),
            'f1_score': lambda x: x['performance_metrics'].get('f1_score', 0),
            'precision': lambda x: x['performance_metrics'].get('precision', 0),
            'recall': lambda x: x['performance_metrics'].get('recall', 0)
        }
        
        for criterion, score_func in criteria.items():
            model_scores = {name: score_func(perf) for name, perf in model_performances.items()}
            rankings[criterion] = sorted(model_scores.keys(), key=lambda x: model_scores[x], reverse=True)
        
        # Business-based rankings
        business_criteria = {
            'roi': lambda x: x['business_metrics'].get('roi_percentage', 0),
            'net_benefit': lambda x: x['business_metrics'].get('net_benefit', 0),
            'revenue_impact': lambda x: x['business_metrics'].get('net_revenue_impact', 0)
        }
        
        for criterion, score_func in business_criteria.items():
            model_scores = {name: score_func(perf) for name, perf in model_performances.items()}
            rankings[criterion] = sorted(model_scores.keys(), key=lambda x: model_scores[x], reverse=True)
        
        # Overall ranking (weighted combination)
        overall_scores = {}
        for model_name, performance in model_performances.items():
            # Combine technical and business metrics
            tech_score = (
                performance['performance_metrics'].get('roc_auc', 0) * 0.4 +
                performance['performance_metrics'].get('f1_score', 0) * 0.3 +
                performance['performance_metrics'].get('precision', 0) * 0.3
            )
            
            # Normalize ROI to 0-1 scale
            roi = performance['business_metrics'].get('roi_percentage', 0)
            business_score = min(1.0, max(0.0, (roi + 100) / 200))  # Assume ROI range -100% to +100%
            
            overall_scores[model_name] = tech_score * 0.6 + business_score * 0.4
        
        rankings['overall'] = sorted(overall_scores.keys(), key=lambda x: overall_scores[x], reverse=True)
        
        return rankings
    
    def _generate_recommendations(self, model_performances: Dict[str, Any], 
                                rankings: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate model selection recommendations"""
        
        recommendations = {
            'best_overall': rankings['overall'][0] if rankings['overall'] else None,
            'best_for_accuracy': rankings['accuracy'][0] if 'accuracy' in rankings else None,
            'best_for_business': rankings['roi'][0] if 'roi' in rankings else None,
            'recommendations_by_use_case': {}
        }
        
        # Use case specific recommendations
        use_cases = {
            'high_precision_needed': {
                'criterion': 'precision',
                'description': 'When false positives are very costly'
            },
            'high_recall_needed': {
                'criterion': 'recall', 
                'description': 'When missing churners is very costly'
            },
            'balanced_performance': {
                'criterion': 'f1_score',
                'description': 'When you need balanced precision and recall'
            },
            'profit_maximization': {
                'criterion': 'roi',
                'description': 'When maximizing ROI is the primary goal'
            }
        }
        
        for use_case, config in use_cases.items():
            criterion = config['criterion']
            if criterion in rankings and rankings[criterion]:
                best_model = rankings[criterion][0]
                recommendations['recommendations_by_use_case'][use_case] = {
                    'recommended_model': best_model,
                    'description': config['description'],
                    'key_metric': criterion
                }
        
        # Performance gaps analysis
        if len(model_performances) > 1:
            recommendations['performance_gaps'] = self._analyze_performance_gaps(model_performances)
        
        return recommendations
    
    def _analyze_performance_gaps(self, model_performances: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance gaps between models"""
        
        gaps = {}
        
        # Get performance metrics for all models
        metrics = ['accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']
        
        for metric in metrics:
            scores = []
            for perf in model_performances.values():
                score = perf['performance_metrics'].get(metric, 0)
                scores.append(score)
            
            if scores:
                gaps[metric] = {
                    'max': max(scores),
                    'min': min(scores),
                    'range': max(scores) - min(scores),
                    'significant_gap': (max(scores) - min(scores)) > 0.05
                }
        
        return gaps
    
    def _statistical_significance_testing(self, models: Dict[str, Any], 
                                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Test statistical significance of performance differences"""
        
        from scipy import stats
        
        # Get predictions from all models
        model_predictions = {}
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    model_predictions[model_name] = model.predict_proba(X_test)[:, 1]
                else:
                    model_predictions[model_name] = model.predict(X_test)
            except Exception as e:
                logger.warning(f"Could not get predictions from {model_name}: {e}")
        
        # Perform pairwise comparisons
        significance_results = {}
        model_names = list(model_predictions.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                
                # Compare AUC scores using DeLong's test (simplified)
                pred1 = model_predictions[model1]
                pred2 = model_predictions[model2]
                
                # Simple significance test using correlation
                correlation, p_value = stats.spearmanr(pred1, pred2)
                
                comparison_key = f"{model1}_vs_{model2}"
                significance_results[comparison_key] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'significantly_different': p_value < 0.05,
                    'models_compared': [model1, model2]
                }
        
        return significance_results
    
    def generate_comparison_report(self) -> str:
        """Generate comprehensive model comparison report"""
        
        if not self.comparison_results:
            return "No comparison results available. Run compare_models() first."
        
        report = []
        report.append("=" * 60)
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 60)
        
        results = self.comparison_results
        
        # Overall rankings
        report.append("\n🏆 OVERALL RANKINGS:")
        if 'overall' in results['rankings']:
            for i, model in enumerate(results['rankings']['overall'], 1):
                report.append(f"  {i}. {model}")
        
        # Best models by category
        report.append("\n🎯 CATEGORY WINNERS:")
        recommendations = results.get('recommendations', {})
        
        for category, info in recommendations.get('recommendations_by_use_case', {}).items():
            model = info['recommended_model']
            description = info['description']
            report.append(f"  {category.replace('_', ' ').title()}: {model}")
            report.append(f"    └─ {description}")
        
        # Performance comparison
        report.append("\n📊 PERFORMANCE COMPARISON:")
        
        for model_name, performance in results['model_performances'].items():
            metrics = performance['performance_metrics']
            business = performance['business_metrics']
            
            report.append(f"\n  {model_name.upper()}:")
            report.append(f"    ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            report.append(f"    F1-Score: {metrics.get('f1_score', 0):.4f}")
            report.append(f"    ROI: {business.get('roi_percentage', 0):.1f}%")
            report.append(f"    Net Benefit: ${business.get('net_benefit', 0):,.2f}")
        
        # Final recommendation
        report.append("\n🔥 FINAL RECOMMENDATION:")
        best_overall = recommendations.get('best_overall')
        if best_overall:
            report.append(f"  Deploy: {best_overall}")
            report.append("  This model provides the best balance of technical performance")
            report.append("  and business value across all evaluation criteria.")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def main():
    """Example usage of evaluation classes"""
    
    logger.info("Testing model evaluation modules...")
    
    # Example would require actual trained models and test data
    print("Model evaluation modules loaded successfully!")
    print("Ready for comprehensive model assessment.")

if __name__ == "__main__":
    main()