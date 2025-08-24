"""
Machine Learning Models Module

This module handles model training, evaluation, and prediction
for gaming player churn prediction.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

from .train_model import (
    ModelTrainer,
    FeaturePreprocessor,
    ModelEvaluator,
    HyperparameterTuner
)

from .predict_model import (
    ChurnPredictor,
    ModelPredictor,
    EnsemblePredictor,
    RealTimePredictor
)

from .model_evaluation import (
    ModelValidator,
    PerformanceAnalyzer,
    BusinessMetricsCalculator,
    ModelComparator
)

__all__ = [
    # Model Training
    "ModelTrainer",
    "FeaturePreprocessor",
    "ModelEvaluator", 
    "HyperparameterTuner",
    
    # Model Prediction
    "ChurnPredictor",
    "ModelPredictor",
    "EnsemblePredictor",
    "RealTimePredictor",
    
    # Model Evaluation
    "ModelValidator",
    "PerformanceAnalyzer",
    "BusinessMetricsCalculator",
    "ModelComparator",
]

# Model configuration constants
SUPPORTED_MODELS = [
    'logistic_regression',
    'random_forest',
    'gradient_boosting',
    'xgboost',
    'svm',
    'neural_network'
]

MODEL_CATEGORIES = {
    'linear': ['logistic_regression', 'svm'],
    'tree_based': ['random_forest', 'gradient_boosting', 'xgboost'],
    'neural': ['neural_network'],
    'ensemble': ['random_forest', 'gradient_boosting', 'xgboost']
}

DEFAULT_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc',
    'precision_recall_auc'
]

BUSINESS_METRICS = [
    'true_positives',
    'false_positives',
    'true_negatives', 
    'false_negatives',
    'cost_benefit_ratio',
    'revenue_impact',
    'intervention_effectiveness'
]

def get_model_info():
    """Return models module information"""
    return {
        "module": "models",
        "version": "1.0.0",
        "supported_models": SUPPORTED_MODELS,
        "model_categories": MODEL_CATEGORIES,
        "default_metrics": DEFAULT_METRICS,
        "business_metrics": BUSINESS_METRICS,
        "maintainer": "Rushikesh Dhumal"
    }

def get_recommended_models():
    """Return recommended models for churn prediction"""
    return {
        "high_accuracy": ["xgboost", "random_forest"],
        "interpretable": ["logistic_regression", "random_forest"],
        "fast_training": ["logistic_regression", "svm"],
        "robust": ["random_forest", "gradient_boosting"],
        "ensemble": ["xgboost", "gradient_boosting", "random_forest"]
    }