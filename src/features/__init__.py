"""
Feature Engineering Module

This module handles feature engineering, transformation, and selection
for gaming player behavior analysis and churn prediction.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

from .feature_engineering import (
    FeatureEngineer,
    EngagementFeatureCreator,
    BehavioralFeatureCreator,
    RiskFeatureCreator
)

from .feature_selection import (
    FeatureSelector,
    ImportanceBasedSelector,
    CorrelationBasedSelector,
    StatisticalSelector
)

from .feature_transformation import (
    FeatureTransformer,
    NumericTransformer,
    CategoricalTransformer,
    DateTimeTransformer
)

__all__ = [
    # Feature Engineering
    "FeatureEngineer",
    "EngagementFeatureCreator",
    "BehavioralFeatureCreator", 
    "RiskFeatureCreator",
    
    # Feature Selection
    "FeatureSelector",
    "ImportanceBasedSelector",
    "CorrelationBasedSelector",
    "StatisticalSelector",
    
    # Feature Transformation
    "FeatureTransformer",
    "NumericTransformer",
    "CategoricalTransformer",
    "DateTimeTransformer",
]

# Feature engineering constants
ENGAGEMENT_FEATURES = [
    'engagement_score',
    'activity_recency',
    'social_engagement',
    'achievement_rate',
    'session_intensity'
]

BEHAVIORAL_FEATURES = [
    'avg_spending_per_purchase',
    'game_exploration',
    'platform_loyalty',
    'genre_diversity',
    'playtime_consistency'
]

RISK_FEATURES = [
    'high_inactivity_risk',
    'low_engagement_risk',
    'no_purchase_risk',
    'social_isolation_risk',
    'total_risk_score'
]

FEATURE_CATEGORIES = {
    'engagement': ENGAGEMENT_FEATURES,
    'behavioral': BEHAVIORAL_FEATURES,
    'risk': RISK_FEATURES
}

def get_feature_info():
    """Return feature engineering module information"""
    return {
        "module": "features",
        "version": "1.0.0",
        "categories": FEATURE_CATEGORIES,
        "total_features": sum(len(features) for features in FEATURE_CATEGORIES.values()),
        "maintainer": "Rushikesh Dhumal"
    }