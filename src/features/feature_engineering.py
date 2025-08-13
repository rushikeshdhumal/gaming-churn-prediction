"""
Feature Engineering Module for Gaming Player Behavior Analysis

This module creates advanced features for churn prediction including
engagement metrics, behavioral patterns, and risk indicators.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Main feature engineering orchestrator"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_columns = list(df.columns)
        self.engineered_features = []
        
    def create_all_features(self) -> pd.DataFrame:
        """Create all engineered features"""
        logger.info("Starting comprehensive feature engineering...")
        
        # Create different categories of features
        self.create_engagement_features()
        self.create_behavioral_features()
        self.create_risk_features()
        self.create_temporal_features()
        self.create_social_features()
        self.create_spending_features()
        self.encode_categorical_features()
        
        # Clean up intermediate columns
        self._cleanup_intermediate_features()
        
        logger.info(f"Feature engineering complete. Created {len(self.engineered_features)} new features")
        logger.info(f"Final dataset shape: {self.df.shape}")
        
        return self.df
    
    def create_engagement_features(self):
        """Create engagement-based features"""
        logger.info("Creating engagement features...")
        
        # Engagement score - composite metric
        self.df['engagement_score'] = (
            np.log1p(self.df.get('total_playtime_hours', 0)) * 0.3 +
            np.log1p(self.df.get('achievements_unlocked', 0)) * 0.2 +
            np.log1p(self.df.get('friends_count', 0)) * 0.2 +
            np.log1p(self.df.get('forum_posts', 0) + self.df.get('reviews_written', 0)) * 0.15 +
            np.log1p(self.df.get('sessions_last_week', 0)) * 0.15
        )
        
        # Activity recency - exponential decay based on last login
        self.df['activity_recency'] = np.exp(-self.df.get('last_login_days_ago', 30) / 7)
        
        # Session quality metrics
        self.df['session_quality'] = np.where(
            self.df.get('avg_session_duration', 0) > 0,
            self.df.get('total_playtime_hours', 0) / (self.df.get('sessions_last_week', 1) * 4),  # Avg weekly sessions
            0
        )
        
        # Achievement rate per hour
        self.df['achievement_rate'] = np.where(
            self.df.get('total_playtime_hours', 0) > 0,
            self.df.get('achievements_unlocked', 0) / self.df.get('total_playtime_hours', 1),
            0
        )
        
        self.engineered_features.extend([
            'engagement_score', 'activity_recency', 'session_quality', 'achievement_rate'
        ])
        
    def create_behavioral_features(self):
        """Create behavioral pattern features"""
        logger.info("Creating behavioral features...")
        
        # Session intensity - playtime per day since registration
        self.df['session_intensity'] = np.where(
            self.df.get('days_since_registration', 1) > 0,
            self.df.get('total_playtime_hours', 0) / self.df.get('days_since_registration', 1),
            0
        )
        
        # Game exploration - variety relative to playtime
        self.df['game_exploration'] = np.where(
            self.df.get('total_playtime_hours', 0) > 0,
            self.df.get('games_owned', 0) / np.log1p(self.df.get('total_playtime_hours', 1)),
            0
        )
        
        # Purchase behavior patterns
        self.df['avg_spending_per_purchase'] = np.where(
            self.df.get('purchases_last_month', 0) > 0,
            self.df.get('total_spent', 0) / self.df.get('purchases_last_month', 1),
            0
        )
        
        # Gaming consistency - how regular is the playing pattern
        self.df['gaming_consistency'] = np.where(
            self.df.get('sessions_last_week', 0) > 0,
            self.df.get('avg_session_duration', 0) / (self.df.get('total_playtime_hours', 1) / self.df.get('sessions_last_week', 1)),
            0
        )
        
        # Account progression rate
        self.df['account_progression'] = np.where(
            self.df.get('days_since_registration', 1) > 0,
            self.df.get('account_level', 0) / np.log1p(self.df.get('days_since_registration', 1)),
            0
        )
        
        self.engineered_features.extend([
            'session_intensity', 'game_exploration', 'avg_spending_per_purchase',
            'gaming_consistency', 'account_progression'
        ])
        
    def create_risk_features(self):
        """Create churn risk indicator features"""
        logger.info("Creating risk assessment features...")
        
        # Binary risk indicators
        self.df['high_inactivity_risk'] = (self.df.get('last_login_days_ago', 0) > 14).astype(int)
        self.df['low_engagement_risk'] = (self.df.get('avg_session_duration', 0) < 20).astype(int)
        self.df['no_purchase_risk'] = (self.df.get('total_spent', 0) == 0).astype(int)
        self.df['social_isolation_risk'] = (self.df.get('friends_count', 0) < 2).astype(int)
        self.df['low_achievement_risk'] = (self.df.get('achievements_unlocked', 0) < 5).astype(int)
        
        # Composite risk score
        self.df['total_risk_score'] = (
            self.df['high_inactivity_risk'] * 0.25 +
            self.df['low_engagement_risk'] * 0.20 +
            self.df['no_purchase_risk'] * 0.20 +
            self.df['social_isolation_risk'] * 0.15 +
            self.df['low_achievement_risk'] * 0.10 +
            (self.df.get('session_intensity', 0) < 0.5).astype(int) * 0.10
        )
        
        # Loyalty indicators
        self.df['loyalty_score'] = (
            (self.df.get('days_since_registration', 0) > 365).astype(int) * 0.3 +
            (self.df.get('total_spent', 0) > 50).astype(int) * 0.3 +
            (self.df.get('achievements_unlocked', 0) > 30).astype(int) * 0.2 +
            (self.df.get('friends_count', 0) > 5).astype(int) * 0.2
        )
        
        self.engineered_features.extend([
            'high_inactivity_risk', 'low_engagement_risk', 'no_purchase_risk',
            'social_isolation_risk', 'low_achievement_risk', 'total_risk_score', 'loyalty_score'
        ])
        
    def create_temporal_features(self):
        """Create time-based features"""
        logger.info("Creating temporal features...")
        
        # Account age categories
        self.df['account_age_category'] = pd.cut(
            self.df.get('days_since_registration', 0),
            bins=[0, 30, 90, 365, 730, float('inf')],
            labels=['New', 'Recent', 'Established', 'Veteran', 'Legacy']
        )
        
        # Activity recency categories
        self.df['activity_category'] = pd.cut(
            self.df.get('last_login_days_ago', 0),
            bins=[0, 1, 7, 30, 90, float('inf')],
            labels=['Today', 'This_Week', 'This_Month', 'Recent', 'Inactive']
        )
        
        # Session frequency relative to account age
        self.df['session_frequency'] = np.where(
            self.df.get('days_since_registration', 1) > 0,
            self.df.get('sessions_last_week', 0) * 52 / self.df.get('days_since_registration', 1),
            0
        )
        
        # Weekend vs weekday player (synthetic feature)
        np.random.seed(42)
        self.df['weekend_player_tendency'] = np.random.beta(2, 5, len(self.df)) * self.df['engagement_score']
        
        self.engineered_features.extend([
            'account_age_category', 'activity_category', 'session_frequency', 'weekend_player_tendency'
        ])
        
    def create_social_features(self):
        """Create social interaction features"""
        logger.info("Creating social features...")
        
        # Social engagement composite score
        self.df['social_engagement'] = (
            self.df.get('friends_count', 0) * 0.4 +
            self.df.get('forum_posts', 0) * 0.3 +
            self.df.get('reviews_written', 0) * 0.3
        )
        
        # Social activity per friend
        self.df['social_activity_per_friend'] = np.where(
            self.df.get('friends_count', 0) > 0,
            self.df['social_engagement'] / self.df.get('friends_count', 1),
            0
        )
        
        # Community participation level
        self.df['community_participation'] = (
            (self.df.get('forum_posts', 0) > 0).astype(int) +
            (self.df.get('reviews_written', 0) > 0).astype(int) +
            (self.df.get('friends_count', 0) > 10).astype(int)
        )
        
        # Social influence score (synthetic)
        self.df['social_influence'] = np.where(
            self.df.get('reviews_written', 0) > 0,
            np.log1p(self.df.get('reviews_written', 0)) * np.log1p(self.df.get('friends_count', 0)),
            0
        )
        
        self.engineered_features.extend([
            'social_engagement', 'social_activity_per_friend', 
            'community_participation', 'social_influence'
        ])
        
    def create_spending_features(self):
        """Create spending and monetization features"""
        logger.info("Creating spending features...")
        
        # Spending categories
        self.df['spending_category'] = pd.cut(
            self.df.get('total_spent', 0),
            bins=[0, 0.01, 20, 50, 100, float('inf')],
            labels=['Free', 'Light', 'Medium', 'Heavy', 'Whale']
        )
        
        # Value per hour played
        self.df['value_per_hour'] = np.where(
            self.df.get('total_playtime_hours', 0) > 0,
            self.df.get('total_spent', 0) / self.df.get('total_playtime_hours', 1),
            0
        )
        
        # Purchase timing (days since registration to first purchase - synthetic)
        np.random.seed(42)
        self.df['days_to_first_purchase'] = np.where(
            self.df.get('total_spent', 0) > 0,
            np.random.exponential(scale=30, size=len(self.df)) * (2 - self.df['engagement_score']),
            self.df.get('days_since_registration', 0)
        )
        
        # Spending momentum (recent vs total)
        self.df['spending_momentum'] = np.where(
            self.df.get('total_spent', 0) > 0,
            self.df.get('purchases_last_month', 0) * 30 / self.df.get('total_spent', 1),
            0
        )
        
        # ROI perception (playtime return on spending)
        self.df['perceived_roi'] = np.where(
            self.df.get('total_spent', 0) > 0,
            self.df.get('total_playtime_hours', 0) / self.df.get('total_spent', 1),
            self.df.get('total_playtime_hours', 0)
        )
        
        self.engineered_features.extend([
            'spending_category', 'value_per_hour', 'days_to_first_purchase',
            'spending_momentum', 'perceived_roi'
        ])
        
    def encode_categorical_features(self):
        """Encode categorical variables"""
        logger.info("Encoding categorical features...")
        
        # Platform preference encoding
        if 'platform_preference' in self.df.columns:
            platform_scores = {'PC': 1.0, 'Mac': 0.7, 'Linux': 0.4}
            self.df['platform_loyalty'] = self.df['platform_preference'].map(platform_scores).fillna(0.5)
            
            # One-hot encode platform preference
            platform_dummies = pd.get_dummies(self.df['platform_preference'], prefix='platform')
            self.df = pd.concat([self.df, platform_dummies], axis=1)
            
            self.engineered_features.extend(['platform_loyalty'] + list(platform_dummies.columns))
        
        # Genre preference encoding
        if 'favorite_genre' in self.df.columns:
            # Genre popularity scores (synthetic weights)
            genre_scores = {
                'Action': 0.9, 'RPG': 0.8, 'Strategy': 0.6, 
                'Simulation': 0.5, 'Indie': 0.4, 'Sports': 0.7, 'Racing': 0.6
            }
            self.df['genre_popularity'] = self.df['favorite_genre'].map(genre_scores).fillna(0.5)
            
            # One-hot encode genres
            genre_dummies = pd.get_dummies(self.df['favorite_genre'], prefix='genre')
            self.df = pd.concat([self.df, genre_dummies], axis=1)
            
            self.engineered_features.extend(['genre_popularity'] + list(genre_dummies.columns))
        
        # Region encoding
        if 'region' in self.df.columns:
            region_scores = {'NA': 1.0, 'EU': 0.9, 'ASIA': 0.8, 'OTHER': 0.6}
            self.df['region_activity'] = self.df['region'].map(region_scores).fillna(0.5)
            
            region_dummies = pd.get_dummies(self.df['region'], prefix='region')
            self.df = pd.concat([self.df, region_dummies], axis=1)
            
            self.engineered_features.extend(['region_activity'] + list(region_dummies.columns))
        
        # Age group encoding
        if 'age_group' in self.df.columns:
            age_scores = {'18-25': 0.9, '26-35': 1.0, '36-45': 0.8, '46+': 0.6}
            self.df['age_engagement'] = self.df['age_group'].map(age_scores).fillna(0.7)
            
            age_dummies = pd.get_dummies(self.df['age_group'], prefix='age')
            self.df = pd.concat([self.df, age_dummies], axis=1)
            
            self.engineered_features.extend(['age_engagement'] + list(age_dummies.columns))
    
    def _cleanup_intermediate_features(self):
        """Remove original categorical columns and intermediate features"""
        columns_to_remove = [
            'registration_date', 'favorite_genre', 'platform_preference', 
            'region', 'age_group', 'engagement_level'
        ]
        
        for col in columns_to_remove:
            if col in self.df.columns and col not in self.engineered_features:
                self.df = self.df.drop(columns=[col])
        
        # Handle missing values in engineered features
        for feature in self.engineered_features:
            if feature in self.df.columns:
                if self.df[feature].dtype in ['object', 'category']:
                    self.df[feature] = self.df[feature].fillna('Unknown')
                else:
                    self.df[feature] = self.df[feature].fillna(0)
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Return features grouped by category for analysis"""
        return {
            'engagement': [f for f in self.engineered_features if 'engagement' in f or 'activity' in f or 'achievement' in f],
            'behavioral': [f for f in self.engineered_features if 'session' in f or 'game' in f or 'consistency' in f],
            'risk': [f for f in self.engineered_features if 'risk' in f or 'loyalty' in f],
            'temporal': [f for f in self.engineered_features if 'age' in f or 'frequency' in f or 'category' in f],
            'social': [f for f in self.engineered_features if 'social' in f or 'community' in f or 'friend' in f],
            'spending': [f for f in self.engineered_features if 'spending' in f or 'purchase' in f or 'value' in f or 'roi' in f],
            'categorical': [f for f in self.engineered_features if f.startswith(('platform_', 'genre_', 'region_', 'age_'))]
        }

class EngagementFeatureCreator:
    """Specialized class for creating engagement-specific features"""
    
    @staticmethod
    def create_advanced_engagement_score(df: pd.DataFrame) -> pd.Series:
        """Create sophisticated engagement score with weighted components"""
        components = {
            'playtime': np.log1p(df.get('total_playtime_hours', 0)),
            'social': np.log1p(df.get('friends_count', 0)),
            'achievements': np.log1p(df.get('achievements_unlocked', 0)),
            'content': np.log1p(df.get('forum_posts', 0) + df.get('reviews_written', 0)),
            'frequency': np.log1p(df.get('sessions_last_week', 0))
        }
        
        weights = {'playtime': 0.25, 'social': 0.20, 'achievements': 0.20, 'content': 0.15, 'frequency': 0.20}
        
        engagement_score = sum(components[comp] * weights[comp] for comp in components)
        return engagement_score

class BehavioralFeatureCreator:
    """Specialized class for creating behavioral pattern features"""
    
    @staticmethod
    def create_player_archetype(df: pd.DataFrame) -> pd.Series:
        """Create player archetype based on behavioral patterns"""
        conditions = [
            (df.get('total_spent', 0) > 100) & (df.get('total_playtime_hours', 0) > 500),  # Whale
            (df.get('friends_count', 0) > 20) & (df.get('forum_posts', 0) > 50),  # Social
            (df.get('achievements_unlocked', 0) > 100) & (df.get('total_playtime_hours', 0) > 200),  # Achiever
            (df.get('games_owned', 0) > 50) & (df.get('total_playtime_hours', 0) < 100),  # Explorer
            (df.get('total_playtime_hours', 0) > 1000),  # Hardcore
        ]
        
        choices = ['Whale', 'Social', 'Achiever', 'Explorer', 'Hardcore']
        
        return pd.Series(np.select(conditions, choices, default='Casual'), index=df.index)

class RiskFeatureCreator:
    """Specialized class for creating churn risk features"""
    
    @staticmethod
    def create_dynamic_risk_score(df: pd.DataFrame) -> pd.Series:
        """Create dynamic risk score that adapts to player behavior"""
        base_risk = 0.1
        
        # Activity-based risk
        activity_risk = np.where(df.get('last_login_days_ago', 0) > 30, 0.4, 0)
        activity_risk += np.where(df.get('avg_session_duration', 0) < 15, 0.3, 0)
        
        # Engagement-based risk
        engagement_risk = np.where(df.get('sessions_last_week', 0) == 0, 0.2, 0)
        engagement_risk += np.where(df.get('achievements_unlocked', 0) == 0, 0.1, 0)
        
        # Social-based risk
        social_risk = np.where(df.get('friends_count', 0) == 0, 0.15, 0)
        
        # Economic-based risk
        economic_risk = np.where(df.get('total_spent', 0) == 0, 0.2, 0)
        
        # Protective factors (reduce risk)
        protective_factors = 0
        protective_factors += np.where(df.get('total_spent', 0) > 50, -0.1, 0)
        protective_factors += np.where(df.get('friends_count', 0) > 10, -0.1, 0)
        protective_factors += np.where(df.get('days_since_registration', 0) > 365, -0.05, 0)
        
        total_risk = base_risk + activity_risk + engagement_risk + social_risk + economic_risk + protective_factors
        return pd.Series(np.clip(total_risk, 0, 1), index=df.index)

def main():
    """Example usage of feature engineering"""
    # This would typically be called from the main analysis pipeline
    from ..data.data_collector import SyntheticDataGenerator
    
    # Generate sample data
    generator = SyntheticDataGenerator()
    sample_data = generator.generate_player_data(1000)
    
    # Apply feature engineering
    engineer = FeatureEngineer(sample_data)
    engineered_data = engineer.create_all_features()
    
    print(f"Original features: {len(sample_data.columns)}")
    print(f"Engineered features: {len(engineered_data.columns)}")
    print(f"New features created: {len(engineer.engineered_features)}")
    
    feature_groups = engineer.get_feature_importance_groups()
    for group, features in feature_groups.items():
        print(f"{group.title()}: {len(features)} features")

if __name__ == "__main__":
    main()