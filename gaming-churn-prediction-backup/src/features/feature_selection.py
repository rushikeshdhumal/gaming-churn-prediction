"""
Feature Selection Module for Gaming Player Behavior Analysis

This module provides advanced feature selection techniques including
importance-based, correlation-based, and statistical selection methods.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, RFECV, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelector:
    """
    Main feature selection orchestrator that combines multiple selection strategies
    
    This class coordinates different feature selection methods and provides
    a unified interface for feature selection in the gaming analytics pipeline.
    """
    
    def __init__(self, target_features: int = 50, selection_strategy: str = 'ensemble'):
        """
        Initialize the feature selector
        
        Args:
            target_features: Number of features to select
            selection_strategy: Strategy to use ('ensemble', 'importance', 'correlation', 'statistical')
        """
        self.target_features = target_features
        self.selection_strategy = selection_strategy
        self.selected_features = []
        self.feature_scores = {}
        self.selection_history = {}
        
        # Initialize selectors
        self.importance_selector = ImportanceBasedSelector()
        self.correlation_selector = CorrelationBasedSelector()
        self.statistical_selector = StatisticalSelector()
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       feature_groups: Dict[str, List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using the specified strategy
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_groups: Optional grouping of features by category
            
        Returns:
            Tuple of (selected features dataframe, list of selected feature names)
        """
        logger.info(f"Starting feature selection with strategy: {self.selection_strategy}")
        logger.info(f"Input shape: {X.shape}, Target features: {self.target_features}")
        
        if self.selection_strategy == 'ensemble':
            return self._ensemble_selection(X, y, feature_groups)
        elif self.selection_strategy == 'importance':
            return self.importance_selector.select_features(X, y, self.target_features)
        elif self.selection_strategy == 'correlation':
            return self.correlation_selector.select_features(X, y, self.target_features)
        elif self.selection_strategy == 'statistical':
            return self.statistical_selector.select_features(X, y, self.target_features)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
    
    def _ensemble_selection(self, X: pd.DataFrame, y: pd.Series, 
                           feature_groups: Dict[str, List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Combine multiple selection methods for robust feature selection"""
        
        # Get selections from each method
        _, importance_features = self.importance_selector.select_features(X, y, self.target_features)
        _, correlation_features = self.correlation_selector.select_features(X, y, self.target_features)
        _, statistical_features = self.statistical_selector.select_features(X, y, self.target_features)
        
        # Score features based on how many methods selected them
        feature_votes = {}
        for feature in X.columns:
            votes = 0
            if feature in importance_features:
                votes += 1
            if feature in correlation_features:
                votes += 1
            if feature in statistical_features:
                votes += 1
            feature_votes[feature] = votes
        
        # Ensure representation from each feature group if provided
        selected_features = []
        
        if feature_groups:
            # Select top features from each group
            features_per_group = max(1, self.target_features // len(feature_groups))
            remaining_features = self.target_features
            
            for group_name, group_features in feature_groups.items():
                group_scores = {f: feature_votes.get(f, 0) for f in group_features if f in X.columns}
                group_selected = sorted(group_scores.keys(), key=lambda x: group_scores[x], reverse=True)
                
                num_to_select = min(features_per_group, len(group_selected), remaining_features)
                selected_features.extend(group_selected[:num_to_select])
                remaining_features -= num_to_select
                
                if remaining_features <= 0:
                    break
        
        # Fill remaining slots with highest-voted features
        if len(selected_features) < self.target_features:
            remaining_votes = {f: v for f, v in feature_votes.items() if f not in selected_features}
            additional_features = sorted(remaining_votes.keys(), 
                                       key=lambda x: remaining_votes[x], reverse=True)
            
            needed = self.target_features - len(selected_features)
            selected_features.extend(additional_features[:needed])
        
        # Store selection results
        self.selected_features = selected_features[:self.target_features]
        self.feature_scores = feature_votes
        
        logger.info(f"Ensemble selection completed. Selected {len(self.selected_features)} features")
        
        return X[self.selected_features], self.selected_features
    
    def get_feature_rankings(self) -> pd.DataFrame:
        """Return detailed feature rankings from all methods"""
        
        rankings_data = []
        
        for feature, score in self.feature_scores.items():
            rankings_data.append({
                'feature': feature,
                'ensemble_votes': score,
                'importance_score': getattr(self.importance_selector, 'feature_scores', {}).get(feature, 0),
                'correlation_score': getattr(self.correlation_selector, 'feature_scores', {}).get(feature, 0),
                'statistical_score': getattr(self.statistical_selector, 'feature_scores', {}).get(feature, 0),
                'selected': feature in self.selected_features
            })
        
        rankings_df = pd.DataFrame(rankings_data)
        return rankings_df.sort_values('ensemble_votes', ascending=False)
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """Plot feature importance scores"""
        try:
            import matplotlib.pyplot as plt
            
            rankings = self.get_feature_rankings().head(top_n)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(rankings)), rankings['ensemble_votes'])
            plt.yticks(range(len(rankings)), rankings['feature'])
            plt.xlabel('Ensemble Votes')
            plt.title(f'Top {top_n} Features by Ensemble Selection')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


class ImportanceBasedSelector:
    """
    Feature selection based on model-based importance scores
    
    Uses multiple tree-based models to assess feature importance
    and selects features based on aggregated importance scores.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize importance-based selector"""
        self.random_state = random_state
        self.feature_scores = {}
        self.importance_methods = {}
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       n_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Select features based on importance scores"""
        
        logger.info("Running importance-based feature selection...")
        
        # Multiple importance-based methods
        methods = {
            'random_forest': self._random_forest_importance,
            'extra_trees': self._extra_trees_importance,
            'lasso': self._lasso_importance,
            'recursive_elimination': self._recursive_elimination
        }
        
        importance_scores = {}
        
        for method_name, method_func in methods.items():
            try:
                scores = method_func(X, y)
                importance_scores[method_name] = scores
                logger.info(f"Completed {method_name} importance calculation")
            except Exception as e:
                logger.warning(f"Failed to calculate {method_name} importance: {e}")
        
        # Aggregate scores across methods
        aggregated_scores = self._aggregate_importance_scores(importance_scores, X.columns)
        
        # Select top features
        selected_features = sorted(aggregated_scores.keys(), 
                                 key=lambda x: aggregated_scores[x], reverse=True)[:n_features]
        
        self.feature_scores = aggregated_scores
        self.importance_methods = importance_scores
        
        logger.info(f"Selected {len(selected_features)} features based on importance")
        
        return X[selected_features], selected_features
    
    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate Random Forest feature importance"""
        
        # Handle mixed data types
        X_numeric = self._prepare_numeric_data(X)
        
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_numeric, y)
        
        return dict(zip(X_numeric.columns, rf.feature_importances_))
    
    def _extra_trees_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate Extra Trees feature importance"""
        
        X_numeric = self._prepare_numeric_data(X)
        
        et = ExtraTreesClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            n_jobs=-1
        )
        et.fit(X_numeric, y)
        
        return dict(zip(X_numeric.columns, et.feature_importances_))
    
    def _lasso_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate LASSO-based feature importance"""
        
        X_numeric = self._prepare_numeric_data(X)
        
        # Scale features for LASSO
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        
        lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=1000)
        lasso.fit(X_scaled, y)
        
        # Use absolute coefficients as importance
        importance = np.abs(lasso.coef_)
        
        return dict(zip(X_numeric.columns, importance))
    
    def _recursive_elimination(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate importance using Recursive Feature Elimination"""
        
        X_numeric = self._prepare_numeric_data(X)
        
        # Limit features for RFE to avoid excessive computation
        if len(X_numeric.columns) > 100:
            # Pre-select top 100 features using Random Forest
            rf_temp = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            rf_temp.fit(X_numeric, y)
            top_indices = np.argsort(rf_temp.feature_importances_)[-100:]
            X_numeric = X_numeric.iloc[:, top_indices]
        
        estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        
        # RFE to select half of the features
        n_features_to_select = max(10, len(X_numeric.columns) // 2)
        rfe = RFE(estimator, n_features_to_select=n_features_to_select)
        rfe.fit(X_numeric, y)
        
        # Convert ranking to importance (lower rank = higher importance)
        max_rank = max(rfe.ranking_)
        importance = [(max_rank - rank + 1) / max_rank for rank in rfe.ranking_]
        
        return dict(zip(X_numeric.columns, importance))
    
    def _prepare_numeric_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for importance calculation by handling non-numeric columns"""
        
        X_numeric = X.copy()
        
        # Convert categorical columns to numeric
        for col in X_numeric.columns:
            if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
                # Use label encoding for categorical variables
                X_numeric[col] = pd.Categorical(X_numeric[col]).codes
        
        # Handle missing values
        X_numeric = X_numeric.fillna(0)
        
        return X_numeric
    
    def _aggregate_importance_scores(self, importance_scores: Dict[str, Dict[str, float]], 
                                   all_features: List[str]) -> Dict[str, float]:
        """Aggregate importance scores across different methods"""
        
        aggregated = {}
        
        for feature in all_features:
            scores = []
            for method_scores in importance_scores.values():
                if feature in method_scores:
                    scores.append(method_scores[feature])
            
            # Use median score to reduce impact of outliers
            if scores:
                aggregated[feature] = np.median(scores)
            else:
                aggregated[feature] = 0.0
        
        return aggregated


class CorrelationBasedSelector:
    """
    Feature selection based on correlation analysis
    
    Removes highly correlated features and selects features with
    strong correlation to target while maintaining diversity.
    """
    
    def __init__(self, correlation_threshold: float = 0.95, 
                 target_correlation_threshold: float = 0.1):
        """
        Initialize correlation-based selector
        
        Args:
            correlation_threshold: Threshold for removing highly correlated features
            target_correlation_threshold: Minimum correlation with target
        """
        self.correlation_threshold = correlation_threshold
        self.target_correlation_threshold = target_correlation_threshold
        self.feature_scores = {}
        self.correlation_matrix = None
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       n_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Select features based on correlation analysis"""
        
        logger.info("Running correlation-based feature selection...")
        
        # Prepare numeric data
        X_numeric = self._prepare_numeric_data(X)
        
        # Calculate correlation with target
        target_correlations = self._calculate_target_correlations(X_numeric, y)
        
        # Remove features with low target correlation
        relevant_features = [
            feature for feature, corr in target_correlations.items()
            if abs(corr) >= self.target_correlation_threshold
        ]
        
        logger.info(f"Features with sufficient target correlation: {len(relevant_features)}")
        
        if len(relevant_features) == 0:
            logger.warning("No features meet target correlation threshold. Using all features.")
            relevant_features = list(X_numeric.columns)
        
        X_relevant = X_numeric[relevant_features]
        
        # Remove highly correlated features
        selected_features = self._remove_correlated_features(
            X_relevant, 
            {f: target_correlations[f] for f in relevant_features}
        )
        
        # Select top features if we have more than needed
        if len(selected_features) > n_features:
            # Sort by absolute target correlation
            selected_features = sorted(
                selected_features,
                key=lambda x: abs(target_correlations[x]),
                reverse=True
            )[:n_features]
        
        self.feature_scores = target_correlations
        
        logger.info(f"Selected {len(selected_features)} features after correlation analysis")
        
        return X[selected_features], selected_features
    
    def _calculate_target_correlations(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate correlation between features and target"""
        
        correlations = {}
        
        for column in X.columns:
            try:
                # Use Spearman correlation for robustness
                corr, _ = spearmanr(X[column], y)
                correlations[column] = corr if not np.isnan(corr) else 0.0
            except Exception as e:
                logger.warning(f"Failed to calculate correlation for {column}: {e}")
                correlations[column] = 0.0
        
        return correlations
    
    def _remove_correlated_features(self, X: pd.DataFrame, 
                                   target_correlations: Dict[str, float]) -> List[str]:
        """Remove highly correlated features, keeping those with higher target correlation"""
        
        # Calculate feature-feature correlation matrix
        self.correlation_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        correlated_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                if self.correlation_matrix.iloc[i, j] >= self.correlation_threshold:
                    feat1 = self.correlation_matrix.columns[i]
                    feat2 = self.correlation_matrix.columns[j]
                    correlated_pairs.append((feat1, feat2, self.correlation_matrix.iloc[i, j]))
        
        logger.info(f"Found {len(correlated_pairs)} highly correlated feature pairs")
        
        # Remove features from correlated pairs (keep one with higher target correlation)
        features_to_remove = set()
        for feat1, feat2, corr_val in correlated_pairs:
            target_corr1 = abs(target_correlations.get(feat1, 0))
            target_corr2 = abs(target_correlations.get(feat2, 0))
            
            if target_corr1 >= target_corr2:
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
        
        selected_features = [f for f in X.columns if f not in features_to_remove]
        
        logger.info(f"Removed {len(features_to_remove)} highly correlated features")
        
        return selected_features
    
    def _prepare_numeric_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for correlation analysis"""
        
        X_numeric = X.copy()
        
        # Convert categorical columns to numeric
        for col in X_numeric.columns:
            if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
                X_numeric[col] = pd.Categorical(X_numeric[col]).codes
        
        # Handle missing values
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        return X_numeric


class StatisticalSelector:
    """
    Feature selection based on statistical tests
    
    Uses various statistical tests to identify features with
    significant relationships to the target variable.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical selector
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        self.feature_scores = {}
        self.test_results = {}
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       n_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Select features based on statistical significance"""
        
        logger.info("Running statistical feature selection...")
        
        # Prepare data
        X_numeric = self._prepare_numeric_data(X)
        
        # Run different statistical tests
        statistical_scores = {}
        
        # ANOVA F-test
        try:
            f_scores = self._anova_f_test(X_numeric, y)
            statistical_scores['anova'] = f_scores
            logger.info("Completed ANOVA F-test")
        except Exception as e:
            logger.warning(f"ANOVA F-test failed: {e}")
        
        # Mutual Information
        try:
            mi_scores = self._mutual_information_test(X_numeric, y)
            statistical_scores['mutual_info'] = mi_scores
            logger.info("Completed Mutual Information test")
        except Exception as e:
            logger.warning(f"Mutual Information test failed: {e}")
        
        # Chi-square test (for non-negative features)
        try:
            chi2_scores = self._chi_square_test(X_numeric, y)
            statistical_scores['chi2'] = chi2_scores
            logger.info("Completed Chi-square test")
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
        
        # Aggregate statistical scores
        aggregated_scores = self._aggregate_statistical_scores(statistical_scores, X_numeric.columns)
        
        # Select top features
        selected_features = sorted(aggregated_scores.keys(), 
                                 key=lambda x: aggregated_scores[x], reverse=True)[:n_features]
        
        self.feature_scores = aggregated_scores
        self.test_results = statistical_scores
        
        logger.info(f"Selected {len(selected_features)} features based on statistical tests")
        
        return X[selected_features], selected_features
    
    def _anova_f_test(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform ANOVA F-test for feature selection"""
        
        selector = SelectKBest(f_classif, k='all')
        selector.fit(X, y)
        
        return dict(zip(X.columns, selector.scores_))
    
    def _mutual_information_test(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform Mutual Information test for feature selection"""
        
        selector = SelectKBest(mutual_info_classif, k='all')
        selector.fit(X, y)
        
        return dict(zip(X.columns, selector.scores_))
    
    def _chi_square_test(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform Chi-square test for feature selection"""
        
        # Make features non-negative for chi-square test
        X_nonneg = X.copy()
        
        # Min-max scale to make non-negative
        for col in X_nonneg.columns:
            col_min = X_nonneg[col].min()
            if col_min < 0:
                X_nonneg[col] = X_nonneg[col] - col_min
        
        # Discretize continuous variables for chi-square
        for col in X_nonneg.columns:
            if X_nonneg[col].nunique() > 20:  # If too many unique values
                X_nonneg[col] = pd.qcut(X_nonneg[col], q=5, duplicates='drop', labels=False)
        
        selector = SelectKBest(chi2, k='all')
        selector.fit(X_nonneg, y)
        
        return dict(zip(X.columns, selector.scores_))
    
    def _aggregate_statistical_scores(self, statistical_scores: Dict[str, Dict[str, float]], 
                                    all_features: List[str]) -> Dict[str, float]:
        """Aggregate scores from different statistical tests"""
        
        aggregated = {}
        
        for feature in all_features:
            scores = []
            for test_name, test_scores in statistical_scores.items():
                if feature in test_scores and not np.isnan(test_scores[feature]):
                    # Normalize scores by test maximum to make them comparable
                    max_score = max(test_scores.values())
                    if max_score > 0:
                        normalized_score = test_scores[feature] / max_score
                        scores.append(normalized_score)
            
            # Use mean of normalized scores
            if scores:
                aggregated[feature] = np.mean(scores)
            else:
                aggregated[feature] = 0.0
        
        return aggregated
    
    def _prepare_numeric_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for statistical tests"""
        
        X_numeric = X.copy()
        
        # Convert categorical columns to numeric
        for col in X_numeric.columns:
            if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
                X_numeric[col] = pd.Categorical(X_numeric[col]).codes
        
        # Handle missing values
        X_numeric = X_numeric.fillna(0)
        
        # Handle infinite values
        X_numeric = X_numeric.replace([np.inf, -np.inf], 0)
        
        return X_numeric
    
    def get_test_summary(self) -> pd.DataFrame:
        """Return summary of statistical test results"""
        
        summary_data = []
        
        for feature in self.feature_scores.keys():
            row = {'feature': feature, 'overall_score': self.feature_scores[feature]}
            
            for test_name, test_scores in self.test_results.items():
                if feature in test_scores:
                    row[f'{test_name}_score'] = test_scores[feature]
                else:
                    row[f'{test_name}_score'] = 0.0
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df.sort_values('overall_score', ascending=False)


def main():
    """Example usage of feature selection classes"""
    
    # This would typically be called from the main analysis pipeline
    from ..data.data_collector import SyntheticDataGenerator
    from .feature_engineering import FeatureEngineer
    
    # Generate sample data
    generator = SyntheticDataGenerator()
    sample_data = generator.generate_player_data(1000)
    
    # Apply feature engineering
    engineer = FeatureEngineer(sample_data)
    engineered_data = engineer.create_all_features()
    
    # Prepare features and target
    X = engineered_data.drop(['churned', 'player_id'], axis=1, errors='ignore')
    y = engineered_data['churned']
    
    print(f"Original features: {X.shape[1]}")
    
    # Test different selection strategies
    strategies = ['importance', 'correlation', 'statistical', 'ensemble']
    
    for strategy in strategies:
        selector = FeatureSelector(target_features=20, selection_strategy=strategy)
        X_selected, selected_features = selector.select_features(X, y)
        
        print(f"{strategy.title()} Selection: {len(selected_features)} features")
        print(f"Top 5 features: {selected_features[:5]}")
        print()

if __name__ == "__main__":
    main()