"""
Feature Transformation Module for Gaming Player Behavior Analysis

This module provides comprehensive feature transformation capabilities including
numeric scaling, categorical encoding, and datetime feature extraction.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
    PowerTransformer, QuantileTransformer, LabelEncoder, 
    OneHotEncoder, OrdinalEncoder, TargetEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureTransformer:
    """
    Main feature transformation orchestrator
    
    Coordinates different transformation methods and provides
    a unified interface for feature transformation in the gaming analytics pipeline.
    """
    
    def __init__(self, transformation_strategy: str = 'comprehensive'):
        """
        Initialize the feature transformer
        
        Args:
            transformation_strategy: Strategy to use ('basic', 'comprehensive', 'custom')
        """
        self.transformation_strategy = transformation_strategy
        self.transformers = {}
        self.transformation_history = {}
        self.fitted_transformers = {}
        
        # Initialize specialized transformers
        self.numeric_transformer = NumericTransformer()
        self.categorical_transformer = CategoricalTransformer()
        self.datetime_transformer = DateTimeTransformer()
        
    def fit_transform(self, df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """
        Fit transformers and transform the data
        
        Args:
            df: Input dataframe
            target: Target variable for supervised transformations
            
        Returns:
            Transformed dataframe
        """
        logger.info(f"Starting feature transformation with strategy: {self.transformation_strategy}")
        logger.info(f"Input shape: {df.shape}")
        
        transformed_df = df.copy()
        
        # Identify column types
        column_types = self._identify_column_types(transformed_df)
        
        # Apply transformations based on strategy
        if self.transformation_strategy == 'comprehensive':
            transformed_df = self._comprehensive_transformation(transformed_df, target, column_types)
        elif self.transformation_strategy == 'basic':
            transformed_df = self._basic_transformation(transformed_df, target, column_types)
        elif self.transformation_strategy == 'custom':
            transformed_df = self._custom_transformation(transformed_df, target, column_types)
        
        logger.info(f"Transformation completed. Output shape: {transformed_df.shape}")
        
        return transformed_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers"""
        
        if not self.fitted_transformers:
            raise ValueError("Transformers must be fitted before transforming new data")
        
        transformed_df = df.copy()
        
        # Apply fitted transformations
        for transformer_name, transformer in self.fitted_transformers.items():
            if hasattr(transformer, 'transform'):
                try:
                    transformed_columns = transformer.get_feature_names_out() if hasattr(transformer, 'get_feature_names_out') else None
                    transformed_data = transformer.transform(transformed_df)
                    
                    if transformed_columns is not None:
                        transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns, index=transformed_df.index)
                    else:
                        transformed_df = pd.DataFrame(transformed_data, index=transformed_df.index)
                        
                except Exception as e:
                    logger.warning(f"Failed to apply {transformer_name}: {e}")
        
        return transformed_df
    
    def _identify_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify different types of columns in the dataframe"""
        
        column_types = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'boolean': [],
            'text': []
        }
        
        for column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                column_types['datetime'].append(column)
            elif pd.api.types.is_bool_dtype(df[column]):
                column_types['boolean'].append(column)
            elif pd.api.types.is_numeric_dtype(df[column]):
                column_types['numeric'].append(column)
            elif df[column].dtype == 'object':
                # Check if it's a date string
                if self._is_date_column(df[column]):
                    column_types['datetime'].append(column)
                else:
                    unique_ratio = df[column].nunique() / len(df)
                    if unique_ratio < 0.1:  # Low cardinality
                        column_types['categorical'].append(column)
                    else:
                        column_types['text'].append(column)
            else:
                column_types['categorical'].append(column)
        
        logger.info(f"Column types identified: {[(k, len(v)) for k, v in column_types.items()]}")
        return column_types
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a string column contains date values"""
        try:
            # Try to parse a sample of non-null values
            sample = series.dropna().head(10)
            parsed_count = 0
            
            for value in sample:
                try:
                    pd.to_datetime(value)
                    parsed_count += 1
                except:
                    pass
            
            return parsed_count / len(sample) > 0.7 if len(sample) > 0 else False
        except:
            return False
    
    def _comprehensive_transformation(self, df: pd.DataFrame, target: pd.Series, 
                                    column_types: Dict[str, List[str]]) -> pd.DataFrame:
        """Apply comprehensive transformation pipeline"""
        
        transformed_df = df.copy()
        
        # Transform datetime columns
        if column_types['datetime']:
            logger.info(f"Transforming {len(column_types['datetime'])} datetime columns")
            datetime_features = self.datetime_transformer.fit_transform(
                transformed_df[column_types['datetime']]
            )
            transformed_df = pd.concat([transformed_df.drop(columns=column_types['datetime']), 
                                      datetime_features], axis=1)
            self.fitted_transformers['datetime'] = self.datetime_transformer
        
        # Transform categorical columns
        if column_types['categorical']:
            logger.info(f"Transforming {len(column_types['categorical'])} categorical columns")
            categorical_features = self.categorical_transformer.fit_transform(
                transformed_df[column_types['categorical']], target
            )
            transformed_df = pd.concat([transformed_df.drop(columns=column_types['categorical']), 
                                      categorical_features], axis=1)
            self.fitted_transformers['categorical'] = self.categorical_transformer
        
        # Transform numeric columns
        numeric_columns = column_types['numeric'] + column_types['boolean']
        if numeric_columns:
            logger.info(f"Transforming {len(numeric_columns)} numeric columns")
            numeric_features = self.numeric_transformer.fit_transform(
                transformed_df[numeric_columns]
            )
            transformed_df = pd.concat([transformed_df.drop(columns=numeric_columns), 
                                      numeric_features], axis=1)
            self.fitted_transformers['numeric'] = self.numeric_transformer
        
        return transformed_df
    
    def _basic_transformation(self, df: pd.DataFrame, target: pd.Series, 
                            column_types: Dict[str, List[str]]) -> pd.DataFrame:
        """Apply basic transformation pipeline"""
        
        transformed_df = df.copy()
        
        # Basic numeric transformation (standardization)
        numeric_columns = column_types['numeric'] + column_types['boolean']
        if numeric_columns:
            scaler = StandardScaler()
            transformed_df[numeric_columns] = scaler.fit_transform(transformed_df[numeric_columns])
            self.fitted_transformers['basic_scaler'] = scaler
        
        # Basic categorical encoding (label encoding)
        if column_types['categorical']:
            for col in column_types['categorical']:
                le = LabelEncoder()
                transformed_df[col] = le.fit_transform(transformed_df[col].astype(str).fillna('unknown'))
                self.fitted_transformers[f'label_encoder_{col}'] = le
        
        return transformed_df
    
    def _custom_transformation(self, df: pd.DataFrame, target: pd.Series, 
                             column_types: Dict[str, List[str]]) -> pd.DataFrame:
        """Apply custom transformation pipeline for gaming-specific features"""
        
        transformed_df = df.copy()
        
        # Gaming-specific transformations
        gaming_transformations = {
            'playtime_log': lambda x: np.log1p(x) if 'playtime' in x.name.lower() else x,
            'spending_sqrt': lambda x: np.sqrt(x) if 'spent' in x.name.lower() or 'purchase' in x.name.lower() else x,
            'session_standardize': lambda x: (x - x.mean()) / x.std() if 'session' in x.name.lower() else x,
            'risk_sigmoid': lambda x: 1 / (1 + np.exp(-x)) if 'risk' in x.name.lower() else x
        }
        
        for col in transformed_df.select_dtypes(include=[np.number]).columns:
            for transform_name, transform_func in gaming_transformations.items():
                try:
                    transformed_df[col] = transform_func(transformed_df[col])
                    break  # Apply only the first matching transformation
                except:
                    continue
        
        return transformed_df
    
    def get_transformation_report(self) -> Dict[str, Any]:
        """Generate a report of applied transformations"""
        
        report = {
            'strategy': self.transformation_strategy,
            'transformers_fitted': list(self.fitted_transformers.keys()),
            'transformation_history': self.transformation_history
        }
        
        return report


class NumericTransformer:
    """
    Specialized transformer for numeric features
    
    Handles scaling, normalization, and distribution corrections
    for numeric features in gaming analytics.
    """
    
    def __init__(self, scaling_method: str = 'robust', handle_outliers: bool = True):
        """
        Initialize numeric transformer
        
        Args:
            scaling_method: Scaling method ('standard', 'robust', 'minmax', 'quantile')
            handle_outliers: Whether to handle outliers
        """
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.scalers = {}
        self.outlier_bounds = {}
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit transformers and transform numeric data"""
        
        logger.info(f"Applying numeric transformation with {self.scaling_method} scaling")
        
        transformed_df = df.copy()
        
        # Handle missing values
        transformed_df = self._handle_missing_values(transformed_df)
        
        # Handle outliers if requested
        if self.handle_outliers:
            transformed_df = self._handle_outliers(transformed_df)
        
        # Apply distribution corrections
        transformed_df = self._correct_distributions(transformed_df)
        
        # Apply scaling
        transformed_df = self._apply_scaling(transformed_df)
        
        return transformed_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in numeric columns"""
        
        for column in df.columns:
            if df[column].isnull().any():
                # Use median for robust imputation
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
                logger.info(f"Imputed {column} with median: {median_value:.2f}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        
        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.outlier_bounds[column] = (lower_bound, upper_bound)
            
            # Cap outliers
            outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            if outliers_count > 0:
                df[column] = np.clip(df[column], lower_bound, upper_bound)
                logger.info(f"Capped {outliers_count} outliers in {column}")
        
        return df
    
    def _correct_distributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply distribution corrections for skewed features"""
        
        corrected_df = df.copy()
        
        for column in df.columns:
            # Check skewness
            skewness = df[column].skew()
            
            if abs(skewness) > 1.5:  # Highly skewed
                if df[column].min() >= 0:  # Non-negative values
                    if 'playtime' in column.lower() or 'hours' in column.lower():
                        # Log transformation for playtime features
                        corrected_df[column] = np.log1p(df[column])
                        logger.info(f"Applied log1p transformation to {column} (skewness: {skewness:.2f})")
                    elif 'spent' in column.lower() or 'price' in column.lower():
                        # Square root transformation for monetary features
                        corrected_df[column] = np.sqrt(df[column])
                        logger.info(f"Applied sqrt transformation to {column} (skewness: {skewness:.2f})")
                    else:
                        # Box-Cox transformation for other positive features
                        try:
                            transformer = PowerTransformer(method='box-cox')
                            corrected_df[column] = transformer.fit_transform(df[[column]]).flatten()
                            logger.info(f"Applied Box-Cox transformation to {column} (skewness: {skewness:.2f})")
                        except:
                            # Fall back to Yeo-Johnson if Box-Cox fails
                            transformer = PowerTransformer(method='yeo-johnson')
                            corrected_df[column] = transformer.fit_transform(df[[column]]).flatten()
                            logger.info(f"Applied Yeo-Johnson transformation to {column} (skewness: {skewness:.2f})")
        
        return corrected_df
    
    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the specified scaling method"""
        
        if self.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.scaling_method == 'robust':
            scaler = RobustScaler()
        elif self.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.scaling_method == 'quantile':
            scaler = QuantileTransformer(output_distribution='uniform')
        else:
            logger.warning(f"Unknown scaling method: {self.scaling_method}. Using robust scaling.")
            scaler = RobustScaler()
        
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
        
        self.scalers['main_scaler'] = scaler
        
        return scaled_df


class CategoricalTransformer:
    """
    Specialized transformer for categorical features
    
    Handles encoding, rare category grouping, and target encoding
    for categorical features in gaming analytics.
    """
    
    def __init__(self, encoding_method: str = 'target', rare_threshold: float = 0.01):
        """
        Initialize categorical transformer
        
        Args:
            encoding_method: Encoding method ('onehot', 'label', 'target', 'ordinal')
            rare_threshold: Threshold for grouping rare categories
        """
        self.encoding_method = encoding_method
        self.rare_threshold = rare_threshold
        self.encoders = {}
        self.rare_categories = {}
        
    def fit_transform(self, df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """Fit transformers and transform categorical data"""
        
        logger.info(f"Applying categorical transformation with {self.encoding_method} encoding")
        
        transformed_df = df.copy()
        
        # Handle missing values
        transformed_df = self._handle_missing_values(transformed_df)
        
        # Group rare categories
        transformed_df = self._group_rare_categories(transformed_df)
        
        # Apply encoding
        transformed_df = self._apply_encoding(transformed_df, target)
        
        return transformed_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in categorical columns"""
        
        for column in df.columns:
            if df[column].isnull().any():
                df[column] = df[column].fillna('unknown')
                logger.info(f"Filled missing values in {column} with 'unknown'")
        
        return df
    
    def _group_rare_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group rare categories to reduce cardinality"""
        
        for column in df.columns:
            value_counts = df[column].value_counts(normalize=True)
            rare_categories = value_counts[value_counts < self.rare_threshold].index.tolist()
            
            if rare_categories:
                self.rare_categories[column] = rare_categories
                df[column] = df[column].replace(rare_categories, 'rare_category')
                logger.info(f"Grouped {len(rare_categories)} rare categories in {column}")
        
        return df
    
    def _apply_encoding(self, df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """Apply the specified encoding method"""
        
        if self.encoding_method == 'onehot':
            return self._onehot_encoding(df)
        elif self.encoding_method == 'label':
            return self._label_encoding(df)
        elif self.encoding_method == 'target':
            return self._target_encoding(df, target)
        elif self.encoding_method == 'ordinal':
            return self._ordinal_encoding(df)
        else:
            logger.warning(f"Unknown encoding method: {self.encoding_method}. Using label encoding.")
            return self._label_encoding(df)
    
    def _onehot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding"""
        
        encoded_dfs = []
        
        for column in df.columns:
            # Limit categories to prevent explosion
            top_categories = df[column].value_counts().head(10).index.tolist()
            df_temp = df[column].apply(lambda x: x if x in top_categories else 'other')
            
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(df_temp.values.reshape(-1, 1))
            
            feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
            
            encoded_dfs.append(encoded_df)
            self.encoders[column] = encoder
        
        return pd.concat(encoded_dfs, axis=1)
    
    def _label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding"""
        
        encoded_df = df.copy()
        
        for column in df.columns:
            encoder = LabelEncoder()
            encoded_df[column] = encoder.fit_transform(df[column].astype(str))
            self.encoders[column] = encoder
        
        return encoded_df
    
    def _target_encoding(self, df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """Apply target encoding (requires target variable)"""
        
        if target is None:
            logger.warning("Target encoding requires target variable. Falling back to label encoding.")
            return self._label_encoding(df)
        
        encoded_df = df.copy()
        
        for column in df.columns:
            # Calculate target mean for each category
            target_means = df.groupby(column)[target.name if hasattr(target, 'name') else 'target'].mean()
            
            # Apply smoothing to handle categories with few samples
            global_mean = target.mean()
            category_counts = df[column].value_counts()
            
            smoothed_means = {}
            for category in target_means.index:
                count = category_counts[category]
                weight = count / (count + 10)  # Smoothing parameter
                smoothed_mean = weight * target_means[category] + (1 - weight) * global_mean
                smoothed_means[category] = smoothed_mean
            
            encoded_df[column] = df[column].map(smoothed_means)
            self.encoders[column] = smoothed_means
        
        return encoded_df
    
    def _ordinal_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ordinal encoding with gaming-specific ordering"""
        
        encoded_df = df.copy()
        
        # Gaming-specific orderings
        gaming_orderings = {
            'age_group': ['18-25', '26-35', '36-45', '46+'],
            'spending_category': ['Free', 'Light', 'Medium', 'Heavy', 'Whale'],
            'account_age_category': ['New', 'Recent', 'Established', 'Veteran', 'Legacy'],
            'activity_category': ['Inactive', 'Recent', 'This_Month', 'This_Week', 'Today']
        }
        
        for column in df.columns:
            if column in gaming_orderings:
                # Use predefined ordering
                ordering = gaming_orderings[column]
                category_map = {cat: i for i, cat in enumerate(ordering)}
                encoded_df[column] = df[column].map(category_map).fillna(-1)
                self.encoders[column] = category_map
            else:
                # Use frequency-based ordering
                value_counts = df[column].value_counts()
                frequency_map = {cat: i for i, cat in enumerate(value_counts.index)}
                encoded_df[column] = df[column].map(frequency_map)
                self.encoders[column] = frequency_map
        
        return encoded_df


class DateTimeTransformer:
    """
    Specialized transformer for datetime features
    
    Extracts temporal features and patterns relevant to gaming analytics.
    """
    
    def __init__(self, extract_features: List[str] = None):
        """
        Initialize datetime transformer
        
        Args:
            extract_features: List of features to extract ('year', 'month', 'day', etc.)
        """
        self.extract_features = extract_features or [
            'year', 'month', 'day', 'dayofweek', 'hour', 
            'quarter', 'is_weekend', 'is_holiday', 'days_since'
        ]
        self.reference_date = datetime.now()
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform datetime columns into multiple temporal features"""
        
        logger.info(f"Extracting datetime features: {self.extract_features}")
        
        all_features = []
        
        for column in df.columns:
            # Convert to datetime if not already
            datetime_series = self._convert_to_datetime(df[column])
            
            # Extract features
            column_features = self._extract_datetime_features(datetime_series, column)
            all_features.append(column_features)
        
        return pd.concat(all_features, axis=1)
    
    def _convert_to_datetime(self, series: pd.Series) -> pd.Series:
        """Convert series to datetime format"""
        
        if not pd.api.types.is_datetime64_any_dtype(series):
            try:
                return pd.to_datetime(series)
            except Exception as e:
                logger.warning(f"Failed to convert {series.name} to datetime: {e}")
                return pd.to_datetime(series, errors='coerce')
        
        return series
    
    def _extract_datetime_features(self, datetime_series: pd.Series, column_name: str) -> pd.DataFrame:
        """Extract specified datetime features"""
        
        features = {}
        
        if 'year' in self.extract_features:
            features[f'{column_name}_year'] = datetime_series.dt.year
        
        if 'month' in self.extract_features:
            features[f'{column_name}_month'] = datetime_series.dt.month
        
        if 'day' in self.extract_features:
            features[f'{column_name}_day'] = datetime_series.dt.day
        
        if 'dayofweek' in self.extract_features:
            features[f'{column_name}_dayofweek'] = datetime_series.dt.dayofweek
        
        if 'hour' in self.extract_features:
            features[f'{column_name}_hour'] = datetime_series.dt.hour
        
        if 'quarter' in self.extract_features:
            features[f'{column_name}_quarter'] = datetime_series.dt.quarter
        
        if 'is_weekend' in self.extract_features:
            features[f'{column_name}_is_weekend'] = (datetime_series.dt.dayofweek >= 5).astype(int)
        
        if 'is_holiday' in self.extract_features:
            # Simplified holiday detection (major holidays)
            features[f'{column_name}_is_holiday'] = self._detect_holidays(datetime_series)
        
        if 'days_since' in self.extract_features:
            features[f'{column_name}_days_since'] = (self.reference_date - datetime_series).dt.days
        
        # Gaming-specific features
        features[f'{column_name}_gaming_season'] = self._get_gaming_season(datetime_series)
        features[f'{column_name}_peak_hours'] = self._get_peak_hours(datetime_series)
        
        return pd.DataFrame(features, index=datetime_series.index)
    
    def _detect_holidays(self, datetime_series: pd.Series) -> pd.Series:
        """Detect major holidays that might affect gaming behavior"""
        
        holidays = []
        
        for date in datetime_series:
            if pd.isna(date):
                holidays.append(0)
                continue
                
            is_holiday = 0
            
            # New Year's Day
            if date.month == 1 and date.day == 1:
                is_holiday = 1
            # Christmas
            elif date.month == 12 and date.day == 25:
                is_holiday = 1
            # Thanksgiving (4th Thursday in November for US)
            elif date.month == 11 and date.day >= 22 and date.day <= 28 and date.dayofweek == 3:
                is_holiday = 1
            # Black Friday (day after Thanksgiving)
            elif date.month == 11 and date.day >= 23 and date.day <= 29 and date.dayofweek == 4:
                is_holiday = 1
            
            holidays.append(is_holiday)
        
        return pd.Series(holidays, index=datetime_series.index)
    
    def _get_gaming_season(self, datetime_series: pd.Series) -> pd.Series:
        """Classify dates into gaming seasons"""
        
        seasons = []
        
        for date in datetime_series:
            if pd.isna(date):
                seasons.append(0)
                continue
            
            month = date.month
            
            if month in [11, 12]:  # Holiday gaming season
                season = 3
            elif month in [6, 7, 8]:  # Summer gaming season
                season = 2
            elif month in [1, 2]:  # Post-holiday season
                season = 1
            else:  # Regular season
                season = 0
            
            seasons.append(season)
        
        return pd.Series(seasons, index=datetime_series.index)
    
    def _get_peak_hours(self, datetime_series: pd.Series) -> pd.Series:
        """Identify peak gaming hours"""
        
        peak_hours = []
        
        for date in datetime_series:
            if pd.isna(date):
                peak_hours.append(0)
                continue
            
            hour = date.hour
            
            # Peak gaming hours: evening (7-11 PM) and late night (11 PM - 1 AM)
            if 19 <= hour <= 23 or 0 <= hour <= 1:
                peak_hours.append(1)
            else:
                peak_hours.append(0)
        
        return pd.Series(peak_hours, index=datetime_series.index)


def main():
    """Example usage of feature transformation classes"""
    
    # This would typically be called from the main analysis pipeline
    from ..data.data_collector import SyntheticDataGenerator
    
    # Generate sample data
    generator = SyntheticDataGenerator()
    sample_data = generator.generate_player_data(1000)
    
    print(f"Original data shape: {sample_data.shape}")
    
    # Test comprehensive transformation
    transformer = FeatureTransformer(transformation_strategy='comprehensive')
    transformed_data = transformer.fit_transform(sample_data, sample_data['churned'])
    
    print(f"Transformed data shape: {transformed_data.shape}")
    print(f"Transformation report: {transformer.get_transformation_report()}")
    
    # Test individual transformers
    numeric_cols = sample_data.select_dtypes(include=[np.number]).columns[:5]
    numeric_transformer = NumericTransformer()
    transformed_numeric = numeric_transformer.fit_transform(sample_data[numeric_cols])
    
    print(f"Numeric transformation: {numeric_cols.tolist()} -> {transformed_numeric.shape[1]} features")

if __name__ == "__main__":
    main()