"""
Data Processing Module for Gaming Analytics

This module handles data cleaning, transformation, and validation
for gaming player behavior analysis.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Comprehensive data cleaning for gaming datasets"""
    
    def __init__(self):
        self.cleaning_report = {}
        
    def clean_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean player behavioral data"""
        logger.info(f"Cleaning player data: {df.shape}")
        
        df_cleaned = df.copy()
        initial_rows = len(df_cleaned)
        
        # Track cleaning operations
        self.cleaning_report['player_data'] = {
            'initial_rows': initial_rows,
            'operations': []
        }
        
        # 1. Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned, 'player')
        
        # 2. Remove duplicates
        duplicates_before = df_cleaned.duplicated().sum()
        df_cleaned = df_cleaned.drop_duplicates(subset=['player_id'], keep='first')
        duplicates_removed = duplicates_before - df_cleaned.duplicated().sum()
        
        if duplicates_removed > 0:
            self.cleaning_report['player_data']['operations'].append(
                f"Removed {duplicates_removed} duplicate player records"
            )
        
        # 3. Fix data types
        df_cleaned = self._fix_data_types(df_cleaned)
        
        # 4. Handle outliers
        df_cleaned = self._handle_outliers(df_cleaned)
        
        # 5. Validate business logic
        df_cleaned = self._validate_business_rules(df_cleaned)
        
        # 6. Clean text fields
        df_cleaned = self._clean_text_fields(df_cleaned)
        
        final_rows = len(df_cleaned)
        self.cleaning_report['player_data']['final_rows'] = final_rows
        self.cleaning_report['player_data']['rows_removed'] = initial_rows - final_rows
        
        logger.info(f"Player data cleaning complete: {df_cleaned.shape}")
        return df_cleaned
    
    def clean_game_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean game catalog data"""
        logger.info(f"Cleaning game data: {df.shape}")
        
        df_cleaned = df.copy()
        initial_rows = len(df_cleaned)
        
        # Track cleaning operations
        self.cleaning_report['game_data'] = {
            'initial_rows': initial_rows,
            'operations': []
        }
        
        # 1. Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned, 'game')
        
        # 2. Remove duplicates by app_id
        duplicates_before = df_cleaned.duplicated(subset=['app_id']).sum()
        df_cleaned = df_cleaned.drop_duplicates(subset=['app_id'], keep='first')
        duplicates_removed = duplicates_before
        
        if duplicates_removed > 0:
            self.cleaning_report['game_data']['operations'].append(
                f"Removed {duplicates_removed} duplicate game records"
            )
        
        # 3. Clean game names
        df_cleaned['name'] = df_cleaned['name'].apply(self._clean_game_name)
        
        # 4. Standardize genres
        if 'genres' in df_cleaned.columns:
            df_cleaned['genres'] = df_cleaned['genres'].apply(self._standardize_genres)
        
        # 5. Fix price data
        if 'price' in df_cleaned.columns:
            df_cleaned = self._fix_price_data(df_cleaned)
        
        # 6. Validate release dates
        if 'release_date' in df_cleaned.columns:
            df_cleaned = self._validate_release_dates(df_cleaned)
        
        # 7. Clean developer/publisher fields
        for field in ['developers', 'publishers']:
            if field in df_cleaned.columns:
                df_cleaned[field] = df_cleaned[field].apply(self._clean_company_names)
        
        final_rows = len(df_cleaned)
        self.cleaning_report['game_data']['final_rows'] = final_rows
        self.cleaning_report['game_data']['rows_removed'] = initial_rows - final_rows
        
        logger.info(f"Game data cleaning complete: {df_cleaned.shape}")
        return df_cleaned
    
    def clean_recommendations_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean game recommendations data"""
        logger.info(f"Cleaning recommendations data: {df.shape}")
        
        df_cleaned = df.copy()
        initial_rows = len(df_cleaned)
        
        # Track cleaning operations
        self.cleaning_report['recommendations_data'] = {
            'initial_rows': initial_rows,
            'operations': []
        }
        
        # 1. Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned, 'recommendation')
        
        # 2. Remove invalid user/app combinations
        invalid_mask = (
            (df_cleaned.get('user_id', '').astype(str).str.len() < 5) |
            (df_cleaned.get('app_id', 0) <= 0)
        )
        invalid_count = invalid_mask.sum()
        df_cleaned = df_cleaned[~invalid_mask]
        
        if invalid_count > 0:
            self.cleaning_report['recommendations_data']['operations'].append(
                f"Removed {invalid_count} invalid user/app combinations"
            )
        
        # 3. Clean review text
        if 'review_text' in df_cleaned.columns:
            df_cleaned['review_text'] = df_cleaned['review_text'].apply(self._clean_review_text)
        
        # 4. Validate helpfulness votes
        for vote_col in ['helpful', 'funny']:
            if vote_col in df_cleaned.columns:
                df_cleaned[vote_col] = pd.to_numeric(df_cleaned[vote_col], errors='coerce').fillna(0)
                df_cleaned[vote_col] = df_cleaned[vote_col].clip(lower=0)
        
        # 5. Fix hours played data
        if 'hours' in df_cleaned.columns:
            df_cleaned['hours'] = pd.to_numeric(df_cleaned['hours'], errors='coerce').fillna(0)
            df_cleaned['hours'] = df_cleaned['hours'].clip(lower=0, upper=50000)  # Cap at reasonable max
        
        # 6. Validate recommendation status
        if 'is_recommended' in df_cleaned.columns:
            df_cleaned['is_recommended'] = df_cleaned['is_recommended'].astype(bool)
        
        final_rows = len(df_cleaned)
        self.cleaning_report['recommendations_data']['final_rows'] = final_rows
        self.cleaning_report['recommendations_data']['rows_removed'] = initial_rows - final_rows
        
        logger.info(f"Recommendations data cleaning complete: {df_cleaned.shape}")
        return df_cleaned
    
    def _handle_missing_values(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Handle missing values based on data type"""
        
        missing_strategies = {
            'player': {
                'total_playtime_hours': 0,
                'total_spent': 0,
                'friends_count': 0,
                'achievements_unlocked': 0,
                'forum_posts': 0,
                'reviews_written': 0,
                'sessions_last_week': 0,
                'purchases_last_month': 0,
                'avg_session_duration': df.get('avg_session_duration', pd.Series([0])).median(),
                'account_level': 1,
                'favorite_genre': 'Unknown',
                'platform_preference': 'PC',
                'region': 'Unknown',
                'age_group': '26-35'
            },
            'game': {
                'price': 0.0,
                'is_free': True,
                'metacritic_score': None,  # Keep as missing for scores
                'genres': 'Indie',
                'developers': 'Unknown',
                'publishers': 'Unknown',
                'platforms': 'Windows'
            },
            'recommendation': {
                'helpful': 0,
                'funny': 0,
                'hours': 0,
                'is_recommended': True,
                'review_text': ''
            }
        }
        
        if data_type in missing_strategies:
            strategy = missing_strategies[data_type]
            for column, fill_value in strategy.items():
                if column in df.columns:
                    if fill_value is not None:
                        df[column] = df[column].fillna(fill_value)
        
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types for proper analysis"""
        
        # Numeric columns that should be integers
        int_columns = [
            'friends_count', 'achievements_unlocked', 'forum_posts', 
            'reviews_written', 'sessions_last_week', 'purchases_last_month',
            'account_level', 'games_owned', 'days_since_registration'
        ]
        
        for col in int_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Numeric columns that should be floats
        float_columns = [
            'total_playtime_hours', 'total_spent', 'avg_session_duration', 'last_login_days_ago'
        ]
        
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Boolean columns
        bool_columns = ['churned', 'is_active']
        
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Date columns
        date_columns = ['registration_date', 'last_login']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        
        numeric_columns = [
            'total_playtime_hours', 'total_spent', 'avg_session_duration'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR  # More conservative than 1.5 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower=max(0, lower_bound), upper=upper_bound)
        
        return df
    
    def _validate_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate business logic rules"""
        
        # Players cannot have negative values for these fields
        non_negative_fields = [
            'total_playtime_hours', 'total_spent', 'friends_count',
            'achievements_unlocked', 'forum_posts', 'reviews_written'
        ]
        
        for field in non_negative_fields:
            if field in df.columns:
                df[field] = df[field].clip(lower=0)
        
        # Session duration should be reasonable (0-500 minutes)
        if 'avg_session_duration' in df.columns:
            df['avg_session_duration'] = df['avg_session_duration'].clip(lower=0, upper=500)
        
        # Account level should be reasonable (1-1000)
        if 'account_level' in df.columns:
            df['account_level'] = df['account_level'].clip(lower=1, upper=1000)
        
        # Last login cannot be in the future
        if 'last_login_days_ago' in df.columns:
            df['last_login_days_ago'] = df['last_login_days_ago'].clip(lower=0)
        
        return df
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text fields"""
        
        text_fields = ['favorite_genre', 'platform_preference', 'region', 'age_group']
        
        for field in text_fields:
            if field in df.columns:
                # Strip whitespace and standardize
                df[field] = df[field].astype(str).str.strip().str.title()
                
                # Handle empty strings
                df[field] = df[field].replace('', 'Unknown')
                df[field] = df[field].replace('Nan', 'Unknown')
        
        return df
    
    def _clean_game_name(self, name: str) -> str:
        """Clean game names"""
        if pd.isna(name) or name == '':
            return 'Unknown Game'
        
        # Remove HTML tags
        name = re.sub(r'<[^>]+>', '', str(name))
        
        # Remove excessive whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Remove special characters at the end
        name = re.sub(r'[^\w\s\-\':&!()]+$', '', name)
        
        return name
    
    def _standardize_genres(self, genres: str) -> str:
        """Standardize game genres"""
        if pd.isna(genres) or genres == '':
            return 'Indie'
        
        # Common genre mappings
        genre_mapping = {
            'rpg': 'RPG',
            'fps': 'FPS', 
            'mmorpg': 'MMORPG',
            'rts': 'RTS',
            'moba': 'MOBA'
        }
        
        genres_lower = str(genres).lower()
        for old, new in genre_mapping.items():
            genres_lower = genres_lower.replace(old, new)
        
        return genres_lower.title()
    
    def _fix_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix price data inconsistencies"""
        
        if 'price' in df.columns:
            # Convert to numeric
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
            
            # Ensure non-negative
            df['price'] = df['price'].clip(lower=0)
            
            # Update is_free status based on price
            if 'is_free' in df.columns:
                df['is_free'] = (df['price'] == 0.0)
        
        return df
    
    def _validate_release_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate game release dates"""
        
        if 'release_date' in df.columns:
            # Convert to datetime
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            
            # Remove future dates (games can't be released in future)
            future_mask = df['release_date'] > datetime.now()
            df.loc[future_mask, 'release_date'] = None
            
            # Remove very old dates (before 1970)
            old_mask = df['release_date'] < datetime(1970, 1, 1)
            df.loc[old_mask, 'release_date'] = None
        
        return df
    
    def _clean_company_names(self, company: str) -> str:
        """Clean developer/publisher names"""
        if pd.isna(company) or company == '':
            return 'Unknown'
        
        # Remove common suffixes
        company = str(company)
        suffixes = ['Inc.', 'LLC', 'Ltd.', 'Corp.', 'Corporation']
        for suffix in suffixes:
            company = company.replace(suffix, '').strip()
        
        # Remove excessive whitespace
        company = re.sub(r'\s+', ' ', company).strip()
        
        return company
    
    def _clean_review_text(self, text: str) -> str:
        """Clean review text"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit length for database storage
        if len(text) > 1000:
            text = text[:1000] + '...'
        
        return text
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get comprehensive cleaning report"""
        return self.cleaning_report

class DataValidator:
    """Validate data quality and integrity"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Comprehensive dataset validation"""
        
        logger.info(f"Validating {dataset_name} dataset: {df.shape}")
        
        results = {
            'dataset_name': dataset_name,
            'validation_date': datetime.now().isoformat(),
            'shape': df.shape,
            'checks': {},
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        # Basic checks
        results['checks']['missing_values'] = df.isnull().sum().to_dict()
        results['checks']['duplicate_rows'] = df.duplicated().sum()
        results['checks']['data_types'] = df.dtypes.to_dict()
        
        # Missing value analysis
        missing_threshold = 0.5  # 50% missing is concerning
        for col, missing_count in results['checks']['missing_values'].items():
            missing_pct = missing_count / len(df)
            if missing_pct > missing_threshold:
                results['issues'].append(f"{col}: {missing_pct:.1%} missing values")
            elif missing_pct > 0.1:  # 10% missing is a warning
                results['warnings'].append(f"{col}: {missing_pct:.1%} missing values")
        
        # Duplicate analysis
        if results['checks']['duplicate_rows'] > 0:
            dup_pct = results['checks']['duplicate_rows'] / len(df)
            if dup_pct > 0.05:  # 5% duplicates is an issue
                results['issues'].append(f"{dup_pct:.1%} duplicate rows")
            else:
                results['warnings'].append(f"{dup_pct:.1%} duplicate rows")
        
        # Dataset-specific validations
        if dataset_name == 'player_data':
            results = self._validate_player_data(df, results)
        elif dataset_name == 'game_data':
            results = self._validate_game_data(df, results)
        elif dataset_name == 'recommendations_data':
            results = self._validate_recommendations_data(df, results)
        
        # Summary
        results['summary'] = {
            'total_checks': len(results['checks']),
            'issues_count': len(results['issues']),
            'warnings_count': len(results['warnings']),
            'data_quality_score': self._calculate_quality_score(results),
            'validation_passed': len(results['issues']) == 0
        }
        
        self.validation_results[dataset_name] = results
        logger.info(f"Validation complete for {dataset_name}: "
                   f"Score {results['summary']['data_quality_score']}/100")
        
        return results
    
    def _validate_player_data(self, df: pd.DataFrame, results: Dict) -> Dict:
        """Player-specific validation checks"""
        
        # Check for negative values in fields that should be positive
        negative_fields = ['total_playtime_hours', 'total_spent', 'friends_count']
        for field in negative_fields:
            if field in df.columns:
                negative_count = (df[field] < 0).sum()
                if negative_count > 0:
                    results['issues'].append(f"{field}: {negative_count} negative values")
        
        # Check for unrealistic values
        if 'avg_session_duration' in df.columns:
            extreme_sessions = (df['avg_session_duration'] > 600).sum()  # > 10 hours
            if extreme_sessions > 0:
                results['warnings'].append(f"{extreme_sessions} players with >10h avg sessions")
        
        if 'total_spent' in df.columns:
            high_spenders = (df['total_spent'] > 10000).sum()  # > $10k
            if high_spenders > 0:
                results['warnings'].append(f"{high_spenders} players spent >$10k")
        
        # Check churn distribution
        if 'churned' in df.columns:
            churn_rate = df['churned'].mean()
            results['checks']['churn_rate'] = churn_rate
            if churn_rate > 0.8 or churn_rate < 0.05:
                results['warnings'].append(f"Unusual churn rate: {churn_rate:.1%}")
        
        return results
    
    def _validate_game_data(self, df: pd.DataFrame, results: Dict) -> Dict:
        """Game-specific validation checks"""
        
        # Check for missing essential fields
        essential_fields = ['name', 'app_id']
        for field in essential_fields:
            if field not in df.columns:
                results['issues'].append(f"Missing essential field: {field}")
            elif df[field].isnull().sum() > 0:
                results['issues'].append(f"Essential field {field} has missing values")
        
        # Check for duplicate app_ids
        if 'app_id' in df.columns:
            duplicate_apps = df['app_id'].duplicated().sum()
            if duplicate_apps > 0:
                results['issues'].append(f"{duplicate_apps} duplicate app_ids")
        
        # Validate price data
        if 'price' in df.columns:
            negative_prices = (df['price'] < 0).sum()
            if negative_prices > 0:
                results['issues'].append(f"{negative_prices} games with negative prices")
        
        return results
    
    def _validate_recommendations_data(self, df: pd.DataFrame, results: Dict) -> Dict:
        """Recommendations-specific validation checks"""
        
        # Check for valid user/app combinations
        if 'user_id' in df.columns and 'app_id' in df.columns:
            invalid_users = df['user_id'].isnull().sum()
            invalid_apps = df['app_id'].isnull().sum()
            
            if invalid_users > 0:
                results['issues'].append(f"{invalid_users} recommendations with invalid user_id")
            if invalid_apps > 0:
                results['issues'].append(f"{invalid_apps} recommendations with invalid app_id")
        
        # Check recommendation distribution
        if 'is_recommended' in df.columns:
            rec_rate = df['is_recommended'].mean()
            results['checks']['recommendation_rate'] = rec_rate
            if rec_rate > 0.95 or rec_rate < 0.05:
                results['warnings'].append(f"Extreme recommendation rate: {rec_rate:.1%}")
        
        return results
    
    def _calculate_quality_score(self, results: Dict) -> int:
        """Calculate overall data quality score (0-100)"""
        score = 100
        
        # Deduct points for issues and warnings
        score -= len(results['issues']) * 15  # Major issues
        score -= len(results['warnings']) * 5   # Minor issues
        
        # Deduct points for high missing value rates
        for col, missing_count in results['checks']['missing_values'].items():
            if missing_count > 0:
                missing_rate = missing_count / results['shape'][0]
                if missing_rate > 0.5:
                    score -= 20
                elif missing_rate > 0.2:
                    score -= 10
                elif missing_rate > 0.1:
                    score -= 5
        
        return max(0, score)

class DataTransformer:
    """Transform data for analysis and modeling"""
    
    def __init__(self):
        self.transformations_applied = []
    
    def prepare_for_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataset for analysis"""
        
        logger.info(f"Preparing data for analysis: {df.shape}")
        
        df_transformed = df.copy()
        
        # Apply transformations
        df_transformed = self._create_derived_features(df_transformed)
        df_transformed = self._normalize_distributions(df_transformed)
        df_transformed = self._encode_categories(df_transformed)
        
        logger.info(f"Data preparation complete: {df_transformed.shape}")
        return df_transformed
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for analysis"""
        
        # Calculate days since registration if date is available
        if 'registration_date' in df.columns:
            df['days_since_registration'] = (
                datetime.now() - pd.to_datetime(df['registration_date'])
            ).dt.days
            self.transformations_applied.append('days_since_registration')
        
        # Calculate value per hour if both columns exist
        if 'total_spent' in df.columns and 'total_playtime_hours' in df.columns:
            df['value_per_hour'] = np.where(
                df['total_playtime_hours'] > 0,
                df['total_spent'] / df['total_playtime_hours'],
                0
            )
            self.transformations_applied.append('value_per_hour')
        
        return df
    
    def _normalize_distributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize skewed distributions"""
        
        # Log transform highly skewed numerical features
        skewed_features = ['total_playtime_hours', 'total_spent']
        
        for feature in skewed_features:
            if feature in df.columns:
                df[f'{feature}_log'] = np.log1p(df[feature])
                self.transformations_applied.append(f'{feature}_log')
        
        return df
    
    def _encode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        
        # One-hot encode low-cardinality categorical features
        categorical_features = ['platform_preference', 'favorite_genre', 'region']
        
        for feature in categorical_features:
            if feature in df.columns and df[feature].nunique() <= 10:
                dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                self.transformations_applied.extend(list(dummies.columns))
        
        return df

def main():
    """Example usage of data processing"""
    from .data_collector import SyntheticDataGenerator
    
    # Generate sample data
    generator = SyntheticDataGenerator()
    sample_data = generator.generate_player_data(1000)
    
    # Clean data
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_player_data(sample_data)
    
    # Validate data
    validator = DataValidator()
    validation_results = validator.validate_dataset(cleaned_data, 'player_data')
    
    # Transform data
    transformer = DataTransformer()
    transformed_data = transformer.prepare_for_analysis(cleaned_data)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(f"Transformed shape: {transformed_data.shape}")
    print(f"Data quality score: {validation_results['summary']['data_quality_score']}/100")
    print(f"Transformations applied: {len(transformer.transformations_applied)}")

if __name__ == "__main__":
    main()