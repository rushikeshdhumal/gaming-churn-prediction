"""
Data Collection Module for Gaming Analytics

Handles data collection from various sources including Steam API, 
Kaggle datasets, and synthetic data generation.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import pandas as pd
import numpy as np
import requests
import logging
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SteamAPICollector:
    """Collect data from Steam Web API with rate limiting"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('STEAM_API_KEY')
        self.base_url = "http://api.steampowered.com/"
        self.rate_limit_delay = 1.0
        
        if not self.api_key:
            logger.warning("No Steam API key provided. Limited functionality available.")
    
    def get_app_list(self) -> List[Dict]:
        """Get list of all Steam applications"""
        try:
            url = f"{self.base_url}ISteamApps/GetAppList/v2/"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('applist', {}).get('apps', [])
            else:
                logger.error(f"Failed to fetch app list: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching Steam app list: {e}")
            return []
    
    def get_game_details(self, app_ids: List[int]) -> pd.DataFrame:
        """Fetch detailed game information for given app IDs"""
        games_data = []
        
        for i, app_id in enumerate(app_ids):
            try:
                time.sleep(self.rate_limit_delay)
                
                url = "https://store.steampowered.com/api/appdetails"
                params = {'appids': app_id, 'format': 'json'}
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    app_data = data.get(str(app_id), {})
                    
                    if app_data.get('success') and 'data' in app_data:
                        game_info = app_data['data']
                        
                        games_data.append({
                            'app_id': app_id,
                            'name': game_info.get('name', ''),
                            'type': game_info.get('type', ''),
                            'is_free': game_info.get('is_free', False),
                            'price': self._extract_price(game_info),
                            'genres': self._extract_genres(game_info),
                            'categories': self._extract_categories(game_info),
                            'release_date': self._extract_release_date(game_info),
                            'metacritic_score': self._extract_metacritic(game_info),
                            'positive_ratings': game_info.get('positive_ratings', 0),
                            'negative_ratings': game_info.get('negative_ratings', 0),
                            'platforms': self._extract_platforms(game_info)
                        })
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(app_ids)} games")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for app_id {app_id}: {e}")
                continue
        
        logger.info(f"Successfully collected data for {len(games_data)} games")
        return pd.DataFrame(games_data)
    
    def _extract_price(self, game_info: Dict) -> float:
        """Extract price information from game data"""
        price_overview = game_info.get('price_overview', {})
        if price_overview:
            return price_overview.get('final', 0) / 100.0
        return 0.0
    
    def _extract_genres(self, game_info: Dict) -> str:
        """Extract genre information"""
        genres = game_info.get('genres', [])
        return ', '.join([genre.get('description', '') for genre in genres])
    
    def _extract_categories(self, game_info: Dict) -> str:
        """Extract category information"""
        categories = game_info.get('categories', [])
        return ', '.join([cat.get('description', '') for cat in categories])
    
    def _extract_release_date(self, game_info: Dict) -> Optional[str]:
        """Extract release date"""
        release_date = game_info.get('release_date', {})
        return release_date.get('date') if release_date.get('coming_soon') is False else None
    
    def _extract_metacritic(self, game_info: Dict) -> Optional[int]:
        """Extract Metacritic score"""
        metacritic = game_info.get('metacritic', {})
        return metacritic.get('score') if metacritic else None
    
    def _extract_platforms(self, game_info: Dict) -> str:
        """Extract supported platforms"""
        platforms = game_info.get('platforms', {})
        supported = []
        if platforms.get('windows'): supported.append('Windows')
        if platforms.get('mac'): supported.append('Mac')
        if platforms.get('linux'): supported.append('Linux')
        return ', '.join(supported)

class SyntheticDataGenerator:
    """Generate realistic synthetic player behavioral data"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
    
    def generate_player_data(self, n_players: int = 10000) -> pd.DataFrame:
        """Generate comprehensive synthetic player dataset"""
        logger.info(f"Generating synthetic data for {n_players} players")
        
        player_ids = [f"player_{i:06d}" for i in range(n_players)]
        registration_dates = self._generate_registration_dates(n_players)
        
        age_groups = np.random.choice(['18-25', '26-35', '36-45', '46+'], 
                                    size=n_players, p=[0.35, 0.40, 0.20, 0.05])
        regions = np.random.choice(['NA', 'EU', 'ASIA', 'OTHER'], 
                                 size=n_players, p=[0.40, 0.35, 0.20, 0.05])
        
        base_data = {
            'player_id': player_ids,
            'registration_date': registration_dates,
            'age_group': age_groups,
            'region': regions,
            'days_since_registration': [(datetime.now().date() - date.date()).days 
                                       for date in registration_dates],
        }
        
        behavioral_data = self._generate_behavioral_features(n_players)
        base_data.update(behavioral_data)
        
        df = pd.DataFrame(base_data)
        df['churned'] = self._generate_churn_labels(df)
        
        logger.info("Synthetic data generation completed")
        return df
    
    def _generate_registration_dates(self, n_players: int) -> List[datetime]:
        """Generate realistic registration date distribution"""
        weights = np.exp(np.linspace(-2, 0, 730))
        weights = weights / weights.sum()
        
        days_ago = np.random.choice(730, size=n_players, p=weights)
        base_date = datetime.now() - timedelta(days=730)
        
        return [base_date + timedelta(days=int(day)) for day in days_ago]
    
    def _generate_behavioral_features(self, n_players: int) -> Dict:
        """Generate correlated behavioral features"""
        
        engagement_level = np.random.beta(2, 5, n_players)
        
        total_playtime = np.random.exponential(scale=100, size=n_players) * (1 + engagement_level * 2)
        avg_session_duration = np.random.lognormal(mean=3, sigma=1, size=n_players) * (0.5 + engagement_level)
        sessions_last_week = np.random.poisson(lam=3, size=n_players) * (1 + engagement_level * 3)
        
        games_owned = np.random.negative_binomial(n=3, p=0.2, size=n_players) * (1 + engagement_level)
        favorite_genres = np.random.choice(
            ['Action', 'RPG', 'Strategy', 'Simulation', 'Indie', 'Sports', 'Racing'], 
            size=n_players, 
            p=[0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08]
        )
        
        friends_count = np.random.negative_binomial(n=2, p=0.3, size=n_players) * (1 + engagement_level * 2)
        forum_posts = np.random.poisson(lam=2, size=n_players) * engagement_level
        reviews_written = np.random.poisson(lam=1, size=n_players) * engagement_level
        
        has_purchases = np.random.binomial(1, 0.7 + 0.2 * engagement_level, size=n_players)
        total_spent = np.where(
            has_purchases,
            np.random.exponential(scale=50, size=n_players) * (1 + engagement_level * 2),
            0
        )
        purchases_last_month = np.where(
            has_purchases,
            np.random.poisson(lam=1, size=n_players) * (1 + engagement_level),
            0
        )
        
        achievements_unlocked = np.random.poisson(lam=20, size=n_players) * (1 + engagement_level * 2)
        account_level = np.random.poisson(lam=8, size=n_players) * (1 + engagement_level)
        
        platform_preference = np.random.choice(
            ['PC', 'Mac', 'Linux'], 
            size=n_players, 
            p=[0.85, 0.12, 0.03]
        )
        
        base_inactivity = np.random.exponential(scale=5, size=n_players)
        last_login_days_ago = base_inactivity * (2 - engagement_level)
        
        return {
            'total_playtime_hours': total_playtime,
            'avg_session_duration': avg_session_duration,
            'sessions_last_week': sessions_last_week,
            'games_owned': games_owned.astype(int),
            'favorite_genre': favorite_genres,
            'friends_count': friends_count.astype(int),
            'forum_posts': forum_posts.astype(int),
            'reviews_written': reviews_written.astype(int),
            'total_spent': total_spent,
            'purchases_last_month': purchases_last_month.astype(int),
            'achievements_unlocked': achievements_unlocked.astype(int),
            'account_level': account_level.astype(int),
            'platform_preference': platform_preference,
            'last_login_days_ago': last_login_days_ago,
            'engagement_level': engagement_level
        }
    
    def _generate_churn_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate realistic churn labels based on player behavior"""
        
        churn_prob = 0.15
        
        churn_prob += 0.4 * (df['last_login_days_ago'] > 30)
        churn_prob += 0.3 * (df['avg_session_duration'] < 15)
        churn_prob += 0.2 * (df['total_spent'] == 0)
        churn_prob += 0.15 * (df['sessions_last_week'] == 0)
        
        churn_prob -= 0.2 * (df['friends_count'] > 5)
        churn_prob -= 0.15 * (df['achievements_unlocked'] > 50)
        churn_prob -= 0.1 * (df['total_spent'] > 100)
        churn_prob -= 0.1 * (df['days_since_registration'] > 365)
        
        churn_prob = np.clip(churn_prob, 0.05, 0.85)
        churned = np.random.binomial(1, churn_prob)
        
        logger.info(f"Generated churn labels with {churned.mean():.2%} churn rate")
        return churned

class KaggleDataLoader:
    """Load and process Kaggle datasets"""
    
    def __init__(self, data_path: str = "data/raw/"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def load_steam_games_dataset(self) -> Optional[pd.DataFrame]:
        """Load Steam games dataset (sample or full)"""
        file_path = self.data_path / "steam_games.csv"
        
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded Steam games dataset: {df.shape}")
                
                if len(df) <= 10000:
                    logger.info("Using sample dataset for demonstration")
                else:
                    logger.info("Using full dataset")
                    
                return df
            except Exception as e:
                logger.error(f"Error loading Steam games dataset: {e}")
                return None
        else:
            logger.warning(f"Steam games dataset not found at {file_path}")
            logger.info("Sample dataset (8,500 games) should be included in repository")
            return None
    
    def load_game_recommendations_dataset(self) -> Optional[pd.DataFrame]:
        """Load game recommendations dataset (sample or full)"""
        file_path = self.data_path / "game_recommendations.csv"
        
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded game recommendations dataset: {df.shape}")
                
                if len(df) <= 100000:
                    logger.info("Using sample dataset for demonstration")
                else:
                    logger.info("Using full dataset")
                    
                return df
            except Exception as e:
                logger.error(f"Error loading recommendations dataset: {e}")
                return None
        else:
            logger.warning(f"Recommendations dataset not found at {file_path}")
            logger.info("Sample dataset (75,000 recommendations) should be included in repository")
            return None

class DataCollectionPipeline:
    """Main data collection pipeline orchestrator"""
    
    def __init__(self, output_path: str = "data/processed/"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.steam_collector = SteamAPICollector()
        self.synthetic_generator = SyntheticDataGenerator()
        self.kaggle_loader = KaggleDataLoader()
    
    def run_full_collection(self, n_synthetic_players: int = 10000) -> Dict[str, pd.DataFrame]:
        """Run complete data collection pipeline"""
        logger.info("Starting full data collection pipeline")
        
        datasets = {}
        
        logger.info("Generating synthetic player data...")
        synthetic_data = self.synthetic_generator.generate_player_data(n_synthetic_players)
        datasets['synthetic_players'] = synthetic_data
        
        synthetic_path = self.output_path / "synthetic_player_data.csv"
        synthetic_data.to_csv(synthetic_path, index=False)
        logger.info(f"Saved synthetic data to {synthetic_path}")
        
        logger.info("Loading Kaggle datasets...")
        steam_games = self.kaggle_loader.load_steam_games_dataset()
        if steam_games is not None:
            datasets['steam_games'] = steam_games
        
        recommendations = self.kaggle_loader.load_game_recommendations_dataset()
        if recommendations is not None:
            datasets['recommendations'] = recommendations
        
        if self.steam_collector.api_key:
            logger.info("Collecting Steam API data...")
            try:
                popular_app_ids = [730, 440, 570, 578080, 271590, 292030, 431960, 359550]
                steam_data = self.steam_collector.get_game_details(popular_app_ids)
                
                if not steam_data.empty:
                    datasets['steam_api_data'] = steam_data
                    
                    steam_path = self.output_path / "steam_api_data.csv"
                    steam_data.to_csv(steam_path, index=False)
                    logger.info(f"Saved Steam API data to {steam_path}")
                    
            except Exception as e:
                logger.error(f"Error collecting Steam API data: {e}")
        
        summary = self._generate_collection_summary(datasets)
        
        summary_path = self.output_path / "data_collection_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Data collection pipeline completed successfully")
        return datasets
    
    def _generate_collection_summary(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """Generate summary of collected data"""
        summary = {
            'collection_date': datetime.now().isoformat(),
            'datasets_collected': len(datasets),
            'total_records': sum(len(df) for df in datasets.values()),
            'dataset_details': {}
        }
        
        for name, df in datasets.items():
            summary['dataset_details'][name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'size_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'missing_values': df.isnull().sum().sum()
            }
        
        return summary

def main():
    """Main function for CLI usage"""
    pipeline = DataCollectionPipeline()
    collected_data = pipeline.run_full_collection(n_synthetic_players=10000)
    
    print("Data collection completed!")
    print(f"Collected {len(collected_data)} datasets:")
    for name, df in collected_data.items():
        print(f"  - {name}: {df.shape}")

if __name__ == "__main__":
    main()