"""
Data Collection and Processing Module

This module handles all data collection, processing, and preparation tasks
including Steam API integration, synthetic data generation, and data validation.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

from .data_collector import (
    SteamAPICollector,
    SyntheticDataGenerator,
    KaggleDataLoader,
    DataCollectionPipeline
)

from .data_processor import (
    DataCleaner,
    DataValidator,
    DataTransformer
)

from .synthetic_generator import (
    PlayerBehaviorSimulator,
    RealisticDataGenerator,
    ChurnPatternGenerator
)

__all__ = [
    # Data Collection
    "SteamAPICollector",
    "SyntheticDataGenerator", 
    "KaggleDataLoader",
    "DataCollectionPipeline",
    
    # Data Processing
    "DataCleaner",
    "DataValidator",
    "DataTransformer",
    
    # Synthetic Generation
    "PlayerBehaviorSimulator",
    "RealisticDataGenerator",
    "ChurnPatternGenerator",
]

# Module configuration
import logging

logger = logging.getLogger(__name__)

# Data collection constants
DEFAULT_SAMPLE_SIZES = {
    "steam_games": 8500,
    "game_recommendations": 75000,
    "synthetic_players": 10000
}

DATA_SOURCES = {
    "steam_api": "Steam Web API",
    "kaggle_games": "https://www.kaggle.com/datasets/fronkongames/steam-games-dataset",
    "kaggle_recommendations": "https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam"
}

def get_data_info():
    """Return data module information"""
    return {
        "module": "data",
        "version": "1.0.0",
        "sample_sizes": DEFAULT_SAMPLE_SIZES,
        "sources": DATA_SOURCES,
        "maintainer": "Rushikesh Dhumal"
    }