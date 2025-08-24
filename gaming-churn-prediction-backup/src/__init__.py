"""
Gaming Player Behavior Analysis & Churn Prediction

A comprehensive data science package for analyzing gaming player behavior 
and predicting player churn using advanced machine learning techniques.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Rushikesh Dhumal"
__email__ = "r.dhumal@rutgers.edu"
__license__ = "MIT"
__description__ = "Advanced machine learning system for predicting player churn in gaming applications"

# Package imports
from . import data
from . import features
from . import models
from . import utils
from . import visualization

# Key classes and functions for easy access
from .data.data_collector import SteamAPICollector, SyntheticDataGenerator, DataCollectionPipeline
from .models.train_model import ModelTrainer, FeaturePreprocessor
from .utils.deployment_utils import ModelDeployment, RealTimeAPI
from .features.feature_engineering import FeatureEngineer

__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    "__description__",
    
    # Modules
    "data",
    "features", 
    "models",
    "utils",
    "visualization",
    
    # Key classes
    "SteamAPICollector",
    "SyntheticDataGenerator", 
    "DataCollectionPipeline",
    "ModelTrainer",
    "FeaturePreprocessor",
    "ModelDeployment",
    "RealTimeAPI",
    "FeatureEngineer",
]

# Package configuration
import logging
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Package metadata
PACKAGE_INFO = {
    "name": "gaming-churn-prediction",
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "description": __description__,
    "url": "https://github.com/rushikeshdhumal/gaming-churn-prediction",
    "license": __license__,
    "python_requires": ">=3.8",
}

def get_package_info():
    """Return package information dictionary"""
    return PACKAGE_INFO.copy()

def print_package_info():
    """Print formatted package information"""
    print(f"""
🎮 Gaming Player Behavior Analysis & Churn Prediction
====================================================
Version: {__version__}
Author: {__author__}
Email: {__email__}
License: {__license__}

Description: {__description__}

Repository: https://github.com/rushikeshdhumal/gaming-churn-prediction
Documentation: https://github.com/rushikeshdhumal/gaming-churn-prediction#readme

Quick Start:
1. from src.data.data_collector import DataCollectionPipeline
2. from src.models.train_model import ModelTrainer
3. from src.utils.deployment_utils import ModelDeployment

Happy analyzing! 🚀
""")

# Validate dependencies on import
def _validate_dependencies():
    """Validate critical dependencies are available"""
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        return True
    except ImportError as e:
        logging.warning(f"Missing dependency: {e}")
        return False

# Run validation
_dependencies_ok = _validate_dependencies()

if not _dependencies_ok:
    logging.warning("Some dependencies are missing. Run: pip install -r requirements.txt")