"""
Configuration Management Module

This module handles configuration settings, environment variables,
and project parameters for the gaming analytics system.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "gaming_analytics.db"
    username: str = ""
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False
    
    @property
    def connection_string(self) -> str:
        """Generate database connection string"""
        if self.type == "sqlite":
            return f"sqlite:///{self.database}"
        elif self.type == "postgresql":
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.type == "mysql":
            return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.type}")

@dataclass
class APIConfig:
    """API configuration settings"""
    steam_api_key: str = ""
    steam_rate_limit: float = 1.0
    steam_timeout: int = 10
    steam_retries: int = 3
    kaggle_username: str = ""
    kaggle_key: str = ""
    
    def __post_init__(self):
        """Load API keys from environment variables"""
        self.steam_api_key = self.steam_api_key or os.getenv('STEAM_API_KEY', '')
        self.kaggle_username = self.kaggle_username or os.getenv('KAGGLE_USERNAME', '')
        self.kaggle_key = self.kaggle_key or os.getenv('KAGGLE_KEY', '')

@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = -1
    
    # Feature selection
    max_features: int = 50
    feature_selection_method: str = "kbest"
    
    # Model hyperparameters
    models_to_train: list = None
    hyperparameter_tuning: bool = True
    tuning_method: str = "grid"  # grid, random, bayesian
    tuning_iterations: int = 100
    
    # Performance thresholds
    min_accuracy: float = 0.85
    min_precision: float = 0.80
    min_recall: float = 0.80
    min_f1_score: float = 0.80
    min_roc_auc: float = 0.85
    
    def __post_init__(self):
        """Set default models if not specified"""
        if self.models_to_train is None:
            self.models_to_train = [
                'logistic_regression',
                'random_forest', 
                'gradient_boosting',
                'xgboost'
            ]

@dataclass
class DataConfig:
    """Data processing configuration"""
    # Data paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    external_data_path: str = "data/external"
    
    # Sampling configuration
    synthetic_players: int = 10000
    steam_games_sample: int = 8500
    recommendations_sample: int = 75000
    
    # Data quality
    missing_threshold: float = 0.5
    outlier_method: str = "iqr"
    outlier_multiplier: float = 3.0
    
    # Data validation
    validate_on_load: bool = True
    auto_clean: bool = True
    backup_raw_data: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/gaming_analytics.log"
    max_file_size: int = 10  # MB
    backup_count: int = 5
    log_to_console: bool = True
    log_to_file: bool = True

@dataclass
class DeploymentConfig:
    """Deployment and production configuration"""
    environment: str = "development"
    debug: bool = True
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    
    # Model serving
    model_path: str = "models"
    prediction_batch_size: int = 1000
    prediction_timeout: int = 30
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_collection: bool = True
    performance_logging: bool = True
    
    # Security
    api_key_required: bool = False
    rate_limiting: bool = True
    max_requests_per_hour: int = 1000

@dataclass 
class EnvironmentConfig:
    """Environment-specific configuration"""
    name: str
    database: DatabaseConfig
    api: APIConfig
    model: ModelConfig
    data: DataConfig
    logging: LoggingConfig
    deployment: DeploymentConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnvironmentConfig':
        """Create config from dictionary"""
        return cls(
            name=config_dict.get('name', 'development'),
            database=DatabaseConfig(**config_dict.get('database', {})),
            api=APIConfig(**config_dict.get('api', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            deployment=DeploymentConfig(**config_dict.get('deployment', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'name': self.name,
            'database': asdict(self.database),
            'api': asdict(self.api),
            'model': asdict(self.model),
            'data': asdict(self.data),
            'logging': asdict(self.logging),
            'deployment': asdict(self.deployment)
        }

class ConfigManager:
    """Manage configuration settings for different environments"""
    
    def __init__(self, environment: str = None, config_path: str = "config"):
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)
        
        # Determine environment
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        
        # Load configuration
        self.config = self._load_config()
        
        logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def _load_config(self) -> EnvironmentConfig:
        """Load configuration from files or defaults"""
        
        # Try to load from YAML file first
        config_file = self.config_path / f"{self.environment}.yaml"
        if config_file.exists():
            return self._load_from_yaml(config_file)
        
        # Try to load from JSON file
        config_file = self.config_path / f"{self.environment}.json"
        if config_file.exists():
            return self._load_from_json(config_file)
        
        # Load default configuration
        return self._get_default_config()
    
    def _load_from_yaml(self, config_file: Path) -> EnvironmentConfig:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            return EnvironmentConfig.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load YAML config: {e}. Using defaults.")
            return self._get_default_config()
    
    def _load_from_json(self, config_file: Path) -> EnvironmentConfig:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            return EnvironmentConfig.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load JSON config: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> EnvironmentConfig:
        """Get default configuration based on environment"""
        
        if self.environment == 'production':
            return self._get_production_config()
        elif self.environment == 'testing':
            return self._get_testing_config()
        elif self.environment == 'staging':
            return self._get_staging_config()
        else:
            return self._get_development_config()
    
    def _get_development_config(self) -> EnvironmentConfig:
        """Development environment configuration"""
        return EnvironmentConfig(
            name='development',
            database=DatabaseConfig(
                type='sqlite',
                database='gaming_analytics_dev.db',
                echo=True
            ),
            api=APIConfig(),
            model=ModelConfig(
                cv_folds=3,
                hyperparameter_tuning=True,
                models_to_train=['logistic_regression', 'random_forest']
            ),
            data=DataConfig(
                synthetic_players=1000,
                validate_on_load=True,
                auto_clean=True
            ),
            logging=LoggingConfig(
                level='DEBUG',
                log_to_console=True,
                log_to_file=False
            ),
            deployment=DeploymentConfig(
                environment='development',
                debug=True,
                enable_monitoring=False
            )
        )
    
    def _get_testing_config(self) -> EnvironmentConfig:
        """Testing environment configuration"""
        return EnvironmentConfig(
            name='testing',
            database=DatabaseConfig(
                type='sqlite',
                database=':memory:',
                echo=False
            ),
            api=APIConfig(
                steam_rate_limit=0.1  # Faster for testing
            ),
            model=ModelConfig(
                cv_folds=2,
                hyperparameter_tuning=False,
                models_to_train=['logistic_regression']
            ),
            data=DataConfig(
                synthetic_players=100,
                validate_on_load=True,
                auto_clean=True
            ),
            logging=LoggingConfig(
                level='WARNING',
                log_to_console=False,
                log_to_file=False
            ),
            deployment=DeploymentConfig(
                environment='testing',
                debug=False,
                enable_monitoring=False
            )
        )
    
    def _get_staging_config(self) -> EnvironmentConfig:
        """Staging environment configuration"""
        return EnvironmentConfig(
            name='staging',
            database=DatabaseConfig(
                type='sqlite',
                database='gaming_analytics_staging.db',
                echo=False
            ),
            api=APIConfig(),
            model=ModelConfig(
                cv_folds=5,
                hyperparameter_tuning=True
            ),
            data=DataConfig(
                synthetic_players=5000,
                validate_on_load=True,
                auto_clean=True
            ),
            logging=LoggingConfig(
                level='INFO',
                log_to_console=True,
                log_to_file=True
            ),
            deployment=DeploymentConfig(
                environment='staging',
                debug=False,
                enable_monitoring=True
            )
        )
    
    def _get_production_config(self) -> EnvironmentConfig:
        """Production environment configuration"""
        return EnvironmentConfig(
            name='production',
            database=DatabaseConfig(
                type='sqlite',
                database='gaming_analytics.db',
                echo=False
            ),
            api=APIConfig(),
            model=ModelConfig(
                cv_folds=5,
                hyperparameter_tuning=True,
                models_to_train=['logistic_regression', 'random_forest', 'xgboost']
            ),
            data=DataConfig(
                synthetic_players=10000,
                validate_on_load=True,
                auto_clean=False  # More careful in production
            ),
            logging=LoggingConfig(
                level='INFO',
                log_to_console=False,
                log_to_file=True
            ),
            deployment=DeploymentConfig(
                environment='production',
                debug=False,
                enable_monitoring=True,
                api_key_required=True,
                rate_limiting=True
            )
        )
    
    def save_config(self, format: str = 'yaml') -> Path:
        """Save current configuration to file"""
        
        if format == 'yaml':
            config_file = self.config_path / f"{self.environment}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.config.to_dict(), f, default_flow_style=False)
        elif format == 'json':
            config_file = self.config_path / f"{self.environment}.json"
            with open(config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {config_file}")
        return config_file
    
    def get_config(self) -> EnvironmentConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration settings"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate API configuration
        if not self.config.api.steam_api_key:
            validation_results['warnings'].append(
                "Steam API key not configured - some features will be limited"
            )
        
        # Validate database configuration
        if self.config.database.type not in ['sqlite', 'postgresql', 'mysql']:
            validation_results['errors'].append(
                f"Unsupported database type: {self.config.database.type}"
            )
            validation_results['valid'] = False
        
        # Validate model configuration
        if self.config.model.test_size <= 0 or self.config.model.test_size >= 1:
            validation_results['errors'].append(
                f"Invalid test size: {self.config.model.test_size}"
            )
            validation_results['valid'] = False
        
        if self.config.model.cv_folds < 2:
            validation_results['errors'].append(
                f"CV folds must be >= 2, got {self.config.model.cv_folds}"
            )
            validation_results['valid'] = False
        
        # Validate data configuration
        if self.config.data.synthetic_players < 100:
            validation_results['warnings'].append(
                "Low number of synthetic players may affect model performance"
            )
        
        # Validate paths exist
        data_path = Path(self.config.data.raw_data_path)
        if not data_path.exists():
            validation_results['warnings'].append(
                f"Data path does not exist: {data_path}"
            )
        
        return validation_results

def get_config(environment: str = None) -> EnvironmentConfig:
    """Convenience function to get configuration"""
    manager = ConfigManager(environment)
    return manager.get_config()

def create_config_template(output_path: str = "config/template.yaml") -> Path:
    """Create a configuration template file"""
    
    template_config = EnvironmentConfig(
        name='template',
        database=DatabaseConfig(),
        api=APIConfig(),
        model=ModelConfig(),
        data=DataConfig(),
        logging=LoggingConfig(),
        deployment=DeploymentConfig()
    )
    
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(template_config.to_dict(), f, default_flow_style=False)
    
    logger.info(f"Configuration template created: {output_file}")
    return output_file

def main():
    """Example usage of configuration management"""
    
    # Create configuration manager
    config_manager = ConfigManager('development')
    
    # Get configuration
    config = config_manager.get_config()
    print(f"Environment: {config.name}")
    print(f"Database: {config.database.connection_string}")
    print(f"Steam API Key configured: {'Yes' if config.api.steam_api_key else 'No'}")
    
    # Validate configuration
    validation = config_manager.validate_config()
    print(f"Configuration valid: {validation['valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    # Save configuration
    config_file = config_manager.save_config('yaml')
    print(f"Configuration saved to: {config_file}")
    
    # Create template
    template_file = create_config_template()
    print(f"Template created: {template_file}")

if __name__ == "__main__":
    main()