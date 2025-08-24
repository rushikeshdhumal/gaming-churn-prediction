"""
Utilities Module

This module provides utility functions, configuration management,
logging, and deployment utilities for the gaming analytics project.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

from .deployment_utils import (
    ModelDeployment,
    ModelMonitoring,
    BatchPredictionPipeline,
    RealTimeAPI,
    DeploymentValidator
)

from .database import (
    DatabaseManager,
    DataValidator,
    QueryBuilder,
    ConnectionPool
)

from .config import (
    ConfigManager,
    EnvironmentConfig,
    ModelConfig,
    DatabaseConfig
)

from .logger import (
    LoggerSetup,
    PerformanceLogger,
    ModelLogger,
    BusinessLogger
)

__all__ = [
    # Deployment Utilities
    "ModelDeployment",
    "ModelMonitoring",
    "BatchPredictionPipeline", 
    "RealTimeAPI",
    "DeploymentValidator",
    
    # Database Utilities
    "DatabaseManager",
    "DataValidator",
    "QueryBuilder",
    "ConnectionPool",
    
    # Configuration Management
    "ConfigManager",
    "EnvironmentConfig",
    "ModelConfig",
    "DatabaseConfig",
    
    # Logging Utilities
    "LoggerSetup",
    "PerformanceLogger",
    "ModelLogger",
    "BusinessLogger",
]

# Utility constants
SUPPORTED_ENVIRONMENTS = ['development', 'testing', 'staging', 'production']

DEPLOYMENT_MODES = ['batch', 'real_time', 'scheduled']

LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

DATABASE_TYPES = ['sqlite', 'postgresql', 'mysql']

API_RATE_LIMITS = {
    'steam_api': 100000,  # requests per day
    'prediction_api': 10000  # requests per hour
}

def get_utils_info():
    """Return utilities module information"""
    return {
        "module": "utils",
        "version": "1.0.0",
        "supported_environments": SUPPORTED_ENVIRONMENTS,
        "deployment_modes": DEPLOYMENT_MODES,
        "log_levels": LOG_LEVELS,
        "database_types": DATABASE_TYPES,
        "api_rate_limits": API_RATE_LIMITS,
        "maintainer": "Rushikesh Dhumal"
    }

def setup_project_environment(environment: str = 'development'):
    """Setup project environment with appropriate configurations"""
    from .config import ConfigManager
    from .logger import LoggerSetup
    
    if environment not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(f"Unsupported environment: {environment}")
    
    # Initialize configuration
    config = ConfigManager(environment)
    
    # Setup logging
    logger_setup = LoggerSetup(environment)
    logger = logger_setup.get_logger('main')
    
    logger.info(f"Project environment initialized: {environment}")
    
    return {
        'config': config,
        'logger': logger,
        'environment': environment
    }