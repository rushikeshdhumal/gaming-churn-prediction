"""
Setup configuration for Gaming Player Behavior Analysis & Churn Prediction

This setup.py file provides professional package configuration for the
gaming analytics project, making it installable and distributable.
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from requirements.txt
def read_requirements(filename):
    """Read requirements from file and return as list"""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f 
                   if line.strip() and not line.startswith('#')]
    return []

# Project metadata
PROJECT_NAME = "gaming-churn-prediction"
VERSION = "1.0.0"
AUTHOR = "Rushikesh Pandurang Dhumal"
AUTHOR_EMAIL = "r.dhumal@rutgers.edu"
DESCRIPTION = "Advanced machine learning system for predicting player churn in gaming applications"
URL = "https://github.com/rushikeshdhumal/gaming-churn-prediction"
LICENSE = "MIT"

# Python version requirement
PYTHON_REQUIRES = ">=3.8"

# Read requirements
install_requires = read_requirements("requirements.txt")

# Development requirements (optional)
dev_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=0.991",
    "pre-commit>=2.20.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.15.0",
]

# Documentation requirements
docs_requires = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "nbsphinx>=0.8.0",
]

# All extra requirements
extras_require = {
    "dev": dev_requires,
    "docs": docs_requires,
    "all": dev_requires + docs_requires,
}

# Package classifiers for PyPI
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Games/Entertainment",
    "Topic :: Office/Business :: Financial :: Investment",
]

# Keywords for discovery
keywords = [
    "machine learning",
    "churn prediction", 
    "gaming analytics",
    "player behavior",
    "data science",
    "retention analysis",
    "xgboost",
    "random forest",
    "steam api",
    "gaming industry",
    "predictive analytics",
    "customer analytics"
]

# Project URLs for PyPI
project_urls = {
    "Homepage": URL,
    "Bug Reports": f"{URL}/issues",
    "Source": URL,
    "Documentation": f"{URL}#readme",
    "Say Thanks!": "https://saythanks.io/to/your-username",  # Optional
}

# Entry points for command-line tools
entry_points = {
    "console_scripts": [
        "gaming-churn-train=src.models.train_model:main",
        "gaming-churn-predict=src.utils.deployment_utils:main", 
        "gaming-churn-setup-db=database.setup_database:main",
        "gaming-churn-collect-data=src.data.data_collector:main",
    ],
}

# Package data to include
package_data = {
    "gaming_churn_prediction": [
        "data/sample_data/*.csv",
        "database/schema.sql",
        "models/model_metadata.json",
        "reports/templates/*.html",
        "config/*.yaml",
        "config/*.json",
    ],
}

# Data files to include (outside the package)
data_files = [
    ("config", ["database/schema.sql"]),
    ("docs", ["README.md", "LICENSE"]),
]

# Custom commands for setup
class CustomCommands:
    """Custom setup commands for project-specific tasks"""
    
    @staticmethod
    def setup_environment():
        """Set up the development environment"""
        print("Setting up gaming churn prediction environment...")
        # Add any custom setup logic here
        
    @staticmethod
    def download_sample_data():
        """Download sample data for testing"""
        print("Sample data can be generated using: gaming-churn-collect-data")

# Main setup configuration
setup(
    # Basic package information
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    project_urls=project_urls,
    license=LICENSE,
    
    # Package discovery and structure
    packages=find_packages(where=".", exclude=["tests*", "docs*", "examples*"]),
    package_dir={"": "."},
    package_data=package_data,
    data_files=data_files,
    include_package_data=True,
    
    # Dependencies
    python_requires=PYTHON_REQUIRES,
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Metadata for PyPI
    classifiers=classifiers,
    keywords=keywords,
    
    # Entry points
    entry_points=entry_points,
    
    # Additional configuration
    zip_safe=False,  # Don't zip the package
    platforms=["any"],
    
    # Testing configuration
    test_suite="tests",
    tests_require=dev_requires,
    
    # Options for different installation methods
    options={
        "bdist_wheel": {
            "universal": False,  # Pure Python but version-specific
        },
        "egg_info": {
            "tag_build": "",
            "tag_date": False,
        },
    },
    
    # Custom setup hooks
    cmdclass={
        # Add custom commands here if needed
    },
)

# Post-installation message
print("""
🎮 Gaming Player Behavior Analysis & Churn Prediction
=====================================================

Installation completed successfully!

Quick Start:
1. Set up the database:
   gaming-churn-setup-db

2. Generate sample data:
   gaming-churn-collect-data

3. Train models:
   gaming-churn-train

4. Make predictions:
   gaming-churn-predict

Documentation: {url}#readme
Issues: {url}/issues

Happy analyzing! 🚀
""".format(url=URL))

# Additional setup validation
if __name__ == "__main__":
    import sys
    
    # Validate Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Check for critical dependencies
    try:
        import pandas
        import numpy
        import sklearn
        print("✅ Core dependencies validated")
    except ImportError as e:
        print(f"⚠️  Warning: Missing core dependency: {e}")
        print("Run: pip install -r requirements.txt")
    
    print("🎯 Setup validation completed")

# Development helpers
def create_dev_environment():
    """Create development environment with all tools"""
    import subprocess
    import sys
    
    commands = [
        "pip install -e .[dev]",
        "pre-commit install",
        "python -m ipykernel install --user --name gaming-churn-env",
    ]
    
    for cmd in commands:
        try:
            subprocess.check_call(cmd.split())
            print(f"✅ {cmd}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed: {cmd}")

def create_production_environment():
    """Create production environment"""
    import subprocess
    
    commands = [
        "pip install .",
        "gaming-churn-setup-db",
    ]
    
    for cmd in commands:
        try:
            subprocess.check_call(cmd.split())
            print(f"✅ {cmd}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed: {cmd}")

# Environment-specific configurations
ENVIRONMENTS = {
    "development": {
        "debug": True,
        "testing": True,
        "database_url": "sqlite:///gaming_analytics_dev.db",
        "log_level": "DEBUG",
    },
    "testing": {
        "debug": True,
        "testing": True,
        "database_url": "sqlite:///:memory:",
        "log_level": "INFO",
    },
    "production": {
        "debug": False,
        "testing": False,
        "database_url": "sqlite:///gaming_analytics.db",
        "log_level": "WARNING",
    },
}

# Export configuration for external use
__version__ = VERSION
__author__ = AUTHOR
__email__ = AUTHOR_EMAIL
__license__ = LICENSE
__description__ = DESCRIPTION

# Make key functions available at package level
__all__ = [
    "create_dev_environment",
    "create_production_environment", 
    "ENVIRONMENTS",
    "__version__",
]
