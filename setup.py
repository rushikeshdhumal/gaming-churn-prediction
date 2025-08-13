"""
Setup configuration for Gaming Player Behavior Analysis & Churn Prediction
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

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
AUTHOR = "Rushikesh Dhumal"
AUTHOR_EMAIL = "r.dhumal@rutgers.edu"
DESCRIPTION = "Advanced machine learning system for predicting player churn in gaming applications"
URL = "https://github.com/rushikeshdhumal/gaming-churn-prediction"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.8"

install_requires = read_requirements("requirements.txt")

dev_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=0.991",
    "pre-commit>=2.20.0",
]

docs_requires = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "nbsphinx>=0.8.0",
]

extras_require = {
    "dev": dev_requires,
    "docs": docs_requires,
    "all": dev_requires + docs_requires,
}

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
]

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

project_urls = {
    "Homepage": URL,
    "Bug Reports": f"{URL}/issues",
    "Source": URL,
    "Documentation": f"{URL}#readme",
}

entry_points = {
    "console_scripts": [
        "gaming-churn-train=src.models.train_model:main",
        "gaming-churn-predict=src.utils.deployment_utils:main", 
        "gaming-churn-setup-db=database.setup_database:main",
        "gaming-churn-collect-data=src.data.data_collector:main",
    ],
}

package_data = {
    "gaming_churn_prediction": [
        "data/raw/steam_games.csv",
        "data/raw/game_recommendations.csv",
        "database/schema.sql",
        "models/model_metadata.json",
        "config/*.yaml",
        "config/*.json",
    ],
}

setup(
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
    
    packages=find_packages(where=".", exclude=["tests*", "docs*", "examples*"]),
    package_dir={"": "."},
    package_data=package_data,
    include_package_data=True,
    
    python_requires=PYTHON_REQUIRES,
    install_requires=install_requires,
    extras_require=extras_require,
    
    classifiers=classifiers,
    keywords=keywords,
    
    entry_points=entry_points,
    
    zip_safe=False,
    platforms=["any"],
    
    test_suite="tests",
    tests_require=dev_requires,
    
    options={
        "bdist_wheel": {
            "universal": False,
        },
    },
)

if __name__ == "__main__":
    import sys
    
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        sys.exit(1)
    
    try:
        import pandas
        import numpy
        import sklearn
        print("✅ Core dependencies validated")
    except ImportError as e:
        print(f"⚠️  Warning: Missing core dependency: {e}")

print("""
🎮 Gaming Player Behavior Analysis & Churn Prediction
=====================================================
Installation completed successfully!

Quick Start:
1. gaming-churn-setup-db
2. gaming-churn-collect-data  
3. gaming-churn-train
4. gaming-churn-predict

Happy analyzing! 🚀
""")

__version__ = VERSION
__author__ = AUTHOR
__email__ = AUTHOR_EMAIL
__license__ = LICENSE