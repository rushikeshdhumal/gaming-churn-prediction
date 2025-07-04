# Gaming Player Behavior Analysis & Churn Prediction

A comprehensive data science project analyzing gaming player behavior patterns and predicting player churn using machine learning models. This project demonstrates end-to-end data science workflow from data collection to model deployment.

## 🎮 Project Overview

This project analyzes player behavior patterns in gaming to predict player churn, helping game developers and publishers understand what factors lead to player retention or abandonment. The analysis combines real gaming data from Steam with synthetic player behavioral data to create robust predictive models.

### Key Features
- **Multi-source Data Integration**: Steam Web API, Kaggle datasets, and synthetic behavioral data
- **Advanced Analytics**: Statistical hypothesis testing, time-series analysis, and feature engineering
- **Machine Learning Models**: Logistic Regression, Random Forest, and XGBoost with 85%+ accuracy
- **Comprehensive Visualizations**: Interactive plots and dashboards for business insights
- **Production-Ready Code**: Proper error handling, logging, and documentation

## 📊 Dataset Information

### Steam Games Dataset
- **Included Sample**: 8,500 games (23 MB) - `data/raw/steam_games.csv`
- **Original Dataset**: 40,000+ games (319 MB) from Kaggle
- **Sampling Method**: Simple random sampling (random_state=42)
- **Coverage**: Representative sample across all game genres and price ranges

### Game Recommendations Dataset
- **Source**: Kaggle - Game Recommendations on Steam
- **Size**: 40M+ recommendations (download separately)
- **Content**: User recommendations, helpful votes, playtime, review text

### Player Behavior Dataset
- **Generated**: Synthetic data for 10,000+ players
- **Method**: Realistic behavioral patterns with statistical correlations
- **Features**: Engagement metrics, purchase behavior, social interactions

### Data Sources
- **Steam Games**: [Kaggle - Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
- **Game Recommendations**: [Kaggle - Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
- **Player Behavior**: Synthetic data generated by the project

### Quick Start with Sample Data
The repository includes a representative sample that allows you to run the complete analysis without downloading large external files:

```bash
# All analysis works with included sample data
python src/data/data_collector.py
jupyter notebook notebooks/01_data_collection.ipynb
```

### Full Dataset Setup (Optional)
For analysis with complete datasets:

```bash
# Install Kaggle API
pip install kaggle

# Download from Kaggle (requires API setup)
kaggle datasets download -d fronkongames/steam-games-dataset
kaggle datasets download -d antonkozyriev/game-recommendations-on-steam

# Extract to data/raw/ (replace sample files)
unzip steam-games-dataset.zip -d data/raw/
unzip game-recommendations-on-steam.zip -d data/raw/
```

## 📊 Business Impact

- **Player Retention**: Identify at-risk players before they churn
- **Revenue Optimization**: Focus retention efforts on high-value players
- **Game Development**: Data-driven insights for game feature development
- **Marketing Strategy**: Targeted campaigns for different player segments

## 🛠️ Technology Stack

- **Languages**: Python 3.8+
- **Data Storage**: SQLite Database
- **ML Libraries**: scikit-learn, XGBoost
- **Data Analysis**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Development**: Jupyter Notebooks, Google Colab
- **APIs**: Steam Web API (free tier)

## 📁 Project Structure

```
gaming-churn-prediction/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── steam_games.csv (8,500 games sample - 23MB)
│   │   ├── game_recommendations.csv (download separately)
│   │   └── player_data_synthetic.csv (generated)
│   ├── processed/
│   │   ├── cleaned_player_data.csv
│   │   ├── features_engineered.csv
│   │   └── model_ready_data.csv
│   └── DATA_INFO.md
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_model_development.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_business_insights.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_collector.py
│   │   ├── data_processor.py
│   │   └── synthetic_generator.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   └── feature_selection.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   ├── predict_model.py
│   │   └── model_evaluation.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── eda_plots.py
│   │   └── model_plots.py
│   └── utils/
│       ├── __init__.py
│       ├── database.py
│       ├── deployment_utils.py
│       ├── config.py
│       └── logger.py
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── model_metadata.json
├── reports/
│   ├── figures/
│   ├── executive_summary.md
│   └── technical_report.md
├── database/
│   ├── schema.sql
│   ├── setup_database.py
│   └── gaming_analytics.db
└── docs/
    ├── api_documentation.md
    ├── model_documentation.md
    └── deployment_guide.md
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/rushikeshdhumal/gaming-churn-prediction.git
cd gaming-churn-prediction
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package with CLI tools
pip install -e .
```

### 3. Initialize Database
```bash
# Using CLI tool (if installed as package)
gaming-churn-setup-db

# Or run directly
python database/setup_database.py
```

### 4. Generate Synthetic Player Data
```bash
# Using CLI tool
gaming-churn-collect-data

# Or run directly
python src/data/data_collector.py
```

### 5. Configure Steam API (Optional)
```bash
# Get free Steam API key from: https://steamcommunity.com/dev/apikey
export STEAM_API_KEY="your_steam_api_key_here"

# On Windows:
set STEAM_API_KEY=your_steam_api_key_here
```

### 6. Run Complete Analysis
```bash
# Start with data collection and analysis
jupyter notebook notebooks/01_data_collection.ipynb

# Or run the complete notebook
jupyter notebook notebooks/complete_analysis.ipynb
```

### 7. Train Models
```bash
# Using CLI tool
gaming-churn-train

# Or run directly
python src/models/train_model.py
```

### 8. Make Predictions
```bash
# Using CLI tool
gaming-churn-predict

# Or run directly
python src/utils/deployment_utils.py
```

Or run in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rushikeshdhumal/gaming-churn-prediction/blob/main/notebooks/)

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 87.2% | 0.85 | 0.82 | 0.83 | 0.91 |
| Random Forest | 89.5% | 0.88 | 0.87 | 0.87 | 0.94 |
| **XGBoost** | **91.3%** | **0.90** | **0.89** | **0.89** | **0.96** |

## 🔍 Key Findings

### Player Behavior Insights
- **Session Duration**: Players with sessions < 30 minutes have 3x higher churn rate
- **Purchase Pattern**: Players who make purchases within first 7 days have 65% lower churn
- **Social Features**: Players with 5+ friends have 40% better retention
- **Game Variety**: Players trying 3+ different genres show 25% better retention

### Churn Prediction Factors
1. **Days Since Last Login** (Importance: 23.4%)
2. **Average Session Duration** (Importance: 18.7%)
3. **Total Playtime Decrease** (Importance: 15.9%)
4. **Purchase Frequency** (Importance: 14.2%)
5. **Social Interactions** (Importance: 12.8%)

## 📊 Sample Visualizations

![Player Behavior Dashboard](reports/figures/player_behavior_dashboard.png)
![Churn Prediction Model Comparison](reports/figures/model_comparison.png)
![Feature Importance](reports/figures/feature_importance.png)

## 🏢 Business Recommendations

### Immediate Actions (0-30 days)
1. **Early Warning System**: Implement real-time churn prediction for players showing risk signals
2. **Onboarding Optimization**: Focus on first 7-day experience to increase purchase likelihood
3. **Engagement Campaigns**: Target players with decreasing playtime with personalized content

### Strategic Initiatives (30-90 days)
1. **Social Features**: Enhance friend systems and community features
2. **Personalization Engine**: Develop recommendation system for games and content
3. **Retention Programs**: Create loyalty programs for high-value players

### Long-term Strategy (3-12 months)
1. **Predictive Analytics Platform**: Build automated churn prediction system
2. **A/B Testing Framework**: Continuous optimization of retention strategies
3. **Cross-game Analytics**: Analyze player behavior across multiple games

## 🔬 Technical Methodology

### Data Collection
- **Steam Web API**: Game details, player statistics, achievements
- **Kaggle Datasets**: Historical game data and recommendations
- **Synthetic Data**: 10,000+ player behavioral records using realistic distributions

### Feature Engineering
- **Temporal Features**: Rolling averages, time-based aggregations
- **Behavioral Metrics**: Session patterns, purchase behavior, social interactions
- **Game-specific Features**: Genre preferences, difficulty progression
- **Engagement Scores**: Composite metrics for player engagement levels

### Statistical Analysis
- **Hypothesis Testing**: Chi-square tests for categorical relationships
- **Time Series Analysis**: Trend analysis for player behavior over time
- **Correlation Analysis**: Feature relationships and multicollinearity detection
- **Survival Analysis**: Time-to-churn modeling

### Model Development
- **Cross-validation**: 5-fold stratified cross-validation
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Feature Selection**: Recursive Feature Elimination and importance-based selection
- **Model Ensembling**: Voting classifier for improved performance

## 🎯 CLI Tools

After installation with `pip install -e .`, you get these command-line tools:

```bash
gaming-churn-setup-db      # Initialize database
gaming-churn-collect-data  # Generate/collect data
gaming-churn-train         # Train ML models
gaming-churn-predict       # Make predictions
```

## 📚 Documentation

- [Dataset Information](data/DATA_INFO.md)
- [API Documentation](docs/api_documentation.md)
- [Model Documentation](docs/model_documentation.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Executive Summary](reports/executive_summary.md)

## 🧪 Testing and Validation

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/test_models.py
pytest tests/test_data_collection.py
```

## 🐳 Docker Support

```bash
# Build Docker image
docker build -t gaming-churn-prediction .

# Run container
docker run -p 8888:8888 gaming-churn-prediction

# Run with volume mounting
docker run -v $(pwd)/data:/app/data gaming-churn-prediction
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
# Install with development dependencies
pip install -e .[dev]

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black src/
flake8 src/

# Run type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Steam Web API for providing gaming data
- Kaggle community for curated datasets
- Open source libraries that made this project possible
- Gaming industry for inspiring this analysis

## 📞 Contact

**Rushikesh Pandurang Dhumal** - r.dhumal@rutgers.edu

Project Link: [https://github.com/rushikeshdhumal/gaming-churn-prediction](https://github.com/rushikeshdhumal/gaming-churn-prediction)

⭐ **Star this repository if you found it helpful!**
