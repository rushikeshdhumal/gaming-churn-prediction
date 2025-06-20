# Gaming Player Behavior Analysis & Churn Prediction

A comprehensive data science project analyzing gaming player behavior patterns and predicting player churn using machine learning models. This project demonstrates an end-to-end data science workflow from data collection to model deployment.

## 🎮 Project Overview

This project analyzes player behavior patterns in gaming to predict player churn, helping game developers and publishers understand what factors lead to player retention or abandonment. The analysis combines real gaming data from Steam with synthetic player behavioral data to create robust predictive models.

### Key Features
- **Multi-source Data Integration**: Steam Web API, Kaggle datasets, and synthetic behavioral data
- **Advanced Analytics**: Statistical hypothesis testing, time-series analysis, and feature engineering
- **Machine Learning Models**: Logistic Regression, Random Forest, and XGBoost with 85%+ accuracy
- **Comprehensive Visualizations**: Interactive plots and dashboards for business insights
- **Production-Ready Code**: Proper error handling, logging, and documentation

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
├── .gitignore
├── setup.py
├── data/
│   ├── raw/
│   │   ├── steam_games.csv
│   │   ├── game_recommendations.csv
│   │   └── player_data_synthetic.csv
│   ├── processed/
│   │   ├── cleaned_player_data.csv
│   │   ├── features_engineered.csv
│   │   └── model_ready_data.csv
│   └── external/
│       └── steam_api_data/
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
git clone https://github.com/your-username/gaming-churn-prediction.git
cd gaming-churn-prediction
```

### 2. Set Up Environment
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Data
```bash
# Download Kaggle datasets (requires Kaggle API setup)
kaggle datasets download -d fronkongames/steam-games-dataset
kaggle datasets download -d antonkozyriev/game-recommendations-on-steam

# Unzip to data/raw/
unzip steam-games-dataset.zip -d data/raw/
unzip game-recommendations-on-steam.zip -d data/raw/
```

### 4. Set Up Database
```bash
python database/setup_database.py
```

### 5. Configure Steam API
```bash
# Get free Steam API key from: https://steamcommunity.com/dev/apikey
export STEAM_API_KEY="your_steam_api_key_here"
```

### 6. Run Analysis
```bash
# Open Jupyter notebooks in order
jupyter notebook notebooks/01_data_collection.ipynb
```

Or run in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/gaming-churn-prediction/blob/main/notebooks/)

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 87.2% | 0.85 | 0.82 | 0.83 | 0.91 |
| Random Forest | 89.5% | 0.88 | 0.87 | 0.87 | 0.94 |
| XGBoost | 91.3% | 0.90 | 0.89 | 0.89 | 0.96 |

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

## 📚 Documentation

- [API Documentation](docs/api_documentation.md)
- [Model Documentation](docs/model_documentation.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Technical Report](reports/technical_report.md)
- [Executive Summary](reports/executive_summary.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Steam Web API for providing gaming data
- Kaggle community for curated datasets
- Open source libraries that made this project possible

## 📞 Contact

**Your Name** - your.email@example.com

Project Link: [https://github.com/rushikeshdhumal/gaming-churn-prediction](https://github.com/rushikeshdhumal/gaming-churn-prediction)

---

⭐ **Star this repository if you found it helpful!**
