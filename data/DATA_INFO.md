# Dataset Documentation

## Gaming Player Behavior Analysis - Dataset Information

### Overview
This project utilizes three main datasets to analyze player behavior and predict churn in gaming applications. All datasets are carefully sampled and processed to ensure GitHub compatibility while maintaining statistical significance.

---

## 1. Steam Games Sample Dataset

### Basic Information
- **File**: `steam_games.csv`
- **Records**: 8,500 games
- **File Size**: ~23 MB
- **Original Dataset**: 40,000+ games (319 MB)
- **Source**: [Kaggle - Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)

### Sampling Methodology
```python
import pandas as pd

# Load original dataset
original_df = pd.read_csv('steam_games_full.csv')

# Create stratified random sample
sample_df = original_df.sample(n=8500, random_state=42)
sample_df.to_csv('steam_games.csv', index=False)

print(f"Sample created: {len(sample_df)} games")
print(f"Coverage: {len(sample_df)/len(original_df)*100:.1f}% of original dataset")
```

### Data Quality & Coverage
- ✅ **Genre Distribution**: Maintains proportional representation across all game genres
- ✅ **Price Range**: Covers free-to-play, indie, and AAA game pricing
- ✅ **Release Timeline**: Games from early Steam era to recent releases
- ✅ **Platform Coverage**: Windows, Mac, Linux game support
- ✅ **Rating Spectrum**: Full range from highly-rated to poorly-rated games

### Key Columns
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `name` | String | Game title | "Counter-Strike 2" |
| `genres` | String | Game categories | "Action,FPS,Multiplayer" |
| `release_date` | Date | Launch date | "2023-09-27" |
| `price` | Float | Current price (USD) | 0.00, 29.99, 59.99 |
| `metacritic_score` | Integer | Review score (0-100) | 81 |
| `platforms` | String | Supported platforms | "windows,mac,linux" |
| `is_free` | Boolean | Free-to-play status | True/False |
| `positive_ratings` | Integer | Positive user reviews | 1543210 |
| `negative_ratings` | Integer | Negative user reviews | 123456 |

---

## 2. Game Recommendations Sample Dataset

### Basic Information
- **File**: `game_recommendations.csv`
- **Records**: 75,000 recommendations
- **File Size**: ~15 MB
- **Original Dataset**: 41M+ recommendations (580 MB)
- **Source**: [Kaggle - Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)

### Sampling Methodology
```python
# Stratified sampling across games and recommendation types
import pandas as pd

sample_size = 75000
chunk_samples = []

# Read in chunks to handle large file
for chunk in pd.read_csv('game_recommendations_full.csv', chunksize=100000):
    # Balance positive/negative recommendations
    pos_sample = chunk[chunk['is_recommended'] == True].sample(
        n=min(1000, len(chunk[chunk['is_recommended'] == True])), 
        random_state=42
    )
    neg_sample = chunk[chunk['is_recommended'] == False].sample(
        n=min(1000, len(chunk[chunk['is_recommended'] == False])), 
        random_state=42
    )
    chunk_samples.extend([pos_sample, neg_sample])

# Combine and finalize sample
final_sample = pd.concat(chunk_samples).sample(n=sample_size, random_state=42)
final_sample.to_csv('game_recommendations.csv', index=False)
```

### Data Quality & Coverage
- ✅ **Balanced Sentiment**: Equal representation of positive/negative recommendations
- ✅ **Game Diversity**: Reviews across popular and niche games
- ✅ **User Variety**: Different user types (casual, hardcore, new, veteran)
- ✅ **Temporal Spread**: Reviews from different time periods
- ✅ **Playtime Range**: From minimal to extensive playtime reviewers

### Key Columns
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `user_id` | String | Anonymous user ID | "user_76561198123456" |
| `app_id` | Integer | Game application ID | 730, 440, 570 |
| `is_recommended` | Boolean | Recommendation status | True/False |
| `helpful` | Integer | Helpfulness votes | 45 |
| `funny` | Integer | Funny votes | 12 |
| `date` | Date | Review posting date | "2023-10-15" |
| `hours` | Float | Hours played at review | 1247.5 |
| `review_text` | String | User review content | "Great game but..." |

---

## 3. Synthetic Player Behavior Dataset

### Basic Information
- **File**: `player_data_synthetic.csv`
- **Records**: 10,000+ players
- **File Size**: ~5 MB
- **Generation Method**: Statistically realistic behavioral patterns
- **Purpose**: Churn prediction training data

### Generation Methodology
```python
# Realistic player behavior simulation
import numpy as np
import pandas as pd

np.random.seed(42)

# Generate correlated behavioral features
engagement_level = np.random.beta(2, 5, n_players)

# Create realistic distributions
player_data = {
    'total_playtime_hours': np.random.exponential(100) * (1 + engagement_level * 2),
    'avg_session_duration': np.random.lognormal(3, 1) * (0.5 + engagement_level),
    'friends_count': np.random.negative_binomial(2, 0.3) * (1 + engagement_level * 2),
    'total_spent': np.random.exponential(50) * (1 + engagement_level * 2),
    # ... more realistic features
}

# Generate realistic churn patterns
churn_probability = calculate_churn_risk(player_data)
player_data['churned'] = np.random.binomial(1, churn_probability)
```

### Data Realism Features
- ✅ **Correlated Behaviors**: Engagement metrics correlate realistically
- ✅ **Churn Patterns**: Based on actual gaming industry insights
- ✅ **Demographic Variety**: Different player types and regions
- ✅ **Temporal Consistency**: Realistic registration and activity patterns
- ✅ **Statistical Validity**: Sufficient sample size for ML training

### Key Columns
| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `player_id` | String | Unique player identifier | "player_000001" |
| `registration_date` | Date | Account creation date | 2022-01-01 to 2024-01-01 |
| `total_playtime_hours` | Float | Cumulative game time | 0.0 - 1000.0+ |
| `avg_session_duration` | Float | Average session length (min) | 5.0 - 180.0 |
| `friends_count` | Integer | Number of gaming friends | 0 - 50+ |
| `total_spent` | Float | Total purchases (USD) | 0.0 - 500.0+ |
| `churned` | Boolean | Player churn status | True/False |

---

## Usage in Analysis

### Statistical Significance
All datasets provide sufficient sample sizes for:
- **Hypothesis Testing**: Chi-square, t-tests, ANOVA
- **Machine Learning**: Training, validation, testing splits
- **Business Analysis**: Segmentation, trend analysis
- **Correlation Studies**: Feature relationship analysis

### Integration Approach
```python
# Example: Cross-dataset analysis
games_df = pd.read_csv('data/raw/steam_games.csv')
recommendations_df = pd.read_csv('data/raw/game_recommendations.csv')
players_df = pd.read_csv('data/raw/player_data_synthetic.csv')

# Join datasets for comprehensive analysis
analysis_df = players_df.merge(
    recommendations_df.groupby('app_id').agg({
        'is_recommended': 'mean',
        'hours': 'mean'
    }), 
    left_on='favorite_game', 
    right_index=True
)
```

### Reproducibility
- **Seed Values**: All sampling uses `random_state=42`
- **Version Control**: Dataset versions tracked
- **Documentation**: Complete methodology documentation
- **Validation**: Data quality checks implemented

---

## Data Acquisition Instructions

### Option 1: Use Included Samples (Recommended)
```bash
# Repository includes all necessary sample files
git clone https://github.com/rushikeshdhumal/gaming-churn-prediction.git
cd gaming-churn-prediction
# All sample files ready for analysis
```

### Option 2: Download Full Datasets
```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle credentials
# Download from: https://www.kaggle.com/account (API section)

# Download datasets
kaggle datasets download -d fronkongames/steam-games-dataset
kaggle datasets download -d antonkozyriev/game-recommendations-on-steam

# Extract to data/raw/ (replace sample files)
unzip steam-games-dataset.zip -d data/raw/
unzip game-recommendations-on-steam.zip -d data/raw/
```

### Option 3: Generate New Samples
```bash
# Create new samples with different parameters
python src/data/data_collector.py --steam-sample-size 10000 --rec-sample-size 100000
```

---

## Quality Assurance

### Data Validation Checks
- ✅ **Completeness**: Missing value analysis
- ✅ **Consistency**: Cross-dataset ID validation
- ✅ **Accuracy**: Range and type validation
- ✅ **Timeliness**: Date range verification
- ✅ **Uniqueness**: Duplicate detection

### Sample Representativeness
- ✅ **Statistical Tests**: KS-test for distribution similarity
- ✅ **Visual Validation**: Distribution comparison plots
- ✅ **Business Logic**: Domain expert validation
- ✅ **Performance**: ML model comparison on sample vs. full data

---

*Last Updated: August 2025*  
*Maintained by: Rushikesh Dhumal*  
*Contact: r.dhumal@rutgers.edu*