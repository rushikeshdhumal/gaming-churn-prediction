# Dataset Documentation

## Steam Games Sample Dataset

### Overview
- **File**: `steam_games.csv`
- **Records**: 8,500 games
- **Size**: ~23 MB
- **Sampling**: Simple random sample from original 40,000+ games

### Sampling Methodology
```python
import pandas as pd

# Original dataset from Kaggle
original_df = pd.read_csv('steam_games_full.csv')

# Create random sample
sample_df = original_df.sample(n=8500, random_state=42)
sample_df.to_csv('steam_games.csv', index=False)
