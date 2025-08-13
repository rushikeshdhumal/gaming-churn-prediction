import os
from dotenv import load_dotenv
from src.data.data_collector import SteamAPICollector

# Load environment variables
load_dotenv()

# Initialize collector
collector = SteamAPICollector()

# Test with popular game IDs
popular_games = [
    730,    # Counter-Strike 2
    440,    # Team Fortress 2  
    570,    # Dota 2
    578080, # PUBG
    271590, # Grand Theft Auto V
    292030, # The Witcher 3
    431960, # Wallpaper Engine
    359550  # Tom Clancy's Rainbow Six Siege
]

print("Testing Steam API connection...")
print(f"API Key configured: {'Yes' if collector.api_key else 'No'}")

# Collect sample data
print("\nFetching game data...")
games_df = collector.get_game_details(popular_games)

print(f"\nCollected data for {len(games_df)} games")
print("\nSample data:")
print(games_df[['name', 'price', 'genres', 'metacritic_score']].head())

# Save to CSV
games_df.to_csv('src/data/raw/steam_api_sample.csv', index=False)
print(f"\nData saved to: src/data/raw/steam_api_sample.csv")