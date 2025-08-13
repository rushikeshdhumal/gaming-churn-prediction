"""
Standalone Steam API Data Collector

This script collects game data from Steam API for the Gaming Churn Prediction project.
No dependencies on existing project code - completely standalone.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu

Usage:
    python standalone_steam_data_collector.py
"""

import requests
import pandas as pd
import time
import json
from datetime import datetime
from typing import List, Dict, Optional

# Configuration
STEAM_API_KEY = "C39A1207334E1D479D532D9D4FF9D222"
BASE_URL = "http://api.steampowered.com/"
STORE_API_URL = "https://store.steampowered.com/api/appdetails"
RATE_LIMIT_DELAY = 1.5  # seconds between requests
OUTPUT_FILE = "steam_games_api_data.csv"

# Popular game app IDs for comprehensive sample
GAME_APP_IDS = [
    # Major titles
    730,    # Counter-Strike 2
    440,    # Team Fortress 2
    570,    # Dota 2
    578080, # PUBG: BATTLEGROUNDS
    271590, # Grand Theft Auto V
    292030, # The Witcher 3: Wild Hunt
    431960, # Wallpaper Engine
    359550, # Tom Clancy's Rainbow Six Siege
    
    # Popular games across genres
    1085660, # Destiny 2
    1174180, # Red Dead Redemption 2
    1151640, # Horizon Zero Dawn
    1245620, # ELDEN RING
    1172470, # Apex Legends
    1237970, # Titanfall 2
    1203220, # NARAKA: BLADEPOINT
    1203630, # Valheim
    
    # Indie and varied genres
    1063730, # New World
    1172620, # Sea of Thieves
    1145360, # Hades
    1086940, # Baldur's Gate 3
    1091500, # Cyberpunk 2077
    1158310, # Crusader Kings III
    1449850, # Yu-Gi-Oh! Master Duel
    1506830, # F1 22
    
    # Free to play
    1269260, # War Thunder
    1599340, # Fall Guys
    1225100, # World of Tanks Blitz
    1276800, # World of Warships
    1180660, # Genshin Impact
    1469040, # Lost Ark
    1824220, # Diablo IV
    1938090, # Call of Duty: Warzone 2.0
    
    # Strategy and simulation
    255710,  # Cities: Skylines
    289070,  # Sid Meier's Civilization VI
    394360,  # Hearts of Iron IV
    236850,  # Europa Universalis IV
    281990,  # Stellaris
    322330,  # Don't Starve Together
    394230,  # Kerbal Space Program
    632360,  # Risk of Rain 2
    
    # Horror and adventure
    323190,  # Frostpunk
    381210,  # Dead by Daylight
    418370,  # Phasmophobia
    367520,  # Hollow Knight
    648800,  # Raft
    435150,  # Divinity: Original Sin 2
    105600,  # Terraria
    427520,  # Factorio
]

class SteamAPICollector:
    """Standalone Steam API data collector"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.collected_data = []
        
    def get_app_list(self) -> List[Dict]:
        """Get complete list of Steam applications"""
        print("Fetching Steam app list...")
        try:
            url = f"{BASE_URL}ISteamApps/GetAppList/v2/"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                apps = data.get('applist', {}).get('apps', [])
                print(f"Found {len(apps)} total Steam applications")
                return apps
            else:
                print(f"Failed to fetch app list: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching app list: {e}")
            return []
    
    def get_game_details(self, app_id: int) -> Optional[Dict]:
        """Get detailed information for a specific game"""
        try:
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
            params = {
                'appids': app_id,
                'format': 'json'
            }
            
            response = self.session.get(STORE_API_URL, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                app_data = data.get(str(app_id), {})
                
                if app_data.get('success') and 'data' in app_data:
                    game_info = app_data['data']
                    
                    # Extract comprehensive game information
                    game_data = {
                        'app_id': app_id,
                        'name': game_info.get('name', ''),
                        'type': game_info.get('type', ''),
                        'is_free': game_info.get('is_free', False),
                        'detailed_description': self._clean_text(game_info.get('detailed_description', '')),
                        'about_the_game': self._clean_text(game_info.get('about_the_game', '')),
                        'short_description': self._clean_text(game_info.get('short_description', '')),
                        'supported_languages': game_info.get('supported_languages', ''),
                        'website': game_info.get('website', ''),
                        'pc_requirements': self._extract_requirements(game_info.get('pc_requirements', {})),
                        'mac_requirements': self._extract_requirements(game_info.get('mac_requirements', {})),
                        'linux_requirements': self._extract_requirements(game_info.get('linux_requirements', {})),
                        'developers': ', '.join(game_info.get('developers', [])),
                        'publishers': ', '.join(game_info.get('publishers', [])),
                        'package_groups': len(game_info.get('package_groups', [])),
                        'platforms': self._extract_platforms(game_info),
                        'genres': self._extract_genres(game_info),
                        'categories': self._extract_categories(game_info),
                        'screenshots_count': len(game_info.get('screenshots', [])),
                        'movies_count': len(game_info.get('movies', [])),
                        'achievements_count': self._extract_achievements_count(game_info),
                        'release_date': self._extract_release_date(game_info),
                        'coming_soon': self._extract_coming_soon(game_info),
                        'price_currency': self._extract_price_currency(game_info),
                        'price_initial': self._extract_price_initial(game_info),
                        'price_final': self._extract_price_final(game_info),
                        'price_discount_percent': self._extract_discount_percent(game_info),
                        'metacritic_score': self._extract_metacritic_score(game_info),
                        'metacritic_url': self._extract_metacritic_url(game_info),
                        'recommendations_total': self._extract_recommendations(game_info),
                        'background': game_info.get('background', ''),
                        'background_raw': game_info.get('background_raw', ''),
                        'content_descriptors': self._extract_content_descriptors(game_info),
                        'header_image': game_info.get('header_image', ''),
                        'capsule_image': game_info.get('capsule_image', ''),
                        'collection_timestamp': datetime.now().isoformat()
                    }
                    
                    return game_data
                else:
                    print(f"  No data available for app_id {app_id}")
                    return None
            else:
                print(f"  HTTP {response.status_code} for app_id {app_id}")
                return None
                
        except Exception as e:
            print(f"  Error fetching app_id {app_id}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean HTML and excessive text from descriptions"""
        if not text:
            return ''
        
        # Remove HTML tags (basic cleaning)
        import re
        clean = re.sub('<.*?>', '', text)
        clean = re.sub(r'\s+', ' ', clean)  # Multiple spaces to single
        clean = clean.strip()
        
        # Limit length for CSV compatibility
        return clean[:500] if len(clean) > 500 else clean
    
    def _extract_requirements(self, req_data) -> str:
        """Extract system requirements"""
        if isinstance(req_data, dict):
            minimum = req_data.get('minimum', '')
            recommended = req_data.get('recommended', '')
            return self._clean_text(f"{minimum} {recommended}")
        return ''
    
    def _extract_platforms(self, game_info: Dict) -> str:
        """Extract supported platforms"""
        platforms = game_info.get('platforms', {})
        supported = []
        if platforms.get('windows'): supported.append('Windows')
        if platforms.get('mac'): supported.append('Mac')
        if platforms.get('linux'): supported.append('Linux')
        return ', '.join(supported)
    
    def _extract_genres(self, game_info: Dict) -> str:
        """Extract game genres"""
        genres = game_info.get('genres', [])
        return ', '.join([genre.get('description', '') for genre in genres])
    
    def _extract_categories(self, game_info: Dict) -> str:
        """Extract game categories"""
        categories = game_info.get('categories', [])
        return ', '.join([cat.get('description', '') for cat in categories])
    
    def _extract_achievements_count(self, game_info: Dict) -> int:
        """Extract number of achievements"""
        achievements = game_info.get('achievements', {})
        return achievements.get('total', 0) if achievements else 0
    
    def _extract_release_date(self, game_info: Dict) -> str:
        """Extract release date"""
        release_date = game_info.get('release_date', {})
        return release_date.get('date', '') if not release_date.get('coming_soon', True) else ''
    
    def _extract_coming_soon(self, game_info: Dict) -> bool:
        """Extract coming soon status"""
        release_date = game_info.get('release_date', {})
        return release_date.get('coming_soon', False)
    
    def _extract_price_currency(self, game_info: Dict) -> str:
        """Extract price currency"""
        price_overview = game_info.get('price_overview', {})
        return price_overview.get('currency', 'USD')
    
    def _extract_price_initial(self, game_info: Dict) -> float:
        """Extract initial price in cents"""
        price_overview = game_info.get('price_overview', {})
        return price_overview.get('initial', 0) / 100.0 if price_overview else 0.0
    
    def _extract_price_final(self, game_info: Dict) -> float:
        """Extract final price in cents"""
        price_overview = game_info.get('price_overview', {})
        return price_overview.get('final', 0) / 100.0 if price_overview else 0.0
    
    def _extract_discount_percent(self, game_info: Dict) -> int:
        """Extract discount percentage"""
        price_overview = game_info.get('price_overview', {})
        return price_overview.get('discount_percent', 0)
    
    def _extract_metacritic_score(self, game_info: Dict) -> Optional[int]:
        """Extract Metacritic score"""
        metacritic = game_info.get('metacritic', {})
        return metacritic.get('score') if metacritic else None
    
    def _extract_metacritic_url(self, game_info: Dict) -> str:
        """Extract Metacritic URL"""
        metacritic = game_info.get('metacritic', {})
        return metacritic.get('url', '') if metacritic else ''
    
    def _extract_recommendations(self, game_info: Dict) -> int:
        """Extract total recommendations"""
        recommendations = game_info.get('recommendations', {})
        return recommendations.get('total', 0) if recommendations else 0
    
    def _extract_content_descriptors(self, game_info: Dict) -> str:
        """Extract content descriptors"""
        descriptors = game_info.get('content_descriptors', {})
        if descriptors:
            notes = descriptors.get('notes', '')
            ids = descriptors.get('ids', [])
            return f"Notes: {notes}, IDs: {', '.join(map(str, ids))}"
        return ''
    
    def collect_games_data(self, app_ids: List[int]) -> List[Dict]:
        """Collect data for multiple games"""
        print(f"Starting collection for {len(app_ids)} games...")
        print(f"Rate limit: {RATE_LIMIT_DELAY} seconds between requests")
        print(f"Estimated time: {len(app_ids) * RATE_LIMIT_DELAY / 60:.1f} minutes")
        print("-" * 60)
        
        collected_games = []
        
        for i, app_id in enumerate(app_ids, 1):
            print(f"[{i:3d}/{len(app_ids)}] Fetching app_id {app_id}...", end=' ')
            
            game_data = self.get_game_details(app_id)
            
            if game_data:
                collected_games.append(game_data)
                print(f"✓ {game_data['name']}")
            else:
                print("✗ Failed")
            
            # Progress update every 10 games
            if i % 10 == 0:
                print(f"\nProgress: {i}/{len(app_ids)} games processed ({len(collected_games)} successful)\n")
        
        print(f"\nCollection complete!")
        print(f"Successfully collected: {len(collected_games)}/{len(app_ids)} games")
        
        return collected_games
    
    def save_to_csv(self, games_data: List[Dict], filename: str):
        """Save collected data to CSV"""
        if not games_data:
            print("No data to save!")
            return
        
        df = pd.DataFrame(games_data)
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"\nData saved to: {filename}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample games:")
        for _, row in df[['name', 'genres', 'price_final', 'metacritic_score']].head().iterrows():
            print(f"  - {row['name']} | {row['genres']} | ${row['price_final']:.2f} | Score: {row['metacritic_score']}")

def main():
    """Main execution function"""
    print("Steam API Data Collector for Gaming Churn Prediction Project")
    print("=" * 65)
    print(f"API Key: {STEAM_API_KEY[:8]}...")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Games to collect: {len(GAME_APP_IDS)}")
    print()
    
    # Initialize collector
    collector = SteamAPICollector(STEAM_API_KEY)
    
    # Collect game data
    games_data = collector.collect_games_data(GAME_APP_IDS)
    
    # Save to CSV
    if games_data:
        collector.save_to_csv(games_data, OUTPUT_FILE)
        
        # Generate summary statistics
        df = pd.DataFrame(games_data)
        print(f"\nSummary Statistics:")
        print(f"- Total games: {len(df)}")
        print(f"- Free games: {df['is_free'].sum()}")
        print(f"- Paid games: {(~df['is_free']).sum()}")
        print(f"- Average price: ${df[df['price_final'] > 0]['price_final'].mean():.2f}")
        print(f"- Games with Metacritic scores: {df['metacritic_score'].notna().sum()}")
        print(f"- Average Metacritic score: {df['metacritic_score'].mean():.1f}")
        print(f"- Unique genres: {len(set(', '.join(df['genres'].fillna('')).split(', ')))}")
        print(f"- Data collection timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    else:
        print("No data collected! Check your API key and internet connection.")

if __name__ == "__main__":
    main()