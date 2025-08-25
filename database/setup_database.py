"""
Database Setup and Management for Gaming Analytics

This module creates and manages the SQLite database for storing
player behavior data, game information, and model predictions.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import sqlite3
import pandas as pd
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GamingAnalyticsDB:
    """Main database management class for gaming analytics"""
    
    def __init__(self, db_path: str = "gaming_analytics.db"):
        self.db_path = Path(db_path)
        self.connection = None
        
    def connect(self):
        """Create database connection"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            logger.info(f"Connected to database: {self.db_path}")
            return self.connection
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def create_schema(self):
        """Create complete database schema"""
        logger.info("Creating database schema...")
        
        with self.connect() as conn:
            cursor = conn.cursor()
            
            # Drop existing tables (for fresh setup)
            cursor.execute("DROP TABLE IF EXISTS model_predictions")
            cursor.execute("DROP TABLE IF EXISTS player_sessions")
            cursor.execute("DROP TABLE IF EXISTS player_purchases")
            cursor.execute("DROP TABLE IF EXISTS game_ratings")
            cursor.execute("DROP TABLE IF EXISTS players")
            cursor.execute("DROP TABLE IF EXISTS games")
            
            # 1. Games table
            cursor.execute('''
                CREATE TABLE games (
                    game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_id INTEGER UNIQUE,
                    name TEXT NOT NULL,
                    genre TEXT,
                    release_date DATE,
                    price REAL DEFAULT 0.0,
                    metacritic_score INTEGER,
                    platforms TEXT,
                    is_free BOOLEAN DEFAULT 0,
                    developers TEXT,
                    publishers TEXT,
                    categories TEXT,
                    screenshots_count INTEGER DEFAULT 0,
                    achievements_count INTEGER DEFAULT 0,
                    recommendations_total INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 2. Players table
            cursor.execute('''
                CREATE TABLE players (
                    player_id TEXT PRIMARY KEY,
                    registration_date DATE NOT NULL,
                    age_group TEXT,
                    region TEXT,
                    platform_preference TEXT,
                    favorite_genre TEXT,
                    account_level INTEGER DEFAULT 1,
                    total_spent REAL DEFAULT 0.0,
                    total_playtime_hours REAL DEFAULT 0.0,
                    games_owned INTEGER DEFAULT 0,
                    friends_count INTEGER DEFAULT 0,
                    achievements_unlocked INTEGER DEFAULT 0,
                    forum_posts INTEGER DEFAULT 0,
                    reviews_written INTEGER DEFAULT 0,
                    last_login DATE,
                    avg_session_duration REAL DEFAULT 0.0,
                    sessions_last_week INTEGER DEFAULT 0,
                    purchases_last_month INTEGER DEFAULT 0,
                    days_since_registration INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    churned BOOLEAN DEFAULT 0,
                    churn_date DATE,
                    churn_reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 3. Player sessions table
            cursor.execute('''
                CREATE TABLE player_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    game_id INTEGER,
                    session_start TIMESTAMP NOT NULL,
                    session_end TIMESTAMP,
                    duration_minutes REAL,
                    achievements_earned INTEGER DEFAULT 0,
                    in_game_purchases REAL DEFAULT 0.0,
                    session_type TEXT DEFAULT 'solo',
                    platform TEXT,
                    session_quality_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players(player_id),
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                )
            ''')
            
            # 4. Player purchases table
            cursor.execute('''
                CREATE TABLE player_purchases (
                    purchase_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    game_id INTEGER,
                    purchase_date TIMESTAMP NOT NULL,
                    amount REAL NOT NULL,
                    item_type TEXT DEFAULT 'game',
                    item_name TEXT,
                    currency TEXT DEFAULT 'USD',
                    discount_applied REAL DEFAULT 0.0,
                    payment_method TEXT,
                    refunded BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players(player_id),
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                )
            ''')
            
            # 5. Game ratings table
            cursor.execute('''
                CREATE TABLE game_ratings (
                    rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    game_id INTEGER NOT NULL,
                    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                    review_text TEXT,
                    helpful_votes INTEGER DEFAULT 0,
                    funny_votes INTEGER DEFAULT 0,
                    playtime_when_reviewed REAL,
                    is_recommended BOOLEAN,
                    rating_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    review_language TEXT DEFAULT 'en',
                    FOREIGN KEY (player_id) REFERENCES players(player_id),
                    FOREIGN KEY (game_id) REFERENCES games(game_id),
                    UNIQUE(player_id, game_id)
                )
            ''')
            
            # 6. Model predictions table
            cursor.execute('''
                CREATE TABLE model_predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    churn_probability REAL NOT NULL,
                    churn_prediction INTEGER NOT NULL,
                    risk_level TEXT CHECK(risk_level IN ('Low', 'Medium', 'High', 'Critical')),
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    feature_values TEXT,
                    confidence_score REAL,
                    intervention_recommended BOOLEAN DEFAULT 0,
                    intervention_type TEXT,
                    intervention_applied BOOLEAN DEFAULT 0,
                    intervention_result TEXT,
                    model_accuracy REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players(player_id)
                )
            ''')
            
            # 7. Model performance tracking table
            cursor.execute('''
                CREATE TABLE model_performance (
                    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    roc_auc REAL,
                    dataset_size INTEGER,
                    training_duration REAL,
                    feature_count INTEGER,
                    hyperparameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            self._create_indexes(cursor)
            
            # Create views for common queries
            self._create_views(cursor)
            
            conn.commit()
            logger.info("Database schema created successfully")
    
    def _create_indexes(self, cursor):
        """Create database indexes for performance optimization"""
        
        indexes = [
            # Player indexes
            "CREATE INDEX IF NOT EXISTS idx_players_registration ON players(registration_date)",
            "CREATE INDEX IF NOT EXISTS idx_players_last_login ON players(last_login)",
            "CREATE INDEX IF NOT EXISTS idx_players_churned ON players(churned)",
            "CREATE INDEX IF NOT EXISTS idx_players_region ON players(region)",
            "CREATE INDEX IF NOT EXISTS idx_players_genre ON players(favorite_genre)",
            "CREATE INDEX IF NOT EXISTS idx_players_spending ON players(total_spent)",
            
            # Session indexes
            "CREATE INDEX IF NOT EXISTS idx_sessions_player ON player_sessions(player_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_start ON player_sessions(session_start)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_duration ON player_sessions(duration_minutes)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_game ON player_sessions(game_id)",
            
            # Purchase indexes
            "CREATE INDEX IF NOT EXISTS idx_purchases_player ON player_purchases(player_id)",
            "CREATE INDEX IF NOT EXISTS idx_purchases_date ON player_purchases(purchase_date)",
            "CREATE INDEX IF NOT EXISTS idx_purchases_amount ON player_purchases(amount)",
            "CREATE INDEX IF NOT EXISTS idx_purchases_game ON player_purchases(game_id)",
            
            # Prediction indexes
            "CREATE INDEX IF NOT EXISTS idx_predictions_player ON model_predictions(player_id)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_date ON model_predictions(prediction_date)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_model ON model_predictions(model_name)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_risk ON model_predictions(risk_level)",
            
            # Game indexes
            "CREATE INDEX IF NOT EXISTS idx_games_genre ON games(genre)",
            "CREATE INDEX IF NOT EXISTS idx_games_price ON games(price)",
            "CREATE INDEX IF NOT EXISTS idx_games_metacritic ON games(metacritic_score)",
            
            # Rating indexes
            "CREATE INDEX IF NOT EXISTS idx_ratings_player ON game_ratings(player_id)",
            "CREATE INDEX IF NOT EXISTS idx_ratings_game ON game_ratings(game_id)",
            "CREATE INDEX IF NOT EXISTS idx_ratings_score ON game_ratings(rating)",
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        logger.info("Database indexes created")
    
    def _create_views(self, cursor):
        """Create database views for common analytical queries"""
        
        # Player activity summary view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS player_activity_summary AS
            SELECT 
                p.player_id,
                p.registration_date,
                p.churned,
                p.total_spent,
                p.total_playtime_hours,
                p.friends_count,
                p.achievements_unlocked,
                p.favorite_genre,
                p.region,
                COUNT(DISTINCT s.session_id) as total_sessions,
                AVG(s.duration_minutes) as avg_session_duration,
                MAX(s.session_start) as last_session_date,
                COUNT(DISTINCT pur.purchase_id) as total_purchases,
                COUNT(DISTINCT r.rating_id) as games_rated,
                AVG(r.rating) as avg_rating_given
            FROM players p
            LEFT JOIN player_sessions s ON p.player_id = s.player_id
            LEFT JOIN player_purchases pur ON p.player_id = pur.player_id
            LEFT JOIN game_ratings r ON p.player_id = r.player_id
            GROUP BY p.player_id
        ''')
        
        # Churn risk analysis view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS churn_risk_analysis AS
            SELECT 
                p.player_id,
                p.churned,
                p.total_spent,
                p.total_playtime_hours,
                p.last_login,
                CASE 
                    WHEN julianday('now') - julianday(p.last_login) > 30 THEN 'High'
                    WHEN julianday('now') - julianday(p.last_login) > 14 THEN 'Medium'
                    ELSE 'Low'
                END as inactivity_risk,
                CASE 
                    WHEN p.total_spent = 0 THEN 'High'
                    WHEN p.total_spent < 20 THEN 'Medium'
                    ELSE 'Low'
                END as spending_risk,
                CASE 
                    WHEN p.friends_count = 0 THEN 'High'
                    WHEN p.friends_count < 3 THEN 'Medium'
                    ELSE 'Low'
                END as social_risk,
                mp.churn_probability as latest_prediction,
                mp.risk_level as model_risk_level
            FROM players p
            LEFT JOIN (
                SELECT player_id, churn_probability, risk_level,
                       ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY prediction_date DESC) as rn
                FROM model_predictions
            ) mp ON p.player_id = mp.player_id AND mp.rn = 1
        ''')
        
        # Game popularity and performance view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS game_popularity AS
            SELECT 
                g.game_id,
                g.name,
                g.genre,
                g.price,
                g.metacritic_score,
                COUNT(DISTINCT s.player_id) as unique_players,
                COUNT(s.session_id) as total_sessions,
                AVG(s.duration_minutes) as avg_session_duration,
                SUM(pur.amount) as total_revenue,
                COUNT(pur.purchase_id) as total_purchases,
                AVG(r.rating) as avg_rating,
                COUNT(r.rating_id) as total_ratings,
                COUNT(CASE WHEN r.is_recommended = 1 THEN 1 END) as positive_recommendations
            FROM games g
            LEFT JOIN player_sessions s ON g.game_id = s.game_id
            LEFT JOIN player_purchases pur ON g.game_id = pur.game_id
            LEFT JOIN game_ratings r ON g.game_id = r.game_id
            GROUP BY g.game_id
        ''')
        
        # Model performance summary view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS model_performance_summary AS
            SELECT 
                model_name,
                model_version,
                COUNT(*) as total_predictions,
                AVG(churn_probability) as avg_churn_probability,
                COUNT(CASE WHEN risk_level = 'High' THEN 1 END) as high_risk_predictions,
                COUNT(CASE WHEN risk_level = 'Critical' THEN 1 END) as critical_risk_predictions,
                COUNT(CASE WHEN intervention_recommended = 1 THEN 1 END) as interventions_recommended,
                COUNT(CASE WHEN intervention_applied = 1 THEN 1 END) as interventions_applied,
                AVG(confidence_score) as avg_confidence,
                MAX(prediction_date) as latest_prediction_date
            FROM model_predictions
            GROUP BY model_name, model_version
        ''')
        
        logger.info("Database views created")
    
    def insert_player_data(self, df: pd.DataFrame):
        """Insert player data from DataFrame"""
        with self.connect() as conn:
            try:
                # Prepare data for insertion
                player_data = df.copy()
                
                # Handle date columns
                if 'registration_date' in player_data.columns:
                    player_data['registration_date'] = pd.to_datetime(player_data['registration_date']).dt.date
                
                # Calculate last_login from last_login_days_ago if present
                if 'last_login_days_ago' in player_data.columns:
                    player_data['last_login'] = (
                        pd.Timestamp.now().date() - 
                        pd.to_timedelta(player_data['last_login_days_ago'], unit='days')
                    )
                
                # Insert data
                player_data.to_sql('players', conn, if_exists='append', index=False)
                logger.info(f"Inserted {len(player_data)} player records")
                
            except Exception as e:
                logger.error(f"Error inserting player data: {e}")
                raise
    
    def insert_game_data(self, df: pd.DataFrame):
        """Insert game data from DataFrame"""
        with self.connect() as conn:
            try:
                game_data = df.copy()
                
                # Handle date columns
                if 'release_date' in game_data.columns:
                    game_data['release_date'] = pd.to_datetime(game_data['release_date'], errors='coerce').dt.date
                
                game_data.to_sql('games', conn, if_exists='append', index=False)
                logger.info(f"Inserted {len(game_data)} game records")
                
            except Exception as e:
                logger.error(f"Error inserting game data: {e}")
                raise
    
    def insert_prediction_data(self, predictions: List[Dict]):
        """Insert model prediction data"""
        with self.connect() as conn:
            cursor = conn.cursor()
            try:
                for pred in predictions:
                    cursor.execute('''
                        INSERT INTO model_predictions 
                        (player_id, model_name, model_version, churn_probability, 
                         churn_prediction, risk_level, feature_values, confidence_score,
                         intervention_recommended, intervention_type, model_accuracy)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        pred['player_id'],
                        pred['model_name'],
                        pred.get('model_version', '1.0'),
                        pred['churn_probability'],
                        pred['churn_prediction'],
                        pred['risk_level'],
                        json.dumps(pred.get('feature_values', {})),
                        pred.get('confidence_score', 0.0),
                        pred.get('intervention_recommended', 0),
                        pred.get('intervention_type', ''),
                        pred.get('model_accuracy', 0.0)
                    ))
                
                conn.commit()
                logger.info(f"Inserted {len(predictions)} prediction records")
                
            except Exception as e:
                logger.error(f"Error inserting prediction data: {e}")
                raise
    
    def get_player_summary(self, player_id: str) -> Optional[Dict]:
        """Get comprehensive player summary"""
        with self.connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM player_activity_summary 
                WHERE player_id = ?
            ''', (player_id,))
            
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def get_churn_analytics(self) -> pd.DataFrame:
        """Get churn analytics data"""
        with self.connect() as conn:
            return pd.read_sql_query('''
                SELECT 
                    churned,
                    region,
                    favorite_genre,
                    COUNT(*) as player_count,
                    AVG(total_spent) as avg_spending,
                    AVG(total_playtime_hours) as avg_playtime,
                    AVG(friends_count) as avg_friends,
                    AVG(account_level) as avg_level,
                    AVG(achievements_unlocked) as avg_achievements
                FROM players
                GROUP BY churned, region, favorite_genre
                ORDER BY churned, player_count DESC
            ''', conn)
    
    def get_high_risk_players(self, limit: int = 100) -> pd.DataFrame:
        """Get players with high churn risk"""
        with self.connect() as conn:
            return pd.read_sql_query('''
                SELECT 
                    p.player_id,
                    p.total_spent,
                    p.total_playtime_hours,
                    p.last_login,
                    p.friends_count,
                    mp.churn_probability,
                    mp.risk_level,
                    mp.intervention_recommended,
                    mp.prediction_date
                FROM players p
                JOIN model_predictions mp ON p.player_id = mp.player_id
                WHERE mp.risk_level IN ('High', 'Critical') 
                AND p.churned = 0
                AND mp.prediction_date >= date('now', '-7 days')
                ORDER BY mp.churn_probability DESC
                LIMIT ?
            ''', conn, params=(limit,))
    
    def generate_database_stats(self) -> Dict[str, Any]:
        """Generate comprehensive database statistics"""
        with self.connect() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Table row counts
            tables = ['players', 'games', 'player_sessions', 'player_purchases', 
                     'game_ratings', 'model_predictions', 'model_performance']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Player statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_players,
                    SUM(churned) as churned_players,
                    AVG(total_spent) as avg_spending,
                    AVG(total_playtime_hours) as avg_playtime,
                    MIN(registration_date) as first_registration,
                    MAX(registration_date) as latest_registration,
                    COUNT(DISTINCT region) as unique_regions,
                    COUNT(DISTINCT favorite_genre) as unique_genres
                FROM players
            ''')
            player_stats = cursor.fetchone()
            stats.update(dict(player_stats))
            
            # Calculate churn rate
            if stats['total_players'] > 0:
                stats['churn_rate'] = stats['churned_players'] / stats['total_players']
            
            # Recent activity
            cursor.execute('''
                SELECT COUNT(*) FROM player_sessions 
                WHERE session_start >= date('now', '-7 days')
            ''')
            stats['sessions_last_week'] = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM player_purchases 
                WHERE purchase_date >= date('now', '-30 days')
            ''')
            stats['purchases_last_month'] = cursor.fetchone()[0]
            
            # Model prediction statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(churn_probability) as avg_churn_probability,
                    COUNT(CASE WHEN risk_level = 'High' THEN 1 END) as high_risk_count,
                    COUNT(CASE WHEN intervention_recommended = 1 THEN 1 END) as intervention_recommended_count
                FROM model_predictions 
                WHERE prediction_date >= date('now', '-30 days')
            ''')
            prediction_stats = cursor.fetchone()
            if prediction_stats:
                stats.update(dict(prediction_stats))
            
            return stats

class DataValidator:
    """Validate data integrity and consistency"""
    
    def __init__(self, db: GamingAnalyticsDB):
        self.db = db
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Run comprehensive data validation checks"""
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'issues': [],
            'summary': {}
        }
        
        with self.db.connect() as conn:
            cursor = conn.cursor()
            
            # Check 1: Players without any activity
            cursor.execute('''
                SELECT COUNT(*) FROM players p
                LEFT JOIN player_sessions s ON p.player_id = s.player_id
                WHERE s.player_id IS NULL AND p.total_playtime_hours = 0
            ''')
            inactive_players = cursor.fetchone()[0]
            validation_results['checks']['completely_inactive_players'] = inactive_players
            
            if inactive_players > 0:
                validation_results['issues'].append(
                    f"{inactive_players} players have no recorded activity"
                )
            
            # Check 2: Negative values
            cursor.execute('''
                SELECT COUNT(*) FROM players 
                WHERE total_spent < 0 OR total_playtime_hours < 0 OR friends_count < 0
            ''')
            negative_values = cursor.fetchone()[0]
            validation_results['checks']['negative_values'] = negative_values
            
            if negative_values > 0:
                validation_results['issues'].append(
                    f"{negative_values} players have negative values"
                )
            
            # Check 3: Future dates
            cursor.execute('''
                SELECT COUNT(*) FROM players 
                WHERE registration_date > date('now') OR last_login > date('now')
            ''')
            future_dates = cursor.fetchone()[0]
            validation_results['checks']['future_dates'] = future_dates
            
            if future_dates > 0:
                validation_results['issues'].append(
                    f"{future_dates} players have future dates"
                )
            
            # Check 4: Orphaned predictions
            cursor.execute('''
                SELECT COUNT(*) FROM model_predictions mp
                LEFT JOIN players p ON mp.player_id = p.player_id
                WHERE p.player_id IS NULL
            ''')
            orphaned_predictions = cursor.fetchone()[0]
            validation_results['checks']['orphaned_predictions'] = orphaned_predictions
            
            if orphaned_predictions > 0:
                validation_results['issues'].append(
                    f"{orphaned_predictions} predictions reference non-existent players"
                )
            
            # Check 5: Data consistency
            cursor.execute('''
                SELECT COUNT(*) FROM players 
                WHERE churned = 1 AND churn_date IS NULL
            ''')
            inconsistent_churn = cursor.fetchone()[0]
            validation_results['checks']['inconsistent_churn_data'] = inconsistent_churn
            
            if inconsistent_churn > 0:
                validation_results['issues'].append(
                    f"{inconsistent_churn} churned players missing churn date"
                )
            
            # Summary
            validation_results['summary'] = {
                'total_checks': len(validation_results['checks']),
                'issues_found': len(validation_results['issues']),
                'data_quality_score': max(0, 100 - len(validation_results['issues']) * 10)
            }
        
        return validation_results

def setup_database(db_path: str = "gaming_analytics.db", 
                  sample_data: bool = True) -> GamingAnalyticsDB:
    """Complete database setup with optional sample data"""
    logger.info("Setting up Gaming Analytics Database...")
    
    # Initialize database
    db = GamingAnalyticsDB(db_path)
    
    # Create schema
    db.create_schema()
    
    if sample_data:
        logger.info("Inserting sample data...")
        
        # Sample games data
        sample_games = pd.DataFrame({
            'app_id': [730, 440, 570, 578080, 271590],
            'name': ['Counter-Strike 2', 'Team Fortress 2', 'Dota 2', 'PUBG', 'Grand Theft Auto V'],
            'genre': ['Action', 'Action', 'MOBA', 'Battle Royale', 'Action'],
            'price': [0.0, 0.0, 0.0, 29.99, 29.99],
            'is_free': [True, True, True, False, False],
            'metacritic_score': [81, 92, 90, 86, 97],
            'developers': ['Valve', 'Valve', 'Valve', 'PUBG Corporation', 'Rockstar Games'],
            'publishers': ['Valve', 'Valve', 'Valve', 'PUBG Corporation', 'Rockstar Games']
        })
        
        db.insert_game_data(sample_games)
        
        # Generate sample player data
        from src.data.data_collector import SyntheticDataGenerator
        generator = SyntheticDataGenerator()
        sample_players = generator.generate_player_data(1000)
        
        db.insert_player_data(sample_players)
        
        logger.info("Sample data inserted successfully")
    
    # Validate data
    validator = DataValidator(db)
    validation_results = validator.validate_data_integrity()
    
    if validation_results['issues']:
        logger.warning(f"Data validation found {len(validation_results['issues'])} issues")
        for issue in validation_results['issues']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Data validation passed - no issues found")
    
    # Generate stats
    stats = db.generate_database_stats()
    logger.info("Database setup completed successfully!")
    logger.info(f"Statistics: {stats.get('total_players', 0)} players, "
               f"{stats.get('churn_rate', 0):.2%} churn rate, "
               f"${stats.get('avg_spending', 0):.2f} avg spending")
    
    return db

def main():
    """Main function for CLI usage"""
    # Setup database with sample data
    db = setup_database("gaming_analytics.db", sample_data=True)
    
    # Generate database statistics
    stats = db.generate_database_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            if 'rate' in key and isinstance(value, float):
                print(f"  {key}: {value:.2%}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    # Test high-risk players query
    high_risk = db.get_high_risk_players(10)
    if len(high_risk) > 0:
        print(f"\nFound {len(high_risk)} high-risk players")
    
    # Disconnect
    db.disconnect()
    print("\nDatabase setup completed successfully!")

if __name__ == "__main__":
    main()