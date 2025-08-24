"""
Database Management Utilities for Gaming Player Churn Prediction

Comprehensive database operations, validation, and connection management.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from contextlib import contextmanager
import queue
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionPool:
    """
    Database connection pool for efficient connection management
    """
    
    def __init__(self, database_path: str, pool_size: int = 10, 
                 timeout: float = 30.0, max_retries: int = 3):
        """
        Initialize connection pool
        
        Args:
            database_path: Path to SQLite database file
            pool_size: Maximum number of connections in pool
            timeout: Connection timeout in seconds
            max_retries: Maximum connection retry attempts
        """
        self.database_path = database_path
        self.pool_size = pool_size
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._connections = queue.Queue(maxsize=pool_size)
        self._all_connections = []
        self._lock = threading.Lock()
        self._closed = False
        
        # Initialize connection pool
        self._initialize_pool()
        
    def _initialize_pool(self) -> None:
        """Initialize the connection pool"""
        
        logger.info(f"Initializing connection pool with {self.pool_size} connections")
        
        for _ in range(self.pool_size):
            try:
                conn = self._create_connection()
                self._connections.put(conn)
                self._all_connections.append(conn)
            except Exception as e:
                logger.error(f"Failed to create connection: {e}")
                raise
        
        logger.info("Connection pool initialized successfully")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection"""
        
        conn = sqlite3.connect(
            self.database_path,
            timeout=self.timeout,
            check_same_thread=False
        )
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Set row factory for easier data access
        conn.row_factory = sqlite3.Row
        
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn = None
        try:
            # Get connection from pool
            conn = self._connections.get(timeout=self.timeout)
            
            # Test connection
            conn.execute("SELECT 1")
            
            yield conn
            
        except queue.Empty:
            raise TimeoutError("Failed to get connection from pool within timeout")
        except sqlite3.Error as e:
            logger.warning(f"Connection error: {e}. Creating new connection.")
            # Create new connection if current one failed
            if conn:
                try:
                    conn.close()
                except:
                    pass
            conn = self._create_connection()
            yield conn
        finally:
            # Return connection to pool
            if conn:
                try:
                    # Rollback any uncommitted transactions
                    conn.rollback()
                    self._connections.put(conn)
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
                    # Create new connection to replace the failed one
                    try:
                        new_conn = self._create_connection()
                        self._connections.put(new_conn)
                    except Exception as e2:
                        logger.error(f"Failed to create replacement connection: {e2}")
    
    def execute_query(self, query: str, params: Tuple = None) -> List[sqlite3.Row]:
        """Execute a query and return results"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            return cursor.fetchall()
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Execute query with multiple parameter sets"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
    
    def execute_script(self, script: str) -> None:
        """Execute SQL script"""
        
        with self.get_connection() as conn:
            conn.executescript(script)
            conn.commit()
    
    def close(self) -> None:
        """Close all connections in the pool"""
        
        with self._lock:
            if self._closed:
                return
            
            self._closed = True
            
            # Close all connections
            for conn in self._all_connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
            
            # Clear queues and lists
            while not self._connections.empty():
                try:
                    self._connections.get_nowait()
                except queue.Empty:
                    break
            
            self._all_connections.clear()
            
            logger.info("Connection pool closed")
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status"""
        
        return {
            'pool_size': self.pool_size,
            'available_connections': self._connections.qsize(),
            'total_connections': len(self._all_connections),
            'closed': self._closed,
            'database_path': self.database_path
        }


class DatabaseManager:
    """
    Comprehensive database manager for gaming analytics data
    """
    
    def __init__(self, database_path: str = "gaming_analytics.db", 
                 connection_pool: ConnectionPool = None):
        """
        Initialize database manager
        
        Args:
            database_path: Path to SQLite database file
            connection_pool: Optional existing connection pool
        """
        self.database_path = database_path
        
        if connection_pool:
            self.connection_pool = connection_pool
        else:
            self.connection_pool = ConnectionPool(database_path)
        
        self.table_schemas = self._define_table_schemas()
        
        # Initialize database
        self._initialize_database()
        
    def _define_table_schemas(self) -> Dict[str, str]:
        """Define database table schemas"""
        
        return {
            'players': '''
                CREATE TABLE IF NOT EXISTS players (
                    player_id TEXT PRIMARY KEY,
                    registration_date DATE NOT NULL,
                    age_group TEXT,
                    region TEXT,
                    platform_preference TEXT,
                    favorite_genre TEXT,
                    account_level INTEGER DEFAULT 1,
                    total_playtime_hours REAL DEFAULT 0,
                    total_spent REAL DEFAULT 0,
                    friends_count INTEGER DEFAULT 0,
                    achievements_unlocked INTEGER DEFAULT 0,
                    games_owned INTEGER DEFAULT 0,
                    last_login_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            
            'player_sessions': '''
                CREATE TABLE IF NOT EXISTS player_sessions (
                    session_id TEXT PRIMARY KEY,
                    player_id TEXT NOT NULL,
                    session_date DATE NOT NULL,
                    session_duration_minutes REAL NOT NULL,
                    activities_completed INTEGER DEFAULT 0,
                    achievements_earned INTEGER DEFAULT 0,
                    social_interactions INTEGER DEFAULT 0,
                    purchases_made INTEGER DEFAULT 0,
                    performance_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players (player_id)
                )
            ''',
            
            'player_purchases': '''
                CREATE TABLE IF NOT EXISTS player_purchases (
                    purchase_id TEXT PRIMARY KEY,
                    player_id TEXT NOT NULL,
                    purchase_date DATE NOT NULL,
                    amount REAL NOT NULL,
                    item_type TEXT,
                    item_name TEXT,
                    currency TEXT DEFAULT 'USD',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players (player_id)
                )
            ''',
            
            'churn_predictions': '''
                CREATE TABLE IF NOT EXISTS churn_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    player_id TEXT NOT NULL,
                    prediction_date DATE NOT NULL,
                    churn_probability REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    features_used TEXT,
                    intervention_recommended BOOLEAN DEFAULT FALSE,
                    intervention_type TEXT,
                    actual_outcome BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players (player_id)
                )
            ''',
            
            'model_performance': '''
                CREATE TABLE IF NOT EXISTS model_performance (
                    performance_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    evaluation_date DATE NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    roc_auc REAL,
                    dataset_size INTEGER,
                    feature_count INTEGER,
                    training_time_seconds REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            
            'data_quality_metrics': '''
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    metric_id TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    metric_date DATE NOT NULL,
                    total_records INTEGER NOT NULL,
                    missing_values INTEGER DEFAULT 0,
                    duplicate_records INTEGER DEFAULT 0,
                    data_quality_score REAL,
                    validation_errors TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''
        }
    
    def _initialize_database(self) -> None:
        """Initialize database tables"""
        
        logger.info("Initializing database tables")
        
        try:
            # Create tables
            for table_name, schema in self.table_schemas.items():
                self.connection_pool.execute_script(schema)
                logger.info(f"Created/verified table: {table_name}")
            
            # Create indexes for performance
            self._create_indexes()
            
            logger.info("Database initialization completed")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_indexes(self) -> None:
        """Create database indexes for performance"""
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_players_registration_date ON players(registration_date)",
            "CREATE INDEX IF NOT EXISTS idx_players_last_login ON players(last_login_date)",
            "CREATE INDEX IF NOT EXISTS idx_players_region ON players(region)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_player_date ON player_sessions(player_id, session_date)",
            "CREATE INDEX IF NOT EXISTS idx_purchases_player_date ON player_purchases(player_id, purchase_date)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_player_date ON churn_predictions(player_id, prediction_date)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON churn_predictions(risk_level)"
        ]
        
        for index_sql in indexes:
            try:
                self.connection_pool.execute_query(index_sql)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
    
    def insert_player(self, player_data: Dict[str, Any]) -> bool:
        """Insert player data"""
        
        try:
            query = '''
                INSERT OR REPLACE INTO players (
                    player_id, registration_date, age_group, region,
                    platform_preference, favorite_genre, account_level,
                    total_playtime_hours, total_spent, friends_count,
                    achievements_unlocked, games_owned, last_login_date,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            '''
            
            params = (
                player_data.get('player_id'),
                player_data.get('registration_date'),
                player_data.get('age_group'),
                player_data.get('region'),
                player_data.get('platform_preference'),
                player_data.get('favorite_genre'),
                player_data.get('account_level', 1),
                player_data.get('total_playtime_hours', 0),
                player_data.get('total_spent', 0),
                player_data.get('friends_count', 0),
                player_data.get('achievements_unlocked', 0),
                player_data.get('games_owned', 0),
                player_data.get('last_login_date')
            )
            
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert player {player_data.get('player_id')}: {e}")
            return False
    
    def insert_players_batch(self, players_data: List[Dict[str, Any]]) -> int:
        """Insert multiple players in batch"""
        
        try:
            query = '''
                INSERT OR REPLACE INTO players (
                    player_id, registration_date, age_group, region,
                    platform_preference, favorite_genre, account_level,
                    total_playtime_hours, total_spent, friends_count,
                    achievements_unlocked, games_owned, last_login_date,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            '''
            
            params_list = []
            for player_data in players_data:
                params = (
                    player_data.get('player_id'),
                    player_data.get('registration_date'),
                    player_data.get('age_group'),
                    player_data.get('region'),
                    player_data.get('platform_preference'),
                    player_data.get('favorite_genre'),
                    player_data.get('account_level', 1),
                    player_data.get('total_playtime_hours', 0),
                    player_data.get('total_spent', 0),
                    player_data.get('friends_count', 0),
                    player_data.get('achievements_unlocked', 0),
                    player_data.get('games_owned', 0),
                    player_data.get('last_login_date')
                )
                params_list.append(params)
            
            rows_affected = self.connection_pool.execute_many(query, params_list)
            logger.info(f"Inserted {rows_affected} players")
            
            return rows_affected
            
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return 0
    
    def get_player(self, player_id: str) -> Optional[Dict[str, Any]]:
        """Get player by ID"""
        
        try:
            query = "SELECT * FROM players WHERE player_id = ?"
            results = self.connection_pool.execute_query(query, (player_id,))
            
            if results:
                return dict(results[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get player {player_id}: {e}")
            return None
    
    def get_players_by_criteria(self, criteria: Dict[str, Any], 
                               limit: int = 1000) -> List[Dict[str, Any]]:
        """Get players by criteria"""
        
        try:
            query_builder = QueryBuilder()
            query, params = query_builder.build_select_query('players', criteria, limit)
            
            results = self.connection_pool.execute_query(query, params)
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get players by criteria: {e}")
            return []
    
    def insert_churn_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Insert churn prediction"""
        
        try:
            query = '''
                INSERT INTO churn_predictions (
                    prediction_id, player_id, prediction_date, churn_probability,
                    risk_level, model_version, features_used,
                    intervention_recommended, intervention_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                prediction_data.get('prediction_id'),
                prediction_data.get('player_id'),
                prediction_data.get('prediction_date'),
                prediction_data.get('churn_probability'),
                prediction_data.get('risk_level'),
                prediction_data.get('model_version'),
                json.dumps(prediction_data.get('features_used', [])),
                prediction_data.get('intervention_recommended', False),
                prediction_data.get('intervention_type')
            )
            
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert prediction: {e}")
            return False
    
    def get_player_analytics(self, player_id: str) -> Dict[str, Any]:
        """Get comprehensive player analytics"""
        
        try:
            analytics = {}
            
            # Basic player info
            player_info = self.get_player(player_id)
            if not player_info:
                return {}
            
            analytics['player_info'] = player_info
            
            # Session history
            session_query = '''
                SELECT COUNT(*) as session_count,
                       AVG(session_duration_minutes) as avg_duration,
                       SUM(session_duration_minutes) as total_duration,
                       MAX(session_date) as last_session_date
                FROM player_sessions 
                WHERE player_id = ?
            '''
            session_results = self.connection_pool.execute_query(session_query, (player_id,))
            if session_results:
                analytics['session_stats'] = dict(session_results[0])
            
            # Purchase history
            purchase_query = '''
                SELECT COUNT(*) as purchase_count,
                       SUM(amount) as total_spent,
                       AVG(amount) as avg_purchase,
                       MAX(purchase_date) as last_purchase_date
                FROM player_purchases 
                WHERE player_id = ?
            '''
            purchase_results = self.connection_pool.execute_query(purchase_query, (player_id,))
            if purchase_results:
                analytics['purchase_stats'] = dict(purchase_results[0])
            
            # Recent predictions
            prediction_query = '''
                SELECT churn_probability, risk_level, prediction_date, model_version
                FROM churn_predictions 
                WHERE player_id = ? 
                ORDER BY prediction_date DESC 
                LIMIT 5
            '''
            prediction_results = self.connection_pool.execute_query(prediction_query, (player_id,))
            analytics['recent_predictions'] = [dict(row) for row in prediction_results]
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get player analytics for {player_id}: {e}")
            return {}
    
    def get_churn_risk_summary(self) -> Dict[str, Any]:
        """Get churn risk summary across all players"""
        
        try:
            summary_query = '''
                SELECT 
                    risk_level,
                    COUNT(*) as player_count,
                    AVG(churn_probability) as avg_probability
                FROM (
                    SELECT DISTINCT player_id, risk_level, churn_probability
                    FROM churn_predictions 
                    WHERE prediction_date = (
                        SELECT MAX(prediction_date) 
                        FROM churn_predictions p2 
                        WHERE p2.player_id = churn_predictions.player_id
                    )
                ) GROUP BY risk_level
            '''
            
            results = self.connection_pool.execute_query(summary_query)
            
            summary = {
                'risk_distribution': {},
                'total_predictions': 0,
                'overall_avg_probability': 0
            }
            
            total_players = 0
            total_probability = 0
            
            for row in results:
                risk_level = row['risk_level']
                player_count = row['player_count']
                avg_prob = row['avg_probability']
                
                summary['risk_distribution'][risk_level] = {
                    'player_count': player_count,
                    'avg_probability': avg_prob
                }
                
                total_players += player_count
                total_probability += avg_prob * player_count
            
            summary['total_predictions'] = total_players
            summary['overall_avg_probability'] = total_probability / total_players if total_players > 0 else 0
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get churn risk summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data beyond retention period"""
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).date()
            
            cleanup_results = {}
            
            # Clean old sessions
            session_query = "DELETE FROM player_sessions WHERE session_date < ?"
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(session_query, (cutoff_date,))
                cleanup_results['sessions_deleted'] = cursor.rowcount
                conn.commit()
            
            # Clean old predictions (keep latest per player)
            prediction_query = '''
                DELETE FROM churn_predictions 
                WHERE prediction_date < ? 
                AND prediction_id NOT IN (
                    SELECT prediction_id FROM (
                        SELECT prediction_id, 
                               ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY prediction_date DESC) as rn
                        FROM churn_predictions
                    ) WHERE rn = 1
                )
            '''
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(prediction_query, (cutoff_date,))
                cleanup_results['predictions_deleted'] = cursor.rowcount
                conn.commit()
            
            logger.info(f"Cleanup completed: {cleanup_results}")
            
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return {}
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        
        try:
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self.connection_pool.get_connection() as conn:
                backup_conn = sqlite3.connect(str(backup_path))
                conn.backup(backup_conn)
                backup_conn.close()
            
            logger.info(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        try:
            stats = {}
            
            # Table sizes
            for table_name in self.table_schemas.keys():
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                result = self.connection_pool.execute_query(count_query)
                stats[f"{table_name}_count"] = result[0]['count'] if result else 0
            
            # Database file size
            db_path = Path(self.database_path)
            if db_path.exists():
                stats['database_size_mb'] = db_path.stat().st_size / (1024 * 1024)
            
            # Connection pool status
            stats['connection_pool'] = self.connection_pool.get_pool_status()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def close(self) -> None:
        """Close database connections"""
        self.connection_pool.close()


class DataValidator:
    """
    Data validation utilities for database operations
    """
    
    def __init__(self):
        self.validation_rules = self._define_validation_rules()
        
    def _define_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define validation rules for different data types"""
        
        return {
            'player_data': {
                'required_fields': ['player_id', 'registration_date'],
                'field_types': {
                    'player_id': str,
                    'total_playtime_hours': (int, float),
                    'total_spent': (int, float),
                    'friends_count': int,
                    'achievements_unlocked': int,
                    'games_owned': int,
                    'account_level': int
                },
                'field_ranges': {
                    'total_playtime_hours': (0, 10000),
                    'total_spent': (0, 100000),
                    'friends_count': (0, 1000),
                    'achievements_unlocked': (0, 10000),
                    'games_owned': (0, 1000),
                    'account_level': (1, 1000)
                },
                'valid_values': {
                    'age_group': ['18-25', '26-35', '36-45', '46+'],
                    'region': ['NA', 'EU', 'ASIA', 'OTHER'],
                    'platform_preference': ['PC', 'Mac', 'Linux', 'Console'],
                    'favorite_genre': ['Action', 'RPG', 'Strategy', 'Simulation', 'Sports', 'Indie', 'Racing']
                }
            },
            
            'session_data': {
                'required_fields': ['session_id', 'player_id', 'session_date', 'session_duration_minutes'],
                'field_types': {
                    'session_id': str,
                    'player_id': str,
                    'session_duration_minutes': (int, float),
                    'activities_completed': int,
                    'achievements_earned': int,
                    'social_interactions': int,
                    'purchases_made': int,
                    'performance_score': (int, float)
                },
                'field_ranges': {
                    'session_duration_minutes': (1, 1440),  # 1 minute to 24 hours
                    'activities_completed': (0, 1000),
                    'achievements_earned': (0, 100),
                    'social_interactions': (0, 1000),
                    'purchases_made': (0, 50),
                    'performance_score': (0, 100)
                }
            },
            
            'prediction_data': {
                'required_fields': ['prediction_id', 'player_id', 'churn_probability', 'risk_level', 'model_version'],
                'field_types': {
                    'prediction_id': str,
                    'player_id': str,
                    'churn_probability': (int, float),
                    'model_version': str,
                    'intervention_recommended': bool
                },
                'field_ranges': {
                    'churn_probability': (0, 1)
                },
                'valid_values': {
                    'risk_level': ['low', 'medium', 'high', 'critical']
                }
            }
        }
    
    def validate_data(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Validate data against defined rules"""
        
        if data_type not in self.validation_rules:
            return {'valid': False, 'errors': [f'Unknown data type: {data_type}']}
        
        rules = self.validation_rules[data_type]
        errors = []
        warnings = []
        
        # Check required fields
        for field in rules.get('required_fields', []):
            if field not in data or data[field] is None:
                errors.append(f'Missing required field: {field}')
        
        # Check field types
        field_types = rules.get('field_types', {})
        for field, expected_type in field_types.items():
            if field in data and data[field] is not None:
                if not isinstance(data[field], expected_type):
                    errors.append(f'Invalid type for {field}: expected {expected_type}, got {type(data[field])}')
        
        # Check field ranges
        field_ranges = rules.get('field_ranges', {})
        for field, (min_val, max_val) in field_ranges.items():
            if field in data and data[field] is not None:
                try:
                    value = float(data[field])
                    if value < min_val or value > max_val:
                        errors.append(f'Value for {field} out of range: {value} (expected {min_val}-{max_val})')
                except (ValueError, TypeError):
                    errors.append(f'Cannot validate range for {field}: not numeric')
        
        # Check valid values
        valid_values = rules.get('valid_values', {})
        for field, valid_list in valid_values.items():
            if field in data and data[field] is not None:
                if data[field] not in valid_list:
                    errors.append(f'Invalid value for {field}: {data[field]} (expected one of {valid_list})')
        
        # Additional validation checks
        if data_type == 'player_data':
            errors.extend(self._validate_player_specific(data))
        elif data_type == 'session_data':
            errors.extend(self._validate_session_specific(data))
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'data_type': data_type
        }
    
    def _validate_player_specific(self, data: Dict[str, Any]) -> List[str]:
        """Player-specific validation logic"""
        
        errors = []
        
        # Check registration date format
        if 'registration_date' in data:
            try:
                datetime.strptime(str(data['registration_date']), '%Y-%m-%d')
            except ValueError:
                errors.append('Invalid registration_date format (expected YYYY-MM-DD)')
        
        # Check consistency between total_spent and games_owned
        if 'total_spent' in data and 'games_owned' in data:
            if data['total_spent'] > 0 and data['games_owned'] == 0:
                errors.append('Player has spending but no games owned')
        
        # Check achievement rate
        if 'achievements_unlocked' in data and 'total_playtime_hours' in data:
            if data['total_playtime_hours'] > 0:
                achievement_rate = data['achievements_unlocked'] / data['total_playtime_hours']
                if achievement_rate > 10:  # More than 10 achievements per hour seems unrealistic
                    errors.append('Unrealistic achievement rate')
        
        return errors
    
    def _validate_session_specific(self, data: Dict[str, Any]) -> List[str]:
        """Session-specific validation logic"""
        
        errors = []
        
        # Check session date format
        if 'session_date' in data:
            try:
                session_date = datetime.strptime(str(data['session_date']), '%Y-%m-%d').date()
                if session_date > datetime.now().date():
                    errors.append('Session date cannot be in the future')
            except ValueError:
                errors.append('Invalid session_date format (expected YYYY-MM-DD)')
        
        # Check achievements vs session duration
        if 'achievements_earned' in data and 'session_duration_minutes' in data:
            if data['achievements_earned'] > 0 and data['session_duration_minutes'] < 5:
                errors.append('Achievements earned in very short session seems unrealistic')
        
        return errors
    
    def validate_dataset(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Validate entire dataset"""
        
        validation_results = {
            'total_records': len(df),
            'valid_records': 0,
            'invalid_records': 0,
            'errors_by_record': {},
            'common_errors': {},
            'data_quality_score': 0,
            'summary': {}
        }
        
        all_errors = []
        
        for idx, row in df.iterrows():
            record_data = row.to_dict()
            validation = self.validate_data(record_data, data_type)
            
            if validation['valid']:
                validation_results['valid_records'] += 1
            else:
                validation_results['invalid_records'] += 1
                validation_results['errors_by_record'][idx] = validation['errors']
                all_errors.extend(validation['errors'])
        
        # Count common errors
        from collections import Counter
        error_counts = Counter(all_errors)
        validation_results['common_errors'] = dict(error_counts.most_common(10))
        
        # Calculate quality score
        if validation_results['total_records'] > 0:
            validation_results['data_quality_score'] = (
                validation_results['valid_records'] / validation_results['total_records']
            )
        
        # Generate summary
        validation_results['summary'] = {
            'validation_passed': validation_results['invalid_records'] == 0,
            'error_rate': validation_results['invalid_records'] / validation_results['total_records'] if validation_results['total_records'] > 0 else 0,
            'most_common_error': error_counts.most_common(1)[0] if error_counts else None
        }
        
        return validation_results


class QueryBuilder:
    """
    Dynamic SQL query builder for flexible database operations
    """
    
    def __init__(self):
        self.supported_operators = {
            'eq': '=',
            'ne': '!=',
            'gt': '>',
            'gte': '>=',
            'lt': '<',
            'lte': '<=',
            'in': 'IN',
            'like': 'LIKE',
            'between': 'BETWEEN'
        }
    
    def build_select_query(self, table: str, criteria: Dict[str, Any] = None, 
                          limit: int = None, order_by: str = None,
                          columns: List[str] = None) -> Tuple[str, Tuple]:
        """Build SELECT query with criteria"""
        
        # Build columns
        if columns:
            columns_str = ', '.join(columns)
        else:
            columns_str = '*'
        
        query = f"SELECT {columns_str} FROM {table}"
        params = []
        
        # Build WHERE clause
        if criteria:
            where_clause, where_params = self._build_where_clause(criteria)
            if where_clause:
                query += f" WHERE {where_clause}"
                params.extend(where_params)
        
        # Add ORDER BY
        if order_by:
            query += f" ORDER BY {order_by}"
        
        # Add LIMIT
        if limit:
            query += f" LIMIT {limit}"
        
        return query, tuple(params)
    
    def build_update_query(self, table: str, updates: Dict[str, Any], 
                          criteria: Dict[str, Any]) -> Tuple[str, Tuple]:
        """Build UPDATE query"""
        
        if not updates:
            raise ValueError("Updates dictionary cannot be empty")
        
        # Build SET clause
        set_clauses = []
        params = []
        
        for column, value in updates.items():
            set_clauses.append(f"{column} = ?")
            params.append(value)
        
        query = f"UPDATE {table} SET {', '.join(set_clauses)}"
        
        # Build WHERE clause
        if criteria:
            where_clause, where_params = self._build_where_clause(criteria)
            if where_clause:
                query += f" WHERE {where_clause}"
                params.extend(where_params)
        
        return query, tuple(params)
    
    def build_delete_query(self, table: str, criteria: Dict[str, Any]) -> Tuple[str, Tuple]:
        """Build DELETE query"""
        
        if not criteria:
            raise ValueError("DELETE query must have criteria to prevent accidental data loss")
        
        query = f"DELETE FROM {table}"
        params = []
        
        # Build WHERE clause
        where_clause, where_params = self._build_where_clause(criteria)
        if where_clause:
            query += f" WHERE {where_clause}"
            params.extend(where_params)
        
        return query, tuple(params)
    
    def build_insert_query(self, table: str, data: Dict[str, Any]) -> Tuple[str, Tuple]:
        """Build INSERT query"""
        
        if not data:
            raise ValueError("Data dictionary cannot be empty")
        
        columns = list(data.keys())
        placeholders = ['?' for _ in columns]
        params = list(data.values())
        
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        
        return query, tuple(params)
    
    def _build_where_clause(self, criteria: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Build WHERE clause from criteria"""
        
        conditions = []
        params = []
        
        for field, condition in criteria.items():
            if isinstance(condition, dict):
                # Complex condition with operator
                for operator, value in condition.items():
                    if operator in self.supported_operators:
                        sql_operator = self.supported_operators[operator]
                        
                        if operator == 'in':
                            if isinstance(value, (list, tuple)):
                                placeholders = ','.join(['?' for _ in value])
                                conditions.append(f"{field} {sql_operator} ({placeholders})")
                                params.extend(value)
                        elif operator == 'between':
                            if isinstance(value, (list, tuple)) and len(value) == 2:
                                conditions.append(f"{field} {sql_operator} ? AND ?")
                                params.extend(value)
                        elif operator == 'like':
                            conditions.append(f"{field} {sql_operator} ?")
                            params.append(f"%{value}%")
                        else:
                            conditions.append(f"{field} {sql_operator} ?")
                            params.append(value)
            else:
                # Simple equality condition
                conditions.append(f"{field} = ?")
                params.append(condition)
        
        where_clause = ' AND '.join(conditions) if conditions else ''
        
        return where_clause, params
    
    def build_aggregation_query(self, table: str, aggregations: Dict[str, str], 
                               group_by: List[str] = None, 
                               criteria: Dict[str, Any] = None) -> Tuple[str, Tuple]:
        """Build aggregation query"""
        
        # Build aggregation columns
        agg_columns = []
        for alias, expression in aggregations.items():
            agg_columns.append(f"{expression} as {alias}")
        
        # Add group by columns
        if group_by:
            select_columns = group_by + agg_columns
        else:
            select_columns = agg_columns
        
        query = f"SELECT {', '.join(select_columns)} FROM {table}"
        params = []
        
        # Build WHERE clause
        if criteria:
            where_clause, where_params = self._build_where_clause(criteria)
            if where_clause:
                query += f" WHERE {where_clause}"
                params.extend(where_params)
        
        # Add GROUP BY
        if group_by:
            query += f" GROUP BY {', '.join(group_by)}"
        
        return query, tuple(params)
    
    def build_join_query(self, main_table: str, joins: List[Dict[str, str]], 
                        columns: List[str] = None, 
                        criteria: Dict[str, Any] = None) -> Tuple[str, Tuple]:
        """Build JOIN query"""
        
        # Build columns
        if columns:
            columns_str = ', '.join(columns)
        else:
            columns_str = '*'
        
        query = f"SELECT {columns_str} FROM {main_table}"
        params = []
        
        # Add JOINs
        for join in joins:
            join_type = join.get('type', 'INNER')
            join_table = join['table']
            join_condition = join['on']
            
            query += f" {join_type} JOIN {join_table} ON {join_condition}"
        
        # Build WHERE clause
        if criteria:
            where_clause, where_params = self._build_where_clause(criteria)
            if where_clause:
                query += f" WHERE {where_clause}"
                params.extend(where_params)
        
        return query, tuple(params)


def main():
    """Example usage of database utilities"""
    
    logger.info("Testing database utilities...")
    
    # Initialize database manager
    db_manager = DatabaseManager("test_gaming_analytics.db")
    
    # Test data validation
    validator = DataValidator()
    
    sample_player = {
        'player_id': 'test_player_001',
        'registration_date': '2024-01-01',
        'age_group': '26-35',
        'region': 'NA',
        'total_playtime_hours': 150.5,
        'total_spent': 89.99
    }
    
    validation_result = validator.validate_data(sample_player, 'player_data')
    print(f"Validation result: {validation_result}")
    
    # Test query builder
    query_builder = QueryBuilder()
    
    query, params = query_builder.build_select_query(
        'players',
        criteria={'region': 'NA', 'total_spent': {'gt': 50}},
        limit=100
    )
    
    print(f"Generated query: {query}")
    print(f"Parameters: {params}")
    
    # Cleanup
    db_manager.close()
    
    print("Database utilities test completed!")

if __name__ == "__main__":
    main()