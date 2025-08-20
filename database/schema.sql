-- Gaming Player Behavior Analysis & Churn Prediction
-- Database Schema Definition
--
-- Author: Rushikesh Dhumal
-- Email: r.dhumal@rutgers.edu
--
-- This schema supports SQLite, PostgreSQL, and MySQL with minor modifications

-- =============================================================================
-- Drop existing tables (for clean setup)
-- =============================================================================

DROP TABLE IF EXISTS model_predictions;
DROP TABLE IF EXISTS player_sessions;
DROP TABLE IF EXISTS player_purchases;
DROP TABLE IF EXISTS game_ratings;
DROP TABLE IF EXISTS players;
DROP TABLE IF EXISTS games;
DROP TABLE IF EXISTS model_performance;

-- =============================================================================
-- Games Table
-- =============================================================================

CREATE TABLE games (
    game_id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id INTEGER UNIQUE NOT NULL,
    name TEXT NOT NULL,
    genre TEXT,
    release_date DATE,
    price REAL DEFAULT 0.0,
    metacritic_score INTEGER CHECK(metacritic_score >= 0 AND metacritic_score <= 100),
    platforms TEXT,
    is_free BOOLEAN DEFAULT 0,
    developers TEXT,
    publishers TEXT,
    categories TEXT,
    screenshots_count INTEGER DEFAULT 0,
    achievements_count INTEGER DEFAULT 0,
    recommendations_total INTEGER DEFAULT 0,
    positive_ratings INTEGER DEFAULT 0,
    negative_ratings INTEGER DEFAULT 0,
    detailed_description TEXT,
    short_description TEXT,
    website TEXT,
    header_image TEXT,
    background_image TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- Players Table
-- =============================================================================

CREATE TABLE players (
    player_id TEXT PRIMARY KEY,
    registration_date DATE NOT NULL,
    age_group TEXT CHECK(age_group IN ('18-25', '26-35', '36-45', '46+')),
    region TEXT,
    platform_preference TEXT CHECK(platform_preference IN ('PC', 'Mac', 'Linux')),
    favorite_genre TEXT,
    account_level INTEGER DEFAULT 1 CHECK(account_level >= 1),
    total_spent REAL DEFAULT 0.0 CHECK(total_spent >= 0),
    total_playtime_hours REAL DEFAULT 0.0 CHECK(total_playtime_hours >= 0),
    games_owned INTEGER DEFAULT 0 CHECK(games_owned >= 0),
    friends_count INTEGER DEFAULT 0 CHECK(friends_count >= 0),
    achievements_unlocked INTEGER DEFAULT 0 CHECK(achievements_unlocked >= 0),
    forum_posts INTEGER DEFAULT 0 CHECK(forum_posts >= 0),
    reviews_written INTEGER DEFAULT 0 CHECK(reviews_written >= 0),
    last_login DATE,
    avg_session_duration REAL DEFAULT 0.0 CHECK(avg_session_duration >= 0),
    sessions_last_week INTEGER DEFAULT 0 CHECK(sessions_last_week >= 0),
    purchases_last_month INTEGER DEFAULT 0 CHECK(purchases_last_month >= 0),
    days_since_registration INTEGER DEFAULT 0 CHECK(days_since_registration >= 0),
    last_login_days_ago REAL DEFAULT 0.0 CHECK(last_login_days_ago >= 0),
    is_active BOOLEAN DEFAULT 1,
    churned BOOLEAN DEFAULT 0,
    churn_date DATE,
    churn_reason TEXT,
    player_value_score REAL DEFAULT 0.0,
    engagement_score REAL DEFAULT 0.0,
    risk_score REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT chk_churn_date CHECK (
        (churned = 0 AND churn_date IS NULL) OR 
        (churned = 1 AND churn_date IS NOT NULL)
    ),
    CONSTRAINT chk_last_login CHECK (
        last_login <= DATE('now')
    ),
    CONSTRAINT chk_registration CHECK (
        registration_date <= DATE('now')
    )
);

-- =============================================================================
-- Player Sessions Table
-- =============================================================================

CREATE TABLE player_sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT NOT NULL,
    game_id INTEGER,
    session_start TIMESTAMP NOT NULL,
    session_end TIMESTAMP,
    duration_minutes REAL CHECK(duration_minutes >= 0),
    achievements_earned INTEGER DEFAULT 0 CHECK(achievements_earned >= 0),
    in_game_purchases REAL DEFAULT 0.0 CHECK(in_game_purchases >= 0),
    session_type TEXT DEFAULT 'solo' CHECK(session_type IN ('solo', 'multiplayer', 'competitive', 'cooperative')),
    platform TEXT,
    session_quality_score REAL DEFAULT 0.0 CHECK(session_quality_score >= 0),
    disconnections INTEGER DEFAULT 0 CHECK(disconnections >= 0),
    peak_concurrent_players INTEGER,
    server_region TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraints
    FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE SET NULL,
    
    -- Check constraints
    CONSTRAINT chk_session_times CHECK (
        session_end IS NULL OR session_end >= session_start
    ),
    CONSTRAINT chk_duration CHECK (
        duration_minutes IS NULL OR 
        (session_end IS NOT NULL AND duration_minutes > 0)
    )
);

-- =============================================================================
-- Player Purchases Table
-- =============================================================================

CREATE TABLE player_purchases (
    purchase_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT NOT NULL,
    game_id INTEGER,
    purchase_date TIMESTAMP NOT NULL,
    amount REAL NOT NULL CHECK(amount >= 0),
    item_type TEXT DEFAULT 'game' CHECK(item_type IN ('game', 'dlc', 'in_game_item', 'season_pass', 'currency')),
    item_name TEXT,
    currency TEXT DEFAULT 'USD',
    discount_applied REAL DEFAULT 0.0 CHECK(discount_applied >= 0 AND discount_applied <= 100),
    payment_method TEXT,
    refunded BOOLEAN DEFAULT 0,
    refund_date TIMESTAMP,
    refund_reason TEXT,
    transaction_id TEXT UNIQUE,
    promotion_code TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraints
    FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE SET NULL,
    
    -- Check constraints
    CONSTRAINT chk_refund_logic CHECK (
        (refunded = 0 AND refund_date IS NULL) OR 
        (refunded = 1 AND refund_date IS NOT NULL)
    ),
    CONSTRAINT chk_purchase_date CHECK (
        purchase_date <= CURRENT_TIMESTAMP
    )
);

-- =============================================================================
-- Game Ratings Table
-- =============================================================================

CREATE TABLE game_ratings (
    rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT NOT NULL,
    game_id INTEGER NOT NULL,
    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
    review_text TEXT,
    helpful_votes INTEGER DEFAULT 0 CHECK(helpful_votes >= 0),
    funny_votes INTEGER DEFAULT 0 CHECK(funny_votes >= 0),
    playtime_when_reviewed REAL CHECK(playtime_when_reviewed >= 0),
    is_recommended BOOLEAN,
    rating_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    review_language TEXT DEFAULT 'en',
    review_length INTEGER DEFAULT 0,
    contains_spoilers BOOLEAN DEFAULT 0,
    verified_purchase BOOLEAN DEFAULT 0,
    edited_date TIMESTAMP,
    
    -- Foreign key constraints
    FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE,
    
    -- Unique constraint
    UNIQUE(player_id, game_id),
    
    -- Check constraints
    CONSTRAINT chk_rating_date CHECK (
        rating_date <= CURRENT_TIMESTAMP
    ),
    CONSTRAINT chk_review_length CHECK (
        review_length >= 0
    )
);

-- =============================================================================
-- Model Predictions Table
-- =============================================================================

CREATE TABLE model_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT DEFAULT '1.0',
    churn_probability REAL NOT NULL CHECK(churn_probability >= 0 AND churn_probability <= 1),
    churn_prediction INTEGER NOT NULL CHECK(churn_prediction IN (0, 1)),
    risk_level TEXT CHECK(risk_level IN ('Low', 'Medium', 'High', 'Critical')),
    confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feature_values TEXT, -- JSON string of feature values
    feature_importance TEXT, -- JSON string of feature importance
    model_accuracy REAL CHECK(model_accuracy >= 0 AND model_accuracy <= 1),
    prediction_latency_ms REAL DEFAULT 0.0,
    
    -- Intervention tracking
    intervention_recommended BOOLEAN DEFAULT 0,
    intervention_type TEXT,
    intervention_priority INTEGER CHECK(intervention_priority >= 1 AND intervention_priority <= 10),
    intervention_applied BOOLEAN DEFAULT 0,
    intervention_date TIMESTAMP,
    intervention_result TEXT,
    intervention_success BOOLEAN,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    
    -- Foreign key constraints
    FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    
    -- Check constraints
    CONSTRAINT chk_intervention_logic CHECK (
        (intervention_applied = 0 AND intervention_date IS NULL) OR 
        (intervention_applied = 1 AND intervention_date IS NOT NULL)
    ),
    CONSTRAINT chk_prediction_date CHECK (
        prediction_date <= CURRENT_TIMESTAMP
    )
);

-- =============================================================================
-- Model Performance Tracking Table
-- =============================================================================

CREATE TABLE model_performance (
    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT DEFAULT '1.0',
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dataset_name TEXT DEFAULT 'test_set',
    dataset_size INTEGER CHECK(dataset_size > 0),
    
    -- Performance metrics
    accuracy REAL CHECK(accuracy >= 0 AND accuracy <= 1),
    precision_score REAL CHECK(precision_score >= 0 AND precision_score <= 1),
    recall REAL CHECK(recall >= 0 AND recall <= 1),
    f1_score REAL CHECK(f1_score >= 0 AND f1_score <= 1),
    roc_auc REAL CHECK(roc_auc >= 0 AND roc_auc <= 1),
    precision_recall_auc REAL CHECK(precision_recall_auc >= 0 AND precision_recall_auc <= 1),
    
    -- Confusion matrix components
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    true_negatives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0,
    
    -- Training metadata
    training_duration REAL DEFAULT 0.0, -- seconds
    feature_count INTEGER CHECK(feature_count > 0),
    hyperparameters TEXT, -- JSON string
    cross_validation_score REAL,
    cross_validation_std REAL,
    
    -- Business metrics
    cost_benefit_ratio REAL,
    revenue_impact REAL,
    intervention_effectiveness REAL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- Indexes for Performance Optimization
-- =============================================================================

-- Player indexes
CREATE INDEX idx_players_registration ON players(registration_date);
CREATE INDEX idx_players_last_login ON players(last_login);
CREATE INDEX idx_players_churned ON players(churned);
CREATE INDEX idx_players_region ON players(region);
CREATE INDEX idx_players_genre ON players(favorite_genre);
CREATE INDEX idx_players_spending ON players(total_spent);
CREATE INDEX idx_players_playtime ON players(total_playtime_hours);
CREATE INDEX idx_players_value ON players(player_value_score);
CREATE INDEX idx_players_engagement ON players(engagement_score);
CREATE INDEX idx_players_risk ON players(risk_score);

-- Session indexes
CREATE INDEX idx_sessions_player ON player_sessions(player_id);
CREATE INDEX idx_sessions_game ON player_sessions(game_id);
CREATE INDEX idx_sessions_start ON player_sessions(session_start);
CREATE INDEX idx_sessions_duration ON player_sessions(duration_minutes);
CREATE INDEX idx_sessions_type ON player_sessions(session_type);

-- Purchase indexes
CREATE INDEX idx_purchases_player ON player_purchases(player_id);
CREATE INDEX idx_purchases_game ON player_purchases(game_id);
CREATE INDEX idx_purchases_date ON player_purchases(purchase_date);
CREATE INDEX idx_purchases_amount ON player_purchases(amount);
CREATE INDEX idx_purchases_type ON player_purchases(item_type);

-- Rating indexes
CREATE INDEX idx_ratings_player ON game_ratings(player_id);
CREATE INDEX idx_ratings_game ON game_ratings(game_id);
CREATE INDEX idx_ratings_score ON game_ratings(rating);
CREATE INDEX idx_ratings_date ON game_ratings(rating_date);
CREATE INDEX idx_ratings_recommended ON game_ratings(is_recommended);

-- Prediction indexes
CREATE INDEX idx_predictions_player ON model_predictions(player_id);
CREATE INDEX idx_predictions_model ON model_predictions(model_name, model_version);
CREATE INDEX idx_predictions_date ON model_predictions(prediction_date);
CREATE INDEX idx_predictions_risk ON model_predictions(risk_level);
CREATE INDEX idx_predictions_probability ON model_predictions(churn_probability);
CREATE INDEX idx_predictions_intervention ON model_predictions(intervention_recommended);

-- Game indexes
CREATE INDEX idx_games_app_id ON games(app_id);
CREATE INDEX idx_games_genre ON games(genre);
CREATE INDEX idx_games_price ON games(price);
CREATE INDEX idx_games_metacritic ON games(metacritic_score);
CREATE INDEX idx_games_release_date ON games(release_date);
CREATE INDEX idx_games_free ON games(is_free);

-- Performance indexes
CREATE INDEX idx_performance_model ON model_performance(model_name, model_version);
CREATE INDEX idx_performance_date ON model_performance(evaluation_date);
CREATE INDEX idx_performance_accuracy ON model_performance(accuracy);

-- =============================================================================
-- Views for Common Analytics Queries
-- =============================================================================

-- Player activity summary view
CREATE VIEW player_activity_summary AS
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
    p.engagement_score,
    p.risk_score,
    COUNT(DISTINCT s.session_id) as total_sessions,
    AVG(s.duration_minutes) as avg_session_duration,
    MAX(s.session_start) as last_session_date,
    COUNT(DISTINCT pur.purchase_id) as total_purchases,
    SUM(pur.amount) as total_purchase_amount,
    COUNT(DISTINCT r.rating_id) as games_rated,
    AVG(r.rating) as avg_rating_given,
    COUNT(DISTINCT CASE WHEN r.is_recommended = 1 THEN r.game_id END) as games_recommended
FROM players p
LEFT JOIN player_sessions s ON p.player_id = s.player_id
LEFT JOIN player_purchases pur ON p.player_id = pur.player_id AND pur.refunded = 0
LEFT JOIN game_ratings r ON p.player_id = r.player_id
GROUP BY p.player_id;

-- Churn risk analysis view
CREATE VIEW churn_risk_analysis AS
SELECT 
    p.player_id,
    p.churned,
    p.total_spent,
    p.total_playtime_hours,
    p.last_login,
    p.engagement_score,
    p.risk_score,
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
    mp.risk_level as model_risk_level,
    mp.intervention_recommended
FROM players p
LEFT JOIN (
    SELECT player_id, churn_probability, risk_level, intervention_recommended,
           ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY prediction_date DESC) as rn
    FROM model_predictions
) mp ON p.player_id = mp.player_id AND mp.rn = 1;

-- Game popularity and performance view
CREATE VIEW game_popularity AS
SELECT 
    g.game_id,
    g.app_id,
    g.name,
    g.genre,
    g.price,
    g.is_free,
    g.metacritic_score,
    COUNT(DISTINCT s.player_id) as unique_players,
    COUNT(s.session_id) as total_sessions,
    AVG(s.duration_minutes) as avg_session_duration,
    SUM(pur.amount) as total_revenue,
    COUNT(pur.purchase_id) as total_purchases,
    AVG(r.rating) as avg_rating,
    COUNT(r.rating_id) as total_ratings,
    COUNT(CASE WHEN r.is_recommended = 1 THEN 1 END) as positive_recommendations,
    COUNT(CASE WHEN r.is_recommended = 0 THEN 1 END) as negative_recommendations
FROM games g
LEFT JOIN player_sessions s ON g.game_id = s.game_id
LEFT JOIN player_purchases pur ON g.game_id = pur.game_id AND pur.refunded = 0
LEFT JOIN game_ratings r ON g.game_id = r.game_id
GROUP BY g.game_id;

-- Model performance summary view
CREATE VIEW model_performance_summary AS
SELECT 
    model_name,
    model_version,
    COUNT(*) as total_predictions,
    AVG(churn_probability) as avg_churn_probability,
    COUNT(CASE WHEN risk_level = 'Low' THEN 1 END) as low_risk_predictions,
    COUNT(CASE WHEN risk_level = 'Medium' THEN 1 END) as medium_risk_predictions,
    COUNT(CASE WHEN risk_level = 'High' THEN 1 END) as high_risk_predictions,
    COUNT(CASE WHEN risk_level = 'Critical' THEN 1 END) as critical_risk_predictions,
    COUNT(CASE WHEN intervention_recommended = 1 THEN 1 END) as interventions_recommended,
    COUNT(CASE WHEN intervention_applied = 1 THEN 1 END) as interventions_applied,
    AVG(confidence_score) as avg_confidence,
    MAX(prediction_date) as latest_prediction_date,
    MIN(prediction_date) as earliest_prediction_date
FROM model_predictions
GROUP BY model_name, model_version;

-- =============================================================================
-- Triggers for Data Integrity and Automation
-- =============================================================================

-- Update player updated_at timestamp
CREATE TRIGGER update_player_timestamp 
    AFTER UPDATE ON players
    FOR EACH ROW
BEGIN
    UPDATE players 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE player_id = NEW.player_id;
END;

-- Update game updated_at timestamp
CREATE TRIGGER update_game_timestamp 
    AFTER UPDATE ON games
    FOR EACH ROW
BEGIN
    UPDATE games 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE game_id = NEW.game_id;
END;

-- Calculate review length when inserting game ratings
CREATE TRIGGER calculate_review_length 
    BEFORE INSERT ON game_ratings
    FOR EACH ROW
BEGIN
    UPDATE game_ratings 
    SET review_length = LENGTH(COALESCE(NEW.review_text, ''))
    WHERE rating_id = NEW.rating_id;
END;

-- =============================================================================
-- Initial Data Seeding (Optional)
-- =============================================================================

-- Insert sample game genres for reference
INSERT OR IGNORE INTO games (app_id, name, genre, price, is_free, metacritic_score) VALUES
(730, 'Counter-Strike 2', 'Action,FPS,Multiplayer', 0.0, 1, 81),
(440, 'Team Fortress 2', 'Action,FPS,Multiplayer', 0.0, 1, 92),
(570, 'Dota 2', 'MOBA,Strategy,Multiplayer', 0.0, 1, 90),
(578080, 'PUBG: BATTLEGROUNDS', 'Action,Battle Royale,Survival', 0.0, 1, 86),
(271590, 'Grand Theft Auto V', 'Action,Adventure,Crime', 29.99, 0, 97);

-- =============================================================================
-- Database Statistics and Maintenance
-- =============================================================================

-- Enable foreign key constraints (SQLite specific)
PRAGMA foreign_keys = ON;

-- Optimize database
PRAGMA optimize;

-- Analyze tables for query optimization
ANALYZE;