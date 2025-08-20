"""
Advanced Synthetic Data Generation for Gaming Analytics

This module provides specialized classes for generating realistic synthetic data
for gaming player behavior analysis and churn prediction.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
import random

logger = logging.getLogger(__name__)

class PlayerBehaviorSimulator:
    """
    Simulates detailed player behaviors and interaction patterns
    
    This class focuses on creating realistic behavioral sequences and 
    interaction patterns that mirror real gaming behavior.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the behavior simulator"""
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        
        # Behavior pattern definitions
        self.player_archetypes = {
            'casual': {'weight': 0.40, 'engagement_base': 0.3, 'variance': 0.2},
            'regular': {'weight': 0.35, 'engagement_base': 0.6, 'variance': 0.15},
            'hardcore': {'weight': 0.20, 'engagement_base': 0.85, 'variance': 0.1},
            'whale': {'weight': 0.05, 'engagement_base': 0.9, 'variance': 0.05}
        }
        
        self.session_patterns = {
            'short_burst': {'duration_mean': 15, 'duration_std': 5, 'frequency_high': True},
            'extended': {'duration_mean': 90, 'duration_std': 30, 'frequency_moderate': True},
            'marathon': {'duration_mean': 240, 'duration_std': 60, 'frequency_low': True},
            'irregular': {'duration_mean': 45, 'duration_std': 40, 'frequency_variable': True}
        }
    
    def simulate_player_archetype(self, n_players: int) -> pd.DataFrame:
        """Simulate player archetypes with associated behavior patterns"""
        
        archetypes = []
        weights = [self.player_archetypes[arch]['weight'] for arch in self.player_archetypes.keys()]
        
        for i in range(n_players):
            archetype = np.random.choice(list(self.player_archetypes.keys()), p=weights)
            archetype_data = self.player_archetypes[archetype]
            
            # Base engagement with variance
            base_engagement = archetype_data['engagement_base']
            variance = archetype_data['variance']
            engagement = np.clip(np.random.normal(base_engagement, variance), 0.1, 1.0)
            
            archetypes.append({
                'player_id': f"player_{i:06d}",
                'archetype': archetype,
                'base_engagement': engagement,
                'behavior_consistency': np.random.beta(2, 1) if archetype == 'regular' else np.random.beta(1, 2)
            })
        
        return pd.DataFrame(archetypes)
    
    def simulate_session_patterns(self, player_archetypes: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """Simulate detailed session patterns for each player over time"""
        
        all_sessions = []
        
        for _, player in player_archetypes.iterrows():
            player_id = player['player_id']
            archetype = player['archetype']
            base_engagement = player['base_engagement']
            
            # Determine session pattern based on archetype
            if archetype == 'casual':
                pattern = 'short_burst'
                sessions_per_day = np.random.poisson(1.5)
            elif archetype == 'regular':
                pattern = 'extended'
                sessions_per_day = np.random.poisson(2.5)
            elif archetype == 'hardcore':
                pattern = np.random.choice(['extended', 'marathon'], p=[0.6, 0.4])
                sessions_per_day = np.random.poisson(4.0)
            else:  # whale
                pattern = 'marathon'
                sessions_per_day = np.random.poisson(3.5)
            
            # Generate sessions over time period
            for day in range(days):
                day_engagement = base_engagement * (0.8 + 0.4 * np.random.random())
                
                # Weekly pattern (lower on weekdays for some players)
                weekday = day % 7
                if weekday < 5 and archetype in ['casual', 'regular']:  # weekdays
                    day_engagement *= 0.7
                
                # Generate sessions for this day
                num_sessions = np.random.poisson(sessions_per_day * day_engagement)
                
                for session in range(num_sessions):
                    session_data = self._generate_single_session(
                        player_id, day, session, pattern, archetype, base_engagement
                    )
                    all_sessions.append(session_data)
        
        return pd.DataFrame(all_sessions)
    
    def _generate_single_session(self, player_id: str, day: int, session_num: int, 
                                pattern: str, archetype: str, engagement: float) -> Dict:
        """Generate a single gaming session with realistic attributes"""
        
        pattern_data = self.session_patterns[pattern]
        
        # Session duration
        duration_mean = pattern_data['duration_mean']
        duration_std = pattern_data['duration_std']
        duration = max(5, np.random.normal(duration_mean, duration_std))
        
        # Session activities based on duration and archetype
        activities = self._simulate_session_activities(duration, archetype, engagement)
        
        # Session performance metrics
        performance_score = np.random.beta(2, 2) * engagement * 100
        
        # Social interactions
        if archetype in ['regular', 'hardcore', 'whale']:
            social_interactions = np.random.poisson(duration / 30)  # More interactions in longer sessions
        else:
            social_interactions = np.random.poisson(duration / 60)
        
        return {
            'player_id': player_id,
            'session_date': datetime.now().date() - timedelta(days=30-day),
            'session_number': session_num,
            'duration_minutes': duration,
            'activities_completed': activities['completed'],
            'activities_started': activities['started'],
            'performance_score': performance_score,
            'social_interactions': social_interactions,
            'purchases_made': activities['purchases'],
            'achievements_earned': activities['achievements'],
            'session_pattern': pattern
        }
    
    def _simulate_session_activities(self, duration: float, archetype: str, engagement: float) -> Dict:
        """Simulate activities within a gaming session"""
        
        # Base activity rates per hour
        base_rates = {
            'casual': {'completion_rate': 2.0, 'start_rate': 3.0, 'purchase_rate': 0.1},
            'regular': {'completion_rate': 3.5, 'start_rate': 4.5, 'purchase_rate': 0.3},
            'hardcore': {'completion_rate': 5.0, 'start_rate': 6.0, 'purchase_rate': 0.2},
            'whale': {'completion_rate': 4.0, 'start_rate': 5.0, 'purchase_rate': 1.5}
        }
        
        rates = base_rates[archetype]
        hours = duration / 60
        
        # Calculate activities
        activities_started = np.random.poisson(rates['start_rate'] * hours * engagement)
        completion_rate = min(0.9, 0.4 + engagement * 0.5)
        activities_completed = np.random.binomial(activities_started, completion_rate)
        
        # Purchases (more likely in longer sessions)
        purchase_probability = rates['purchase_rate'] * hours * engagement / 100
        purchases = np.random.poisson(purchase_probability)
        
        # Achievements (based on completion and engagement)
        achievement_rate = activities_completed * 0.1 * engagement
        achievements = np.random.poisson(achievement_rate)
        
        return {
            'started': activities_started,
            'completed': activities_completed,
            'purchases': purchases,
            'achievements': achievements
        }
    
    def simulate_social_behaviors(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Simulate social interactions and network effects"""
        
        social_data = []
        
        for _, player in player_data.iterrows():
            player_id = player['player_id']
            archetype = player.get('archetype', 'regular')
            
            # Friend network simulation
            if archetype == 'casual':
                friends_mean, friends_std = 3, 2
                community_activity = np.random.exponential(0.5)
            elif archetype == 'regular':
                friends_mean, friends_std = 8, 4
                community_activity = np.random.exponential(2.0)
            elif archetype == 'hardcore':
                friends_mean, friends_std = 15, 8
                community_activity = np.random.exponential(5.0)
            else:  # whale
                friends_mean, friends_std = 12, 6
                community_activity = np.random.exponential(3.0)
            
            friends_count = max(0, int(np.random.normal(friends_mean, friends_std)))
            
            # Social activities
            forum_posts = np.random.poisson(community_activity)
            reviews_written = np.random.poisson(community_activity * 0.3)
            guild_participation = np.random.choice([0, 1], p=[0.7, 0.3]) if friends_count > 5 else 0
            
            social_data.append({
                'player_id': player_id,
                'friends_count': friends_count,
                'forum_posts': forum_posts,
                'reviews_written': reviews_written,
                'guild_member': guild_participation,
                'social_engagement_score': (friends_count * 0.4 + forum_posts * 0.3 + 
                                          reviews_written * 0.3) / 10
            })
        
        return pd.DataFrame(social_data)


class RealisticDataGenerator:
    """
    Generates realistic data distributions for various player attributes
    
    This class focuses on creating statistically realistic distributions
    that match real-world gaming industry patterns.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the realistic data generator"""
        np.random.seed(seed)
        self.seed = seed
        
        # Industry-based realistic distributions
        self.spending_distributions = {
            'free_player': {'mean': 0, 'std': 0, 'weight': 0.60},
            'light_spender': {'mean': 15, 'std': 8, 'weight': 0.25},
            'moderate_spender': {'mean': 50, 'std': 20, 'weight': 0.12},
            'heavy_spender': {'mean': 150, 'std': 75, 'weight': 0.025},
            'whale': {'mean': 500, 'std': 300, 'weight': 0.005}
        }
        
        self.playtime_patterns = {
            'casual': {'hours_per_week': 3, 'variance': 2, 'weight': 0.4},
            'regular': {'hours_per_week': 10, 'variance': 5, 'weight': 0.35},
            'enthusiast': {'hours_per_week': 25, 'variance': 10, 'weight': 0.2},
            'hardcore': {'hours_per_week': 50, 'variance': 20, 'weight': 0.05}
        }
    
    def generate_spending_patterns(self, n_players: int, duration_weeks: int = 52) -> pd.DataFrame:
        """Generate realistic spending patterns over time"""
        
        spending_data = []
        
        for i in range(n_players):
            player_id = f"player_{i:06d}"
            
            # Assign spending category
            categories = list(self.spending_distributions.keys())
            weights = [self.spending_distributions[cat]['weight'] for cat in categories]
            spending_category = np.random.choice(categories, p=weights)
            
            category_data = self.spending_distributions[spending_category]
            
            # Generate weekly spending
            weekly_spending = []
            for week in range(duration_weeks):
                if spending_category == 'free_player':
                    # Small chance of making a purchase
                    if np.random.random() < 0.02:  # 2% chance per week
                        weekly_spend = np.random.exponential(5)
                    else:
                        weekly_spend = 0
                else:
                    # Regular spenders with seasonal variations
                    base_spend = max(0, np.random.normal(
                        category_data['mean'] / duration_weeks, 
                        category_data['std'] / duration_weeks
                    ))
                    
                    # Seasonal multiplier (holidays, sales)
                    seasonal_multiplier = self._get_seasonal_multiplier(week)
                    weekly_spend = base_spend * seasonal_multiplier
                
                weekly_spending.append(weekly_spend)
            
            total_spent = sum(weekly_spending)
            spending_volatility = np.std(weekly_spending) / (np.mean(weekly_spending) + 0.01)
            
            spending_data.append({
                'player_id': player_id,
                'spending_category': spending_category,
                'total_spent': total_spent,
                'weekly_spending_pattern': weekly_spending,
                'spending_volatility': spending_volatility,
                'first_purchase_week': next((i for i, x in enumerate(weekly_spending) if x > 0), None),
                'purchase_frequency': sum(1 for x in weekly_spending if x > 0) / duration_weeks
            })
        
        return pd.DataFrame(spending_data)
    
    def generate_playtime_distributions(self, n_players: int, duration_weeks: int = 52) -> pd.DataFrame:
        """Generate realistic playtime patterns"""
        
        playtime_data = []
        
        for i in range(n_players):
            player_id = f"player_{i:06d}"
            
            # Assign playtime pattern
            patterns = list(self.playtime_patterns.keys())
            weights = [self.playtime_patterns[pattern]['weight'] for pattern in patterns]
            playtime_pattern = np.random.choice(patterns, p=weights)
            
            pattern_data = self.playtime_patterns[playtime_pattern]
            
            # Generate weekly playtime with decay/growth trends
            base_hours = pattern_data['hours_per_week']
            variance = pattern_data['variance']
            
            weekly_playtime = []
            engagement_trend = np.random.choice([-0.01, 0, 0.005], p=[0.3, 0.5, 0.2])  # decline, stable, growth
            
            for week in range(duration_weeks):
                # Base hours with trend
                trend_multiplier = 1 + (engagement_trend * week)
                expected_hours = base_hours * max(0.1, trend_multiplier)
                
                # Weekly variation
                week_hours = max(0, np.random.normal(expected_hours, variance))
                
                # Life events (occasional breaks)
                if np.random.random() < 0.05:  # 5% chance of a break week
                    week_hours *= 0.1
                
                weekly_playtime.append(week_hours)
            
            # Calculate metrics
            total_hours = sum(weekly_playtime)
            avg_session_hours = np.random.lognormal(np.log(2), 0.5) if total_hours > 0 else 0
            
            playtime_data.append({
                'player_id': player_id,
                'playtime_pattern': playtime_pattern,
                'total_playtime_hours': total_hours,
                'weekly_playtime_pattern': weekly_playtime,
                'avg_weekly_hours': total_hours / duration_weeks,
                'avg_session_duration': avg_session_hours * 60,  # Convert to minutes
                'playtime_consistency': 1 - (np.std(weekly_playtime) / (np.mean(weekly_playtime) + 0.01)),
                'engagement_trend': engagement_trend
            })
        
        return pd.DataFrame(playtime_data)
    
    def generate_demographic_profiles(self, n_players: int) -> pd.DataFrame:
        """Generate realistic demographic profiles"""
        
        demographic_data = []
        
        # Realistic gaming demographic distributions
        age_groups = ['18-25', '26-35', '36-45', '46+']
        age_weights = [0.35, 0.40, 0.20, 0.05]
        
        regions = ['North America', 'Europe', 'Asia', 'Other']
        region_weights = [0.40, 0.35, 0.20, 0.05]
        
        platforms = ['PC', 'Console', 'Mobile', 'Multiple']
        platform_weights = [0.45, 0.30, 0.15, 0.10]
        
        genres = ['Action', 'RPG', 'Strategy', 'Simulation', 'Sports', 'Indie', 'Racing']
        genre_weights = [0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08]
        
        for i in range(n_players):
            player_id = f"player_{i:06d}"
            
            age_group = np.random.choice(age_groups, p=age_weights)
            region = np.random.choice(regions, p=region_weights)
            primary_platform = np.random.choice(platforms, p=platform_weights)
            favorite_genre = np.random.choice(genres, p=genre_weights)
            
            # Registration date with realistic distribution (more recent players)
            days_ago = np.random.exponential(180)  # Exponential decay
            registration_date = datetime.now() - timedelta(days=min(days_ago, 1095))  # Max 3 years ago
            
            # Account level based on time and engagement
            days_active = (datetime.now() - registration_date).days
            base_level = max(1, int(days_active / 30 * np.random.beta(2, 5)))
            
            demographic_data.append({
                'player_id': player_id,
                'age_group': age_group,
                'region': region,
                'primary_platform': primary_platform,
                'favorite_genre': favorite_genre,
                'registration_date': registration_date.date(),
                'days_since_registration': days_active,
                'account_level': base_level,
                'language_preference': self._assign_language(region)
            })
        
        return pd.DataFrame(demographic_data)
    
    def _get_seasonal_multiplier(self, week: int) -> float:
        """Calculate seasonal spending multiplier"""
        # Holiday seasons and sales events
        if week in [47, 48, 49, 50]:  # Black Friday / Cyber Monday / Christmas
            return np.random.uniform(1.5, 3.0)
        elif week in [1, 2]:  # New Year
            return np.random.uniform(1.2, 1.8)
        elif week in [12, 13, 14]:  # Spring sales
            return np.random.uniform(1.1, 1.4)
        elif week in [26, 27, 28]:  # Summer sales
            return np.random.uniform(1.2, 1.6)
        else:
            return np.random.uniform(0.8, 1.2)
    
    def _assign_language(self, region: str) -> str:
        """Assign language preference based on region"""
        language_map = {
            'North America': np.random.choice(['English', 'Spanish'], p=[0.8, 0.2]),
            'Europe': np.random.choice(['English', 'German', 'French', 'Spanish'], p=[0.4, 0.2, 0.2, 0.2]),
            'Asia': np.random.choice(['English', 'Chinese', 'Japanese', 'Korean'], p=[0.3, 0.3, 0.2, 0.2]),
            'Other': 'English'
        }
        return language_map.get(region, 'English')


class ChurnPatternGenerator:
    """
    Generates realistic churn patterns and risk factors
    
    This class specializes in creating churn labels and risk factors
    based on industry research and behavioral patterns.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the churn pattern generator"""
        np.random.seed(seed)
        self.seed = seed
        
        # Churn risk factors based on gaming industry research
        self.risk_factors = {
            'inactivity': {'weight': 0.4, 'threshold_days': 14},
            'low_engagement': {'weight': 0.25, 'threshold_sessions': 2},
            'no_social_connections': {'weight': 0.15, 'threshold_friends': 1},
            'no_purchases': {'weight': 0.1, 'threshold_spent': 0},
            'poor_performance': {'weight': 0.05, 'threshold_percentile': 20},
            'technical_issues': {'weight': 0.05, 'base_probability': 0.1}
        }
        
        # Cohort-based churn rates
        self.cohort_churn_rates = {
            'week_1': 0.45,   # First week dropout
            'month_1': 0.65,  # Month 1 churn
            'month_3': 0.75,  # Quarter churn
            'month_6': 0.80,  # Semi-annual
            'year_1': 0.85    # Annual churn
        }
    
    def generate_churn_labels(self, player_data: pd.DataFrame, 
                            behavioral_data: pd.DataFrame = None,
                            social_data: pd.DataFrame = None) -> pd.DataFrame:
        """Generate realistic churn labels based on player behavior"""
        
        churn_data = []
        
        for _, player in player_data.iterrows():
            player_id = player['player_id']
            
            # Get behavioral data for this player
            player_behavior = None
            player_social = None
            
            if behavioral_data is not None:
                player_behavior = behavioral_data[behavioral_data['player_id'] == player_id].iloc[0] if len(behavioral_data[behavioral_data['player_id'] == player_id]) > 0 else None
            
            if social_data is not None:
                player_social = social_data[social_data['player_id'] == player_id].iloc[0] if len(social_data[social_data['player_id'] == player_id]) > 0 else None
            
            # Calculate churn probability
            churn_probability = self._calculate_churn_probability(player, player_behavior, player_social)
            
            # Determine churn status
            churned = np.random.binomial(1, churn_probability)
            
            # If churned, determine churn timing
            if churned:
                churn_week = self._determine_churn_timing(player)
                churn_reason = self._determine_churn_reason(player, player_behavior, player_social)
            else:
                churn_week = None
                churn_reason = None
            
            # Risk scores for different factors
            risk_scores = self._calculate_risk_scores(player, player_behavior, player_social)
            
            churn_data.append({
                'player_id': player_id,
                'churned': churned,
                'churn_probability': churn_probability,
                'churn_week': churn_week,
                'churn_reason': churn_reason,
                'risk_score_inactivity': risk_scores['inactivity'],
                'risk_score_engagement': risk_scores['engagement'],
                'risk_score_social': risk_scores['social'],
                'risk_score_monetary': risk_scores['monetary'],
                'risk_score_overall': risk_scores['overall']
            })
        
        return pd.DataFrame(churn_data)
    
    def _calculate_churn_probability(self, player: pd.Series, 
                                   behavior: pd.Series = None, 
                                   social: pd.Series = None) -> float:
        """Calculate churn probability based on multiple factors"""
        
        base_churn_rate = 0.15  # Industry baseline
        
        # Time-based risk (newer players more likely to churn)
        days_active = player.get('days_since_registration', 30)
        if days_active < 7:
            time_risk = 0.3
        elif days_active < 30:
            time_risk = 0.2
        elif days_active < 90:
            time_risk = 0.1
        else:
            time_risk = 0.05
        
        # Behavioral risk factors
        behavior_risk = 0.0
        if behavior is not None:
            # Inactivity risk
            last_login = behavior.get('last_login_days_ago', 0)
            if last_login > 14:
                behavior_risk += 0.25
            elif last_login > 7:
                behavior_risk += 0.1
            
            # Low engagement risk
            avg_session = behavior.get('avg_session_duration', 60)
            if avg_session < 15:
                behavior_risk += 0.15
            
            sessions_week = behavior.get('sessions_last_week', 3)
            if sessions_week == 0:
                behavior_risk += 0.2
            elif sessions_week < 2:
                behavior_risk += 0.05
        
        # Social risk factors
        social_risk = 0.0
        if social is not None:
            friends = social.get('friends_count', 5)
            if friends == 0:
                social_risk += 0.1
            elif friends < 3:
                social_risk += 0.05
        
        # Monetary risk factors
        monetary_risk = 0.0
        total_spent = player.get('total_spent', 0)
        if total_spent == 0:
            monetary_risk += 0.05
        
        # Combine all risk factors
        total_risk = base_churn_rate + time_risk + behavior_risk + social_risk + monetary_risk
        
        # Cap at reasonable maximum
        return min(total_risk, 0.85)
    
    def _determine_churn_timing(self, player: pd.Series) -> int:
        """Determine when player churned (week number)"""
        
        days_active = player.get('days_since_registration', 30)
        weeks_active = days_active // 7
        
        # Early churn is more likely
        if weeks_active < 4:
            # High early churn probability
            churn_week = np.random.geometric(p=0.3)
        elif weeks_active < 12:
            # Medium churn probability
            churn_week = np.random.geometric(p=0.1) + 4
        else:
            # Lower, steady churn probability
            churn_week = np.random.geometric(p=0.05) + 12
        
        return min(churn_week, weeks_active)
    
    def _determine_churn_reason(self, player: pd.Series, 
                               behavior: pd.Series = None, 
                               social: pd.Series = None) -> str:
        """Determine primary reason for churn"""
        
        reasons = []
        
        # Check various risk factors
        if behavior is not None:
            last_login = behavior.get('last_login_days_ago', 0)
            if last_login > 14:
                reasons.append('inactivity')
            
            sessions = behavior.get('sessions_last_week', 3)
            if sessions == 0:
                reasons.append('disengagement')
            
            session_duration = behavior.get('avg_session_duration', 60)
            if session_duration < 15:
                reasons.append('poor_retention')
        
        if social is not None:
            friends = social.get('friends_count', 5)
            if friends == 0:
                reasons.append('social_isolation')
        
        total_spent = player.get('total_spent', 0)
        if total_spent == 0:
            reasons.append('low_investment')
        
        days_active = player.get('days_since_registration', 30)
        if days_active < 7:
            reasons.append('early_abandonment')
        
        # If no specific reason found, assign general reasons
        if not reasons:
            reasons = ['natural_attrition', 'competitor_game', 'life_change', 'game_completion']
        
        # Return primary reason (most common first)
        reason_priority = {
            'inactivity': 1, 'disengagement': 2, 'early_abandonment': 3,
            'social_isolation': 4, 'poor_retention': 5, 'low_investment': 6,
            'natural_attrition': 7, 'competitor_game': 8, 'life_change': 9, 'game_completion': 10
        }
        
        reasons.sort(key=lambda x: reason_priority.get(x, 999))
        return reasons[0]
    
    def _calculate_risk_scores(self, player: pd.Series, 
                              behavior: pd.Series = None, 
                              social: pd.Series = None) -> Dict[str, float]:
        """Calculate individual risk scores for different factors"""
        
        risk_scores = {}
        
        # Inactivity risk
        if behavior is not None:
            last_login = behavior.get('last_login_days_ago', 0)
            risk_scores['inactivity'] = min(1.0, last_login / 30)
        else:
            risk_scores['inactivity'] = 0.2
        
        # Engagement risk
        if behavior is not None:
            sessions = behavior.get('sessions_last_week', 3)
            duration = behavior.get('avg_session_duration', 60)
            engagement_score = (sessions / 10) * (duration / 120)  # Normalize
            risk_scores['engagement'] = max(0.0, 1.0 - engagement_score)
        else:
            risk_scores['engagement'] = 0.3
        
        # Social risk
        if social is not None:
            friends = social.get('friends_count', 5)
            social_score = min(1.0, friends / 20)  # Normalize to 20 friends max
            risk_scores['social'] = 1.0 - social_score
        else:
            risk_scores['social'] = 0.5
        
        # Monetary risk
        total_spent = player.get('total_spent', 0)
        monetary_score = min(1.0, total_spent / 100)  # Normalize to $100
        risk_scores['monetary'] = 1.0 - monetary_score
        
        # Overall risk (weighted average)
        weights = [0.3, 0.3, 0.2, 0.2]  # inactivity, engagement, social, monetary
        risk_scores['overall'] = sum(
            w * risk_scores[factor] for w, factor in 
            zip(weights, ['inactivity', 'engagement', 'social', 'monetary'])
        )
        
        return risk_scores
    
    def generate_retention_interventions(self, churn_data: pd.DataFrame) -> pd.DataFrame:
        """Generate intervention recommendations for at-risk players"""
        
        interventions = []
        
        for _, player in churn_data.iterrows():
            player_id = player['player_id']
            overall_risk = player['risk_score_overall']
            
            # Determine intervention type based on risk profile
            if overall_risk < 0.3:
                intervention = 'monitoring'
                priority = 'low'
            elif overall_risk < 0.6:
                intervention = 'engagement_campaign'
                priority = 'medium'
            else:
                intervention = 'retention_campaign'
                priority = 'high'
            
            # Specific intervention recommendations
            recommendations = []
            
            if player['risk_score_inactivity'] > 0.6:
                recommendations.append('send_reactivation_email')
            
            if player['risk_score_engagement'] > 0.6:
                recommendations.append('offer_new_content')
                recommendations.append('personalized_challenges')
            
            if player['risk_score_social'] > 0.6:
                recommendations.append('friend_recommendations')
                recommendations.append('guild_invitations')
            
            if player['risk_score_monetary'] > 0.6:
                recommendations.append('special_offers')
                recommendations.append('starter_pack_discount')
            
            interventions.append({
                'player_id': player_id,
                'intervention_type': intervention,
                'priority': priority,
                'recommendations': recommendations,
                'estimated_cost': self._calculate_intervention_cost(intervention),
                'expected_success_rate': self._calculate_success_rate(overall_risk, intervention)
            })
        
        return pd.DataFrame(interventions)
    
    def _calculate_intervention_cost(self, intervention_type: str) -> float:
        """Calculate estimated cost for intervention"""
        cost_map = {
            'monitoring': 1.0,
            'engagement_campaign': 10.0,
            'retention_campaign': 25.0
        }
        return cost_map.get(intervention_type, 5.0)
    
    def _calculate_success_rate(self, risk_score: float, intervention_type: str) -> float:
        """Calculate expected success rate for intervention"""
        base_rates = {
            'monitoring': 0.05,
            'engagement_campaign': 0.25,
            'retention_campaign': 0.40
        }
        
        base_rate = base_rates.get(intervention_type, 0.1)
        
        # Higher risk players are harder to retain
        risk_penalty = risk_score * 0.3
        
        return max(0.05, base_rate - risk_penalty)