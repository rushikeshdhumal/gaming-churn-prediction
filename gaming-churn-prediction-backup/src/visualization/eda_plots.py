"""
Exploratory Data Analysis Visualization Module

Comprehensive visualization tools for player behavior analysis and churn exploration.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PlayerBehaviorVisualizer:
    """
    Visualize player behavior patterns and engagement metrics
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def plot_playtime_distribution(self, df: pd.DataFrame, interactive: bool = True) -> Any:
        """Plot playtime distribution across players"""
        
        if interactive:
            fig = px.histogram(
                df, x='total_playtime_hours',
                title='Player Playtime Distribution',
                labels={'total_playtime_hours': 'Total Playtime (Hours)', 'count': 'Number of Players'},
                marginal='box'
            )
            fig.update_layout(height=500)
            return fig
        else:
            plt.figure(figsize=self.figsize)
            plt.hist(df['total_playtime_hours'], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Total Playtime (Hours)')
            plt.ylabel('Number of Players')
            plt.title('Player Playtime Distribution')
            plt.grid(True, alpha=0.3)
            return plt.gcf()
    
    def plot_spending_patterns(self, df: pd.DataFrame, interactive: bool = True) -> Any:
        """Analyze player spending patterns"""
        
        if interactive:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Spending Distribution', 'Spending vs Playtime', 
                               'Spending by Age Group', 'Spending by Region'],
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}]]
            )
            
            # Spending distribution
            fig.add_trace(
                go.Histogram(x=df['total_spent'], name='Spending Distribution'),
                row=1, col=1
            )
            
            # Spending vs Playtime
            fig.add_trace(
                go.Scatter(
                    x=df['total_playtime_hours'], 
                    y=df['total_spent'],
                    mode='markers',
                    name='Spending vs Playtime',
                    opacity=0.6
                ),
                row=1, col=2
            )
            
            # Spending by Age Group
            if 'age_group' in df.columns:
                spending_by_age = df.groupby('age_group')['total_spent'].mean().reset_index()
                fig.add_trace(
                    go.Bar(x=spending_by_age['age_group'], y=spending_by_age['total_spent'], 
                           name='Avg Spending by Age'),
                    row=2, col=1
                )
            
            # Spending by Region
            if 'region' in df.columns:
                spending_by_region = df.groupby('region')['total_spent'].mean().reset_index()
                fig.add_trace(
                    go.Bar(x=spending_by_region['region'], y=spending_by_region['total_spent'],
                           name='Avg Spending by Region'),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, title_text="Player Spending Analysis")
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Spending distribution
            axes[0, 0].hist(df['total_spent'], bins=50, alpha=0.7)
            axes[0, 0].set_xlabel('Total Spent ($)')
            axes[0, 0].set_ylabel('Number of Players')
            axes[0, 0].set_title('Spending Distribution')
            
            # Spending vs Playtime
            axes[0, 1].scatter(df['total_playtime_hours'], df['total_spent'], alpha=0.6)
            axes[0, 1].set_xlabel('Total Playtime (Hours)')
            axes[0, 1].set_ylabel('Total Spent ($)')
            axes[0, 1].set_title('Spending vs Playtime')
            
            # Spending by Age Group
            if 'age_group' in df.columns:
                df.groupby('age_group')['total_spent'].mean().plot(kind='bar', ax=axes[1, 0])
                axes[1, 0].set_title('Average Spending by Age Group')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Spending by Region
            if 'region' in df.columns:
                df.groupby('region')['total_spent'].mean().plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Average Spending by Region')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig
    
    def plot_engagement_metrics(self, df: pd.DataFrame, interactive: bool = True) -> Any:
        """Visualize player engagement metrics"""
        
        engagement_cols = ['avg_session_duration', 'sessions_last_week', 'friends_count', 'achievements_unlocked']
        available_cols = [col for col in engagement_cols if col in df.columns]
        
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[col.replace('_', ' ').title() for col in available_cols[:4]]
            )
            
            for i, col in enumerate(available_cols[:4]):
                row = (i // 2) + 1
                col_num = (i % 2) + 1
                
                fig.add_trace(
                    go.Histogram(x=df[col], name=col),
                    row=row, col=col_num
                )
            
            fig.update_layout(height=600, title_text="Player Engagement Metrics")
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            axes = axes.flatten()
            
            for i, col in enumerate(available_cols[:4]):
                axes[i].hist(df[col], bins=30, alpha=0.7)
                axes[i].set_xlabel(col.replace('_', ' ').title())
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}')
            
            plt.tight_layout()
            return fig
    
    def plot_player_lifecycle(self, df: pd.DataFrame, interactive: bool = True) -> Any:
        """Analyze player lifecycle patterns"""
        
        if 'registration_date' not in df.columns:
            print("Registration date not available for lifecycle analysis")
            return None
        
        # Convert registration date
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        df['days_since_registration'] = (pd.Timestamp.now() - df['registration_date']).dt.days
        
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Registration Timeline', 'Player Retention', 
                               'Activity by Tenure', 'Spending by Tenure']
            )
            
            # Registration timeline
            reg_timeline = df.groupby(df['registration_date'].dt.date).size().reset_index()
            reg_timeline.columns = ['date', 'new_players']
            
            fig.add_trace(
                go.Scatter(x=reg_timeline['date'], y=reg_timeline['new_players'], 
                          mode='lines', name='Daily Registrations'),
                row=1, col=1
            )
            
            # Player retention by tenure
            tenure_groups = pd.cut(df['days_since_registration'], 
                                 bins=[0, 30, 90, 180, 365, float('inf')],
                                 labels=['0-30 days', '31-90 days', '91-180 days', '181-365 days', '365+ days'])
            
            retention_data = df.groupby([tenure_groups, 'churned']).size().unstack(fill_value=0)
            if 1 in retention_data.columns:
                retention_rate = retention_data[0] / (retention_data[0] + retention_data[1])
            else:
                retention_rate = pd.Series([1.0] * len(retention_data), index=retention_data.index)
            
            fig.add_trace(
                go.Bar(x=retention_rate.index, y=retention_rate.values, name='Retention Rate'),
                row=1, col=2
            )
            
            # Activity by tenure
            activity_by_tenure = df.groupby(tenure_groups)['total_playtime_hours'].mean()
            fig.add_trace(
                go.Bar(x=activity_by_tenure.index, y=activity_by_tenure.values, name='Avg Playtime'),
                row=2, col=1
            )
            
            # Spending by tenure
            spending_by_tenure = df.groupby(tenure_groups)['total_spent'].mean()
            fig.add_trace(
                go.Bar(x=spending_by_tenure.index, y=spending_by_tenure.values, name='Avg Spending'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text="Player Lifecycle Analysis")
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Registration timeline
            reg_timeline = df.groupby(df['registration_date'].dt.date).size()
            reg_timeline.plot(ax=axes[0, 0])
            axes[0, 0].set_title('Daily Player Registrations')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Tenure distribution
            axes[0, 1].hist(df['days_since_registration'], bins=50, alpha=0.7)
            axes[0, 1].set_xlabel('Days Since Registration')
            axes[0, 1].set_ylabel('Number of Players')
            axes[0, 1].set_title('Player Tenure Distribution')
            
            # Activity by tenure
            tenure_groups = pd.cut(df['days_since_registration'], bins=10)
            df.groupby(tenure_groups)['total_playtime_hours'].mean().plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Average Playtime by Tenure')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Spending by tenure
            df.groupby(tenure_groups)['total_spent'].mean().plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Average Spending by Tenure')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig
    
    def plot_social_behavior(self, df: pd.DataFrame, interactive: bool = True) -> Any:
        """Analyze social behavior patterns"""
        
        social_cols = ['friends_count', 'forum_posts', 'reviews_written']
        available_cols = [col for col in social_cols if col in df.columns]
        
        if not available_cols:
            print("No social behavior data available")
            return None
        
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Friends Distribution', 'Social Activity vs Playtime',
                               'Social Players vs Non-Social', 'Social Impact on Spending']
            )
            
            # Friends distribution
            fig.add_trace(
                go.Histogram(x=df['friends_count'], name='Friends Count'),
                row=1, col=1
            )
            
            # Social activity vs playtime
            if 'forum_posts' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['total_playtime_hours'],
                        y=df['forum_posts'] + df.get('reviews_written', 0),
                        mode='markers',
                        name='Social Activity vs Playtime',
                        opacity=0.6
                    ),
                    row=1, col=2
                )
            
            # Social vs non-social players
            df['is_social'] = df['friends_count'] > 5
            social_comparison = df.groupby('is_social')[['total_playtime_hours', 'total_spent']].mean()
            
            fig.add_trace(
                go.Bar(
                    x=['Non-Social', 'Social'],
                    y=social_comparison['total_playtime_hours'],
                    name='Avg Playtime'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=['Non-Social', 'Social'],
                    y=social_comparison['total_spent'],
                    name='Avg Spending'
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text="Social Behavior Analysis")
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Friends distribution
            axes[0, 0].hist(df['friends_count'], bins=30, alpha=0.7)
            axes[0, 0].set_xlabel('Number of Friends')
            axes[0, 0].set_ylabel('Number of Players')
            axes[0, 0].set_title('Friends Count Distribution')
            
            # Social activity correlation
            if 'forum_posts' in df.columns:
                axes[0, 1].scatter(df['friends_count'], df['forum_posts'], alpha=0.6)
                axes[0, 1].set_xlabel('Friends Count')
                axes[0, 1].set_ylabel('Forum Posts')
                axes[0, 1].set_title('Friends vs Forum Activity')
            
            # Social impact on engagement
            df['social_group'] = pd.cut(df['friends_count'], 
                                      bins=[0, 0, 5, 15, float('inf')],
                                      labels=['No Friends', 'Few Friends', 'Many Friends', 'Very Social'])
            
            df.groupby('social_group')['total_playtime_hours'].mean().plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Playtime by Social Group')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            df.groupby('social_group')['total_spent'].mean().plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Spending by Social Group')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig


class ChurnAnalysisPlotter:
    """
    Specialized visualization for churn analysis and patterns
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.churn_colors = {'churned': '#d62728', 'retained': '#2ca02c'}
        
    def plot_churn_overview(self, df: pd.DataFrame, interactive: bool = True) -> Any:
        """Create comprehensive churn overview"""
        
        if 'churned' not in df.columns:
            print("Churn column not found in data")
            return None
        
        churn_rate = df['churned'].mean()
        churn_counts = df['churned'].value_counts()
        
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Churn Distribution', 'Churn Rate by Segment',
                               'Feature Impact on Churn', 'Churn Timeline'],
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}]]
            )
            
            # Churn pie chart
            fig.add_trace(
                go.Pie(
                    labels=['Retained', 'Churned'],
                    values=[churn_counts[0], churn_counts[1]],
                    marker_colors=['#2ca02c', '#d62728']
                ),
                row=1, col=1
            )
            
            # Churn by segments
            if 'age_group' in df.columns:
                churn_by_age = df.groupby('age_group')['churned'].mean()
                fig.add_trace(
                    go.Bar(x=churn_by_age.index, y=churn_by_age.values, name='Churn Rate by Age'),
                    row=1, col=2
                )
            
            # Feature impact
            numeric_cols = ['total_playtime_hours', 'total_spent', 'friends_count']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if available_cols:
                feature_impact = []
                for col in available_cols[:3]:
                    churned_mean = df[df['churned'] == 1][col].mean()
                    retained_mean = df[df['churned'] == 0][col].mean()
                    impact = abs(churned_mean - retained_mean) / retained_mean if retained_mean > 0 else 0
                    feature_impact.append((col, impact))
                
                features, impacts = zip(*feature_impact)
                fig.add_trace(
                    go.Bar(x=list(features), y=list(impacts), name='Feature Impact'),
                    row=2, col=1
                )
            
            # Churn timeline
            if 'registration_date' in df.columns:
                df['registration_date'] = pd.to_datetime(df['registration_date'])
                churn_timeline = df.groupby(df['registration_date'].dt.date)['churned'].mean()
                
                fig.add_trace(
                    go.Scatter(x=churn_timeline.index, y=churn_timeline.values,
                              mode='lines', name='Daily Churn Rate'),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, title_text=f"Churn Analysis Overview (Rate: {churn_rate:.1%})")
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Churn distribution
            churn_counts.plot(kind='pie', ax=axes[0, 0], autopct='%1.1f%%',
                             colors=[self.churn_colors['retained'], self.churn_colors['churned']])
            axes[0, 0].set_title(f'Churn Distribution (Rate: {churn_rate:.1%})')
            
            # Churn by category
            if 'age_group' in df.columns:
                df.groupby('age_group')['churned'].mean().plot(kind='bar', ax=axes[0, 1],
                                                              color=self.churn_colors['churned'])
                axes[0, 1].set_title('Churn Rate by Age Group')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Feature comparison
            numeric_cols = ['total_playtime_hours', 'total_spent', 'friends_count']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if available_cols:
                churned_data = df[df['churned'] == 1][available_cols].mean()
                retained_data = df[df['churned'] == 0][available_cols].mean()
                
                x = np.arange(len(available_cols))
                width = 0.35
                
                axes[1, 0].bar(x - width/2, retained_data, width, label='Retained', 
                              color=self.churn_colors['retained'])
                axes[1, 0].bar(x + width/2, churned_data, width, label='Churned',
                              color=self.churn_colors['churned'])
                axes[1, 0].set_xlabel('Features')
                axes[1, 0].set_ylabel('Average Value')
                axes[1, 0].set_title('Feature Comparison: Churned vs Retained')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels([col.replace('_', ' ').title() for col in available_cols])
                axes[1, 0].legend()
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Churn by playtime bins
            if 'total_playtime_hours' in df.columns:
                playtime_bins = pd.cut(df['total_playtime_hours'], bins=10)
                churn_by_playtime = df.groupby(playtime_bins)['churned'].mean()
                churn_by_playtime.plot(kind='bar', ax=axes[1, 1], color=self.churn_colors['churned'])
                axes[1, 1].set_title('Churn Rate by Playtime')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig
    
    def plot_churn_by_segments(self, df: pd.DataFrame, segment_cols: List[str] = None,
                              interactive: bool = True) -> Any:
        """Analyze churn across different player segments"""
        
        if segment_cols is None:
            segment_cols = ['age_group', 'region', 'platform_preference', 'favorite_genre']
        
        available_cols = [col for col in segment_cols if col in df.columns]
        
        if not available_cols:
            print("No segment columns available")
            return None
        
        if interactive:
            n_cols = len(available_cols)
            rows = (n_cols + 1) // 2
            
            fig = make_subplots(
                rows=rows, cols=2,
                subplot_titles=[f'Churn Rate by {col.replace("_", " ").title()}' for col in available_cols]
            )
            
            for i, col in enumerate(available_cols):
                row = (i // 2) + 1
                col_num = (i % 2) + 1
                
                churn_by_segment = df.groupby(col)['churned'].mean().reset_index()
                
                fig.add_trace(
                    go.Bar(
                        x=churn_by_segment[col],
                        y=churn_by_segment['churned'],
                        name=f'Churn by {col}',
                        marker_color='#d62728'
                    ),
                    row=row, col=col_num
                )
            
            fig.update_layout(height=400*rows, title_text="Churn Analysis by Segments")
            return fig
        else:
            n_cols = len(available_cols)
            rows = (n_cols + 1) // 2
            
            fig, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, col in enumerate(available_cols):
                row = i // 2
                col_idx = i % 2
                
                churn_by_segment = df.groupby(col)['churned'].mean()
                churn_by_segment.plot(kind='bar', ax=axes[row, col_idx], 
                                     color=self.churn_colors['churned'])
                axes[row, col_idx].set_title(f'Churn Rate by {col.replace("_", " ").title()}')
                axes[row, col_idx].tick_params(axis='x', rotation=45)
                axes[row, col_idx].set_ylabel('Churn Rate')
            
            # Hide empty subplots
            for i in range(len(available_cols), rows * 2):
                row = i // 2
                col_idx = i % 2
                axes[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            return fig
    
    def plot_churn_risk_factors(self, df: pd.DataFrame, interactive: bool = True) -> Any:
        """Identify and visualize key churn risk factors"""
        
        # Calculate risk factors
        risk_factors = self._calculate_risk_factors(df)
        
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Risk Factor Importance', 'High-Risk Player Characteristics',
                               'Risk Score Distribution', 'Intervention Opportunities']
            )
            
            # Risk factor importance
            factors, importance = zip(*list(risk_factors.items())[:10])
            fig.add_trace(
                go.Bar(x=list(importance), y=list(factors), orientation='h', name='Importance'),
                row=1, col=1
            )
            
            # High-risk characteristics
            high_risk_threshold = df['churned'].quantile(0.8)
            high_risk_players = df[df['churned'] >= high_risk_threshold]
            
            characteristics = {
                'Low Playtime': (high_risk_players['total_playtime_hours'] < df['total_playtime_hours'].median()).mean(),
                'No Spending': (high_risk_players['total_spent'] == 0).mean(),
                'Few Friends': (high_risk_players['friends_count'] < 3).mean(),
                'Inactive': (high_risk_players.get('last_login_days_ago', 0) > 7).mean()
            }
            
            fig.add_trace(
                go.Bar(x=list(characteristics.keys()), y=list(characteristics.values()),
                      name='High-Risk Characteristics'),
                row=1, col=2
            )
            
            # Risk score distribution
            if 'total_risk_score' in df.columns:
                fig.add_trace(
                    go.Histogram(x=df['total_risk_score'], name='Risk Score Distribution'),
                    row=2, col=1
                )
            
            # Intervention opportunities
            intervention_data = {
                'Immediate': len(df[df['churned'] > 0.8]),
                'Near-term': len(df[(df['churned'] > 0.6) & (df['churned'] <= 0.8)]),
                'Watch': len(df[(df['churned'] > 0.3) & (df['churned'] <= 0.6)]),
                'Stable': len(df[df['churned'] <= 0.3])
            }
            
            fig.add_trace(
                go.Bar(x=list(intervention_data.keys()), y=list(intervention_data.values()),
                      name='Intervention Priorities'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text="Churn Risk Factor Analysis")
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Risk factor importance
            factors = list(risk_factors.keys())[:10]
            importance = list(risk_factors.values())[:10]
            
            axes[0, 0].barh(factors, importance)
            axes[0, 0].set_title('Top Risk Factors')
            axes[0, 0].set_xlabel('Importance Score')
            
            # Feature distributions by churn
            numeric_features = ['total_playtime_hours', 'total_spent', 'friends_count']
            available_features = [f for f in numeric_features if f in df.columns]
            
            if available_features:
                feature = available_features[0]
                df[df['churned'] == 0][feature].hist(alpha=0.5, label='Retained', 
                                                    bins=30, ax=axes[0, 1])
                df[df['churned'] == 1][feature].hist(alpha=0.5, label='Churned', 
                                                    bins=30, ax=axes[0, 1])
                axes[0, 1].set_title(f'{feature.replace("_", " ").title()} Distribution by Churn')
                axes[0, 1].legend()
            
            # Churn rate by bins
            if 'total_playtime_hours' in df.columns:
                playtime_bins = pd.qcut(df['total_playtime_hours'], q=5, duplicates='drop')
                churn_by_bins = df.groupby(playtime_bins)['churned'].mean()
                churn_by_bins.plot(kind='bar', ax=axes[1, 0], color=self.churn_colors['churned'])
                axes[1, 0].set_title('Churn Rate by Playtime Quintiles')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Risk score distribution
            if 'total_risk_score' in df.columns:
                axes[1, 1].hist(df['total_risk_score'], bins=30, alpha=0.7)
                axes[1, 1].set_title('Risk Score Distribution')
                axes[1, 1].set_xlabel('Total Risk Score')
                axes[1, 1].set_ylabel('Number of Players')
            
            plt.tight_layout()
            return fig
    
    def _calculate_risk_factors(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate importance of different risk factors"""
        
        risk_factors = {}
        
        # Correlation-based importance
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'churned' and col in df.columns:
                corr = abs(df[col].corr(df['churned']))
                if not np.isnan(corr):
                    risk_factors[col] = corr
        
        # Sort by importance
        risk_factors = dict(sorted(risk_factors.items(), key=lambda x: x[1], reverse=True))
        
        return risk_factors


class FeatureDistributionPlotter:
    """
    Visualize feature distributions and relationships
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_feature_distributions(self, df: pd.DataFrame, features: List[str] = None,
                                  interactive: bool = True) -> Any:
        """Plot distributions of selected features"""
        
        if features is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = list(numeric_cols)[:12]  # Limit to 12 features
        
        if interactive:
            n_features = len(features)
            cols = 3
            rows = (n_features + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f.replace('_', ' ').title() for f in features]
            )
            
            for i, feature in enumerate(features):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                fig.add_trace(
                    go.Histogram(x=df[feature], name=feature, showlegend=False),
                    row=row, col=col
                )
            
            fig.update_layout(height=300*rows, title_text="Feature Distributions")
            return fig
        else:
            n_features = len(features)
            cols = 3
            rows = (n_features + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i, feature in enumerate(features):
                row = i // cols
                col = i % cols
                
                axes[row, col].hist(df[feature], bins=30, alpha=0.7, edgecolor='black')
                axes[row, col].set_title(f'{feature.replace("_", " ").title()}')
                axes[row, col].set_xlabel(feature)
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(features), rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            return fig
    
    def plot_feature_boxplots(self, df: pd.DataFrame, features: List[str] = None,
                             group_by: str = 'churned', interactive: bool = True) -> Any:
        """Create box plots for features grouped by target variable"""
        
        if features is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if col != group_by][:8]
        
        if group_by not in df.columns:
            print(f"Group by column '{group_by}' not found")
            return None
        
        if interactive:
            n_features = len(features)
            cols = 2
            rows = (n_features + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f.replace('_', ' ').title() for f in features]
            )
            
            for i, feature in enumerate(features):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                for group_val in df[group_by].unique():
                    group_data = df[df[group_by] == group_val][feature]
                    
                    fig.add_trace(
                        go.Box(y=group_data, name=f'{group_by}={group_val}', 
                              showlegend=(i == 0)),
                        row=row, col=col
                    )
            
            fig.update_layout(height=400*rows, title_text=f"Feature Distributions by {group_by}")
            return fig
        else:
            n_features = len(features)
            cols = 2
            rows = (n_features + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 5*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, feature in enumerate(features):
                row = i // cols
                col = i % cols
                
                data_by_group = [df[df[group_by] == val][feature] for val in df[group_by].unique()]
                labels = [f'{group_by}={val}' for val in df[group_by].unique()]
                
                axes[row, col].boxplot(data_by_group, labels=labels)
                axes[row, col].set_title(f'{feature.replace("_", " ").title()}')
                axes[row, col].set_ylabel(feature)
                axes[row, col].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(features), rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            return fig
    
    def plot_feature_relationships(self, df: pd.DataFrame, feature_pairs: List[Tuple[str, str]] = None,
                                  color_by: str = 'churned', interactive: bool = True) -> Any:
        """Plot relationships between feature pairs"""
        
        if feature_pairs is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != color_by]
            
            # Create some default pairs
            feature_pairs = []
            for i in range(0, min(len(numeric_cols), 6), 2):
                if i + 1 < len(numeric_cols):
                    feature_pairs.append((numeric_cols[i], numeric_cols[i + 1]))
        
        if interactive:
            n_pairs = len(feature_pairs)
            cols = 2
            rows = (n_pairs + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f'{pair[0]} vs {pair[1]}' for pair in feature_pairs]
            )
            
            for i, (feat1, feat2) in enumerate(feature_pairs):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                if color_by in df.columns:
                    for color_val in df[color_by].unique():
                        subset = df[df[color_by] == color_val]
                        fig.add_trace(
                            go.Scatter(
                                x=subset[feat1],
                                y=subset[feat2],
                                mode='markers',
                                name=f'{color_by}={color_val}',
                                showlegend=(i == 0),
                                opacity=0.6
                            ),
                            row=row, col=col
                        )
                else:
                    fig.add_trace(
                        go.Scatter(x=df[feat1], y=df[feat2], mode='markers', 
                                  name=f'{feat1} vs {feat2}', opacity=0.6),
                        row=row, col=col
                    )
            
            fig.update_layout(height=400*rows, title_text="Feature Relationships")
            return fig
        else:
            n_pairs = len(feature_pairs)
            cols = 2
            rows = (n_pairs + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 5*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (feat1, feat2) in enumerate(feature_pairs):
                row = i // cols
                col = i % cols
                
                if color_by in df.columns:
                    for color_val in df[color_by].unique():
                        subset = df[df[color_by] == color_val]
                        axes[row, col].scatter(subset[feat1], subset[feat2], 
                                             label=f'{color_by}={color_val}', alpha=0.6)
                    axes[row, col].legend()
                else:
                    axes[row, col].scatter(df[feat1], df[feat2], alpha=0.6)
                
                axes[row, col].set_xlabel(feat1.replace('_', ' ').title())
                axes[row, col].set_ylabel(feat2.replace('_', ' ').title())
                axes[row, col].set_title(f'{feat1} vs {feat2}')
                axes[row, col].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(feature_pairs), rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            return fig


class CorrelationVisualizer:
    """
    Visualize feature correlations and relationships
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        self.figsize = figsize
        
    def plot_correlation_matrix(self, df: pd.DataFrame, features: List[str] = None,
                               method: str = 'pearson', interactive: bool = True) -> Any:
        """Create correlation matrix heatmap"""
        
        if features is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = list(numeric_cols)[:20]  # Limit for readability
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr(method=method)
        
        if interactive:
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={'size': 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f'Feature Correlation Matrix ({method.title()})',
                width=800,
                height=800
            )
            
            return fig
        else:
            plt.figure(figsize=self.figsize)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})
            plt.title(f'Feature Correlation Matrix ({method.title()})')
            plt.tight_layout()
            return plt.gcf()
    
    def plot_correlation_with_target(self, df: pd.DataFrame, target_col: str = 'churned',
                                   top_n: int = 15, interactive: bool = True) -> Any:
        """Plot correlations of features with target variable"""
        
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found")
            return None
        
        # Calculate correlations with target
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for col in numeric_cols:
            if col != target_col:
                corr = df[col].corr(df[target_col])
                if not np.isnan(corr):
                    correlations[col] = corr
        
        # Sort and get top N
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        features, corr_values = zip(*sorted_corr)
        
        if interactive:
            colors = ['red' if x < 0 else 'blue' for x in corr_values]
            
            fig = go.Figure(data=go.Bar(
                x=list(corr_values),
                y=list(features),
                orientation='h',
                marker_color=colors
            ))
            
            fig.update_layout(
                title=f'Feature Correlations with {target_col.title()}',
                xaxis_title='Correlation Coefficient',
                yaxis_title='Features',
                height=max(400, len(features) * 30)
            )
            
            return fig
        else:
            plt.figure(figsize=(10, max(6, len(features) * 0.4)))
            colors = ['red' if x < 0 else 'blue' for x in corr_values]
            plt.barh(range(len(features)), corr_values, color=colors)
            plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
            plt.xlabel('Correlation Coefficient')
            plt.title(f'Feature Correlations with {target_col.title()}')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            return plt.gcf()
    
    def plot_feature_clusters(self, df: pd.DataFrame, features: List[str] = None,
                             interactive: bool = True) -> Any:
        """Cluster features based on correlation and visualize"""
        
        if features is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = list(numeric_cols)[:15]
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr().abs()
        
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance
        distance_matrix = 1 - corr_matrix
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='average')
        
        if interactive:
            # Create dendrogram
            fig = ff.create_dendrogram(
                linkage_matrix,
                labels=[f.replace('_', ' ').title() for f in features],
                orientation='left'
            )
            
            fig.update_layout(
                title='Feature Clustering Dendrogram',
                height=max(400, len(features) * 30)
            )
            
            return fig
        else:
            plt.figure(figsize=(12, 8))
            dendrogram(linkage_matrix, 
                      labels=[f.replace('_', ' ').title() for f in features],
                      orientation='left')
            plt.title('Feature Clustering Dendrogram')
            plt.xlabel('Distance')
            plt.tight_layout()
            return plt.gcf()
    
    def plot_partial_correlations(self, df: pd.DataFrame, target_features: List[str],
                                 control_features: List[str] = None, 
                                 interactive: bool = True) -> Any:
        """Plot partial correlations controlling for specified features"""
        
        try:
            from scipy.stats import pearsonr
            from sklearn.linear_model import LinearRegression
            
            if control_features is None:
                # Select some features as controls
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                control_features = list(numeric_cols)[:3]
            
            partial_corr_results = {}
            
            # Calculate partial correlations
            for i, feat1 in enumerate(target_features):
                for feat2 in target_features[i+1:]:
                    # Regress out control variables
                    X_control = df[control_features].fillna(0)
                    
                    # Residuals for feat1
                    reg1 = LinearRegression().fit(X_control, df[feat1].fillna(0))
                    residuals1 = df[feat1].fillna(0) - reg1.predict(X_control)
                    
                    # Residuals for feat2
                    reg2 = LinearRegression().fit(X_control, df[feat2].fillna(0))
                    residuals2 = df[feat2].fillna(0) - reg2.predict(X_control)
                    
                    # Correlation of residuals
                    partial_corr, _ = pearsonr(residuals1, residuals2)
                    partial_corr_results[f'{feat1} vs {feat2}'] = partial_corr
            
            # Plot results
            pairs = list(partial_corr_results.keys())
            correlations = list(partial_corr_results.values())
            
            if interactive:
                fig = go.Figure(data=go.Bar(
                    x=correlations,
                    y=pairs,
                    orientation='h',
                    marker_color=['red' if x < 0 else 'blue' for x in correlations]
                ))
                
                fig.update_layout(
                    title='Partial Correlations (Controlling for Other Features)',
                    xaxis_title='Partial Correlation',
                    height=max(400, len(pairs) * 40)
                )
                
                return fig
            else:
                plt.figure(figsize=(10, max(6, len(pairs) * 0.4)))
                colors = ['red' if x < 0 else 'blue' for x in correlations]
                plt.barh(range(len(pairs)), correlations, color=colors)
                plt.yticks(range(len(pairs)), pairs)
                plt.xlabel('Partial Correlation')
                plt.title('Partial Correlations (Controlling for Other Features)')
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                return plt.gcf()
                
        except ImportError:
            print("Scipy required for partial correlation analysis")
            return None


def main():
    """Example usage of EDA visualization classes"""
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_players = 1000
    
    sample_data = pd.DataFrame({
        'player_id': [f'player_{i}' for i in range(n_players)],
        'total_playtime_hours': np.random.exponential(100, n_players),
        'total_spent': np.random.exponential(50, n_players),
        'friends_count': np.random.poisson(5, n_players),
        'achievements_unlocked': np.random.poisson(20, n_players),
        'avg_session_duration': np.random.normal(45, 15, n_players),
        'sessions_last_week': np.random.poisson(3, n_players),
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], n_players),
        'region': np.random.choice(['NA', 'EU', 'ASIA', 'OTHER'], n_players),
        'churned': np.random.binomial(1, 0.2, n_players)
    })
    
    # Initialize visualizers
    behavior_viz = PlayerBehaviorVisualizer()
    churn_viz = ChurnAnalysisPlotter()
    feature_viz = FeatureDistributionPlotter()
    corr_viz = CorrelationVisualizer()
    
    print("EDA Visualization classes initialized successfully!")
    
    # Example plots (commented out to avoid display in headless environment)
    # behavior_viz.plot_playtime_distribution(sample_data)
    # churn_viz.plot_churn_overview(sample_data)
    # feature_viz.plot_feature_distributions(sample_data)
    # corr_viz.plot_correlation_matrix(sample_data)

if __name__ == "__main__":
    main()