"""
Business Intelligence Dashboard and Visualization Module

Comprehensive business-focused visualizations for ROI, revenue impact, and player segmentation.

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
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BusinessDashboard:
    """
    Comprehensive business intelligence dashboard for gaming analytics
    """
    
    def __init__(self, business_params: Dict[str, float] = None):
        """Initialize business dashboard with parameters"""
        self.business_params = business_params or {
            'cost_per_intervention': 10.0,
            'revenue_per_retained_player': 50.0,
            'intervention_success_rate': 0.25,
            'player_lifetime_months': 12,
            'monthly_revenue_per_player': 15.0,
            'acquisition_cost_per_player': 25.0
        }
        
        self.colors = {
            'profit': '#2ca02c',
            'cost': '#d62728', 
            'revenue': '#1f77b4',
            'neutral': '#7f7f7f',
            'warning': '#ff7f0e'
        }
    
    def create_executive_dashboard(self, df: pd.DataFrame, predictions: pd.DataFrame = None,
                                 interactive: bool = True) -> Any:
        """Create comprehensive executive dashboard"""
        
        if interactive:
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=[
                    'Player Overview', 'Churn Risk Distribution', 'Revenue at Risk',
                    'Player Value Segments', 'Retention ROI', 'Monthly Trends',
                    'Key Metrics', 'Intervention Priorities', 'Performance KPIs'
                ],
                specs=[
                    [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                    [{"type": "indicator"}, {"type": "xy"}, {"type": "indicator"}]
                ]
            )
            
            # Player Overview
            total_players = len(df)
            active_players = len(df[df.get('last_login_days_ago', 0) <= 7])
            
            fig.add_trace(
                go.Bar(x=['Total', 'Active'], y=[total_players, active_players],
                      marker_color=[self.colors['neutral'], self.colors['profit']]),
                row=1, col=1
            )
            
            # Churn Risk Distribution
            if 'churned' in df.columns or predictions is not None:
                if predictions is not None and 'risk_level' in predictions.columns:
                    risk_dist = predictions['risk_level'].value_counts()
                else:
                    # Create risk levels from churn column
                    risk_levels = pd.cut(df.get('churned', np.random.random(len(df))), 
                                       bins=[0, 0.3, 0.6, 0.8, 1.0],
                                       labels=['Low', 'Medium', 'High', 'Critical'])
                    risk_dist = risk_levels.value_counts()
                
                fig.add_trace(
                    go.Pie(labels=risk_dist.index, values=risk_dist.values,
                          marker_colors=['#2ca02c', '#ff7f0e', '#d62728', '#8b0000']),
                    row=1, col=2
                )
            
            # Revenue at Risk
            revenue_data = self._calculate_revenue_at_risk(df)
            fig.add_trace(
                go.Bar(x=list(revenue_data.keys()), y=list(revenue_data.values()),
                      marker_color=self.colors['warning']),
                row=1, col=3
            )
            
            # Player Value Segments
            segments = self._create_player_segments(df)
            segment_counts = segments.value_counts()
            
            fig.add_trace(
                go.Bar(x=segment_counts.index, y=segment_counts.values,
                      marker_color=px.colors.qualitative.Set1),
                row=2, col=1
            )
            
            # ROI Analysis
            roi_data = self._calculate_roi_metrics(df, predictions)
            
            fig.add_trace(
                go.Waterfall(
                    x=['Revenue', 'Costs', 'Net Benefit'],
                    y=[roi_data['revenue'], -roi_data['costs'], roi_data['net_benefit']],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    decreasing={"marker": {"color": self.colors['cost']}},
                    increasing={"marker": {"color": self.colors['profit']}},
                    totals={"marker": {"color": self.colors['revenue']}}
                ),
                row=2, col=2
            )
            
            # Monthly Trends (simulated)
            monthly_data = self._generate_monthly_trends()
            for metric, values in monthly_data.items():
                fig.add_trace(
                    go.Scatter(x=list(range(12)), y=values, mode='lines+markers', name=metric),
                    row=2, col=3
                )
            
            # Key Metrics Indicators
            churn_rate = df.get('churned', np.random.random(len(df))).mean()
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=churn_rate * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Rate %"},
                    gauge={'axis': {'range': [None, 50]},
                          'bar': {'color': "darkblue"},
                          'steps': [{'range': [0, 20], 'color': "lightgray"},
                                   {'range': [20, 35], 'color': "yellow"},
                                   {'range': [35, 50], 'color': "red"}],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 25}}
                ),
                row=3, col=1
            )
            
            # Intervention Priorities
            intervention_data = self._calculate_intervention_priorities(df, predictions)
            fig.add_trace(
                go.Bar(x=list(intervention_data.keys()), y=list(intervention_data.values()),
                      marker_color=[self.colors['cost'], self.colors['warning'], 
                                   self.colors['profit'], self.colors['neutral']]),
                row=3, col=2
            )
            
            # ROI Indicator
            roi_percentage = roi_data.get('roi_percentage', 0)
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=roi_percentage,
                    title={'text': "ROI %"},
                    delta={'reference': 20, 'valueformat': '.1f'},
                    number={'valueformat': '.1f', 'suffix': '%'}
                ),
                row=3, col=3
            )
            
            fig.update_layout(height=1200, title_text="Executive Business Dashboard")
            return fig
        else:
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            
            # Player Overview
            total_players = len(df)
            active_players = len(df[df.get('last_login_days_ago', 0) <= 7])
            axes[0, 0].bar(['Total', 'Active'], [total_players, active_players])
            axes[0, 0].set_title('Player Overview')
            
            # Churn Risk Distribution
            if 'churned' in df.columns:
                risk_levels = pd.cut(df['churned'], bins=[0, 0.3, 0.6, 0.8, 1.0],
                                   labels=['Low', 'Medium', 'High', 'Critical'])
                risk_dist = risk_levels.value_counts()
                risk_dist.plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%')
                axes[0, 1].set_title('Churn Risk Distribution')
            
            # Revenue at Risk
            revenue_data = self._calculate_revenue_at_risk(df)
            axes[0, 2].bar(revenue_data.keys(), revenue_data.values())
            axes[0, 2].set_title('Revenue at Risk')
            axes[0, 2].tick_params(axis='x', rotation=45)
            
            # Continue with other subplots...
            segments = self._create_player_segments(df)
            segments.value_counts().plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Player Value Segments')
            
            plt.tight_layout()
            return fig
    
    def _calculate_revenue_at_risk(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate revenue at risk by segment"""
        
        segments = self._create_player_segments(df)
        churn_risk = df.get('churned', np.random.random(len(df)))
        
        revenue_at_risk = {}
        for segment in segments.unique():
            segment_mask = segments == segment
            segment_revenue = df[segment_mask]['total_spent'].sum()
            segment_risk = churn_risk[segment_mask].mean()
            revenue_at_risk[segment] = segment_revenue * segment_risk
        
        return revenue_at_risk
    
    def _create_player_segments(self, df: pd.DataFrame) -> pd.Series:
        """Create player value segments"""
        
        # Calculate player value score
        spending_score = pd.qcut(df.get('total_spent', 0), q=4, labels=['Low', 'Medium', 'High', 'VIP'], duplicates='drop')
        return spending_score.fillna('Low')
    
    def _calculate_roi_metrics(self, df: pd.DataFrame, predictions: pd.DataFrame = None) -> Dict[str, float]:
        """Calculate ROI metrics for interventions"""
        
        if predictions is not None:
            predicted_churners = len(predictions[predictions.get('churn_prediction', False)])
        else:
            predicted_churners = int(len(df) * df.get('churned', np.random.random(len(df))).mean())
        
        intervention_cost = predicted_churners * self.business_params['cost_per_intervention']
        
        # Estimate successful interventions
        successful_interventions = predicted_churners * self.business_params['intervention_success_rate']
        revenue_saved = successful_interventions * self.business_params['revenue_per_retained_player']
        
        net_benefit = revenue_saved - intervention_cost
        roi_percentage = (net_benefit / intervention_cost * 100) if intervention_cost > 0 else 0
        
        return {
            'costs': intervention_cost,
            'revenue': revenue_saved,
            'net_benefit': net_benefit,
            'roi_percentage': roi_percentage
        }
    
    def _generate_monthly_trends(self) -> Dict[str, List[float]]:
        """Generate sample monthly trend data"""
        
        np.random.seed(42)
        return {
            'Revenue': np.random.normal(100000, 10000, 12).tolist(),
            'Churn Rate': (np.random.beta(2, 8, 12) * 100).tolist(),
            'New Players': np.random.poisson(500, 12).tolist()
        }
    
    def _calculate_intervention_priorities(self, df: pd.DataFrame, predictions: pd.DataFrame = None) -> Dict[str, int]:
        """Calculate intervention priorities"""
        
        if predictions is not None and 'risk_level' in predictions.columns:
            return predictions['risk_level'].value_counts().to_dict()
        else:
            # Simulate based on churn probabilities
            churn_probs = df.get('churned', np.random.random(len(df)))
            immediate = np.sum(churn_probs > 0.8)
            high = np.sum((churn_probs > 0.6) & (churn_probs <= 0.8))
            medium = np.sum((churn_probs > 0.3) & (churn_probs <= 0.6))
            low = np.sum(churn_probs <= 0.3)
            
            return {
                'Immediate': immediate,
                'High Priority': high, 
                'Medium Priority': medium,
                'Low Priority': low
            }


class ROIAnalysisPlotter:
    """
    Specialized visualization for ROI and financial impact analysis
    """
    
    def __init__(self, business_params: Dict[str, float] = None):
        self.business_params = business_params or {
            'cost_per_intervention': 10.0,
            'revenue_per_retained_player': 50.0,
            'intervention_success_rate': 0.25,
            'player_lifetime_months': 12,
            'monthly_revenue_per_player': 15.0
        }
        
    def plot_roi_waterfall(self, intervention_costs: float, revenue_saved: float,
                          additional_costs: Dict[str, float] = None,
                          interactive: bool = True) -> Any:
        """Create ROI waterfall chart"""
        
        if additional_costs is None:
            additional_costs = {}
        
        # Build waterfall data
        categories = ['Revenue Saved']
        values = [revenue_saved]
        
        # Add costs
        values.append(-intervention_costs)
        categories.append('Intervention Costs')
        
        for cost_name, cost_value in additional_costs.items():
            values.append(-cost_value)
            categories.append(cost_name)
        
        # Net benefit
        net_benefit = sum(values)
        values.append(net_benefit)
        categories.append('Net Benefit')
        
        if interactive:
            fig = go.Figure()
            
            # Calculate cumulative values for waterfall
            cumulative = [0]
            for i, val in enumerate(values[:-1]):
                if val > 0:
                    cumulative.append(cumulative[-1] + val)
                else:
                    cumulative.append(cumulative[-1])
            
            # Revenue bar
            fig.add_trace(go.Bar(
                x=[categories[0]], y=[values[0]],
                marker_color='green', name='Revenue'
            ))
            
            # Cost bars
            for i in range(1, len(values)-1):
                fig.add_trace(go.Bar(
                    x=[categories[i]], y=[-values[i]],
                    marker_color='red', name='Costs'
                ))
            
            # Net benefit bar
            net_color = 'green' if net_benefit > 0 else 'red'
            fig.add_trace(go.Bar(
                x=[categories[-1]], y=[net_benefit],
                marker_color=net_color, name='Net Result'
            ))
            
            fig.update_layout(
                title='ROI Waterfall Analysis',
                xaxis_title='Components',
                yaxis_title='Amount ($)',
                showlegend=False
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create waterfall chart manually
            colors = []
            bottom_values = []
            bar_values = []
            
            cumulative = 0
            for i, val in enumerate(values):
                if i == 0:  # Revenue
                    colors.append('green')
                    bottom_values.append(0)
                    bar_values.append(val)
                    cumulative = val
                elif i == len(values) - 1:  # Net benefit
                    colors.append('green' if val > 0 else 'red')
                    bottom_values.append(0)
                    bar_values.append(val)
                else:  # Costs
                    colors.append('red')
                    bottom_values.append(cumulative + val)
                    bar_values.append(-val)
                    cumulative += val
            
            bars = ax.bar(categories, bar_values, bottom=bottom_values, color=colors, alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2,
                       f'${value:,.0f}', ha='center', va='center', fontweight='bold')
            
            ax.set_title('ROI Waterfall Analysis')
            ax.set_ylabel('Amount ($)')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            return fig
    
    def plot_roi_sensitivity(self, base_params: Dict[str, float], 
                           param_ranges: Dict[str, Tuple[float, float]],
                           interactive: bool = True) -> Any:
        """Analyze ROI sensitivity to parameter changes"""
        
        sensitivity_results = {}
        
        for param_name, (min_val, max_val) in param_ranges.items():
            param_values = np.linspace(min_val, max_val, 20)
            roi_values = []
            
            for param_val in param_values:
                # Update parameters
                test_params = base_params.copy()
                test_params[param_name] = param_val
                
                # Calculate ROI
                roi = self._calculate_roi_for_params(test_params)
                roi_values.append(roi)
            
            sensitivity_results[param_name] = (param_values, roi_values)
        
        if interactive:
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set1
            
            for i, (param_name, (param_vals, roi_vals)) in enumerate(sensitivity_results.items()):
                fig.add_trace(go.Scatter(
                    x=param_vals, y=roi_vals,
                    mode='lines+markers',
                    name=param_name.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="red", 
                         annotation_text="Break-even")
            
            fig.update_layout(
                title='ROI Sensitivity Analysis',
                xaxis_title='Parameter Value',
                yaxis_title='ROI (%)',
                height=500
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for param_name, (param_vals, roi_vals) in sensitivity_results.items():
                ax.plot(param_vals, roi_vals, 'o-', label=param_name.replace('_', ' ').title())
            
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('ROI (%)')
            ax.set_title('ROI Sensitivity Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def plot_cost_benefit_matrix(self, scenarios: Dict[str, Dict[str, float]],
                                interactive: bool = True) -> Any:
        """Create cost-benefit analysis matrix for different scenarios"""
        
        scenario_names = list(scenarios.keys())
        costs = [scenario['total_cost'] for scenario in scenarios.values()]
        benefits = [scenario['total_benefit'] for scenario in scenarios.values()]
        roi_values = [scenario.get('roi', 0) for scenario in scenarios.values()]
        
        if interactive:
            fig = go.Figure()
            
            # Bubble chart with ROI as bubble size
            fig.add_trace(go.Scatter(
                x=costs, y=benefits,
                mode='markers+text',
                text=scenario_names,
                textposition="middle center",
                marker=dict(
                    size=[abs(roi) * 2 + 10 for roi in roi_values],  # Scale bubble size
                    color=roi_values,
                    colorscale='RdYlGn',
                    colorbar=dict(title="ROI (%)"),
                    line=dict(width=2, color='black')
                ),
                hovertemplate='<b>%{text}</b><br>' +
                             'Cost: $%{x:,.0f}<br>' +
                             'Benefit: $%{y:,.0f}<br>' +
                             'ROI: %{marker.color:.1f}%<extra></extra>'
            ))
            
            # Add diagonal line for break-even
            max_val = max(max(costs), max(benefits))
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                name='Break-even',
                line=dict(dash='dash', color='red')
            ))
            
            fig.update_layout(
                title='Cost-Benefit Analysis Matrix',
                xaxis_title='Total Cost ($)',
                yaxis_title='Total Benefit ($)',
                height=600
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create scatter plot with ROI as color
            scatter = ax.scatter(costs, benefits, c=roi_values, s=[abs(roi) * 10 + 50 for roi in roi_values],
                               cmap='RdYlGn', alpha=0.7, edgecolors='black')
            
            # Add labels for each point
            for i, name in enumerate(scenario_names):
                ax.annotate(name, (costs[i], benefits[i]), ha='center', va='center', fontweight='bold')
            
            # Add diagonal line for break-even
            max_val = max(max(costs), max(benefits))
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Break-even')
            
            ax.set_xlabel('Total Cost ($)')
            ax.set_ylabel('Total Benefit ($)')
            ax.set_title('Cost-Benefit Analysis Matrix')
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('ROI (%)')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def _calculate_roi_for_params(self, params: Dict[str, float]) -> float:
        """Calculate ROI for given parameters"""
        
        # Simplified ROI calculation
        intervention_cost = params.get('cost_per_intervention', 10) * 100  # Assume 100 interventions
        success_rate = params.get('intervention_success_rate', 0.25)
        revenue_per_player = params.get('revenue_per_retained_player', 50)
        
        total_cost = intervention_cost
        total_benefit = intervention_cost / params.get('cost_per_intervention', 10) * success_rate * revenue_per_player
        
        roi = ((total_benefit - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        return roi


class PlayerSegmentationVisualizer:
    """
    Visualize player segmentation and value analysis
    """
    
    def __init__(self):
        self.segment_colors = {
            'VIP': '#d4af37',      # Gold
            'High': '#c0392b',     # Red
            'Medium': '#f39c12',   # Orange  
            'Low': '#95a5a6'       # Gray
        }
        
    def plot_player_segments(self, df: pd.DataFrame, segment_method: str = 'rfm',
                           interactive: bool = True) -> Any:
        """Create player segmentation visualization"""
        
        if segment_method == 'rfm':
            segments = self._create_rfm_segments(df)
        elif segment_method == 'value':
            segments = self._create_value_segments(df)
        else:
            segments = self._create_behavioral_segments(df)
        
        segment_stats = self._calculate_segment_statistics(df, segments)
        
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Segment Distribution', 'Revenue by Segment',
                               'Segment Characteristics', 'Churn Risk by Segment'],
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}]]
            )
            
            # Segment distribution
            segment_counts = segments.value_counts()
            fig.add_trace(
                go.Pie(labels=segment_counts.index, values=segment_counts.values,
                      marker_colors=[self.segment_colors.get(seg, 'gray') for seg in segment_counts.index]),
                row=1, col=1
            )
            
            # Revenue by segment
            revenue_by_segment = df.groupby(segments)['total_spent'].sum().reset_index()
            fig.add_trace(
                go.Bar(x=revenue_by_segment[segments.name], y=revenue_by_segment['total_spent'],
                      marker_color=[self.segment_colors.get(seg, 'gray') for seg in revenue_by_segment[segments.name]]),
                row=1, col=2
            )
            
            # Segment characteristics radar chart
            characteristics = ['Avg Spending', 'Avg Playtime', 'Avg Friends', 'Retention Rate']
            
            for segment in segment_stats.keys():
                stats = segment_stats[segment]
                values = [
                    stats['avg_spending'] / 100,  # Normalize
                    stats['avg_playtime'] / 100,
                    stats['avg_friends'] / 20,
                    stats['retention_rate']
                ]
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=values + [values[0]],  # Close the polygon
                        theta=characteristics + [characteristics[0]],
                        fill='toself',
                        name=segment,
                        line_color=self.segment_colors.get(segment, 'gray')
                    ),
                    row=2, col=1
                )
            
            # Churn risk by segment
            churn_by_segment = df.groupby(segments).get('churned', df.index % 2).mean().reset_index()  # Fallback for missing churn
            fig.add_trace(
                go.Bar(x=churn_by_segment[segments.name], y=churn_by_segment.iloc[:, 1],
                      marker_color='red', opacity=0.7),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text=f"Player Segmentation Analysis ({segment_method.upper()})")
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Segment distribution
            segment_counts = segments.value_counts()
            colors = [self.segment_colors.get(seg, 'gray') for seg in segment_counts.index]
            segment_counts.plot(kind='pie', ax=axes[0, 0], colors=colors, autopct='%1.1f%%')
            axes[0, 0].set_title('Segment Distribution')
            
            # Revenue by segment
            revenue_by_segment = df.groupby(segments)['total_spent'].sum()
            colors = [self.segment_colors.get(seg, 'gray') for seg in revenue_by_segment.index]
            revenue_by_segment.plot(kind='bar', ax=axes[0, 1], color=colors)
            axes[0, 1].set_title('Revenue by Segment')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Player characteristics by segment
            char_data = []
            for segment in segments.unique():
                segment_data = df[segments == segment]
                char_data.append([
                    segment_data['total_spent'].mean(),
                    segment_data['total_playtime_hours'].mean(),
                    segment_data['friends_count'].mean()
                ])
            
            char_df = pd.DataFrame(char_data, 
                                 columns=['Avg Spending', 'Avg Playtime', 'Avg Friends'],
                                 index=segments.unique())
            
            char_df.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Segment Characteristics')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Churn risk by segment
            if 'churned' in df.columns:
                churn_by_segment = df.groupby(segments)['churned'].mean()
            else:
                churn_by_segment = pd.Series([0.1, 0.2, 0.3, 0.4], index=segments.unique())
            
            churn_by_segment.plot(kind='bar', ax=axes[1, 1], color='red', alpha=0.7)
            axes[1, 1].set_title('Churn Risk by Segment')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig
    
    def plot_segment_migration(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                              time_period: str = 'month', interactive: bool = True) -> Any:
        """Visualize player segment migration over time"""
        
        # Create segments for both periods
        segments_before = self._create_value_segments(df_before)
        segments_after = self._create_value_segments(df_after)
        
        # Create migration matrix
        migration_matrix = pd.crosstab(segments_before, segments_after, normalize='index')
        
        if interactive:
            fig = go.Figure(data=go.Heatmap(
                z=migration_matrix.values,
                x=migration_matrix.columns,
                y=migration_matrix.index,
                colorscale='RdYlBu_r',
                text=np.round(migration_matrix.values * 100, 1),
                texttemplate='%{text}%',
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f'Player Segment Migration ({time_period})',
                xaxis_title='Segment After',
                yaxis_title='Segment Before',
                width=600,
                height=600
            )
            
            return fig
        else:
            plt.figure(figsize=(10, 8))
            sns.heatmap(migration_matrix * 100, annot=True, fmt='.1f', cmap='RdYlBu_r',
                       cbar_kws={'label': 'Migration Rate (%)'})
            plt.title(f'Player Segment Migration ({time_period})')
            plt.xlabel('Segment After')
            plt.ylabel('Segment Before')
            return plt.gcf()
    
    def plot_segment_lifetime_value(self, df: pd.DataFrame, interactive: bool = True) -> Any:
        """Analyze lifetime value by player segment"""
        
        segments = self._create_value_segments(df)
        
        # Calculate LTV metrics
        ltv_data = []
        for segment in segments.unique():
            segment_data = df[segments == segment]
            
            avg_spending = segment_data['total_spent'].mean()
            avg_playtime = segment_data['total_playtime_hours'].mean()
            retention_rate = 1 - segment_data.get('churned', 0.2).mean()  # Fallback retention
            
            # Estimated LTV (simplified calculation)
            estimated_ltv = avg_spending * retention_rate * 2  # Assume 2x multiplier for lifetime
            
            ltv_data.append({
                'Segment': segment,
                'Current Value': avg_spending,
                'Estimated LTV': estimated_ltv,
                'Player Count': len(segment_data),
                'Retention Rate': retention_rate
            })
        
        ltv_df = pd.DataFrame(ltv_data)
        
        if interactive:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['LTV by Segment', 'LTV vs Current Value']
            )
            
            # LTV by segment
            fig.add_trace(
                go.Bar(x=ltv_df['Segment'], y=ltv_df['Estimated LTV'],
                      marker_color=[self.segment_colors.get(seg, 'gray') for seg in ltv_df['Segment']],
                      name='Estimated LTV'),
                row=1, col=1
            )
            
            # LTV vs Current Value scatter
            fig.add_trace(
                go.Scatter(
                    x=ltv_df['Current Value'], y=ltv_df['Estimated LTV'],
                    mode='markers+text',
                    text=ltv_df['Segment'],
                    textposition="top center",
                    marker=dict(
                        size=ltv_df['Player Count'] / 50,  # Scale by player count
                        color=[self.segment_colors.get(seg, 'gray') for seg in ltv_df['Segment']],
                        line=dict(width=2, color='black')
                    ),
                    name='Segments'
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=500, title_text="Lifetime Value Analysis by Segment")
            return fig
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # LTV by segment
            colors = [self.segment_colors.get(seg, 'gray') for seg in ltv_df['Segment']]
            axes[0].bar(ltv_df['Segment'], ltv_df['Estimated LTV'], color=colors)
            axes[0].set_title('Estimated LTV by Segment')
            axes[0].set_ylabel('LTV ($)')
            axes[0].tick_params(axis='x', rotation=45)
            
            # LTV vs Current Value
            scatter = axes[1].scatter(ltv_df['Current Value'], ltv_df['Estimated LTV'],
                                    s=ltv_df['Player Count']/10, c=range(len(ltv_df)), alpha=0.7)
            
            for i, segment in enumerate(ltv_df['Segment']):
                axes[1].annotate(segment, (ltv_df.iloc[i]['Current Value'], ltv_df.iloc[i]['Estimated LTV']),
                               xytext=(5, 5), textcoords='offset points')
            
            axes[1].set_xlabel('Current Value ($)')
            axes[1].set_ylabel('Estimated LTV ($)')
            axes[1].set_title('LTV vs Current Value')
            
            plt.tight_layout()
            return fig
    
    def _create_rfm_segments(self, df: pd.DataFrame) -> pd.Series:
        """Create RFM-based player segments"""
        
        # Recency (days since last login)
        recency = df.get('last_login_days_ago', np.random.randint(0, 30, len(df)))
        
        # Frequency (approximated by sessions or playtime)
        frequency = df.get('total_playtime_hours', np.random.exponential(100, len(df)))
        
        # Monetary (total spent)
        monetary = df.get('total_spent', np.random.exponential(50, len(df)))
        
        # Create quintiles for each dimension
        r_score = pd.qcut(recency, q=5, labels=[5, 4, 3, 2, 1])  # Lower recency = higher score
        f_score = pd.qcut(frequency, q=5, labels=[1, 2, 3, 4, 5])
        m_score = pd.qcut(monetary, q=5, labels=[1, 2, 3, 4, 5])
        
        # Combine scores
        rfm_score = r_score.astype(int) + f_score.astype(int) + m_score.astype(int)
        
        # Create segments based on combined score
        segments = pd.cut(rfm_score, bins=[0, 6, 9, 12, 15], labels=['Low', 'Medium', 'High', 'VIP'])
        
        return segments.fillna('Low')
    
    def _create_value_segments(self, df: pd.DataFrame) -> pd.Series:
        """Create value-based player segments"""
        
        # Calculate value score based on spending and engagement
        spending_score = pd.qcut(df.get('total_spent', 0), q=4, labels=[1, 2, 3, 4], duplicates='drop')
        engagement_score = pd.qcut(df.get('total_playtime_hours', 0), q=4, labels=[1, 2, 3, 4], duplicates='drop')
        
        # Combine scores
        value_score = spending_score.astype(float).fillna(1) + engagement_score.astype(float).fillna(1)
        
        # Create segments
        segments = pd.cut(value_score, bins=[0, 3, 5, 7, 8], labels=['Low', 'Medium', 'High', 'VIP'])
        
        return segments.fillna('Low')
    
    def _create_behavioral_segments(self, df: pd.DataFrame) -> pd.Series:
        """Create behavior-based player segments"""
        
        # Define behavioral patterns
        conditions = [
            (df.get('total_spent', 0) > df.get('total_spent', 0).quantile(0.9)),  # High spenders
            (df.get('friends_count', 0) > df.get('friends_count', 0).quantile(0.8)),  # Social players
            (df.get('total_playtime_hours', 0) > df.get('total_playtime_hours', 0).quantile(0.8)),  # Engaged players
        ]
        
        choices = ['High Spender', 'Social Player', 'Engaged Player']
        
        segments = pd.Series(np.select(conditions, choices, default='Casual Player'), index=df.index)
        
        return segments
    
    def _calculate_segment_statistics(self, df: pd.DataFrame, segments: pd.Series) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each segment"""
        
        stats = {}
        
        for segment in segments.unique():
            segment_data = df[segments == segment]
            
            stats[segment] = {
                'avg_spending': segment_data.get('total_spent', 0).mean(),
                'avg_playtime': segment_data.get('total_playtime_hours', 0).mean(),
                'avg_friends': segment_data.get('friends_count', 0).mean(),
                'retention_rate': 1 - segment_data.get('churned', 0.2).mean(),
                'player_count': len(segment_data)
            }
        
        return stats


class RevenueImpactPlotter:
    """
    Visualize revenue impact and projections
    """
    
    def __init__(self):
        self.colors = {
            'actual': '#1f77b4',
            'projected': '#ff7f0e',
            'at_risk': '#d62728',
            'saved': '#2ca02c'
        }
        
    def plot_revenue_impact_forecast(self, historical_revenue: List[float],
                                   churn_impact: List[float],
                                   intervention_impact: List[float] = None,
                                   periods: int = 12, interactive: bool = True) -> Any:
        """Plot revenue impact forecast with and without interventions"""
        
        if intervention_impact is None:
            intervention_impact = [impact * 0.7 for impact in churn_impact]  # 30% improvement
        
        # Generate time periods
        time_periods = list(range(len(historical_revenue), len(historical_revenue) + periods))
        
        # Project baseline revenue (without intervention)
        baseline_revenue = [historical_revenue[-1] - impact for impact in churn_impact]
        
        # Project revenue with intervention
        intervention_revenue = [historical_revenue[-1] - impact for impact in intervention_impact]
        
        if interactive:
            fig = go.Figure()
            
            # Historical revenue
            fig.add_trace(go.Scatter(
                x=list(range(len(historical_revenue))),
                y=historical_revenue,
                mode='lines+markers',
                name='Historical Revenue',
                line=dict(color=self.colors['actual'], width=3)
            ))
            
            # Baseline projection (no intervention)
            fig.add_trace(go.Scatter(
                x=time_periods,
                y=baseline_revenue,
                mode='lines+markers',
                name='Projected (No Intervention)',
                line=dict(color=self.colors['at_risk'], dash='dash')
            ))
            
            # Intervention projection
            fig.add_trace(go.Scatter(
                x=time_periods,
                y=intervention_revenue,
                mode='lines+markers',
                name='Projected (With Intervention)',
                line=dict(color=self.colors['saved'])
            ))
            
            # Fill area between projections to show savings
            fig.add_trace(go.Scatter(
                x=time_periods + time_periods[::-1],
                y=intervention_revenue + baseline_revenue[::-1],
                fill='toself',
                fillcolor='rgba(46, 160, 44, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Revenue Saved',
                showlegend=True
            ))
            
            # Add vertical line to separate historical from projected
            fig.add_vline(x=len(historical_revenue)-1, line_dash="dot", line_color="gray",
                         annotation_text="Forecast Start")
            
            fig.update_layout(
                title='Revenue Impact Forecast',
                xaxis_title='Time Period',
                yaxis_title='Revenue ($)',
                height=500
            )
            
            return fig
        else:
            plt.figure(figsize=(12, 6))
            
            # Historical revenue
            hist_x = list(range(len(historical_revenue)))
            plt.plot(hist_x, historical_revenue, 'o-', linewidth=3, 
                    color=self.colors['actual'], label='Historical Revenue')
            
            # Projections
            plt.plot(time_periods, baseline_revenue, '--', 
                    color=self.colors['at_risk'], label='Projected (No Intervention)')
            plt.plot(time_periods, intervention_revenue, '-', 
                    color=self.colors['saved'], label='Projected (With Intervention)')
            
            # Fill area for savings
            plt.fill_between(time_periods, baseline_revenue, intervention_revenue,
                           alpha=0.3, color=self.colors['saved'], label='Revenue Saved')
            
            # Vertical line
            plt.axvline(x=len(historical_revenue)-1, color='gray', linestyle=':', 
                       alpha=0.7, label='Forecast Start')
            
            plt.xlabel('Time Period')
            plt.ylabel('Revenue ($)')
            plt.title('Revenue Impact Forecast')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            return plt.gcf()
    
    def plot_revenue_at_risk_breakdown(self, risk_segments: Dict[str, float],
                                     intervention_effectiveness: Dict[str, float] = None,
                                     interactive: bool = True) -> Any:
        """Break down revenue at risk by segment and intervention potential"""
        
        if intervention_effectiveness is None:
            intervention_effectiveness = {seg: 0.3 for seg in risk_segments.keys()}
        
        segments = list(risk_segments.keys())
        at_risk = list(risk_segments.values())
        can_save = [risk_segments[seg] * intervention_effectiveness[seg] for seg in segments]
        will_lose = [risk_segments[seg] - can_save[i] for i, seg in enumerate(segments)]
        
        if interactive:
            fig = go.Figure()
            
            # Revenue at risk
            fig.add_trace(go.Bar(
                name='Will Lose (No Intervention)',
                x=segments,
                y=will_lose,
                marker_color=self.colors['at_risk']
            ))
            
            # Revenue can save
            fig.add_trace(go.Bar(
                name='Can Save (With Intervention)',
                x=segments,
                y=can_save,
                marker_color=self.colors['saved']
            ))
            
            fig.update_layout(
                title='Revenue at Risk Breakdown',
                xaxis_title='Player Segments',
                yaxis_title='Revenue ($)',
                barmode='stack',
                height=500
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(segments))
            width = 0.6
            
            # Stacked bars
            ax.bar(x, will_lose, width, label='Will Lose (No Intervention)', 
                  color=self.colors['at_risk'])
            ax.bar(x, can_save, width, bottom=will_lose, 
                  label='Can Save (With Intervention)', color=self.colors['saved'])
            
            ax.set_xlabel('Player Segments')
            ax.set_ylabel('Revenue ($)')
            ax.set_title('Revenue at Risk Breakdown')
            ax.set_xticks(x)
            ax.set_xticklabels(segments)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def plot_ltv_impact(self, segment_data: Dict[str, Dict[str, float]],
                       intervention_scenarios: List[str] = None,
                       interactive: bool = True) -> Any:
        """Visualize lifetime value impact of interventions"""
        
        if intervention_scenarios is None:
            intervention_scenarios = ['No Intervention', 'Basic Intervention', 'Advanced Intervention']
        
        # Calculate LTV for each scenario
        ltv_data = {}
        
        for scenario in intervention_scenarios:
            scenario_ltv = {}
            for segment, data in segment_data.items():
                base_ltv = data.get('base_ltv', 100)
                
                if scenario == 'No Intervention':
                    ltv_multiplier = 1.0
                elif scenario == 'Basic Intervention':
                    ltv_multiplier = 1.2
                else:  # Advanced Intervention
                    ltv_multiplier = 1.5
                
                scenario_ltv[segment] = base_ltv * ltv_multiplier
            
            ltv_data[scenario] = scenario_ltv
        
        if interactive:
            fig = go.Figure()
            
            segments = list(segment_data.keys())
            x = np.arange(len(segments))
            
            for i, scenario in enumerate(intervention_scenarios):
                ltv_values = [ltv_data[scenario][seg] for seg in segments]
                fig.add_trace(go.Bar(
                    name=scenario,
                    x=segments,
                    y=ltv_values,
                    offsetgroup=i
                ))
            
            fig.update_layout(
                title='Lifetime Value Impact by Intervention',
                xaxis_title='Player Segments',
                yaxis_title='LTV ($)',
                barmode='group',
                height=500
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            segments = list(segment_data.keys())
            x = np.arange(len(segments))
            width = 0.25
            
            for i, scenario in enumerate(intervention_scenarios):
                ltv_values = [ltv_data[scenario][seg] for seg in segments]
                ax.bar(x + i*width, ltv_values, width, label=scenario)
            
            ax.set_xlabel('Player Segments')
            ax.set_ylabel('LTV ($)')
            ax.set_title('Lifetime Value Impact by Intervention')
            ax.set_xticks(x + width)
            ax.set_xticklabels(segments)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig


def main():
    """Example usage of business visualization classes"""
    
    # Generate sample data
    np.random.seed(42)
    n_players = 1000
    
    sample_data = pd.DataFrame({
        'player_id': [f'player_{i}' for i in range(n_players)],
        'total_spent': np.random.exponential(50, n_players),
        'total_playtime_hours': np.random.exponential(100, n_players),
        'friends_count': np.random.poisson(5, n_players),
        'last_login_days_ago': np.random.exponential(7, n_players),
        'churned': np.random.binomial(1, 0.2, n_players)
    })
    
    # Initialize visualizers
    dashboard = BusinessDashboard()
    roi_plotter = ROIAnalysisPlotter()
    segmentation_viz = PlayerSegmentationVisualizer()
    revenue_plotter = RevenueImpactPlotter()
    
    print("Business visualization classes initialized successfully!")
    
    # Example usage (commented out for headless environment)
    # dashboard.create_executive_dashboard(sample_data)
    # segmentation_viz.plot_player_segments(sample_data)

if __name__ == "__main__":
    main()