"""
Visualization Module

This module provides comprehensive visualization capabilities for gaming analytics
including EDA plots, model performance visualization, and business dashboards.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

from .eda_plots import (
    PlayerBehaviorVisualizer,
    ChurnAnalysisPlotter,
    FeatureDistributionPlotter,
    CorrelationVisualizer
)

from .model_plots import (
    ModelPerformancePlotter,
    FeatureImportancePlotter,
    ModelComparisonVisualizer,
    PredictionAnalysisPlotter
)

from .business_dashboard import (
    BusinessDashboard,
    ROIAnalysisPlotter,
    PlayerSegmentationVisualizer,
    RevenueImpactPlotter
)

__all__ = [
    # EDA Visualization
    "PlayerBehaviorVisualizer",
    "ChurnAnalysisPlotter",
    "FeatureDistributionPlotter",
    "CorrelationVisualizer",
    
    # Model Visualization
    "ModelPerformancePlotter",
    "FeatureImportancePlotter", 
    "ModelComparisonVisualizer",
    "PredictionAnalysisPlotter",
    
    # Business Visualization
    "BusinessDashboard",
    "ROIAnalysisPlotter",
    "PlayerSegmentationVisualizer",
    "RevenueImpactPlotter",
]

# Visualization configuration
PLOT_STYLES = {
    'presentation': 'seaborn-v0_8-darkgrid',
    'publication': 'seaborn-v0_8-whitegrid',
    'dashboard': 'seaborn-v0_8-dark',
    'minimal': 'seaborn-v0_8-white'
}

COLOR_PALETTES = {
    'churn': ['#2E8B57', '#DC143C'],  # Green for retained, Red for churned
    'risk': ['#32CD32', '#FFD700', '#FF6347', '#8B0000'],  # Low to Critical risk
    'revenue': ['#4169E1', '#32CD32', '#FFD700'],  # Revenue tiers
    'engagement': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
}

DEFAULT_FIGSIZE = {
    'single': (10, 6),
    'wide': (15, 6), 
    'tall': (10, 12),
    'dashboard': (20, 15),
    'comparison': (12, 8)
}

def get_visualization_info():
    """Return visualization module information"""
    return {
        "module": "visualization",
        "version": "1.0.0",
        "plot_styles": list(PLOT_STYLES.keys()),
        "color_palettes": list(COLOR_PALETTES.keys()),
        "default_figsizes": list(DEFAULT_FIGSIZE.keys()),
        "maintainer": "Rushikesh Dhumal"
    }

def setup_plot_style(style: str = 'presentation'):
    """Setup matplotlib/seaborn plotting style"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if style in PLOT_STYLES:
        plt.style.use(PLOT_STYLES[style])
    
    # Set default parameters
    plt.rcParams.update({
        'figure.figsize': DEFAULT_FIGSIZE['single'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })
    
    # Set color palette
    sns.set_palette("husl")