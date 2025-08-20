"""
Complete Gaming Player Behavior Analysis & Churn Prediction Pipeline

This script runs the entire analysis pipeline from data collection to model deployment,
demonstrating end-to-end data science workflow in a production-ready format.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu

Usage:
    python scripts/run_complete_analysis.py [--environment dev|prod] [--quick-run]
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.data.data_collector import DataCollectionPipeline, SyntheticDataGenerator
from src.data.data_processor import DataCleaner, DataValidator, DataTransformer
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import ModelTrainer, ModelEvaluator
from src.utils.config import ConfigManager
from src.utils.logger import LoggerSetup, PerformanceLogger, ModelLogger, BusinessLogger
from database.setup_database import GamingAnalyticsDB, setup_database
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveAnalysisPipeline:
    """Complete analysis pipeline orchestrator"""
    
    def __init__(self, environment: str = 'development', quick_run: bool = False):
        self.environment = environment
        self.quick_run = quick_run
        self.results = {}
        
        # Setup configuration and logging
        self.config_manager = ConfigManager(environment)
        self.config = self.config_manager.get_config()
        
        # Setup loggers
        self.setup = LoggerSetup(environment)
        self.main_logger = self.setup.get_logger('main')
        self.perf_logger = PerformanceLogger()
        self.model_logger = ModelLogger()
        self.business_logger = BusinessLogger()
        
        # Create output directories
        self.create_directories()
        
        self.main_logger.info(f"🎮 Starting Gaming Analytics Pipeline - Environment: {environment}")
        
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            'data/raw', 'data/processed', 'data/external',
            'models', 'reports/figures', 'logs', 'scripts/outputs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        self.main_logger.info("📁 Created project directories")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        
        pipeline_start_time = time.time()
        self.main_logger.info("🚀 Starting complete analysis pipeline...")
        
        try:
            # Step 1: Data Collection and Setup
            with self.perf_logger.log_block_timing("Step 1: Data Collection"):
                self.step1_data_collection()
            
            # Step 2: Data Processing and Validation
            with self.perf_logger.log_block_timing("Step 2: Data Processing"):
                self.step2_data_processing()
            
            # Step 3: Exploratory Data Analysis
            with self.perf_logger.log_block_timing("Step 3: Exploratory Analysis"):
                self.step3_exploratory_analysis()
            
            # Step 4: Feature Engineering
            with self.perf_logger.log_block_timing("Step 4: Feature Engineering"):
                self.step4_feature_engineering()
            
            # Step 5: Statistical Analysis
            with self.perf_logger.log_block_timing("Step 5: Statistical Analysis"):
                self.step5_statistical_analysis()
            
            # Step 6: Model Development
            with self.perf_logger.log_block_timing("Step 6: Model Training"):
                self.step6_model_development()
            
            # Step 7: Model Evaluation
            with self.perf_logger.log_block_timing("Step 7: Model Evaluation"):
                self.step7_model_evaluation()
            
            # Step 8: Business Analysis
            with self.perf_logger.log_block_timing("Step 8: Business Analysis"):
                self.step8_business_analysis()
            
            # Step 9: Generate Reports
            with self.perf_logger.log_block_timing("Step 9: Report Generation"):
                self.step9_generate_reports()
            
            # Pipeline summary
            total_time = time.time() - pipeline_start_time
            self.results['pipeline_summary'] = {
                'total_execution_time': total_time,
                'environment': self.environment,
                'quick_run': self.quick_run,
                'completion_time': datetime.now().isoformat(),
                'success': True
            }
            
            self.main_logger.info(f"✅ Pipeline completed successfully in {total_time:.2f} seconds")
            return self.results
            
        except Exception as e:
            self.main_logger.error(f"❌ Pipeline failed: {str(e)}")
            self.results['pipeline_summary'] = {
                'success': False,
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            }
            raise
    
    def step1_data_collection(self):
        """Step 1: Data Collection and Database Setup"""
        self.main_logger.info("📊 Step 1: Data Collection and Database Setup")
        
        # Setup database
        db = setup_database("gaming_analytics.db", sample_data=True)
        
        # Initialize data collection pipeline
        n_players = 1000 if self.quick_run else self.config.data.synthetic_players
        
        pipeline = DataCollectionPipeline()
        collected_data = pipeline.run_full_collection(n_synthetic_players=n_players)
        
        # Store results
        self.results['data_collection'] = {
            'datasets_collected': len(collected_data),
            'total_records': sum(len(df) for df in collected_data.values()),
            'synthetic_players': n_players
        }
        
        # Store data for next steps
        self.collected_data = collected_data
        self.database = db
        
        self.main_logger.info(f"✅ Collected {len(collected_data)} datasets with {self.results['data_collection']['total_records']} total records")
    
    def step2_data_processing(self):
        """Step 2: Data Processing and Validation"""
        self.main_logger.info("🔧 Step 2: Data Processing and Validation")
        
        # Initialize processors
        cleaner = DataCleaner()
        validator = DataValidator()
        transformer = DataTransformer()
        
        processing_results = {}
        
        # Process each dataset
        for dataset_name, df in self.collected_data.items():
            self.main_logger.info(f"Processing {dataset_name}...")
            
            # Clean data
            if dataset_name == 'synthetic_players':
                cleaned_df = cleaner.clean_player_data(df)
            elif dataset_name == 'steam_games':
                cleaned_df = cleaner.clean_game_data(df)
            elif dataset_name == 'recommendations':
                cleaned_df = cleaner.clean_recommendations_data(df)
            else:
                cleaned_df = df  # Skip cleaning for unknown datasets
            
            # Validate data
            validation_results = validator.validate_dataset(cleaned_df, dataset_name)
            
            # Transform data for analysis
            transformed_df = transformer.prepare_for_analysis(cleaned_df)
            
            processing_results[dataset_name] = {
                'original_shape': df.shape,
                'cleaned_shape': cleaned_df.shape,
                'transformed_shape': transformed_df.shape,
                'data_quality_score': validation_results['summary']['data_quality_score'],
                'issues_found': len(validation_results['issues'])
            }
            
            # Store processed data
            self.collected_data[dataset_name] = transformed_df
        
        self.results['data_processing'] = processing_results
        
        # Get cleaning report
        cleaning_report = cleaner.get_cleaning_report()
        self.results['cleaning_report'] = cleaning_report
        
        self.main_logger.info("✅ Data processing and validation completed")
    
    def step3_exploratory_analysis(self):
        """Step 3: Exploratory Data Analysis"""
        self.main_logger.info("🔍 Step 3: Exploratory Data Analysis")
        
        # Focus on player data for main analysis
        player_data = self.collected_data.get('synthetic_players')
        if player_data is None:
            self.main_logger.warning("No player data available for EDA")
            return
        
        # Basic statistics
        eda_results = {
            'dataset_shape': player_data.shape,
            'missing_values': player_data.isnull().sum().to_dict(),
            'churn_distribution': player_data['churned'].value_counts().to_dict(),
            'churn_rate': player_data['churned'].mean(),
        }
        
        # Correlation analysis
        numeric_cols = player_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = player_data[numeric_cols].corr()
        churn_correlations = correlation_matrix['churned'].sort_values(key=abs, ascending=False)
        
        eda_results['top_churn_correlations'] = churn_correlations.head(10).to_dict()
        
        # Feature distributions
        feature_stats = {}
        key_features = ['total_playtime_hours', 'total_spent', 'friends_count', 
                       'avg_session_duration', 'achievements_unlocked']
        
        for feature in key_features:
            if feature in player_data.columns:
                feature_stats[feature] = {
                    'mean': float(player_data[feature].mean()),
                    'median': float(player_data[feature].median()),
                    'std': float(player_data[feature].std()),
                    'min': float(player_data[feature].min()),
                    'max': float(player_data[feature].max())
                }
        
        eda_results['feature_statistics'] = feature_stats
        
        # Genre and platform analysis
        if 'favorite_genre' in player_data.columns:
            genre_churn = player_data.groupby('favorite_genre')['churned'].mean().to_dict()
            eda_results['churn_by_genre'] = genre_churn
        
        if 'platform_preference' in player_data.columns:
            platform_churn = player_data.groupby('platform_preference')['churned'].mean().to_dict()
            eda_results['churn_by_platform'] = platform_churn
        
        self.results['exploratory_analysis'] = eda_results
        
        self.main_logger.info(f"✅ EDA completed - Churn rate: {eda_results['churn_rate']:.2%}")
    
    def step4_feature_engineering(self):
        """Step 4: Feature Engineering"""
        self.main_logger.info("⚙️ Step 4: Feature Engineering")
        
        player_data = self.collected_data.get('synthetic_players')
        if player_data is None:
            self.main_logger.warning("No player data available for feature engineering")
            return
        
        # Apply feature engineering
        feature_engineer = FeatureEngineer(player_data)
        engineered_data = feature_engineer.create_all_features()
        
        # Store engineered data
        self.engineered_data = engineered_data
        
        # Feature engineering results
        feature_results = {
            'original_features': len(player_data.columns),
            'engineered_features': len(engineered_data.columns),
            'new_features_created': len(feature_engineer.engineered_features),
            'feature_groups': feature_engineer.get_feature_importance_groups()
        }
        
        # Count features by category
        feature_counts = {}
        for category, features in feature_results['feature_groups'].items():
            feature_counts[category] = len(features)
        
        feature_results['features_by_category'] = feature_counts
        
        self.results['feature_engineering'] = feature_results
        
        self.main_logger.info(f"✅ Created {feature_results['new_features_created']} new features")
    
    def step5_statistical_analysis(self):
        """Step 5: Statistical Analysis"""
        self.main_logger.info("📈 Step 5: Statistical Analysis")
        
        if not hasattr(self, 'engineered_data'):
            self.main_logger.warning("No engineered data available for statistical analysis")
            return
        
        from scipy import stats
        
        statistical_results = {}
        
        # T-test: Session duration vs churn
        churned_sessions = self.engineered_data[self.engineered_data['churned'] == 1]['avg_session_duration']
        retained_sessions = self.engineered_data[self.engineered_data['churned'] == 0]['avg_session_duration']
        
        if len(churned_sessions) > 0 and len(retained_sessions) > 0:
            t_stat, p_value = stats.ttest_ind(churned_sessions, retained_sessions)
            statistical_results['session_duration_ttest'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'churned_mean': float(churned_sessions.mean()),
                'retained_mean': float(retained_sessions.mean())
            }
        
        # Mann-Whitney U test: Spending vs churn
        churned_spending = self.engineered_data[self.engineered_data['churned'] == 1]['total_spent']
        retained_spending = self.engineered_data[self.engineered_data['churned'] == 0]['total_spent']
        
        if len(churned_spending) > 0 and len(retained_spending) > 0:
            u_stat, p_value = stats.mannwhitneyu(churned_spending, retained_spending, alternative='two-sided')
            statistical_results['spending_mannwhitney'] = {
                'u_statistic': float(u_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'churned_median': float(churned_spending.median()),
                'retained_median': float(retained_spending.median())
            }
        
        # Chi-square test: Social engagement vs churn
        social_engaged = (self.engineered_data['friends_count'] > self.engineered_data['friends_count'].median()).astype(int)
        contingency_table = pd.crosstab(social_engaged, self.engineered_data['churned'])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        statistical_results['social_engagement_chi2'] = {
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < 0.05
        }
        
        self.results['statistical_analysis'] = statistical_results
        
        self.main_logger.info("✅ Statistical analysis completed")
    
    def step6_model_development(self):
        """Step 6: Model Development and Training"""
        self.main_logger.info("🤖 Step 6: Model Development and Training")
        
        if not hasattr(self, 'engineered_data'):
            self.main_logger.warning("No engineered data available for modeling")
            return
        
        # Prepare data for modeling
        X = self.engineered_data.drop(['churned', 'player_id'], axis=1, errors='ignore')
        y = self.engineered_data['churned']
        
        self.main_logger.info(f"Training models on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Initialize model trainer
        trainer = ModelTrainer(random_state=self.config.model.random_state)
        
        # Train models (limited set for quick run)
        if self.quick_run:
            # Quick run: only train essential models
            trainer.model_configs = {
                'logistic_regression': trainer.prepare_models()['logistic_regression'],
                'random_forest': trainer.prepare_models()['random_forest']
            }
        
        # Start model training logging
        run_id = self.model_logger.log_training_start('ensemble', {
            'models': list(trainer.model_configs.keys()) if hasattr(trainer, 'model_configs') else ['all'],
            'quick_run': self.quick_run,
            'dataset_shape': X.shape
        })
        
        try:
            # Train all models
            model_results = trainer.train_all_models(X, y, 
                                                   test_size=self.config.model.test_size,
                                                   cv_folds=3 if self.quick_run else self.config.model.cv_folds)
            
            # Save models
            model_path = trainer.save_models()
            
            # Log successful training
            best_model_metrics = trainer.best_model['metrics'] if trainer.best_model else {}
            self.model_logger.log_training_complete(run_id, best_model_metrics, str(model_path))
            
            # Store results
            self.results['model_development'] = {
                'models_trained': len(model_results),
                'best_model': trainer.best_model['name'] if trainer.best_model else None,
                'best_model_metrics': best_model_metrics,
                'model_path': str(model_path)
            }
            
            # Store trainer for evaluation
            self.trainer = trainer
            
            self.main_logger.info(f"✅ Trained {len(model_results)} models. Best: {trainer.best_model['name'] if trainer.best_model else 'None'}")
            
        except Exception as e:
            self.model_logger.log_training_failed(run_id, e)
            raise
    
    def step7_model_evaluation(self):
        """Step 7: Model Evaluation and Validation"""
        self.main_logger.info("📊 Step 7: Model Evaluation and Validation")
        
        if not hasattr(self, 'trainer'):
            self.main_logger.warning("No trained models available for evaluation")
            return
        
        # Generate model comparison report
        comparison_df = self.trainer.generate_model_comparison_report()
        
        # Get feature importance
        feature_importance = self.trainer.get_feature_importance()
        
        # Business evaluation
        evaluator = ModelEvaluator(self.trainer)
        business_metrics = evaluator.generate_business_metrics()
        threshold_optimization = evaluator.threshold_optimization()
        
        # Store evaluation results
        self.results['model_evaluation'] = {
            'model_comparison': comparison_df.to_dict('records'),
            'feature_importance_available': len(feature_importance) > 0,
            'business_metrics': business_metrics,
            'threshold_optimization': threshold_optimization
        }
        
        # Log model performance
        if self.trainer.best_model:
            self.model_logger.log_model_performance(
                self.trainer.best_model['name'],
                'test_set',
                self.trainer.best_model['metrics']
            )
            
            # Log feature importance for best model
            if feature_importance and self.trainer.best_model['name'] in feature_importance:
                importance_dict = dict(zip(
                    self.trainer.feature_names,
                    feature_importance[self.trainer.best_model['name']]
                ))
                self.model_logger.log_feature_importance(
                    self.trainer.best_model['name'],
                    importance_dict
                )
        
        self.main_logger.info("✅ Model evaluation completed")
    
    def step8_business_analysis(self):
        """Step 8: Business Impact Analysis"""
        self.main_logger.info("💼 Step 8: Business Impact Analysis")
        
        if not hasattr(self, 'trainer') or not self.trainer.best_model:
            self.main_logger.warning("No trained models available for business analysis")
            return
        
        # Player segmentation analysis
        if hasattr(self, 'engineered_data'):
            # Create value and risk scores for segmentation
            self.engineered_data['player_value'] = (
                np.log1p(self.engineered_data.get('total_spent', 0)) * 0.4 +
                np.log1p(self.engineered_data.get('total_playtime_hours', 0)) * 0.3 +
                self.engineered_data.get('loyalty_score', 0) * 0.3
            )
            
            # Segment players
            value_quartiles = pd.qcut(self.engineered_data['player_value'], 
                                    q=4, labels=['Low', 'Medium', 'High', 'VIP'])
            
            if 'total_risk_score' in self.engineered_data.columns:
                risk_quartiles = pd.qcut(self.engineered_data['total_risk_score'], 
                                       q=4, labels=['Low', 'Medium', 'High', 'Critical'])
            else:
                risk_quartiles = pd.Series(['Medium'] * len(self.engineered_data))
            
            # Count segments
            segments = {}
            for value in ['Low', 'Medium', 'High', 'VIP']:
                for risk in ['Low', 'Medium', 'High', 'Critical']:
                    count = ((value_quartiles == value) & (risk_quartiles == risk)).sum()
                    segments[f"{value}_Value_{risk}_Risk"] = count
        else:
            segments = {'Unknown': 100}  # Fallback
        
        # ROI Analysis
        total_players = len(self.engineered_data) if hasattr(self, 'engineered_data') else 1000
        if hasattr(self, 'trainer') and self.trainer.best_model:
            predicted_churners = sum(self.trainer.best_model['metrics'].get('true_positives', 0) + 
                                   self.trainer.best_model['metrics'].get('false_positives', 0) for _ in [1])
        else:
            predicted_churners = int(total_players * 0.3)  # Estimate
        
        # Business calculations
        cost_per_intervention = 10
        revenue_per_retained_player = 50
        intervention_success_rate = 0.25
        
        intervention_cost = predicted_churners * cost_per_intervention
        potential_revenue_saved = predicted_churners * intervention_success_rate * revenue_per_retained_player
        net_benefit = potential_revenue_saved - intervention_cost
        roi_percentage = (net_benefit / intervention_cost) * 100 if intervention_cost > 0 else 0
        
        business_analysis = {
            'player_segmentation': segments,
            'roi_analysis': {
                'total_players': total_players,
                'predicted_churners': predicted_churners,
                'intervention_cost': intervention_cost,
                'potential_revenue_saved': potential_revenue_saved,
                'net_benefit': net_benefit,
                'roi_percentage': roi_percentage
            },
            'recommendations': [
                "Implement real-time churn alerts for high-risk players",
                "Create targeted retention campaigns for players inactive >14 days",
                "Develop social features to increase friend connections",
                "Launch loyalty programs for high-value players"
            ]
        }
        
        self.results['business_analysis'] = business_analysis
        
        # Log business metrics
        self.business_logger.log_player_segmentation(segments)
        self.business_logger.log_roi_analysis(
            intervention_cost, potential_revenue_saved, roi_percentage/100
        )
        
        self.main_logger.info(f"✅ Business analysis completed - ROI: {roi_percentage:.1f}%")
    
    def step9_generate_reports(self):
        """Step 9: Generate Comprehensive Reports"""
        self.main_logger.info("📋 Step 9: Generating Reports")
        
        # Create comprehensive summary report
        summary_report = {
            'project_info': {
                'title': 'Gaming Player Behavior Analysis & Churn Prediction',
                'author': 'Rushikesh Dhumal',
                'email': 'r.dhumal@rutgers.edu',
                'completion_date': datetime.now().isoformat(),
                'environment': self.environment,
                'quick_run': self.quick_run
            },
            'executive_summary': self._generate_executive_summary(),
            'technical_details': self._generate_technical_summary(),
            'business_impact': self.results.get('business_analysis', {}),
            'performance_metrics': self.perf_logger.get_performance_summary()
        }
        
        # Save summary report
        report_file = Path('scripts/outputs/analysis_summary_report.json')
        with open(report_file, 'w') as f:
            import json
            json.dump(summary_report, f, indent=2, default=str)
        
        # Generate markdown report
        markdown_report = self._generate_markdown_report(summary_report)
        markdown_file = Path('scripts/outputs/analysis_report.md')
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
        
        self.results['reports_generated'] = {
            'summary_report': str(report_file),
            'markdown_report': str(markdown_file),
            'reports_created': 2
        }
        
        self.main_logger.info(f"✅ Generated reports: {report_file}, {markdown_file}")
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary"""
        
        # Extract key metrics
        churn_rate = self.results.get('exploratory_analysis', {}).get('churn_rate', 0)
        best_model = self.results.get('model_development', {}).get('best_model', 'Unknown')
        best_accuracy = 0
        
        if hasattr(self, 'trainer') and self.trainer.best_model:
            best_accuracy = self.trainer.best_model['metrics'].get('roc_auc', 0)
        
        roi_percentage = self.results.get('business_analysis', {}).get('roi_analysis', {}).get('roi_percentage', 0)
        potential_savings = self.results.get('business_analysis', {}).get('roi_analysis', {}).get('potential_revenue_saved', 0)
        
        return {
            'key_findings': [
                f"Dataset churn rate: {churn_rate:.1%}",
                f"Best model: {best_model} with {best_accuracy:.1%} ROC-AUC",
                f"Potential annual savings: ${potential_savings:,.0f}",
                f"ROI on intervention: {roi_percentage:.1f}%"
            ],
            'datasets_analyzed': len(self.results.get('data_collection', {}).get('datasets_collected', 0)),
            'models_trained': self.results.get('model_development', {}).get('models_trained', 0),
            'features_engineered': self.results.get('feature_engineering', {}).get('new_features_created', 0)
        }
    
    def _generate_technical_summary(self) -> Dict[str, Any]:
        """Generate technical summary"""
        return {
            'data_collection': self.results.get('data_collection', {}),
            'data_processing': self.results.get('data_processing', {}),
            'feature_engineering': self.results.get('feature_engineering', {}),
            'model_development': self.results.get('model_development', {}),
            'model_evaluation': self.results.get('model_evaluation', {}),
            'statistical_analysis': self.results.get('statistical_analysis', {})
        }
    
    def _generate_markdown_report(self, summary_report: Dict[str, Any]) -> str:
        """Generate markdown report"""
        
        report = f"""# Gaming Player Behavior Analysis & Churn Prediction
## Analysis Report

**Author:** {summary_report['project_info']['author']}  
**Email:** {summary_report['project_info']['email']}  
**Date:** {summary_report['project_info']['completion_date'][:10]}  
**Environment:** {summary_report['project_info']['environment']}

## Executive Summary

### Key Findings
"""
        
        for finding in summary_report['executive_summary']['key_findings']:
            report += f"- {finding}\n"
        
        report += f"""

### Project Statistics
- **Datasets Analyzed:** {summary_report['executive_summary']['datasets_analyzed']}
- **Models Trained:** {summary_report['executive_summary']['models_trained']}
- **Features Engineered:** {summary_report['executive_summary']['features_engineered']}

## Business Impact

### ROI Analysis
"""
        
        roi_data = summary_report['business_impact'].get('roi_analysis', {})
        report += f"""- **Total Players:** {roi_data.get('total_players', 0):,}
- **Predicted Churners:** {roi_data.get('predicted_churners', 0):,}
- **Intervention Cost:** ${roi_data.get('intervention_cost', 0):,.2f}
- **Potential Revenue Saved:** ${roi_data.get('potential_revenue_saved', 0):,.2f}
- **Net Benefit:** ${roi_data.get('net_benefit', 0):,.2f}
- **ROI:** {roi_data.get('roi_percentage', 0):.1f}%

## Recommendations
"""
        
        for rec in summary_report['business_impact'].get('recommendations', []):
            report += f"- {rec}\n"
        
        report += """

## Technical Implementation

This analysis demonstrates:
- ✅ End-to-end data science workflow
- ✅ Advanced feature engineering
- ✅ Multiple machine learning models
- ✅ Statistical hypothesis testing
- ✅ Business impact analysis
- ✅ Production-ready code structure

*Generated by automated analysis pipeline*
"""
        
        return report

def main():
    """Main function to run the complete analysis"""
    
    parser = argparse.ArgumentParser(description='Run complete gaming analytics pipeline')
    parser.add_argument('--environment', choices=['development', 'testing', 'staging', 'production'],
                       default='development', help='Environment to run in')
    parser.add_argument('--quick-run', action='store_true',
                       help='Run with reduced dataset and model set for faster execution')
    parser.add_argument('--output-dir', default='scripts/outputs',
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("🎮 Gaming Player Behavior Analysis & Churn Prediction")
    print("=" * 60)
    print(f"Environment: {args.environment}")
    print(f"Quick Run: {args.quick_run}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)
    
    try:
        # Initialize and run pipeline
        pipeline = ComprehensiveAnalysisPipeline(
            environment=args.environment,
            quick_run=args.quick_run
        )
        
        results = pipeline.run_complete_pipeline()
        
        # Print final summary
        print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        summary = results['pipeline_summary']
        print(f"⏱️  Total Execution Time: {summary['total_execution_time']:.2f} seconds")
        print(f"📊 Environment: {summary['environment']}")
        print(f"🚀 Quick Run: {summary['quick_run']}")
        
        # Key results
        if 'executive_summary' in results.get('reports_generated', {}):
            exec_summary = results['reports_generated']['executive_summary']
            print(f"\n📈 KEY RESULTS:")
            for finding in exec_summary.get('key_findings', []):
                print(f"   • {finding}")
        
        # Business impact
        if 'business_analysis' in results:
            roi_data = results['business_analysis'].get('roi_analysis', {})
            print(f"\n💰 BUSINESS IMPACT:")
            print(f"   • ROI: {roi_data.get('roi_percentage', 0):.1f}%")
            print(f"   • Potential Savings: ${roi_data.get('potential_revenue_saved', 0):,.0f}")
            print(f"   • Net Benefit: ${roi_data.get('net_benefit', 0):,.0f}")
        
        # Model performance
        if 'model_development' in results:
            model_info = results['model_development']
            print(f"\n🤖 MODEL PERFORMANCE:")
            print(f"   • Best Model: {model_info.get('best_model', 'Unknown')}")
            
            if 'best_model_metrics' in model_info:
                metrics = model_info['best_model_metrics']
                print(f"   • ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
                print(f"   • Accuracy: {metrics.get('accuracy', 0):.3f}")
                print(f"   • Precision: {metrics.get('precision', 0):.3f}")
                print(f"   • Recall: {metrics.get('recall', 0):.3f}")
        
        # Reports generated
        if 'reports_generated' in results:
            reports = results['reports_generated']
            print(f"\n📋 REPORTS GENERATED:")
            print(f"   • Summary Report: {reports.get('summary_report', 'N/A')}")
            print(f"   • Markdown Report: {reports.get('markdown_report', 'N/A')}")
        
        print("\n🎯 NEXT STEPS:")
        print("   1. Review generated reports in scripts/outputs/")
        print("   2. Check trained models in models/ directory")
        print("   3. Examine logs in logs/ directory")
        print("   4. Use CLI tools for predictions: gaming-churn-predict")
        
        print(f"\n✅ Analysis pipeline completed successfully!")
        print("🏆 Ready for production deployment and portfolio showcase!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED!")
        print(f"Error: {str(e)}")
        print(f"Check logs in logs/ directory for detailed error information")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)