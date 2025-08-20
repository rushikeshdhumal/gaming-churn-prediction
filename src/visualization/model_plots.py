"""
Model Performance Visualization Module

Comprehensive visualization tools for model evaluation, comparison, and analysis.

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
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

class ModelPerformancePlotter:
    """
    Visualize model performance metrics and evaluation results
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e'
        }
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             labels: List[str] = None, interactive: bool = True) -> Any:
        """Plot confusion matrix with annotations"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = ['Retained', 'Churned']
        
        if interactive:
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={'size': 16},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                width=500,
                height=500
            )
            
            return fig
        else:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            return plt.gcf()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = 'Model', interactive: bool = True) -> Any:
        """Plot ROC curve with AUC score"""
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        if interactive:
            fig = go.Figure()
            
            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.3f})',
                line=dict(color=self.colors['primary'], width=2)
            ))
            
            # Random classifier line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=600,
                height=600,
                showlegend=True
            )
            
            return fig
        else:
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color=self.colors['primary'], lw=2, 
                    label=f'{model_name} (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', 
                    label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            return plt.gcf()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = 'Model', interactive: bool = True) -> Any:
        """Plot Precision-Recall curve"""
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        if interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{model_name} (AP = {avg_precision:.3f})',
                line=dict(color=self.colors['primary'], width=2)
            ))
            
            # Baseline (random classifier)
            baseline = np.mean(y_true)
            fig.add_shape(
                type="line",
                x0=0, y0=baseline, x1=1, y1=baseline,
                line=dict(color="red", width=1, dash="dash")
            )
            
            fig.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                width=600,
                height=600,
                showlegend=True
            )
            
            return fig
        else:
            plt.figure(figsize=(8, 8))
            plt.plot(recall, precision, color=self.colors['primary'], lw=2,
                    label=f'{model_name} (AP = {avg_precision:.3f})')
            plt.axhline(y=np.mean(y_true), color='red', linestyle='--', 
                       label=f'Baseline (AP = {np.mean(y_true):.3f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            return plt.gcf()
    
    def plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  labels: List[str] = None, interactive: bool = True) -> Any:
        """Visualize classification report metrics"""
        
        from sklearn.metrics import classification_report
        
        if labels is None:
            labels = ['Retained', 'Churned']
        
        # Get classification report as dict
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        
        # Extract metrics for each class
        metrics_data = []
        for label in labels:
            if label in report:
                metrics_data.append({
                    'Class': label,
                    'Precision': report[label]['precision'],
                    'Recall': report[label]['recall'],
                    'F1-Score': report[label]['f1-score']
                })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        if interactive:
            fig = go.Figure()
            
            metrics = ['Precision', 'Recall', 'F1-Score']
            x = np.arange(len(labels))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=[f"{label}_{metric}" for label in labels],
                    y=df_metrics[metric],
                    marker_color=px.colors.qualitative.Set1[i]
                ))
            
            fig.update_layout(
                title='Classification Report',
                xaxis_title='Classes',
                yaxis_title='Score',
                barmode='group',
                width=600,
                height=400
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(labels))
            width = 0.25
            
            metrics = ['Precision', 'Recall', 'F1-Score']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, metric in enumerate(metrics):
                ax.bar(x + i*width, df_metrics[metric], width, 
                      label=metric, color=colors[i], alpha=0.8)
            
            ax.set_xlabel('Classes')
            ax.set_ylabel('Score')
            ax.set_title('Classification Report')
            ax.set_xticks(x + width)
            ax.set_xticklabels(labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def plot_learning_curves(self, train_scores: np.ndarray, val_scores: np.ndarray,
                            train_sizes: np.ndarray, metric_name: str = 'Score',
                            interactive: bool = True) -> Any:
        """Plot learning curves showing training vs validation performance"""
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        if interactive:
            fig = go.Figure()
            
            # Training scores
            fig.add_trace(go.Scatter(
                x=train_sizes, y=train_mean,
                mode='lines+markers',
                name='Training Score',
                line=dict(color=self.colors['primary']),
                error_y=dict(type='data', array=train_std, visible=True)
            ))
            
            # Validation scores
            fig.add_trace(go.Scatter(
                x=train_sizes, y=val_mean,
                mode='lines+markers',
                name='Validation Score',
                line=dict(color=self.colors['secondary']),
                error_y=dict(type='data', array=val_std, visible=True)
            ))
            
            fig.update_layout(
                title='Learning Curves',
                xaxis_title='Training Set Size',
                yaxis_title=metric_name,
                width=800,
                height=500
            )
            
            return fig
        else:
            plt.figure(figsize=self.figsize)
            
            plt.plot(train_sizes, train_mean, 'o-', color=self.colors['primary'], 
                    label='Training Score')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                           alpha=0.1, color=self.colors['primary'])
            
            plt.plot(train_sizes, val_mean, 'o-', color=self.colors['secondary'],
                    label='Validation Score')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                           alpha=0.1, color=self.colors['secondary'])
            
            plt.xlabel('Training Set Size')
            plt.ylabel(metric_name)
            plt.title('Learning Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            return plt.gcf()
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               interactive: bool = True) -> Any:
        """Plot performance metrics across different decision thresholds"""
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            if np.sum(y_pred_thresh) == 0 or np.sum(y_pred_thresh) == len(y_pred_thresh):
                continue
            
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(y_true, y_pred_thresh, zero_division=0)
            recall = recall_score(y_true, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        valid_thresholds = thresholds[:len(precisions)]
        
        if interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=valid_thresholds, y=precisions,
                mode='lines+markers',
                name='Precision',
                line=dict(color=self.colors['primary'])
            ))
            
            fig.add_trace(go.Scatter(
                x=valid_thresholds, y=recalls,
                mode='lines+markers',
                name='Recall',
                line=dict(color=self.colors['secondary'])
            ))
            
            fig.add_trace(go.Scatter(
                x=valid_thresholds, y=f1_scores,
                mode='lines+markers',
                name='F1-Score',
                line=dict(color=self.colors['success'])
            ))
            
            fig.update_layout(
                title='Performance vs Decision Threshold',
                xaxis_title='Decision Threshold',
                yaxis_title='Score',
                width=800,
                height=500
            )
            
            return fig
        else:
            plt.figure(figsize=self.figsize)
            
            plt.plot(valid_thresholds, precisions, 'o-', label='Precision', 
                    color=self.colors['primary'])
            plt.plot(valid_thresholds, recalls, 'o-', label='Recall',
                    color=self.colors['secondary'])
            plt.plot(valid_thresholds, f1_scores, 'o-', label='F1-Score',
                    color=self.colors['success'])
            
            plt.xlabel('Decision Threshold')
            plt.ylabel('Score')
            plt.title('Performance vs Decision Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            return plt.gcf()


class FeatureImportancePlotter:
    """
    Visualize feature importance from various models and methods
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray,
                               title: str = 'Feature Importance', top_n: int = 20,
                               interactive: bool = True) -> Any:
        """Plot feature importance scores"""
        
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)[::-1][:top_n]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]
        
        if interactive:
            fig = go.Figure(data=go.Bar(
                x=sorted_scores,
                y=sorted_features,
                orientation='h',
                marker_color=px.colors.sequential.Viridis
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=max(400, top_n * 25),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
        else:
            plt.figure(figsize=(10, max(6, top_n * 0.3)))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
            bars = plt.barh(range(len(sorted_features)), sorted_scores, color=colors)
            
            plt.yticks(range(len(sorted_features)), 
                      [f.replace('_', ' ').title() for f in sorted_features])
            plt.xlabel('Importance Score')
            plt.title(title)
            plt.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
                plt.text(score + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f}', va='center', fontsize=8)
            
            plt.tight_layout()
            return plt.gcf()
    
    def plot_feature_importance_comparison(self, importance_dict: Dict[str, Dict[str, float]],
                                         top_n: int = 15, interactive: bool = True) -> Any:
        """Compare feature importance across multiple models"""
        
        # Get common features
        all_features = set()
        for model_importances in importance_dict.values():
            all_features.update(model_importances.keys())
        
        # Calculate average importance across models
        feature_avg_importance = {}
        for feature in all_features:
            importances = [model_imp.get(feature, 0) for model_imp in importance_dict.values()]
            feature_avg_importance[feature] = np.mean(importances)
        
        # Get top features
        top_features = sorted(feature_avg_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:top_n]
        top_feature_names = [f[0] for f in top_features]
        
        if interactive:
            fig = go.Figure()
            
            for model_name, importances in importance_dict.items():
                model_scores = [importances.get(feat, 0) for feat in top_feature_names]
                
                fig.add_trace(go.Bar(
                    name=model_name,
                    x=top_feature_names,
                    y=model_scores
                ))
            
            fig.update_layout(
                title='Feature Importance Comparison Across Models',
                xaxis_title='Features',
                yaxis_title='Importance Score',
                barmode='group',
                height=600,
                xaxis={'tickangle': 45}
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            x = np.arange(len(top_feature_names))
            width = 0.8 / len(importance_dict)
            
            for i, (model_name, importances) in enumerate(importance_dict.items()):
                model_scores = [importances.get(feat, 0) for feat in top_feature_names]
                offset = (i - len(importance_dict)/2 + 0.5) * width
                
                ax.bar(x + offset, model_scores, width, label=model_name, alpha=0.8)
            
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance Score')
            ax.set_title('Feature Importance Comparison Across Models')
            ax.set_xticks(x)
            ax.set_xticklabels([f.replace('_', ' ').title() for f in top_feature_names], 
                              rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
    
    def plot_permutation_importance(self, feature_names: List[str], 
                                   importance_means: np.ndarray,
                                   importance_stds: np.ndarray,
                                   top_n: int = 20, interactive: bool = True) -> Any:
        """Plot permutation importance with error bars"""
        
        # Sort by importance
        sorted_idx = np.argsort(importance_means)[::-1][:top_n]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_means = importance_means[sorted_idx]
        sorted_stds = importance_stds[sorted_idx]
        
        if interactive:
            fig = go.Figure(data=go.Bar(
                x=sorted_means,
                y=sorted_features,
                orientation='h',
                error_x=dict(type='data', array=sorted_stds),
                marker_color='lightblue',
                marker_line_color='darkblue',
                marker_line_width=1
            ))
            
            fig.update_layout(
                title='Permutation Feature Importance',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=max(400, top_n * 25),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
        else:
            plt.figure(figsize=(10, max(6, top_n * 0.3)))
            
            y_pos = np.arange(len(sorted_features))
            bars = plt.barh(y_pos, sorted_means, xerr=sorted_stds, 
                           color='lightblue', edgecolor='darkblue', alpha=0.8)
            
            plt.yticks(y_pos, [f.replace('_', ' ').title() for f in sorted_features])
            plt.xlabel('Importance Score')
            plt.title('Permutation Feature Importance')
            plt.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            return plt.gcf()
    
    def plot_shap_importance(self, shap_values: np.ndarray, feature_names: List[str],
                           top_n: int = 20, interactive: bool = True) -> Any:
        """Plot SHAP feature importance (requires shap library)"""
        
        try:
            import shap
            
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Sort features
            sorted_idx = np.argsort(mean_shap)[::-1][:top_n]
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_importance = mean_shap[sorted_idx]
            
            if interactive:
                fig = go.Figure(data=go.Bar(
                    x=sorted_importance,
                    y=sorted_features,
                    orientation='h',
                    marker_color='coral'
                ))
                
                fig.update_layout(
                    title='SHAP Feature Importance',
                    xaxis_title='Mean |SHAP Value|',
                    yaxis_title='Features',
                    height=max(400, top_n * 25),
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                return fig
            else:
                plt.figure(figsize=(10, max(6, top_n * 0.3)))
                
                y_pos = np.arange(len(sorted_features))
                plt.barh(y_pos, sorted_importance, color='coral', alpha=0.8)
                
                plt.yticks(y_pos, [f.replace('_', ' ').title() for f in sorted_features])
                plt.xlabel('Mean |SHAP Value|')
                plt.title('SHAP Feature Importance')
                plt.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                return plt.gcf()
                
        except ImportError:
            print("SHAP library required for SHAP importance plots")
            return None
    
    def plot_feature_selection_impact(self, n_features_list: List[int], 
                                    performance_scores: List[float],
                                    metric_name: str = 'Performance',
                                    interactive: bool = True) -> Any:
        """Plot impact of feature selection on model performance"""
        
        if interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=n_features_list,
                y=performance_scores,
                mode='lines+markers',
                name=metric_name,
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            # Find optimal number of features
            best_idx = np.argmax(performance_scores)
            best_n_features = n_features_list[best_idx]
            best_score = performance_scores[best_idx]
            
            fig.add_trace(go.Scatter(
                x=[best_n_features],
                y=[best_score],
                mode='markers',
                name=f'Optimal ({best_n_features} features)',
                marker=dict(color='red', size=12, symbol='star')
            ))
            
            fig.update_layout(
                title='Feature Selection Impact on Model Performance',
                xaxis_title='Number of Features',
                yaxis_title=metric_name,
                width=800,
                height=500
            )
            
            return fig
        else:
            plt.figure(figsize=self.figsize)
            
            plt.plot(n_features_list, performance_scores, 'bo-', linewidth=2, markersize=6)
            
            # Highlight optimal point
            best_idx = np.argmax(performance_scores)
            best_n_features = n_features_list[best_idx]
            best_score = performance_scores[best_idx]
            
            plt.plot(best_n_features, best_score, 'r*', markersize=15, 
                    label=f'Optimal ({best_n_features} features)')
            
            plt.xlabel('Number of Features')
            plt.ylabel(metric_name)
            plt.title('Feature Selection Impact on Model Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            return plt.gcf()


class ModelComparisonVisualizer:
    """
    Visualize and compare multiple models' performance
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_model_comparison_metrics(self, model_results: Dict[str, Dict[str, float]],
                                    metrics: List[str] = None, interactive: bool = True) -> Any:
        """Compare multiple models across different metrics"""
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Filter available metrics
        available_metrics = []
        for metric in metrics:
            if any(metric in results for results in model_results.values()):
                available_metrics.append(metric)
        
        if interactive:
            fig = go.Figure()
            
            for metric in available_metrics:
                model_names = []
                metric_values = []
                
                for model_name, results in model_results.items():
                    if metric in results:
                        model_names.append(model_name)
                        metric_values.append(results[metric])
                
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=model_names,
                    y=metric_values
                ))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Models',
                yaxis_title='Score',
                barmode='group',
                height=600
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            model_names = list(model_results.keys())
            x = np.arange(len(model_names))
            width = 0.8 / len(available_metrics)
            
            for i, metric in enumerate(available_metrics):
                metric_values = [model_results[model].get(metric, 0) for model in model_names]
                offset = (i - len(available_metrics)/2 + 0.5) * width
                
                ax.bar(x + offset, metric_values, width, 
                      label=metric.replace('_', ' ').title(), alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
    
    def plot_roc_comparison(self, model_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                           interactive: bool = True) -> Any:
        """Compare ROC curves of multiple models"""
        
        if interactive:
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set1
            
            for i, (model_name, (fpr, tpr, auc)) in enumerate(model_curves.items()):
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {auc:.3f})',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            # Random classifier line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='black', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title='ROC Curve Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=700,
                height=700
            )
            
            return fig
        else:
            plt.figure(figsize=(10, 10))
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(model_curves)))
            
            for (model_name, (fpr, tpr, auc)), color in zip(model_curves.items(), colors):
                plt.plot(fpr, tpr, color=color, lw=2,
                        label=f'{model_name} (AUC = {auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Comparison')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            return plt.gcf()
    
    def plot_precision_recall_comparison(self, model_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                                       interactive: bool = True) -> Any:
        """Compare Precision-Recall curves of multiple models"""
        
        if interactive:
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set1
            
            for i, (model_name, (precision, recall, ap)) in enumerate(model_curves.items()):
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'{model_name} (AP = {ap:.3f})',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig.update_layout(
                title='Precision-Recall Curve Comparison',
                xaxis_title='Recall',
                yaxis_title='Precision',
                width=700,
                height=700
            )
            
            return fig
        else:
            plt.figure(figsize=(10, 10))
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(model_curves)))
            
            for (model_name, (precision, recall, ap)), color in zip(model_curves.items(), colors):
                plt.plot(recall, precision, color=color, lw=2,
                        label=f'{model_name} (AP = {ap:.3f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            return plt.gcf()
    
    def plot_model_ranking(self, model_results: Dict[str, Dict[str, float]],
                          ranking_metric: str = 'roc_auc', interactive: bool = True) -> Any:
        """Rank models by specified metric"""
        
        # Extract ranking metric
        model_scores = []
        for model_name, results in model_results.items():
            if ranking_metric in results:
                model_scores.append((model_name, results[ranking_metric]))
        
        # Sort by score
        model_scores.sort(key=lambda x: x[1], reverse=True)
        model_names, scores = zip(*model_scores)
        
        if interactive:
            colors = ['gold' if i == 0 else 'silver' if i == 1 else 'brown' if i == 2 else 'lightblue' 
                     for i in range(len(model_names))]
            
            fig = go.Figure(data=go.Bar(
                x=list(scores),
                y=list(model_names),
                orientation='h',
                marker_color=colors
            ))
            
            fig.update_layout(
                title=f'Model Ranking by {ranking_metric.replace("_", " ").title()}',
                xaxis_title=ranking_metric.replace('_', ' ').title(),
                yaxis_title='Models',
                height=max(400, len(model_names) * 40),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
        else:
            plt.figure(figsize=(10, max(6, len(model_names) * 0.4)))
            
            colors = ['gold' if i == 0 else 'silver' if i == 1 else '#CD7F32' if i == 2 else 'lightblue' 
                     for i in range(len(model_names))]
            
            bars = plt.barh(range(len(model_names)), scores, color=colors, alpha=0.8)
            
            plt.yticks(range(len(model_names)), model_names)
            plt.xlabel(ranking_metric.replace('_', ' ').title())
            plt.title(f'Model Ranking by {ranking_metric.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3, axis='x')
            
            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, scores)):
                plt.text(score + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            return plt.gcf()


class PredictionAnalysisPlotter:
    """
    Analyze and visualize model predictions and errors
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_prediction_distribution(self, y_pred_proba: np.ndarray, y_true: np.ndarray = None,
                                   interactive: bool = True) -> Any:
        """Plot distribution of prediction probabilities"""
        
        if interactive:
            fig = make_subplots(rows=1, cols=2, subplot_titles=['All Predictions', 'By True Class'])
            
            # Overall distribution
            fig.add_trace(
                go.Histogram(x=y_pred_proba, name='Predictions', nbinsx=50),
                row=1, col=1
            )
            
            # Distribution by true class
            if y_true is not None:
                fig.add_trace(
                    go.Histogram(x=y_pred_proba[y_true == 0], name='True Negative', 
                               nbinsx=25, opacity=0.7),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Histogram(x=y_pred_proba[y_true == 1], name='True Positive', 
                               nbinsx=25, opacity=0.7),
                    row=1, col=2
                )
            
            fig.update_layout(height=400, title_text="Prediction Probability Distribution")
            return fig
        else:
            if y_true is not None:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Overall distribution
                axes[0].hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
                axes[0].set_xlabel('Prediction Probability')
                axes[0].set_ylabel('Frequency')
                axes[0].set_title('All Predictions')
                axes[0].grid(True, alpha=0.3)
                
                # By true class
                axes[1].hist(y_pred_proba[y_true == 0], bins=25, alpha=0.7, 
                           label='True Negative', color='blue')
                axes[1].hist(y_pred_proba[y_true == 1], bins=25, alpha=0.7, 
                           label='True Positive', color='red')
                axes[1].set_xlabel('Prediction Probability')
                axes[1].set_ylabel('Frequency')
                axes[1].set_title('Predictions by True Class')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                return fig
            else:
                plt.figure(figsize=(8, 6))
                plt.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('Prediction Probability')
                plt.ylabel('Frequency')
                plt.title('Prediction Probability Distribution')
                plt.grid(True, alpha=0.3)
                return plt.gcf()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                              n_bins: int = 10, interactive: bool = True) -> Any:
        """Plot model calibration curve"""
        
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        if interactive:
            fig = go.Figure()
            
            # Calibration curve
            fig.add_trace(go.Scatter(
                x=mean_predicted_value, y=fraction_of_positives,
                mode='lines+markers',
                name='Model',
                line=dict(color='blue', width=2)
            ))
            
            # Perfect calibration line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title='Calibration Curve',
                xaxis_title='Mean Predicted Probability',
                yaxis_title='Fraction of Positives',
                width=600,
                height=600
            )
            
            return fig
        else:
            plt.figure(figsize=(8, 8))
            
            plt.plot(mean_predicted_value, fraction_of_positives, 'bo-', 
                    linewidth=2, markersize=6, label='Model')
            plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect Calibration')
            
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            return plt.gcf()
    
    def plot_error_analysis(self, X: pd.DataFrame, y_true: np.ndarray, 
                           y_pred: np.ndarray, y_pred_proba: np.ndarray = None,
                           feature_names: List[str] = None, interactive: bool = True) -> Any:
        """Analyze prediction errors across different feature ranges"""
        
        if feature_names is None:
            feature_names = list(X.columns)[:4]  # Limit to 4 features
        
        # Identify errors
        errors = (y_true != y_pred)
        false_positives = (y_true == 0) & (y_pred == 1)
        false_negatives = (y_true == 1) & (y_pred == 0)
        
        if interactive:
            n_features = len(feature_names)
            rows = (n_features + 1) // 2
            
            fig = make_subplots(
                rows=rows, cols=2,
                subplot_titles=[f'Errors by {feat}' for feat in feature_names]
            )
            
            for i, feature in enumerate(feature_names):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                # Create feature bins
                feature_values = X[feature]
                bins = pd.qcut(feature_values, q=5, duplicates='drop')
                
                # Calculate error rates by bin
                error_rates = []
                bin_labels = []
                
                for bin_val in bins.cat.categories:
                    mask = bins == bin_val
                    if mask.sum() > 0:
                        error_rate = errors[mask].mean()
                        error_rates.append(error_rate)
                        bin_labels.append(f'{bin_val.left:.2f}-{bin_val.right:.2f}')
                
                fig.add_trace(
                    go.Bar(x=bin_labels, y=error_rates, name=f'{feature} Errors', showlegend=False),
                    row=row, col=col
                )
            
            fig.update_layout(height=300*rows, title_text="Error Analysis by Feature Ranges")
            return fig
        else:
            n_features = len(feature_names)
            rows = (n_features + 1) // 2
            
            fig, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, feature in enumerate(feature_names):
                row = i // 2
                col = i % 2
                
                # Create feature bins
                feature_values = X[feature]
                bins = pd.qcut(feature_values, q=5, duplicates='drop')
                
                # Calculate error rates by bin
                error_rates_by_bin = []
                bin_labels = []
                
                for bin_val in bins.cat.categories:
                    mask = bins == bin_val
                    if mask.sum() > 0:
                        error_rate = errors[mask].mean()
                        error_rates_by_bin.append(error_rate)
                        bin_labels.append(f'{bin_val.left:.1f}-{bin_val.right:.1f}')
                
                axes[row, col].bar(range(len(bin_labels)), error_rates_by_bin, alpha=0.7)
                axes[row, col].set_xticks(range(len(bin_labels)))
                axes[row, col].set_xticklabels(bin_labels, rotation=45)
                axes[row, col].set_ylabel('Error Rate')
                axes[row, col].set_title(f'Errors by {feature}')
                axes[row, col].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(feature_names), rows * 2):
                row = i // 2
                col = i % 2
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            return fig
    
    def plot_residual_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                              interactive: bool = True) -> Any:
        """Plot residual analysis for probability predictions"""
        
        residuals = y_true - y_pred_proba
        
        if interactive:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Residuals vs Predicted', 'Residual Distribution']
            )
            
            # Residuals vs predicted
            fig.add_trace(
                go.Scatter(x=y_pred_proba, y=residuals, mode='markers', 
                          name='Residuals', opacity=0.6),
                row=1, col=1
            )
            
            # Add horizontal line at y=0
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Residual distribution
            fig.add_trace(
                go.Histogram(x=residuals, name='Residual Distribution', nbinsx=30),
                row=1, col=2
            )
            
            fig.update_layout(height=400, title_text="Residual Analysis")
            return fig
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Residuals vs predicted
            axes[0].scatter(y_pred_proba, residuals, alpha=0.6)
            axes[0].axhline(y=0, color='red', linestyle='--')
            axes[0].set_xlabel('Predicted Probability')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs Predicted')
            axes[0].grid(True, alpha=0.3)
            
            # Residual distribution
            axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Residuals')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Residual Distribution')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig


def main():
    """Example usage of model visualization classes"""
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Sample true labels and predictions
    y_true = np.random.binomial(1, 0.3, n_samples)
    y_pred_proba = np.random.beta(2, 5, n_samples)
    y_pred_proba[y_true == 1] += np.random.normal(0.3, 0.2, np.sum(y_true == 1))
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Initialize visualizers
    perf_plotter = ModelPerformancePlotter()
    importance_plotter = FeatureImportancePlotter()
    comparison_plotter = ModelComparisonVisualizer()
    prediction_plotter = PredictionAnalysisPlotter()
    
    print("Model visualization classes initialized successfully!")
    
    # Example plots (commented out to avoid display in headless environment)
    # perf_plotter.plot_confusion_matrix(y_true, y_pred)
    # perf_plotter.plot_roc_curve(y_true, y_pred_proba)
    # prediction_plotter.plot_prediction_distribution(y_pred_proba, y_true)

if __name__ == "__main__":
    main()