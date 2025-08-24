"""
Logging Utilities Module

This module provides comprehensive logging capabilities for the gaming analytics system
including performance logging, model tracking, and business metrics logging.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import logging
import logging.handlers
import sys
import time
import functools
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json
import threading
from contextlib import contextmanager

class LoggerSetup:
    """Setup and configure logging for the application"""
    
    def __init__(self, environment: str = 'development'):
        self.environment = environment
        self.loggers = {}
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the specified name"""
        
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(self._get_log_level())
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Add console handler
        if self._should_log_to_console():
            console_handler = self._create_console_handler()
            logger.addHandler(console_handler)
        
        # Add file handler
        if self._should_log_to_file():
            file_handler = self._create_file_handler(name)
            logger.addHandler(file_handler)
        
        # Add error file handler for errors and above
        error_handler = self._create_error_file_handler()
        logger.addHandler(error_handler)
        
        # Prevent duplicate logs
        logger.propagate = False
        
        self.loggers[name] = logger
        return logger
    
    def _get_log_level(self) -> int:
        """Get log level based on environment"""
        levels = {
            'development': logging.DEBUG,
            'testing': logging.WARNING,
            'staging': logging.INFO,
            'production': logging.INFO
        }
        return levels.get(self.environment, logging.INFO)
    
    def _should_log_to_console(self) -> bool:
        """Determine if should log to console"""
        return self.environment in ['development', 'staging']
    
    def _should_log_to_file(self) -> bool:
        """Determine if should log to file"""
        return True  # Always log to file
    
    def _create_console_handler(self) -> logging.StreamHandler:
        """Create console handler with formatting"""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        
        # Color formatter for development
        if self.environment == 'development':
            formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        return handler
    
    def _create_file_handler(self, logger_name: str) -> logging.handlers.RotatingFileHandler:
        """Create rotating file handler"""
        log_file = self.log_dir / f"{logger_name}.log"
        
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)
        return handler
    
    def _create_error_file_handler(self) -> logging.handlers.RotatingFileHandler:
        """Create error file handler for errors and above"""
        error_file = self.log_dir / "errors.log"
        
        handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        handler.setLevel(logging.ERROR)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n'
            'Traceback: %(exc_info)s'
        )
        handler.setFormatter(formatter)
        return handler

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        
        return super().format(record)

class PerformanceLogger:
    """Log performance metrics and timing information"""
    
    def __init__(self, logger_name: str = 'performance'):
        self.logger = LoggerSetup().get_logger(logger_name)
        self.metrics = {}
        self.lock = threading.Lock()
    
    def log_function_timing(self, func: Callable) -> Callable:
        """Decorator to log function execution time"""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                self.logger.info(
                    f"Function {function_name} executed successfully in {execution_time:.4f}s"
                )
                
                # Store metric
                with self.lock:
                    if function_name not in self.metrics:
                        self.metrics[function_name] = []
                    self.metrics[function_name].append(execution_time)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(
                    f"Function {function_name} failed after {execution_time:.4f}s: {str(e)}"
                )
                raise
        
        return wrapper
    
    @contextmanager
    def log_block_timing(self, block_name: str):
        """Context manager to log execution time of code blocks"""
        start_time = time.time()
        self.logger.debug(f"Starting {block_name}")
        
        try:
            yield
            execution_time = time.time() - start_time
            self.logger.info(f"{block_name} completed in {execution_time:.4f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"{block_name} failed after {execution_time:.4f}s: {str(e)}")
            raise
    
    def log_memory_usage(self, label: str = "Memory Usage"):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"{label}: {memory_mb:.2f} MB")
            
        except ImportError:
            self.logger.warning("psutil not available for memory logging")
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all tracked functions"""
        summary = {}
        
        with self.lock:
            for func_name, times in self.metrics.items():
                if times:
                    summary[func_name] = {
                        'count': len(times),
                        'total_time': sum(times),
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times)
                    }
        
        return summary
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        with self.lock:
            self.metrics.clear()
        self.logger.info("Performance metrics reset")

class ModelLogger:
    """Specialized logger for machine learning model operations"""
    
    def __init__(self, logger_name: str = 'model'):
        self.logger = LoggerSetup().get_logger(logger_name)
        self.model_runs = []
        
    def log_training_start(self, model_name: str, config: Dict[str, Any]):
        """Log the start of model training"""
        run_info = {
            'model_name': model_name,
            'start_time': datetime.now().isoformat(),
            'config': config,
            'status': 'training'
        }
        
        self.model_runs.append(run_info)
        
        self.logger.info(
            f"Started training {model_name} with config: {json.dumps(config, indent=2)}"
        )
        
        return len(self.model_runs) - 1  # Return run index
    
    def log_training_complete(self, run_index: int, metrics: Dict[str, float], 
                            model_path: Optional[str] = None):
        """Log completion of model training"""
        
        if run_index < len(self.model_runs):
            run_info = self.model_runs[run_index]
            run_info['end_time'] = datetime.now().isoformat()
            run_info['status'] = 'completed'
            run_info['metrics'] = metrics
            run_info['model_path'] = model_path
            
            start_time = datetime.fromisoformat(run_info['start_time'])
            end_time = datetime.fromisoformat(run_info['end_time'])
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info(
                f"Completed training {run_info['model_name']} in {duration:.2f}s. "
                f"Metrics: {json.dumps(metrics, indent=2)}"
            )
    
    def log_training_failed(self, run_index: int, error: Exception):
        """Log failed model training"""
        
        if run_index < len(self.model_runs):
            run_info = self.model_runs[run_index]
            run_info['end_time'] = datetime.now().isoformat()
            run_info['status'] = 'failed'
            run_info['error'] = str(error)
            run_info['traceback'] = traceback.format_exc()
            
            self.logger.error(
                f"Training failed for {run_info['model_name']}: {str(error)}"
            )
    
    def log_prediction_batch(self, model_name: str, batch_size: int, 
                           prediction_time: float, avg_confidence: float):
        """Log batch prediction information"""
        self.logger.info(
            f"Model {model_name} processed {batch_size} predictions in "
            f"{prediction_time:.4f}s (avg confidence: {avg_confidence:.3f})"
        )
    
    def log_model_performance(self, model_name: str, dataset: str, metrics: Dict[str, float]):
        """Log model performance on a dataset"""
        self.logger.info(
            f"Model {model_name} performance on {dataset}: "
            f"{json.dumps(metrics, indent=2)}"
        )
    
    def log_feature_importance(self, model_name: str, feature_importance: Dict[str, float]):
        """Log feature importance for interpretability"""
        # Log top 10 most important features
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        
        self.logger.info(
            f"Top 10 features for {model_name}: "
            f"{json.dumps(dict(sorted_features), indent=2)}"
        )
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get complete training history"""
        return self.model_runs.copy()

class BusinessLogger:
    """Logger for business metrics and insights"""
    
    def __init__(self, logger_name: str = 'business'):
        self.logger = LoggerSetup().get_logger(logger_name)
        
    def log_churn_prediction_impact(self, predictions: Dict[str, Any]):
        """Log business impact of churn predictions"""
        self.logger.info(
            f"Churn prediction impact: {json.dumps(predictions, indent=2)}"
        )
    
    def log_revenue_impact(self, revenue_metrics: Dict[str, float]):
        """Log revenue impact analysis"""
        self.logger.info(
            f"Revenue impact analysis: {json.dumps(revenue_metrics, indent=2)}"
        )
    
    def log_player_segmentation(self, segments: Dict[str, int]):
        """Log player segmentation results"""
        total_players = sum(segments.values())
        segment_percentages = {
            segment: (count / total_players) * 100 
            for segment, count in segments.items()
        }
        
        self.logger.info(
            f"Player segmentation: {json.dumps(segment_percentages, indent=2)}%"
        )
    
    def log_intervention_results(self, intervention_type: str, 
                               success_rate: float, players_affected: int):
        """Log results of player intervention campaigns"""
        self.logger.info(
            f"Intervention '{intervention_type}' results: "
            f"{success_rate:.1%} success rate, {players_affected} players affected"
        )
    
    def log_roi_analysis(self, investment: float, return_value: float, 
                        roi_percentage: float):
        """Log ROI analysis results"""
        self.logger.info(
            f"ROI Analysis - Investment: ${investment:,.2f}, "
            f"Return: ${return_value:,.2f}, ROI: {roi_percentage:.1%}"
        )

class AuditLogger:
    """Logger for audit trails and compliance"""
    
    def __init__(self, logger_name: str = 'audit'):
        self.logger = LoggerSetup().get_logger(logger_name)
        
    def log_data_access(self, user: str, data_type: str, action: str, 
                       record_count: Optional[int] = None):
        """Log data access for audit purposes"""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'data_type': data_type,
            'action': action,
            'record_count': record_count
        }
        
        self.logger.info(f"Data access: {json.dumps(audit_entry)}")
    
    def log_model_deployment(self, model_name: str, version: str, 
                           deployed_by: str, environment: str):
        """Log model deployment events"""
        deployment_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'version': version,
            'deployed_by': deployed_by,
            'environment': environment,
            'action': 'deployment'
        }
        
        self.logger.info(f"Model deployment: {json.dumps(deployment_entry)}")
    
    def log_prediction_request(self, request_id: str, user: str, 
                             model_name: str, input_hash: str):
        """Log prediction requests for audit"""
        prediction_entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'user': user,
            'model_name': model_name,
            'input_hash': input_hash,
            'action': 'prediction_request'
        }
        
        self.logger.info(f"Prediction request: {json.dumps(prediction_entry)}")

def get_logger(name: str, environment: str = 'development') -> logging.Logger:
    """Convenience function to get a logger"""
    setup = LoggerSetup(environment)
    return setup.get_logger(name)

def setup_logging(environment: str = 'development') -> Dict[str, logging.Logger]:
    """Setup all loggers for the application"""
    setup = LoggerSetup(environment)
    
    loggers = {
        'main': setup.get_logger('main'),
        'data': setup.get_logger('data'),
        'model': setup.get_logger('model'),
        'api': setup.get_logger('api'),
        'business': setup.get_logger('business'),
        'performance': setup.get_logger('performance'),
        'audit': setup.get_logger('audit')
    }
    
    loggers['main'].info(f"Logging setup complete for environment: {environment}")
    return loggers

def main():
    """Example usage of logging utilities"""
    
    # Setup loggers
    loggers = setup_logging('development')
    
    # Example performance logging
    perf_logger = PerformanceLogger()
    
    @perf_logger.log_function_timing
    def example_function():
        time.sleep(0.1)  # Simulate work
        return "result"
    
    # Test function timing
    result = example_function()
    
    # Test block timing
    with perf_logger.log_block_timing("Data processing"):
        time.sleep(0.05)  # Simulate processing
    
    # Example model logging
    model_logger = ModelLogger()
    run_id = model_logger.log_training_start('xgboost', {'n_estimators': 100})
    model_logger.log_training_complete(run_id, {'accuracy': 0.91, 'roc_auc': 0.95})
    
    # Example business logging
    business_logger = BusinessLogger()
    business_logger.log_churn_prediction_impact({
        'high_risk_players': 150,
        'potential_revenue_saved': 75000,
        'intervention_cost': 15000
    })
    
    # Get performance summary
    summary = perf_logger.get_performance_summary()
    print("Performance Summary:")
    for func, metrics in summary.items():
        print(f"  {func}: {metrics}")

if __name__ == "__main__":
    main()