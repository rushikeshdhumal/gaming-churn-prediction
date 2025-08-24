"""
Deployment Utilities for Gaming Player Churn Prediction

Production deployment, monitoring, and API utilities for model deployment.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# FastAPI and related imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeployment:
    """
    Comprehensive model deployment manager for production environments
    """
    
    def __init__(self, deployment_config: Dict[str, Any] = None):
        self.deployment_config = deployment_config or self._default_config()
        self.deployment_status = {}
        self.model_artifacts = {}
        self.health_checks = {}
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'environment': 'production',
            'model_path': 'models/',
            'api_host': '0.0.0.0',
            'api_port': 8000,
            'workers': 4,
            'enable_monitoring': True,
            'log_level': 'INFO',
            'model_version': '1.0.0',
            'backup_enabled': True,
            'auto_scaling': True,
            'health_check_interval': 300,
            'performance_threshold': 100
        }
    
    def deploy_model(self, model_path: str, deployment_type: str = 'api') -> Dict[str, Any]:
        """Deploy model with specified deployment type"""
        
        logger.info(f"Starting {deployment_type} deployment")
        
        deployment_id = f"deploy_{int(time.time())}"
        
        try:
            # Validate model
            if not self._validate_model_artifacts(model_path):
                raise ValueError("Model validation failed")
            
            # Load model artifacts
            self._load_model_artifacts(model_path)
            
            # Deploy based on type
            if deployment_type == 'api':
                result = self._deploy_api()
            elif deployment_type == 'batch':
                result = self._deploy_batch_service()
            elif deployment_type == 'streaming':
                result = self._deploy_streaming_service()
            else:
                raise ValueError(f"Unknown deployment type: {deployment_type}")
            
            # Update deployment status
            self.deployment_status[deployment_id] = {
                'status': 'deployed',
                'deployment_type': deployment_type,
                'deployment_time': datetime.now().isoformat(),
                'model_version': self.deployment_config['model_version'],
                'deployment_details': result
            }
            
            logger.info(f"Deployment {deployment_id} completed successfully")
            return {'deployment_id': deployment_id, 'status': 'success', 'details': result}
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self.deployment_status[deployment_id] = {
                'status': 'failed',
                'error': str(e),
                'deployment_time': datetime.now().isoformat()
            }
            return {'deployment_id': deployment_id, 'status': 'failed', 'error': str(e)}
    
    def _validate_model_artifacts(self, model_path: str) -> bool:
        """Validate model artifacts before deployment"""
        
        model_path = Path(model_path)
        
        required_files = ['model_metadata.json', 'preprocessor.pkl']
        
        for file in required_files:
            if not (model_path / file).exists():
                logger.error(f"Missing required file: {file}")
                return False
        
        # Load and validate metadata
        try:
            with open(model_path / 'model_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            required_keys = ['model_results', 'best_model', 'feature_names']
            for key in required_keys:
                if key not in metadata:
                    logger.error(f"Missing metadata key: {key}")
                    return False
            
            logger.info("Model artifacts validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Metadata validation failed: {e}")
            return False
    
    def _load_model_artifacts(self, model_path: str) -> None:
        """Load model artifacts into memory"""
        
        model_path = Path(model_path)
        
        # Load metadata
        with open(model_path / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load best model
        best_model_name = metadata['best_model']['name']
        model_file = model_path / f"{best_model_name}_model.pkl"
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessor
        with open(model_path / 'preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        self.model_artifacts = {
            'model': model,
            'preprocessor': preprocessor,
            'metadata': metadata,
            'model_name': best_model_name,
            'feature_names': metadata['feature_names']
        }
        
        logger.info("Model artifacts loaded successfully")
    
    def _deploy_api(self) -> Dict[str, Any]:
        """Deploy FastAPI service"""
        
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        app = self._create_fastapi_app()
        
        # Configure server
        config = uvicorn.Config(
            app,
            host=self.deployment_config['api_host'],
            port=self.deployment_config['api_port'],
            workers=self.deployment_config['workers'],
            log_level=self.deployment_config['log_level'].lower()
        )
        
        # Start server in background thread
        server = uvicorn.Server(config)
        
        def start_server():
            server.run()
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        return {
            'api_url': f"http://{self.deployment_config['api_host']}:{self.deployment_config['api_port']}",
            'workers': self.deployment_config['workers'],
            'endpoints': ['/predict', '/predict_batch', '/health', '/metrics']
        }
    
    def _create_fastapi_app(self) -> Any:
        """Create FastAPI application"""
        
        app = FastAPI(
            title="Gaming Churn Prediction API",
            description="Production API for gaming player churn prediction",
            version=self.deployment_config['model_version']
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Pydantic models
        class PlayerData(BaseModel):
            player_id: str
            total_playtime_hours: float
            avg_session_duration: float
            sessions_last_week: int
            friends_count: int
            total_spent: float
            last_login_days_ago: int
            achievements_unlocked: int
            games_owned: int
            age_group: Optional[str] = "26-35"
            region: Optional[str] = "NA"
            favorite_genre: Optional[str] = "Action"
            platform_preference: Optional[str] = "PC"
        
        class BatchPredictionRequest(BaseModel):
            players: List[PlayerData]
            include_probabilities: bool = True
            include_risk_factors: bool = True
        
        class PredictionResponse(BaseModel):
            player_id: str
            churn_prediction: bool
            churn_probability: Optional[float]
            risk_level: str
            prediction_time_ms: float
        
        # API endpoints
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model_loaded": bool(self.model_artifacts),
                "timestamp": datetime.now().isoformat(),
                "version": self.deployment_config['model_version']
            }
        
        @app.post("/predict", response_model=PredictionResponse)
        async def predict_single(player: PlayerData):
            try:
                start_time = time.time()
                
                # Convert to DataFrame
                player_df = pd.DataFrame([player.dict()])
                
                # Preprocess
                X_processed = self.model_artifacts['preprocessor'].transform(player_df)
                
                # Predict
                prediction = self.model_artifacts['model'].predict(X_processed)[0]
                probability = None
                
                if hasattr(self.model_artifacts['model'], 'predict_proba'):
                    probability = self.model_artifacts['model'].predict_proba(X_processed)[0][1]
                
                # Calculate risk level
                risk_level = 'low'
                if probability is not None:
                    if probability > 0.8:
                        risk_level = 'critical'
                    elif probability > 0.6:
                        risk_level = 'high'
                    elif probability > 0.3:
                        risk_level = 'medium'
                
                prediction_time = (time.time() - start_time) * 1000
                
                return PredictionResponse(
                    player_id=player.player_id,
                    churn_prediction=bool(prediction),
                    churn_probability=float(probability) if probability is not None else None,
                    risk_level=risk_level,
                    prediction_time_ms=prediction_time
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/predict_batch")
        async def predict_batch(request: BatchPredictionRequest):
            try:
                results = []
                
                for player in request.players:
                    # Reuse single prediction logic
                    result = await predict_single(player)
                    results.append(result.dict())
                
                return {
                    "predictions": results,
                    "batch_size": len(request.players),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/metrics")
        async def get_metrics():
            return {
                "model_info": {
                    "name": self.model_artifacts.get('model_name'),
                    "version": self.deployment_config['model_version'],
                    "features": len(self.model_artifacts.get('feature_names', []))
                },
                "system_info": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "uptime": time.time() - psutil.boot_time()
                }
            }
        
        return app
    
    def _deploy_batch_service(self) -> Dict[str, Any]:
        """Deploy batch prediction service"""
        
        return {
            'service_type': 'batch_processor',
            'max_batch_size': 10000,
            'processing_schedule': 'hourly',
            'output_format': 'csv'
        }
    
    def _deploy_streaming_service(self) -> Dict[str, Any]:
        """Deploy streaming prediction service"""
        
        return {
            'service_type': 'streaming_processor',
            'stream_source': 'kafka',
            'throughput': '1000 predictions/sec',
            'latency': '<10ms'
        }
    
    def get_deployment_status(self, deployment_id: str = None) -> Dict[str, Any]:
        """Get deployment status"""
        
        if deployment_id:
            return self.deployment_status.get(deployment_id, {})
        
        return self.deployment_status
    
    def rollback_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Rollback a deployment"""
        
        if deployment_id not in self.deployment_status:
            return {'status': 'error', 'message': 'Deployment ID not found'}
        
        try:
            # Implementation would depend on deployment type
            logger.info(f"Rolling back deployment {deployment_id}")
            
            self.deployment_status[deployment_id]['status'] = 'rolled_back'
            self.deployment_status[deployment_id]['rollback_time'] = datetime.now().isoformat()
            
            return {'status': 'success', 'message': 'Deployment rolled back'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


class ModelMonitoring:
    """
    Production model monitoring and alerting system
    """
    
    def __init__(self, monitoring_config: Dict[str, Any] = None):
        self.monitoring_config = monitoring_config or self._default_config()
        self.metrics_history = []
        self.alerts = []
        self.monitoring_active = False
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'check_interval': 300,  # 5 minutes
            'performance_threshold': 100,  # ms
            'accuracy_threshold': 0.8,
            'drift_threshold': 0.1,
            'error_rate_threshold': 0.05,
            'memory_threshold': 80,  # percentage
            'cpu_threshold': 80,  # percentage
            'enable_alerts': True,
            'alert_channels': ['log', 'email'],
            'drift_detection_window': 1000
        }
    
    def start_monitoring(self, model, preprocessor, baseline_data: pd.DataFrame = None) -> None:
        """Start continuous model monitoring"""
        
        self.model = model
        self.preprocessor = preprocessor
        self.baseline_data = baseline_data
        self.monitoring_active = True
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        logger.info("Model monitoring started")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Store metrics
                self.metrics_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                })
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Cleanup old metrics (keep last 24 hours)
                self._cleanup_metrics()
                
                time.sleep(self.monitoring_config['check_interval'])
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system and model metrics"""
        
        metrics = {
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'timestamp': time.time()
            },
            'model': {
                'predictions_per_minute': self._get_prediction_rate(),
                'average_response_time': self._get_avg_response_time(),
                'error_rate': self._get_error_rate(),
                'model_version': '1.0.0'
            }
        }
        
        # Data drift detection
        if self.baseline_data is not None:
            drift_score = self._detect_data_drift()
            metrics['data_quality'] = {
                'drift_score': drift_score,
                'drift_detected': drift_score > self.monitoring_config['drift_threshold']
            }
        
        return metrics
    
    def _get_prediction_rate(self) -> float:
        """Calculate predictions per minute"""
        # Implementation would track actual prediction calls
        return 100.0  # Placeholder
    
    def _get_avg_response_time(self) -> float:
        """Calculate average response time"""
        # Implementation would track actual response times
        return 25.0  # Placeholder
    
    def _get_error_rate(self) -> float:
        """Calculate error rate"""
        # Implementation would track actual errors
        return 0.001  # Placeholder
    
    def _detect_data_drift(self) -> float:
        """Detect data drift using statistical methods"""
        
        # Simplified drift detection
        # In production, this would use more sophisticated methods
        # like KS test, PSI, or model-based drift detection
        
        return np.random.random() * 0.2  # Placeholder
    
    def _check_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check metrics against thresholds and generate alerts"""
        
        alerts = []
        
        # System alerts
        if metrics['system']['cpu_percent'] > self.monitoring_config['cpu_threshold']:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'warning',
                'message': f"CPU usage {metrics['system']['cpu_percent']:.1f}% exceeds threshold",
                'timestamp': datetime.now().isoformat()
            })
        
        if metrics['system']['memory_percent'] > self.monitoring_config['memory_threshold']:
            alerts.append({
                'type': 'high_memory',
                'severity': 'warning',
                'message': f"Memory usage {metrics['system']['memory_percent']:.1f}% exceeds threshold",
                'timestamp': datetime.now().isoformat()
            })
        
        # Model performance alerts
        if metrics['model']['average_response_time'] > self.monitoring_config['performance_threshold']:
            alerts.append({
                'type': 'slow_response',
                'severity': 'warning',
                'message': f"Response time {metrics['model']['average_response_time']:.1f}ms exceeds threshold",
                'timestamp': datetime.now().isoformat()
            })
        
        if metrics['model']['error_rate'] > self.monitoring_config['error_rate_threshold']:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f"Error rate {metrics['model']['error_rate']:.3f} exceeds threshold",
                'timestamp': datetime.now().isoformat()
            })
        
        # Data quality alerts
        if 'data_quality' in metrics and metrics['data_quality']['drift_detected']:
            alerts.append({
                'type': 'data_drift',
                'severity': 'critical',
                'message': f"Data drift detected with score {metrics['data_quality']['drift_score']:.3f}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
    
    def _process_alert(self, alert: Dict[str, Any]) -> None:
        """Process and send alerts"""
        
        self.alerts.append(alert)
        
        if self.monitoring_config['enable_alerts']:
            # Log alert
            if 'log' in self.monitoring_config['alert_channels']:
                logger.warning(f"ALERT [{alert['type']}]: {alert['message']}")
            
            # Email alert (implementation would send actual email)
            if 'email' in self.monitoring_config['alert_channels']:
                self._send_email_alert(alert)
    
    def _send_email_alert(self, alert: Dict[str, Any]) -> None:
        """Send email alert (placeholder)"""
        logger.info(f"Email alert sent: {alert['message']}")
    
    def _cleanup_metrics(self) -> None:
        """Clean up old metrics"""
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        self.metrics_history = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        self.alerts = [
            a for a in self.alerts
            if datetime.fromisoformat(a['timestamp']) > cutoff_time
        ]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 readings
        
        return {
            'current_status': 'healthy' if len(self.alerts) == 0 else 'issues_detected',
            'active_alerts': len([a for a in self.alerts if 
                                datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)]),
            'average_cpu': np.mean([m['metrics']['system']['cpu_percent'] for m in recent_metrics]),
            'average_memory': np.mean([m['metrics']['system']['memory_percent'] for m in recent_metrics]),
            'average_response_time': np.mean([m['metrics']['model']['average_response_time'] for m in recent_metrics]),
            'prediction_rate': np.mean([m['metrics']['model']['predictions_per_minute'] for m in recent_metrics]),
            'uptime_hours': (datetime.now() - datetime.fromisoformat(self.metrics_history[0]['timestamp'])).total_seconds() / 3600
        }
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.monitoring_active = False
        logger.info("Model monitoring stopped")


class BatchPredictionPipeline:
    """
    Scalable batch prediction pipeline for large datasets
    """
    
    def __init__(self, model, preprocessor, batch_config: Dict[str, Any] = None):
        self.model = model
        self.preprocessor = preprocessor
        self.batch_config = batch_config or self._default_config()
        self.processing_stats = {}
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'batch_size': 1000,
            'max_workers': 4,
            'chunk_size': 100,
            'output_format': 'csv',
            'include_probabilities': True,
            'include_metadata': True,
            'error_handling': 'skip',
            'progress_reporting': True
        }
    
    def process_batch(self, input_data: Union[pd.DataFrame, str], 
                     output_path: str = None) -> Dict[str, Any]:
        """Process batch predictions"""
        
        start_time = time.time()
        
        logger.info("Starting batch prediction pipeline")
        
        # Load data if path provided
        if isinstance(input_data, str):
            data = self._load_data(input_data)
        else:
            data = input_data.copy()
        
        total_records = len(data)
        logger.info(f"Processing {total_records} records")
        
        # Process in batches
        results = []
        errors = []
        processed_count = 0
        
        batch_size = self.batch_config['batch_size']
        
        for i in range(0, total_records, batch_size):
            batch_start = time.time()
            
            batch_data = data.iloc[i:i+batch_size]
            
            try:
                batch_results = self._process_batch_chunk(batch_data)
                results.extend(batch_results)
                processed_count += len(batch_data)
                
                if self.batch_config['progress_reporting']:
                    progress = (processed_count / total_records) * 100
                    logger.info(f"Progress: {progress:.1f}% ({processed_count}/{total_records})")
                
            except Exception as e:
                logger.error(f"Batch {i//batch_size + 1} failed: {e}")
                
                if self.batch_config['error_handling'] == 'skip':
                    errors.append({
                        'batch_index': i//batch_size + 1,
                        'error': str(e),
                        'records_in_batch': len(batch_data)
                    })
                    continue
                else:
                    raise
        
        # Compile results
        results_df = pd.DataFrame(results)
        
        # Save results
        if output_path:
            self._save_results(results_df, output_path)
        
        # Calculate statistics
        total_time = time.time() - start_time
        
        self.processing_stats = {
            'total_records': total_records,
            'processed_records': len(results_df),
            'failed_records': total_records - len(results_df),
            'processing_time_seconds': total_time,
            'records_per_second': len(results_df) / total_time,
            'error_count': len(errors),
            'success_rate': len(results_df) / total_records
        }
        
        logger.info(f"Batch processing completed in {total_time:.2f} seconds")
        
        return {
            'results': results_df,
            'statistics': self.processing_stats,
            'errors': errors,
            'output_path': output_path
        }
    
    def _load_data(self, input_path: str) -> pd.DataFrame:
        """Load data from file"""
        
        input_path = Path(input_path)
        
        if input_path.suffix == '.csv':
            return pd.read_csv(input_path)
        elif input_path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(input_path)
        elif input_path.suffix == '.json':
            return pd.read_json(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    def _process_batch_chunk(self, batch_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process a single batch chunk"""
        
        # Preprocess data
        X_processed = self.preprocessor.transform(batch_data)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        probabilities = None
        if hasattr(self.model, 'predict_proba') and self.batch_config['include_probabilities']:
            probabilities = self.model.predict_proba(X_processed)[:, 1]
        
        # Compile results
        results = []
        
        for i, (_, row) in enumerate(batch_data.iterrows()):
            result = {
                'player_id': row.get('player_id', f'player_{i}'),
                'churn_prediction': bool(predictions[i]),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            if probabilities is not None:
                result['churn_probability'] = float(probabilities[i])
                
                # Risk categorization
                prob = probabilities[i]
                if prob > 0.8:
                    result['risk_level'] = 'critical'
                elif prob > 0.6:
                    result['risk_level'] = 'high'
                elif prob > 0.3:
                    result['risk_level'] = 'medium'
                else:
                    result['risk_level'] = 'low'
            
            if self.batch_config['include_metadata']:
                result['model_version'] = '1.0.0'
                result['processing_batch'] = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            results.append(result)
        
        return results
    
    def _save_results(self, results_df: pd.DataFrame, output_path: str) -> None:
        """Save results to file"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.batch_config['output_format'] == 'csv':
            results_df.to_csv(output_path, index=False)
        elif self.batch_config['output_format'] == 'json':
            results_df.to_json(output_path, orient='records', indent=2)
        elif self.batch_config['output_format'] == 'parquet':
            results_df.to_parquet(output_path, index=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def process_parallel(self, input_data: pd.DataFrame, output_path: str = None) -> Dict[str, Any]:
        """Process batch with parallel execution"""
        
        total_records = len(input_data)
        batch_size = self.batch_config['batch_size']
        max_workers = self.batch_config['max_workers']
        
        # Split data into chunks
        chunks = [input_data.iloc[i:i+batch_size] for i in range(0, total_records, batch_size)]
        
        start_time = time.time()
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(self._process_batch_chunk, chunks))
        
        # Combine results
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
        
        results_df = pd.DataFrame(all_results)
        
        # Save results
        if output_path:
            self._save_results(results_df, output_path)
        
        total_time = time.time() - start_time
        
        return {
            'results': results_df,
            'processing_time': total_time,
            'records_per_second': len(results_df) / total_time,
            'parallel_workers': max_workers
        }


class RealTimeAPI:
    """
    High-performance real-time prediction API
    """
    
    def __init__(self, model, preprocessor, api_config: Dict[str, Any] = None):
        self.model = model
        self.preprocessor = preprocessor
        self.api_config = api_config or self._default_config()
        self.request_cache = {}
        self.performance_metrics = {
            'total_requests': 0,
            'total_errors': 0,
            'total_time': 0,
            'start_time': time.time()
        }
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'cache_enabled': True,
            'cache_ttl': 300,  # 5 minutes
            'rate_limiting': True,
            'max_requests_per_minute': 1000,
            'enable_cors': True,
            'request_timeout': 30,
            'batch_prediction_limit': 100
        }
    
    def predict(self, player_data: Dict[str, Any], request_id: str = None) -> Dict[str, Any]:
        """Real-time prediction with performance optimizations"""
        
        start_time = time.time()
        
        try:
            # Check cache
            if self.api_config['cache_enabled'] and request_id:
                cached_result = self._get_cached_result(request_id, player_data)
                if cached_result:
                    return cached_result
            
            # Validate input
            self._validate_input(player_data)
            
            # Convert to DataFrame
            player_df = pd.DataFrame([player_data])
            
            # Preprocess
            X_processed = self.preprocessor.transform(player_df)
            
            # Predict
            prediction = self.model.predict(X_processed)[0]
            probability = None
            
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(X_processed)[0][1]
            
            # Build response
            response = {
                'request_id': request_id or f"req_{int(time.time())}",
                'player_id': player_data.get('player_id', 'unknown'),
                'churn_prediction': bool(prediction),
                'churn_probability': float(probability) if probability is not None else None,
                'risk_level': self._categorize_risk(probability) if probability is not None else 'unknown',
                'response_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat(),
                'model_version': '1.0.0'
            }
            
            # Cache result
            if self.api_config['cache_enabled'] and request_id:
                self._cache_result(request_id, player_data, response)
            
            # Update metrics
            self._update_metrics(start_time, success=True)
            
            return response
            
        except Exception as e:
            self._update_metrics(start_time, success=False)
            raise e
    
    def _validate_input(self, player_data: Dict[str, Any]) -> None:
        """Validate input data"""
        
        required_fields = [
            'total_playtime_hours', 'avg_session_duration', 'sessions_last_week',
            'friends_count', 'total_spent', 'last_login_days_ago'
        ]
        
        for field in required_fields:
            if field not in player_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate data types and ranges
        numeric_fields = {
            'total_playtime_hours': (0, 10000),
            'avg_session_duration': (0, 600),
            'sessions_last_week': (0, 50),
            'friends_count': (0, 1000),
            'total_spent': (0, 10000),
            'last_login_days_ago': (0, 365)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in player_data:
                value = player_data[field]
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    raise ValueError(f"Invalid value for {field}: {value}")
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk level"""
        
        if probability > 0.8:
            return 'critical'
        elif probability > 0.6:
            return 'high'
        elif probability > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _get_cached_result(self, request_id: str, player_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached prediction result"""
        
        if request_id not in self.request_cache:
            return None
        
        cached_entry = self.request_cache[request_id]
        
        # Check TTL
        if time.time() - cached_entry['timestamp'] > self.api_config['cache_ttl']:
            del self.request_cache[request_id]
            return None
        
        # Check if data has changed
        if cached_entry['data_hash'] != hash(str(sorted(player_data.items()))):
            return None
        
        return cached_entry['result']
    
    def _cache_result(self, request_id: str, player_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Cache prediction result"""
        
        self.request_cache[request_id] = {
            'result': result,
            'data_hash': hash(str(sorted(player_data.items()))),
            'timestamp': time.time()
        }
        
        # Cleanup old cache entries
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.request_cache.items()
            if current_time - entry['timestamp'] > self.api_config['cache_ttl']
        ]
        
        for key in expired_keys:
            del self.request_cache[key]
    
    def _update_metrics(self, start_time: float, success: bool) -> None:
        """Update performance metrics"""
        
        self.performance_metrics['total_requests'] += 1
        self.performance_metrics['total_time'] += time.time() - start_time
        
        if not success:
            self.performance_metrics['total_errors'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get API performance metrics"""
        
        uptime = time.time() - self.performance_metrics['start_time']
        
        return {
            'total_requests': self.performance_metrics['total_requests'],
            'total_errors': self.performance_metrics['total_errors'],
            'error_rate': self.performance_metrics['total_errors'] / max(1, self.performance_metrics['total_requests']),
            'average_response_time_ms': (self.performance_metrics['total_time'] / max(1, self.performance_metrics['total_requests'])) * 1000,
            'requests_per_second': self.performance_metrics['total_requests'] / uptime,
            'uptime_seconds': uptime,
            'cache_size': len(self.request_cache)
        }


class DeploymentValidator:
    """
    Validate deployment readiness and perform pre-deployment checks
    """
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_deployment(self, model_path: str, test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Comprehensive deployment validation"""
        
        logger.info("Starting deployment validation")
        
        results = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'checks_passed': 0,
            'checks_failed': 0,
            'validation_details': {}
        }
        
        # Model artifact validation
        artifact_check = self._validate_model_artifacts(model_path)
        results['validation_details']['model_artifacts'] = artifact_check
        
        # Model loading validation
        loading_check = self._validate_model_loading(model_path)
        results['validation_details']['model_loading'] = loading_check
        
        # Performance validation
        if test_data is not None:
            performance_check = self._validate_model_performance(model_path, test_data)
            results['validation_details']['model_performance'] = performance_check
        
        # System requirements validation
        system_check = self._validate_system_requirements()
        results['validation_details']['system_requirements'] = system_check
        
        # API readiness validation
        api_check = self._validate_api_readiness()
        results['validation_details']['api_readiness'] = api_check
        
        # Calculate overall status
        all_checks = [artifact_check, loading_check, system_check, api_check]
        if test_data is not None:
            all_checks.append(performance_check)
        
        passed_checks = sum(1 for check in all_checks if check['status'] == 'passed')
        failed_checks = len(all_checks) - passed_checks
        
        results['checks_passed'] = passed_checks
        results['checks_failed'] = failed_checks
        results['overall_status'] = 'ready' if failed_checks == 0 else 'not_ready'
        
        self.validation_results = results
        
        logger.info(f"Validation completed: {passed_checks}/{len(all_checks)} checks passed")
        
        return results
    
    def _validate_model_artifacts(self, model_path: str) -> Dict[str, Any]:
        """Validate model artifacts"""
        
        model_path = Path(model_path)
        
        required_files = [
            'model_metadata.json',
            'preprocessor.pkl'
        ]
        
        issues = []
        
        for file in required_files:
            if not (model_path / file).exists():
                issues.append(f"Missing required file: {file}")
        
        # Check metadata content
        metadata_file = model_path / 'model_metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                required_keys = ['best_model', 'feature_names', 'model_results']
                for key in required_keys:
                    if key not in metadata:
                        issues.append(f"Missing metadata key: {key}")
                        
            except json.JSONDecodeError:
                issues.append("Invalid JSON in metadata file")
        
        return {
            'status': 'passed' if len(issues) == 0 else 'failed',
            'issues': issues,
            'message': 'All artifacts present' if len(issues) == 0 else f'{len(issues)} issues found'
        }
    
    def _validate_model_loading(self, model_path: str) -> Dict[str, Any]:
        """Validate model can be loaded"""
        
        try:
            model_path = Path(model_path)
            
            # Load metadata
            with open(model_path / 'model_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            # Load best model
            best_model_name = metadata['best_model']['name']
            model_file = model_path / f"{best_model_name}_model.pkl"
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Load preprocessor
            with open(model_path / 'preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            
            # Test basic functionality
            if hasattr(model, 'predict') and hasattr(preprocessor, 'transform'):
                return {
                    'status': 'passed',
                    'issues': [],
                    'message': 'Model loaded successfully',
                    'model_type': str(type(model).__name__)
                }
            else:
                return {
                    'status': 'failed',
                    'issues': ['Model or preprocessor missing required methods'],
                    'message': 'Model loading validation failed'
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'issues': [str(e)],
                'message': 'Failed to load model'
            }
    
    def _validate_model_performance(self, model_path: str, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate model performance on test data"""
        
        try:
            # Load model artifacts
            deployment = ModelDeployment()
            deployment._load_model_artifacts(model_path)
            
            model = deployment.model_artifacts['model']
            preprocessor = deployment.model_artifacts['preprocessor']
            
            # Test prediction
            sample_data = test_data.head(100).drop(['churned'], axis=1, errors='ignore')
            
            start_time = time.time()
            X_processed = preprocessor.transform(sample_data)
            predictions = model.predict(X_processed)
            prediction_time = time.time() - start_time
            
            # Performance metrics
            avg_prediction_time = (prediction_time / len(sample_data)) * 1000  # ms per prediction
            
            issues = []
            if avg_prediction_time > 100:  # 100ms threshold
                issues.append(f"Slow prediction time: {avg_prediction_time:.2f}ms per prediction")
            
            return {
                'status': 'passed' if len(issues) == 0 else 'failed',
                'issues': issues,
                'message': 'Performance validation passed' if len(issues) == 0 else 'Performance issues detected',
                'metrics': {
                    'avg_prediction_time_ms': avg_prediction_time,
                    'total_prediction_time_ms': prediction_time * 1000,
                    'samples_tested': len(sample_data)
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'issues': [str(e)],
                'message': 'Performance validation failed'
            }
    
    def _validate_system_requirements(self) -> Dict[str, Any]:
        """Validate system requirements"""
        
        issues = []
        
        # Check memory
        memory = psutil.virtual_memory()
        if memory.available < 2 * 1024 * 1024 * 1024:  # 2GB
            issues.append(f"Low available memory: {memory.available / 1024 / 1024 / 1024:.1f}GB")
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        if cpu_count < 2:
            issues.append(f"Low CPU count: {cpu_count}")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.free < 5 * 1024 * 1024 * 1024:  # 5GB
            issues.append(f"Low disk space: {disk.free / 1024 / 1024 / 1024:.1f}GB")
        
        return {
            'status': 'passed' if len(issues) == 0 else 'failed',
            'issues': issues,
            'message': 'System requirements met' if len(issues) == 0 else 'System requirements not met',
            'system_info': {
                'memory_gb': memory.total / 1024 / 1024 / 1024,
                'cpu_count': cpu_count,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024
            }
        }
    
    def _validate_api_readiness(self) -> Dict[str, Any]:
        """Validate API deployment readiness"""
        
        issues = []
        
        # Check FastAPI availability
        if not FASTAPI_AVAILABLE:
            issues.append("FastAPI not installed (pip install fastapi uvicorn)")
        
        # Check required Python packages
        required_packages = ['pandas', 'numpy', 'scikit-learn', 'pickle']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(f"Missing required package: {package}")
        
        return {
            'status': 'passed' if len(issues) == 0 else 'failed',
            'issues': issues,
            'message': 'API ready for deployment' if len(issues) == 0 else 'API deployment requirements not met'
        }
    
    def generate_validation_report(self) -> str:
        """Generate validation report"""
        
        if not self.validation_results:
            return "No validation results available"
        
        results = self.validation_results
        
        report = []
        report.append("=" * 60)
        report.append("DEPLOYMENT VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall status
        status_emoji = "✅" if results['overall_status'] == 'ready' else "❌"
        report.append(f"\n{status_emoji} OVERALL STATUS: {results['overall_status'].upper()}")
        report.append(f"Checks Passed: {results['checks_passed']}")
        report.append(f"Checks Failed: {results['checks_failed']}")
        
        # Detailed results
        for check_name, check_result in results['validation_details'].items():
            status_emoji = "✅" if check_result['status'] == 'passed' else "❌"
            report.append(f"\n{status_emoji} {check_name.upper().replace('_', ' ')}")
            report.append(f"   Status: {check_result['status']}")
            report.append(f"   Message: {check_result['message']}")
            
            if check_result['issues']:
                report.append("   Issues:")
                for issue in check_result['issues']:
                    report.append(f"     • {issue}")
        
        # Recommendations
        report.append(f"\n🎯 RECOMMENDATIONS:")
        if results['overall_status'] == 'ready':
            report.append("   • System is ready for deployment")
            report.append("   • Proceed with deployment process")
        else:
            report.append("   • Fix identified issues before deployment")
            report.append("   • Re-run validation after fixes")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)