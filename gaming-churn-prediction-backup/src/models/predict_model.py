"""
Model Prediction Module for Gaming Player Churn Prediction

This module provides comprehensive prediction capabilities including
single predictions, ensemble predictions, and real-time prediction services.

Author: Rushikesh Dhumal
Email: r.dhumal@rutgers.edu
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Base class for model prediction with common functionality
    
    Provides core prediction capabilities that can be extended
    by specialized predictor classes.
    """
    
    def __init__(self, model_path: str = None, model_object: Any = None):
        """
        Initialize the model predictor
        
        Args:
            model_path: Path to saved model file
            model_object: Pre-loaded model object
        """
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_metadata = None
        
        if model_path:
            self.load_model(model_path)
        elif model_object:
            self.model = model_object
        
    def load_model(self, model_path: str) -> None:
        """Load model and associated artifacts"""
        
        model_path = Path(model_path)
        
        if model_path.is_file():
            # Single model file
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded model from {model_path}")
        elif model_path.is_dir():
            # Model directory with artifacts
            self._load_model_artifacts(model_path)
        else:
            raise ValueError(f"Invalid model path: {model_path}")
    
    def _load_model_artifacts(self, model_dir: Path) -> None:
        """Load complete model artifacts from directory"""
        
        # Load metadata
        metadata_file = model_dir / "model_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.model_metadata = json.load(f)
            
            # Load best model
            best_model_name = self.model_metadata.get('best_model', {}).get('name')
            if best_model_name:
                model_file = model_dir / f"{best_model_name}_model.pkl"
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        self.model = pickle.load(f)
                    logger.info(f"Loaded best model: {best_model_name}")
        
        # Load preprocessor
        preprocessor_file = model_dir / "preprocessor.pkl"
        if preprocessor_file.exists():
            with open(preprocessor_file, 'rb') as f:
                self.preprocessor = pickle.load(f)
            logger.info("Loaded preprocessor")
        
        # Extract feature names
        if self.model_metadata:
            self.feature_names = self.model_metadata.get('feature_names', [])
    
    def predict(self, X: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """Make predictions on input data"""
        
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        # Convert input to DataFrame if needed
        X_df = self._prepare_input(X)
        
        # Preprocess if preprocessor available
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X_df)
        else:
            X_processed = X_df.values
        
        # Make prediction
        predictions = self.model.predict(X_processed)
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """Predict class probabilities"""
        
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        # Convert input to DataFrame if needed
        X_df = self._prepare_input(X)
        
        # Preprocess if preprocessor available
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X_df)
        else:
            X_processed = X_df.values
        
        # Make probability prediction
        probabilities = self.model.predict_proba(X_processed)
        return probabilities
    
    def _prepare_input(self, X: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """Convert various input formats to DataFrame"""
        
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, dict):
            return pd.DataFrame([X])
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            return pd.DataFrame(X)
        else:
            raise ValueError("Input must be DataFrame, dict, or list of dicts")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information and metadata"""
        
        info = {
            'model_loaded': self.model is not None,
            'preprocessor_loaded': self.preprocessor is not None,
            'feature_names': self.feature_names,
            'model_type': str(type(self.model).__name__) if self.model else None,
        }
        
        if self.model_metadata:
            info['metadata'] = self.model_metadata
        
        return info


class ChurnPredictor(ModelPredictor):
    """
    Specialized predictor for player churn prediction
    
    Provides gaming-specific prediction capabilities with
    business-relevant outputs and interpretations.
    """
    
    def __init__(self, model_path: str = None, model_object: Any = None):
        """Initialize churn predictor"""
        super().__init__(model_path, model_object)
        
        # Churn-specific thresholds and parameters
        self.churn_threshold = 0.5
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
    def predict_churn(self, player_data: Union[pd.DataFrame, Dict, List[Dict]], 
                     include_probability: bool = True, 
                     include_risk_factors: bool = True) -> List[Dict[str, Any]]:
        """
        Predict churn with detailed player-specific insights
        
        Args:
            player_data: Player behavioral data
            include_probability: Include churn probability scores
            include_risk_factors: Include risk factor analysis
            
        Returns:
            List of prediction dictionaries with insights
        """
        
        # Convert to DataFrame
        X_df = self._prepare_input(player_data)
        
        # Get predictions and probabilities
        predictions = self.predict(X_df)
        probabilities = self.predict_proba(X_df)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Generate detailed results
        results = []
        for i, (_, row) in enumerate(X_df.iterrows()):
            player_result = {
                'player_id': row.get('player_id', f'player_{i}'),
                'churn_prediction': bool(predictions[i]),
                'prediction_confidence': 'high' if abs(probabilities[i] - 0.5) > 0.3 else 'medium' if abs(probabilities[i] - 0.5) > 0.15 else 'low'
            }
            
            if include_probability:
                player_result['churn_probability'] = float(probabilities[i]) if probabilities is not None else None
                player_result['risk_level'] = self._categorize_risk(probabilities[i]) if probabilities is not None else 'unknown'
            
            if include_risk_factors:
                player_result['risk_factors'] = self._analyze_risk_factors(row)
                player_result['recommendations'] = self._generate_recommendations(row, probabilities[i] if probabilities is not None else 0.5)
            
            results.append(player_result)
        
        return results
    
    def predict_churn_batch(self, player_data: pd.DataFrame, 
                           batch_size: int = 1000) -> pd.DataFrame:
        """Predict churn for large batches of players efficiently"""
        
        results = []
        
        for start_idx in range(0, len(player_data), batch_size):
            end_idx = min(start_idx + batch_size, len(player_data))
            batch = player_data.iloc[start_idx:end_idx]
            
            # Get predictions for batch
            predictions = self.predict(batch)
            probabilities = self.predict_proba(batch)[:, 1] if hasattr(self.model, 'predict_proba') else np.full(len(batch), 0.5)
            
            # Create results DataFrame for batch
            batch_results = pd.DataFrame({
                'player_id': batch.get('player_id', range(start_idx, end_idx)),
                'churn_prediction': predictions,
                'churn_probability': probabilities,
                'risk_level': [self._categorize_risk(p) for p in probabilities]
            })
            
            results.append(batch_results)
            
            logger.info(f"Processed batch {start_idx//batch_size + 1}: {len(batch)} players")
        
        return pd.concat(results, ignore_index=True)
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize churn risk based on probability"""
        
        if probability < self.risk_thresholds['low']:
            return 'low'
        elif probability < self.risk_thresholds['medium']:
            return 'medium'
        elif probability < self.risk_thresholds['high']:
            return 'high'
        else:
            return 'critical'
    
    def _analyze_risk_factors(self, player_data: pd.Series) -> List[str]:
        """Analyze specific risk factors for a player"""
        
        risk_factors = []
        
        # Inactivity risk
        if player_data.get('last_login_days_ago', 0) > 14:
            risk_factors.append('long_inactivity_period')
        elif player_data.get('last_login_days_ago', 0) > 7:
            risk_factors.append('recent_inactivity')
        
        # Engagement risk
        if player_data.get('avg_session_duration', 60) < 15:
            risk_factors.append('short_session_duration')
        
        if player_data.get('sessions_last_week', 3) == 0:
            risk_factors.append('no_recent_sessions')
        elif player_data.get('sessions_last_week', 3) < 2:
            risk_factors.append('low_session_frequency')
        
        # Social risk
        if player_data.get('friends_count', 5) == 0:
            risk_factors.append('no_social_connections')
        elif player_data.get('friends_count', 5) < 3:
            risk_factors.append('limited_social_network')
        
        # Monetary risk
        if player_data.get('total_spent', 0) == 0:
            risk_factors.append('no_monetary_investment')
        
        # Achievement risk
        if player_data.get('achievements_unlocked', 10) < 5:
            risk_factors.append('low_achievement_progress')
        
        return risk_factors
    
    def _generate_recommendations(self, player_data: pd.Series, churn_probability: float) -> List[str]:
        """Generate retention recommendations based on player profile"""
        
        recommendations = []
        
        if churn_probability > self.risk_thresholds['high']:
            recommendations.append('immediate_intervention_required')
            recommendations.append('personal_outreach_campaign')
        
        # Specific recommendations based on risk factors
        if player_data.get('last_login_days_ago', 0) > 7:
            recommendations.append('send_comeback_incentive')
            recommendations.append('highlight_new_content')
        
        if player_data.get('friends_count', 5) < 3:
            recommendations.append('social_features_promotion')
            recommendations.append('friend_recommendation_system')
        
        if player_data.get('total_spent', 0) == 0:
            recommendations.append('starter_pack_offer')
            recommendations.append('trial_premium_features')
        
        if player_data.get('avg_session_duration', 60) < 30:
            recommendations.append('engagement_boosting_content')
            recommendations.append('tutorial_improvement')
        
        return recommendations
    
    def set_custom_threshold(self, threshold: float) -> None:
        """Set custom churn prediction threshold"""
        if 0 <= threshold <= 1:
            self.churn_threshold = threshold
            logger.info(f"Churn threshold set to {threshold}")
        else:
            raise ValueError("Threshold must be between 0 and 1")


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models for robust predictions
    
    Supports various ensemble methods including voting, averaging,
    and stacking for improved prediction accuracy.
    """
    
    def __init__(self, models: Dict[str, Any] = None, ensemble_method: str = 'average'):
        """
        Initialize ensemble predictor
        
        Args:
            models: Dictionary of {model_name: model_object}
            ensemble_method: Method for combining predictions ('vote', 'average', 'weighted')
        """
        self.models = models or {}
        self.ensemble_method = ensemble_method
        self.model_weights = {}
        self.preprocessors = {}
        
    def add_model(self, name: str, model: Any, weight: float = 1.0, 
                  preprocessor: Any = None) -> None:
        """Add a model to the ensemble"""
        
        self.models[name] = model
        self.model_weights[name] = weight
        
        if preprocessor:
            self.preprocessors[name] = preprocessor
        
        logger.info(f"Added model {name} to ensemble with weight {weight}")
    
    def load_models_from_directory(self, models_dir: str, 
                                  model_names: List[str] = None) -> None:
        """Load multiple models from a directory"""
        
        models_path = Path(models_dir)
        
        # Load metadata to get model names if not provided
        metadata_file = models_path / "model_metadata.json"
        if metadata_file.exists() and model_names is None:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            model_names = list(metadata.get('model_results', {}).keys())
        
        # Load each model
        for model_name in model_names or []:
            model_file = models_path / f"{model_name}_model.pkl"
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                # Set weight based on model performance if available
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    model_results = metadata.get('model_results', {})
                    weight = model_results.get(model_name, {}).get('roc_auc', 1.0)
                else:
                    weight = 1.0
                
                self.add_model(model_name, model, weight)
        
        # Load shared preprocessor
        preprocessor_file = models_path / "preprocessor.pkl"
        if preprocessor_file.exists():
            with open(preprocessor_file, 'rb') as f:
                shared_preprocessor = pickle.load(f)
            
            # Assign to all models that don't have their own preprocessor
            for model_name in self.models.keys():
                if model_name not in self.preprocessors:
                    self.preprocessors[model_name] = shared_preprocessor
    
    def predict(self, X: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """Make ensemble predictions"""
        
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Convert input to DataFrame
        if isinstance(X, (dict, list)):
            X = pd.DataFrame([X] if isinstance(X, dict) else X)
        
        # Get predictions from each model
        model_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Preprocess if needed
                if model_name in self.preprocessors:
                    X_processed = self.preprocessors[model_name].transform(X)
                else:
                    X_processed = X.values
                
                predictions = model.predict(X_processed)
                model_predictions[model_name] = predictions
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed to predict: {e}")
                continue
        
        if not model_predictions:
            raise ValueError("No models could make predictions")
        
        # Combine predictions based on ensemble method
        return self._combine_predictions(model_predictions)
    
    def predict_proba(self, X: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """Make ensemble probability predictions"""
        
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Convert input to DataFrame
        if isinstance(X, (dict, list)):
            X = pd.DataFrame([X] if isinstance(X, dict) else X)
        
        # Get probability predictions from each model
        model_probabilities = {}
        
        for model_name, model in self.models.items():
            try:
                if not hasattr(model, 'predict_proba'):
                    continue
                
                # Preprocess if needed
                if model_name in self.preprocessors:
                    X_processed = self.preprocessors[model_name].transform(X)
                else:
                    X_processed = X.values
                
                probabilities = model.predict_proba(X_processed)
                model_probabilities[model_name] = probabilities
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed to predict probabilities: {e}")
                continue
        
        if not model_probabilities:
            raise ValueError("No models could make probability predictions")
        
        # Combine probabilities based on ensemble method
        return self._combine_probabilities(model_probabilities)
    
    def _combine_predictions(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine binary predictions from multiple models"""
        
        if self.ensemble_method == 'vote':
            # Majority voting
            prediction_matrix = np.column_stack(list(model_predictions.values()))
            ensemble_predictions = (prediction_matrix.mean(axis=1) >= 0.5).astype(int)
            
        elif self.ensemble_method == 'weighted':
            # Weighted voting
            total_weight = 0
            weighted_sum = np.zeros(len(list(model_predictions.values())[0]))
            
            for model_name, predictions in model_predictions.items():
                weight = self.model_weights.get(model_name, 1.0)
                weighted_sum += predictions * weight
                total_weight += weight
            
            ensemble_predictions = (weighted_sum / total_weight >= 0.5).astype(int)
            
        else:  # average
            prediction_matrix = np.column_stack(list(model_predictions.values()))
            ensemble_predictions = (prediction_matrix.mean(axis=1) >= 0.5).astype(int)
        
        return ensemble_predictions
    
    def _combine_probabilities(self, model_probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine probability predictions from multiple models"""
        
        if self.ensemble_method == 'weighted':
            # Weighted average of probabilities
            total_weight = 0
            weighted_sum = np.zeros(list(model_probabilities.values())[0].shape)
            
            for model_name, probabilities in model_probabilities.items():
                weight = self.model_weights.get(model_name, 1.0)
                weighted_sum += probabilities * weight
                total_weight += weight
            
            ensemble_probabilities = weighted_sum / total_weight
            
        else:  # average or vote
            # Simple average of probabilities
            prob_array = np.stack(list(model_probabilities.values()))
            ensemble_probabilities = prob_array.mean(axis=0)
        
        return ensemble_probabilities
    
    def get_model_contributions(self, X: Union[pd.DataFrame, Dict, List[Dict]]) -> Dict[str, Dict]:
        """Get individual model contributions to ensemble prediction"""
        
        # Convert input to DataFrame
        if isinstance(X, (dict, list)):
            X = pd.DataFrame([X] if isinstance(X, dict) else X)
        
        contributions = {}
        
        for model_name, model in self.models.items():
            try:
                # Get predictions and probabilities
                if model_name in self.preprocessors:
                    X_processed = self.preprocessors[model_name].transform(X)
                else:
                    X_processed = X.values
                
                predictions = model.predict(X_processed)
                probabilities = model.predict_proba(X_processed) if hasattr(model, 'predict_proba') else None
                
                contributions[model_name] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'weight': self.model_weights.get(model_name, 1.0)
                }
                
            except Exception as e:
                logger.warning(f"Could not get contributions from {model_name}: {e}")
        
        return contributions


class RealTimePredictor:
    """
    Real-time prediction service for streaming/online churn prediction
    
    Optimized for low-latency predictions with caching and
    efficient preprocessing for production deployment.
    """
    
    def __init__(self, model_path: str = None, cache_size: int = 1000):
        """
        Initialize real-time predictor
        
        Args:
            model_path: Path to model directory
            cache_size: Size of prediction cache
        """
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.cache_size = cache_size
        self.prediction_cache = {}
        self.cache_timestamps = {}
        self.prediction_count = 0
        self.total_prediction_time = 0
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load model optimized for real-time prediction"""
        
        model_path = Path(model_path)
        
        # Load metadata
        metadata_file = model_path / "model_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load the fastest model for real-time use
            model_results = metadata.get('model_results', {})
            if model_results:
                # Prefer logistic regression or SVM for speed
                priority_models = ['logistic_regression', 'svm', 'random_forest']
                selected_model = None
                
                for preferred in priority_models:
                    if preferred in model_results:
                        selected_model = preferred
                        break
                
                if not selected_model:
                    # Fall back to best performing model
                    selected_model = max(model_results.keys(), 
                                       key=lambda x: model_results[x].get('roc_auc', 0))
                
                # Load selected model
                model_file = model_path / f"{selected_model}_model.pkl"
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        self.model = pickle.load(f)
                    logger.info(f"Loaded {selected_model} for real-time prediction")
        
        # Load preprocessor
        preprocessor_file = model_path / "preprocessor.pkl"
        if preprocessor_file.exists():
            with open(preprocessor_file, 'rb') as f:
                self.preprocessor = pickle.load(f)
    
    def predict_single(self, player_data: Dict[str, Any], 
                      player_id: str = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Make real-time prediction for a single player
        
        Args:
            player_data: Player behavioral data
            player_id: Unique player identifier
            use_cache: Whether to use prediction caching
            
        Returns:
            Prediction result with timing information
        """
        
        start_time = datetime.now()
        
        # Check cache if enabled
        if use_cache and player_id:
            cached_result = self._get_cached_prediction(player_id, player_data)
            if cached_result:
                return cached_result
        
        # Prepare input
        X = pd.DataFrame([player_data])
        
        # Preprocess
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X.values
        
        # Make prediction
        prediction = self.model.predict(X_processed)[0]
        probability = None
        
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(X_processed)[0][1]
        
        # Calculate timing
        prediction_time = (datetime.now() - start_time).total_seconds() * 1000  # milliseconds
        
        result = {
            'player_id': player_id,
            'churn_prediction': bool(prediction),
            'churn_probability': float(probability) if probability is not None else None,
            'prediction_time_ms': prediction_time,
            'timestamp': start_time.isoformat(),
            'model_version': 'real_time_v1'
        }
        
        # Cache result
        if use_cache and player_id:
            self._cache_prediction(player_id, player_data, result)
        
        # Update performance metrics
        self.prediction_count += 1
        self.total_prediction_time += prediction_time
        
        return result
    
    def predict_batch_realtime(self, player_data_batch: List[Dict[str, Any]], 
                              batch_timeout_ms: float = 100) -> List[Dict[str, Any]]:
        """
        Make real-time predictions for a batch of players with timeout
        
        Args:
            player_data_batch: List of player data dictionaries
            batch_timeout_ms: Maximum time to spend on batch (milliseconds)
            
        Returns:
            List of prediction results
        """
        
        start_time = datetime.now()
        results = []
        
        for i, player_data in enumerate(player_data_batch):
            # Check timeout
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            if elapsed_ms > batch_timeout_ms:
                logger.warning(f"Batch timeout reached after {i} predictions")
                break
            
            player_id = player_data.get('player_id', f'batch_player_{i}')
            result = self.predict_single(player_data, player_id, use_cache=True)
            results.append(result)
        
        return results
    
    def _get_cached_prediction(self, player_id: str, player_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available and fresh"""
        
        if player_id not in self.prediction_cache:
            return None
        
        cached_entry = self.prediction_cache[player_id]
        cache_time = self.cache_timestamps[player_id]
        
        # Check if cache is still fresh (within 1 hour)
        if (datetime.now() - cache_time).total_seconds() > 3600:
            del self.prediction_cache[player_id]
            del self.cache_timestamps[player_id]
            return None
        
        # Check if player data has changed significantly
        cached_hash = cached_entry.get('data_hash')
        current_hash = hash(frozenset(player_data.items()))
        
        if cached_hash != current_hash:
            return None
        
        return cached_entry['result']
    
    def _cache_prediction(self, player_id: str, player_data: Dict[str, Any], 
                         result: Dict[str, Any]) -> None:
        """Cache prediction result"""
        
        # Implement LRU-style cache management
        if len(self.prediction_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_id = min(self.cache_timestamps.keys(), 
                          key=lambda x: self.cache_timestamps[x])
            del self.prediction_cache[oldest_id]
            del self.cache_timestamps[oldest_id]
        
        # Add new entry
        data_hash = hash(frozenset(player_data.items()))
        self.prediction_cache[player_id] = {
            'result': result,
            'data_hash': data_hash
        }
        self.cache_timestamps[player_id] = datetime.now()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get real-time prediction performance statistics"""
        
        avg_prediction_time = (self.total_prediction_time / self.prediction_count 
                             if self.prediction_count > 0 else 0)
        
        return {
            'total_predictions': self.prediction_count,
            'average_prediction_time_ms': avg_prediction_time,
            'cache_size': len(self.prediction_cache),
            'cache_hit_potential': len(self.prediction_cache) / max(1, self.prediction_count),
            'model_type': str(type(self.model).__name__) if self.model else None
        }
    
    def clear_cache(self) -> None:
        """Clear prediction cache"""
        self.prediction_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Prediction cache cleared")


def main():
    """Example usage of prediction classes"""
    
    # This would typically be called from a web service or batch processing script
    logger.info("Testing prediction modules...")
    
    # Example player data
    sample_player = {
        'player_id': 'test_player_001',
        'total_playtime_hours': 150.5,
        'avg_session_duration': 45.2,
        'sessions_last_week': 8,
        'friends_count': 12,
        'total_spent': 89.99,
        'last_login_days_ago': 2,
        'achievements_unlocked': 23,
        'games_owned': 15
    }
    
    # Test individual predictor (would need trained model)
    try:
        predictor = ChurnPredictor()
        # Would load actual model: predictor.load_model("models/")
        
        print("Prediction modules loaded successfully!")
        print("Ready for production deployment.")
        
    except Exception as e:
        print(f"Note: {e}")
        print("Prediction modules are ready but need trained models to run.")

if __name__ == "__main__":
    main()