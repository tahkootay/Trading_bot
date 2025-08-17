"""
ML Predictor with Ensemble Voting for SOL/USDT Trading Algorithm

Implements the machine learning prediction system as specified in the algorithm
with ensemble voting, regime-adaptive thresholds, and optimal stop loss prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import joblib
import logging
from datetime import datetime

@dataclass
class MLPrediction:
    """ML prediction results"""
    p_up: float
    p_down: float
    p_flat: float
    direction: str  # 'long', 'short', 'flat'
    confidence: float
    agreement: float
    margin: float
    expected_move_size: str  # 'small', 'medium', 'large'
    expected_time: float  # minutes to target
    optimal_sl_multiplier: float
    passes_ml: bool
    confidence_required: float
    margin_required: float
    agreement_required: float

@dataclass
class ModelPrediction:
    """Individual model prediction"""
    up: float
    down: float
    flat: float
    size: str = 'medium'
    time_to_target: float = 60.0
    optimal_sl: float = 1.0
    confidence: float = 0.0

class FeatureCalculator:
    """Calculate ML features from market data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, market_data: Dict) -> Dict[str, float]:
        """
        Calculate all ML features as specified in algorithm
        
        Args:
            market_data: Dictionary containing market indicators and data
            
        Returns:
            Dictionary of calculated features
        """
        features = {}
        
        try:
            # Price features
            features.update(self._price_features(market_data))
            
            # Technical indicators
            features.update(self._technical_features(market_data))
            
            # Volume features
            features.update(self._volume_features(market_data))
            
            # Market structure
            features.update(self._structure_features(market_data))
            
            # Order flow
            features.update(self._order_flow_features(market_data))
            
            # Cross-market
            features.update(self._cross_market_features(market_data))
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            
        return features
    
    def _price_features(self, market_data: Dict) -> Dict[str, float]:
        """Calculate price-based features"""
        features = {}
        
        close_data = market_data.get('close', [])
        if len(close_data) >= 60:
            # Price changes over different periods
            features['price_change_1m'] = self._safe_pct_change(close_data, 1)
            features['price_change_5m'] = self._safe_pct_change(close_data, 5)
            features['price_change_15m'] = self._safe_pct_change(close_data, 15)
            features['price_change_1h'] = self._safe_pct_change(close_data, 60)
        
        return features
    
    def _technical_features(self, market_data: Dict) -> Dict[str, float]:
        """Calculate technical indicator features"""
        features = {}
        
        close = market_data.get('close', 0)
        if isinstance(close, list) and close:
            close = close[-1]
        
        # RSI
        features['rsi'] = market_data.get('rsi', 50)
        
        # ADX
        features['adx'] = market_data.get('adx', 20)
        
        # ATR normalized by price
        atr = market_data.get('atr', 0)
        if close > 0:
            features['atr_normalized'] = atr / close
        else:
            features['atr_normalized'] = 0
        
        # EMA distance
        ema21 = market_data.get('ema_21', close)
        if ema21 > 0:
            features['ema_distance'] = (close - ema21) / ema21
        else:
            features['ema_distance'] = 0
        
        return features
    
    def _volume_features(self, market_data: Dict) -> Dict[str, float]:
        """Calculate volume features"""
        features = {}
        
        # Volume ratio (current vs average)
        volume = market_data.get('volume', 0)
        volume_ma = market_data.get('volume_ma', volume)
        if volume_ma > 0:
            features['volume_ratio'] = volume / volume_ma
        else:
            features['volume_ratio'] = 1.0
        
        # Z-score volume
        features['zvol'] = market_data.get('zvol', 0)
        
        return features
    
    def _structure_features(self, market_data: Dict) -> Dict[str, float]:
        """Calculate market structure features"""
        features = {}
        
        close = market_data.get('close', 0)
        if isinstance(close, list) and close:
            close = close[-1]
        
        # Distance to nearest resistance/support
        nearest_resistance = market_data.get('nearest_resistance', close * 1.02)
        nearest_support = market_data.get('nearest_support', close * 0.98)
        
        if close > 0:
            features['distance_to_resistance'] = (nearest_resistance - close) / close
            features['distance_to_support'] = (close - nearest_support) / close
        else:
            features['distance_to_resistance'] = 0.02
            features['distance_to_support'] = 0.02
        
        return features
    
    def _order_flow_features(self, market_data: Dict) -> Dict[str, float]:
        """Calculate order flow features"""
        features = {}
        
        # CVD normalized by volume
        cvd = market_data.get('cvd', 0)
        volume = market_data.get('volume', 1)
        features['cvd_normalized'] = cvd / volume if volume > 0 else 0
        
        # Order imbalance
        features['order_imbalance'] = market_data.get('order_imbalance', 1.0)
        
        # Trade intensity
        features['trade_intensity'] = market_data.get('trade_intensity', 0)
        
        return features
    
    def _cross_market_features(self, market_data: Dict) -> Dict[str, float]:
        """Calculate cross-market features"""
        features = {}
        
        # BTC correlation
        features['btc_correlation'] = market_data.get('btc_correlation', 0.7)
        
        # BTC trend (simplified)
        btc_trend = market_data.get('btc_trend', 0)
        features['btc_trend'] = btc_trend
        
        return features
    
    def _safe_pct_change(self, data: List[float], periods: int) -> float:
        """Safely calculate percentage change"""
        if len(data) <= periods:
            return 0.0
        
        current = data[-1]
        previous = data[-periods-1]
        
        if previous == 0:
            return 0.0
        
        return (current - previous) / previous

class MLPredictor:
    """
    ML Prediction system with ensemble voting
    Implements algorithm specification exactly
    """
    
    def __init__(self, models_path: str, config: Dict = None):
        self.models_path = Path(models_path)
        self.config = config or {}
        self.models = {}
        self.feature_calculator = FeatureCalculator()
        self.logger = logging.getLogger(__name__)
        
        # Algorithm constants from specification
        self.ENTRY_FILTERS = {
            'default': {
                'ml_margin_min': 0.15,
                'ml_conf_min': 0.70,
                'ml_agreement_min': 0.60
            },
            'trending': {
                'ml_margin_min': 0.12,
                'ml_conf_min': 0.65,
                'ml_agreement_min': 0.55
            },
            'ranging': {
                'ml_margin_min': 0.18,
                'ml_conf_min': 0.75,
                'ml_agreement_min': 0.65
            },
            'high_volatility': {
                'ml_margin_min': 0.20,
                'ml_conf_min': 0.75,
                'ml_agreement_min': 0.70
            }
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load trained ML models"""
        try:
            # For now, create mock models since actual trained models aren't available
            # In production, this would load actual trained models
            self.models = {
                'xgboost_main': self._create_mock_model('xgboost'),
                'lightgbm_main': self._create_mock_model('lightgbm'),
                'neural_net': self._create_mock_model('neural_net'),
                'ensemble_meta': self._create_mock_model('ensemble')
            }
            self.logger.info(f"Loaded {len(self.models)} ML models")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.models = {}
    
    def _create_mock_model(self, model_type: str) -> Dict:
        """Create mock model for testing (replace with actual model loading)"""
        return {
            'type': model_type,
            'version': '1.0.0',
            'trained_date': datetime.now().isoformat(),
            'features': ['rsi', 'adx', 'volume_ratio', 'ema_distance'],
            'mock': True
        }
    
    def predict(self, market_data: Dict, regime: str) -> MLPrediction:
        """
        Generate ML predictions with ensemble voting
        
        Args:
            market_data: Current market state and indicators
            regime: Current market regime
            
        Returns:
            MLPrediction with ensemble results and thresholds
        """
        try:
            # Calculate features
            features = self.feature_calculator.calculate(market_data)
            
            if not features:
                return self._default_prediction(regime)
            
            # Get predictions from each model
            predictions = []
            for model_name, model in self.models.items():
                pred = self._predict_single_model(model, features, model_name)
                if pred:
                    predictions.append(pred)
            
            if not predictions:
                return self._default_prediction(regime)
            
            # Ensemble voting
            ensemble_result = self._ensemble_vote(predictions)
            
            # Adjust thresholds based on regime
            adjusted_result = self._adjust_for_regime(ensemble_result, regime)
            
            return adjusted_result
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return self._default_prediction(regime)
    
    def _predict_single_model(self, model: Dict, features: Dict, model_name: str) -> Optional[ModelPrediction]:
        """Get prediction from a single model"""
        try:
            # Mock prediction logic - replace with actual model inference
            if model.get('mock', False):
                return self._mock_prediction(features, model_name)
            
            # In production, this would be:
            # model_object = model['object']
            # feature_vector = self._features_to_vector(features, model['features'])
            # probabilities = model_object.predict_proba(feature_vector)
            # return self._parse_model_output(probabilities, model_name)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error predicting with model {model_name}: {e}")
            return None
    
    def _mock_prediction(self, features: Dict, model_name: str) -> ModelPrediction:
        """Generate mock prediction for testing"""
        # Create realistic-looking predictions based on features
        rsi = features.get('rsi', 50)
        adx = features.get('adx', 20)
        volume_ratio = features.get('volume_ratio', 1.0)
        ema_distance = features.get('ema_distance', 0)
        
        # Simple logic for mock predictions
        if rsi > 60 and adx > 25 and ema_distance > 0:
            # Bullish bias
            up_prob = min(0.8, 0.5 + (rsi - 50) / 100 + adx / 100)
            down_prob = max(0.1, 0.4 - (rsi - 50) / 100)
        elif rsi < 40 and adx > 25 and ema_distance < 0:
            # Bearish bias
            up_prob = max(0.1, 0.4 + (rsi - 50) / 100)
            down_prob = min(0.8, 0.5 - (rsi - 50) / 100 + adx / 100)
        else:
            # Neutral
            up_prob = 0.4 + np.random.normal(0, 0.1)
            down_prob = 0.4 + np.random.normal(0, 0.1)
        
        flat_prob = max(0.1, 1.0 - up_prob - down_prob)
        
        # Normalize probabilities
        total = up_prob + down_prob + flat_prob
        up_prob /= total
        down_prob /= total
        flat_prob /= total
        
        # Expected move size based on volatility
        atr_norm = features.get('atr_normalized', 0.02)
        if atr_norm > 0.03:
            size = 'large'
            time_to_target = 30
        elif atr_norm > 0.02:
            size = 'medium'
            time_to_target = 60
        else:
            size = 'small'
            time_to_target = 120
        
        # Optimal stop loss based on volatility and ADX
        optimal_sl = max(0.8, min(2.0, 1.0 + atr_norm * 10 + (25 - adx) / 50))
        
        return ModelPrediction(
            up=up_prob,
            down=down_prob,
            flat=flat_prob,
            size=size,
            time_to_target=time_to_target,
            optimal_sl=optimal_sl,
            confidence=max(up_prob, down_prob, flat_prob)
        )
    
    def _ensemble_vote(self, predictions: List[ModelPrediction]) -> MLPrediction:
        """Combine predictions from multiple models using ensemble voting"""
        
        # Average probabilities
        p_up = np.mean([p.up for p in predictions])
        p_down = np.mean([p.down for p in predictions])
        p_flat = np.mean([p.flat for p in predictions])
        
        # Calculate agreement (how many models agree on direction)
        directions = [self._get_direction(p) for p in predictions]
        most_common = max(set(directions), key=directions.count)
        agreement = directions.count(most_common) / len(directions)
        
        # Calculate confidence and margin
        confidence = max(p_up, p_down, p_flat)
        margin = abs(p_up - p_down)
        
        # Aggregate other predictions
        size_predictions = [p.size for p in predictions]
        time_predictions = [p.time_to_target for p in predictions]
        sl_predictions = [p.optimal_sl for p in predictions]
        
        expected_move_size = max(set(size_predictions), key=size_predictions.count)
        expected_time = np.median(time_predictions)
        optimal_sl_multiplier = np.mean(sl_predictions)
        
        return MLPrediction(
            p_up=p_up,
            p_down=p_down,
            p_flat=p_flat,
            direction=most_common,
            confidence=confidence,
            agreement=agreement,
            margin=margin,
            expected_move_size=expected_move_size,
            expected_time=expected_time,
            optimal_sl_multiplier=optimal_sl_multiplier,
            passes_ml=False,  # Will be set in adjust_for_regime
            confidence_required=0.7,  # Will be set in adjust_for_regime
            margin_required=0.15,  # Will be set in adjust_for_regime
            agreement_required=0.6  # Will be set in adjust_for_regime
        )
    
    def _adjust_for_regime(self, prediction: MLPrediction, regime: str) -> MLPrediction:
        """Adjust ML thresholds based on market regime"""
        
        # Get regime-specific adjustments from algorithm specification
        regime_adjustments = {
            'strong_trend': {
                'confidence_mult': 0.9,
                'margin_mult': 0.8,
                'agreement_mult': 0.9
            },
            'trending': {
                'confidence_mult': 0.95,
                'margin_mult': 0.9,
                'agreement_mult': 0.95
            },
            'normal_range': {
                'confidence_mult': 1.0,
                'margin_mult': 1.0,
                'agreement_mult': 1.0
            },
            'low_volatility_range': {
                'confidence_mult': 1.1,
                'margin_mult': 1.2,
                'agreement_mult': 1.1
            },
            'volatile_choppy': {
                'confidence_mult': 1.2,
                'margin_mult': 1.3,
                'agreement_mult': 1.2
            }
        }
        
        adj = regime_adjustments.get(regime, regime_adjustments['normal_range'])
        
        # Get base thresholds
        base_filters = self.ENTRY_FILTERS.get('default')
        
        # Apply regime adjustments
        prediction.confidence_required = base_filters['ml_conf_min'] * adj['confidence_mult']
        prediction.margin_required = base_filters['ml_margin_min'] * adj['margin_mult']
        prediction.agreement_required = base_filters['ml_agreement_min'] * adj['agreement_mult']
        
        # Check if ML filters pass
        prediction.passes_ml = (
            prediction.confidence >= prediction.confidence_required and
            prediction.margin >= prediction.margin_required and
            prediction.agreement >= prediction.agreement_required
        )
        
        return prediction
    
    def _get_direction(self, prediction: ModelPrediction) -> str:
        """Get direction from model prediction"""
        if prediction.up > prediction.down and prediction.up > prediction.flat:
            return 'long'
        elif prediction.down > prediction.up and prediction.down > prediction.flat:
            return 'short'
        else:
            return 'flat'
    
    def _default_prediction(self, regime: str) -> MLPrediction:
        """Return default prediction when models unavailable"""
        default = MLPrediction(
            p_up=0.33,
            p_down=0.33,
            p_flat=0.34,
            direction='flat',
            confidence=0.34,
            agreement=0.33,
            margin=0.0,
            expected_move_size='medium',
            expected_time=60.0,
            optimal_sl_multiplier=1.0,
            passes_ml=False,
            confidence_required=0.7,
            margin_required=0.15,
            agreement_required=0.6
        )
        
        return self._adjust_for_regime(default, regime)
    
    def retrain_models(self, training_data: pd.DataFrame) -> bool:
        """Retrain models with new data (placeholder for future implementation)"""
        try:
            self.logger.info("Model retraining not implemented yet")
            return False
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")
            return False
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for loaded models"""
        performance = {}
        
        for model_name, model in self.models.items():
            performance[model_name] = {
                'type': model.get('type', 'unknown'),
                'version': model.get('version', '0.0.0'),
                'mock': model.get('mock', False),
                'status': 'loaded' if model else 'failed'
            }
        
        return performance