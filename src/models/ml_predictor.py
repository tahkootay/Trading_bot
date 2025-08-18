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
    """Calculate ML features from market data as per algorithm specification"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, market_data: Dict) -> Dict[str, float]:
        """
        Calculate all ML features as specified in algorithm document
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Dictionary of calculated features
        """
        try:
            features = {}
            
            # Price features
            close_prices = market_data.get('close', [])
            if isinstance(close_prices, list) and len(close_prices) >= 60:
                features['price_change_1m'] = self._safe_pct_change(close_prices, 1)
                features['price_change_5m'] = self._safe_pct_change(close_prices, 5)
                features['price_change_15m'] = self._safe_pct_change(close_prices, 15)
                features['price_change_1h'] = self._safe_pct_change(close_prices, 60)
            else:
                # Fallback for single values
                current_price = close_prices[-1] if close_prices else market_data.get('close', 0)
                features['price_change_1m'] = 0.0
                features['price_change_5m'] = 0.0
                features['price_change_15m'] = 0.0
                features['price_change_1h'] = 0.0
            
            # Technical indicators
            features['rsi'] = market_data.get('rsi', 50.0)
            features['adx'] = market_data.get('adx', 25.0)
            features['atr_normalized'] = market_data.get('atr', 0) / max(market_data.get('close', 1), 1)
            
            # EMA distance
            close = market_data.get('close', [])[-1] if isinstance(market_data.get('close', []), list) else market_data.get('close', 0)
            ema21 = market_data.get('ema_21', close)
            features['ema_distance'] = (close - ema21) / ema21 if ema21 > 0 else 0
            
            # Volume features
            volume = market_data.get('volume', 0)
            volume_ma = market_data.get('volume_ma', volume)
            features['volume_ratio'] = volume / volume_ma if volume_ma > 0 else 1.0
            features['zvol'] = market_data.get('zvol', 0.0)
            
            # Market structure
            features['distance_to_resistance'] = market_data.get('nearest_resistance', close) - close
            features['distance_to_support'] = close - market_data.get('nearest_support', close)
            
            # Order flow
            cvd = market_data.get('cvd', 0)
            volume_current = market_data.get('volume', 1)
            features['cvd_normalized'] = cvd / volume_current if volume_current > 0 else 0
            features['order_imbalance'] = market_data.get('order_imbalance', 1.0)
            features['trade_intensity'] = market_data.get('trade_intensity', 0.0)
            
            # Cross-market
            features['btc_correlation'] = market_data.get('btc_correlation', 0.7)
            features['btc_trend'] = market_data.get('btc_trend', 0)
            
            # Additional features for robustness
            features['volatility_regime'] = self._classify_volatility_regime(market_data)
            features['momentum_score'] = self._calculate_momentum_score(market_data)
            features['mean_reversion_score'] = self._calculate_mean_reversion_score(market_data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            return self._get_default_features()
    
    def _safe_pct_change(self, prices: List[float], periods: int) -> float:
        """Safely calculate percentage change"""
        try:
            if len(prices) < periods + 1:
                return 0.0
            
            current = prices[-1]
            previous = prices[-(periods + 1)]
            
            return (current - previous) / previous if previous != 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _classify_volatility_regime(self, market_data: Dict) -> float:
        """Classify volatility regime (0.0 = low, 1.0 = high)"""
        try:
            atr = market_data.get('atr', 0)
            atr_avg = market_data.get('atr_20d_avg', atr)
            
            if atr_avg > 0:
                ratio = atr / atr_avg
                return min(1.0, max(0.0, (ratio - 0.5) / 1.5))  # Normalize to 0-1
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_momentum_score(self, market_data: Dict) -> float:
        """Calculate momentum score (-1.0 to 1.0)"""
        try:
            # Combine multiple momentum indicators
            rsi = market_data.get('rsi', 50)
            adx = market_data.get('adx', 20)
            ema_distance = market_data.get('ema_distance', 0)
            
            # RSI component (-1 to 1)
            rsi_score = (rsi - 50) / 50
            
            # ADX component (0 to 1)
            adx_score = min(1.0, adx / 50)
            
            # EMA distance component
            ema_score = max(-1.0, min(1.0, ema_distance * 10))
            
            # Weighted combination
            momentum_score = (rsi_score * 0.4 + ema_score * 0.6) * adx_score
            
            return max(-1.0, min(1.0, momentum_score))
            
        except Exception:
            return 0.0
    
    def _calculate_mean_reversion_score(self, market_data: Dict) -> float:
        """Calculate mean reversion score (0.0 to 1.0)"""
        try:
            # Distance from VWAP
            close = market_data.get('close', 0)
            if isinstance(close, list):
                close = close[-1] if close else 0
            
            vwap = market_data.get('vwap', close)
            vwap_upper = market_data.get('vwap_upper_2sigma', close * 1.02)
            vwap_lower = market_data.get('vwap_lower_2sigma', close * 0.98)
            
            if close > vwap_upper:
                # Price above upper band - mean reversion opportunity (short)
                return (close - vwap_upper) / (vwap_upper - vwap) if vwap_upper > vwap else 0
            elif close < vwap_lower:
                # Price below lower band - mean reversion opportunity (long)
                return (vwap_lower - close) / (vwap - vwap_lower) if vwap > vwap_lower else 0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default features when calculation fails"""
        return {
            'price_change_1m': 0.0,
            'price_change_5m': 0.0,
            'price_change_15m': 0.0,
            'price_change_1h': 0.0,
            'rsi': 50.0,
            'adx': 25.0,
            'atr_normalized': 0.015,
            'ema_distance': 0.0,
            'volume_ratio': 1.0,
            'zvol': 0.0,
            'distance_to_resistance': 10.0,
            'distance_to_support': 10.0,
            'cvd_normalized': 0.0,
            'order_imbalance': 1.0,
            'trade_intensity': 0.0,
            'btc_correlation': 0.7,
            'btc_trend': 0.0,
            'volatility_regime': 0.5,
            'momentum_score': 0.0,
            'mean_reversion_score': 0.0
        }


class MLPredictor:
    """
    ML Predictor with ensemble voting as per algorithm specification
    
    Implements:
    - Multiple model ensemble with voting
    - Regime-adaptive thresholds
    - Optimal stop loss prediction
    - Agreement and confidence metrics
    """
    
    def __init__(self, models_path: str, config: Dict = None):
        self.models_path = Path(models_path)
        self.config = config or {}
        self.models = {}
        self.feature_calculator = FeatureCalculator()
        self.logger = logging.getLogger(__name__)
        
        # Algorithm constants from specification
        from ..utils.algorithm_constants import ENTRY_FILTERS
        self.ENTRY_FILTERS = ENTRY_FILTERS
        
        # Model weights for ensemble (can be trained/optimized)
        self.ensemble_weights = self.config.get('ensemble_weights', {
            'xgboost': 0.3,
            'lightgbm': 0.3,
            'random_forest': 0.2,
            'neural_network': 0.2
        })
        
        # Load models if available
        self.models_loaded = False
        self._load_models()
    
    def _load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            model_files = {
                'xgboost': 'xgb_model.pkl',
                'lightgbm': 'lgb_model.pkl',
                'random_forest': 'rf_model.pkl',
                'neural_network': 'nn_model.pkl'
            }
            
            loaded_count = 0
            for model_name, filename in model_files.items():
                model_path = self.models_path / filename
                if model_path.exists():
                    try:
                        self.models[model_name] = joblib.load(model_path)
                        loaded_count += 1
                        self.logger.info(f"Loaded {model_name} from {model_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load {model_name}: {e}")
            
            self.models_loaded = loaded_count > 0
            
            if not self.models_loaded:
                self.logger.warning("No ML models loaded - using dummy predictions")
                self._create_dummy_models()
            
            return self.models_loaded
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self._create_dummy_models()
            return False
    
    def _create_dummy_models(self):
        """Create dummy models for testing when real models not available"""
        self.models = {
            'dummy_trend': self._create_dummy_trend_model(),
            'dummy_mean_reversion': self._create_dummy_mean_reversion_model(),
            'dummy_momentum': self._create_dummy_momentum_model()
        }
        
        self.ensemble_weights = {
            'dummy_trend': 0.4,
            'dummy_mean_reversion': 0.3,
            'dummy_momentum': 0.3
        }
    
    def _create_dummy_trend_model(self):
        """Create dummy trend-following model"""
        class DummyTrendModel:
            def predict(self, features):
                momentum = features.get('momentum_score', 0)
                adx = features.get('adx', 25)
                
                if momentum > 0.2 and adx > 25:
                    return {
                        'up': 0.7, 'down': 0.2, 'flat': 0.1,
                        'size': 'medium', 'time_to_target': 90,
                        'optimal_sl': 1.2, 'confidence': 0.7
                    }
                elif momentum < -0.2 and adx > 25:
                    return {
                        'up': 0.2, 'down': 0.7, 'flat': 0.1,
                        'size': 'medium', 'time_to_target': 90,
                        'optimal_sl': 1.2, 'confidence': 0.7
                    }
                else:
                    return {
                        'up': 0.35, 'down': 0.35, 'flat': 0.3,
                        'size': 'small', 'time_to_target': 60,
                        'optimal_sl': 1.0, 'confidence': 0.5
                    }
        
        return DummyTrendModel()
    
    def _create_dummy_mean_reversion_model(self):
        """Create dummy mean reversion model"""
        class DummyMeanReversionModel:
            def predict(self, features):
                mean_rev_score = features.get('mean_reversion_score', 0)
                rsi = features.get('rsi', 50)
                
                if mean_rev_score > 0.5 and rsi > 70:
                    return {
                        'up': 0.2, 'down': 0.7, 'flat': 0.1,
                        'size': 'small', 'time_to_target': 45,
                        'optimal_sl': 0.8, 'confidence': 0.65
                    }
                elif mean_rev_score > 0.5 and rsi < 30:
                    return {
                        'up': 0.7, 'down': 0.2, 'flat': 0.1,
                        'size': 'small', 'time_to_target': 45,
                        'optimal_sl': 0.8, 'confidence': 0.65
                    }
                else:
                    return {
                        'up': 0.4, 'down': 0.4, 'flat': 0.2,
                        'size': 'small', 'time_to_target': 30,
                        'optimal_sl': 1.0, 'confidence': 0.4
                    }
        
        return DummyMeanReversionModel()
    
    def _create_dummy_momentum_model(self):
        """Create dummy momentum model"""
        class DummyMomentumModel:
            def predict(self, features):
                volume_ratio = features.get('volume_ratio', 1)
                zvol = features.get('zvol', 0)
                price_change_5m = features.get('price_change_5m', 0)
                
                if volume_ratio > 2 and zvol > 2 and abs(price_change_5m) > 0.01:
                    direction = 'up' if price_change_5m > 0 else 'down'
                    if direction == 'up':
                        return {
                            'up': 0.8, 'down': 0.1, 'flat': 0.1,
                            'size': 'large', 'time_to_target': 30,
                            'optimal_sl': 1.0, 'confidence': 0.8
                        }
                    else:
                        return {
                            'up': 0.1, 'down': 0.8, 'flat': 0.1,
                            'size': 'large', 'time_to_target': 30,
                            'optimal_sl': 1.0, 'confidence': 0.8
                        }
                else:
                    return {
                        'up': 0.35, 'down': 0.35, 'flat': 0.3,
                        'size': 'medium', 'time_to_target': 60,
                        'optimal_sl': 1.0, 'confidence': 0.5
                    }
        
        return DummyMomentumModel()
    
    def predict(self, market_data: Dict, regime: str) -> MLPrediction:
        """
        Generate ML predictions with ensemble voting as per algorithm specification
        
        Args:
            market_data: Market data for feature calculation
            regime: Current market regime
            
        Returns:
            MLPrediction with ensemble results and regime adjustments
        """
        try:
            # Calculate features
            features = self.feature_calculator.calculate(market_data)
            
            # Get predictions from each model
            predictions = []
            for model_name, model in self.models.items():
                try:
                    pred_dict = model.predict(features)
                    pred = ModelPrediction(
                        up=pred_dict.get('up', 0.33),
                        down=pred_dict.get('down', 0.33),
                        flat=pred_dict.get('flat', 0.34),
                        size=pred_dict.get('size', 'medium'),
                        time_to_target=pred_dict.get('time_to_target', 60),
                        optimal_sl=pred_dict.get('optimal_sl', 1.0),
                        confidence=pred_dict.get('confidence', 0.5)
                    )
                    predictions.append((model_name, pred))
                except Exception as e:
                    self.logger.warning(f"Model {model_name} prediction failed: {e}")
            
            if not predictions:
                return self._get_default_prediction(regime)
            
            # Ensemble voting with weights
            ensemble_result = self._ensemble_vote(predictions)
            
            # Adjust thresholds based on regime
            adjusted_result = self._adjust_for_regime(ensemble_result, regime)
            
            return adjusted_result
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return self._get_default_prediction(regime)
    
    def _ensemble_vote(self, predictions: List[Tuple[str, ModelPrediction]]) -> MLPrediction:
        """
        Combine predictions from multiple models using weighted voting
        
        Args:
            predictions: List of (model_name, prediction) tuples
            
        Returns:
            Ensemble MLPrediction
        """
        try:
            if not predictions:
                return self._get_default_prediction()
            
            # Weighted average of probabilities
            total_weight = 0
            weighted_up = 0
            weighted_down = 0
            weighted_flat = 0
            
            size_votes = []
            time_predictions = []
            sl_predictions = []
            confidence_scores = []
            
            for model_name, pred in predictions:
                weight = self.ensemble_weights.get(model_name, 1.0 / len(predictions))
                total_weight += weight
                
                weighted_up += pred.up * weight
                weighted_down += pred.down * weight
                weighted_flat += pred.flat * weight
                
                size_votes.append(pred.size)
                time_predictions.append(pred.time_to_target)
                sl_predictions.append(pred.optimal_sl)
                confidence_scores.append(pred.confidence)
            
            # Normalize probabilities
            if total_weight > 0:
                p_up = weighted_up / total_weight
                p_down = weighted_down / total_weight
                p_flat = weighted_flat / total_weight
            else:
                p_up = p_down = p_flat = 1.0 / 3
            
            # Determine direction
            direction = self._get_direction_from_probabilities(p_up, p_down, p_flat)
            
            # Calculate agreement (how many models agree on direction)
            directions = [self._get_direction_from_probabilities(p.up, p.down, p.flat) for _, p in predictions]
            most_common_direction = max(set(directions), key=directions.count)
            agreement = directions.count(most_common_direction) / len(directions)
            
            # Calculate confidence and margin
            confidence = max(p_up, p_down, p_flat)
            margin = abs(p_up - p_down)
            
            # Aggregate other predictions
            expected_move_size = max(set(size_votes), key=size_votes.count) if size_votes else 'medium'
            expected_time = np.median(time_predictions) if time_predictions else 60.0
            optimal_sl_multiplier = np.mean(sl_predictions) if sl_predictions else 1.0
            
            return MLPrediction(
                p_up=p_up,
                p_down=p_down,
                p_flat=p_flat,
                direction=direction,
                confidence=confidence,
                agreement=agreement,
                margin=margin,
                expected_move_size=expected_move_size,
                expected_time=expected_time,
                optimal_sl_multiplier=optimal_sl_multiplier,
                passes_ml=False,  # Will be set in adjust_for_regime
                confidence_required=0.0,
                margin_required=0.0,
                agreement_required=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Error in ensemble voting: {e}")
            return self._get_default_prediction()
    
    def _adjust_for_regime(self, prediction: MLPrediction, regime: str) -> MLPrediction:
        """
        Adjust ML thresholds based on market regime as per algorithm specification
        
        Args:
            prediction: Base prediction from ensemble
            regime: Current market regime
            
        Returns:
            Adjusted prediction with regime-specific thresholds
        """
        try:
            # Regime adjustment multipliers from algorithm
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
            
            # Get base thresholds from algorithm constants
            base_filters = self.ENTRY_FILTERS['default']
            
            # Calculate adjusted thresholds
            confidence_required = base_filters['ml_conf_min'] * adj['confidence_mult']
            margin_required = base_filters['ml_margin_min'] * adj['margin_mult']
            agreement_required = base_filters['ml_agreement_min'] * adj['agreement_mult']
            
            # Check if prediction passes ML filters
            passes_ml = (
                prediction.confidence >= confidence_required and
                prediction.margin >= margin_required and
                prediction.agreement >= agreement_required
            )
            
            # Create adjusted prediction
            adjusted_prediction = MLPrediction(
                p_up=prediction.p_up,
                p_down=prediction.p_down,
                p_flat=prediction.p_flat,
                direction=prediction.direction,
                confidence=prediction.confidence,
                agreement=prediction.agreement,
                margin=prediction.margin,
                expected_move_size=prediction.expected_move_size,
                expected_time=prediction.expected_time,
                optimal_sl_multiplier=prediction.optimal_sl_multiplier,
                passes_ml=passes_ml,
                confidence_required=confidence_required,
                margin_required=margin_required,
                agreement_required=agreement_required
            )
            
            return adjusted_prediction
            
        except Exception as e:
            self.logger.error(f"Error adjusting for regime: {e}")
            return prediction
    
    def _get_direction_from_probabilities(self, p_up: float, p_down: float, p_flat: float) -> str:
        """Get direction from probabilities"""
        if p_up > p_down and p_up > p_flat:
            return 'long'
        elif p_down > p_up and p_down > p_flat:
            return 'short'
        else:
            return 'flat'
    
    def _get_default_prediction(self, regime: str = 'normal_range') -> MLPrediction:
        """Get default prediction when models fail"""
        return MLPrediction(
            p_up=0.33,
            p_down=0.33,
            p_flat=0.34,
            direction='flat',
            confidence=0.4,
            agreement=0.5,
            margin=0.0,
            expected_move_size='medium',
            expected_time=60.0,
            optimal_sl_multiplier=1.0,
            passes_ml=False,
            confidence_required=0.7,
            margin_required=0.15,
            agreement_required=0.6
        )
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return {
            'models_loaded': list(self.models.keys()),
            'models_count': len(self.models),
            'ensemble_weights': self.ensemble_weights,
            'feature_calculator_ready': self.feature_calculator is not None,
            'models_path': str(self.models_path),
            'has_real_models': self.models_loaded
        }
    
    def retrain_models(self, training_data: pd.DataFrame, target_column: str = 'target'):
        """
        Retrain models with new data (placeholder for future implementation)
        
        Args:
            training_data: Training dataset
            target_column: Target column name
        """
        self.logger.info("Model retraining not implemented yet")
        # TODO: Implement model retraining pipeline
        pass