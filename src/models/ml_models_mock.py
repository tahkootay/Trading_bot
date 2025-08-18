"""Mock ML models for testing without XGBoost dependency."""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from ..utils.types import Signal, SignalType
from ..utils.logger import TradingLogger


class MLModelPredictor:
    """Mock ML predictor that simulates ML predictions."""
    
    def __init__(self, models_path: str = "models", config: Dict = None):
        self.models_path = Path(models_path)
        self.config = config or {}
        self.logger = TradingLogger("ml_models")
        
        # Mock model parameters
        self.is_trained = True
        self.model_version = "mock_v1.0"
        
    def predict(
        self,
        features: pd.DataFrame,
        regime: str = "normal_range",
        return_proba: bool = True
    ) -> Dict[str, Any]:
        """Mock prediction based on simple technical rules."""
        if features.empty:
            return self._empty_prediction()
        
        try:
            # Simple rule-based predictions
            last_row = features.iloc[-1] if len(features) > 0 else None
            if last_row is None:
                return self._empty_prediction()
            
            # Mock features (use available columns or defaults)
            rsi = last_row.get('rsi', 50)
            ema_trend = last_row.get('ema_trend', 0)
            volume_ratio = last_row.get('volume_ratio', 1.0)
            adx = last_row.get('adx', 20)
            
            # Simple prediction logic
            bullish_score = 0.5
            
            # RSI influence
            if rsi < 35:
                bullish_score += 0.2  # Oversold
            elif rsi > 65:
                bullish_score -= 0.2  # Overbought
            
            # EMA trend influence  
            if ema_trend > 0:
                bullish_score += 0.15
            elif ema_trend < 0:
                bullish_score -= 0.15
            
            # Volume influence
            if volume_ratio > 1.5:
                bullish_score += 0.1
            
            # ADX influence (trend strength)
            if adx > 25:
                bullish_score += 0.05
            
            # Regime influence
            regime_adjustments = {
                'strong_trend': 0.1,
                'trending': 0.05,
                'normal_range': 0.0,
                'low_volatility_range': -0.05,
                'volatile_choppy': -0.1
            }
            bullish_score += regime_adjustments.get(regime, 0.0)
            
            # Clip to [0, 1] range
            bullish_score = max(0.0, min(1.0, bullish_score))
            bearish_score = 1.0 - bullish_score
            
            # Determine signal
            if bullish_score > 0.65:
                signal = 'BUY'
                confidence = bullish_score
            elif bearish_score > 0.65:
                signal = 'SELL'  
                confidence = bearish_score
            else:
                signal = 'HOLD'
                confidence = max(bullish_score, bearish_score)
            
            # Calculate margin (difference between top two probabilities)
            probs = sorted([bullish_score, bearish_score], reverse=True)
            margin = probs[0] - probs[1]
            
            return {
                'signal': signal,
                'confidence': confidence,
                'margin': margin,
                'probabilities': {
                    'BUY': bullish_score,
                    'SELL': bearish_score,
                    'HOLD': 1.0 - max(bullish_score, bearish_score)
                },
                'features_used': list(features.columns),
                'regime': regime,
                'model_version': self.model_version,
                'prediction_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log_error(
                error_type="prediction_failed",
                component="ml_models",
                error_message=str(e)
            )
            return self._empty_prediction()
    
    def _empty_prediction(self) -> Dict[str, Any]:
        """Return empty/neutral prediction."""
        return {
            'signal': 'HOLD',
            'confidence': 0.5,
            'margin': 0.0,
            'probabilities': {'BUY': 0.33, 'SELL': 0.33, 'HOLD': 0.34},
            'features_used': [],
            'regime': 'unknown',
            'model_version': self.model_version,
            'prediction_time': datetime.now().isoformat()
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Mock feature importance."""
        return {
            'rsi': 0.25,
            'ema_trend': 0.20,
            'volume_ratio': 0.15,
            'adx': 0.15,
            'macd_histogram': 0.10,
            'price_momentum': 0.10,
            'volatility': 0.05
        }
    
    def is_model_trained(self) -> bool:
        """Check if model is trained."""
        return self.is_trained
    
    def retrain_model(
        self,
        training_data: pd.DataFrame,
        target_column: str = 'target',
        **kwargs
    ) -> Dict[str, Any]:
        """Mock retraining."""
        self.logger.log_system_event(
            event_type="model_retrain_mock",
            component="ml_models",
            status="completed",
            details={"samples": len(training_data)}
        )
        
        return {
            'status': 'completed',
            'samples_used': len(training_data),
            'accuracy': 0.67,  # Mock accuracy
            'model_version': self.model_version
        }


class MLSignalGenerator:
    """Signal generator using ML predictions."""
    
    def __init__(self, models: List[MLModelPredictor], config: Dict = None):
        self.models = models
        self.config = config or {}
        self.logger = TradingLogger("ml_signal_generator")
        
        # Thresholds from config
        self.min_confidence = self.config.get('min_confidence', 0.65)
        self.min_margin = self.config.get('min_margin', 0.15)
        self.min_agreement = self.config.get('min_agreement', 0.6)
    
    def generate_signal(
        self,
        features: pd.DataFrame,
        regime: str,
        current_price: float
    ) -> Optional[Signal]:
        """Generate trading signal from ML models."""
        try:
            if not self.models:
                return None
            
            # Get predictions from all models
            predictions = []
            for model in self.models:
                pred = model.predict(features, regime)
                predictions.append(pred)
            
            if not predictions:
                return None
            
            # Ensemble logic
            ensemble_result = self._ensemble_predictions(predictions)
            
            # Check if signal meets criteria
            if not self._meets_signal_criteria(ensemble_result, regime):
                return None
            
            # Create signal
            signal_type = SignalType.BUY if ensemble_result['signal'] == 'BUY' else SignalType.SELL
            if ensemble_result['signal'] == 'HOLD':
                return None
            
            return Signal(
                timestamp=datetime.now(),
                symbol='SOLUSDT',
                signal_type=signal_type,
                confidence=ensemble_result['confidence'],
                entry_price=current_price,
                stop_loss=self._calculate_stop_loss(current_price, signal_type),
                take_profits=self._calculate_take_profits(current_price, signal_type),
                features=dict(ensemble_result),
                setup_type='ml_ensemble',
                timeframe='5m',
                regime=regime
            )
            
        except Exception as e:
            self.logger.log_error(
                error_type="signal_generation_failed",
                component="ml_signal_generator",
                error_message=str(e)
            )
            return None
    
    def _ensemble_predictions(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Combine predictions from multiple models."""
        if not predictions:
            return {'signal': 'HOLD', 'confidence': 0.5, 'margin': 0.0}
        
        # Average probabilities
        buy_probs = [p['probabilities']['BUY'] for p in predictions]
        sell_probs = [p['probabilities']['SELL'] for p in predictions]
        
        avg_buy = np.mean(buy_probs)
        avg_sell = np.mean(sell_probs)
        avg_hold = 1.0 - max(avg_buy, avg_sell)
        
        # Determine signal
        if avg_buy > avg_sell and avg_buy > avg_hold:
            signal = 'BUY'
            confidence = avg_buy
        elif avg_sell > avg_buy and avg_sell > avg_hold:
            signal = 'SELL'
            confidence = avg_sell
        else:
            signal = 'HOLD'
            confidence = max(avg_buy, avg_sell, avg_hold)
        
        # Calculate agreement (how many models agree)
        main_signals = [p['signal'] for p in predictions]
        agreement = main_signals.count(signal) / len(main_signals)
        
        # Calculate margin
        probs = sorted([avg_buy, avg_sell, avg_hold], reverse=True)
        margin = probs[0] - probs[1]
        
        return {
            'signal': signal,
            'confidence': confidence,
            'margin': margin,
            'agreement': agreement,
            'probabilities': {
                'BUY': avg_buy,
                'SELL': avg_sell,
                'HOLD': avg_hold
            }
        }
    
    def _meets_signal_criteria(self, ensemble_result: Dict, regime: str) -> bool:
        """Check if signal meets minimum criteria."""
        # Basic thresholds
        if ensemble_result['confidence'] < self.min_confidence:
            return False
        
        if ensemble_result['margin'] < self.min_margin:
            return False
        
        if ensemble_result['agreement'] < self.min_agreement:
            return False
        
        # Regime-specific adjustments
        regime_multipliers = {
            'strong_trend': 0.85,      # Lower thresholds in strong trend
            'trending': 0.90,
            'normal_range': 1.0,
            'low_volatility_range': 1.1,  # Higher thresholds in low vol
            'volatile_choppy': 1.15        # Highest thresholds in choppy markets
        }
        
        multiplier = regime_multipliers.get(regime, 1.0)
        adjusted_min_confidence = self.min_confidence * multiplier
        
        return ensemble_result['confidence'] >= adjusted_min_confidence
    
    def _calculate_stop_loss(self, entry_price: float, signal_type: SignalType, atr: float = None) -> float:
        """Calculate stop loss price."""
        # Use ATR or fallback to percentage
        stop_distance = atr * 1.5 if atr else entry_price * 0.015  # 1.5%
        
        if signal_type == SignalType.BUY:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def _calculate_take_profits(self, entry_price: float, signal_type: SignalType, atr: float = None) -> List[float]:
        """Calculate take profit levels."""
        # Multiple TP levels
        tp_distances = [1.0, 1.5, 2.5] if atr else [0.01, 0.015, 0.025]  # ATR multiples or percentages
        
        take_profits = []
        for i, distance in enumerate(tp_distances):
            if atr:
                tp_distance = atr * distance
            else:
                tp_distance = entry_price * distance
            
            if signal_type == SignalType.BUY:
                tp_price = entry_price + tp_distance
            else:
                tp_price = entry_price - tp_distance
            
            take_profits.append(tp_price)
        
        return take_profits