"""Advanced trading signal generation system."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from ..data_collector.market_data_collector import MarketDataCollector
from ..feature_engine.technical_indicators import FeatureEngine, TechnicalIndicatorCalculator
from ..models.ml_models import MLModelPredictor
from ..utils.types import (
    Signal, SignalType, TimeFrame, MarketData, TechnicalIndicators
)
from ..utils.logger import TradingLogger


class TradingSignalGenerator:
    """Generate trading signals using ML models and technical analysis."""
    
    def __init__(
        self,
        market_data_collector: MarketDataCollector,
        ml_predictor: MLModelPredictor,
        config: Dict[str, Any],
    ):
        self.market_data_collector = market_data_collector
        self.ml_predictor = ml_predictor
        self.config = config
        
        self.logger = TradingLogger("signal_generator")
        
        # Feature engines
        self.feature_engine = FeatureEngine()
        self.indicator_calc = TechnicalIndicatorCalculator()
        
        # Signal filtering parameters - optimized for real market conditions
        self.min_ml_confidence = config.get("min_signal_confidence", 0.08)  # Lowered from 0.15
        self.min_volume_ratio = config.get("min_volume_ratio", 0.6)  # Reduced from 1.0
        self.min_adx = config.get("min_adx", 15.0)  # Reduced from 20.0
        
        # Signal tracking
        self.signal_history: List[Signal] = []
        self.last_signals: Dict[str, Signal] = {}
        
        # Performance tracking
        self.signal_performance: Dict[str, Dict] = {}
    
    async def generate_signal(
        self,
        symbol: str,
        primary_timeframe: TimeFrame = TimeFrame.M5,
    ) -> Optional[Signal]:
        """Generate trading signal for given symbol."""
        try:
            # Get market data
            market_data = self.market_data_collector.get_market_data(symbol)
            if not market_data:
                return None
            
            # Check data quality
            if not self._is_data_sufficient(market_data, primary_timeframe):
                return None
            
            # Generate features
            features = self._generate_comprehensive_features(market_data, symbol, primary_timeframe)
            if not features:
                return None
            
            # Get ML prediction
            ml_result = self.ml_predictor.predict(
                features=features,
                symbol=symbol,
                current_price=market_data.ticker.price if market_data.ticker else 0.0,
            )
            
            if not ml_result:
                return None
            
            prediction, confidence, individual_predictions = ml_result
            
            # Apply filters
            signal_type = self._apply_signal_filters(
                market_data=market_data,
                features=features,
                ml_prediction=prediction,
                ml_confidence=confidence,
                symbol=symbol,
                primary_timeframe=primary_timeframe,
            )
            
            if signal_type is None:
                return None
            
            # Calculate signal parameters
            current_price = market_data.ticker.price if market_data.ticker else 0.0
            stop_loss, take_profit = self._calculate_signal_levels(
                market_data=market_data,
                signal_type=signal_type,
                current_price=current_price,
                features=features,
            )
            
            # Create signal
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                timestamp=datetime.now(),
                confidence=confidence,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=self._generate_reasoning(
                    signal_type, features, prediction, confidence, individual_predictions
                ),
                features=features,
            )
            
            # Track signal
            self.signal_history.append(signal)
            self.last_signals[symbol] = signal
            
            # Keep only last 1000 signals
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            self.logger.log_signal(
                symbol=symbol,
                signal_type=signal_type.value,
                confidence=confidence,
                price=current_price,
                reasoning=signal.reasoning,
                features={k: v for k, v in features.items() if isinstance(v, (int, float))},
            )
            
            return signal
        
        except Exception as e:
            self.logger.log_error(
                error_type="signal_generation_failed",
                component="signal_generator",
                error_message=str(e),
                details={"symbol": symbol, "timeframe": primary_timeframe.value},
            )
            return None
    
    def _is_data_sufficient(self, market_data: MarketData, timeframe: TimeFrame) -> bool:
        """Check if market data is sufficient for signal generation."""
        if timeframe not in market_data.candles:
            return False
        
        df = market_data.candles[timeframe]
        
        # Need minimum candles for indicators
        if len(df) < 55:
            return False
        
        # Check for recent data
        last_candle_time = df.index[-1]
        time_diff = datetime.now() - last_candle_time
        
        # Allow max 2 intervals of staleness
        max_staleness = {
            TimeFrame.M1: timedelta(minutes=2),
            TimeFrame.M3: timedelta(minutes=6),
            TimeFrame.M5: timedelta(minutes=10),
            TimeFrame.M15: timedelta(minutes=30),
        }.get(timeframe, timedelta(minutes=10))
        
        if time_diff > max_staleness:
            return False
        
        # Check for valid ticker
        if not market_data.ticker:
            return False
        
        return True
    
    def _generate_comprehensive_features(
        self,
        market_data: MarketData,
        symbol: str,
        primary_timeframe: TimeFrame,
    ) -> Optional[Dict[str, float]]:
        """Generate comprehensive feature set."""
        try:
            # Convert market data to required format
            candles_dict = {
                symbol: market_data.candles
            }
            
            # Generate ML features
            features = self.feature_engine.generate_features(
                market_data=candles_dict,
                symbol=symbol,
                primary_timeframe=primary_timeframe,
            )
            
            if not features:
                return None
            
            # Add market microstructure features
            features.update(self._add_microstructure_features(market_data))
            
            # Add inter-market features
            features.update(self._add_intermarket_features(market_data))
            
            # Add momentum features
            features.update(self._add_momentum_features(market_data, primary_timeframe))
            
            return features
        
        except Exception as e:
            self.logger.log_error(
                error_type="feature_generation_failed",
                component="signal_generator",
                error_message=str(e),
                details={"symbol": symbol},
            )
            return None
    
    def _add_microstructure_features(self, market_data: MarketData) -> Dict[str, float]:
        """Add market microstructure features."""
        features = {}
        
        if market_data.orderbook:
            orderbook = market_data.orderbook
            
            # Bid-ask spread
            if orderbook.bids and orderbook.asks:
                best_bid = orderbook.bids[0][0]
                best_ask = orderbook.asks[0][0]
                spread = (best_ask - best_bid) / best_bid
                features['bid_ask_spread'] = spread
                
                # Order book imbalance
                bid_volume = sum([level[1] for level in orderbook.bids[:5]])
                ask_volume = sum([level[1] for level in orderbook.asks[:5]])
                if ask_volume > 0:
                    features['orderbook_imbalance'] = bid_volume / ask_volume - 1
                else:
                    features['orderbook_imbalance'] = 0
        
        # Recent trades analysis
        if market_data.recent_trades:
            trades = market_data.recent_trades[-100:]  # Last 100 trades
            
            # Buy/sell pressure
            buy_volume = sum([t.quantity for t in trades if t.side.value == "BUY"])
            sell_volume = sum([t.quantity for t in trades if t.side.value == "SELL"])
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                features['buy_pressure'] = buy_volume / total_volume
                features['trade_imbalance'] = (buy_volume - sell_volume) / total_volume
            
            # Large trade detection
            if trades:
                volumes = [t.quantity for t in trades]
                avg_volume = np.mean(volumes)
                large_trades = [v for v in volumes if v > avg_volume * 3]
                features['large_trade_ratio'] = len(large_trades) / len(trades)
        
        return features
    
    def _add_intermarket_features(self, market_data: MarketData) -> Dict[str, float]:
        """Add inter-market correlation features."""
        features = {}
        
        # Funding rate analysis
        if market_data.funding_rate is not None:
            features['funding_rate'] = market_data.funding_rate
            features['funding_rate_extreme'] = 1 if abs(market_data.funding_rate) > 0.01 else 0
        
        # Open interest
        if market_data.open_interest is not None:
            features['open_interest'] = market_data.open_interest
        
        # Long/short ratio
        if market_data.long_short_ratio is not None:
            features['long_short_ratio'] = market_data.long_short_ratio
            features['ls_ratio_extreme'] = 1 if market_data.long_short_ratio > 3 or market_data.long_short_ratio < 0.33 else 0
        
        return features
    
    def _add_momentum_features(self, market_data: MarketData, timeframe: TimeFrame) -> Dict[str, float]:
        """Add momentum-specific features."""
        features = {}
        
        if timeframe in market_data.candles:
            df = market_data.candles[timeframe]
            
            # Price momentum
            close = df['close']
            
            # Recent price acceleration
            returns_1 = close.pct_change(1).iloc[-1]
            returns_3 = close.pct_change(3).iloc[-1]
            returns_5 = close.pct_change(5).iloc[-1]
            
            if not pd.isna(returns_1) and not pd.isna(returns_3):
                features['momentum_acceleration'] = returns_1 - returns_3
            
            # Volume momentum
            volume = df['volume']
            vol_ratio_1 = volume.iloc[-1] / volume.rolling(5).mean().iloc[-1]
            vol_ratio_2 = volume.rolling(5).mean().iloc[-1] / volume.rolling(20).mean().iloc[-1]
            
            if not pd.isna(vol_ratio_1):
                features['volume_momentum_1'] = vol_ratio_1 - 1
            if not pd.isna(vol_ratio_2):
                features['volume_momentum_2'] = vol_ratio_2 - 1
            
            # Volatility momentum
            returns = close.pct_change()
            vol_1 = returns.rolling(5).std().iloc[-1]
            vol_2 = returns.rolling(20).std().iloc[-1]
            
            if not pd.isna(vol_1) and not pd.isna(vol_2) and vol_2 > 0:
                features['volatility_momentum'] = vol_1 / vol_2 - 1
        
        return features
    
    def _apply_signal_filters(
        self,
        market_data: MarketData,
        features: Dict[str, float],
        ml_prediction: float,
        ml_confidence: float,
        symbol: str,
        primary_timeframe: TimeFrame,
    ) -> Optional[SignalType]:
        """Apply multi-level signal filters."""
        try:
            # Filter 1: ML confidence threshold
            if ml_confidence < self.min_ml_confidence:
                return None
            
            # Filter 2: ML prediction strength
            if abs(ml_prediction) < self.min_ml_confidence:
                return None
            
            # Filter 3: Volume confirmation (relaxed for real market conditions)
            volume_ratio = features.get('volume_ratio', 1.0)
            if volume_ratio < self.min_volume_ratio:
                return None
            
            # Filter 4: Trend strength (ADX)
            adx = features.get('adx', 0.0)
            if adx < self.min_adx:
                return None
            
            # Filter 5: Market hours (if configured)
            if self._is_low_activity_period():
                return None
            
            # Filter 6: Technical alignment
            if not self._check_technical_alignment(features, ml_prediction):
                return None
            
            # Filter 7: Risk management (avoid overtrading)
            if self._should_skip_signal(symbol):
                return None
            
            # Determine signal type
            if ml_prediction > 0:
                return SignalType.BUY
            else:
                return SignalType.SELL
        
        except Exception as e:
            self.logger.log_error(
                error_type="signal_filtering_failed",
                component="signal_generator",
                error_message=str(e),
            )
            return None
    
    def _check_technical_alignment(self, features: Dict[str, float], ml_prediction: float) -> bool:
        """Check if technical indicators align with ML prediction."""
        try:
            if ml_prediction > 0:  # Bullish prediction
                # Check bullish conditions
                conditions = [
                    features.get('price_vs_ema_8', 0) > -0.02,  # Price near EMA8
                    features.get('rsi', 50) > 30,  # Not oversold
                    features.get('macd_bullish', 0) == 1 or features.get('macd_hist', 0) > 0,  # MACD alignment
                    features.get('price_vs_vwap', 0) > -0.01,  # Price near VWAP
                ]
                
                # Need at least 2 out of 4 conditions
                return sum(conditions) >= 2
            
            else:  # Bearish prediction
                # Check bearish conditions
                conditions = [
                    features.get('price_vs_ema_8', 0) < 0.02,  # Price near EMA8
                    features.get('rsi', 50) < 70,  # Not overbought
                    features.get('macd_bullish', 1) == 0 or features.get('macd_hist', 0) < 0,  # MACD alignment
                    features.get('price_vs_vwap', 0) < 0.01,  # Price near VWAP
                ]
                
                # Need at least 2 out of 4 conditions
                return sum(conditions) >= 2
        
        except Exception:
            return True  # Default to allow signal if check fails
    
    def _is_low_activity_period(self) -> bool:
        """Check if current time is low activity period."""
        now = datetime.now()
        hour = now.hour
        
        # Avoid very low activity periods (UTC)
        # 22:00 - 01:00 UTC is typically low activity
        if 22 <= hour or hour <= 1:
            return True
        
        # Weekend trading is typically lower volume
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return True
        
        return False
    
    def _should_skip_signal(self, symbol: str) -> bool:
        """Check if should skip signal due to recent signals."""
        if symbol not in self.last_signals:
            return False
        
        last_signal = self.last_signals[symbol]
        time_since_last = datetime.now() - last_signal.timestamp
        
        # Minimum time between signals (reduced for more opportunities)
        min_interval = timedelta(minutes=15)
        
        return time_since_last < min_interval
    
    def _calculate_signal_levels(
        self,
        market_data: MarketData,
        signal_type: SignalType,
        current_price: float,
        features: Dict[str, float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        try:
            atr = features.get('atr_ratio', 0.02) * current_price
            if atr == 0:
                atr = current_price * 0.015  # Fallback to 1.5%
            
            # Stop loss
            sl_multiplier = self.config.get('stop_loss_atr', 1.2)
            
            if signal_type == SignalType.BUY:
                stop_loss = current_price - (atr * sl_multiplier)
                take_profit = current_price + (atr * 2.0)  # 1:1.67 R:R
            else:
                stop_loss = current_price + (atr * sl_multiplier)
                take_profit = current_price - (atr * 2.0)
            
            return stop_loss, take_profit
        
        except Exception as e:
            self.logger.log_error(
                error_type="level_calculation_failed",
                component="signal_generator",
                error_message=str(e),
            )
            return None, None
    
    def _generate_reasoning(
        self,
        signal_type: SignalType,
        features: Dict[str, float],
        ml_prediction: float,
        ml_confidence: float,
        individual_predictions: Dict[str, float],
    ) -> str:
        """Generate human-readable reasoning for the signal."""
        reasoning_parts = []
        
        # ML prediction
        reasoning_parts.append(
            f"ML Ensemble: {ml_prediction:.3f} (confidence: {ml_confidence:.3f})"
        )
        
        # Individual model agreement
        agreements = [
            f"{model}: {pred:.3f}" 
            for model, pred in individual_predictions.items()
        ]
        reasoning_parts.append(f"Models: {', '.join(agreements)}")
        
        # Technical factors
        tech_factors = []
        
        if features.get('ema_bullish_alignment', 0) == 1:
            tech_factors.append("EMA bullish alignment")
        elif features.get('ema_bearish_alignment', 0) == 1:
            tech_factors.append("EMA bearish alignment")
        
        if features.get('macd_bullish', 0) == 1:
            tech_factors.append("MACD bullish")
        elif features.get('macd_hist', 0) < 0:
            tech_factors.append("MACD bearish")
        
        if features.get('strong_trend', 0) == 1:
            tech_factors.append(f"Strong trend (ADX: {features.get('adx', 0):.1f})")
        
        volume_ratio = features.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            tech_factors.append(f"High volume ({volume_ratio:.1f}x)")
        
        if tech_factors:
            reasoning_parts.append(f"Technical: {', '.join(tech_factors)}")
        
        # Market structure
        structure_factors = []
        
        if features.get('price_vs_vwap', 0) > 0.005:
            structure_factors.append("Above VWAP")
        elif features.get('price_vs_vwap', 0) < -0.005:
            structure_factors.append("Below VWAP")
        
        if features.get('buy_pressure', 0.5) > 0.6:
            structure_factors.append("Strong buy pressure")
        elif features.get('buy_pressure', 0.5) < 0.4:
            structure_factors.append("Strong sell pressure")
        
        if structure_factors:
            reasoning_parts.append(f"Structure: {', '.join(structure_factors)}")
        
        return " | ".join(reasoning_parts)
    
    def get_signal_performance(self, days: int = 7) -> Dict[str, Any]:
        """Get signal performance statistics."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            recent_signals = [
                s for s in self.signal_history
                if s.timestamp >= cutoff_time
            ]
            
            if not recent_signals:
                return {"total_signals": 0, "period_days": days}
            
            # Basic statistics
            buy_signals = [s for s in recent_signals if s.signal_type == SignalType.BUY]
            sell_signals = [s for s in recent_signals if s.signal_type == SignalType.SELL]
            
            # Confidence distribution
            confidences = [s.confidence for s in recent_signals]
            
            performance = {
                "total_signals": len(recent_signals),
                "buy_signals": len(buy_signals),
                "sell_signals": len(sell_signals),
                "avg_confidence": np.mean(confidences),
                "min_confidence": np.min(confidences),
                "max_confidence": np.max(confidences),
                "high_confidence_signals": len([s for s in recent_signals if s.confidence > 0.7]),
                "period_days": days,
            }
            
            # Symbols traded
            symbols = list(set([s.symbol for s in recent_signals]))
            performance["symbols_traded"] = symbols
            performance["signals_per_symbol"] = {
                symbol: len([s for s in recent_signals if s.symbol == symbol])
                for symbol in symbols
            }
            
            return performance
        
        except Exception as e:
            self.logger.log_error(
                error_type="performance_calculation_failed",
                component="signal_generator",
                error_message=str(e),
            )
            return {"error": str(e)}
    
    def get_recent_signals(self, count: int = 10) -> List[Signal]:
        """Get most recent signals."""
        return self.signal_history[-count:] if self.signal_history else []