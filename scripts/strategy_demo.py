#!/usr/bin/env python3
"""Demonstration of modular strategy replacement in the trading bot system."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from abc import ABC, abstractmethod

# Mock imports for demonstration (would use real ones in production)
class SignalType:
    BUY = "BUY"
    SELL = "SELL"

class Signal:
    def __init__(self, symbol, signal_type, timestamp, confidence, price, stop_loss, take_profit, reasoning):
        self.symbol = symbol
        self.signal_type = signal_type
        self.timestamp = timestamp
        self.confidence = confidence
        self.price = price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.reasoning = reasoning

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, params: Dict):
        self.params = params
        self.name = self.__class__.__name__
    
    @abstractmethod
    def generate_signal(self, market_data: Dict, features: Dict[str, float]) -> Optional[Signal]:
        """Generate trading signal based on market data and features."""
        pass
    
    @abstractmethod
    def update_parameters(self, new_params: Dict):
        """Update strategy parameters."""
        pass
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information and current parameters."""
        return {
            "name": self.name,
            "parameters": self.params,
            "description": self.__doc__ or "No description available"
        }

class EMATrendStrategy(BaseStrategy):
    """
    Simple EMA-based trend following strategy.
    Generates signals based on EMA crossovers and trend alignment.
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            "ema_fast": 8,
            "ema_medium": 21, 
            "ema_slow": 55,
            "min_confidence": 0.6,
            "min_volume_ratio": 1.2,
            "atr_sl_multiplier": 1.5,
            "atr_tp_multiplier": 2.0
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def generate_signal(self, market_data: Dict, features: Dict[str, float]) -> Optional[Signal]:
        """Generate EMA-based signal."""
        
        # Extract required features
        ema_8 = features.get('ema_8', 0)
        ema_21 = features.get('ema_21', 0)
        ema_55 = features.get('ema_55', 0)
        current_price = features.get('current_price', 0)
        volume_ratio = features.get('volume_ratio', 1.0)
        atr = features.get('atr', current_price * 0.02)
        
        # Check volume condition
        if volume_ratio < self.params['min_volume_ratio']:
            return None
        
        # Check trend alignment for BUY
        if ema_8 > ema_21 > ema_55 and current_price > ema_8:
            confidence = min(0.9, self.params['min_confidence'] + 
                           (volume_ratio - 1) * 0.1 + 
                           ((ema_8 - ema_21) / ema_21) * 10)
            
            if confidence >= self.params['min_confidence']:
                return Signal(
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.BUY,
                    timestamp=datetime.now(),
                    confidence=confidence,
                    price=current_price,
                    stop_loss=current_price - atr * self.params['atr_sl_multiplier'],
                    take_profit=current_price + atr * self.params['atr_tp_multiplier'],
                    reasoning=f"EMA Trend: {ema_8:.2f} > {ema_21:.2f} > {ema_55:.2f}, Vol: {volume_ratio:.1f}x"
                )
        
        # Check trend alignment for SELL
        elif ema_8 < ema_21 < ema_55 and current_price < ema_8:
            confidence = min(0.9, self.params['min_confidence'] + 
                           (volume_ratio - 1) * 0.1 + 
                           ((ema_21 - ema_8) / ema_21) * 10)
            
            if confidence >= self.params['min_confidence']:
                return Signal(
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SELL,
                    timestamp=datetime.now(),
                    confidence=confidence,
                    price=current_price,
                    stop_loss=current_price + atr * self.params['atr_sl_multiplier'],
                    take_profit=current_price - atr * self.params['atr_tp_multiplier'],
                    reasoning=f"EMA Trend: {ema_8:.2f} < {ema_21:.2f} < {ema_55:.2f}, Vol: {volume_ratio:.1f}x"
                )
        
        return None
    
    def update_parameters(self, new_params: Dict):
        """Update strategy parameters."""
        self.params.update(new_params)

class RSIStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy.
    Generates signals based on RSI overbought/oversold levels.
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "rsi_extreme_oversold": 20,
            "rsi_extreme_overbought": 80,
            "min_confidence": 0.5,
            "min_volume_ratio": 1.0,
            "atr_sl_multiplier": 2.0,
            "atr_tp_multiplier": 1.5
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def generate_signal(self, market_data: Dict, features: Dict[str, float]) -> Optional[Signal]:
        """Generate RSI-based signal."""
        
        # Extract required features
        rsi = features.get('rsi', 50)
        current_price = features.get('current_price', 0)
        volume_ratio = features.get('volume_ratio', 1.0)
        atr = features.get('atr', current_price * 0.02)
        
        # Check volume condition
        if volume_ratio < self.params['min_volume_ratio']:
            return None
        
        # BUY signal (oversold)
        if rsi <= self.params['rsi_oversold']:
            # Higher confidence for extreme oversold
            if rsi <= self.params['rsi_extreme_oversold']:
                confidence = min(0.9, self.params['min_confidence'] + 0.3)
            else:
                confidence = self.params['min_confidence']
            
            confidence += (volume_ratio - 1) * 0.1  # Volume bonus
            
            if confidence >= self.params['min_confidence']:
                return Signal(
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.BUY,
                    timestamp=datetime.now(),
                    confidence=min(0.95, confidence),
                    price=current_price,
                    stop_loss=current_price - atr * self.params['atr_sl_multiplier'],
                    take_profit=current_price + atr * self.params['atr_tp_multiplier'],
                    reasoning=f"RSI Oversold: {rsi:.1f} ({'Extreme' if rsi <= self.params['rsi_extreme_oversold'] else 'Normal'})"
                )
        
        # SELL signal (overbought)
        elif rsi >= self.params['rsi_overbought']:
            # Higher confidence for extreme overbought
            if rsi >= self.params['rsi_extreme_overbought']:
                confidence = min(0.9, self.params['min_confidence'] + 0.3)
            else:
                confidence = self.params['min_confidence']
            
            confidence += (volume_ratio - 1) * 0.1  # Volume bonus
            
            if confidence >= self.params['min_confidence']:
                return Signal(
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SELL,
                    timestamp=datetime.now(),
                    confidence=min(0.95, confidence),
                    price=current_price,
                    stop_loss=current_price + atr * self.params['atr_sl_multiplier'],
                    take_profit=current_price - atr * self.params['atr_tp_multiplier'],
                    reasoning=f"RSI Overbought: {rsi:.1f} ({'Extreme' if rsi >= self.params['rsi_extreme_overbought'] else 'Normal'})"
                )
        
        return None
    
    def update_parameters(self, new_params: Dict):
        """Update strategy parameters."""
        self.params.update(new_params)

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands breakout strategy.
    Generates signals based on price breakouts from Bollinger Bands.
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "breakout_threshold": 0.001,  # 0.1% beyond band
            "min_confidence": 0.6,
            "min_volume_ratio": 1.5,
            "atr_sl_multiplier": 1.0,
            "atr_tp_multiplier": 2.5
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def generate_signal(self, market_data: Dict, features: Dict[str, float]) -> Optional[Signal]:
        """Generate Bollinger Bands breakout signal."""
        
        # Extract required features
        current_price = features.get('current_price', 0)
        bb_upper = features.get('bb_upper', current_price)
        bb_lower = features.get('bb_lower', current_price)
        bb_middle = features.get('bb_middle', current_price)
        volume_ratio = features.get('volume_ratio', 1.0)
        atr = features.get('atr', current_price * 0.02)
        
        # Check volume condition (breakouts need volume)
        if volume_ratio < self.params['min_volume_ratio']:
            return None
        
        # Calculate band distances
        upper_distance = (current_price - bb_upper) / bb_upper
        lower_distance = (bb_lower - current_price) / bb_lower
        
        # BUY signal (breakout above upper band)
        if upper_distance > self.params['breakout_threshold']:
            confidence = self.params['min_confidence'] + min(0.3, upper_distance * 100)
            confidence += (volume_ratio - 1.5) * 0.1  # Volume bonus
            
            if confidence >= self.params['min_confidence']:
                return Signal(
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.BUY,
                    timestamp=datetime.now(),
                    confidence=min(0.95, confidence),
                    price=current_price,
                    stop_loss=max(bb_middle, current_price - atr * self.params['atr_sl_multiplier']),
                    take_profit=current_price + atr * self.params['atr_tp_multiplier'],
                    reasoning=f"BB Breakout: {upper_distance*100:.2f}% above upper band, Vol: {volume_ratio:.1f}x"
                )
        
        # SELL signal (breakdown below lower band)
        elif lower_distance > self.params['breakout_threshold']:
            confidence = self.params['min_confidence'] + min(0.3, lower_distance * 100)
            confidence += (volume_ratio - 1.5) * 0.1  # Volume bonus
            
            if confidence >= self.params['min_confidence']:
                return Signal(
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SELL,
                    timestamp=datetime.now(),
                    confidence=min(0.95, confidence),
                    price=current_price,
                    stop_loss=min(bb_middle, current_price + atr * self.params['atr_sl_multiplier']),
                    take_profit=current_price - atr * self.params['atr_tp_multiplier'],
                    reasoning=f"BB Breakdown: {lower_distance*100:.2f}% below lower band, Vol: {volume_ratio:.1f}x"
                )
        
        return None
    
    def update_parameters(self, new_params: Dict):
        """Update strategy parameters."""
        self.params.update(new_params)

class StrategyManager:
    """
    Manager class that demonstrates how easy it is to switch between strategies.
    This shows the power of the modular design.
    """
    
    def __init__(self):
        self.strategies = {
            "ema_trend": EMATrendStrategy(),
            "rsi_reversion": RSIStrategy(),
            "bollinger_breakout": BollingerBandsStrategy()
        }
        self.current_strategy = "ema_trend"
    
    def switch_strategy(self, strategy_name: str, params: Dict = None):
        """Switch to a different strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        self.current_strategy = strategy_name
        
        if params:
            self.strategies[strategy_name].update_parameters(params)
        
        print(f"✅ Switched to strategy: {strategy_name}")
        return self.strategies[strategy_name]
    
    def get_current_strategy(self) -> BaseStrategy:
        """Get the currently active strategy."""
        return self.strategies[self.current_strategy]
    
    def generate_signal(self, market_data: Dict, features: Dict[str, float]) -> Optional[Signal]:
        """Generate signal using current strategy."""
        return self.get_current_strategy().generate_signal(market_data, features)
    
    def list_strategies(self):
        """List all available strategies."""
        print("📋 Available Strategies:")
        for name, strategy in self.strategies.items():
            info = strategy.get_strategy_info()
            status = "🟢 ACTIVE" if name == self.current_strategy else "⚪ INACTIVE"
            print(f"  {status} {name}: {info['description']}")
    
    def optimize_strategy(self, strategy_name: str, optimization_params: Dict):
        """Optimize strategy parameters (placeholder for ML optimization)."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # This would typically run backtests with different parameters
        # and select the best performing combination
        print(f"🔧 Optimizing {strategy_name} with parameters: {optimization_params}")
        
        # For demo, just update with provided params
        self.strategies[strategy_name].update_parameters(optimization_params)
        print(f"✅ Strategy {strategy_name} optimized!")

def demonstrate_strategy_switching():
    """Demonstrate how easy it is to switch strategies."""
    
    print("🚀 Trading Bot Strategy Modularity Demonstration")
    print("=" * 60)
    
    # Create strategy manager
    manager = StrategyManager()
    
    # Show available strategies
    manager.list_strategies()
    print()
    
    # Sample market data and features
    market_data = {
        "symbol": "SOLUSDT",
        "timestamp": datetime.now()
    }
    
    features = {
        "current_price": 195.50,
        "ema_8": 195.20,
        "ema_21": 194.80,
        "ema_55": 193.50,
        "rsi": 45.0,
        "volume_ratio": 1.8,
        "atr": 3.90,
        "bb_upper": 198.50,
        "bb_lower": 192.50,
        "bb_middle": 195.50
    }
    
    # Test each strategy
    strategies_to_test = ["ema_trend", "rsi_reversion", "bollinger_breakout"]
    
    for strategy_name in strategies_to_test:
        print(f"🔄 Testing {strategy_name} strategy:")
        
        # Switch strategy
        strategy = manager.switch_strategy(strategy_name)
        
        # Generate signal
        signal = manager.generate_signal(market_data, features)
        
        if signal:
            print(f"  ✅ Signal: {signal.signal_type}")
            print(f"  📊 Confidence: {signal.confidence:.2f}")
            print(f"  💰 Price: ${signal.price:.2f}")
            print(f"  🛑 Stop Loss: ${signal.stop_loss:.2f}")
            print(f"  🎯 Take Profit: ${signal.take_profit:.2f}")
            print(f"  💭 Reasoning: {signal.reasoning}")
        else:
            print("  ❌ No signal generated")
        
        print()
    
    # Demonstrate parameter optimization
    print("🔧 Demonstrating Parameter Optimization:")
    
    # Optimize EMA strategy for more aggressive trading
    manager.optimize_strategy("ema_trend", {
        "min_confidence": 0.4,  # Lower threshold
        "min_volume_ratio": 1.0,  # Lower volume requirement
        "atr_sl_multiplier": 1.0,  # Tighter stop loss
    })
    
    # Test optimized strategy
    manager.switch_strategy("ema_trend")
    optimized_signal = manager.generate_signal(market_data, features)
    
    if optimized_signal:
        print(f"  ✅ Optimized Signal: {optimized_signal.signal_type}")
        print(f"  📊 Confidence: {optimized_signal.confidence:.2f}")
        print(f"  🛑 New Stop Loss: ${optimized_signal.stop_loss:.2f}")
    
    print("\n🎯 Key Benefits of This Architecture:")
    print("  ✅ Easy to add new strategies")
    print("  ✅ Strategies can be switched in real-time")
    print("  ✅ Parameters can be optimized independently")
    print("  ✅ A/B testing multiple strategies simultaneously")
    print("  ✅ Risk management and execution remain unchanged")
    print("  ✅ Backtesting works with any strategy")

if __name__ == "__main__":
    demonstrate_strategy_switching()