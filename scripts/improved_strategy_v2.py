#!/usr/bin/env python3
"""Improved strategy V2 with adaptive parameters and breakout detection."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.enhanced_backtest import EnhancedBacktestEngine
from src.utils.types import SignalType

class ImprovedStrategyV2(EnhancedBacktestEngine):
    """Enhanced strategy with improved signal detection."""
    
    def __init__(self, initial_balance: float = 10000.0):
        super().__init__(initial_balance)
        
        # More adaptive parameters
        self.strategy_params.update({
            'min_confidence': 0.08,           # Much lower confidence threshold
            'min_volume_ratio': 0.8,          # Lower volume requirement
            'min_adx': 12.0,                  # Lower trend strength requirement
            'breakout_threshold': 0.015,      # 1.5% breakout detection
            'momentum_window': 5,             # 5-period momentum
            'volatility_multiplier': 1.5,    # Volatility-based position sizing
        })
    
    def _generate_enhanced_signal(
        self,
        symbol: str,
        current_price: float,
        current_candle: pd.Series,
        features: Dict[str, float],
        timestamp: datetime,
    ) -> Optional['Signal']:
        """Enhanced signal generation with breakout detection."""
        
        # Import Signal class
        from src.utils.types import Signal
        
        # Get features
        ema_8 = features.get('ema_8', current_price)
        ema_21 = features.get('ema_21', current_price)
        ema_55 = features.get('ema_55', current_price)
        rsi = features.get('rsi', 50)
        adx = features.get('adx', 0)
        volume_ratio = features.get('volume_ratio', 1.0)
        atr_ratio = features.get('atr_ratio', 0.02)
        
        # Check minimum requirements (relaxed)
        if adx < self.strategy_params['min_adx']:
            return None
        
        if volume_ratio < self.strategy_params['min_volume_ratio']:
            return None
        
        # Breakout detection
        price_change = current_candle.get('price_change_5m', 0)
        is_breakout = abs(price_change) > self.strategy_params['breakout_threshold']
        
        # Enhanced confidence calculation
        confidence = 0.0
        signal_type = None
        
        # Bullish conditions with breakout support
        bullish_score = 0.0
        bullish_conditions = [
            ema_8 > ema_21,                    # Trend alignment (weight: 2)
            current_price > ema_8,             # Price above fast EMA (weight: 2)
            rsi > 35 and rsi < 80,             # RSI not extreme (weight: 1)
            volume_ratio > 0.9,                # Reasonable volume (weight: 1)
            price_change > 0,                  # Positive momentum (weight: 1)
        ]
        
        # Weight the conditions
        weights = [2, 2, 1, 1, 1]
        bullish_score = sum(condition * weight for condition, weight in zip(bullish_conditions, weights)) / sum(weights)
        
        # Bearish conditions
        bearish_score = 0.0
        bearish_conditions = [
            ema_8 < ema_21,                    # Trend alignment (weight: 2)
            current_price < ema_8,             # Price below fast EMA (weight: 2)
            rsi > 20 and rsi < 65,             # RSI not extreme (weight: 1)
            volume_ratio > 0.9,                # Reasonable volume (weight: 1)
            price_change < 0,                  # Negative momentum (weight: 1)
        ]
        
        bearish_score = sum(condition * weight for condition, weight in zip(bearish_conditions, weights)) / sum(weights)
        
        # Signal determination with breakout bonus
        breakout_bonus = 0.15 if is_breakout else 0
        
        if bullish_score > 0.4:  # Lower threshold
            signal_type = SignalType.BUY
            confidence = bullish_score + breakout_bonus
            if price_change > self.strategy_params['breakout_threshold']:
                confidence += 0.1  # Breakout momentum bonus
                
        elif bearish_score > 0.4:  # Lower threshold
            signal_type = SignalType.SELL
            confidence = bearish_score + breakout_bonus
            if price_change < -self.strategy_params['breakout_threshold']:
                confidence += 0.1  # Breakout momentum bonus
        
        # Final confidence check
        if not signal_type or confidence < self.strategy_params['min_confidence']:
            return None
        
        # Dynamic stop loss based on volatility
        base_atr = atr_ratio * current_price
        volatility_multiplier = min(2.5, max(1.0, abs(price_change) * 10))  # Scale with price movement
        dynamic_sl_mult = self.strategy_params['atr_sl_multiplier'] * volatility_multiplier
        
        if signal_type == SignalType.BUY:
            stop_loss = current_price - (base_atr * dynamic_sl_mult)
            take_profit = current_price + (base_atr * self.strategy_params['atr_tp_multiplier'])
        else:
            stop_loss = current_price + (base_atr * dynamic_sl_mult)
            take_profit = current_price - (base_atr * self.strategy_params['atr_tp_multiplier'])
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            timestamp=timestamp,
            confidence=min(0.95, confidence),  # Cap confidence
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"Enhanced V2: {signal_type.value} conf {confidence:.3f}, "
                     f"Bullish: {bullish_score:.2f}, Bearish: {bearish_score:.2f}, "
                     f"Breakout: {is_breakout}, Change: {price_change:.3f}",
            features=features,
        )

def test_improved_v2():
    """Test the improved V2 strategy."""
    print("🚀 Testing Improved Strategy V2")
    print("=" * 50)
    
    engine = ImprovedStrategyV2(initial_balance=10000)
    
    print("📋 Strategy V2 Parameters:")
    for key, value in engine.strategy_params.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        data = engine.load_data_from_block("august_12_single_day")
        
        # Add price change calculation to the data
        if 'M5' in data:
            df = data['M5']
            df['price_change_5m'] = df['close'].pct_change()
            data['M5'] = df
        
        results = engine.run_backtest(symbol="SOLUSDT", data=data)
        
        print("📊 STRATEGY V2 RESULTS")
        print("-" * 40)
        
        if "error" in results:
            print(f"❌ {results['error']}")
        else:
            print(f"✅ Total trades: {results['total_trades']}")
            print(f"💰 Total P&L: ${results['total_pnl']:.2f}")
            print(f"📈 Final balance: ${results['final_balance']:.2f}")
            print(f"📊 Return: {results['total_return_pct']:.2f}%")
            print(f"🎯 Win rate: {results['win_rate']:.1%}")
            
            # Show trade details
            if engine.trades:
                print(f"\n🔍 First 5 Trades:")
                for i, trade in enumerate(engine.trades[:5], 1):
                    pnl = trade.get('net_pnl', 0)
                    print(f"  {i}. {trade['signal_type'].value} @ ${trade['entry_price']:.2f} → "
                          f"${trade['exit_price']:.2f} | P&L: ${pnl:.2f} | "
                          f"Reason: {trade['exit_reason']}")
        
        # Summary comparison
        print(f"\n💡 Strategy Comparison Summary:")
        print(f"📊 Original Strategy: 0 trades on +8.7% day")
        if "error" not in results:
            print(f"🚀 Improved V2: {results['total_trades']} trades, {results['total_return_pct']:.2f}% return")
        else:
            print(f"🚀 Improved V2: Still 0 trades - need further optimization")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_v2()