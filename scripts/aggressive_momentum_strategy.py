#!/usr/bin/env python3
"""Aggressive momentum strategy optimized for SOL/USDT intraday trading."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.enhanced_backtest import EnhancedBacktestEngine
from src.utils.types import SignalType, Signal

class AggressiveMomentumStrategy(EnhancedBacktestEngine):
    """Aggressive momentum strategy for capturing $2+ moves."""
    
    def __init__(self, initial_balance: float = 10000.0):
        super().__init__(initial_balance)
        
        # Aggressive parameters optimized for SOL/USDT
        self.strategy_params = {
            'min_confidence': 0.05,           # Very low threshold - 5%
            'min_volume_ratio': 0.4,          # Minimal volume requirement
            'min_adx': 8.0,                   # Very low trend requirement
            'atr_sl_multiplier': 0.8,         # Tighter stops
            'atr_tp_multiplier': 3.0,         # Higher targets
            'max_position_time_hours': 6,     # Longer hold time
            'position_size_pct': 0.03,        # 3% position size
            
            # Momentum parameters
            'breakout_threshold': 0.008,      # 0.8% breakout detection
            'strong_breakout_threshold': 0.015, # 1.5% strong breakout
            'momentum_lookback': 3,           # 3-period momentum
            'volume_spike_threshold': 2.0,    # 2x volume spike
            
            # Adaptive parameters
            'volatility_boost': True,         # Boost signals in volatile periods
            'trend_follow_boost': True,       # Boost trending signals
            'immediate_entry_threshold': 0.02, # 2% immediate entry
        }
    
    def _calculate_momentum_features(self, df: pd.DataFrame, current_idx: int) -> Dict[str, float]:
        """Calculate momentum-specific features."""
        if current_idx < self.strategy_params['momentum_lookback']:
            return {}
        
        lookback = self.strategy_params['momentum_lookback']
        recent_data = df.iloc[current_idx-lookback:current_idx+1]
        
        # Price momentum
        price_momentum_1 = (df.iloc[current_idx]['close'] / df.iloc[current_idx-1]['close'] - 1)
        price_momentum_3 = (df.iloc[current_idx]['close'] / df.iloc[current_idx-lookback]['close'] - 1)
        
        # Volume momentum
        avg_volume = recent_data['volume'].mean()
        current_volume = df.iloc[current_idx]['volume']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volatility momentum
        price_changes = recent_data['close'].pct_change().dropna()
        volatility = price_changes.std() if len(price_changes) > 1 else 0.01
        
        # Trend momentum
        closes = recent_data['close'].values
        if len(closes) > 1:
            trend_slope = np.polyfit(range(len(closes)), closes, 1)[0] / closes[-1]
        else:
            trend_slope = 0
        
        return {
            'price_momentum_1': price_momentum_1,
            'price_momentum_3': price_momentum_3,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'trend_slope': trend_slope,
            'volume_spike': volume_ratio > self.strategy_params['volume_spike_threshold']
        }
    
    def _generate_enhanced_signal(
        self,
        symbol: str,
        current_price: float,
        current_candle: pd.Series,
        features: Dict[str, float],
        timestamp: datetime,
    ) -> Optional[Signal]:
        """Aggressive momentum signal generation."""
        
        # Get basic features
        ema_8 = features.get('ema_8', current_price)
        ema_21 = features.get('ema_21', current_price)
        ema_55 = features.get('ema_55', current_price)
        rsi = features.get('rsi', 50)
        adx = features.get('adx', 0)
        atr_ratio = features.get('atr_ratio', 0.02)
        
        # Get momentum features
        momentum_features = getattr(self, '_current_momentum_features', {})
        price_momentum_1 = momentum_features.get('price_momentum_1', 0)
        price_momentum_3 = momentum_features.get('price_momentum_3', 0)
        volume_ratio = momentum_features.get('volume_ratio', 1.0)
        volatility = momentum_features.get('volatility', 0.01)
        trend_slope = momentum_features.get('trend_slope', 0)
        volume_spike = momentum_features.get('volume_spike', False)
        
        # IMMEDIATE ENTRY CONDITIONS (for strong moves)
        immediate_entry = abs(price_momentum_1) >= self.strategy_params['immediate_entry_threshold']
        if immediate_entry:
            signal_type = SignalType.BUY if price_momentum_1 > 0 else SignalType.SELL
            confidence = 0.9  # High confidence for immediate entries
            
            return self._create_signal(
                symbol, signal_type, timestamp, confidence, current_price, atr_ratio,
                f"IMMEDIATE ENTRY: {price_momentum_1*100:.1f}% move, Vol: {volume_ratio:.1f}x"
            )
        
        # BASIC FILTERING (very permissive)
        if adx < self.strategy_params['min_adx'] and not volume_spike:
            return None
            
        if volume_ratio < self.strategy_params['min_volume_ratio'] and not immediate_entry:
            return None
        
        # SIGNAL SCORING SYSTEM
        
        # === BULLISH SIGNAL ===
        bullish_score = 0.0
        bullish_reasons = []
        
        # Trend alignment (30% weight)
        if current_price > ema_21:
            bullish_score += 0.15
            bullish_reasons.append("price>EMA21")
        
        if ema_8 > ema_21:
            bullish_score += 0.15
            bullish_reasons.append("EMA8>EMA21")
        
        # Momentum conditions (40% weight)
        if price_momentum_1 > 0.002:  # >0.2% momentum
            bullish_score += 0.10
            bullish_reasons.append("pos_momentum")
        
        if price_momentum_3 > 0.005:  # >0.5% 3-period momentum
            bullish_score += 0.10
            bullish_reasons.append("strong_momentum")
        
        if trend_slope > 0:
            bullish_score += 0.10
            bullish_reasons.append("uptrend")
        
        if price_momentum_1 > self.strategy_params['breakout_threshold']:
            bullish_score += 0.10
            bullish_reasons.append("breakout")
        
        # Volume conditions (20% weight)
        if volume_spike:
            bullish_score += 0.15
            bullish_reasons.append("volume_spike")
        elif volume_ratio > 1.0:
            bullish_score += 0.05
            bullish_reasons.append("good_volume")
        
        # RSI conditions (10% weight)
        if 30 < rsi < 80:
            bullish_score += 0.05
            bullish_reasons.append("rsi_ok")
        
        if rsi < 40:  # Oversold bounce
            bullish_score += 0.05
            bullish_reasons.append("oversold_bounce")
        
        # === BEARISH SIGNAL ===
        bearish_score = 0.0
        bearish_reasons = []
        
        # Trend alignment (30% weight)
        if current_price < ema_21:
            bearish_score += 0.15
            bearish_reasons.append("price<EMA21")
        
        if ema_8 < ema_21:
            bearish_score += 0.15
            bearish_reasons.append("EMA8<EMA21")
        
        # Momentum conditions (40% weight)
        if price_momentum_1 < -0.002:  # <-0.2% momentum
            bearish_score += 0.10
            bearish_reasons.append("neg_momentum")
        
        if price_momentum_3 < -0.005:  # <-0.5% 3-period momentum
            bearish_score += 0.10
            bearish_reasons.append("strong_neg_momentum")
        
        if trend_slope < 0:
            bearish_score += 0.10
            bearish_reasons.append("downtrend")
        
        if price_momentum_1 < -self.strategy_params['breakout_threshold']:
            bearish_score += 0.10
            bearish_reasons.append("breakdown")
        
        # Volume conditions (20% weight)
        if volume_spike:
            bearish_score += 0.15
            bearish_reasons.append("volume_spike")
        elif volume_ratio > 1.0:
            bearish_score += 0.05
            bearish_reasons.append("good_volume")
        
        # RSI conditions (10% weight)
        if 20 < rsi < 70:
            bearish_score += 0.05
            bearish_reasons.append("rsi_ok")
        
        if rsi > 60:  # Overbought reversal
            bearish_score += 0.05
            bearish_reasons.append("overbought_reversal")
        
        # ADAPTIVE BOOSTING
        volatility_boost = 0.0
        if self.strategy_params['volatility_boost'] and volatility > 0.015:
            volatility_boost = 0.1
        
        # SIGNAL DECISION
        signal_type = None
        confidence = 0.0
        reasons = []
        
        # Apply volatility boost
        bullish_score += volatility_boost
        bearish_score += volatility_boost
        
        # Determine signal
        if bullish_score > bearish_score and bullish_score >= self.strategy_params['min_confidence']:
            signal_type = SignalType.BUY
            confidence = min(0.95, bullish_score)
            reasons = bullish_reasons
        elif bearish_score > bullish_score and bearish_score >= self.strategy_params['min_confidence']:
            signal_type = SignalType.SELL
            confidence = min(0.95, bearish_score)
            reasons = bearish_reasons
        
        if not signal_type:
            return None
        
        # Create signal with detailed reasoning
        reasoning = f"Aggressive {signal_type.value}: conf={confidence:.3f}, " \
                   f"momentum_1={price_momentum_1*100:.2f}%, " \
                   f"vol={volume_ratio:.1f}x, reasons=[{','.join(reasons[:3])}]"
        
        return self._create_signal(symbol, signal_type, timestamp, confidence, current_price, atr_ratio, reasoning)
    
    def _create_signal(
        self, 
        symbol: str, 
        signal_type: SignalType, 
        timestamp: datetime, 
        confidence: float, 
        current_price: float, 
        atr_ratio: float, 
        reasoning: str
    ) -> Signal:
        """Create signal with dynamic stop loss and take profit."""
        
        # Dynamic ATR multiplier based on confidence and momentum
        base_atr = atr_ratio * current_price
        
        # Tighter stops for lower confidence, wider for higher confidence
        sl_multiplier = self.strategy_params['atr_sl_multiplier'] * (0.5 + confidence)
        tp_multiplier = self.strategy_params['atr_tp_multiplier'] * (0.8 + confidence * 0.4)
        
        if signal_type == SignalType.BUY:
            stop_loss = current_price - (base_atr * sl_multiplier)
            take_profit = current_price + (base_atr * tp_multiplier)
        else:
            stop_loss = current_price + (base_atr * sl_multiplier)
            take_profit = current_price - (base_atr * tp_multiplier)
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            timestamp=timestamp,
            confidence=confidence,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            features={}
        )
    
    def run_backtest(
        self,
        symbol: str,
        data: Dict['TimeFrame', pd.DataFrame],
        primary_timeframe = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Override run_backtest to add momentum feature calculation."""
        
        from src.utils.types import TimeFrame
        
        if primary_timeframe is None:
            primary_timeframe = TimeFrame.M5
        
        # Get primary data and add momentum features
        df_primary = data[primary_timeframe].copy()
        
        # Pre-calculate momentum features for all candles
        print("🔧 Pre-calculating momentum features...")
        
        for i in range(len(df_primary)):
            momentum_features = self._calculate_momentum_features(df_primary, i)
            
            # Store in class for access during signal generation
            if i == 0:  # Initialize storage
                self._momentum_features_cache = {}
            
            self._momentum_features_cache[i] = momentum_features
        
        print("✅ Momentum features calculated")
        
        # Store modified data
        data[primary_timeframe] = df_primary
        
        # Run the original backtest
        return super().run_backtest(symbol, data, primary_timeframe, start_date, end_date)
    
    def _process_candle_momentum(self, current_idx: int, df_primary: pd.DataFrame):
        """Set current momentum features for signal generation."""
        if hasattr(self, '_momentum_features_cache') and current_idx in self._momentum_features_cache:
            self._current_momentum_features = self._momentum_features_cache[current_idx]
        else:
            self._current_momentum_features = self._calculate_momentum_features(df_primary, current_idx)

def test_aggressive_strategy():
    """Test the aggressive momentum strategy."""
    print("🚀 Testing Aggressive Momentum Strategy")
    print("=" * 60)
    
    engine = AggressiveMomentumStrategy(initial_balance=10000)
    
    print("📋 Aggressive Strategy Parameters:")
    for key, value in engine.strategy_params.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Load data
        data = engine.load_data_from_block("august_12_single_day")
        
        # Add momentum features to data processing
        from src.utils.types import TimeFrame
        if TimeFrame.M5 in data:
            df = data[TimeFrame.M5].copy()
            
            # Add price change columns
            df['price_change_1'] = df['close'].pct_change()
            df['price_change_3'] = df['close'].pct_change(periods=3)
            
            # Add volume moving average
            df['volume_ma'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            data[TimeFrame.M5] = df
        
        # Override the signal processing to include momentum features
        original_generate = engine._generate_enhanced_signal
        
        def enhanced_generate_with_momentum(*args, **kwargs):
            # Set momentum features before signal generation
            if hasattr(engine, '_current_candle_idx'):
                engine._process_candle_momentum(engine._current_candle_idx, data[TimeFrame.M5])
            return original_generate(*args, **kwargs)
        
        engine._generate_enhanced_signal = enhanced_generate_with_momentum
        
        # Run backtest
        results = engine.run_backtest(symbol="SOLUSDT", data=data)
        
        print("📊 AGGRESSIVE STRATEGY RESULTS")
        print("-" * 50)
        
        if "error" in results:
            print(f"❌ {results['error']}")
        else:
            print(f"✅ Total trades: {results['total_trades']}")
            print(f"💰 Total P&L: ${results['total_pnl']:.2f}")
            print(f"📈 Final balance: ${results['final_balance']:.2f}")
            print(f"📊 Return: {results['total_return_pct']:.2f}%")
            print(f"🎯 Win rate: {results['win_rate']:.1%}")
            print(f"📉 Max drawdown: {results['max_drawdown']:.2%}")
            
            if engine.trades:
                print(f"\n🔍 Trade Analysis:")
                total_profit = sum(t.get('net_pnl', 0) for t in engine.trades if t.get('net_pnl', 0) > 0)
                total_loss = sum(t.get('net_pnl', 0) for t in engine.trades if t.get('net_pnl', 0) < 0)
                
                print(f"  💚 Profitable trades: {sum(1 for t in engine.trades if t.get('net_pnl', 0) > 0)}")
                print(f"  ❌ Losing trades: {sum(1 for t in engine.trades if t.get('net_pnl', 0) < 0)}")
                print(f"  💰 Total profit: ${total_profit:.2f}")
                print(f"  💸 Total loss: ${total_loss:.2f}")
                
                print(f"\n📝 First 5 Trades:")
                for i, trade in enumerate(engine.trades[:5], 1):
                    pnl = trade.get('net_pnl', 0)
                    entry_time = trade['entry_time'].strftime('%H:%M') if 'entry_time' in trade else 'N/A'
                    exit_time = trade['exit_time'].strftime('%H:%M') if 'exit_time' in trade else 'N/A'
                    
                    print(f"  {i}. {entry_time}-{exit_time} {trade['signal_type'].value} @ "
                          f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
                          f"P&L: ${pnl:+.2f} | {trade['exit_reason']}")
        
        # Strategy comparison
        print(f"\n💡 Strategy Evolution Summary:")
        print(f"📊 Original (Conservative): 0 trades on +8.7% day")
        print(f"🔧 Improved V1: 0 trades (still too conservative)")
        print(f"🔧 Improved V2: 0 trades (still blocked by filters)")
        if "error" not in results:
            captured_movement = (results['total_return_pct'] / 8.7) * 100 if results['total_return_pct'] != 0 else 0
            print(f"🚀 Aggressive Momentum: {results['total_trades']} trades, "
                  f"{results['total_return_pct']:.2f}% return ({captured_movement:.1f}% of market move)")
        else:
            print(f"🚀 Aggressive Momentum: Still 0 trades - need to debug signal generation")
            
        # Save results if trades occurred
        if "error" not in results and results.get('total_trades', 0) > 0:
            import json
            
            output_file = f"Output/aggressive_momentum_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Serialize results
            serializable_results = {}
            for key, value in results.items():
                if hasattr(value, 'value'):  # Enum
                    serializable_results[key] = value.value
                else:
                    serializable_results[key] = value
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {output_file}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_aggressive_strategy()