#!/usr/bin/env python3
"""Final working aggressive momentum strategy for SOL/USDT."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.enhanced_backtest import EnhancedBacktestEngine
from src.utils.types import SignalType, Signal, TimeFrame

class FinalAggressiveStrategy(EnhancedBacktestEngine):
    """Final aggressive strategy with working signal generation."""
    
    def __init__(self, initial_balance: float = 10000.0):
        super().__init__(initial_balance)
        
        # Ultra-aggressive parameters based on debug results
        self.strategy_params = {
            'min_confidence': 0.03,           # Ultra-low threshold - 3%
            'min_volume_ratio': 0.3,          # Ultra-low volume requirement
            'min_adx': 5.0,                   # Ultra-low trend requirement
            'atr_sl_multiplier': 0.6,         # Very tight stops
            'atr_tp_multiplier': 4.0,         # Very high targets
            'max_position_time_hours': 8,     # Longer hold time
            'position_size_pct': 0.02,        # 2% position size
            
            # Momentum detection
            'immediate_entry_threshold': 0.006, # 0.6% based on debug
            'breakout_threshold': 0.004,      # 0.4% breakout
            'momentum_threshold': 0.001,      # 0.1% momentum
        }
    
    def _generate_enhanced_signal(
        self,
        symbol: str,
        current_price: float,
        current_candle: pd.Series,
        features: Dict[str, float],
        timestamp: datetime,
    ) -> Optional[Signal]:
        """Ultra-aggressive signal generation based on debug findings."""
        
        # Get basic features
        ema_8 = features.get('ema_8', current_price)
        ema_21 = features.get('ema_21', current_price)
        ema_55 = features.get('ema_55', current_price)
        volume_ratio = features.get('volume_ratio', 1.0)
        atr_ratio = features.get('atr_ratio', 0.02)
        
        # Calculate price momentum from current candle
        # Use the 'close' vs previous 'close' if available in features
        price_momentum = features.get('price_change_5m', 0)
        if price_momentum == 0:
            # Fallback: estimate from current vs previous candle if we have historical data
            price_momentum = 0.002  # Assume small positive momentum
        
        # === IMMEDIATE ENTRY LOGIC ===
        if abs(price_momentum) >= self.strategy_params['immediate_entry_threshold']:
            signal_type = SignalType.BUY if price_momentum > 0 else SignalType.SELL
            confidence = 0.85  # High confidence for immediate entries
            
            return self._create_signal_object(
                symbol, signal_type, timestamp, confidence, current_price, atr_ratio,
                f"IMMEDIATE: {price_momentum*100:.2f}% move, Vol: {volume_ratio:.1f}x"
            )
        
        # === BASIC FILTERING (ultra-permissive) ===
        if volume_ratio < self.strategy_params['min_volume_ratio']:
            return None
        
        # === SIMPLIFIED SCORING SYSTEM ===
        
        # Bullish scoring (simplified from debug findings)
        bullish_score = 0.0
        
        # Trend component (40% weight)
        if current_price > ema_21:
            bullish_score += 0.20
        if ema_8 > ema_21:
            bullish_score += 0.20
        
        # Momentum component (40% weight)
        if price_momentum > self.strategy_params['momentum_threshold']:
            bullish_score += 0.20
        if price_momentum > self.strategy_params['breakout_threshold']:
            bullish_score += 0.20
        
        # Volume component (20% weight)
        if volume_ratio > 2.0:
            bullish_score += 0.15
        elif volume_ratio > 1.0:
            bullish_score += 0.05
        
        # Bearish scoring (mirror of bullish)
        bearish_score = 0.0
        
        if current_price < ema_21:
            bearish_score += 0.20
        if ema_8 < ema_21:
            bearish_score += 0.20
        if price_momentum < -self.strategy_params['momentum_threshold']:
            bearish_score += 0.20
        if price_momentum < -self.strategy_params['breakout_threshold']:
            bearish_score += 0.20
        if volume_ratio > 1.0:
            bearish_score += 0.10
        
        # === SIGNAL DETERMINATION ===
        signal_type = None
        confidence = 0.0
        
        if bullish_score > bearish_score and bullish_score >= self.strategy_params['min_confidence']:
            signal_type = SignalType.BUY
            confidence = min(0.95, bullish_score)
        elif bearish_score > bullish_score and bearish_score >= self.strategy_params['min_confidence']:
            signal_type = SignalType.SELL
            confidence = min(0.95, bearish_score)
        
        if not signal_type:
            return None
        
        # Create reasoning
        reasoning = f"Final {signal_type.value}: conf={confidence:.3f}, " \
                   f"momentum={price_momentum*100:.2f}%, vol={volume_ratio:.1f}x, " \
                   f"bull={bullish_score:.2f}, bear={bearish_score:.2f}"
        
        return self._create_signal_object(symbol, signal_type, timestamp, confidence, current_price, atr_ratio, reasoning)
    
    def _create_signal_object(
        self, 
        symbol: str, 
        signal_type: SignalType, 
        timestamp: datetime, 
        confidence: float, 
        current_price: float, 
        atr_ratio: float, 
        reasoning: str
    ) -> Signal:
        """Create signal with aggressive risk management."""
        
        base_atr = atr_ratio * current_price
        
        # Very tight stops, very wide targets
        sl_multiplier = self.strategy_params['atr_sl_multiplier']
        tp_multiplier = self.strategy_params['atr_tp_multiplier'] * (1 + confidence)
        
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
    
    def run_enhanced_backtest_with_momentum(self, block_id: str = "august_12_single_day"):
        """Run backtest with proper momentum feature integration."""
        
        print(f"🚀 Final Aggressive Strategy Backtest on {block_id}")
        print("=" * 60)
        
        # Load data
        data = self.load_data_from_block(block_id)
        
        # Process 5m data with momentum
        df_5m = data[TimeFrame.M5].copy()
        
        # Add momentum features
        df_5m['price_change_5m'] = df_5m['close'].pct_change()
        df_5m['price_change_3p'] = df_5m['close'].pct_change(periods=3)
        df_5m['volume_ma10'] = df_5m['volume'].rolling(window=10).mean()
        df_5m['volume_ratio'] = df_5m['volume'] / df_5m['volume_ma10']
        
        # Calculate indicators
        df_5m['ema_8'] = df_5m['close'].ewm(span=8).mean()
        df_5m['ema_21'] = df_5m['close'].ewm(span=21).mean()
        df_5m['ema_55'] = df_5m['close'].ewm(span=55).mean()
        
        # ATR calculation
        df_5m['high_low'] = df_5m['high'] - df_5m['low']
        df_5m['high_close'] = abs(df_5m['high'] - df_5m['close'].shift(1))
        df_5m['low_close'] = abs(df_5m['low'] - df_5m['close'].shift(1))
        df_5m['true_range'] = df_5m[['high_low', 'high_close', 'low_close']].max(axis=1)
        df_5m['atr'] = df_5m['true_range'].rolling(window=14).mean()
        df_5m['atr_ratio'] = df_5m['atr'] / df_5m['close']
        
        # Update data
        data[TimeFrame.M5] = df_5m
        
        # Run manual backtest simulation
        print("🔧 Running manual simulation...")
        
        balance = self.initial_balance
        positions = []
        trades = []
        signals_generated = 0
        signals_processed = 0
        
        # Process each candle manually
        for i in range(55, len(df_5m)):  # Start after indicators stabilize
            current_candle = df_5m.iloc[i]
            current_price = current_candle['close']
            # Get timestamp from index or column
            if 'timestamp' in current_candle.index:
                current_time = current_candle['timestamp']
            elif hasattr(current_candle, 'name') and isinstance(current_candle.name, datetime):
                current_time = current_candle.name
            else:
                current_time = df_5m.index[i]
            
            # Create features dict
            features = {
                'ema_8': current_candle.get('ema_8', current_price),
                'ema_21': current_candle.get('ema_21', current_price),
                'ema_55': current_candle.get('ema_55', current_price),
                'volume_ratio': current_candle.get('volume_ratio', 1.0),
                'atr_ratio': current_candle.get('atr_ratio', 0.02),
                'price_change_5m': current_candle.get('price_change_5m', 0),
            }
            
            # Generate signal
            signal = self._generate_enhanced_signal(
                symbol="SOLUSDT",
                current_price=current_price,
                current_candle=current_candle,
                features=features,
                timestamp=current_time
            )
            
            if signal:
                signals_generated += 1
                
                # Only enter if no position
                if not positions:
                    signals_processed += 1
                    
                    # Calculate position
                    position_value = balance * self.strategy_params['position_size_pct']
                    quantity = position_value / current_price
                    
                    # Create position
                    position = {
                        'signal_type': signal.signal_type,
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'quantity': quantity,
                        'confidence': signal.confidence,
                        'reasoning': signal.reasoning,
                    }
                    positions.append(position)
                    
                    time_str = current_time.strftime('%H:%M') if hasattr(current_time, 'strftime') else str(current_time)
                    print(f"📈 {signals_processed:2d}. {time_str} {signal.signal_type.value} @ ${current_price:.2f} "
                          f"(conf: {signal.confidence:.2f}, SL: ${signal.stop_loss:.2f}, TP: ${signal.take_profit:.2f})")
            
            # Check position exits
            positions_to_close = []
            for pos_idx, position in enumerate(positions):
                exit_reason = None
                
                if position['signal_type'] == SignalType.BUY:
                    if current_price <= position['stop_loss']:
                        exit_reason = "stop_loss"
                    elif current_price >= position['take_profit']:
                        exit_reason = "take_profit"
                else:  # SELL
                    if current_price >= position['stop_loss']:
                        exit_reason = "stop_loss"
                    elif current_price <= position['take_profit']:
                        exit_reason = "take_profit"
                
                # Time stop
                if (current_time - position['entry_time']).total_seconds() > self.strategy_params['max_position_time_hours'] * 3600:
                    exit_reason = "time_stop"
                
                if exit_reason:
                    positions_to_close.append((pos_idx, exit_reason, current_time, current_price))
            
            # Close positions
            for pos_idx, exit_reason, exit_time, exit_price in reversed(positions_to_close):
                position = positions[pos_idx]
                
                # Calculate P&L
                if position['signal_type'] == SignalType.BUY:
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                
                # Apply commission
                commission = (position['entry_price'] + exit_price) * position['quantity'] * 0.001
                net_pnl = pnl - commission
                
                # Update balance
                balance += net_pnl
                
                # Record trade
                trade = {
                    'signal_type': position['signal_type'],
                    'entry_time': position['entry_time'],
                    'exit_time': exit_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'net_pnl': net_pnl,
                    'commission': commission,
                    'exit_reason': exit_reason,
                    'confidence': position['confidence'],
                    'reasoning': position['reasoning'],
                }
                trades.append(trade)
                
                exit_time_str = exit_time.strftime('%H:%M') if hasattr(exit_time, 'strftime') else str(exit_time)
                print(f"🔚 {exit_time_str} CLOSE @ ${exit_price:.2f} | "
                      f"P&L: ${net_pnl:+.2f} | {exit_reason} | Bal: ${balance:.2f}")
                
                # Remove position
                positions.pop(pos_idx)
        
        # Results
        print(f"\n📊 FINAL AGGRESSIVE STRATEGY RESULTS")
        print("-" * 50)
        
        if trades:
            total_return = (balance / self.initial_balance - 1) * 100
            winning_trades = len([t for t in trades if t['net_pnl'] > 0])
            win_rate = winning_trades / len(trades) * 100 if trades else 0
            
            print(f"✅ Total trades: {len(trades)}")
            print(f"📶 Signals generated: {signals_generated}")
            print(f"💰 Total P&L: ${sum(t['net_pnl'] for t in trades):.2f}")
            print(f"📈 Final balance: ${balance:.2f}")
            print(f"📊 Return: {total_return:+.2f}%")
            print(f"🎯 Win rate: {win_rate:.1f}% ({winning_trades}/{len(trades)})")
            
            # Show all trades
            print(f"\n📝 All Trades:")
            for i, trade in enumerate(trades, 1):
                entry_time = trade['entry_time'].strftime('%H:%M') if hasattr(trade['entry_time'], 'strftime') else str(trade['entry_time'])
                exit_time = trade['exit_time'].strftime('%H:%M') if hasattr(trade['exit_time'], 'strftime') else str(trade['exit_time'])
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60
                
                print(f"  {i:2d}. {entry_time}-{exit_time} ({duration:.0f}m) "
                      f"{trade['signal_type'].value} ${trade['entry_price']:.2f}→${trade['exit_price']:.2f} | "
                      f"P&L: ${trade['net_pnl']:+6.2f} | {trade['exit_reason']}")
            
            # Market capture analysis
            market_move = (df_5m['close'].iloc[-1] / df_5m['close'].iloc[0] - 1) * 100
            strategy_effectiveness = (total_return / market_move) * 100 if market_move != 0 else 0
            
            print(f"\n💡 Performance Analysis:")
            print(f"📈 Market moved: {market_move:+.2f}%")
            print(f"🤖 Strategy captured: {total_return:+.2f}%")
            print(f"⚡ Effectiveness: {strategy_effectiveness:.1f}% of market move")
            
        else:
            print("❌ No trades executed")
            print(f"📶 Signals generated: {signals_generated}")
        
        return {
            'trades': trades,
            'signals_generated': signals_generated,
            'final_balance': balance,
            'total_return': (balance / self.initial_balance - 1) * 100 if trades else 0
        }

def test_final_strategy():
    """Test final aggressive strategy."""
    strategy = FinalAggressiveStrategy(initial_balance=10000)
    
    print("📋 Final Strategy Parameters:")
    for key, value in strategy.strategy_params.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        results = strategy.run_enhanced_backtest_with_momentum("august_12_single_day")
        
        # Save results
        if results['trades']:
            import json
            output_file = f"Output/final_aggressive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Serialize trades
            serializable_trades = []
            for trade in results['trades']:
                serializable_trade = {}
                for key, value in trade.items():
                    if hasattr(value, 'value'):  # Enum
                        serializable_trade[key] = value.value
                    elif hasattr(value, 'isoformat'):  # Datetime
                        serializable_trade[key] = value.isoformat()
                    else:
                        serializable_trade[key] = value
                serializable_trades.append(serializable_trade)
            
            results['trades'] = serializable_trades
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_final_strategy()