#!/usr/bin/env python3
"""Improved aggressive strategy based on performance analysis."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.enhanced_backtest import EnhancedBacktestEngine
from src.utils.types import SignalType, Signal, TimeFrame

class ImprovedAggressiveStrategy(EnhancedBacktestEngine):
    """Improved aggressive strategy with enhanced filters and risk management."""
    
    def __init__(self, initial_balance: float = 10000.0):
        super().__init__(initial_balance)
        
        # Improved parameters based on analysis
        self.strategy_params = {
            # Selectivity improvements
            'min_confidence': 0.06,           # Raised from 3% to 6% for better win rate
            'min_volume_ratio': 0.8,          # More conservative volume filter
            'min_adx': 8.0,                   # Slightly higher trend requirement
            
            # Risk management improvements  
            'atr_sl_multiplier': 1.1,         # Wider stops: 0.6 → 1.1
            'atr_tp_multiplier': 3.0,         # More realistic targets: 4.0 → 3.0
            'max_position_time_hours': 6,     # Shorter max hold time
            
            # Adaptive position sizing
            'high_confidence_threshold': 0.70,  # >70% confidence
            'medium_confidence_threshold': 0.50, # 50-70% confidence
            'high_confidence_size': 0.03,       # 3% for high confidence
            'medium_confidence_size': 0.02,     # 2% for medium confidence  
            'low_confidence_size': 0.01,        # 1% for low confidence
            
            # Confirmation filters
            'require_confirmation': True,        # Wait for confirmation
            'confirmation_candles': 2,          # 2 candles in direction
            'volume_confirmation': 1.3,         # Volume >1.3x average
            'momentum_confirmation': 0.003,     # Price change >0.3%
            
            # Volatility adaptation
            'volatility_adaptation': True,
            'high_volatility_threshold': 0.015, # 1.5% volatility
            'volatility_sl_multiplier': 1.4,    # Extra stop buffer in high vol
        }
        
        # Track performance metrics
        self.performance_metrics = {
            'signals_generated': 0,
            'signals_filtered': 0,
            'confirmation_failures': 0,
            'volatility_adjustments': 0,
            'adaptive_sizing_used': 0
        }
    
    def _calculate_enhanced_features(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced technical features."""
        
        # Basic EMAs
        df_5m['ema_8'] = df_5m['close'].ewm(span=8).mean()
        df_5m['ema_20'] = df_5m['close'].ewm(span=20).mean()
        df_5m['ema_50'] = df_5m['close'].ewm(span=50).mean()
        
        # Enhanced RSI with smoothing
        delta = df_5m['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_5m['rsi'] = 100 - (100 / (1 + rs))
        df_5m['rsi_smooth'] = df_5m['rsi'].rolling(3).mean()  # Smoothed RSI
        
        # Enhanced MACD
        ema_12 = df_5m['close'].ewm(span=12).mean()
        ema_26 = df_5m['close'].ewm(span=26).mean()
        df_5m['macd'] = ema_12 - ema_26
        df_5m['macd_signal'] = df_5m['macd'].ewm(span=9).mean()
        df_5m['macd_histogram'] = df_5m['macd'] - df_5m['macd_signal']
        
        # Enhanced volume analysis
        df_5m['volume_sma'] = df_5m['volume'].rolling(20).mean()
        df_5m['volume_ratio'] = df_5m['volume'] / df_5m['volume_sma']
        df_5m['volume_trend'] = df_5m['volume_ratio'].rolling(5).mean()
        
        # Volatility measures
        df_5m['price_change'] = df_5m['close'].pct_change()
        df_5m['volatility'] = df_5m['price_change'].rolling(20).std()
        df_5m['atr'] = self._calculate_atr(df_5m)
        df_5m['atr_pct'] = df_5m['atr'] / df_5m['close']
        
        # Momentum indicators
        df_5m['momentum_5'] = df_5m['close'].pct_change(5)  # 5-period momentum
        df_5m['momentum_3'] = df_5m['close'].pct_change(3)  # 3-period momentum
        
        # Trend strength
        df_5m['trend_strength'] = abs(df_5m['ema_8'] - df_5m['ema_20']) / df_5m['close']
        
        # Support/Resistance levels (simplified)
        df_5m['resistance'] = df_5m['high'].rolling(20).max()
        df_5m['support'] = df_5m['low'].rolling(20).min()
        df_5m['price_position'] = (df_5m['close'] - df_5m['support']) / (df_5m['resistance'] - df_5m['support'])
        
        return df_5m
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(period).mean()
    
    def _check_confirmation_filters(self, df_5m: pd.DataFrame, current_idx: int, 
                                   signal_type: str) -> Dict[str, bool]:
        """Check confirmation filters for signal validation."""
        
        if not self.strategy_params['require_confirmation']:
            return {'passed': True, 'reason': 'confirmation_disabled'}
        
        if current_idx < self.strategy_params['confirmation_candles']:
            return {'passed': False, 'reason': 'insufficient_history'}
        
        confirmations = {}
        
        # Check consecutive candles in signal direction
        direction_confirmations = 0
        for i in range(1, self.strategy_params['confirmation_candles'] + 1):
            prev_idx = current_idx - i
            current_close = df_5m.iloc[current_idx]['close']
            prev_close = df_5m.iloc[prev_idx]['close']
            
            if signal_type == 'BUY' and current_close > prev_close:
                direction_confirmations += 1
            elif signal_type == 'SELL' and current_close < prev_close:
                direction_confirmations += 1
        
        confirmations['direction'] = direction_confirmations >= self.strategy_params['confirmation_candles']
        
        # Volume confirmation
        current_volume_ratio = df_5m.iloc[current_idx]['volume_ratio']
        confirmations['volume'] = current_volume_ratio >= self.strategy_params['volume_confirmation']
        
        # Momentum confirmation
        current_momentum = df_5m.iloc[current_idx]['momentum_3']
        if signal_type == 'BUY':
            confirmations['momentum'] = current_momentum >= self.strategy_params['momentum_confirmation']
        else:
            confirmations['momentum'] = current_momentum <= -self.strategy_params['momentum_confirmation']
        
        # Overall confirmation
        passed = all(confirmations.values())
        
        return {
            'passed': passed,
            'details': confirmations,
            'reason': 'confirmed' if passed else 'failed_confirmation'
        }
    
    def _calculate_adaptive_position_size(self, confidence: float, balance: float) -> float:
        """Calculate adaptive position size based on confidence."""
        
        if confidence >= self.strategy_params['high_confidence_threshold']:
            size_pct = self.strategy_params['high_confidence_size']
            self.performance_metrics['adaptive_sizing_used'] += 1
        elif confidence >= self.strategy_params['medium_confidence_threshold']:
            size_pct = self.strategy_params['medium_confidence_size']
            self.performance_metrics['adaptive_sizing_used'] += 1
        else:
            size_pct = self.strategy_params['low_confidence_size']
        
        return balance * size_pct
    
    def _calculate_adaptive_stop_loss(self, entry_price: float, atr_value: float, 
                                    signal_type: str, volatility: float) -> float:
        """Calculate adaptive stop loss based on volatility."""
        
        # Base stop loss multiplier
        sl_multiplier = self.strategy_params['atr_sl_multiplier']
        
        # Adjust for high volatility
        if (self.strategy_params['volatility_adaptation'] and 
            volatility > self.strategy_params['high_volatility_threshold']):
            sl_multiplier *= self.strategy_params['volatility_sl_multiplier']
            self.performance_metrics['volatility_adjustments'] += 1
        
        # Calculate stop loss
        if signal_type == 'BUY':
            stop_loss = entry_price - (atr_value * sl_multiplier)
        else:
            stop_loss = entry_price + (atr_value * sl_multiplier)
        
        return stop_loss
    
    def _generate_enhanced_signal(self, symbol: str, current_price: float,
                                current_candle: pd.Series, features: Dict[str, float],
                                timestamp: datetime) -> Optional[Signal]:
        """Enhanced signal generation with improved filters."""
        
        # Get current market data
        current_idx = features.get('current_index', 0)
        df_5m = features.get('dataframe')
        
        if df_5m is None or current_idx < 50:
            return None
        
        self.performance_metrics['signals_generated'] += 1
        
        # Extract enhanced features
        ema_8 = features.get('ema_8', current_price)
        ema_20 = features.get('ema_20', current_price)
        ema_50 = features.get('ema_50', current_price)
        rsi = features.get('rsi_smooth', 50)
        macd = features.get('macd', 0)
        macd_signal = features.get('macd_signal', 0)
        macd_histogram = features.get('macd_histogram', 0)
        volume_ratio = features.get('volume_ratio', 1.0)
        volume_trend = features.get('volume_trend', 1.0)
        volatility = features.get('volatility', 0.01)
        momentum_5 = features.get('momentum_5', 0)
        momentum_3 = features.get('momentum_3', 0)
        trend_strength = features.get('trend_strength', 0)
        price_position = features.get('price_position', 0.5)
        
        # Basic filtering
        if volume_ratio < self.strategy_params['min_volume_ratio']:
            self.performance_metrics['signals_filtered'] += 1
            return None
        
        # Enhanced scoring system
        bullish_score = 0.0
        bearish_score = 0.0
        
        # Trend analysis (35% weight)
        if ema_8 > ema_20 > ema_50:
            bullish_score += 0.20
            if trend_strength > 0.005:  # Strong trend
                bullish_score += 0.15
        elif ema_8 < ema_20 < ema_50:
            bearish_score += 0.20
            if trend_strength > 0.005:
                bearish_score += 0.15
        
        # Momentum analysis (25% weight)
        if momentum_5 > 0.005 and momentum_3 > 0.002:
            bullish_score += 0.15
        elif momentum_5 < -0.005 and momentum_3 < -0.002:
            bearish_score += 0.15
        
        if macd > macd_signal and macd_histogram > 0:
            bullish_score += 0.10
        elif macd < macd_signal and macd_histogram < 0:
            bearish_score += 0.10
        
        # RSI analysis (15% weight)
        if 30 < rsi < 70:  # Not overbought/oversold
            if rsi > 55:
                bullish_score += 0.08
            elif rsi < 45:
                bearish_score += 0.08
        elif rsi < 30:  # Oversold
            bullish_score += 0.15
        elif rsi > 70:  # Overbought
            bearish_score += 0.15
        
        # Volume analysis (15% weight)
        if volume_ratio > 1.5 and volume_trend > 1.2:
            if bullish_score > bearish_score:
                bullish_score += 0.15
            else:
                bearish_score += 0.15
        elif volume_ratio > 1.2:
            if bullish_score > bearish_score:
                bullish_score += 0.08
            else:
                bearish_score += 0.08
        
        # Price position analysis (10% weight)
        if price_position > 0.7:  # Near resistance
            bearish_score += 0.05
        elif price_position < 0.3:  # Near support
            bullish_score += 0.05
        
        # Determine signal
        bullish_confidence = min(bullish_score, 1.0)
        bearish_confidence = min(bearish_score, 1.0)
        
        signal_type = None
        confidence = 0
        
        if bullish_confidence > bearish_confidence + 0.05 and bullish_confidence >= self.strategy_params['min_confidence']:
            signal_type = 'BUY'
            confidence = bullish_confidence
        elif bearish_confidence > bullish_confidence + 0.05 and bearish_confidence >= self.strategy_params['min_confidence']:
            signal_type = 'SELL'
            confidence = bearish_confidence
        else:
            self.performance_metrics['signals_filtered'] += 1
            return None
        
        # Check confirmation filters
        confirmation = self._check_confirmation_filters(df_5m, current_idx, signal_type)
        if not confirmation['passed']:
            self.performance_metrics['confirmation_failures'] += 1
            return None
        
        # Calculate adaptive parameters
        atr_value = features.get('atr', current_price * 0.02)
        
        # Adaptive stop loss
        stop_loss = self._calculate_adaptive_stop_loss(
            current_price, atr_value, signal_type, volatility
        )
        
        # Take profit
        if signal_type == 'BUY':
            take_profit = current_price + (atr_value * self.strategy_params['atr_tp_multiplier'])
        else:
            take_profit = current_price - (atr_value * self.strategy_params['atr_tp_multiplier'])
        
        # Create signal with enhanced reasoning
        reasoning = (f"Enhanced {signal_type}: conf={confidence:.3f}, "
                    f"bull={bullish_score:.2f}, bear={bearish_score:.2f}, "
                    f"vol={volume_ratio:.1f}x, trend={trend_strength:.3f}, "
                    f"mom={momentum_3:.3f}, rsi={rsi:.1f}")
        
        return {
            'symbol': symbol,
            'signal_type': SignalType.BUY if signal_type == 'BUY' else SignalType.SELL,
            'timestamp': timestamp,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reasoning': reasoning,
            'atr_value': atr_value,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'confirmation_details': confirmation['details']
        }
    
    def run_enhanced_backtest_with_improvements(self, block_id: str = "august_12_single_day"):
        """Run backtest with improved strategy."""
        
        print(f"🚀 Improved Aggressive Strategy Backtest")
        print(f"📦 Block: {block_id}")
        print("=" * 60)
        
        # Load data
        data = self.load_data_from_block(block_id)
        df_5m = data['5m'].copy()
        df_1m = data['1m'].copy()
        
        # Calculate enhanced features
        df_5m = self._calculate_enhanced_features(df_5m)
        
        # Reset performance metrics
        self.performance_metrics = {k: 0 for k in self.performance_metrics}
        
        # Run enhanced signal generation
        signals = []
        trades = []
        balance = self.initial_balance
        current_position = None
        
        print("🔧 Running enhanced simulation...")
        
        for i in range(50, len(df_5m)):  # Start after feature calculation period
            current_candle = df_5m.iloc[i]
            current_time = df_5m.index[i]
            current_price = current_candle['close']
            
            # Prepare features dictionary
            features = {
                'current_index': i,
                'dataframe': df_5m,
                'ema_8': current_candle['ema_8'],
                'ema_20': current_candle['ema_20'],
                'ema_50': current_candle['ema_50'],
                'rsi_smooth': current_candle['rsi_smooth'],
                'macd': current_candle['macd'],
                'macd_signal': current_candle['macd_signal'],
                'macd_histogram': current_candle['macd_histogram'],
                'volume_ratio': current_candle['volume_ratio'],
                'volume_trend': current_candle['volume_trend'],
                'volatility': current_candle['volatility'],
                'momentum_5': current_candle['momentum_5'],
                'momentum_3': current_candle['momentum_3'],
                'trend_strength': current_candle['trend_strength'],
                'price_position': current_candle['price_position'],
                'atr': current_candle['atr'],
            }
            
            # Check for exit conditions first
            if current_position is not None:
                exit_reason = None
                
                # Check stop loss and take profit
                if current_position['signal_type'] == SignalType.BUY:
                    if current_price <= current_position['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif current_price >= current_position['take_profit']:
                        exit_reason = 'take_profit'
                else:  # SELL
                    if current_price >= current_position['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif current_price <= current_position['take_profit']:
                        exit_reason = 'take_profit'
                
                # Check time stop
                time_diff = current_time - current_position['entry_time']
                if time_diff.total_seconds() / 3600 >= self.strategy_params['max_position_time_hours']:
                    exit_reason = 'time_stop'
                
                # Close position if exit condition met
                if exit_reason:
                    # Calculate P&L
                    if current_position['signal_type'] == SignalType.BUY:
                        pnl = (current_price - current_position['entry_price']) * current_position['quantity']
                    else:
                        pnl = (current_position['entry_price'] - current_price) * current_position['quantity']
                    
                    # Calculate commission (0.06% each side)
                    commission = (current_position['entry_price'] * current_position['quantity'] * 0.0006 +
                                current_price * current_position['quantity'] * 0.0006)
                    net_pnl = pnl - commission
                    
                    balance += net_pnl
                    
                    # Record trade
                    trade = {
                        'signal_type': current_position['signal_type'].value if hasattr(current_position['signal_type'], 'value') else current_position['signal_type'],
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': current_position['entry_price'],
                        'exit_price': current_price,
                        'quantity': current_position['quantity'],
                        'pnl': pnl,
                        'net_pnl': net_pnl,
                        'commission': commission,
                        'exit_reason': exit_reason,
                        'confidence': current_position['confidence'],
                        'reasoning': current_position['reasoning'],
                        'volatility_adjusted': current_position.get('volatility_adjusted', False),
                        'position_size_type': current_position.get('position_size_type', 'standard')
                    }
                    
                    trades.append(trade)
                    
                    # Print trade result
                    duration = (current_time - current_position['entry_time']).total_seconds() / 60
                    print(f"🔚 {current_time.strftime('%H:%M')} CLOSE @ ${current_price:.2f} | "
                          f"P&L: ${net_pnl:+.2f} | {exit_reason} | Bal: ${balance:.2f}")
                    
                    current_position = None
                    continue
            
            # Generate new signal if no position
            if current_position is None:
                signal = self._generate_enhanced_signal(
                    "SOLUSDT", current_price, current_candle, features, current_time
                )
                
                if signal:
                    # Calculate adaptive position size
                    position_value = self._calculate_adaptive_position_size(signal['confidence'], balance)
                    quantity = position_value / current_price
                    
                    # Determine position size type for tracking
                    if signal['confidence'] >= self.strategy_params['high_confidence_threshold']:
                        size_type = 'high'
                    elif signal['confidence'] >= self.strategy_params['medium_confidence_threshold']:
                        size_type = 'medium'
                    else:
                        size_type = 'low'
                    
                    current_position = {
                        'signal_type': signal['signal_type'],
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'quantity': quantity,
                        'confidence': signal['confidence'],
                        'reasoning': signal['reasoning'],
                        'volatility_adjusted': signal['volatility'] > self.strategy_params['high_volatility_threshold'],
                        'position_size_type': size_type
                    }
                    
                    signals.append(signal)
                    
                    # Print entry
                    signal_str = signal['signal_type'].value if hasattr(signal['signal_type'], 'value') else signal['signal_type']
                    print(f"📈 {len(trades)+1}. {current_time.strftime('%H:%M')} {signal_str} @ ${current_price:.2f} "
                          f"(conf: {signal['confidence']:.2f}, SL: ${signal['stop_loss']:.2f}, "
                          f"TP: ${signal['take_profit']:.2f}, size: {size_type})")
        
        # Close any remaining position
        if current_position is not None:
            final_time = df_5m.index[-1]
            final_price = df_5m.iloc[-1]['close']
            
            if current_position['signal_type'] == SignalType.BUY:
                pnl = (final_price - current_position['entry_price']) * current_position['quantity']
            else:
                pnl = (current_position['entry_price'] - final_price) * current_position['quantity']
            
            commission = (current_position['entry_price'] * current_position['quantity'] * 0.0006 +
                         final_price * current_position['quantity'] * 0.0006)
            net_pnl = pnl - commission
            balance += net_pnl
            
            trade = {
                'signal_type': current_position['signal_type'].value if hasattr(current_position['signal_type'], 'value') else current_position['signal_type'],
                'entry_time': current_position['entry_time'],
                'exit_time': final_time,
                'entry_price': current_position['entry_price'],
                'exit_price': final_price,
                'quantity': current_position['quantity'],
                'pnl': pnl,
                'net_pnl': net_pnl,
                'commission': commission,
                'exit_reason': 'end_of_data',
                'confidence': current_position['confidence'],
                'reasoning': current_position['reasoning'],
                'volatility_adjusted': current_position.get('volatility_adjusted', False),
                'position_size_type': current_position.get('position_size_type', 'standard')
            }
            
            trades.append(trade)
        
        # Calculate results
        winning_trades = [t for t in trades if t['net_pnl'] > 0]
        losing_trades = [t for t in trades if t['net_pnl'] <= 0]
        
        results = {
            'trades': trades,
            'signals_generated': len(signals),
            'final_balance': balance,
            'total_return': (balance / self.initial_balance - 1) * 100,
            'initial_balance': self.initial_balance,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
            'avg_win': np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0,
            'performance_metrics': self.performance_metrics
        }
        
        # Print final results
        self._print_enhanced_results(results)
        
        return results
    
    def _print_enhanced_results(self, results: Dict[str, Any]):
        """Print enhanced results with detailed metrics."""
        
        print(f"\n📊 IMPROVED AGGRESSIVE STRATEGY RESULTS")
        print("=" * 55)
        print(f"✅ Total trades: {len(results['trades'])}")
        print(f"🎯 Win rate: {results['win_rate']:.1f}% ({results['winning_trades']}/{len(results['trades'])})")
        print(f"📶 Signals generated: {results['signals_generated']}")
        print(f"💰 Total P&L: ${results['final_balance'] - results['initial_balance']:+.2f}")
        print(f"📈 Final balance: ${results['final_balance']:.2f}")
        print(f"📊 Return: {results['total_return']:+.2f}%")
        
        if results['winning_trades'] > 0:
            print(f"🏆 Average win: ${results['avg_win']:+.2f}")
        if results['losing_trades'] > 0:
            print(f"💸 Average loss: ${results['avg_loss']:+.2f}")
        
        # Enhanced metrics
        print(f"\n🔧 Enhancement Metrics:")
        print(f"• Signals filtered: {self.performance_metrics['signals_filtered']}")
        print(f"• Confirmation failures: {self.performance_metrics['confirmation_failures']}")
        print(f"• Volatility adjustments: {self.performance_metrics['volatility_adjustments']}")
        print(f"• Adaptive sizing used: {self.performance_metrics['adaptive_sizing_used']}")
        
        # Position sizing breakdown
        trades = results['trades']
        high_conf_trades = [t for t in trades if t.get('position_size_type') == 'high']
        medium_conf_trades = [t for t in trades if t.get('position_size_type') == 'medium']
        low_conf_trades = [t for t in trades if t.get('position_size_type') == 'low']
        
        if trades:
            print(f"\n💼 Position Sizing:")
            print(f"• High confidence (3%): {len(high_conf_trades)} trades")
            print(f"• Medium confidence (2%): {len(medium_conf_trades)} trades")
            print(f"• Low confidence (1%): {len(low_conf_trades)} trades")
        
        # Volatility adjustments
        vol_adjusted = [t for t in trades if t.get('volatility_adjusted', False)]
        if vol_adjusted:
            print(f"• Volatility adjusted: {len(vol_adjusted)} trades")


def main():
    """Main function to run improved strategy."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved Aggressive Strategy")
    parser.add_argument('--block', default='august_12_single_day', help='Data block to use')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    
    args = parser.parse_args()
    
    # Run improved strategy
    strategy = ImprovedAggressiveStrategy(initial_balance=args.balance)
    results = strategy.run_enhanced_backtest_with_improvements(args.block)
    
    return results

if __name__ == "__main__":
    main()