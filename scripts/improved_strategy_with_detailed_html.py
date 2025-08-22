#!/usr/bin/env python3
"""Improved strategy with ultra-detailed HTML reports for algorithm analysis."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.improved_aggressive_strategy import ImprovedAggressiveStrategy

class ImprovedStrategyWithDetailedHTML(ImprovedAggressiveStrategy):
    """Improved strategy with ultra-detailed HTML analysis reports."""
    
    def __init__(self, initial_balance: float = 10000.0):
        super().__init__(initial_balance)
        self.detailed_trade_data = []  # Store ultra-detailed trade analysis
        
    def capture_ultra_detailed_trade_data(self, trade_type: str, timestamp: datetime, 
                                        current_candle: pd.Series, features: Dict,
                                        signal_data: Dict = None, exit_data: Dict = None):
        """Capture ultra-detailed data for each trade decision."""
        
        # Get current market context
        current_idx = features.get('current_index', 0)
        df_5m = features.get('dataframe')
        
        # Calculate additional analysis metrics
        analysis_data = {
            'timestamp': timestamp,
            'trade_type': trade_type,  # 'entry', 'exit', 'signal_filtered'
            'price': current_candle['close'],
            'ohlcv': {
                'open': current_candle['open'],
                'high': current_candle['high'],
                'low': current_candle['low'],
                'close': current_candle['close'],
                'volume': current_candle['volume']
            },
            
            # Technical indicators with detailed analysis
            'indicators': {
                'ema_8': current_candle['ema_8'],
                'ema_20': current_candle['ema_20'],
                'ema_50': current_candle['ema_50'],
                'rsi': current_candle['rsi'],
                'rsi_smooth': current_candle['rsi_smooth'],
                'macd': current_candle['macd'],
                'macd_signal': current_candle['macd_signal'],
                'macd_histogram': current_candle['macd_histogram'],
                'volume_ratio': current_candle['volume_ratio'],
                'volume_trend': current_candle['volume_trend'],
                'volatility': current_candle['volatility'],
                'atr': current_candle['atr'],
                'atr_pct': current_candle['atr_pct'],
                'momentum_5': current_candle['momentum_5'],
                'momentum_3': current_candle['momentum_3'],
                'trend_strength': current_candle['trend_strength'],
                'price_position': current_candle['price_position'],
                'support': current_candle['support'],
                'resistance': current_candle['resistance']
            },
            
            # Market condition analysis
            'market_conditions': {
                'trend_direction': self._analyze_trend_direction(current_candle),
                'trend_strength_level': self._analyze_trend_strength(current_candle),
                'volatility_regime': self._analyze_volatility_regime(current_candle),
                'volume_regime': self._analyze_volume_regime(current_candle),
                'rsi_regime': self._analyze_rsi_regime(current_candle),
                'momentum_strength': self._analyze_momentum_strength(current_candle),
                'price_level_analysis': self._analyze_price_levels(current_candle),
                'macd_status': self._analyze_macd_status(current_candle)
            },
            
            # Historical context (last 5 candles)
            'historical_context': self._get_historical_context(df_5m, current_idx),
            
            # Strategy scoring breakdown
            'scoring_breakdown': self._get_detailed_scoring(current_candle, features) if trade_type == 'entry' else None,
            
            # Confirmation analysis
            'confirmation_analysis': features.get('confirmation_details') if trade_type == 'entry' else None,
            
            # Signal specific data
            'signal_data': signal_data,
            'exit_data': exit_data
        }
        
        self.detailed_trade_data.append(analysis_data)
        return len(self.detailed_trade_data) - 1  # Return index for linking
    
    def _analyze_trend_direction(self, candle: pd.Series) -> str:
        """Analyze trend direction with detailed explanation."""
        ema8 = candle['ema_8']
        ema20 = candle['ema_20']
        ema50 = candle['ema_50']
        
        if ema8 > ema20 > ema50:
            strength = (ema8 - ema50) / ema50 * 100
            return f"Strong Bullish (EMAs aligned, {strength:.2f}% spread)"
        elif ema8 < ema20 < ema50:
            strength = (ema50 - ema8) / ema50 * 100
            return f"Strong Bearish (EMAs aligned, {strength:.2f}% spread)"
        elif ema8 > ema20:
            return "Weak Bullish (EMA8 > EMA20, but EMA20 < EMA50)"
        elif ema8 < ema20:
            return "Weak Bearish (EMA8 < EMA20, but EMA20 > EMA50)"
        else:
            return "Sideways/Consolidation (EMAs converging)"
    
    def _analyze_trend_strength(self, candle: pd.Series) -> str:
        """Analyze trend strength level."""
        strength = candle['trend_strength']
        if strength > 0.01:
            return f"Very Strong ({strength:.3f})"
        elif strength > 0.005:
            return f"Strong ({strength:.3f})"
        elif strength > 0.002:
            return f"Moderate ({strength:.3f})"
        else:
            return f"Weak ({strength:.3f})"
    
    def _analyze_volatility_regime(self, candle: pd.Series) -> str:
        """Analyze volatility regime."""
        vol = candle['volatility']
        atr_pct = candle['atr_pct']
        
        if vol > 0.02:
            return f"High Volatility (σ={vol:.3f}, ATR={atr_pct:.2%})"
        elif vol > 0.01:
            return f"Medium Volatility (σ={vol:.3f}, ATR={atr_pct:.2%})"
        else:
            return f"Low Volatility (σ={vol:.3f}, ATR={atr_pct:.2%})"
    
    def _analyze_volume_regime(self, candle: pd.Series) -> str:
        """Analyze volume regime."""
        vol_ratio = candle['volume_ratio']
        vol_trend = candle['volume_trend']
        
        if vol_ratio > 2.0:
            return f"Exceptional Volume ({vol_ratio:.1f}x avg, trend: {vol_trend:.1f}x)"
        elif vol_ratio > 1.5:
            return f"High Volume ({vol_ratio:.1f}x avg, trend: {vol_trend:.1f}x)"
        elif vol_ratio > 1.2:
            return f"Above Average Volume ({vol_ratio:.1f}x avg, trend: {vol_trend:.1f}x)"
        elif vol_ratio > 0.8:
            return f"Normal Volume ({vol_ratio:.1f}x avg, trend: {vol_trend:.1f}x)"
        else:
            return f"Low Volume ({vol_ratio:.1f}x avg, trend: {vol_trend:.1f}x)"
    
    def _analyze_rsi_regime(self, candle: pd.Series) -> str:
        """Analyze RSI regime."""
        rsi = candle['rsi']
        rsi_smooth = candle['rsi_smooth']
        
        if rsi > 80:
            return f"Extremely Overbought (RSI={rsi:.1f}, smooth={rsi_smooth:.1f})"
        elif rsi > 70:
            return f"Overbought (RSI={rsi:.1f}, smooth={rsi_smooth:.1f})"
        elif rsi > 60:
            return f"Bullish Zone (RSI={rsi:.1f}, smooth={rsi_smooth:.1f})"
        elif rsi > 40:
            return f"Neutral Zone (RSI={rsi:.1f}, smooth={rsi_smooth:.1f})"
        elif rsi > 30:
            return f"Bearish Zone (RSI={rsi:.1f}, smooth={rsi_smooth:.1f})"
        elif rsi > 20:
            return f"Oversold (RSI={rsi:.1f}, smooth={rsi_smooth:.1f})"
        else:
            return f"Extremely Oversold (RSI={rsi:.1f}, smooth={rsi_smooth:.1f})"
    
    def _analyze_momentum_strength(self, candle: pd.Series) -> str:
        """Analyze momentum strength."""
        mom3 = candle['momentum_3']
        mom5 = candle['momentum_5']
        
        if mom3 > 0.01:
            return f"Strong Bullish Momentum (3p: {mom3:.2%}, 5p: {mom5:.2%})"
        elif mom3 > 0.005:
            return f"Moderate Bullish Momentum (3p: {mom3:.2%}, 5p: {mom5:.2%})"
        elif mom3 > 0.002:
            return f"Weak Bullish Momentum (3p: {mom3:.2%}, 5p: {mom5:.2%})"
        elif mom3 > -0.002:
            return f"Neutral Momentum (3p: {mom3:.2%}, 5p: {mom5:.2%})"
        elif mom3 > -0.005:
            return f"Weak Bearish Momentum (3p: {mom3:.2%}, 5p: {mom5:.2%})"
        elif mom3 > -0.01:
            return f"Moderate Bearish Momentum (3p: {mom3:.2%}, 5p: {mom5:.2%})"
        else:
            return f"Strong Bearish Momentum (3p: {mom3:.2%}, 5p: {mom5:.2%})"
    
    def _analyze_price_levels(self, candle: pd.Series) -> str:
        """Analyze price level positioning."""
        price = candle['close']
        support = candle['support']
        resistance = candle['resistance']
        position = candle['price_position']
        
        range_size = resistance - support
        range_pct = range_size / price * 100
        
        if position > 0.8:
            return f"Near Resistance (pos={position:.1%}, range={range_pct:.1%})"
        elif position > 0.6:
            return f"Upper Range (pos={position:.1%}, range={range_pct:.1%})"
        elif position > 0.4:
            return f"Middle Range (pos={position:.1%}, range={range_pct:.1%})"
        elif position > 0.2:
            return f"Lower Range (pos={position:.1%}, range={range_pct:.1%})"
        else:
            return f"Near Support (pos={position:.1%}, range={range_pct:.1%})"
    
    def _analyze_macd_status(self, candle: pd.Series) -> str:
        """Analyze MACD status."""
        macd = candle['macd']
        signal = candle['macd_signal']
        histogram = candle['macd_histogram']
        
        if macd > signal and histogram > 0:
            return f"Bullish MACD (line={macd:.3f}, signal={signal:.3f}, hist={histogram:.3f})"
        elif macd < signal and histogram < 0:
            return f"Bearish MACD (line={macd:.3f}, signal={signal:.3f}, hist={histogram:.3f})"
        elif macd > signal:
            return f"Weakening Bullish MACD (line={macd:.3f}, signal={signal:.3f}, hist={histogram:.3f})"
        else:
            return f"Weakening Bearish MACD (line={macd:.3f}, signal={signal:.3f}, hist={histogram:.3f})"
    
    def _get_historical_context(self, df_5m: pd.DataFrame, current_idx: int) -> Dict:
        """Get historical context for the last 5 candles."""
        if current_idx < 5:
            return {"error": "Insufficient historical data"}
        
        history = []
        for i in range(5, 0, -1):  # Last 5 candles
            idx = current_idx - i
            candle = df_5m.iloc[idx]
            
            history.append({
                'candles_ago': i,
                'timestamp': df_5m.index[idx],
                'price': candle['close'],
                'price_change': candle['close'] / df_5m.iloc[idx-1]['close'] - 1 if idx > 0 else 0,
                'volume_ratio': candle['volume_ratio'],
                'rsi': candle['rsi'],
                'trend': 'up' if candle['ema_8'] > candle['ema_20'] else 'down'
            })
        
        return {
            'last_5_candles': history,
            'price_trend': 'rising' if history[0]['price'] < history[-1]['price'] else 'falling',
            'volume_trend': 'increasing' if np.mean([h['volume_ratio'] for h in history[:2]]) < np.mean([h['volume_ratio'] for h in history[-2:]]) else 'decreasing'
        }
    
    def _get_detailed_scoring(self, candle: pd.Series, features: Dict) -> Dict:
        """Get detailed scoring breakdown for signal generation."""
        
        scoring = {
            'trend_analysis': {
                'weight': 35,
                'factors': {}
            },
            'momentum_analysis': {
                'weight': 25,
                'factors': {}
            },
            'rsi_analysis': {
                'weight': 15,
                'factors': {}
            },
            'volume_analysis': {
                'weight': 15,
                'factors': {}
            },
            'price_position': {
                'weight': 10,
                'factors': {}
            }
        }
        
        # Trend analysis scoring
        if candle['ema_8'] > candle['ema_20'] > candle['ema_50']:
            scoring['trend_analysis']['factors']['ema_alignment_bullish'] = {'score': 20, 'reason': 'EMA8 > EMA20 > EMA50'}
            if candle['trend_strength'] > 0.005:
                scoring['trend_analysis']['factors']['strong_trend'] = {'score': 15, 'reason': f'Trend strength {candle["trend_strength"]:.3f} > 0.005'}
        elif candle['ema_8'] < candle['ema_20'] < candle['ema_50']:
            scoring['trend_analysis']['factors']['ema_alignment_bearish'] = {'score': 20, 'reason': 'EMA8 < EMA20 < EMA50'}
            if candle['trend_strength'] > 0.005:
                scoring['trend_analysis']['factors']['strong_trend'] = {'score': 15, 'reason': f'Trend strength {candle["trend_strength"]:.3f} > 0.005'}
        
        # Momentum analysis scoring
        if candle['momentum_5'] > 0.005 and candle['momentum_3'] > 0.002:
            scoring['momentum_analysis']['factors']['strong_bullish_momentum'] = {'score': 15, 'reason': f'5p momentum {candle["momentum_5"]:.3f} > 0.005, 3p momentum {candle["momentum_3"]:.3f} > 0.002'}
        elif candle['momentum_5'] < -0.005 and candle['momentum_3'] < -0.002:
            scoring['momentum_analysis']['factors']['strong_bearish_momentum'] = {'score': 15, 'reason': f'5p momentum {candle["momentum_5"]:.3f} < -0.005, 3p momentum {candle["momentum_3"]:.3f} < -0.002'}
        
        if candle['macd'] > candle['macd_signal'] and candle['macd_histogram'] > 0:
            scoring['momentum_analysis']['factors']['bullish_macd'] = {'score': 10, 'reason': f'MACD {candle["macd"]:.3f} > Signal {candle["macd_signal"]:.3f}, Histogram {candle["macd_histogram"]:.3f} > 0'}
        elif candle['macd'] < candle['macd_signal'] and candle['macd_histogram'] < 0:
            scoring['momentum_analysis']['factors']['bearish_macd'] = {'score': 10, 'reason': f'MACD {candle["macd"]:.3f} < Signal {candle["macd_signal"]:.3f}, Histogram {candle["macd_histogram"]:.3f} < 0'}
        
        # RSI analysis scoring
        rsi = candle['rsi_smooth']
        if 30 < rsi < 70:
            if rsi > 55:
                scoring['rsi_analysis']['factors']['bullish_rsi'] = {'score': 8, 'reason': f'RSI {rsi:.1f} in bullish zone (55-70)'}
            elif rsi < 45:
                scoring['rsi_analysis']['factors']['bearish_rsi'] = {'score': 8, 'reason': f'RSI {rsi:.1f} in bearish zone (30-45)'}
        elif rsi < 30:
            scoring['rsi_analysis']['factors']['oversold_rsi'] = {'score': 15, 'reason': f'RSI {rsi:.1f} oversold < 30'}
        elif rsi > 70:
            scoring['rsi_analysis']['factors']['overbought_rsi'] = {'score': 15, 'reason': f'RSI {rsi:.1f} overbought > 70'}
        
        # Volume analysis scoring
        if candle['volume_ratio'] > 1.5 and candle['volume_trend'] > 1.2:
            scoring['volume_analysis']['factors']['strong_volume_confirmation'] = {'score': 15, 'reason': f'Volume ratio {candle["volume_ratio"]:.1f}x > 1.5, trend {candle["volume_trend"]:.1f}x > 1.2'}
        elif candle['volume_ratio'] > 1.2:
            scoring['volume_analysis']['factors']['volume_confirmation'] = {'score': 8, 'reason': f'Volume ratio {candle["volume_ratio"]:.1f}x > 1.2'}
        
        # Price position analysis
        if candle['price_position'] > 0.7:
            scoring['price_position']['factors']['near_resistance'] = {'score': 5, 'reason': f'Price position {candle["price_position"]:.1%} > 70% (bearish)'}
        elif candle['price_position'] < 0.3:
            scoring['price_position']['factors']['near_support'] = {'score': 5, 'reason': f'Price position {candle["price_position"]:.1%} < 30% (bullish)'}
        
        return scoring
    
    def run_enhanced_backtest_with_ultra_detailed_reporting(self, block_id: str = "august_12_single_day"):
        """Run backtest with ultra-detailed reporting and data capture."""
        
        print(f"🔬 Ultra-Detailed Strategy Analysis")
        print(f"📦 Block: {block_id}")
        print("=" * 60)
        
        # Load data
        data = self.load_data_from_block(block_id)
        df_5m = data['5m'].copy()
        df_1m = data['1m'].copy()
        
        # Calculate enhanced features
        df_5m = self._calculate_enhanced_features(df_5m)
        
        # Reset detailed data and performance metrics
        self.detailed_trade_data = []
        self.performance_metrics = {k: 0 for k in self.performance_metrics}
        
        # Run enhanced signal generation with detailed capture
        signals = []
        trades = []
        balance = self.initial_balance
        current_position = None
        
        print("🔧 Running ultra-detailed simulation...")
        
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
                'rsi': current_candle['rsi'],
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
                if current_position['signal_type'] == 'BUY':
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
                    # Capture exit data
                    exit_idx = self.capture_ultra_detailed_trade_data(
                        'exit', current_time, current_candle, features,
                        exit_data={'reason': exit_reason, 'price': current_price}
                    )
                    
                    # Calculate P&L
                    if current_position['signal_type'] == 'BUY':
                        pnl = (current_price - current_position['entry_price']) * current_position['quantity']
                    else:
                        pnl = (current_position['entry_price'] - current_price) * current_position['quantity']
                    
                    # Calculate commission
                    commission = (current_position['entry_price'] * current_position['quantity'] * 0.0006 +
                                current_price * current_position['quantity'] * 0.0006)
                    net_pnl = pnl - commission
                    
                    balance += net_pnl
                    
                    # Record trade with detailed data indices
                    trade = {
                        'signal_type': current_position['signal_type'],
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
                        'position_size_type': current_position.get('position_size_type', 'standard'),
                        'entry_detailed_data_idx': current_position.get('detailed_data_index'),
                        'exit_detailed_data_idx': exit_idx
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
                signal = self._generate_enhanced_signal_with_capture(
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
                        'position_size_type': size_type,
                        'detailed_data_index': signal.get('detailed_data_index')
                    }
                    
                    signals.append(signal)
                    
                    # Print entry
                    signal_str = signal['signal_type'].value if hasattr(signal['signal_type'], 'value') else str(signal['signal_type'])
                    print(f"📈 {len(trades)+1}. {current_time.strftime('%H:%M')} {signal_str} @ ${current_price:.2f} "
                          f"(conf: {signal['confidence']:.2f}, SL: ${signal['stop_loss']:.2f}, "
                          f"TP: ${signal['take_profit']:.2f}, size: {size_type})")
        
        # Close any remaining position
        if current_position is not None:
            final_time = df_5m.index[-1]
            final_price = df_5m.iloc[-1]['close']
            final_candle = df_5m.iloc[-1]
            
            # Capture final exit
            exit_idx = self.capture_ultra_detailed_trade_data(
                'exit', final_time, final_candle, features,
                exit_data={'reason': 'end_of_data', 'price': final_price}
            )
            
            if current_position['signal_type'] == 'BUY':
                pnl = (final_price - current_position['entry_price']) * current_position['quantity']
            else:
                pnl = (current_position['entry_price'] - final_price) * current_position['quantity']
            
            commission = (current_position['entry_price'] * current_position['quantity'] * 0.0006 +
                         final_price * current_position['quantity'] * 0.0006)
            net_pnl = pnl - commission
            balance += net_pnl
            
            trade = {
                'signal_type': current_position['signal_type'],
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
                'position_size_type': current_position.get('position_size_type', 'standard'),
                'entry_detailed_data_idx': current_position.get('detailed_data_index'),
                'exit_detailed_data_idx': exit_idx
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
            'performance_metrics': self.performance_metrics,
            'detailed_trade_data': self.detailed_trade_data
        }
        
        # Print final results
        self._print_enhanced_results(results)
        
        # Generate ultra-detailed HTML report
        if results.get('trades'):
            html_file = self.generate_ultra_detailed_html_report(results, block_id)
            print(f"\n🔬 Ultra-detailed HTML report: {html_file}")
            results['detailed_html_report'] = html_file
        
        return results
    
    def _generate_enhanced_signal_with_capture(self, symbol: str, current_price: float,
                                             current_candle: pd.Series, features: Dict[str, float],
                                             timestamp: datetime) -> Optional[Dict]:
        """Enhanced signal generation with detailed data capture."""
        
        # Capture detailed data for this signal attempt
        detailed_idx = self.capture_ultra_detailed_trade_data(
            'signal_attempt', timestamp, current_candle, features
        )
        
        # Call parent method logic but with data capture
        current_idx = features.get('current_index', 0)
        df_5m = features.get('dataframe')
        
        if df_5m is None or current_idx < 50:
            return None
        
        self.performance_metrics['signals_generated'] += 1
        
        # Extract enhanced features (same as parent)
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
        
        # Enhanced scoring system (same logic as parent)
        bullish_score = 0.0
        bearish_score = 0.0
        
        # Trend analysis
        if ema_8 > ema_20 > ema_50:
            bullish_score += 0.20
            if trend_strength > 0.005:
                bullish_score += 0.15
        elif ema_8 < ema_20 < ema_50:
            bearish_score += 0.20
            if trend_strength > 0.005:
                bearish_score += 0.15
        
        # Momentum analysis
        if momentum_5 > 0.005 and momentum_3 > 0.002:
            bullish_score += 0.15
        elif momentum_5 < -0.005 and momentum_3 < -0.002:
            bearish_score += 0.15
        
        if macd > macd_signal and macd_histogram > 0:
            bullish_score += 0.10
        elif macd < macd_signal and macd_histogram < 0:
            bearish_score += 0.10
        
        # RSI analysis
        if 30 < rsi < 70:
            if rsi > 55:
                bullish_score += 0.08
            elif rsi < 45:
                bearish_score += 0.08
        elif rsi < 30:
            bullish_score += 0.15
        elif rsi > 70:
            bearish_score += 0.15
        
        # Volume analysis
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
        
        # Price position analysis
        if price_position > 0.7:
            bearish_score += 0.05
        elif price_position < 0.3:
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
        
        # Check confirmation filters with capture
        confirmation = self._check_confirmation_filters(df_5m, current_idx, signal_type)
        if not confirmation['passed']:
            self.performance_metrics['confirmation_failures'] += 1
            return None
        
        # Store confirmation details in features for capture
        features['confirmation_details'] = confirmation['details']
        
        # Capture entry data with all details
        entry_idx = self.capture_ultra_detailed_trade_data(
            'entry', timestamp, current_candle, features,
            signal_data={
                'signal_type': signal_type,
                'confidence': confidence,
                'bullish_score': bullish_score,
                'bearish_score': bearish_score
            }
        )
        
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
        
        from src.utils.types import SignalType
        
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
            'confirmation_details': confirmation['details'],
            'detailed_data_index': entry_idx
        }
    
    def generate_ultra_detailed_html_report(self, results: Dict[str, Any], block_id: str) -> str:
        """Generate ultra-detailed HTML report with complete algorithm analysis."""
        
        trades = results['trades']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_file = f"Output/Ultra_Detailed_Strategy_Report_{block_id}_{timestamp}.html"
        
        winning_trades = [t for t in trades if t.get('net_pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('net_pnl', 0) <= 0]
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultra-Detailed Strategy Analysis - {block_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .trades-table th,
        .trades-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .trades-table th {{
            background-color: #34495e;
            color: white;
            position: sticky;
            top: 0;
        }}
        .trades-table tr:hover {{
            background-color: #f0f8ff;
            cursor: pointer;
        }}
        .profit {{ color: #27ae60; font-weight: bold; }}
        .loss {{ color: #e74c3c; font-weight: bold; }}
        
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }}
        .modal-content {{
            background-color: #fefefe;
            margin: 1% auto;
            padding: 20px;
            border-radius: 10px;
            width: 95%;
            max-width: 1400px;
            max-height: 95vh;
            overflow-y: auto;
        }}
        .close {{
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{ color: black; }}
        
        .analysis-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .analysis-section {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .analysis-section h4 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        
        .indicator-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }}
        .indicator-item {{
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ddd;
            font-size: 13px;
        }}
        .indicator-label {{
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .scoring-breakdown {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .score-category {{
            margin: 10px 0;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            border-left: 3px solid #3498db;
        }}
        .score-factor {{
            margin: 5px 0 5px 20px;
            font-size: 13px;
            color: #555;
        }}
        
        .market-conditions {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }}
        .condition-item {{
            background-color: white;
            padding: 12px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        .condition-label {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .historical-context {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #ffeaa7;
        }}
        .history-candle {{
            display: inline-block;
            margin: 5px;
            padding: 8px;
            background-color: white;
            border-radius: 4px;
            border: 1px solid #ddd;
            font-size: 12px;
        }}
        
        .confirmation-analysis {{
            background-color: #e8f5e8;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #c3e6c3;
        }}
        
        .risk-calculation {{
            background-color: #ffebee;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #ffcdd2;
        }}
        .risk-item {{
            margin: 8px 0;
            padding: 8px;
            background-color: white;
            border-radius: 4px;
        }}
        
        .tabs {{
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px;
        }}
        .tab-button {{
            background: none;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            font-size: 14px;
        }}
        .tab-button.active {{
            border-bottom-color: #3498db;
            color: #3498db;
            font-weight: bold;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card.green {{
            background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        }}
        .stat-card.red {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        }}
        
        .enhanced-explanation {{
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
            border: 2px solid #3498db;
        }}
        .explanation-title {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        .explanation-section {{
            margin: 15px 0;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Ultra-Detailed Strategy Analysis Report</h1>
        
        <div class="summary-stats">
            <div class="stat-card">
                <h3>Total Trades</h3>
                <div style="font-size: 24px; font-weight: bold;">{len(trades)}</div>
            </div>
            <div class="stat-card green">
                <h3>Winning Trades</h3>
                <div style="font-size: 24px; font-weight: bold;">{len(winning_trades)}</div>
                <div style="font-size: 14px; opacity: 0.9;">{len(winning_trades) / len(trades) * 100:.1f}%</div>
            </div>
            <div class="stat-card red">
                <h3>Losing Trades</h3>
                <div style="font-size: 24px; font-weight: bold;">{len(losing_trades)}</div>
                <div style="font-size: 14px; opacity: 0.9;">{len(losing_trades) / len(trades) * 100:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Total Return</h3>
                <div style="font-size: 24px; font-weight: bold;">{results['total_return']:+.2f}%</div>
            </div>
            <div class="stat-card">
                <h3>Enhancement Metrics</h3>
                <div style="font-size: 14px;">Filtered: {results['performance_metrics']['signals_filtered']}</div>
                <div style="font-size: 14px;">Confirmations: {results['performance_metrics']['confirmation_failures']}</div>
                <div style="font-size: 14px;">Adaptive: {results['performance_metrics']['adaptive_sizing_used']}</div>
            </div>
        </div>
        
        <h2>💼 Interactive Trade Analysis</h2>
        <p><strong>Кликните на любую сделку для ПОЛНОГО анализа алгоритма:</strong></p>
        <ul>
            <li>🎯 <strong>Факторы входа</strong> - почему именно тогда был сделан вход</li>
            <li>📊 <strong>Расчёт TP/SL</strong> - как были рассчитаны уровни</li>
            <li>🔍 <strong>Технические индикаторы</strong> - все значения на момент решения</li>
            <li>📈 <strong>Состояние рынка</strong> - полный анализ рыночных условий</li>
            <li>📋 <strong>Подтверждающие факторы</strong> - что подтвердило сигнал</li>
            <li>⏱️ <strong>Исторический контекст</strong> - что происходило до этого</li>
        </ul>
        
        <table class="trades-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Duration</th>
                    <th>Type</th>
                    <th>Confidence</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>P&L</th>
                    <th>Exit Reason</th>
                    <th>Position Size</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add trade rows with ultra-detailed data
        for i, trade in enumerate(trades, 1):
            pnl = trade.get('net_pnl', 0)
            pnl_class = 'profit' if pnl > 0 else 'loss'
            
            # Find corresponding detailed trade data
            entry_data = None
            exit_data = None
            
            # Look for detailed data (simplified - you might need to match by timestamp)
            for detailed_data in self.detailed_trade_data:
                if (detailed_data['trade_type'] == 'entry' and 
                    abs((detailed_data['timestamp'] - trade['entry_time']).total_seconds()) < 300):  # Within 5 minutes
                    entry_data = detailed_data
                    break
            
            # Calculate duration
            if hasattr(trade['exit_time'], '__sub__'):
                duration_minutes = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60
                duration_str = f"{duration_minutes:.0f}m"
            else:
                duration_str = "N/A"
            
            # Prepare ultra-detailed data for JavaScript
            ultra_detailed_data = {
                'trade_number': i,
                'trade_info': {
                    'entry_time': trade['entry_time'].isoformat() if hasattr(trade['entry_time'], 'isoformat') else str(trade['entry_time']),
                    'exit_time': trade['exit_time'].isoformat() if hasattr(trade['exit_time'], 'isoformat') else str(trade['exit_time']),
                    'duration': duration_str,
                    'signal_type': trade['signal_type'],
                    'confidence': trade['confidence'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'pnl': pnl,
                    'exit_reason': trade['exit_reason'],
                    'position_size_type': trade.get('position_size_type', 'standard'),
                    'volatility_adjusted': trade.get('volatility_adjusted', False),
                    'reasoning': trade.get('reasoning', '')
                },
                'entry_analysis': entry_data if entry_data else {'error': 'No detailed entry data found'},
            }
            
            html_content += f"""
                <tr onclick="showUltraDetailedAnalysis({json.dumps(ultra_detailed_data, default=str).replace('"', '&quot;')})">
                    <td>{i}</td>
                    <td>{trade['entry_time'].strftime('%H:%M') if hasattr(trade['entry_time'], 'strftime') else trade['entry_time']}</td>
                    <td>{trade['exit_time'].strftime('%H:%M') if hasattr(trade['exit_time'], 'strftime') else trade['exit_time']}</td>
                    <td>{duration_str}</td>
                    <td>{trade['signal_type']}</td>
                    <td>{trade['confidence']:.2f}</td>
                    <td>${trade['entry_price']:.2f}</td>
                    <td>${trade['exit_price']:.2f}</td>
                    <td class="{pnl_class}">${pnl:+.2f}</td>
                    <td>{trade['exit_reason']}</td>
                    <td>{trade.get('position_size_type', 'std')}</td>
                </tr>
            """

        html_content += """
            </tbody>
        </table>
    </div>
    
    <!-- Ultra-detailed modal -->
    <div id="ultraDetailedModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeUltraDetailedModal()">&times;</span>
            <div id="ultraDetailedContent">
                <!-- Ultra-detailed analysis will be loaded here -->
            </div>
        </div>
    </div>
    
    <script>
        function showUltraDetailedAnalysis(data) {
            const modal = document.getElementById('ultraDetailedModal');
            const content = document.getElementById('ultraDetailedContent');
            
            const trade = data.trade_info;
            const entry = data.entry_analysis;
            
            let html = `
                <h2>🔬 Trade #${data.trade_number} - Complete Algorithm Analysis</h2>
                
                <div class="enhanced-explanation">
                    <div class="explanation-title">🎯 Why This Trade Was Opened</div>
                    <div class="explanation-section">
                        <h4>🚀 Entry Decision Summary</h4>
                        <p><strong>Signal:</strong> ${trade.signal_type} at $${trade.entry_price} with ${(trade.confidence * 100).toFixed(1)}% confidence</p>
                        <p><strong>Algorithm Reasoning:</strong> ${trade.reasoning}</p>
                        <p><strong>Position Size:</strong> ${trade.position_size_type} confidence level (${trade.volatility_adjusted ? 'volatility adjusted' : 'standard sizing'})</p>
                    </div>
                </div>
                
                <div class="tabs">
                    <button class="tab-button active" onclick="showTab('market-conditions')">🌍 Market Conditions</button>
                    <button class="tab-button" onclick="showTab('technical-indicators')">📊 Technical Indicators</button>
                    <button class="tab-button" onclick="showTab('scoring-breakdown')">🎯 Scoring Breakdown</button>
                    <button class="tab-button" onclick="showTab('risk-calculation')">⚖️ Risk Calculation</button>
                    <button class="tab-button" onclick="showTab('historical-context')">⏱️ Historical Context</button>
                    <button class="tab-button" onclick="showTab('confirmation-analysis')">✅ Confirmation Analysis</button>
                </div>
            `;
            
            // Market Conditions Tab
            html += `
                <div id="market-conditions" class="tab-content active">
                    <h3>🌍 Market Conditions Analysis</h3>
                    <div class="market-conditions">
            `;
            
            if (entry.market_conditions) {
                Object.entries(entry.market_conditions).forEach(([key, value]) => {
                    html += `
                        <div class="condition-item">
                            <div class="condition-label">${key.replace(/_/g, ' ').toUpperCase()}</div>
                            <div>${value}</div>
                        </div>
                    `;
                });
            } else {
                html += `<div class="condition-item">Market conditions data not available</div>`;
            }
            
            html += `
                    </div>
                </div>
            `;
            
            // Technical Indicators Tab
            html += `
                <div id="technical-indicators" class="tab-content">
                    <h3>📊 Technical Indicators at Entry</h3>
                    <div class="indicator-grid">
            `;
            
            if (entry.indicators) {
                Object.entries(entry.indicators).forEach(([key, value]) => {
                    html += `
                        <div class="indicator-item">
                            <span class="indicator-label">${key.toUpperCase()}:</span><br>
                            ${typeof value === 'number' ? value.toFixed(4) : value}
                        </div>
                    `;
                });
            } else {
                html += `<div class="indicator-item">Technical indicators data not available</div>`;
            }
            
            html += `
                    </div>
                    
                    <h4>📈 OHLCV Data</h4>
                    <div class="indicator-grid">
            `;
            
            if (entry.ohlcv) {
                Object.entries(entry.ohlcv).forEach(([key, value]) => {
                    html += `
                        <div class="indicator-item">
                            <span class="indicator-label">${key.toUpperCase()}:</span><br>
                            ${typeof value === 'number' ? value.toFixed(2) : value}
                        </div>
                    `;
                });
            }
            
            html += `
                    </div>
                </div>
            `;
            
            // Scoring Breakdown Tab
            html += `
                <div id="scoring-breakdown" class="tab-content">
                    <h3>🎯 Algorithm Scoring Breakdown</h3>
                    <div class="scoring-breakdown">
                        <h4>How the algorithm calculated confidence: ${(trade.confidence * 100).toFixed(1)}%</h4>
            `;
            
            if (entry.scoring_breakdown) {
                Object.entries(entry.scoring_breakdown).forEach(([category, data]) => {
                    html += `
                        <div class="score-category">
                            <h5>${category.replace(/_/g, ' ').toUpperCase()} (Weight: ${data.weight}%)</h5>
                    `;
                    
                    if (data.factors && Object.keys(data.factors).length > 0) {
                        Object.entries(data.factors).forEach(([factor, details]) => {
                            html += `
                                <div class="score-factor">
                                    ✓ <strong>${factor.replace(/_/g, ' ')}:</strong> +${details.score} points - ${details.reason}
                                </div>
                            `;
                        });
                    } else {
                        html += `<div class="score-factor">No factors contributed from this category</div>`;
                    }
                    
                    html += `</div>`;
                });
            } else {
                html += `<div>Scoring breakdown data not available</div>`;
            }
            
            html += `
                    </div>
                </div>
            `;
            
            // Risk Calculation Tab
            html += `
                <div id="risk-calculation" class="tab-content">
                    <h3>⚖️ Risk Management Calculation</h3>
                    <div class="risk-calculation">
                        <div class="risk-item">
                            <strong>Entry Price:</strong> $${trade.entry_price}
                        </div>
                        <div class="risk-item">
                            <strong>Stop Loss Calculation:</strong><br>
                            Based on ATR × ${trade.volatility_adjusted ? 'volatility-adjusted ' : ''}multiplier<br>
                            ${entry.indicators ? `ATR: ${entry.indicators.atr ? entry.indicators.atr.toFixed(3) : 'N/A'} (${entry.indicators.atr_pct ? (entry.indicators.atr_pct * 100).toFixed(2) : 'N/A'}% of price)` : 'ATR data not available'}
                        </div>
                        <div class="risk-item">
                            <strong>Position Sizing:</strong><br>
                            ${trade.position_size_type} confidence → ${
                                trade.position_size_type === 'high' ? '3% of balance' :
                                trade.position_size_type === 'medium' ? '2% of balance' :
                                '1% of balance'
                            }<br>
                            ${trade.volatility_adjusted ? 'Adjusted for high volatility conditions' : 'Standard sizing applied'}
                        </div>
                        <div class="risk-item">
                            <strong>Take Profit Target:</strong><br>
                            Calculated as Entry ± (ATR × 3.0 multiplier)
                        </div>
                    </div>
                </div>
            `;
            
            // Historical Context Tab
            html += `
                <div id="historical-context" class="tab-content">
                    <h3>⏱️ Historical Context (Last 5 Candles)</h3>
                    <div class="historical-context">
            `;
            
            if (entry.historical_context && entry.historical_context.last_5_candles) {
                html += `<h4>Market Leading Up To Entry:</h4>`;
                entry.historical_context.last_5_candles.forEach(candle => {
                    html += `
                        <div class="history-candle">
                            <strong>${candle.candles_ago} candles ago</strong><br>
                            Price: $${candle.price.toFixed(2)} (${(candle.price_change * 100).toFixed(2)}%)<br>
                            Volume: ${candle.volume_ratio.toFixed(1)}x<br>
                            RSI: ${candle.rsi.toFixed(1)}<br>
                            Trend: ${candle.trend}
                        </div>
                    `;
                });
                
                html += `
                    <p><strong>Price Trend:</strong> ${entry.historical_context.price_trend}</p>
                    <p><strong>Volume Trend:</strong> ${entry.historical_context.volume_trend}</p>
                `;
            } else {
                html += `<div>Historical context data not available</div>`;
            }
            
            html += `
                    </div>
                </div>
            `;
            
            // Confirmation Analysis Tab
            html += `
                <div id="confirmation-analysis" class="tab-content">
                    <h3>✅ Confirmation Filters Analysis</h3>
                    <div class="confirmation-analysis">
            `;
            
            if (entry.confirmation_analysis) {
                html += `<h4>All confirmation filters passed:</h4>`;
                Object.entries(entry.confirmation_analysis).forEach(([filter, passed]) => {
                    html += `
                        <div style="margin: 8px 0;">
                            ${passed ? '✅' : '❌'} <strong>${filter.replace(/_/g, ' ').toUpperCase()}:</strong> ${passed ? 'PASSED' : 'FAILED'}
                        </div>
                    `;
                });
            } else {
                html += `<div>Confirmation analysis data not available</div>`;
            }
            
            html += `
                    </div>
                    
                    <div class="analysis-section">
                        <h4>💡 Why This Trade ${trade.pnl > 0 ? 'Succeeded' : 'Failed'}</h4>
                        <p><strong>Exit Reason:</strong> ${trade.exit_reason}</p>
                        <p><strong>Final P&L:</strong> $${trade.pnl.toFixed(2)}</p>
                        <p><strong>Duration:</strong> ${trade.duration}</p>
                        ${trade.pnl > 0 ? 
                            '<p style="color: #27ae60;">✅ <strong>Success factors:</strong> Market moved in predicted direction and reached take profit target.</p>' :
                            '<p style="color: #e74c3c;">❌ <strong>Failure analysis:</strong> ' + 
                            (trade.exit_reason === 'stop_loss' ? 'Price moved against position and hit stop loss.' : 
                             trade.exit_reason === 'time_stop' ? 'Position held too long without significant movement.' : 
                             'Position closed due to end of data.') + '</p>'
                        }
                    </div>
                </div>
            `;
            
            content.innerHTML = html;
            modal.style.display = 'block';
        }
        
        function closeUltraDetailedModal() {
            document.getElementById('ultraDetailedModal').style.display = 'none';
        }
        
        function showTab(tabName) {
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tab buttons
            const tabButtons = document.querySelectorAll('.tab-button');
            tabButtons.forEach(button => button.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('ultraDetailedModal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
    </script>
</body>
</html>
        """
        
        # Write HTML file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file
    
    def _generate_enhanced_signal(self, symbol: str, current_price: float,
                                current_candle: pd.Series, features: Dict[str, float],
                                timestamp: datetime) -> Optional[Dict]:
        """Enhanced signal generation with detailed data capture."""
        
        # Capture detailed data for this signal attempt
        detailed_idx = self.capture_ultra_detailed_trade_data(
            'signal_attempt', timestamp, current_candle, features
        )
        
        # Call parent method to get signal
        signal = super()._generate_enhanced_signal(symbol, current_price, current_candle, features, timestamp)
        
        if signal:
            # Capture entry data
            entry_idx = self.capture_ultra_detailed_trade_data(
                'entry', timestamp, current_candle, features, signal_data=signal
            )
            signal['detailed_data_index'] = entry_idx
        
        return signal


def main():
    """Main function to run ultra-detailed strategy analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Detailed Strategy Analysis")
    parser.add_argument('--block', default='august_12_single_day', help='Data block to use')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    
    args = parser.parse_args()
    
    # Run ultra-detailed analysis
    strategy = ImprovedStrategyWithDetailedHTML(initial_balance=args.balance)
    results = strategy.run_enhanced_backtest_with_ultra_detailed_reporting(args.block)
    
    return results

if __name__ == "__main__":
    main()