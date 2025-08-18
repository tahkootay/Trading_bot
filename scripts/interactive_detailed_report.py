#!/usr/bin/env python3
"""Interactive detailed report with clickable trades showing full decision factors."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.final_aggressive_strategy import FinalAggressiveStrategy

class InteractiveDetailedReport(FinalAggressiveStrategy):
    """Enhanced strategy with interactive detailed reporting."""
    
    def __init__(self, initial_balance: float = 10000.0):
        super().__init__(initial_balance)
        self.detailed_decisions = []  # Store detailed decision data for each signal
        self.market_context = {}  # Store market context for each timestamp
    
    def generate_human_readable_explanation(self, decision: str, confidence: float, 
                                          indicators: Dict, conditions: Dict, 
                                          market_context: Dict) -> str:
        """Generate human-readable explanation for decision."""
        
        if decision == 'HOLD':
            return "Сделка не была открыта, так как не было достаточных сигналов для входа в позицию."
        
        explanation_parts = []
        
        # Main decision reason
        if decision == 'BUY':
            explanation_parts.append(f"🟢 ПОКУПКА была открыта (уверенность {confidence*100:.1f}%) потому что:")
        else:
            explanation_parts.append(f"🔴 ПРОДАЖА была открыта (уверенность {confidence*100:.1f}%) потому что:")
        
        # Trend analysis
        if decision == 'BUY':
            if conditions.get('trend_bullish', False):
                explanation_parts.append("  📈 Восходящий тренд: EMA8 > EMA20 > EMA50 - цена движется вверх")
            if conditions.get('price_above_vwap', False):
                explanation_parts.append("  🎯 Цена выше VWAP - институциональные покупки")
        else:
            if conditions.get('trend_bearish', False):
                explanation_parts.append("  📉 Нисходящий тренд: EMA8 < EMA20 < EMA50 - цена движется вниз")
            if not conditions.get('price_above_vwap', True):
                explanation_parts.append("  🎯 Цена ниже VWAP - институциональные продажи")
        
        # Volume analysis
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 2.0:
            explanation_parts.append(f"  🔊 Очень высокий объём ({volume_ratio:.1f}x) - сильный интерес")
        elif volume_ratio > 1.2:
            explanation_parts.append(f"  📊 Повышенный объём ({volume_ratio:.1f}x) - подтверждение движения")
        
        # Momentum analysis
        if conditions.get('momentum_strong', False):
            explanation_parts.append("  ⚡ Сильный моментум по MACD - импульсное движение")
        
        # RSI conditions
        rsi = indicators.get('rsi', 50)
        if decision == 'BUY' and conditions.get('rsi_oversold', False):
            explanation_parts.append(f"  📉➡️📈 RSI {rsi:.1f} - актив перепродан, возможен отскок")
        elif decision == 'SELL' and conditions.get('rsi_overbought', False):
            explanation_parts.append(f"  📈➡️📉 RSI {rsi:.1f} - актив перекуплен, возможна коррекция")
        
        # Volatility context
        volatility = market_context.get('volatility', 0)
        if volatility > 2.0:
            explanation_parts.append(f"  🌪️ Высокая волатильность {volatility:.1f}% - благоприятно для скальпинга")
        
        # Momentum context
        momentum_pct = market_context.get('momentum', 0)
        if abs(momentum_pct) > 0.5:
            direction = "вверх" if momentum_pct > 0 else "вниз"
            explanation_parts.append(f"  🚀 Сильное движение цены {momentum_pct:+.2f}% {direction} за последние 5 минут")
        
        # Volume surge
        if market_context.get('volume_surge', False):
            explanation_parts.append("  💥 Всплеск объёма - крупные игроки активны")
        
        # Position in Bollinger Bands
        bb_position = market_context.get('price_position', 'middle')
        if decision == 'BUY' and bb_position == 'lower':
            explanation_parts.append("  📊 Цена у нижней границы Боллинджера - потенциал роста")
        elif decision == 'SELL' and bb_position == 'upper':
            explanation_parts.append("  📊 Цена у верхней границы Боллинджера - потенциал падения")
        
        # Trend strength
        trend_strength = market_context.get('trend_strength', 20)
        if trend_strength > 25:
            explanation_parts.append(f"  💪 Сильный тренд (ADX {trend_strength:.0f}) - высокая вероятность продолжения")
        
        # Risk factors (if any)
        risk_factors = []
        if volume_ratio < 0.8:
            risk_factors.append("низкий объём")
        if volatility < 1.0:
            risk_factors.append("низкая волатильность")
        
        if risk_factors:
            explanation_parts.append(f"  ⚠️ Факторы риска: {', '.join(risk_factors)}")
        
        return "\n".join(explanation_parts)

    def capture_detailed_decision(self, timestamp, candle_data: Dict, decision: str, 
                                 confidence: float, indicators: Dict, reasoning: str):
        """Capture detailed decision data for interactive report."""
        
        # Calculate market conditions
        conditions = {
            'trend_bullish': indicators.get('ema_8', 0) > indicators.get('ema_20', 0) > indicators.get('ema_50', 0),
            'trend_bearish': indicators.get('ema_8', 0) < indicators.get('ema_20', 0) < indicators.get('ema_50', 0),
            'price_above_vwap': candle_data['close'] > indicators.get('vwap', candle_data['close']),
            'volume_high': indicators.get('volume_ratio', 0) > 1.0,
            'momentum_strong': abs(indicators.get('macd_histogram', 0)) > 0.1,
            'rsi_oversold': indicators.get('rsi', 50) < 30,
            'rsi_overbought': indicators.get('rsi', 50) > 70,
            'atr_high': indicators.get('atr', 0) / candle_data['close'] > 0.01,
        }
        
        # Calculate market context
        market_context = {
            'volatility': indicators.get('atr', 0) / candle_data['close'] * 100,
            'trend_strength': indicators.get('adx', 0),
            'momentum': (candle_data['close'] - candle_data.get('open', candle_data['close'])) / candle_data.get('open', candle_data['close']) * 100,
            'volume_surge': indicators.get('volume_ratio', 1.0) > 2.0,
            'price_position': 'upper' if candle_data['close'] > indicators.get('bb_middle', candle_data['close']) else 'lower',
        }
        
        # Generate human-readable explanation
        human_explanation = self.generate_human_readable_explanation(
            decision, confidence, indicators, conditions, market_context
        )
        
        decision_data = {
            'timestamp': timestamp,
            'decision': decision,
            'confidence': confidence,
            'reasoning': reasoning,
            'human_explanation': human_explanation,
            'price': candle_data['close'],
            'volume': candle_data['volume'],
            'indicators': {
                'ema_8': indicators.get('ema_8', 0),
                'ema_20': indicators.get('ema_20', 0),
                'ema_50': indicators.get('ema_50', 0),
                'rsi': indicators.get('rsi', 0),
                'macd': indicators.get('macd', 0),
                'macd_signal': indicators.get('macd_signal', 0),
                'macd_histogram': indicators.get('macd_histogram', 0),
                'adx': indicators.get('adx', 0),
                'atr': indicators.get('atr', 0),
                'volume_ratio': indicators.get('volume_ratio', 0),
                'bb_upper': indicators.get('bb_upper', 0),
                'bb_middle': indicators.get('bb_middle', 0),
                'bb_lower': indicators.get('bb_lower', 0),
                'vwap': indicators.get('vwap', 0),
            },
            'conditions': conditions,
            'market_context': market_context
        }
        
        self.detailed_decisions.append(decision_data)
    
    def enhanced_signal_generation_with_details(self, df_5m: pd.DataFrame, df_1m: pd.DataFrame) -> List[Dict]:
        """Generate signals with detailed decision capture."""
        
        print("🔍 Enhanced signal generation with detailed capture...")
        signals = []
        
        for i in range(50, len(df_5m)):  # Skip initial candles for indicators
            current_candle = df_5m.iloc[i]
            
            if hasattr(current_candle, 'name') and isinstance(current_candle.name, datetime):
                current_time = current_candle.name
            elif 'timestamp' in current_candle.index:
                current_time = current_candle['timestamp']
            else:
                current_time = df_5m.index[i]
            
            # Get technical indicators
            indicators = {
                'ema_8': current_candle.get('ema_8', 0),
                'ema_20': current_candle.get('ema_20', 0),
                'ema_50': current_candle.get('ema_50', 0),
                'rsi': current_candle.get('rsi', 50),
                'macd': current_candle.get('macd', 0),
                'macd_signal': current_candle.get('macd_signal', 0),
                'macd_histogram': current_candle.get('macd_histogram', 0),
                'adx': current_candle.get('adx', 0),
                'atr': current_candle.get('atr', 0),
                'volume_ratio': current_candle.get('volume_ratio', 1.0),
                'bb_upper': current_candle.get('bb_upper', current_candle['close']),
                'bb_middle': current_candle.get('bb_middle', current_candle['close']),
                'bb_lower': current_candle.get('bb_lower', current_candle['close']),
                'vwap': current_candle.get('vwap', current_candle['close']),
            }
            
            # Calculate confidence scores
            bullish_factors = 0
            bearish_factors = 0
            
            # Trend analysis
            if indicators['ema_8'] > indicators['ema_20'] > indicators['ema_50']:
                bullish_factors += 0.20
            elif indicators['ema_8'] < indicators['ema_20'] < indicators['ema_50']:
                bearish_factors += 0.20
            
            # Price vs VWAP
            if current_candle['close'] > indicators['vwap']:
                bullish_factors += 0.15
            else:
                bearish_factors += 0.15
            
            # MACD momentum
            if indicators['macd'] > indicators['macd_signal'] and indicators['macd_histogram'] > 0:
                bullish_factors += 0.15
            elif indicators['macd'] < indicators['macd_signal'] and indicators['macd_histogram'] < 0:
                bearish_factors += 0.15
            
            # Volume confirmation
            if indicators['volume_ratio'] > 1.2:
                if bullish_factors > bearish_factors:
                    bullish_factors += 0.10
                else:
                    bearish_factors += 0.10
            
            # RSI conditions
            if indicators['rsi'] < 40 and bullish_factors > 0.15:
                bullish_factors += 0.10
            elif indicators['rsi'] > 60 and bearish_factors > 0.15:
                bearish_factors += 0.10
            
            # Price momentum (5min change)
            if i > 0:
                momentum = (current_candle['close'] - df_5m.iloc[i-1]['close']) / df_5m.iloc[i-1]['close']
                if momentum > 0.006:  # Strong upward momentum
                    bullish_factors += 0.20
                elif momentum < -0.006:  # Strong downward momentum
                    bearish_factors += 0.20
            
            # Determine signal type and confidence
            bullish_confidence = min(bullish_factors, 1.0)
            bearish_confidence = min(bearish_factors, 1.0)
            
            signal_type = None
            confidence = 0
            reasoning = ""
            
            if bullish_confidence > bearish_confidence + 0.05 and bullish_confidence >= self.strategy_params['min_confidence']:
                signal_type = 'BUY'
                confidence = bullish_confidence
                reasoning = f"BUY: bull={bullish_confidence:.2f}, bear={bearish_confidence:.2f}, vol={indicators['volume_ratio']:.1f}x"
            elif bearish_confidence > bullish_confidence + 0.05 and bearish_confidence >= self.strategy_params['min_confidence']:
                signal_type = 'SELL'
                confidence = bearish_confidence  
                reasoning = f"SELL: bull={bullish_confidence:.2f}, bear={bearish_confidence:.2f}, vol={indicators['volume_ratio']:.1f}x"
            else:
                reasoning = f"NO SIGNAL: bull={bullish_confidence:.2f}, bear={bearish_confidence:.2f}, insufficient confidence"
            
            # Capture detailed decision data
            candle_data = {
                'open': current_candle['open'],
                'high': current_candle['high'], 
                'low': current_candle['low'],
                'close': current_candle['close'],
                'volume': current_candle['volume']
            }
            
            decision = signal_type if signal_type else 'HOLD'
            self.capture_detailed_decision(
                current_time, candle_data, decision, 
                confidence, indicators, reasoning
            )
            
            if signal_type and confidence >= self.strategy_params['min_confidence']:
                signals.append({
                    'timestamp': current_time,
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'price': current_candle['close'],
                    'reasoning': reasoning,
                    'volume_ratio': indicators['volume_ratio'],
                    'decision_index': len(self.detailed_decisions) - 1  # Link to detailed decision
                })
        
        return signals
    
    def _add_basic_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to dataframe."""
        
        # EMAs
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()  
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume ratio
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ATR  
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        
        # ADX (simplified)
        df['adx'] = 20.0  # Simplified for now
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # VWAP (simplified)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df
    
    def run_detailed_backtest_with_decisions(self, block_id: str = "august_12_single_day"):
        """Run backtest capturing detailed decisions."""
        
        print(f"🚀 Interactive Detailed Strategy Report")
        print(f"📦 Block: {block_id}")
        print("=" * 60)
        
        # Load data
        data = self.load_data_from_block(block_id)
        df_5m = data['5m'].copy()
        df_1m = data['1m'].copy()
        
        # Add technical indicators
        df_5m = self._add_basic_technical_indicators(df_5m)
        
        # Generate signals with detailed capture
        signals = self.enhanced_signal_generation_with_details(df_5m, df_1m)
        
        # Run trading simulation
        trades = []
        balance = self.initial_balance
        current_position = None
        
        for signal in signals:
            if current_position is None:
                # Enter position
                entry_price = signal['price']
                quantity = (balance * self.strategy_params['position_size_pct']) / entry_price
                
                # Calculate stop loss and take profit
                atr_value = df_5m.loc[df_5m.index <= signal['timestamp']]['atr'].iloc[-1]
                if pd.isna(atr_value):
                    atr_value = entry_price * 0.01
                
                if signal['signal_type'] == 'BUY':
                    stop_loss = entry_price - (atr_value * self.strategy_params['atr_sl_multiplier'])
                    take_profit = entry_price + (atr_value * self.strategy_params['atr_tp_multiplier'])
                else:
                    stop_loss = entry_price + (atr_value * self.strategy_params['atr_sl_multiplier']) 
                    take_profit = entry_price - (atr_value * self.strategy_params['atr_tp_multiplier'])
                
                current_position = {
                    'signal_type': signal['signal_type'],
                    'entry_time': signal['timestamp'],
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': signal['confidence'],
                    'reasoning': signal['reasoning'],
                    'entry_decision_index': signal['decision_index']
                }
                
                continue
            
            # Check for exit conditions
            current_price = signal['price']
            
            exit_reason = None
            if signal['signal_type'] == 'BUY' and current_position['signal_type'] == 'BUY':
                if current_price <= current_position['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price >= current_position['take_profit']:
                    exit_reason = 'take_profit'
            elif signal['signal_type'] == 'SELL' and current_position['signal_type'] == 'SELL':
                if current_price >= current_position['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price <= current_position['take_profit']:
                    exit_reason = 'take_profit'
            
            if exit_reason:
                # Close position
                if current_position['signal_type'] == 'BUY':
                    pnl = (current_price - current_position['entry_price']) * current_position['quantity']
                else:
                    pnl = (current_position['entry_price'] - current_price) * current_position['quantity']
                
                commission = (current_position['entry_price'] * current_position['quantity'] * 0.0006 + 
                             current_price * current_position['quantity'] * 0.0006)
                net_pnl = pnl - commission
                
                balance += net_pnl
                
                trade = {
                    'signal_type': current_position['signal_type'],
                    'entry_time': current_position['entry_time'],
                    'exit_time': signal['timestamp'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': current_price,
                    'quantity': current_position['quantity'],
                    'pnl': pnl,
                    'net_pnl': net_pnl,
                    'commission': commission,
                    'exit_reason': exit_reason,
                    'confidence': current_position['confidence'],
                    'reasoning': current_position['reasoning'],
                    'entry_decision_index': current_position['entry_decision_index'],
                    'exit_decision_index': signal['decision_index']
                }
                
                trades.append(trade)
                current_position = None
        
        results = {
            'trades': trades,
            'detailed_decisions': self.detailed_decisions,
            'signals_generated': len(signals),
            'final_balance': balance,
            'total_return': (balance / self.initial_balance - 1) * 100,
            'initial_balance': self.initial_balance
        }
        
        # Generate interactive HTML report
        html_file = self.generate_interactive_html_report(results, block_id)
        print(f"\n📄 Interactive HTML Report: {html_file}")
        
        return results
    
    def generate_interactive_html_report(self, results: Dict[str, Any], block_id: str) -> str:
        """Generate interactive HTML report with clickable trades."""
        
        trades = results['trades']
        decisions = results['detailed_decisions']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_file = f"Output/Interactive_Detailed_Report_{block_id}_{timestamp}.html"
        
        # Calculate metrics
        winning_trades = [t for t in trades if t.get('net_pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('net_pnl', 0) < 0]
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Trading Analysis - {block_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
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
        }}
        .trades-table tr:hover {{
            background-color: #f0f8ff;
            cursor: pointer;
        }}
        .profit {{
            color: #27ae60;
            font-weight: bold;
        }}
        .loss {{
            color: #e74c3c;
            font-weight: bold;
        }}
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
            margin: 2% auto;
            padding: 20px;
            border-radius: 10px;
            width: 90%;
            max-width: 1200px;
            max-height: 90vh;
            overflow-y: auto;
        }}
        .close {{
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{
            color: black;
        }}
        .decision-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .decision-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
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
        }}
        .condition-true {{
            color: #27ae60;
            font-weight: bold;
        }}
        .condition-false {{
            color: #e74c3c;
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
        .confidence-bar {{
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }}
        .confidence-fill {{
            height: 100%;
            background-color: #3498db;
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Interactive Trading Analysis Report</h1>
        
        <div class="summary-stats">
            <div class="stat-card">
                <h3>Total Trades</h3>
                <div style="font-size: 24px; font-weight: bold;">{len(trades)}</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #27ae60 0%, #229954 100%);">
                <h3>Winning Trades</h3>
                <div style="font-size: 24px; font-weight: bold;">{len(winning_trades)}</div>
                <div style="font-size: 14px; opacity: 0.9;">{len(winning_trades) / len(trades) * 100:.1f}%</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);">
                <h3>Losing Trades</h3>
                <div style="font-size: 24px; font-weight: bold;">{len(losing_trades)}</div>
                <div style="font-size: 14px; opacity: 0.9;">{len(losing_trades) / len(trades) * 100:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Total Return</h3>
                <div style="font-size: 24px; font-weight: bold;">{results['total_return']:+.2f}%</div>
            </div>
            <div class="stat-card">
                <h3>Signals Generated</h3>
                <div style="font-size: 24px; font-weight: bold;">{results['signals_generated']}</div>
            </div>
        </div>
        
        <h2>💼 Trades (Click for Details)</h2>
        <p><em>Кликните на любую сделку для подробного анализа факторов принятия решения</em></p>
        
        <table class="trades-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Type</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>P&L</th>
                    <th>Exit Reason</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add trade rows with click handlers
        for i, trade in enumerate(trades, 1):
            pnl = trade.get('net_pnl', 0)
            pnl_class = 'profit' if pnl > 0 else 'loss'
            
            # Get entry and exit decision data
            entry_decision = decisions[trade['entry_decision_index']]
            exit_decision = decisions[trade['exit_decision_index']] if 'exit_decision_index' in trade else None
            
            # Store decision data in JavaScript
            decision_data = {
                'entry': {
                    'timestamp': entry_decision['timestamp'].isoformat() if hasattr(entry_decision['timestamp'], 'isoformat') else str(entry_decision['timestamp']),
                    'decision': entry_decision['decision'],
                    'confidence': entry_decision['confidence'],
                    'reasoning': entry_decision['reasoning'],
                    'human_explanation': entry_decision['human_explanation'],
                    'price': entry_decision['price'],
                    'indicators': entry_decision['indicators'],
                    'conditions': entry_decision['conditions'],
                    'market_context': entry_decision['market_context']
                },
                'exit': exit_decision if exit_decision else None,
                'trade': {
                    'type': trade['signal_type'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'pnl': pnl,
                    'exit_reason': trade['exit_reason'],
                    'duration': str(trade['exit_time'] - trade['entry_time']) if hasattr(trade['exit_time'], '__sub__') else 'N/A'
                }
            }
            
            html_content += f"""
                <tr onclick="showTradeDetails({i}, {json.dumps(decision_data, default=str).replace('"', '&quot;')})">
                    <td>{i}</td>
                    <td>{trade['entry_time'].strftime('%H:%M') if hasattr(trade['entry_time'], 'strftime') else trade['entry_time']}</td>
                    <td>{trade['exit_time'].strftime('%H:%M') if hasattr(trade['exit_time'], 'strftime') else trade['exit_time']}</td>
                    <td>{trade['signal_type']}</td>
                    <td>${trade['entry_price']:.2f}</td>
                    <td>${trade['exit_price']:.2f}</td>
                    <td class="{pnl_class}">${pnl:+.2f}</td>
                    <td>{trade['exit_reason']}</td>
                    <td>{trade['confidence']:.2f}</td>
                </tr>
            """

        html_content += """
            </tbody>
        </table>
    </div>
    
    <!-- Modal for trade details -->
    <div id="tradeModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <div id="modalContent">
                <!-- Trade details will be loaded here -->
            </div>
        </div>
    </div>
    
    <script>
        function showTradeDetails(tradeNumber, data) {
            const modal = document.getElementById('tradeModal');
            const content = document.getElementById('modalContent');
            
            const entry = data.entry;
            const trade = data.trade;
            
            let html = `
                <h2>🔍 Trade #${tradeNumber} - Detailed Analysis</h2>
                
                <div class="decision-grid">
                    <div class="decision-card">
                        <h3>📊 Trade Summary</h3>
                        <p><strong>Type:</strong> ${trade.type}</p>
                        <p><strong>Entry:</strong> $${trade.entry_price.toFixed(2)}</p>
                        <p><strong>Exit:</strong> $${trade.exit_price.toFixed(2)}</p>
                        <p><strong>P&L:</strong> <span class="${trade.pnl > 0 ? 'profit' : 'loss'}">$${trade.pnl.toFixed(2)}</span></p>
                        <p><strong>Exit Reason:</strong> ${trade.exit_reason}</p>
                        <p><strong>Duration:</strong> ${trade.duration}</p>
                    </div>
                    
                    <div class="decision-card">
                        <h3>🎯 Entry Decision</h3>
                        <p><strong>Time:</strong> ${new Date(entry.timestamp).toLocaleString()}</p>
                        <p><strong>Decision:</strong> ${entry.decision}</p>
                        <p><strong>Confidence:</strong> ${(entry.confidence * 100).toFixed(1)}%</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${entry.confidence * 100}%"></div>
                        </div>
                        <p><strong>Reasoning:</strong> ${entry.reasoning}</p>
                    </div>
                </div>
                
                <div class="decision-card" style="grid-column: 1 / -1;">
                    <h3>🤖 Подробное объяснение решения</h3>
                    <div style="white-space: pre-line; font-size: 14px; line-height: 1.6;">
                        ${entry.human_explanation}
                    </div>
                </div>
                
                <h3>📈 Technical Indicators at Entry</h3>
                <div class="indicator-grid">
            `;
            
            // Add indicators
            Object.entries(entry.indicators).forEach(([key, value]) => {
                html += `
                    <div class="indicator-item">
                        <strong>${key.toUpperCase()}:</strong> ${typeof value === 'number' ? value.toFixed(3) : value}
                    </div>
                `;
            });
            
            html += `
                </div>
                
                <h3>✅ Market Conditions</h3>
                <div class="indicator-grid">
            `;
            
            // Add conditions
            Object.entries(entry.conditions).forEach(([key, value]) => {
                const className = value ? 'condition-true' : 'condition-false';
                const icon = value ? '✅' : '❌';
                html += `
                    <div class="indicator-item">
                        <span class="${className}"><strong>${key.replace(/_/g, ' ').toUpperCase()}:</strong> ${icon}</span>
                    </div>
                `;
            });
            
            html += `
                </div>
                
                <h3>🌍 Market Context</h3>
                <div class="indicator-grid">
            `;
            
            // Add market context
            Object.entries(entry.market_context).forEach(([key, value]) => {
                html += `
                    <div class="indicator-item">
                        <strong>${key.replace(/_/g, ' ').toUpperCase()}:</strong> ${typeof value === 'number' ? value.toFixed(3) : value}
                    </div>
                `;
            });
            
            html += `</div>`;
            
            content.innerHTML = html;
            modal.style.display = 'block';
        }
        
        function closeModal() {
            document.getElementById('tradeModal').style.display = 'none';
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('tradeModal');
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


def main():
    """Main function to run interactive detailed report."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Detailed Trading Report")
    parser.add_argument('--block', default='august_12_single_day', help='Data block to use')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    
    args = parser.parse_args()
    
    # Generate interactive report
    reporter = InteractiveDetailedReport(initial_balance=args.balance)
    results = reporter.run_detailed_backtest_with_decisions(args.block)
    
    print(f"\n🎯 Interactive report completed!")
    print(f"📊 Captured {len(results['detailed_decisions'])} decision points")
    print(f"💼 Generated {len(results['trades'])} detailed trades")

if __name__ == "__main__":
    main()