#!/usr/bin/env python3
"""Analyze strategy performance and identify improvement opportunities."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.final_aggressive_strategy import FinalAggressiveStrategy

class StrategyPerformanceAnalyzer:
    """Analyzer for strategy performance and improvement opportunities."""
    
    def __init__(self):
        self.strategy = FinalAggressiveStrategy()
        
    def analyze_price_movements(self, df_5m: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price movements and identify patterns."""
        
        print("📈 Analyzing price movements...")
        
        # Calculate price changes
        df_5m['price_change'] = df_5m['close'].pct_change()
        df_5m['price_change_abs'] = df_5m['price_change'].abs()
        
        # Identify significant movements (>1%)
        significant_moves = df_5m[df_5m['price_change_abs'] > 0.01].copy()
        
        # Analyze directional moves
        strong_up_moves = df_5m[df_5m['price_change'] > 0.015].copy()  # >1.5% up
        strong_down_moves = df_5m[df_5m['price_change'] < -0.015].copy()  # >1.5% down
        
        # Calculate volatility periods
        df_5m['volatility_20'] = df_5m['price_change'].rolling(20).std()
        high_vol_periods = df_5m[df_5m['volatility_20'] > df_5m['volatility_20'].quantile(0.8)]
        
        # Identify trend periods
        df_5m['ema_8'] = df_5m['close'].ewm(span=8).mean()
        df_5m['ema_20'] = df_5m['close'].ewm(span=20).mean()
        df_5m['trend_bullish'] = df_5m['ema_8'] > df_5m['ema_20']
        
        # Calculate trend strength
        trend_changes = df_5m['trend_bullish'].diff().fillna(0)
        trend_periods = []
        current_trend_start = 0
        
        for i, change in enumerate(trend_changes):
            if change != 0:  # Trend change
                if i > current_trend_start:
                    trend_periods.append({
                        'start': current_trend_start,
                        'end': i,
                        'duration': i - current_trend_start,
                        'bullish': df_5m.iloc[current_trend_start]['trend_bullish'],
                        'price_change': (df_5m.iloc[i]['close'] / df_5m.iloc[current_trend_start]['close'] - 1) * 100
                    })
                current_trend_start = i
        
        analysis = {
            'total_5min_candles': len(df_5m),
            'significant_moves': len(significant_moves),
            'strong_up_moves': len(strong_up_moves),
            'strong_down_moves': len(strong_down_moves),
            'max_5min_gain': df_5m['price_change'].max() * 100,
            'max_5min_loss': df_5m['price_change'].min() * 100,
            'avg_volatility': df_5m['volatility_20'].mean() * 100,
            'high_vol_periods': len(high_vol_periods),
            'trend_periods': trend_periods,
            'price_range': {
                'low': df_5m['low'].min(),
                'high': df_5m['high'].max(),
                'total_range': df_5m['high'].max() - df_5m['low'].min(),
                'range_pct': (df_5m['high'].max() / df_5m['low'].min() - 1) * 100
            }
        }
        
        return analysis
    
    def analyze_failed_trades(self, trades: List[Dict], df_5m: pd.DataFrame) -> Dict[str, Any]:
        """Analyze why trades failed (hit stop loss)."""
        
        print("🔍 Analyzing failed trades...")
        
        failed_trades = [t for t in trades if t.get('exit_reason') == 'stop_loss']
        successful_trades = [t for t in trades if t.get('exit_reason') == 'take_profit']
        
        failed_analysis = []
        
        for trade in failed_trades:
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            
            # Find corresponding candles
            entry_idx = df_5m.index.get_indexer([entry_time], method='nearest')[0]
            exit_idx = df_5m.index.get_indexer([exit_time], method='nearest')[0]
            
            # Analyze what happened during the trade
            trade_candles = df_5m.iloc[entry_idx:exit_idx+1]
            
            if len(trade_candles) > 1:
                # Check if price moved in our direction initially
                if trade['signal_type'] == 'BUY':
                    max_favorable = (trade_candles['high'].max() - trade['entry_price']) / trade['entry_price'] * 100
                    immediate_move = (trade_candles.iloc[1]['close'] - trade['entry_price']) / trade['entry_price'] * 100
                else:
                    max_favorable = (trade['entry_price'] - trade_candles['low'].min()) / trade['entry_price'] * 100
                    immediate_move = (trade['entry_price'] - trade_candles.iloc[1]['close']) / trade['entry_price'] * 100
                
                failed_analysis.append({
                    'trade_index': trades.index(trade),
                    'signal_type': trade['signal_type'],
                    'confidence': trade['confidence'],
                    'duration_minutes': (exit_time - entry_time).total_seconds() / 60,
                    'max_favorable_move': max_favorable,
                    'immediate_move': immediate_move,
                    'was_direction_correct': max_favorable > 0.2,  # Did we get at least 0.2% in our favor?
                    'quick_reversal': abs(immediate_move) > 0.5 and immediate_move * max_favorable < 0,
                    'stop_too_tight': max_favorable > 0.5 and max_favorable < 1.0,
                })
        
        # Analyze successful trades for comparison
        successful_analysis = []
        for trade in successful_trades:
            entry_time = trade['entry_time'] 
            exit_time = trade['exit_time']
            
            entry_idx = df_5m.index.get_indexer([entry_time], method='nearest')[0]
            exit_idx = df_5m.index.get_indexer([exit_time], method='nearest')[0]
            
            trade_candles = df_5m.iloc[entry_idx:exit_idx+1]
            
            if len(trade_candles) > 1:
                if trade['signal_type'] == 'BUY':
                    immediate_move = (trade_candles.iloc[1]['close'] - trade['entry_price']) / trade['entry_price'] * 100
                else:
                    immediate_move = (trade['entry_price'] - trade_candles.iloc[1]['close']) / trade['entry_price'] * 100
                
                successful_analysis.append({
                    'confidence': trade['confidence'],
                    'duration_minutes': (exit_time - entry_time).total_seconds() / 60,
                    'immediate_move': immediate_move,
                    'final_profit': trade['net_pnl']
                })
        
        return {
            'failed_trades': failed_analysis,
            'successful_trades': successful_analysis,
            'failure_patterns': {
                'quick_reversals': sum(1 for t in failed_analysis if t.get('quick_reversal', False)),
                'direction_wrong': sum(1 for t in failed_analysis if not t.get('was_direction_correct', True)),
                'stop_too_tight': sum(1 for t in failed_analysis if t.get('stop_too_tight', False)),
                'avg_max_favorable': np.mean([t['max_favorable_move'] for t in failed_analysis]) if failed_analysis else 0,
                'avg_immediate_move': np.mean([t['immediate_move'] for t in failed_analysis]) if failed_analysis else 0,
            }
        }
    
    def analyze_missed_opportunities(self, df_5m: pd.DataFrame, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze major price movements that were missed."""
        
        print("🎯 Analyzing missed opportunities...")
        
        # Find all major moves (>2% in 5-20 minutes)
        major_moves = []
        
        for i in range(20, len(df_5m)-20):
            current_price = df_5m.iloc[i]['close']
            
            # Check next 4 candles (20 minutes) for significant moves
            for j in range(1, 5):
                if i + j < len(df_5m):
                    future_price = df_5m.iloc[i + j]['close']
                    price_change = (future_price / current_price - 1) * 100
                    
                    if abs(price_change) > 2.0:  # >2% move
                        major_moves.append({
                            'timestamp': df_5m.index[i],
                            'entry_price': current_price,
                            'peak_price': future_price,
                            'move_pct': price_change,
                            'direction': 'UP' if price_change > 0 else 'DOWN',
                            'duration_minutes': j * 5,
                            'was_captured': False
                        })
                        break
        
        # Check which moves were captured by our trades
        for move in major_moves:
            move_time = move['timestamp']
            for trade in trades:
                entry_time = trade['entry_time']
                exit_time = trade['exit_time']
                
                # Check if our trade overlapped with this move
                if (entry_time <= move_time <= exit_time and 
                    ((move['direction'] == 'UP' and trade['signal_type'] == 'BUY') or
                     (move['direction'] == 'DOWN' and trade['signal_type'] == 'SELL'))):
                    move['was_captured'] = True
                    break
        
        captured_moves = [m for m in major_moves if m['was_captured']]
        missed_moves = [m for m in major_moves if not m['was_captured']]
        
        return {
            'total_major_moves': len(major_moves),
            'captured_moves': len(captured_moves),
            'missed_moves': len(missed_moves),
            'capture_rate': len(captured_moves) / len(major_moves) * 100 if major_moves else 0,
            'missed_opportunities': missed_moves[:10],  # Top 10 missed
            'avg_missed_move': np.mean([abs(m['move_pct']) for m in missed_moves]) if missed_moves else 0,
            'total_missed_profit': sum([abs(m['move_pct']) for m in missed_moves]) if missed_moves else 0
        }
    
    def generate_improvement_recommendations(self, price_analysis: Dict, failed_analysis: Dict, 
                                           missed_analysis: Dict) -> List[str]:
        """Generate specific recommendations for strategy improvement."""
        
        recommendations = []
        
        # Analyze stop loss issues
        if failed_analysis['failure_patterns']['stop_too_tight'] > 3:
            recommendations.append(
                "🎯 **Расширить стоп-лоссы**: "
                f"{failed_analysis['failure_patterns']['stop_too_tight']} сделок показали правильное направление, "
                f"но были закрыты слишком тесными стопами. "
                f"Средний максимальный ход в нашу пользу: {failed_analysis['failure_patterns']['avg_max_favorable']:.2f}%. "
                "Рекомендация: увеличить ATR множитель с 0.6 до 1.0-1.2."
            )
        
        # Analyze entry timing
        if failed_analysis['failure_patterns']['quick_reversals'] > 5:
            recommendations.append(
                "⏱️ **Улучшить тайминг входа**: "
                f"{failed_analysis['failure_patterns']['quick_reversals']} сделок показали быстрые развороты. "
                "Рекомендации: добавить фильтр подтверждения (например, ждать 2-3 свечи подряд в нужном направлении) "
                "или использовать pullback стратегию вместо breakout."
            )
        
        # Analyze missed opportunities
        if missed_analysis['capture_rate'] < 30:
            recommendations.append(
                f"📈 **Повысить охват движений**: Захвачено только {missed_analysis['capture_rate']:.1f}% "
                f"крупных движений. Упущено {missed_analysis['missed_moves']} движений "
                f"средней величиной {missed_analysis['avg_missed_move']:.1f}%. "
                "Рекомендации: снизить порог уверенности с 3% до 1-2% или добавить momentum фильтры."
            )
        
        # Analyze volatility periods
        if price_analysis['high_vol_periods'] > 50:
            recommendations.append(
                "🌪️ **Оптимизация для волатильности**: "
                f"Обнаружено {price_analysis['high_vol_periods']} периодов высокой волатильности. "
                "Рекомендации: адаптивные стоп-лоссы (больше в волатильные периоды), "
                "или специальные параметры для высоковолатильных условий."
            )
        
        # Analyze trend following
        strong_trends = [t for t in price_analysis['trend_periods'] if abs(t['price_change']) > 3]
        if len(strong_trends) > 2:
            recommendations.append(
                f"📊 **Улучшить trend following**: Найдено {len(strong_trends)} сильных трендов. "
                "Рекомендации: добавить trailing stop для прибыльных позиций, "
                "увеличить time-to-profit в трендовых условиях, "
                "или использовать пирамидинг в сильных трендах."
            )
        
        # Analyze confidence vs success rate
        if len(failed_analysis['successful_trades']) > 0:
            avg_successful_confidence = np.mean([t['confidence'] for t in failed_analysis['successful_trades']])
            avg_failed_confidence = np.mean([t['confidence'] for t in failed_analysis['failed_trades']])
            
            if avg_successful_confidence > avg_failed_confidence + 0.1:
                recommendations.append(
                    f"🎯 **Повысить порог уверенности**: "
                    f"Успешные сделки имели среднюю уверенность {avg_successful_confidence:.2f}, "
                    f"неудачные - {avg_failed_confidence:.2f}. "
                    "Рекомендация: поднять минимальный порог уверенности до 5-10%."
                )
        
        # Position sizing recommendations
        if len(missed_analysis['missed_opportunities']) > 0:
            large_missed = [m for m in missed_analysis['missed_opportunities'] if abs(m['move_pct']) > 4]
            if len(large_missed) > 2:
                recommendations.append(
                    f"💰 **Оптимизировать размер позиций**: "
                    f"Упущено {len(large_missed)} движений >4%. "
                    "Рекомендации: увеличить размер позиции для высокоуверенных сигналов (>70%), "
                    "или добавить дополнительные входы при подтверждении движения."
                )
        
        return recommendations
    
    def run_full_analysis(self, block_id: str = "august_12_single_day"):
        """Run complete performance analysis."""
        
        print("🔬 Starting comprehensive strategy analysis...")
        print("=" * 70)
        
        # Load data and run strategy
        data = self.strategy.load_data_from_block(block_id)
        df_5m = data['5m'].copy()
        
        # Run strategy to get trades
        results = self.strategy.run_enhanced_backtest_with_momentum(block_id)
        trades = results.get('trades', [])
        
        if not trades:
            print("❌ No trades found to analyze")
            return
        
        # Perform analyses
        price_analysis = self.analyze_price_movements(df_5m)
        failed_analysis = self.analyze_failed_trades(trades, df_5m)
        missed_analysis = self.analyze_missed_opportunities(df_5m, trades)
        
        # Generate recommendations
        recommendations = self.generate_improvement_recommendations(
            price_analysis, failed_analysis, missed_analysis
        )
        
        # Create comprehensive report
        self.generate_analysis_report(
            price_analysis, failed_analysis, missed_analysis, 
            recommendations, trades, block_id
        )
    
    def generate_analysis_report(self, price_analysis: Dict, failed_analysis: Dict,
                               missed_analysis: Dict, recommendations: List[str],
                               trades: List[Dict], block_id: str):
        """Generate detailed analysis report."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"Output/Strategy_Analysis_Report_{block_id}_{timestamp}.md"
        
        report_content = f"""# 🔬 Strategy Performance Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Data Block:** {block_id}  
**Total Trades:** {len(trades)}

## 📊 Executive Summary

**Current Performance:**
- Win Rate: {len([t for t in trades if t.get('net_pnl', 0) > 0]) / len(trades) * 100:.1f}%
- Major Moves Captured: {missed_analysis['capture_rate']:.1f}%
- Failed Trades: {len(failed_analysis['failed_trades'])}
- Successful Trades: {len(failed_analysis['successful_trades'])}

## 📈 Price Movement Analysis

**Market Characteristics:**
- Total 5-min candles: {price_analysis['total_5min_candles']}
- Significant moves (>1%): {price_analysis['significant_moves']}
- Strong up moves (>1.5%): {price_analysis['strong_up_moves']}
- Strong down moves (>1.5%): {price_analysis['strong_down_moves']}
- Max 5-min gain: {price_analysis['max_5min_gain']:.2f}%
- Max 5-min loss: {price_analysis['max_5min_loss']:.2f}%
- Average volatility: {price_analysis['avg_volatility']:.2f}%
- High volatility periods: {price_analysis['high_vol_periods']}
- Total price range: {price_analysis['price_range']['range_pct']:.2f}%

## ❌ Failed Trades Analysis

**Failure Patterns:**
- Quick reversals: {failed_analysis['failure_patterns']['quick_reversals']}
- Wrong direction: {failed_analysis['failure_patterns']['direction_wrong']}
- Stop too tight: {failed_analysis['failure_patterns']['stop_too_tight']}
- Avg max favorable move: {failed_analysis['failure_patterns']['avg_max_favorable']:.2f}%
- Avg immediate move: {failed_analysis['failure_patterns']['avg_immediate_move']:.2f}%

**Key Issues:**
"""

        # Add specific failed trade analysis
        if failed_analysis['failure_patterns']['stop_too_tight'] > 0:
            report_content += f"- **{failed_analysis['failure_patterns']['stop_too_tight']} сделок** были правильными по направлению, но стоп-лоссы слишком тесные\n"
        
        if failed_analysis['failure_patterns']['quick_reversals'] > 0:
            report_content += f"- **{failed_analysis['failure_patterns']['quick_reversals']} сделок** показали быстрые развороты против нас\n"
        
        if failed_analysis['failure_patterns']['direction_wrong'] > 0:
            report_content += f"- **{failed_analysis['failure_patterns']['direction_wrong']} сделок** были неправильными по направлению\n"

        report_content += f"""

## 🎯 Missed Opportunities Analysis

**Opportunity Capture:**
- Total major moves (>2%): {missed_analysis['total_major_moves']}
- Captured: {missed_analysis['captured_moves']} ({missed_analysis['capture_rate']:.1f}%)
- Missed: {missed_analysis['missed_moves']}
- Avg missed move size: {missed_analysis['avg_missed_move']:.2f}%
- Total missed profit potential: {missed_analysis['total_missed_profit']:.1f}%

**Top Missed Opportunities:**
"""

        for i, missed in enumerate(missed_analysis['missed_opportunities'][:5], 1):
            report_content += f"{i}. {missed['timestamp'].strftime('%H:%M')} - {missed['direction']} {missed['move_pct']:+.2f}% in {missed['duration_minutes']}min\n"

        report_content += f"""

## 💡 Improvement Recommendations

"""

        for i, rec in enumerate(recommendations, 1):
            report_content += f"{i}. {rec}\n\n"

        report_content += f"""

## 🔧 Specific Parameter Suggestions

**Based on this analysis:**

### Stop Loss Optimization
- Current: ATR * 0.6
- Suggested: ATR * 1.0-1.2 (reduces tight stop failures)

### Confidence Threshold
- Current: 3%
- Suggested: 5-7% (improves win rate) OR 1-2% (captures more moves)

### Take Profit Strategy
- Current: ATR * 4.0
- Suggested: Add trailing stop at 50% of TP target

### Entry Confirmation
- Add: Wait for 2 consecutive candles in signal direction
- Add: Volume confirmation (>1.5x average)
- Add: Momentum filter (price change >0.3% in signal direction)

### Position Management
- High confidence (>70%): 3% position size
- Medium confidence (40-70%): 2% position size  
- Low confidence (<40%): 1% position size

## 📋 Next Steps

1. **Immediate Fixes:**
   - Adjust stop loss multiplier to 1.0
   - Add entry confirmation filter
   - Implement adaptive position sizing

2. **Testing Required:**
   - Backtest with new parameters on multiple blocks
   - Compare performance metrics
   - Validate on different market conditions

3. **Long-term Improvements:**
   - Machine learning for dynamic stop losses
   - Market regime detection
   - Advanced momentum filters

---

*Report generated by Strategy Performance Analyzer*  
*Next recommended action: Implement parameter adjustments and retest*
"""

        # Write report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📄 Analysis report generated: {report_file}")
        
        # Print summary to console
        print("\n" + "="*70)
        print("🎯 KEY FINDINGS & RECOMMENDATIONS")
        print("="*70)
        
        for i, rec in enumerate(recommendations[:3], 1):  # Top 3 recommendations
            print(f"{i}. {rec}")
        
        print(f"\n📊 QUICK STATS:")
        print(f"• Win Rate: {len([t for t in trades if t.get('net_pnl', 0) > 0]) / len(trades) * 100:.1f}%")
        print(f"• Captured Major Moves: {missed_analysis['capture_rate']:.1f}%")
        print(f"• Stop Too Tight Issues: {failed_analysis['failure_patterns']['stop_too_tight']} trades")
        print(f"• Quick Reversals: {failed_analysis['failure_patterns']['quick_reversals']} trades")
        
        return report_file


def main():
    """Main function to run strategy analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Strategy Performance Analysis")
    parser.add_argument('--block', default='august_12_single_day', help='Data block to analyze')
    
    args = parser.parse_args()
    
    analyzer = StrategyPerformanceAnalyzer()
    analyzer.run_full_analysis(args.block)

if __name__ == "__main__":
    main()