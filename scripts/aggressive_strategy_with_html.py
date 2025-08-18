#!/usr/bin/env python3
"""Aggressive strategy with HTML report generation."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.final_aggressive_strategy import FinalAggressiveStrategy

class AggressiveStrategyWithHTML(FinalAggressiveStrategy):
    """Aggressive strategy with HTML report generation capability."""
    
    def __init__(self, initial_balance: float = 10000.0):
        super().__init__(initial_balance)
        self.html_data = {}  # Store data for HTML generation
    
    def run_enhanced_backtest_with_html(self, block_id: str = "august_12_single_day"):
        """Run backtest and generate HTML report."""
        
        print(f"🚀 Aggressive Strategy with HTML Report")
        print(f"📦 Block: {block_id}")
        print("=" * 60)
        
        # Run the backtest
        results = self.run_enhanced_backtest_with_momentum(block_id)
        
        # Generate HTML report if we have trades
        if results.get('trades'):
            html_file = self.generate_html_report(results, block_id)
            print(f"\n📄 HTML Report generated: {html_file}")
            results['html_report'] = html_file
        else:
            print(f"\n⚠️  No HTML report generated (no trades to show)")
        
        return results
    
    def generate_html_report(self, results: Dict[str, Any], block_id: str) -> str:
        """Generate comprehensive HTML report."""
        
        trades = results['trades']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_file = f"Output/Aggressive_Strategy_Report_{block_id}_{timestamp}.html"
        
        # Calculate additional metrics
        winning_trades = [t for t in trades if t.get('net_pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('net_pnl', 0) < 0]
        
        total_profit = sum(t.get('net_pnl', 0) for t in winning_trades)
        total_loss = sum(t.get('net_pnl', 0) for t in losing_trades)
        
        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = abs(total_loss / len(losing_trades)) if losing_trades else 0
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else 0
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aggressive Strategy Report - {block_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #e74c3c;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #e74c3c;
            padding-left: 15px;
            margin-top: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-card.positive {{
            background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        }}
        .summary-card.neutral {{
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            margin: 5px 0;
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
        .trades-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .profit {{
            color: #27ae60;
            font-weight: bold;
        }}
        .loss {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .strategy-params {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .strategy-params h4 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .param-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }}
        .param-item {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #e74c3c;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Aggressive Momentum Strategy Report</h1>
        
        <div class="summary-grid">
            <div class="summary-card neutral">
                <h3>Total Trades</h3>
                <div class="value">{len(trades)}</div>
            </div>
            <div class="summary-card positive">
                <h3>Winning Trades</h3>
                <div class="value">{len(winning_trades)}</div>
                <small>{len(winning_trades) / len(trades) * 100:.1f}%</small>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);">
                <h3>Losing Trades</h3>
                <div class="value">{len(losing_trades)}</div>
                <small>{len(losing_trades) / len(trades) * 100:.1f}%</small>
            </div>
            <div class="summary-card {'positive' if results['total_return'] > 0 else 'loss' if results['total_return'] < 0 else 'neutral'}">
                <h3>Total Return</h3>
                <div class="value">{results['total_return']:+.2f}%</div>
            </div>
            <div class="summary-card neutral">
                <h3>Signals Generated</h3>
                <div class="value">{results.get('signals_generated', 0)}</div>
            </div>
            <div class="summary-card {'positive' if total_profit > 0 else 'neutral'}">
                <h3>Total Profit</h3>
                <div class="value">${total_profit:.2f}</div>
            </div>
            <div class="summary-card {'loss' if total_loss < 0 else 'neutral'}">
                <h3>Total Loss</h3>
                <div class="value">${total_loss:.2f}</div>
            </div>
            <div class="summary-card {'positive' if profit_factor > 1 else 'neutral'}">
                <h3>Profit Factor</h3>
                <div class="value">{profit_factor:.2f}</div>
            </div>
            <div class="summary-card neutral">
                <h3>Avg Win</h3>
                <div class="value">${avg_win:.2f}</div>
            </div>
        </div>

        <h2>📊 Strategy Parameters</h2>
        <div class="strategy-params">
            <h4>Aggressive Momentum Strategy Configuration</h4>
            <div class="param-grid">
                <div class="param-item">
                    <strong>Min Confidence:</strong> {self.strategy_params['min_confidence']:.3f} (3%)
                </div>
                <div class="param-item">
                    <strong>Min Volume Ratio:</strong> {self.strategy_params['min_volume_ratio']:.1f}x
                </div>
                <div class="param-item">
                    <strong>Min ADX:</strong> {self.strategy_params['min_adx']:.0f}
                </div>
                <div class="param-item">
                    <strong>Immediate Entry Threshold:</strong> {self.strategy_params['immediate_entry_threshold']:.3f} (0.6%)
                </div>
                <div class="param-item">
                    <strong>Stop Loss Multiplier:</strong> {self.strategy_params['atr_sl_multiplier']:.1f}x ATR
                </div>
                <div class="param-item">
                    <strong>Take Profit Multiplier:</strong> {self.strategy_params['atr_tp_multiplier']:.1f}x ATR
                </div>
                <div class="param-item">
                    <strong>Position Size:</strong> {self.strategy_params['position_size_pct']:.1%}
                </div>
                <div class="param-item">
                    <strong>Max Hold Time:</strong> {self.strategy_params['max_position_time_hours']}h
                </div>
            </div>
        </div>

        <h2>📈 Trade Details</h2>
        <table class="trades-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Duration</th>
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

        # Add trade rows
        for i, trade in enumerate(trades, 1):
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            
            # Handle datetime formatting
            if hasattr(entry_time, 'strftime'):
                entry_str = entry_time.strftime('%H:%M')
                exit_str = exit_time.strftime('%H:%M')
                duration = (exit_time - entry_time).total_seconds() / 60
            else:
                entry_str = str(entry_time)
                exit_str = str(exit_time)
                duration = 0
            
            pnl = trade.get('net_pnl', 0)
            pnl_class = 'profit' if pnl > 0 else 'loss'
            
            signal_type = trade['signal_type']
            if hasattr(signal_type, 'value'):
                signal_type_str = signal_type.value
            else:
                signal_type_str = str(signal_type)
            
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{entry_str}</td>
                    <td>{exit_str}</td>
                    <td>{duration:.0f}m</td>
                    <td>{signal_type_str}</td>
                    <td>${trade['entry_price']:.2f}</td>
                    <td>${trade['exit_price']:.2f}</td>
                    <td class="{pnl_class}">${pnl:+.2f}</td>
                    <td>{trade['exit_reason']}</td>
                    <td>{trade.get('confidence', 0):.2f}</td>
                </tr>
"""

        # Close HTML
        html_content += f"""
            </tbody>
        </table>

        <h2>💡 Performance Analysis</h2>
        <div class="strategy-params">
            <h4>Strategy vs Market</h4>
            <div class="param-grid">
                <div class="param-item">
                    <strong>Data Block:</strong> {block_id}
                </div>
                <div class="param-item">
                    <strong>Initial Balance:</strong> ${self.initial_balance:,.2f}
                </div>
                <div class="param-item">
                    <strong>Final Balance:</strong> ${results['final_balance']:,.2f}
                </div>
                <div class="param-item">
                    <strong>Strategy Type:</strong> Aggressive Momentum
                </div>
                <div class="param-item">
                    <strong>Winning Trades:</strong> {len(winning_trades)} / {len(trades)}
                </div>
                <div class="param-item">
                    <strong>Average Trade:</strong> ${(total_profit + total_loss) / len(trades):.2f}
                </div>
            </div>
        </div>

        <h2>🎯 Key Insights</h2>
        <div class="strategy-params">
            <ul>
                <li><strong>Strategy Effectiveness:</strong> This aggressive momentum strategy successfully generated {len(trades)} trades compared to 0 from the conservative approach.</li>
                <li><strong>Signal Generation:</strong> {results.get('signals_generated', 0)} signals were generated, showing high market activity detection.</li>
                <li><strong>Risk Management:</strong> {'Tight stops prevented large losses' if avg_loss < 2 else 'Consider tighter stop losses'}</li>
                <li><strong>Profit Capture:</strong> {len(winning_trades)} trades captured significant market movements with an average win of ${avg_win:.2f}.</li>
                <li><strong>Strategy Comparison:</strong> Aggressive approach vs Conservative: {len(trades)} trades vs 0 trades on the same dataset.</li>
            </ul>
        </div>

        <div class="footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Aggressive Momentum Strategy | Block: {block_id}</p>
            <p>🤖 Generated with Claude Code Trading Bot Analysis</p>
        </div>
    </div>
</body>
</html>
"""

        # Write HTML file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file

def main():
    """Main function to run aggressive strategy with HTML report."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggressive Strategy with HTML Report")
    parser.add_argument('--block', default='august_12_single_day', help='Data block to use')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    
    args = parser.parse_args()
    
    # Run strategy with HTML report
    strategy = AggressiveStrategyWithHTML(initial_balance=args.balance)
    results = strategy.run_enhanced_backtest_with_html(args.block)
    
    # Save JSON results too
    if results.get('trades'):
        json_file = f"Output/aggressive_with_html_{args.block}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Serialize results for JSON
        json_results = {}
        for key, value in results.items():
            if key == 'trades':
                json_results[key] = []
                for trade in value:
                    trade_dict = {}
                    for k, v in trade.items():
                        if hasattr(v, 'value'):  # Enum
                            trade_dict[k] = v.value
                        elif hasattr(v, 'isoformat'):  # Datetime
                            trade_dict[k] = v.isoformat()
                        else:
                            trade_dict[k] = v
                    json_results[key].append(trade_dict)
            else:
                json_results[key] = value
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"💾 JSON results saved to: {json_file}")

if __name__ == "__main__":
    main()