#!/usr/bin/env python3
"""Create final summary report combining all analysis results."""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def load_backtest_results():
    """Load the latest backtest results."""
    output_dir = Path("Output")
    
    # Find the latest enhanced backtest results
    backtest_files = list(output_dir.glob("backtest_results_SOLUSDT_*.json"))
    if backtest_files:
        latest_backtest = max(backtest_files, key=lambda x: x.stat().st_mtime)
        with open(latest_backtest, 'r') as f:
            backtest_data = json.load(f)
        print(f"📊 Loaded backtest results from: {latest_backtest.name}")
        return backtest_data
    else:
        print("⚠️  No backtest results found")
        return None

def load_real_data_stats():
    """Load real market data statistics."""
    try:
        # Locate latest available 5m real file
        from pathlib import Path
        candidates = sorted(Path('data').glob('SOLUSDT_5m_real_*.csv'), key=lambda x: x.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError("No SOLUSDT_5m_real_*.csv files found")
        latest = candidates[-1]
        df_5m = pd.read_csv(latest)
        df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
        df_5m.set_index('timestamp', inplace=True)
        
        # Calculate market stats
        price_start = df_5m['close'].iloc[0]
        price_end = df_5m['close'].iloc[-1]
        price_min = df_5m['low'].min()
        price_max = df_5m['high'].max()
        
        # Price movements
        price_changes = df_5m['close'].diff().abs()
        movements_over_2 = (price_changes >= 2.0).sum()
        
        # Volatility
        returns = df_5m['close'].pct_change()
        volatility = returns.std() * (288 ** 0.5) * 100  # Annualized
        
        return {
            'period_start': df_5m.index[0],
            'period_end': df_5m.index[-1],
            'price_start': price_start,
            'price_end': price_end,
            'price_min': price_min,
            'price_max': price_max,
            'total_return_pct': (price_end / price_start - 1) * 100,
            'price_range': price_max - price_min,
            'movements_over_2_usd': movements_over_2,
            'volatility_pct': volatility,
            'total_candles': len(df_5m),
            'total_volume': df_5m['volume'].sum(),
            'avg_volume': df_5m['volume'].mean(),
            'max_5min_move': price_changes.max()
        }
    except Exception as e:
        print(f"⚠️  Could not load real data stats: {e}")
        return None

def create_final_summary_report():
    """Create comprehensive final summary report."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    backtest_results = load_backtest_results()
    market_stats = load_real_data_stats()
    
    if not market_stats:
        print("❌ Cannot generate report without market data")
        return
    
    # Determine overall assessment
    def assess_strategy_performance():
        if not backtest_results or backtest_results.get('total_trades', 0) == 0:
            return {
                'status': 'NEEDS_OPTIMIZATION',
                'color': '#f39c12',
                'message': 'Strategy requires parameter optimization - no trades executed',
                'recommendations': [
                    'Lower confidence threshold from 15% to 8-10%',
                    'Reduce volume requirements for real market conditions',
                    'Consider using 1m timeframe for faster signal detection',
                    'Optimize strategy parameters for current market volatility'
                ]
            }
        
        total_trades = backtest_results.get('total_trades', 0)
        win_rate = backtest_results.get('win_rate', 0)
        profit_factor = backtest_results.get('profit_factor', 0)
        total_pnl = backtest_results.get('total_pnl', 0)
        
        if total_trades > 5 and win_rate >= 0.6 and profit_factor >= 1.5 and total_pnl > 0:
            return {
                'status': 'EXCELLENT',
                'color': '#27ae60',
                'message': 'Strategy performs well on real market data',
                'recommendations': [
                    'Consider increasing position size gradually',
                    'Implement risk scaling based on volatility',
                    'Test on longer time periods for validation',
                    'Begin paper trading with real-time data'
                ]
            }
        elif total_trades > 3 and (win_rate >= 0.4 or profit_factor >= 1.0):
            return {
                'status': 'PROMISING',
                'color': '#f39c12',
                'message': 'Strategy shows potential but needs refinement',
                'recommendations': [
                    'Fine-tune stop loss and take profit levels',
                    'Optimize entry signal confidence thresholds',
                    'Consider market regime detection',
                    'Test different timeframe combinations'
                ]
            }
        else:
            return {
                'status': 'NEEDS_WORK',
                'color': '#e74c3c',
                'message': 'Strategy requires significant improvement',
                'recommendations': [
                    'Review and simplify signal generation logic',
                    'Consider alternative technical indicators',
                    'Implement adaptive parameters based on market conditions',
                    'Reduce complexity and focus on core profitable patterns'
                ]
            }
    
    assessment = assess_strategy_performance()
    
    # Calculate opportunity score
    opportunity_score = min(100, max(0, 
        (market_stats['movements_over_2_usd'] * 10) +  # Movement frequency
        (market_stats['volatility_pct'] * 2) +         # Volatility bonus
        (market_stats['total_return_pct'] * 2)         # Trend bonus
    ))
    
    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SOL/USDT Trading Bot - Final Analysis Summary</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 0 30px rgba(0,0,0,0.2);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            border-radius: 10px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .status-card {{
            background-color: {assessment['color']};
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .status-card h2 {{
            margin: 0 0 10px 0;
            font-size: 1.8em;
        }}
        .status-card p {{
            margin: 0;
            font-size: 1.2em;
            opacity: 0.95;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #3498db;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-card .value {{
            font-size: 2.2em;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }}
        .metric-card .subtext {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .section {{
            margin: 40px 0;
            padding: 25px;
            background-color: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #3498db;
        }}
        .section h2 {{
            color: #2c3e50;
            margin-top: 0;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }}
        .recommendations {{
            background-color: #e8f4fd;
            border: 1px solid #bee5eb;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .recommendations h4 {{
            color: #0c5460;
            margin-top: 0;
        }}
        .recommendations ul {{
            margin: 10px 0 0 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            color: #0c5460;
            margin: 8px 0;
            line-height: 1.5;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .comparison-table th {{
            background-color: #34495e;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: bold;
        }}
        .comparison-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .comparison-table tr:hover {{
            background-color: #f8f9fa;
        }}
        .progress-bar {{
            background-color: #ecf0f1;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #27ae60 0%, #2ecc71 100%);
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        .key-insights {{
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border: 1px solid #ffeaa7;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .key-insights h4 {{
            color: #856404;
            margin-top: 0;
        }}
        .insights-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .insight-item {{
            background-color: rgba(255,255,255,0.7);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #f39c12;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #6c757d;
        }}
        .data-source {{
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            text-align: center;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 SOL/USDT Trading Bot Analysis</h1>
            <p>Comprehensive 7-Day Real Market Data Backtest Results</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="status-card">
            <h2>🎯 Overall Assessment: {assessment['status'].replace('_', ' ')}</h2>
            <p>{assessment['message']}</p>
        </div>
        
        <div class="data-source">
            <strong>📡 Data Source:</strong> Real Bybit market data • 
            <strong>Period:</strong> {market_stats['period_start'].strftime('%Y-%m-%d')} to {market_stats['period_end'].strftime('%Y-%m-%d')} • 
            <strong>Candles:</strong> {market_stats['total_candles']:,} (5-minute intervals)
        </div>
        
        <div class="section">
            <h2>🏪 Market Performance Analysis</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Price Movement</h3>
                    <div class="value">{market_stats['total_return_pct']:+.2f}%</div>
                    <div class="subtext">${market_stats['price_start']:.2f} → ${market_stats['price_end']:.2f}</div>
                </div>
                <div class="metric-card">
                    <h3>Trading Range</h3>
                    <div class="value">${market_stats['price_range']:.2f}</div>
                    <div class="subtext">${market_stats['price_min']:.2f} - ${market_stats['price_max']:.2f}</div>
                </div>
                <div class="metric-card">
                    <h3>Volatility</h3>
                    <div class="value">{market_stats['volatility_pct']:.1f}%</div>
                    <div class="subtext">Annualized</div>
                </div>
                <div class="metric-card">
                    <h3>Movements ≥$2</h3>
                    <div class="value">{market_stats['movements_over_2_usd']}</div>
                    <div class="subtext">Target opportunities</div>
                </div>
                <div class="metric-card">
                    <h3>Max 5min Move</h3>
                    <div class="value">${market_stats['max_5min_move']:.2f}</div>
                    <div class="subtext">Single candle</div>
                </div>
                <div class="metric-card">
                    <h3>Opportunity Score</h3>
                    <div class="value">{opportunity_score:.0f}/100</div>
                    <div class="subtext">Market suitability</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>⚡ Strategy Performance</h2>
    """
    
    if backtest_results and backtest_results.get('total_trades', 0) > 0:
        html_content += f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Total Trades</h3>
                    <div class="value">{backtest_results['total_trades']}</div>
                    <div class="subtext">From {backtest_results.get('signals_generated', 0)} signals</div>
                </div>
                <div class="metric-card">
                    <h3>Win Rate</h3>
                    <div class="value">{backtest_results['win_rate']:.1%}</div>
                    <div class="subtext">{backtest_results['winning_trades']}W / {backtest_results['losing_trades']}L</div>
                </div>
                <div class="metric-card">
                    <h3>Total P&L</h3>
                    <div class="value">${backtest_results['total_pnl']:+.2f}</div>
                    <div class="subtext">Commission: ${backtest_results.get('total_commission', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <h3>Profit Factor</h3>
                    <div class="value">{backtest_results['profit_factor']:.2f}</div>
                    <div class="subtext">Risk/Reward ratio</div>
                </div>
                <div class="metric-card">
                    <h3>Avg Duration</h3>
                    <div class="value">{backtest_results['avg_trade_duration_hours']:.1f}h</div>
                    <div class="subtext">Per trade</div>
                </div>
                <div class="metric-card">
                    <h3>Movements Captured</h3>
                    <div class="value">{backtest_results.get('movements_over_2_usd', 0)}</div>
                    <div class="subtext">≥$2 moves</div>
                </div>
            </div>
            
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Result</th>
                        <th>Target</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Win Rate</td>
                        <td>{backtest_results['win_rate']:.1%}</td>
                        <td>≥55%</td>
                        <td>{'✅ Good' if backtest_results['win_rate'] >= 0.55 else '⚠️ Below target' if backtest_results['win_rate'] >= 0.40 else '❌ Poor'}</td>
                    </tr>
                    <tr>
                        <td>Profit Factor</td>
                        <td>{backtest_results['profit_factor']:.2f}</td>
                        <td>≥1.3</td>
                        <td>{'✅ Good' if backtest_results['profit_factor'] >= 1.3 else '⚠️ Below target' if backtest_results['profit_factor'] >= 1.0 else '❌ Poor'}</td>
                    </tr>
                    <tr>
                        <td>Max Drawdown</td>
                        <td>{backtest_results['max_drawdown']:.1%}</td>
                        <td>≤10%</td>
                        <td>{'✅ Good' if backtest_results['max_drawdown'] <= 0.10 else '⚠️ Elevated' if backtest_results['max_drawdown'] <= 0.15 else '❌ High'}</td>
                    </tr>
                    <tr>
                        <td>$2+ Movements Captured</td>
                        <td>{backtest_results.get('movements_over_2_usd', 0)}/{market_stats['movements_over_2_usd']}</td>
                        <td>≥30%</td>
                        <td>{'✅ Good' if backtest_results.get('movements_over_2_usd', 0) >= market_stats['movements_over_2_usd'] * 0.3 else '⚠️ Moderate' if backtest_results.get('movements_over_2_usd', 0) > 0 else '❌ Poor'}</td>
                    </tr>
                </tbody>
            </table>
        """
    else:
        html_content += """
            <div style="text-align: center; padding: 40px; background-color: #fff3cd; border: 2px dashed #f39c12; border-radius: 10px;">
                <h3>⚠️ No Trades Executed</h3>
                <p>The strategy did not generate any trades during the 7-day analysis period.</p>
                <p>This indicates that the current signal criteria are too restrictive for the market conditions.</p>
            </div>
        """
    
    html_content += f"""
        </div>
        
        <div class="key-insights">
            <h4>💡 Key Insights</h4>
            <div class="insights-grid">
                <div class="insight-item">
                    <strong>Market Opportunity:</strong><br>
                    SOL/USDT showed <span class="highlight">{market_stats['movements_over_2_usd']} movements ≥$2</span> over 7 days, 
                    indicating {'excellent' if market_stats['movements_over_2_usd'] >= 20 else 'good' if market_stats['movements_over_2_usd'] >= 10 else 'limited'} 
                    trading opportunities for the target strategy.
                </div>
                <div class="insight-item">
                    <strong>Volatility Assessment:</strong><br>
                    Market volatility of <span class="highlight">{market_stats['volatility_pct']:.1f}%</span> is 
                    {'high' if market_stats['volatility_pct'] > 10 else 'moderate' if market_stats['volatility_pct'] > 5 else 'low'}, 
                    {'providing good' if market_stats['volatility_pct'] > 5 else 'limiting'} trading opportunities.
                </div>
                <div class="insight-item">
                    <strong>Strategy Effectiveness:</strong><br>
                    {'Strategy successfully captured profitable movements' if backtest_results and backtest_results.get('total_trades', 0) > 0 and backtest_results.get('total_pnl', 0) > 0 else 'Strategy needs optimization' if backtest_results and backtest_results.get('total_trades', 0) > 0 else 'Strategy requires significant parameter adjustment'} 
                    based on real market data analysis.
                </div>
                <div class="insight-item">
                    <strong>Risk Management:</strong><br>
                    {'Effective risk controls maintained low drawdown' if backtest_results and backtest_results.get('max_drawdown', 0) < 0.10 else 'Risk management parameters may need adjustment' if backtest_results and backtest_results.get('max_drawdown', 0) < 0.20 else 'Risk controls need strengthening'}.
                </div>
            </div>
        </div>
        
        <div class="recommendations">
            <h4>🎯 Actionable Recommendations</h4>
            <ul>
    """
    
    for rec in assessment['recommendations']:
        html_content += f"<li>{rec}</li>"
    
    html_content += f"""
            </ul>
        </div>
        
        <div class="section">
            <h2>📈 Next Steps</h2>
            <div style="padding: 20px; background-color: white; border-radius: 8px; border: 1px solid #dee2e6;">
                <h4>Immediate Actions:</h4>
                <ol>
                    <li><strong>Parameter Optimization:</strong> {'Fine-tune existing parameters' if backtest_results and backtest_results.get('total_trades', 0) > 0 else 'Lower signal thresholds to generate trades'}</li>
                    <li><strong>Extended Testing:</strong> Run backtests on longer periods (30+ days) for validation</li>
                    <li><strong>Real-time Validation:</strong> Implement paper trading with live market data</li>
                    <li><strong>Risk Scaling:</strong> Implement dynamic position sizing based on volatility</li>
                </ol>
                
                <h4>Long-term Strategy:</h4>
                <ol>
                    <li>Develop market regime detection for adaptive parameters</li>
                    <li>Implement multi-timeframe signal confirmation</li>
                    <li>Add machine learning components for signal enhancement</li>
                    <li>Create automated performance monitoring and alerting</li>
                </ol>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Analysis completed on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</strong></p>
            <p>Based on {market_stats['total_candles']:,} candles of real Bybit SOL/USDT market data</p>
            <p>Report generated by SOL/USDT Trading Bot Analysis System</p>
        </div>
    </div>
</body>
</html>
    """
    
    # Save report
    Path("Output").mkdir(exist_ok=True)
    report_filename = f"Output/SOL_USDT_Final_Summary_{timestamp}.html"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_filename, assessment

def main():
    """Generate final summary report."""
    
    print("📋 Creating Final Summary Report")
    print("=" * 50)
    
    report_filename, assessment = create_final_summary_report()
    
    print(f"✅ Final summary report generated: {report_filename}")
    print(f"🎯 Overall assessment: {assessment['status'].replace('_', ' ')}")
    print(f"💬 {assessment['message']}")
    
    return report_filename

if __name__ == "__main__":
    main()