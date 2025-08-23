#!/usr/bin/env python3
"""
Generate comprehensive HTML backtest report based on backtest_report_concept.md
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def generate_backtest_report(results_file: str, price_data_file: str):
    """Generate comprehensive HTML backtest report."""
    
    # Load backtest results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load price data
    df_price = pd.read_csv(price_data_file)
    df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
    df_price = df_price.set_index('timestamp')
    
    # Prepare data
    trades = results['trades']
    initial_balance = results['initial_balance']
    final_balance = results['final_balance']
    total_pnl = results['total_pnl']
    total_return_pct = results['total_return_pct']
    
    # Calculate equity curve
    equity_curve = [initial_balance]
    balance = initial_balance
    
    for trade in trades:
        balance += trade['pnl']
        equity_curve.append(balance)
    
    # Create HTML report
    html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SOL/USDT Trading Bot - Backtest Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2rem;
            opacity: 0.9;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
        }}
        .metric-card.positive {{
            border-left-color: #28a745;
        }}
        .metric-card.negative {{
            border-left-color: #dc3545;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        .metric-value.positive {{
            color: #28a745;
        }}
        .metric-value.negative {{
            color: #dc3545;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .section {{
            padding: 30px;
            border-bottom: 1px solid #eee;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section h2 {{
            margin: 0 0 20px 0;
            font-size: 1.8rem;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .chart-container {{
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .trades-table th,
        .trades-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .trades-table th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}
        .trades-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .pnl-positive {{
            color: #28a745;
            font-weight: 600;
        }}
        .pnl-negative {{
            color: #dc3545;
            font-weight: 600;
        }}
        .side-long {{
            background-color: #d4edda;
            color: #155724;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .side-short {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9rem;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🤖 SOL/USDT Trading Bot</h1>
            <p>Backtest Report - August 10-17, 2025</p>
            <p>ML Enhanced Strategy (90d Dataset)</p>
        </div>

        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card {'positive' if total_return_pct > 0 else 'negative'}">
                <div class="metric-value {'positive' if total_return_pct > 0 else 'negative'}">
                    {total_return_pct:+.2f}%
                </div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card {'positive' if total_pnl > 0 else 'negative'}">
                <div class="metric-value {'positive' if total_pnl > 0 else 'negative'}">
                    ${total_pnl:+,.2f}
                </div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(trades)}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card {'positive' if results['win_rate'] > 50 else 'negative'}">
                <div class="metric-value {'positive' if results['win_rate'] > 50 else 'negative'}">
                    {results['win_rate']:.1f}%
                </div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{results['max_drawdown_pct']:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${final_balance:,.2f}</div>
                <div class="metric-label">Final Balance</div>
            </div>
        </div>

        <!-- Equity Curve Section -->
        <div class="section">
            <h2>📈 Equity Curve</h2>
            <div class="chart-container">
                <div id="equity-chart"></div>
            </div>
        </div>

        <!-- Price Chart with Trades -->
        <div class="section">
            <h2>💹 Price Chart with Trade Markers</h2>
            <div class="chart-container">
                <div id="price-chart"></div>
            </div>
        </div>

        <!-- Performance Summary -->
        <div class="section">
            <h2>📊 Performance Analysis</h2>
            <div class="summary-stats">
                <div class="stat-box">
                    <h3>🎯 Strategy Details</h3>
                    <p><strong>Model:</strong> {results['model_version']}</p>
                    <p><strong>Period:</strong> {results['dataset_records']} data points</p>
                    <p><strong>Position Size:</strong> {results['max_position_pct']:.1f}% per trade</p>
                    <p><strong>Commission:</strong> {results['commission_rate']*100:.1f}%</p>
                </div>
                <div class="stat-box">
                    <h3>📈 Trade Statistics</h3>
                    <p><strong>Profitable:</strong> {results['profitable_trades']} trades</p>
                    <p><strong>Losing:</strong> {len(trades) - results['profitable_trades']} trades</p>
                    <p><strong>Avg PnL:</strong> ${sum(t['pnl'] for t in trades) / len(trades) if trades else 0:+.2f}</p>
                    <p><strong>Best Trade:</strong> ${max(t['pnl'] for t in trades) if trades else 0:+.2f}</p>
                    <p><strong>Worst Trade:</strong> ${min(t['pnl'] for t in trades) if trades else 0:+.2f}</p>
                </div>
                <div class="stat-box">
                    <h3>🎛️ Risk Management</h3>
                    <p><strong>Stop Loss:</strong> {results['stop_loss_pct']:.1f}%</p>
                    <p><strong>Take Profit:</strong> {results['take_profit_pct']:.1f}%</p>
                    <p><strong>ML Confidence:</strong> >{results['min_confidence']:.2f}</p>
                    <p><strong>Max Drawdown:</strong> {results['max_drawdown_pct']:.2f}%</p>
                </div>
                <div class="stat-box">
                    <h3>🤖 ML Performance</h3>
                    <p><strong>Predictions:</strong> {results['successful_predictions']:,}</p>
                    <p><strong>Failed:</strong> {results['failed_predictions']}</p>
                    <p><strong>Trades Opened:</strong> {results['trades_opened']}</p>
                    <p><strong>Selectivity:</strong> {(results['trades_opened']/results['successful_predictions']*100) if results['successful_predictions'] > 0 else 0:.2f}%</p>
                </div>
            </div>
        </div>

        <!-- Trades Table -->
        <div class="section">
            <h2>📋 Detailed Trades</h2>
            <div style="overflow-x: auto;">
                <table class="trades-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Entry Time</th>
                            <th>Entry Price</th>
                            <th>Exit Price</th>
                            <th>Side</th>
                            <th>Position Size</th>
                            <th>P&L ($)</th>
                            <th>P&L (%)</th>
                            <th>Exit Reason</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    # Add trades to table
    for i, trade in enumerate(trades, 1):
        pnl_class = "pnl-positive" if trade['pnl'] > 0 else "pnl-negative"
        side_class = "side-long" if trade['side'] == 'LONG' else "side-short"
        entry_time = pd.to_datetime(trade['entry_time']).strftime('%m-%d %H:%M')
        
        html_content += f"""
                        <tr>
                            <td>{i}</td>
                            <td>{entry_time}</td>
                            <td>${trade['entry_price']:.2f}</td>
                            <td>${trade['exit_price']:.2f}</td>
                            <td><span class="{side_class}">{trade['side']}</span></td>
                            <td>${trade['position_size_usd']:.0f}</td>
                            <td class="{pnl_class}">${trade['pnl']:+.2f}</td>
                            <td class="{pnl_class}">{trade['pnl_pct']:+.2f}%</td>
                            <td>{trade['exit_reason']}</td>
                        </tr>
        """

    # Complete HTML
    html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Footer -->
        <div class="section">
            <div class="timestamp">
                Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                Strategy: ML Enhanced (CatBoost + Random Forest) | 
                Dataset: 90-day Bybit Futures SOLUSDT
            </div>
        </div>
    </div>

    <script>
        // Equity Curve Chart
        const equityTrace = {{
            x: {[f"'{pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}'" for trade in trades]},
            y: {equity_curve[1:]},  // Skip initial balance
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Equity',
            line: {{color: '#667eea', width: 3}},
            marker: {{size: 6}}
        }};

        const equityLayout = {{
            title: 'Portfolio Equity Curve',
            xaxis: {{title: 'Time'}},
            yaxis: {{title: 'Balance ($)'}},
            showlegend: false,
            height: 400
        }};

        Plotly.newPlot('equity-chart', [equityTrace], equityLayout);

        // Price Chart with Trade Markers
        const priceData = {df_price.to_dict('list')};
        const timestamps = {[f"'{ts}'" for ts in df_price.index.strftime('%Y-%m-%d %H:%M')]};
        
        const priceTrace = {{
            x: timestamps,
            y: priceData.close,
            type: 'scatter',
            mode: 'lines',
            name: 'SOL/USDT Price',
            line: {{color: '#ffa500', width: 1}}
        }};

        // Entry markers
        const entryTrace = {{
            x: {[f"'{pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}'" for trade in trades]},
            y: {[trade['entry_price'] for trade in trades]},
            mode: 'markers',
            name: 'Entry',
            marker: {{
                symbol: 'triangle-up',
                size: 12,
                color: {['"#28a745"' if trade['side'] == 'LONG' else '"#dc3545"' for trade in trades]}
            }},
            text: {[f"'{trade['side']} Entry: ${trade['entry_price']:.2f}'" for trade in trades]},
            hovertemplate: '%{{text}}<extra></extra>'
        }};

        // Exit markers  
        const exitTrace = {{
            x: {[f"'{pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}'" for trade in trades]}, // Using entry_time as proxy
            y: {[trade['exit_price'] for trade in trades]},
            mode: 'markers',
            name: 'Exit',
            marker: {{
                symbol: 'triangle-down',
                size: 12,
                color: {['"#dc3545"' if trade['side'] == 'LONG' else '"#28a745"' for trade in trades]}
            }},
            text: {[f"'{trade['side']} Exit: ${trade['exit_price']:.2f} | P&L: ${trade['pnl']:+.2f} ({trade['exit_reason']})'" for trade in trades]},
            hovertemplate: '%{{text}}<extra></extra>'
        }};

        const priceLayout = {{
            title: 'SOL/USDT Price Action with Trade Entries/Exits',
            xaxis: {{title: 'Time'}},
            yaxis: {{title: 'Price ($)'}},
            showlegend: true,
            height: 500
        }};

        Plotly.newPlot('price-chart', [priceTrace, entryTrace, exitTrace], priceLayout);
    </script>
</body>
</html>
    """

    return html_content

if __name__ == "__main__":
    # Generate report for the latest backtest
    results_file = "output/proper_ml_backtest_august_20250823_141722.json"
    price_data_file = "SOLUSDT_5m_aug10_17.csv"
    
    print("📊 Generating comprehensive backtest report...")
    
    try:
        html_report = generate_backtest_report(results_file, price_data_file)
        
        # Save report
        output_path = Path("output") / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"✅ Report generated successfully: {output_path}")
        print(f"🌐 Open in browser: file://{output_path.absolute()}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")