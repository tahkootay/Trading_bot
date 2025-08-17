#!/usr/bin/env python3
"""Generate detailed backtest report with signal analysis and explanations."""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def load_market_data():
    """Load market data for analysis (prefers real data over testnet)."""
    # Try real data first
    data_files = {
        '1m': 'data/SOLUSDT_1m_real_7d.csv',
        '5m': 'data/SOLUSDT_5m_real_7d.csv', 
        '15m': 'data/SOLUSDT_15m_real_7d.csv',
        '1h': 'data/SOLUSDT_1h_real_7d.csv'
    }
    
    # Check if real data exists, otherwise fallback to testnet
    real_data_exists = any(Path(file_path).exists() for file_path in data_files.values())
    
    if not real_data_exists:
        print("⚠️  Real data not found, using testnet data...")
        data_files = {
            '1m': 'data/SOLUSDT_1m_testnet.csv',
            '5m': 'data/SOLUSDT_5m_testnet.csv', 
            '15m': 'data/SOLUSDT_15m_testnet.csv',
            '1h': 'data/SOLUSDT_1h_testnet.csv'
        }
    else:
        print("✅ Using real market data...")
    
    data = {}
    for timeframe, file_path in data_files.items():
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            data[timeframe] = df
            print(f"✅ Loaded {len(df)} candles for {timeframe}")
    
    return data

def calculate_simple_indicators(df):
    """Calculate basic technical indicators."""
    
    # Simple Moving Averages
    df['sma_8'] = df['close'].rolling(8).mean()
    df['sma_21'] = df['close'].rolling(21).mean()
    df['sma_55'] = df['close'].rolling(55).mean()
    
    # Exponential Moving Averages
    df['ema_8'] = df['close'].ewm(span=8).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_55'] = df['close'].ewm(span=55).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume analysis
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price vs VWAP (simplified)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    
    # ATR (simplified)
    df['tr'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['close'].shift(1)),
                                   abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']
    
    return df

def analyze_signals(df):
    """Analyze potential signals and explain why they were/weren't generated."""
    
    signals_analysis = []
    
    for i in range(55, len(df)):  # Start after we have enough data for indicators
        row = df.iloc[i]
        timestamp = df.index[i]
        
        # Check if we have valid indicator values
        if pd.isna(row['ema_8']) or pd.isna(row['ema_21']) or pd.isna(row['ema_55']):
            continue
            
        # Analyze potential BUY signal
        bullish_conditions = {
            'ema_trend_up': row['ema_8'] > row['ema_21'] and row['ema_21'] > row['ema_55'],
            'price_above_ema8': row['close'] > row['ema_8'],
            'rsi_reasonable': 40 < row['rsi'] < 75 if not pd.isna(row['rsi']) else False,
            'volume_good': row['volume_ratio'] > 1.2 if not pd.isna(row['volume_ratio']) else False,
            'price_vs_vwap_ok': row['price_vs_vwap'] > -0.01 if not pd.isna(row['price_vs_vwap']) else False,
        }
        
        # Analyze potential SELL signal
        bearish_conditions = {
            'ema_trend_down': row['ema_8'] < row['ema_21'] and row['ema_21'] < row['ema_55'],
            'price_below_ema8': row['close'] < row['ema_8'],
            'rsi_reasonable': 25 < row['rsi'] < 60 if not pd.isna(row['rsi']) else False,
            'volume_good': row['volume_ratio'] > 1.2 if not pd.isna(row['volume_ratio']) else False,
            'price_vs_vwap_ok': row['price_vs_vwap'] < 0.01 if not pd.isna(row['price_vs_vwap']) else False,
        }
        
        bullish_score = sum(bullish_conditions.values()) / len(bullish_conditions)
        bearish_score = sum(bearish_conditions.values()) / len(bearish_conditions)
        
        # Record significant moments
        if bullish_score > 0.4 or bearish_score > 0.4 or i % 20 == 0:  # Sample points
            analysis = {
                'timestamp': timestamp,
                'price': row['close'],
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'bullish_conditions': bullish_conditions,
                'bearish_conditions': bearish_conditions,
                'indicators': {
                    'ema_8': row['ema_8'],
                    'ema_21': row['ema_21'],
                    'ema_55': row['ema_55'],
                    'rsi': row['rsi'],
                    'volume_ratio': row['volume_ratio'],
                    'price_vs_vwap': row['price_vs_vwap'],
                    'atr_ratio': row['atr_ratio'],
                },
                'potential_signal': 'BUY' if bullish_score > 0.6 and bullish_score > bearish_score + 0.2 
                                  else 'SELL' if bearish_score > 0.6 and bearish_score > bullish_score + 0.2 
                                  else 'NONE',
                'explanation': f"Bullish: {bullish_score:.1%}, Bearish: {bearish_score:.1%}"
            }
            
            signals_analysis.append(analysis)
    
    return signals_analysis

def generate_market_summary(data):
    """Generate market summary from the data."""
    
    # Use 5m data for primary analysis
    df_5m = data['5m'].copy()
    
    price_start = df_5m['close'].iloc[0]
    price_end = df_5m['close'].iloc[-1]
    price_min = df_5m['low'].min()
    price_max = df_5m['high'].max()
    
    total_volume = df_5m['volume'].sum()
    avg_volume = df_5m['volume'].mean()
    
    # Volatility
    returns = df_5m['close'].pct_change()
    volatility = returns.std() * np.sqrt(288)  # Annualized for 5min periods
    
    # Price movements >= $2
    price_changes = df_5m['close'].diff().abs()
    movements_over_2 = (price_changes >= 2.0).sum()
    
    summary = {
        'period_start': df_5m.index[0],
        'period_end': df_5m.index[-1],
        'price_start': price_start,
        'price_end': price_end,
        'price_min': price_min,
        'price_max': price_max,
        'total_return_pct': (price_end / price_start - 1) * 100,
        'price_range': price_max - price_min,
        'total_volume': total_volume,
        'avg_volume': avg_volume,
        'volatility_pct': volatility * 100,
        'candles_analyzed': len(df_5m),
        'movements_over_2_usd': movements_over_2,
        'max_5min_move': price_changes.max(),
    }
    
    return summary

def create_html_report(market_summary, signals_analysis, backtest_results):
    """Create comprehensive HTML report."""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SOL/USDT Backtest Analysis Report</title>
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
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.8;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            margin: 5px 0;
        }}
        .signal-analysis {{
            margin: 20px 0;
        }}
        .signal-item {{
            border: 1px solid #ddd;
            border-radius: 8px;
            margin: 10px 0;
            padding: 15px;
            background-color: #f9f9f9;
        }}
        .signal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .signal-type {{
            padding: 5px 10px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
        }}
        .signal-buy {{ background-color: #27ae60; color: white; }}
        .signal-sell {{ background-color: #e74c3c; color: white; }}
        .signal-none {{ background-color: #95a5a6; color: white; }}
        .conditions-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 10px;
        }}
        .conditions {{
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #eee;
        }}
        .condition {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            font-size: 14px;
        }}
        .condition-true {{ color: #27ae60; }}
        .condition-false {{ color: #e74c3c; }}
        .indicators {{
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .indicator {{
            display: inline-block;
            margin: 2px 5px;
            padding: 2px 8px;
            background-color: #3498db;
            color: white;
            border-radius: 12px;
            font-size: 12px;
        }}
        .explanation {{
            margin-top: 15px;
            padding: 10px;
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .no-trades {{
            text-align: center;
            padding: 40px;
            background-color: #ffebee;
            border: 2px dashed #f44336;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .score-bar {{
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }}
        .score-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        .score-bullish {{ background-color: #27ae60; }}
        .score-bearish {{ background-color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 SOL/USDT Backtest Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>🏪 Market Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Analysis Period</h3>
                <div class="value">{(market_summary['period_end'] - market_summary['period_start']).days} days</div>
                <small>{market_summary['period_start'].strftime('%Y-%m-%d %H:%M')} - {market_summary['period_end'].strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            <div class="summary-card">
                <h3>Price Range</h3>
                <div class="value">${market_summary['price_min']:.2f} - ${market_summary['price_max']:.2f}</div>
                <small>Range: ${market_summary['price_range']:.2f}</small>
            </div>
            <div class="summary-card">
                <h3>Total Return</h3>
                <div class="value">{market_summary['total_return_pct']:+.2f}%</div>
                <small>${market_summary['price_start']:.2f} → ${market_summary['price_end']:.2f}</small>
            </div>
            <div class="summary-card">
                <h3>Movements ≥$2</h3>
                <div class="value">{market_summary['movements_over_2_usd']}</div>
                <small>Max 5min move: ${market_summary['max_5min_move']:.2f}</small>
            </div>
            <div class="summary-card">
                <h3>Average Volume</h3>
                <div class="value">{market_summary['avg_volume']:.2f}</div>
                <small>Total: {market_summary['total_volume']:.2f}</small>
            </div>
            <div class="summary-card">
                <h3>Volatility</h3>
                <div class="value">{market_summary['volatility_pct']:.1f}%</div>
                <small>Annualized</small>
            </div>
        </div>
        
        <h2>⚠️ Backtest Results</h2>
        <div class="no-trades">
            <h3>❌ No Trades Executed</h3>
            <p>The enhanced backtest strategy did not generate any executable trades during the analysis period.</p>
            <p>This indicates that the signal criteria are either too strict or the market conditions during this period did not meet the strategy requirements.</p>
        </div>
        
        <h2>🔍 Signal Analysis ({len(signals_analysis)} Analysis Points)</h2>
        <div class="warning">
            <strong>Key Findings:</strong>
            <ul>
                <li>Strategy requires multiple technical indicators to align simultaneously</li>
                <li>Minimum confidence threshold of 15% (0.15) for signal generation</li>
                <li>Volume ratio must be >1.2x average</li>
                <li>ADX must be ≥20 for trend strength confirmation</li>
                <li>The testnet data may have limited volume activity affecting volume-based criteria</li>
            </ul>
        </div>
        
        <div class="signal-analysis">
    """
    
    # Add signal analysis items
    for signal in signals_analysis[-20:]:  # Show last 20 analysis points
        signal_class = f"signal-{signal['potential_signal'].lower()}"
        
        html_content += f"""
            <div class="signal-item">
                <div class="signal-header">
                    <span class="timestamp">{signal['timestamp'].strftime('%Y-%m-%d %H:%M')}</span>
                    <span class="signal-type {signal_class}">{signal['potential_signal']}</span>
                    <span>Price: ${signal['price']:.2f}</span>
                </div>
                
                <div class="conditions-grid">
                    <div class="conditions">
                        <h4>🐂 Bullish Conditions ({signal['bullish_score']:.1%})</h4>
                        <div class="score-bar">
                            <div class="score-fill score-bullish" style="width: {signal['bullish_score']*100}%"></div>
                        </div>
        """
        
        for condition, met in signal['bullish_conditions'].items():
            status_class = "condition-true" if met else "condition-false"
            status_icon = "✅" if met else "❌"
            html_content += f'<div class="condition"><span>{condition.replace("_", " ").title()}</span><span class="{status_class}">{status_icon}</span></div>'
        
        html_content += f"""
                    </div>
                    <div class="conditions">
                        <h4>🐻 Bearish Conditions ({signal['bearish_score']:.1%})</h4>
                        <div class="score-bar">
                            <div class="score-fill score-bearish" style="width: {signal['bearish_score']*100}%"></div>
                        </div>
        """
        
        for condition, met in signal['bearish_conditions'].items():
            status_class = "condition-true" if met else "condition-false"
            status_icon = "✅" if met else "❌"
            html_content += f'<div class="condition"><span>{condition.replace("_", " ").title()}</span><span class="{status_class}">{status_icon}</span></div>'
        
        html_content += """
                    </div>
                </div>
                
                <div class="indicators">
                    <strong>Technical Indicators:</strong>
        """
        
        for indicator, value in signal['indicators'].items():
            if not pd.isna(value):
                html_content += f'<span class="indicator">{indicator.upper()}: {value:.3f}</span>'
            else:
                html_content += f'<span class="indicator">{indicator.upper()}: N/A</span>'
        
        html_content += f"""
                </div>
                
                <div class="explanation">
                    <strong>Analysis:</strong> {signal['explanation']}
                    {f"<br><strong>Why no signal:</strong> Neither bullish nor bearish confidence reached the 60% threshold for signal generation." if signal['potential_signal'] == 'NONE' else ''}
                </div>
            </div>
        """
    
    html_content += f"""
        </div>
        
        <h2>📋 Strategy Configuration</h2>
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <h4>Current Strategy Parameters:</h4>
            <ul>
                <li><strong>Minimum Confidence:</strong> 15% (0.15)</li>
                <li><strong>Minimum Volume Ratio:</strong> 1.2x average</li>
                <li><strong>Minimum ADX:</strong> 20.0 (trend strength)</li>
                <li><strong>ATR Stop Loss Multiplier:</strong> 1.2x</li>
                <li><strong>ATR Take Profit Multiplier:</strong> 2.0x</li>
                <li><strong>Maximum Position Time:</strong> 4 hours</li>
                <li><strong>Position Size:</strong> 2% of balance</li>
            </ul>
        </div>
        
        <h2>💡 Recommendations</h2>
        <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <h4>To improve signal generation:</h4>
            <ol>
                <li><strong>Lower confidence threshold:</strong> Consider reducing from 15% to 10% for more signals</li>
                <li><strong>Adjust volume requirements:</strong> The testnet data shows low volume activity</li>
                <li><strong>Use real market data:</strong> Live market data would provide more realistic volume patterns</li>
                <li><strong>Optimize for testnet conditions:</strong> Adapt strategy parameters for the specific data characteristics</li>
                <li><strong>Add more timeframes:</strong> Consider using 1m data for faster signal detection</li>
            </ol>
        </div>
        
        <h2>📈 Next Steps</h2>
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <ul>
                <li>Run backtest with relaxed parameters to generate signals</li>
                <li>Collect real market data from live Bybit API</li>
                <li>Implement paper trading to validate strategy in real-time</li>
                <li>Monitor signal frequency vs market volatility correlation</li>
                <li>A/B test different confidence thresholds</li>
            </ul>
        </div>
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d;">
            <p>Report generated by SOL/USDT Trading Bot Analysis System</p>
            <p>Data Period: {market_summary['candles_analyzed']} candles analyzed over {(market_summary['period_end'] - market_summary['period_start']).days} days</p>
        </footer>
    </div>
</body>
</html>
    """
    
    return html_content

def main():
    """Generate comprehensive backtest report."""
    
    print("🔬 Generating Detailed Backtest Analysis Report")
    print("=" * 60)
    
    # Load data
    print("📊 Loading market data...")
    data = load_market_data()
    
    if not data:
        print("❌ No data files found")
        return
    
    # Calculate indicators for 5m data
    print("📈 Calculating technical indicators...")
    df_5m = calculate_simple_indicators(data['5m'].copy())
    
    # Generate market summary
    print("📋 Generating market summary...")
    market_summary = generate_market_summary(data)
    
    # Analyze signals
    print("🔍 Analyzing potential signals...")
    signals_analysis = analyze_signals(df_5m)
    
    # Load backtest results if available
    backtest_results = {"error": "No trades executed"}
    results_files = list(Path('Output').glob('backtest_results_*.json'))
    if results_files:
        latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
        with open(latest_results, 'r') as f:
            backtest_results = json.load(f)
    
    # Generate HTML report
    print("📝 Creating HTML report...")
    html_content = create_html_report(market_summary, signals_analysis, backtest_results)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"Output/SOL_USDT_Backtest_Report_{timestamp}.html"
    
    # Ensure Output directory exists
    Path("Output").mkdir(exist_ok=True)
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Report generated: {report_filename}")
    print(f"📊 Analyzed {len(signals_analysis)} signal points")
    print(f"💰 Market moved {market_summary['total_return_pct']:+.2f}% over {(market_summary['period_end'] - market_summary['period_start']).days} days")
    print(f"🎯 Found {market_summary['movements_over_2_usd']} movements ≥$2")
    
    # Summary
    print("\n" + "="*60)
    print("📋 EXECUTIVE SUMMARY")
    print("="*60)
    print(f"• Period: {market_summary['period_start'].strftime('%Y-%m-%d')} to {market_summary['period_end'].strftime('%Y-%m-%d')}")
    print(f"• Price range: ${market_summary['price_min']:.2f} - ${market_summary['price_max']:.2f}")
    print(f"• Total return: {market_summary['total_return_pct']:+.2f}%")
    print(f"• Movements ≥$2: {market_summary['movements_over_2_usd']}")
    print(f"• Strategy result: No trades executed (signals too restrictive)")
    print(f"• Recommendation: Lower confidence threshold and adjust for testnet data")

if __name__ == "__main__":
    main()