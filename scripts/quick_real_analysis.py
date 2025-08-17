#!/usr/bin/env python3
"""Quick analysis of real SOL/USDT data with simplified backtesting."""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt

def load_real_data():
    """Load real market data."""
    data_files = {
        '1m': 'data/SOLUSDT_1m_real_7d.csv',
        '5m': 'data/SOLUSDT_5m_real_7d.csv', 
        '15m': 'data/SOLUSDT_15m_real_7d.csv',
        '1h': 'data/SOLUSDT_1h_real_7d.csv'
    }
    
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

def calculate_technical_indicators(df, fast_mode=True):
    """Calculate technical indicators optimized for speed."""
    
    # EMAs (faster than SMAs)
    df['ema_8'] = df['close'].ewm(span=8).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_55'] = df['close'].ewm(span=55).mean()
    
    # RSI (simplified)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume analysis
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # ATR (simplified)
    df['tr'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['close'].shift(1)),
                                   abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    # Price momentum
    df['price_change_5m'] = df['close'].diff()
    df['price_change_1h'] = df['close'].diff(12)  # For 5m data
    
    return df

def simple_signal_analysis(df):
    """Simple but fast signal analysis."""
    
    signals = []
    trades = []
    
    # Simple strategy parameters
    min_atr_pct = 0.5  # Minimum volatility
    min_volume_ratio = 1.5
    position_size = 200  # Fixed size for simplicity
    
    current_position = None
    
    for i in range(55, len(df)):  # Start after indicators are calculated
        row = df.iloc[i]
        timestamp = df.index[i]
        
        if pd.isna(row['ema_8']) or pd.isna(row['ema_21']) or pd.isna(row['ema_55']):
            continue
        
        # Check if we have minimum volatility and volume
        if row['atr_pct'] < min_atr_pct or row['volume_ratio'] < min_volume_ratio:
            continue
        
        # Simple trend following signals
        bullish = (row['ema_8'] > row['ema_21'] > row['ema_55'] and 
                  row['close'] > row['ema_8'] and
                  row['rsi'] > 45 and row['rsi'] < 75)
        
        bearish = (row['ema_8'] < row['ema_21'] < row['ema_55'] and 
                  row['close'] < row['ema_8'] and
                  row['rsi'] > 25 and row['rsi'] < 55)
        
        # Entry logic
        if current_position is None:
            if bullish:
                current_position = {
                    'type': 'BUY',
                    'entry_time': timestamp,
                    'entry_price': row['close'],
                    'stop_loss': row['close'] - row['atr'] * 1.5,
                    'take_profit': row['close'] + row['atr'] * 2.0,
                    'size': position_size
                }
                signals.append({
                    'timestamp': timestamp,
                    'type': 'BUY',
                    'price': row['close'],
                    'confidence': 0.7,  # Simplified
                    'reasoning': f"EMA trend up, RSI {row['rsi']:.1f}, Vol {row['volume_ratio']:.1f}x"
                })
            
            elif bearish:
                current_position = {
                    'type': 'SELL',
                    'entry_time': timestamp,
                    'entry_price': row['close'],
                    'stop_loss': row['close'] + row['atr'] * 1.5,
                    'take_profit': row['close'] - row['atr'] * 2.0,
                    'size': position_size
                }
                signals.append({
                    'timestamp': timestamp,
                    'type': 'SELL', 
                    'price': row['close'],
                    'confidence': 0.7,
                    'reasoning': f"EMA trend down, RSI {row['rsi']:.1f}, Vol {row['volume_ratio']:.1f}x"
                })
        
        # Exit logic
        elif current_position is not None:
            exit_trade = False
            exit_reason = ""
            
            if current_position['type'] == 'BUY':
                if row['close'] <= current_position['stop_loss']:
                    exit_trade = True
                    exit_reason = "Stop Loss"
                elif row['close'] >= current_position['take_profit']:
                    exit_trade = True
                    exit_reason = "Take Profit"
                elif (timestamp - current_position['entry_time']).total_seconds() > 4 * 3600:  # 4 hours
                    exit_trade = True
                    exit_reason = "Time Stop"
            
            elif current_position['type'] == 'SELL':
                if row['close'] >= current_position['stop_loss']:
                    exit_trade = True
                    exit_reason = "Stop Loss"
                elif row['close'] <= current_position['take_profit']:
                    exit_trade = True
                    exit_reason = "Take Profit"
                elif (timestamp - current_position['entry_time']).total_seconds() > 4 * 3600:
                    exit_trade = True
                    exit_reason = "Time Stop"
            
            if exit_trade:
                # Calculate P&L
                if current_position['type'] == 'BUY':
                    pnl = (row['close'] - current_position['entry_price']) * current_position['size'] / current_position['entry_price']
                else:
                    pnl = (current_position['entry_price'] - row['close']) * current_position['size'] / current_position['entry_price']
                
                trade = {
                    'entry_time': current_position['entry_time'],
                    'exit_time': timestamp,
                    'type': current_position['type'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': row['close'],
                    'size': current_position['size'],
                    'pnl_usd': pnl,
                    'pnl_pct': pnl / current_position['size'] * 100,
                    'duration_hours': (timestamp - current_position['entry_time']).total_seconds() / 3600,
                    'exit_reason': exit_reason
                }
                
                trades.append(trade)
                signals.append({
                    'timestamp': timestamp,
                    'type': 'EXIT',
                    'price': row['close'],
                    'confidence': 1.0,
                    'reasoning': f"Exit {current_position['type']} - {exit_reason}"
                })
                
                current_position = None
    
    return signals, trades

def analyze_market_performance(data):
    """Analyze market performance from real data."""
    
    df_5m = data['5m'].copy()
    
    # Calculate comprehensive market stats
    price_start = df_5m['close'].iloc[0]
    price_end = df_5m['close'].iloc[-1]
    price_min = df_5m['low'].min()
    price_max = df_5m['high'].max()
    
    # Returns analysis
    returns = df_5m['close'].pct_change().dropna()
    
    # Price movements
    price_changes = df_5m['close'].diff().abs()
    movements_over_1 = (price_changes >= 1.0).sum()
    movements_over_2 = (price_changes >= 2.0).sum()
    movements_over_5 = (price_changes >= 5.0).sum()
    
    # Volume analysis
    avg_volume = df_5m['volume'].mean()
    high_volume_candles = (df_5m['volume'] > avg_volume * 2).sum()
    
    # Volatility periods
    df_5m['atr'] = df_5m['high'].sub(df_5m['low']).rolling(14).mean()
    high_volatility = (df_5m['atr'] > df_5m['atr'].quantile(0.8)).sum()
    
    return {
        'period_start': df_5m.index[0],
        'period_end': df_5m.index[-1],
        'total_hours': (df_5m.index[-1] - df_5m.index[0]).total_seconds() / 3600,
        'price_start': price_start,
        'price_end': price_end,
        'price_min': price_min,
        'price_max': price_max,
        'total_return_pct': (price_end / price_start - 1) * 100,
        'price_range': price_max - price_min,
        'max_5min_gain': returns.max() * 100,
        'max_5min_loss': returns.min() * 100,
        'avg_5min_return': returns.mean() * 100,
        'volatility_pct': returns.std() * np.sqrt(288) * 100,  # Annualized
        'movements_over_1_usd': movements_over_1,
        'movements_over_2_usd': movements_over_2,
        'movements_over_5_usd': movements_over_5,
        'max_5min_move': price_changes.max(),
        'avg_volume': avg_volume,
        'high_volume_periods': high_volume_candles,
        'high_volatility_periods': high_volatility,
        'total_candles': len(df_5m)
    }

def create_quick_report(market_stats, signals, trades):
    """Create quick analysis report."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Trading performance
    if trades:
        winning_trades = [t for t in trades if t['pnl_usd'] > 0]
        losing_trades = [t for t in trades if t['pnl_usd'] <= 0]
        
        total_pnl = sum(t['pnl_usd'] for t in trades)
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl_usd'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_usd'] for t in losing_trades]) if losing_trades else 0
        avg_duration = np.mean([t['duration_hours'] for t in trades])
        
        trade_stats = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_duration_hours': avg_duration,
            'profit_factor': abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if avg_loss != 0 else 0
        }
    else:
        trade_stats = {
            'total_trades': 0,
            'error': 'No trades executed'
        }
    
    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SOL/USDT Real Data Quick Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-card h3 {{ margin: 0; font-size: 14px; opacity: 0.9; }}
        .stat-card .value {{ font-size: 20px; font-weight: bold; margin: 5px 0; }}
        .trades-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .trades-table th, .trades-table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        .trades-table th {{ background-color: #f2f2f2; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 SOL/USDT Real Market Data Analysis</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>🏪 Market Overview</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Analysis Period</h3>
                <div class="value">{market_stats['total_hours']:.1f}h</div>
                <small>{market_stats['period_start'].strftime('%m-%d %H:%M')} - {market_stats['period_end'].strftime('%m-%d %H:%M')}</small>
            </div>
            <div class="stat-card">
                <h3>Price Range</h3>
                <div class="value">${market_stats['price_min']:.2f} - ${market_stats['price_max']:.2f}</div>
                <small>Range: ${market_stats['price_range']:.2f}</small>
            </div>
            <div class="stat-card">
                <h3>Total Return</h3>
                <div class="value">{market_stats['total_return_pct']:+.2f}%</div>
                <small>${market_stats['price_start']:.2f} → ${market_stats['price_end']:.2f}</small>
            </div>
            <div class="stat-card">
                <h3>Max 5min Move</h3>
                <div class="value">${market_stats['max_5min_move']:.2f}</div>
                <small>Volatility: {market_stats['volatility_pct']:.1f}%</small>
            </div>
            <div class="stat-card">
                <h3>Movements ≥$1</h3>
                <div class="value">{market_stats['movements_over_1_usd']}</div>
                <small>≥$2: {market_stats['movements_over_2_usd']}, ≥$5: {market_stats['movements_over_5_usd']}</small>
            </div>
            <div class="stat-card">
                <h3>Average Volume</h3>
                <div class="value">{market_stats['avg_volume']:,.0f}</div>
                <small>High vol periods: {market_stats['high_volume_periods']}</small>
            </div>
        </div>
        
        <h2>⚡ Trading Performance</h2>
    """
    
    if trade_stats['total_trades'] > 0:
        html_content += f"""
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Trades</h3>
                <div class="value">{trade_stats['total_trades']}</div>
                <small>Signals: {len(signals)}</small>
            </div>
            <div class="stat-card">
                <h3>Win Rate</h3>
                <div class="value">{trade_stats['win_rate']:.1%}</div>
                <small>{trade_stats['winning_trades']}W / {trade_stats['losing_trades']}L</small>
            </div>
            <div class="stat-card">
                <h3>Total P&L</h3>
                <div class="value">${trade_stats['total_pnl']:+.2f}</div>
                <small>Avg: ${trade_stats['total_pnl']/trade_stats['total_trades']:+.2f}</small>
            </div>
            <div class="stat-card">
                <h3>Avg Duration</h3>
                <div class="value">{trade_stats['avg_duration_hours']:.1f}h</div>
                <small>Position time</small>
            </div>
            <div class="stat-card">
                <h3>Profit Factor</h3>
                <div class="value">{trade_stats['profit_factor']:.2f}</div>
                <small>Risk/Reward</small>
            </div>
        </div>
        
        <h2>📋 Trade History</h2>
        <table class="trades-table">
            <thead>
                <tr>
                    <th>Entry Time</th>
                    <th>Type</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>P&L</th>
                    <th>Duration</th>
                    <th>Exit Reason</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for trade in trades[-10:]:  # Show last 10 trades
            pnl_class = "positive" if trade['pnl_usd'] >= 0 else "negative"
            html_content += f"""
                <tr>
                    <td>{trade['entry_time'].strftime('%m-%d %H:%M')}</td>
                    <td>{trade['type']}</td>
                    <td>${trade['entry_price']:.2f}</td>
                    <td>${trade['exit_price']:.2f}</td>
                    <td class="{pnl_class}">${trade['pnl_usd']:+.2f}</td>
                    <td>{trade['duration_hours']:.1f}h</td>
                    <td>{trade['exit_reason']}</td>
                </tr>
            """
        
        html_content += """
            </tbody>
        </table>
        """
    else:
        html_content += """
        <div class="summary">
            <h3>❌ No Trades Executed</h3>
            <p>The strategy did not generate any trades during this period. This could indicate:</p>
            <ul>
                <li>Market conditions didn't meet entry criteria</li>
                <li>Strategy parameters are too conservative</li>
                <li>Insufficient volatility or volume</li>
            </ul>
        </div>
        """
    
    html_content += f"""
        <h2>📈 Key Insights</h2>
        <div class="summary">
            <h4>Market Characteristics:</h4>
            <ul>
                <li><strong>Price volatility:</strong> {market_stats['volatility_pct']:.1f}% annualized - {'High' if market_stats['volatility_pct'] > 10 else 'Moderate' if market_stats['volatility_pct'] > 5 else 'Low'}</li>
                <li><strong>Movement frequency:</strong> {market_stats['movements_over_2_usd']} movements ≥$2 over {market_stats['total_hours']:.1f} hours</li>
                <li><strong>Trading opportunities:</strong> {'Excellent' if market_stats['movements_over_2_usd'] > 20 else 'Good' if market_stats['movements_over_2_usd'] > 10 else 'Limited'} based on $2+ movements</li>
                <li><strong>Volume activity:</strong> {market_stats['high_volume_periods']} high-volume periods detected</li>
            </ul>
            
            <h4>Strategy Performance:</h4>
            <ul>
                <li><strong>Signal generation:</strong> {len(signals)} total signals over {market_stats['total_hours']:.1f} hours</li>
                <li><strong>Trade execution:</strong> {trade_stats.get('total_trades', 0)} trades completed</li>
                {'<li><strong>Profitability:</strong> ' + ('Profitable' if trade_stats.get('total_pnl', 0) > 0 else 'Unprofitable') + f" with {trade_stats.get('win_rate', 0):.1%} win rate</li>" if trade_stats.get('total_trades', 0) > 0 else '<li><strong>Issue:</strong> No trades executed - strategy may need optimization</li>'}
            </ul>
        </div>
        
        <footer style="margin-top: 30px; text-align: center; color: #7f8c8d;">
            <p>Real market data analysis • {market_stats['total_candles']} candles analyzed</p>
        </footer>
    </div>
</body>
</html>
    """
    
    # Save report
    Path("Output").mkdir(exist_ok=True)
    report_filename = f"Output/SOL_USDT_Quick_Analysis_{timestamp}.html"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_filename, trade_stats

def main():
    """Run quick analysis of real market data."""
    
    print("⚡ Quick Real Market Data Analysis")
    print("=" * 50)
    
    # Load real data
    print("📊 Loading real market data...")
    data = load_real_data()
    
    if '5m' not in data:
        print("❌ No 5m data found")
        return
    
    # Calculate indicators
    print("📈 Calculating technical indicators...")
    df_5m = calculate_technical_indicators(data['5m'].copy())
    
    # Analyze market
    print("🔍 Analyzing market performance...")
    market_stats = analyze_market_performance(data)
    
    # Run simple strategy
    print("⚡ Running simplified strategy...")
    signals, trades = simple_signal_analysis(df_5m)
    
    # Generate report
    print("📝 Creating analysis report...")
    report_filename, trade_stats = create_quick_report(market_stats, signals, trades)
    
    print(f"\n✅ Quick analysis completed!")
    print(f"📄 Report saved: {report_filename}")
    
    # Console summary
    print(f"\n📊 QUICK SUMMARY")
    print("=" * 30)
    print(f"• Analysis period: {market_stats['total_hours']:.1f} hours")
    print(f"• Price range: ${market_stats['price_min']:.2f} - ${market_stats['price_max']:.2f}")
    print(f"• Total return: {market_stats['total_return_pct']:+.2f}%")
    print(f"• Movements ≥$2: {market_stats['movements_over_2_usd']}")
    print(f"• Signals generated: {len(signals)}")
    print(f"• Trades executed: {trade_stats.get('total_trades', 0)}")
    
    if trade_stats.get('total_trades', 0) > 0:
        print(f"• Win rate: {trade_stats['win_rate']:.1%}")
        print(f"• Total P&L: ${trade_stats['total_pnl']:+.2f}")
    
    return report_filename

if __name__ == "__main__":
    main()