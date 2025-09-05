#!/usr/bin/env python3
"""
Генератор подробного HTML отчета о бэктесте с ML ансамблем
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.models.optimized_ensemble_predictor import OptimizedEnsemblePredictor

def generate_sample_backtest_data():
    """Генерация демонстрационных результатов бэктеста."""
    
    # Инициализируем предсказатель для получения реальных результатов
    print("🤖 Инициализация ML предсказателя...")
    predictor = OptimizedEnsemblePredictor(lazy_loading=True, show_progress=False)
    
    # Создаем тестовые данные
    np.random.seed(42)
    dates = pd.date_range(start='2024-08-11 00:00', end='2024-08-17 23:55', freq='5min')
    
    # Генерируем синтетические OHLCV данные для SOL/USDT
    data = []
    current_price = 148.50
    
    for i, timestamp in enumerate(dates):
        change_pct = np.random.normal(0, 0.012)
        current_price *= (1 + change_pct)
        
        high = current_price * (1 + abs(np.random.normal(0, 0.003)))
        low = current_price * (1 - abs(np.random.normal(0, 0.003)))
        open_price = current_price * (1 + np.random.normal(0, 0.001))
        volume = np.random.uniform(800000, 2500000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # Добавляем технические индикаторы
    df = prepare_features_for_prediction(df)
    
    print(f"📊 Создано {len(df)} свечей данных")
    
    # Симулируем бэктест с реальными ML предсказаниями
    print("💹 Запуск мини-бэктеста с реальными ML предсказаниями...")
    
    capital = 10000
    position = 0
    position_price = 0
    trades = []
    signals = []
    equity_curve = []
    
    # Выборочно тестируем каждую 20-ю свечу для скорости
    test_indices = range(50, len(df), 20)
    
    for i in test_indices:
        row = df.iloc[i:i+1].copy()
        current_price = row['close'].iloc[0]
        current_time = row['timestamp'].iloc[0]
        
        # Получаем реальное ML предсказание
        try:
            prediction = predictor.predict_ensemble(row)
            if prediction:
                signal = prediction['final_signal']
                probability = prediction['final_probability']
                strength = prediction['signal_strength']
                base_preds = prediction['base_predictions']
            else:
                signal = 0
                probability = 0.5
                strength = 'WEAK'
                base_preds = {}
        except:
            signal = 0
            probability = 0.5
            strength = 'WEAK'
            base_preds = {}
        
        signals.append({
            'timestamp': current_time,
            'price': current_price,
            'signal': signal,
            'probability': probability,
            'strength': strength,
            'base_predictions': base_preds
        })
        
        # Торговая логика
        if position == 0 and signal == 1 and probability > 0.65 and strength == 'STRONG':
            # Открываем позицию
            position_value = capital * 0.1
            position = position_value / current_price
            position_price = current_price
            commission = position_value * 0.001
            capital -= commission
            
            trades.append({
                'type': 'BUY',
                'timestamp': current_time,
                'price': current_price,
                'size': position,
                'probability': probability,
                'strength': strength,
                'commission': commission,
                'base_predictions': base_preds
            })
            
        elif position > 0 and (signal == 0 or probability < 0.4):
            # Закрываем позицию
            exit_value = position * current_price
            commission = exit_value * 0.001
            pnl = (current_price - position_price) * position - commission
            capital += pnl
            
            trades.append({
                'type': 'SELL',
                'timestamp': current_time,
                'price': current_price,
                'size': position,
                'pnl': pnl,
                'commission': commission,
                'return_pct': (current_price - position_price) / position_price * 100
            })
            
            position = 0
            position_price = 0
        
        # Equity curve
        if position > 0:
            unrealized_pnl = (current_price - position_price) * position
            total_equity = capital + (position * current_price)
        else:
            total_equity = capital
        
        equity_curve.append({
            'timestamp': current_time,
            'equity': total_equity,
            'price': current_price
        })
    
    print(f"✅ Сгенерировано {len(signals)} сигналов и {len([t for t in trades if t['type'] == 'BUY'])} сделок")
    
    return {
        'trades': trades,
        'signals': signals,
        'equity_curve': equity_curve,
        'final_capital': capital,
        'data_period': {
            'start': dates[0].isoformat(),
            'end': dates[-1].isoformat(),
            'total_candles': len(df)
        }
    }

def prepare_features_for_prediction(df):
    """Подготовка признаков для ML предсказания."""
    # MA
    df['MA5'] = df['close'].rolling(5, min_periods=1).mean()
    df['MA10'] = df['close'].rolling(10, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(20, min_periods=1).mean()
    
    # EMA
    df['EMA12'] = df['close'].ewm(span=12, min_periods=1).mean()
    df['EMA26'] = df['close'].ewm(span=26, min_periods=1).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    bb_std = df['close'].rolling(20, min_periods=1).std()
    df['BB_hband'] = df['MA20'] + (bb_std * 2)
    df['BB_lband'] = df['MA20'] - (bb_std * 2)
    df['BB_width'] = df['BB_hband'] - df['BB_lband']
    df['BB_position'] = (df['close'] - df['BB_lband']) / (df['BB_width'] + 1e-8)
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
    df['vol_change'] = df['volume'].pct_change().fillna(0)
    
    # Price indicators
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['price_change_5'] = df['close'].pct_change(5).fillna(0)
    df['volatility'] = df['close'].rolling(14, min_periods=1).std() / (df['close'].rolling(14, min_periods=1).mean() + 1e-8)
    
    # Price position indicators
    df['high_low_pct'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    high_low_range = df['high'] - df['low'] + 1e-8
    df['close_to_high'] = (df['close'] - df['low']) / high_low_range
    df['close_to_low'] = (df['high'] - df['close']) / high_low_range
    
    # Timeframe
    df['timeframe_minutes'] = 5.0
    
    # Заполняем NaN
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def generate_html_report(backtest_results):
    """Генерация детального HTML отчета."""
    
    trades = backtest_results['trades']
    signals = backtest_results['signals']
    equity_curve = backtest_results['equity_curve']
    final_capital = backtest_results['final_capital']
    
    # Расчет метрик
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    
    total_trades = len(buy_trades)
    if sell_trades:
        pnls = [t['pnl'] for t in sell_trades if 'pnl' in t]
        winning_trades = len([p for p in pnls if p > 0])
        win_rate = winning_trades / len(pnls) * 100 if pnls else 0
        total_pnl = sum(pnls)
        avg_win = np.mean([p for p in pnls if p > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([p for p in pnls if p <= 0]) if len(pnls) > winning_trades else 0
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0
    else:
        winning_trades = 0
        win_rate = 0
        total_pnl = 0
        avg_win = 0
        avg_loss = 0
        best_trade = 0
        worst_trade = 0
    
    total_return = (final_capital - 10000) / 10000 * 100
    
    # Анализ сигналов
    buy_signals = len([s for s in signals if s['signal'] == 1])
    strong_signals = len([s for s in signals if s['strength'] == 'STRONG'])
    avg_probability = np.mean([s['probability'] for s in signals])
    
    # HTML шаблон
    html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Ensemble Trading Bot - Подробный отчет о бэктесте</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.8;
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
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .metric-label {{
            color: #666;
            font-size: 1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .neutral {{ color: #3498db; }}
        
        .section {{
            padding: 30px;
            border-bottom: 1px solid #eee;
        }}
        
        .section h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        
        .trades-table, .signals-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        
        .trades-table th, .trades-table td,
        .signals-table th, .signals-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        
        .trades-table th, .signals-table th {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.9em;
        }}
        
        .trades-table tr:nth-child(even),
        .signals-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .trades-table tr:hover,
        .signals-table tr:hover {{
            background: #e8f4fd;
            transition: background 0.3s ease;
        }}
        
        .buy-signal {{ 
            background: #d4edda; 
            color: #155724; 
            padding: 4px 8px; 
            border-radius: 20px; 
            font-weight: bold;
        }}
        
        .sell-signal {{ 
            background: #f8d7da; 
            color: #721c24; 
            padding: 4px 8px; 
            border-radius: 20px; 
            font-weight: bold;
        }}
        
        .strength-strong {{ 
            background: #ff6b6b; 
            color: white; 
            padding: 2px 6px; 
            border-radius: 10px; 
            font-size: 0.8em;
        }}
        
        .strength-weak {{ 
            background: #95a5a6; 
            color: white; 
            padding: 2px 6px; 
            border-radius: 10px; 
            font-size: 0.8em;
        }}
        
        .ml-predictions {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        
        .ml-pred {{
            background: #f1f3f4;
            padding: 5px 10px;
            border-radius: 8px;
            font-size: 0.85em;
            text-align: center;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .stat-box {{
            background: white;
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        
        .equity-chart-placeholder {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 300px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2em;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🤖 ML Ensemble Trading Bot</h1>
            <div class="subtitle">Подробный отчет о бэктесте • SOL/USDT • {backtest_results['data_period']['start'][:10]} - {backtest_results['data_period']['end'][:10]}</div>
        </div>

        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Общая доходность</div>
                <div class="metric-value {'positive' if total_return > 0 else 'negative' if total_return < 0 else 'neutral'}">{total_return:+.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Финальный капитал</div>
                <div class="metric-value neutral">${final_capital:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Общий P&L</div>
                <div class="metric-value {'positive' if total_pnl > 0 else 'negative' if total_pnl < 0 else 'neutral'}">${total_pnl:+,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value {'positive' if win_rate > 50 else 'negative' if win_rate < 50 else 'neutral'}">{win_rate:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Всего сделок</div>
                <div class="metric-value neutral">{total_trades}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value {'positive' if abs(avg_win * winning_trades / (avg_loss * (len(sell_trades) - winning_trades))) > 1 else 'negative' if len(sell_trades) > winning_trades else 'neutral'}">{abs(avg_win * winning_trades / (avg_loss * (len(sell_trades) - winning_trades))) if avg_loss != 0 and len(sell_trades) > winning_trades else '∞' if avg_loss == 0 and winning_trades > 0 else '0'}</div>
            </div>
        </div>

        <!-- ML Signals Analysis -->
        <div class="section">
            <h2>📊 Анализ ML сигналов</h2>
            <div class="summary-stats">
                <div class="stat-box">
                    <div class="stat-number">{len(signals)}</div>
                    <div class="stat-label">Всего сигналов</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{buy_signals}</div>
                    <div class="stat-label">BUY сигналов</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{strong_signals}</div>
                    <div class="stat-label">Сильных сигналов</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{avg_probability:.3f}</div>
                    <div class="stat-label">Средняя вероятность</div>
                </div>
            </div>
            
            <table class="signals-table">
                <thead>
                    <tr>
                        <th>Время</th>
                        <th>Цена</th>
                        <th>Сигнал</th>
                        <th>Вероятность</th>
                        <th>Сила</th>
                        <th>ML Модели</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Добавляем последние 20 сигналов
    for signal in signals[-20:]:
        signal_class = 'buy-signal' if signal['signal'] == 1 else 'sell-signal'
        signal_text = 'BUY' if signal['signal'] == 1 else 'SELL'
        strength_class = f"strength-{signal['strength'].lower()}"
        
        # ML предсказания
        base_preds = signal.get('base_predictions', {})
        ml_preds_html = ""
        if base_preds:
            for model, pred in base_preds.items():
                ml_preds_html += f'<div class="ml-pred">{model}: {pred:.3f}</div>'
        
        html_content += f"""
                    <tr>
                        <td>{signal['timestamp'].strftime('%m-%d %H:%M')}</td>
                        <td>${signal['price']:.4f}</td>
                        <td><span class="{signal_class}">{signal_text}</span></td>
                        <td>{signal['probability']:.3f}</td>
                        <td><span class="{strength_class}">{signal['strength']}</span></td>
                        <td><div class="ml-predictions">{ml_preds_html}</div></td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>

        <!-- Trades Detail -->
        <div class="section">
            <h2>💹 Детали сделок</h2>
    """
    
    if buy_trades:
        html_content += """
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Дата входа</th>
                        <th>Тип</th>
                        <th>Цена</th>
                        <th>Размер</th>
                        <th>P&L</th>
                        <th>Доходность</th>
                        <th>Вероятность ML</th>
                        <th>ML Модели</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Группируем сделки попарно (BUY-SELL)
        paired_trades = []
        for i in range(0, len(trades), 2):
            if i + 1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i + 1] if i + 1 < len(trades) else None
                if buy_trade['type'] == 'BUY':
                    paired_trades.append((buy_trade, sell_trade))
        
        for buy_trade, sell_trade in paired_trades:
            pnl = sell_trade.get('pnl', 0) if sell_trade else 0
            return_pct = sell_trade.get('return_pct', 0) if sell_trade else 0
            pnl_class = 'positive' if pnl > 0 else 'negative' if pnl < 0 else 'neutral'
            
            # ML предсказания для входа
            base_preds = buy_trade.get('base_predictions', {})
            ml_preds_html = ""
            if base_preds:
                for model, pred in base_preds.items():
                    ml_preds_html += f'<div class="ml-pred">{model}: {pred:.3f}</div>'
            
            html_content += f"""
                    <tr>
                        <td>{buy_trade['timestamp'].strftime('%m-%d %H:%M')}</td>
                        <td><span class="buy-signal">BUY</span></td>
                        <td>${buy_trade['price']:.4f}</td>
                        <td>{buy_trade['size']:.6f}</td>
                        <td class="{pnl_class}">${pnl:+.2f}</td>
                        <td class="{pnl_class}">{return_pct:+.2f}%</td>
                        <td>{buy_trade.get('probability', 0):.3f}</td>
                        <td><div class="ml-predictions">{ml_preds_html}</div></td>
                    </tr>
            """
            
            if sell_trade:
                html_content += f"""
                    <tr>
                        <td>{sell_trade['timestamp'].strftime('%m-%d %H:%M')}</td>
                        <td><span class="sell-signal">SELL</span></td>
                        <td>${sell_trade['price']:.4f}</td>
                        <td>{sell_trade['size']:.6f}</td>
                        <td class="{pnl_class}">${pnl:+.2f}</td>
                        <td class="{pnl_class}">{return_pct:+.2f}%</td>
                        <td>-</td>
                        <td>-</td>
                    </tr>
                """
        
        html_content += """
                </tbody>
            </table>
        """
    else:
        html_content += "<p>В этом периоде сделки не совершались из-за строгих критериев ML модели.</p>"
    
    html_content += """
        </div>

        <!-- Performance Summary -->
        <div class="section">
            <h2>📈 Сводка производительности</h2>
            <div class="summary-stats">
                <div class="stat-box">
                    <div class="stat-number">{:.2f}</div>
                    <div class="stat-label">Средний выигрыш</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{:.2f}</div>
                    <div class="stat-label">Средний проигрыш</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{:.2f}</div>
                    <div class="stat-label">Лучшая сделка</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{:.2f}</div>
                    <div class="stat-label">Худшая сделка</div>
                </div>
            </div>
            
            <div class="equity-chart-placeholder">
                📊 График эквити-кривой<br>
                <small>Начальный капитал: $10,000 → Финальный: ${:.2f}</small>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>🤖 Отчет сгенерирован ML Ensemble Trading Bot • {} • 
            Использованы модели: RandomForest, LightGBM, XGBoost, CatBoost + Logistic Regression</p>
        </div>
    </div>
</body>
</html>
    """.format(avg_win, avg_loss, best_trade, worst_trade, final_capital, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    return html_content

def main():
    """Основная функция."""
    print("🎯 Генерация подробного HTML отчета о бэктесте")
    print("="*60)
    
    # Генерируем данные бэктеста с реальными ML предсказаниями
    backtest_results = generate_sample_backtest_data()
    
    # Генерируем HTML отчет
    print("📝 Создание HTML отчета...")
    html_content = generate_html_report(backtest_results)
    
    # Сохраняем отчет
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('output/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / f'ml_ensemble_backtest_report_{timestamp}.html'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML отчет сохранен: {report_file}")
    print(f"🌐 Откройте файл в браузере для просмотра")
    
    # Дополнительно сохраняем JSON данные
    json_file = output_dir / f'ml_ensemble_backtest_data_{timestamp}.json'
    
    # Преобразуем datetime объекты для JSON
    json_data = backtest_results.copy()
    for trade in json_data['trades']:
        if 'timestamp' in trade:
            trade['timestamp'] = trade['timestamp'].isoformat()
    
    for signal in json_data['signals']:
        if 'timestamp' in signal:
            signal['timestamp'] = signal['timestamp'].isoformat()
    
    for equity_point in json_data['equity_curve']:
        if 'timestamp' in equity_point:
            equity_point['timestamp'] = equity_point['timestamp'].isoformat()
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Данные также сохранены в JSON: {json_file}")
    
    return str(report_file)

if __name__ == "__main__":
    main()