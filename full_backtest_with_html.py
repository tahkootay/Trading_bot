#!/usr/bin/env python3
"""
Полный бэктест с ML ансамблем и генерацией подробного HTML отчета
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

from src.models.optimized_ensemble_predictor import OptimizedEnsemblePredictor, ProgressBar

def load_real_data():
    """Загрузка реальных данных SOL/USDT за 7 дней."""
    print("📥 Загрузка реальных данных SOL/USDT...")
    
    data_file = None
    possible_files = [
        "data/SOLUSDT_5m_real_2025-08-10_to_2025-08-17.csv",
        "data/bybit_futures_solusdt_5m.csv",
        "data/SOLUSDT_5m.csv"
    ]
    
    for file_path in possible_files:
        if Path(file_path).exists():
            data_file = file_path
            break
    
    if not data_file:
        print("❌ Файлы данных не найдены!")
        return None
    
    print(f"📂 Загрузка из: {data_file}")
    data = pd.read_csv(data_file)
    
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    else:
        data['timestamp'] = pd.date_range(start='2024-08-11', periods=len(data), freq='5min')
    
    # Берем данные за последние 7 дней (ограничиваем до 2000 свечей для разумного времени выполнения)
    if len(data) > 2000:
        data = data.tail(2000).copy()
    
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Загружено {len(data)} свечей")
    print(f"📅 Период: {data['timestamp'].min()} - {data['timestamp'].max()}")
    
    return data

def prepare_technical_indicators(df):
    """Подготовка технических индикаторов для ML."""
    print("📊 Расчет технических индикаторов...")
    
    # Moving Averages
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
    
    print("✅ Технические индикаторы рассчитаны")
    return df

def run_full_backtest(data):
    """Запуск полного бэктеста с ML ансамблем."""
    print("🤖 Запуск полного бэктеста с ML ансамблем...")
    
    # Инициализация ML предсказателя
    predictor = OptimizedEnsemblePredictor(lazy_loading=True, show_progress=True)
    
    # Торговые параметры
    initial_capital = 10000
    position_size_pct = 0.1
    commission_rate = 0.001
    slippage = 0.0005
    
    # Состояние торговли
    capital = initial_capital
    position = 0
    position_price = 0
    position_entry_time = None
    
    # Результаты
    trades = []
    signals = []
    equity_curve = []
    
    # Пропускаем первые 50 свечей для стабилизации индикаторов
    start_idx = 50
    total_candles = len(data) - start_idx
    
    print(f"💹 Обработка {total_candles} свечей...")
    
    # Создаем прогресс-бар
    progress = ProgressBar(total_candles, "🔄 Полный бэктест")
    
    backtest_start_time = time.time()
    prediction_times = []
    
    for i in range(start_idx, len(data)):
        current_row = data.iloc[i:i+1].copy()
        current_price = current_row['close'].iloc[0]
        current_time = current_row['timestamp'].iloc[0]
        
        # Получаем ML предсказание
        pred_start = time.time()
        try:
            prediction = predictor.predict_ensemble(current_row)
            pred_time = time.time() - pred_start
            prediction_times.append(pred_time)
            
            if prediction:
                signal = prediction['final_signal']
                probability = prediction['final_probability']
                strength = prediction['signal_strength']
                base_predictions = prediction['base_predictions']
            else:
                signal = 0
                probability = 0.5
                strength = 'WEAK'
                base_predictions = {}
        except Exception as e:
            signal = 0
            probability = 0.5
            strength = 'WEAK'
            base_predictions = {}
            pred_time = time.time() - pred_start
            prediction_times.append(pred_time)
        
        # Записываем сигнал
        signals.append({
            'timestamp': current_time,
            'price': current_price,
            'signal': signal,
            'probability': probability,
            'strength': strength,
            'base_predictions': base_predictions
        })
        
        # Торговая логика
        if position == 0:  # Нет позиции
            # Условия входа: BUY сигнал с высокой вероятностью
            if signal == 1 and probability > 0.7 and strength == 'STRONG':
                # Открываем лонг позицию
                position_value = capital * position_size_pct
                entry_price = current_price * (1 + slippage)
                position = position_value / entry_price
                position_price = entry_price
                position_entry_time = current_time
                
                commission = position_value * commission_rate
                capital -= commission
                
                trades.append({
                    'type': 'BUY',
                    'timestamp': current_time,
                    'price': entry_price,
                    'size': position,
                    'probability': probability,
                    'strength': strength,
                    'commission': commission,
                    'base_predictions': base_predictions,
                    'capital_before': capital + commission
                })
        
        else:  # Есть позиция
            # Условия выхода
            should_exit = False
            exit_reason = ''
            
            # Стоп-лосс (-3%)
            if current_price <= position_price * 0.97:
                should_exit = True
                exit_reason = 'stop_loss'
            # Тейк-профит (+6%)  
            elif current_price >= position_price * 1.06:
                should_exit = True
                exit_reason = 'take_profit'
            # Сигнал на выход
            elif signal == 0 or (probability < 0.3 and strength == 'STRONG'):
                should_exit = True
                exit_reason = 'signal_exit'
            # Тайм-стоп (более 4 часов)
            elif (current_time - position_entry_time).total_seconds() > 4 * 3600:
                should_exit = True
                exit_reason = 'time_stop'
            
            if should_exit:
                exit_price = current_price * (1 - slippage)
                position_value = position * exit_price
                commission = position_value * commission_rate
                
                gross_pnl = (exit_price - position_price) * position
                net_pnl = gross_pnl - commission
                capital += net_pnl
                
                return_pct = (exit_price - position_price) / position_price * 100
                
                trades.append({
                    'type': 'SELL',
                    'timestamp': current_time,
                    'price': exit_price,
                    'size': position,
                    'pnl_gross': gross_pnl,
                    'pnl_net': net_pnl,
                    'return_pct': return_pct,
                    'commission': commission,
                    'exit_reason': exit_reason,
                    'holding_time': current_time - position_entry_time,
                    'capital_after': capital
                })
                
                position = 0
                position_price = 0
                position_entry_time = None
        
        # Обновляем equity curve
        if position > 0:
            unrealized_pnl = (current_price - position_price) * position
            total_equity = capital + (position * current_price)
        else:
            total_equity = capital
        
        equity_curve.append({
            'timestamp': current_time,
            'equity': total_equity,
            'price': current_price,
            'capital': capital,
            'position_value': position * current_price if position > 0 else 0
        })
        
        progress.update(1)
    
    # Закрываем оставшуюся позицию
    if position > 0:
        final_time = data.iloc[-1]['timestamp']
        final_price = data.iloc[-1]['close']
        
        exit_price = final_price * (1 - slippage)
        position_value = position * exit_price
        commission = position_value * commission_rate
        
        gross_pnl = (exit_price - position_price) * position
        net_pnl = gross_pnl - commission
        capital += net_pnl
        
        trades.append({
            'type': 'SELL',
            'timestamp': final_time,
            'price': exit_price,
            'size': position,
            'pnl_gross': gross_pnl,
            'pnl_net': net_pnl,
            'return_pct': (exit_price - position_price) / position_price * 100,
            'commission': commission,
            'exit_reason': 'backtest_end',
            'holding_time': final_time - position_entry_time,
            'capital_after': capital
        })
    
    backtest_time = time.time() - backtest_start_time
    avg_prediction_time = np.mean(prediction_times) if prediction_times else 0
    
    print(f"\n✅ Бэктест завершен за {backtest_time:.1f}s")
    print(f"   Среднее время ML предсказания: {avg_prediction_time*1000:.1f}ms")
    print(f"   Сгенерировано {len(signals)} сигналов")
    print(f"   Совершено {len([t for t in trades if t['type'] == 'BUY'])} сделок")
    
    return {
        'trades': trades,
        'signals': signals,
        'equity_curve': equity_curve,
        'initial_capital': initial_capital,
        'final_capital': capital,
        'backtest_time': backtest_time,
        'avg_prediction_time': avg_prediction_time,
        'data_period': {
            'start': data['timestamp'].iloc[0].isoformat(),
            'end': data['timestamp'].iloc[-1].isoformat(),
            'total_candles': len(data)
        }
    }

def generate_detailed_html_report(results):
    """Генерация детального HTML отчета."""
    print("📝 Создание подробного HTML отчета...")
    
    trades = results['trades']
    signals = results['signals']
    equity_curve = results['equity_curve']
    initial_capital = results['initial_capital']
    final_capital = results['final_capital']
    
    # Расчет метрик
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL' and 'pnl_net' in t]
    
    total_trades = len(buy_trades)
    
    if sell_trades:
        pnls = [t['pnl_net'] for t in sell_trades]
        winning_trades = len([p for p in pnls if p > 0])
        win_rate = winning_trades / len(sell_trades) * 100
        total_pnl = sum(pnls)
        avg_win = np.mean([p for p in pnls if p > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([p for p in pnls if p <= 0]) if len(pnls) > winning_trades else 0
        best_trade = max(pnls)
        worst_trade = min(pnls)
        profit_factor = abs(sum(p for p in pnls if p > 0) / sum(p for p in pnls if p <= 0)) if sum(p for p in pnls if p <= 0) != 0 else float('inf')
        
        # Анализ времени удержания
        holding_times = [t.get('holding_time') for t in sell_trades if 'holding_time' in t]
        avg_holding_hours = np.mean([ht.total_seconds() / 3600 for ht in holding_times if ht]) if holding_times else 0
    else:
        winning_trades = 0
        win_rate = 0
        total_pnl = 0
        avg_win = 0
        avg_loss = 0
        best_trade = 0
        worst_trade = 0
        profit_factor = 0
        avg_holding_hours = 0
    
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    # Анализ сигналов
    buy_signals = len([s for s in signals if s['signal'] == 1])
    strong_signals = len([s for s in signals if s['strength'] == 'STRONG'])
    avg_probability = np.mean([s['probability'] for s in signals])
    
    # ML модели статистика
    ml_stats = {}
    for signal in signals:
        base_preds = signal.get('base_predictions', {})
        for model, pred in base_preds.items():
            if model not in ml_stats:
                ml_stats[model] = []
            ml_stats[model].append(pred)
    
    # Усреднение по моделям
    ml_averages = {model: np.mean(preds) for model, preds in ml_stats.items()}
    
    html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Ensemble Trading Bot - Полный отчет бэктеста</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.15);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 20"><defs><radialGradient id="gradient"><stop offset="20%" stop-color="%23ffffff" stop-opacity="0.1"/><stop offset="80%" stop-color="%23ffffff" stop-opacity="0"/></radialGradient></defs><rect width="100" height="20" fill="url(%23gradient)"/></svg>') repeat-x;
            opacity: 0.1;
        }}
        
        .header h1 {{
            font-size: 3em;
            margin-bottom: 15px;
            font-weight: 700;
            position: relative;
            z-index: 1;
        }}
        
        .header .subtitle {{
            font-size: 1.3em;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }}
        
        .performance-banner {{
            background: linear-gradient(135deg, {'#27ae60' if total_return > 0 else '#e74c3c'} 0%, {'#2ecc71' if total_return > 0 else '#c0392b'} 100%);
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .metric-card {{
            background: white;
            padding: 30px 25px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            text-align: center;
            transition: all 0.3s ease;
            border-left: 5px solid #3498db;
        }}
        
        .metric-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }}
        
        .metric-card.positive {{ border-left-color: #27ae60; }}
        .metric-card.negative {{ border-left-color: #e74c3c; }}
        .metric-card.warning {{ border-left-color: #f39c12; }}
        
        .metric-value {{
            font-size: 2.8em;
            font-weight: 800;
            margin: 15px 0;
            line-height: 1;
        }}
        
        .metric-label {{
            color: #7f8c8d;
            font-size: 1em;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 600;
        }}
        
        .metric-sublabel {{
            color: #95a5a6;
            font-size: 0.85em;
            margin-top: 8px;
        }}
        
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .neutral {{ color: #3498db; }}
        .warning {{ color: #f39c12; }}
        
        .section {{
            padding: 40px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        .section:last-child {{ border-bottom: none; }}
        
        .section h2 {{
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2.2em;
            font-weight: 700;
            border-left: 6px solid #3498db;
            padding-left: 20px;
            display: flex;
            align-items: center;
        }}
        
        .section h2::before {{
            font-size: 0.8em;
            margin-right: 15px;
        }}
        
        .trades-table, .signals-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 25px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            background: white;
        }}
        
        .trades-table th, .trades-table td,
        .signals-table th, .signals-table td {{
            padding: 15px 18px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .trades-table th, .signals-table th {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-size: 0.9em;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        .trades-table tr:nth-child(even),
        .signals-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .trades-table tr:hover,
        .signals-table tr:hover {{
            background: #e3f2fd;
            transition: background 0.3s ease;
        }}
        
        .signal-badge {{
            padding: 6px 12px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .buy-signal {{ 
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            color: white;
        }}
        
        .sell-signal {{ 
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
        }}
        
        .strength-badge {{
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .strength-strong {{ 
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
        }}
        
        .strength-weak {{ 
            background: linear-gradient(135deg, #95a5a6, #7f8c8d);
            color: white;
        }}
        
        .ml-predictions {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 8px;
            margin-top: 8px;
        }}
        
        .ml-pred {{
            background: linear-gradient(135deg, #ecf0f1, #bdc3c7);
            padding: 6px 10px;
            border-radius: 8px;
            font-size: 0.8em;
            text-align: center;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }}
        
        .stat-box {{
            background: white;
            border: 2px solid #ecf0f1;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .stat-box:hover {{
            border-color: #3498db;
            transform: translateY(-5px);
        }}
        
        .stat-number {{
            font-size: 2.2em;
            font-weight: 800;
            color: #2c3e50;
            margin-bottom: 8px;
        }}
        
        .stat-label {{
            color: #7f8c8d;
            margin-top: 8px;
            font-weight: 600;
        }}
        
        .performance-summary {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
        }}
        
        .footer {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            text-align: center;
            padding: 30px;
            font-size: 1em;
        }}
        
        .footer .tech-info {{
            opacity: 0.8;
            margin-top: 15px;
            font-size: 0.9em;
        }}
        
        .table-container {{
            max-height: 600px;
            overflow-y: auto;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        
        .exit-reason {{
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            font-weight: 600;
        }}
        
        .exit-stop-loss {{ background: #ffebee; color: #c62828; }}
        .exit-take-profit {{ background: #e8f5e8; color: #2e7d32; }}
        .exit-signal-exit {{ background: #fff3e0; color: #f57c00; }}
        .exit-time-stop {{ background: #f3e5f5; color: #7b1fa2; }}
        .exit-backtest-end {{ background: #e3f2fd; color: #1976d2; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🤖 ML Ensemble Trading Bot</h1>
            <div class="subtitle">
                Полный отчет бэктеста • SOL/USDT • {results['data_period']['start'][:10]} - {results['data_period']['end'][:10]}<br>
                {results['data_period']['total_candles']} свечей • Время выполнения: {results['backtest_time']:.1f}s
            </div>
        </div>

        <!-- Performance Banner -->
        <div class="performance-banner">
            {'🎉 ПРИБЫЛЬНАЯ СТРАТЕГИЯ' if total_return > 0 else '⚠️ УБЫТОЧНАЯ СТРАТЕГИЯ'}: {total_return:+.2f}% за период
        </div>

        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card {'positive' if total_return > 0 else 'negative' if total_return < 0 else 'neutral'}">
                <div class="metric-label">Общая доходность</div>
                <div class="metric-value {'positive' if total_return > 0 else 'negative' if total_return < 0 else 'neutral'}">{total_return:+.2f}%</div>
                <div class="metric-sublabel">За {(pd.to_datetime(results['data_period']['end']) - pd.to_datetime(results['data_period']['start'])).days} дней</div>
            </div>
            
            <div class="metric-card neutral">
                <div class="metric-label">Финальный капитал</div>
                <div class="metric-value neutral">${final_capital:,.2f}</div>
                <div class="metric-sublabel">Начальный: ${initial_capital:,.2f}</div>
            </div>
            
            <div class="metric-card {'positive' if total_pnl > 0 else 'negative' if total_pnl < 0 else 'neutral'}">
                <div class="metric-label">Чистый P&L</div>
                <div class="metric-value {'positive' if total_pnl > 0 else 'negative' if total_pnl < 0 else 'neutral'}">${total_pnl:+,.2f}</div>
                <div class="metric-sublabel">Лучшая: ${best_trade:+.2f} • Худшая: ${worst_trade:+.2f}</div>
            </div>
            
            <div class="metric-card {'positive' if win_rate > 60 else 'warning' if win_rate > 40 else 'negative'}">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value {'positive' if win_rate > 60 else 'warning' if win_rate > 40 else 'negative'}">{win_rate:.1f}%</div>
                <div class="metric-sublabel">{winning_trades} из {len(sell_trades)} сделок</div>
            </div>
            
            <div class="metric-card neutral">
                <div class="metric-label">Всего сделок</div>
                <div class="metric-value neutral">{total_trades}</div>
                <div class="metric-sublabel">Среднее удержание: {avg_holding_hours:.1f}ч</div>
            </div>
            
            <div class="metric-card {'positive' if profit_factor > 1.5 else 'warning' if profit_factor > 1 else 'negative'}">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value {'positive' if profit_factor > 1.5 else 'warning' if profit_factor > 1 else 'negative'}">{profit_factor:.2f}</div>
                <div class="metric-sublabel">Ср. выигрыш: ${avg_win:.2f} • проигрыш: ${avg_loss:.2f}</div>
            </div>
        </div>

        <!-- ML Analysis -->
        <div class="section">
            <h2>🧠 Анализ ML сигналов</h2>
            <div class="performance-summary">
                <h3>Статистика сигналов ансамбля</h3>
                <div class="summary-stats">
                    <div class="stat-box">
                        <div class="stat-number">{len(signals)}</div>
                        <div class="stat-label">Всего сигналов обработано</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{buy_signals}</div>
                        <div class="stat-label">BUY сигналов ({buy_signals/len(signals)*100:.1f}%)</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{strong_signals}</div>
                        <div class="stat-label">Сильных сигналов ({strong_signals/len(signals)*100:.1f}%)</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{avg_probability:.3f}</div>
                        <div class="stat-label">Средняя вероятность</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{results['avg_prediction_time']*1000:.1f}ms</div>
                        <div class="stat-label">Среднее время предсказания</div>
                    </div>
                </div>
                
                <h4>Средние предсказания по моделям:</h4>
                <div class="ml-predictions">
    """
    
    for model, avg_pred in ml_averages.items():
        html_content += f'<div class="ml-pred">{model}: {avg_pred:.3f}</div>'
    
    html_content += """
                </div>
            </div>
            
            <h3>Последние 30 сигналов</h3>
            <div class="table-container">
                <table class="signals-table">
                    <thead>
                        <tr>
                            <th>Время</th>
                            <th>Цена</th>
                            <th>Сигнал</th>
                            <th>Вероятность</th>
                            <th>Сила</th>
                            <th>ML Предсказания</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Показываем последние 30 сигналов
    for signal in signals[-30:]:
        signal_class = 'buy-signal' if signal['signal'] == 1 else 'sell-signal'
        signal_text = 'BUY' if signal['signal'] == 1 else 'SELL'
        strength_class = f"strength-{signal['strength'].lower()}"
        
        base_preds = signal.get('base_predictions', {})
        ml_preds_html = ""
        if base_preds:
            for model, pred in base_preds.items():
                ml_preds_html += f'<div class="ml-pred">{model}: {pred:.3f}</div>'
        
        html_content += f"""
                        <tr>
                            <td>{signal['timestamp'].strftime('%m-%d %H:%M')}</td>
                            <td>${signal['price']:.4f}</td>
                            <td><span class="signal-badge {signal_class}">{signal_text}</span></td>
                            <td>{signal['probability']:.3f}</td>
                            <td><span class="strength-badge {strength_class}">{signal['strength']}</span></td>
                            <td><div class="ml-predictions">{ml_preds_html}</div></td>
                        </tr>
        """
    
    html_content += """
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Trades Detail -->
        <div class="section">
            <h2>💹 Детализация всех сделок</h2>
    """
    
    if buy_trades:
        html_content += """
            <div class="table-container">
                <table class="trades-table">
                    <thead>
                        <tr>
                            <th>Время</th>
                            <th>Тип</th>
                            <th>Цена</th>
                            <th>Размер</th>
                            <th>P&L</th>
                            <th>Доходность</th>
                            <th>Причина выхода</th>
                            <th>Время удержания</th>
                            <th>ML Вероятность</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Группируем сделки попарно
        for i in range(0, len(trades), 2):
            buy_trade = trades[i] if i < len(trades) else None
            sell_trade = trades[i + 1] if i + 1 < len(trades) and trades[i + 1]['type'] == 'SELL' else None
            
            if buy_trade and buy_trade['type'] == 'BUY':
                pnl = sell_trade.get('pnl_net', 0) if sell_trade else 0
                return_pct = sell_trade.get('return_pct', 0) if sell_trade else 0
                pnl_class = 'positive' if pnl > 0 else 'negative' if pnl < 0 else 'neutral'
                
                exit_reason = sell_trade.get('exit_reason', 'unknown') if sell_trade else 'open'
                holding_time = sell_trade.get('holding_time') if sell_trade else None
                holding_str = f"{holding_time.total_seconds()/3600:.1f}ч" if holding_time else 'N/A'
                
                html_content += f"""
                        <tr>
                            <td>{buy_trade['timestamp'].strftime('%m-%d %H:%M')}</td>
                            <td><span class="signal-badge buy-signal">BUY</span></td>
                            <td>${buy_trade['price']:.4f}</td>
                            <td>{buy_trade['size']:.6f}</td>
                            <td rowspan="2" class="{pnl_class}">${pnl:+.2f}</td>
                            <td rowspan="2" class="{pnl_class}">{return_pct:+.2f}%</td>
                            <td rowspan="2"><span class="exit-reason exit-{exit_reason.replace('_', '-')}">{exit_reason}</span></td>
                            <td rowspan="2">{holding_str}</td>
                            <td>{buy_trade.get('probability', 0):.3f}</td>
                        </tr>
                """
                
                if sell_trade:
                    html_content += f"""
                        <tr>
                            <td>{sell_trade['timestamp'].strftime('%m-%d %H:%M')}</td>
                            <td><span class="signal-badge sell-signal">SELL</span></td>
                            <td>${sell_trade['price']:.4f}</td>
                            <td>{sell_trade['size']:.6f}</td>
                            <td>-</td>
                        </tr>
                    """
        
        html_content += """
                    </tbody>
                </table>
            </div>
        """
    else:
        html_content += "<p>В данном периоде сделки не совершались из-за строгих критериев ML фильтрации.</p>"
    
    html_content += f"""
        </div>

        <!-- Footer -->
        <div class="footer">
            <div>
                🤖 Отчет сгенерирован ML Ensemble Trading Bot • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            <div class="tech-info">
                Использованные ML модели: RandomForest, LightGBM, XGBoost, CatBoost + Logistic Regression (мета-модель)<br>
                Всего обработано {results['data_period']['total_candles']} 5-минутных свечей • {len(signals)} ML предсказаний • {total_trades} торговых сделок
            </div>
        </div>
    </div>
</body>
</html>
    """
    
    return html_content

def main():
    """Основная функция."""
    print("🎯 ПОЛНЫЙ БЭКТЕСТ С ДЕТАЛЬНЫМ HTML ОТЧЕТОМ")
    print("="*70)
    
    # Загрузка данных
    data = load_real_data()
    if data is None:
        print("❌ Не удалось загрузить данные")
        return
    
    # Подготовка технических индикаторов
    data = prepare_technical_indicators(data)
    
    # Запуск полного бэктеста
    results = run_full_backtest(data)
    
    # Генерация HTML отчета
    html_content = generate_detailed_html_report(results)
    
    # Сохранение отчета
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('output/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html_file = output_dir / f'full_ml_backtest_report_{timestamp}.html'
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ ПОЛНЫЙ HTML ОТЧЕТ СОХРАНЕН: {html_file}")
    print(f"🌐 Откройте файл в браузере для просмотра")
    
    # Дополнительно сохраняем JSON данные
    json_file = output_dir / f'full_ml_backtest_data_{timestamp}.json'
    
    # Подготавливаем данные для JSON (конвертируем datetime)
    json_data = results.copy()
    for trade in json_data['trades']:
        if 'timestamp' in trade:
            trade['timestamp'] = trade['timestamp'].isoformat()
        if 'holding_time' in trade and trade['holding_time']:
            trade['holding_time'] = str(trade['holding_time'])
    
    for signal in json_data['signals']:
        if 'timestamp' in signal:
            signal['timestamp'] = signal['timestamp'].isoformat()
    
    for equity_point in json_data['equity_curve']:
        if 'timestamp' in equity_point:
            equity_point['timestamp'] = equity_point['timestamp'].isoformat()
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Данные также сохранены в JSON: {json_file}")
    
    # Краткая сводка результатов
    buy_trades = [t for t in results['trades'] if t['type'] == 'BUY']
    sell_trades = [t for t in results['trades'] if t['type'] == 'SELL' and 'pnl_net' in t]
    
    if sell_trades:
        total_pnl = sum(t['pnl_net'] for t in sell_trades)
        win_rate = len([t for t in sell_trades if t['pnl_net'] > 0]) / len(sell_trades) * 100
        total_return = (results['final_capital'] - results['initial_capital']) / results['initial_capital'] * 100
        
        print(f"\n📊 КРАТКИЕ РЕЗУЛЬТАТЫ:")
        print(f"   Сделок: {len(buy_trades)}")
        print(f"   P&L: ${total_pnl:+.2f}")
        print(f"   Доходность: {total_return:+.2f}%")
        print(f"   Win Rate: {win_rate:.1f}%")
    
    return str(html_file)

if __name__ == "__main__":
    main()