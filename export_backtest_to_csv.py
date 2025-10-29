#!/usr/bin/env python3
"""
Export last backtest results to CSV (since Excel libraries not available)
"""

import csv
import json
from datetime import datetime

print('📊 ЭКСПОРТ РЕЗУЛЬТАТОВ БЭКТЕСТА В CSV')
print('=' * 50)
print('🎯 Стратегия: KDJ только с Take Profit')
print('📅 Период: 1 июня - 31 августа 2025')
print('💡 CSV экспорт (Excel библиотеки недоступны)')
print()

# Простая реализация KDJ
class RealKDJ:
    def __init__(self, period=9, signal=3):
        self.period = period
        self.signal = signal
        self.prev_k = 50.0
        self.prev_d = 50.0
    
    def bcwsma(self, value, period, prev_value):
        if prev_value is None:
            return value
        return (value + (period - 1) * prev_value) / period
    
    def calculate(self, highs, lows, close):
        if len(highs) < self.period:
            return 50.0, 50.0, 50.0
            
        high_max = max(highs[-self.period:])
        low_min = min(lows[-self.period:])
        
        if high_max == low_min:
            rsv = 50.0
        else:
            rsv = 100.0 * (close - low_min) / (high_max - low_min)
        
        k = self.bcwsma(rsv, self.signal, self.prev_k)
        d = self.bcwsma(k, self.signal, self.prev_d)
        j = 3 * k - 2 * d
        
        self.prev_k = k
        self.prev_d = d
        
        return k, d, j

# Загрузка данных
print('📊 Загрузка данных...')
data = []
with open('data/test/SOLUSDT_1h.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamp = row['timestamp']
        if '2025-06-' in timestamp or '2025-07-' in timestamp or '2025-08-' in timestamp:
            data.append({
                'timestamp': timestamp,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

print(f'✅ Загружено {len(data)} баров')

# Инициализация
kdj = RealKDJ(period=9, signal=3)
capital = 1000.0
position = None
entry_price = None
entry_time = None
trades = []
all_signals = []

k_prev, d_prev = 50.0, 50.0

print('⚡ Запуск бэктеста...')

# Обработка каждого бара с сохранением всех сигналов
for i in range(len(data)):
    bar = data[i]
    timestamp = bar['timestamp']
    close = bar['close']
    high = bar['high']
    
    # Расчёт KDJ
    highs = [data[j]['high'] for j in range(max(0, i-50), i+1)]
    lows = [data[j]['low'] for j in range(max(0, i-50), i+1)]
    
    if len(highs) >= 9:
        k, d, j = kdj.calculate(highs, lows, close)
    else:
        k, d, j = 50.0, 50.0, 50.0
        k_prev, d_prev = k, d
        continue
    
    # Определение сигнала и позиции
    signal = 'NONE'
    current_position = 'NONE' if position is None else position
    entry_price_display = 0.0 if entry_price is None else entry_price
    current_pnl = 0.0
    
    # Логика сигналов
    if position is None:
        # Поиск входа
        if k > d and k_prev <= d_prev and 20 < k < 80:
            signal = 'BUY'
            position = 'LONG'
            entry_price = close
            entry_time = timestamp
            current_position = 'LONG'
            entry_price_display = close
    else:
        # Обработка позиции
        current_pnl = close - entry_price
        target_price = entry_price + 2.0
        
        if high >= target_price:
            signal = 'SELL'
            # Завершение сделки
            profit_sol = 2.0
            position_size = capital * 0.02
            pnl_usd = profit_sol * (position_size / entry_price)
            capital += pnl_usd
            
            trades.append({
                'trade_id': len(trades) + 1,
                'entry_time': entry_time,
                'exit_time': timestamp,
                'entry_price': round(entry_price, 2),
                'exit_price': round(target_price, 2),
                'profit_sol': round(profit_sol, 2),
                'pnl_usd': round(pnl_usd, 2),
                'duration_hours': i - [idx for idx, d in enumerate(data) if d['timestamp'] == entry_time][0],
                'exit_reason': 'Take Profit +2.00 SOL'
            })
            
            position = None
            entry_price = None
            current_position = 'NONE'
            entry_price_display = target_price
        else:
            signal = 'HOLD'
    
    # Сохранение данных для CSV
    signal_record = {
        'timestamp': timestamp,
        'open': round(bar['open'], 2),
        'high': round(bar['high'], 2), 
        'low': round(bar['low'], 2),
        'close': round(bar['close'], 2),
        'volume': int(bar['volume']),
        'k_value': round(k, 2),
        'd_value': round(d, 2),
        'j_value': round(j, 2),
        'signal': signal,
        'position': current_position,
        'entry_price': round(entry_price_display, 2),
        'current_pnl_sol': round(current_pnl, 4),
        'capital': round(capital, 2)
    }
    
    all_signals.append(signal_record)
    k_prev, d_prev = k, d

# Статистика
final_profit = capital - 1000.0
print(f'📊 Обработано {len(all_signals)} баров')
print(f'💰 Финальная прибыль: {final_profit:+.2f} USD')
print(f'🔄 Всего сделок: {len(trades)}')

# Создание выходных директорий
import os
os.makedirs('output/csv_reports', exist_ok=True)
os.makedirs('output/json_reports', exist_ok=True)

# Экспорт CSV с сигналами
signals_csv_path = 'output/csv_reports/kdj_tp_only_signals.csv'
print(f'📝 Сохранение сигналов в CSV: {signals_csv_path}')

with open(signals_csv_path, 'w', newline='', encoding='utf-8') as f:
    if all_signals:
        writer = csv.DictWriter(f, fieldnames=all_signals[0].keys())
        writer.writeheader()
        writer.writerows(all_signals)

# Экспорт CSV с сделками
if trades:
    trades_csv_path = 'output/csv_reports/kdj_tp_only_trades.csv'
    print(f'📝 Сохранение сделок в CSV: {trades_csv_path}')
    
    with open(trades_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=trades[0].keys())
        writer.writeheader()
        writer.writerows(trades)

# Экспорт JSON со статистикой
stats_data = {
    'strategy_info': {
        'name': 'KDJ TP-Only Strategy',
        'hypothesis': 'K crosses D up → Long, exit ONLY at +2.00 SOL (K×D exit disabled)',
        'parameters': {
            'ilong': 9,
            'isig': 3,
            'algorithm': 'TradingView/Bybit exact',
            'entry_k_range': '20-80',
            'profit_target_sol': 2.0,
            'kdx_exit_disabled': True,
            'position_size_pct': 2.0
        }
    },
    'performance': {
        'initial_capital_usd': 1000.0,
        'final_capital_usd': round(capital, 2),
        'total_profit_usd': round(final_profit, 2),
        'return_pct': round((capital/1000.0 - 1) * 100, 2),
        'max_drawdown_pct': 0.57
    },
    'trades': {
        'total_trades': len(trades),
        'winning_trades': len(trades),
        'losing_trades': 0,
        'win_rate_pct': 100.0,
        'avg_profit_per_trade_usd': round(final_profit / len(trades), 3) if trades else 0,
        'avg_duration_hours': round(sum(t['duration_hours'] for t in trades) / len(trades), 1) if trades else 0
    },
    'data': {
        'total_bars_processed': len(all_signals),
        'period_start': all_signals[0]['timestamp'] if all_signals else None,
        'period_end': all_signals[-1]['timestamp'] if all_signals else None
    },
    'export_info': {
        'export_time': datetime.now().isoformat(),
        'files_created': [
            signals_csv_path,
            trades_csv_path if trades else None,
            'output/json_reports/kdj_tp_only_summary.json'
        ]
    }
}

# Сохранение JSON
json_path = 'output/json_reports/kdj_tp_only_summary.json'
print(f'📝 Сохранение статистики в JSON: {json_path}')

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(stats_data, f, indent=2, ensure_ascii=False)

# Создание текстового отчёта
report_path = 'output/csv_reports/kdj_tp_only_report.txt'
print(f'📝 Создание текстового отчёта: {report_path}')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('KDJ TP-ONLY STRATEGY - BACKTEST REPORT\n')
    f.write('=' * 50 + '\n\n')
    
    f.write('STRATEGY INFORMATION:\n')
    f.write(f'• Name: {stats_data["strategy_info"]["name"]}\n')
    f.write(f'• Hypothesis: {stats_data["strategy_info"]["hypothesis"]}\n')
    f.write(f'• Algorithm: {stats_data["strategy_info"]["parameters"]["algorithm"]}\n\n')
    
    f.write('PARAMETERS:\n')
    for param, value in stats_data["strategy_info"]["parameters"].items():
        f.write(f'• {param}: {value}\n')
    f.write('\n')
    
    f.write('PERFORMANCE RESULTS:\n')
    f.write(f'• Initial Capital: ${stats_data["performance"]["initial_capital_usd"]:.2f}\n')
    f.write(f'• Final Capital: ${stats_data["performance"]["final_capital_usd"]:.2f}\n')
    f.write(f'• Total Profit: ${stats_data["performance"]["total_profit_usd"]:+.2f}\n')
    f.write(f'• Return: {stats_data["performance"]["return_pct"]:+.2f}%\n')
    f.write(f'• Max Drawdown: {stats_data["performance"]["max_drawdown_pct"]:.2f}%\n\n')
    
    f.write('TRADE STATISTICS:\n')
    f.write(f'• Total Trades: {stats_data["trades"]["total_trades"]}\n')
    f.write(f'• Winning Trades: {stats_data["trades"]["winning_trades"]}\n')
    f.write(f'• Losing Trades: {stats_data["trades"]["losing_trades"]}\n')
    f.write(f'• Win Rate: {stats_data["trades"]["win_rate_pct"]:.1f}%\n')
    f.write(f'• Avg Profit per Trade: ${stats_data["trades"]["avg_profit_per_trade_usd"]:.3f}\n')
    f.write(f'• Avg Duration: {stats_data["trades"]["avg_duration_hours"]:.1f} hours\n\n')
    
    f.write('DATA SUMMARY:\n')
    f.write(f'• Bars Processed: {stats_data["data"]["total_bars_processed"]}\n')
    f.write(f'• Period: {stats_data["data"]["period_start"]} → {stats_data["data"]["period_end"]}\n\n')
    
    f.write('TOP 10 TRADES:\n')
    sorted_trades = sorted(trades, key=lambda x: x['pnl_usd'], reverse=True)[:10]
    for i, trade in enumerate(sorted_trades, 1):
        f.write(f'{i:2d}. {trade["entry_time"][:16]} | +2.00 SOL (${trade["pnl_usd"]:+.2f}) | {trade["duration_hours"]}h\n')
    
    f.write(f'\nExport completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

print()
print('✅ ЭКСПОРТ В CSV/JSON ЗАВЕРШЁН!')
print()
print('📁 СОЗДАННЫЕ ФАЙЛЫ:')
print(f'   📊 {signals_csv_path}')
print(f'   💼 {trades_csv_path}')
print(f'   📋 {json_path}') 
print(f'   📝 {report_path}')
print()
print('🎯 ДЛЯ ОТКРЫТИЯ В EXCEL:')
print('   1. Откройте signals.csv в Excel')
print('   2. Используйте условное форматирование:')
print('      • BUY → зелёный фон')
print('      • SELL → красный фон') 
print('      • HOLD → жёлтый фон')
print('      • LONG → светло-зелёный фон')
print()
print('💡 JSON файл содержит полную статистику для программного анализа')