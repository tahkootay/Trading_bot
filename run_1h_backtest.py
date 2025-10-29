#!/usr/bin/env python3
"""
KDJ TP-Only Strategy - 1-Hour Timeframe Backtest (Sep 2024 - Aug 2025)
Comprehensive analysis on hourly data
"""

import csv
import sys
import os
from datetime import datetime, timedelta

print('📊 KDJ TP-ONLY STRATEGY - 1-ЧАСОВОЙ БЭКТЕСТ')
print('=' * 55)
print('🎯 Стратегия: K×D вход, только Take Profit +2.00 SOL выход')
print('📅 Период: 01.09.2024 - 31.08.2025 (12 месяцев)')
print('⏰ Таймфрейм: 1 час')
print('💡 Алгоритм: TradingView/Bybit точный KDJ')
print()

class KDJTradingView:
    def __init__(self, ilong=9, isig=3):
        self.ilong = ilong
        self.isig = isig
        self.prev_k = 50.0
        self.prev_d = 50.0
    
    def bcwsma(self, value, period, prev_value):
        if prev_value is None:
            return value
        return (value + (period - 1) * prev_value) / period
    
    def calculate(self, highs, lows, close):
        if len(highs) < self.ilong:
            return 50.0, 50.0, 50.0
            
        high_max = max(highs[-self.ilong:])
        low_min = min(lows[-self.ilong:])
        
        if high_max == low_min:
            rsv = 50.0
        else:
            rsv = 100.0 * (close - low_min) / (high_max - low_min)
        
        k = self.bcwsma(rsv, self.isig, self.prev_k)
        d = self.bcwsma(k, self.isig, self.prev_d)
        j = 3 * k - 2 * d
        
        self.prev_k = k
        self.prev_d = d
        
        return k, d, j

# Загрузка 1-часовых данных
print('📊 Загрузка 1-часовых данных...')
data = []

with open('data/raw/SOLUSDT_1h_20240901_20250831.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            'timestamp': row['timestamp'],
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })

print(f'✅ Загружено {len(data)} часовых баров')
print(f'📅 Период: {data[0]["timestamp"]} → {data[-1]["timestamp"]}')

# Расчёт статистики периода
start_date = datetime.strptime(data[0]["timestamp"], '%Y-%m-%d %H:%M:%S')
end_date = datetime.strptime(data[-1]["timestamp"], '%Y-%m-%d %H:%M:%S')
total_days = (end_date - start_date).days
total_months = total_days / 30.44

print(f'📊 Длительность: {total_days} дней ({total_months:.1f} месяцев)')

# Инициализация стратегии
kdj = KDJTradingView(ilong=9, isig=3)
initial_capital = 1000.0
capital = initial_capital
position = None
entry_price = None
entry_time = None
exit_price = None
trades = []
all_signals = []

# Переменные для расширенной аналитики
max_capital = capital
max_drawdown = 0.0
consecutive_wins = 0
max_consecutive_wins = 0
total_holding_time = 0
monthly_stats = {}

k_prev, d_prev = 50.0, 50.0

print('⚡ Запуск 1-часового бэктеста...')

# Основной цикл бэктеста
for i in range(len(data)):
    bar = data[i]
    timestamp = bar['timestamp']
    close = bar['close']
    high = bar['high']
    
    # Отслеживание месячной статистики
    bar_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    month_key = bar_date.strftime('%Y-%m')
    
    if month_key not in monthly_stats:
        monthly_stats[month_key] = {
            'start_capital': capital,
            'trades': 0,
            'profit': 0.0
        }
    
    # Расчёт KDJ
    highs = [data[j]['high'] for j in range(max(0, i-50), i+1)]
    lows = [data[j]['low'] for j in range(max(0, i-50), i+1)]
    
    if len(highs) >= 9:
        k, d, j = kdj.calculate(highs, lows, close)
    else:
        k, d, j = 50.0, 50.0, 50.0
        k_prev, d_prev = k, d
        continue
    
    # Инициализация переменных для записи
    signal = 'NONE'
    current_position = 'NONE' if position is None else position
    entry_price_display = 0.0 if entry_price is None else entry_price
    exit_price_display = 0.0 if exit_price is None else exit_price
    current_pnl = 0.0
    
    # Торговая логика
    if position is None:
        # Поиск входа: K пересекает D вверх в зоне 20-80
        if k > d and k_prev <= d_prev and 20 < k < 80:
            signal = 'BUY'
            position = 'LONG'
            entry_price = close
            entry_time = timestamp
            current_position = 'LONG'
            entry_price_display = close
            exit_price_display = 0.0
    else:
        # Обработка открытой позиции
        current_pnl = close - entry_price
        target_price = entry_price + 2.0
        
        # Проверка достижения Take Profit
        if high >= target_price:
            signal = 'SELL'
            exit_price = target_price  # Точно +2.00 SOL
            
            # Расчёт прибыли
            profit_sol = 2.0
            position_size = capital * 0.02
            pnl_usd = profit_sol * (position_size / entry_price)
            capital += pnl_usd
            
            # Обновление максимального капитала и просадки
            if capital > max_capital:
                max_capital = capital
            current_drawdown = (max_capital - capital) / max_capital * 100
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
            
            # Найти индекс входа для расчёта длительности
            entry_idx = next(idx for idx, d in enumerate(data) if d['timestamp'] == entry_time)
            duration_hours = i - entry_idx
            total_holding_time += duration_hours
            
            # Отслеживание серии побед
            consecutive_wins += 1
            if consecutive_wins > max_consecutive_wins:
                max_consecutive_wins = consecutive_wins
            
            # Обновление месячной статистики
            monthly_stats[month_key]['trades'] += 1
            monthly_stats[month_key]['profit'] += pnl_usd
            
            # Сохранение сделки
            trades.append({
                'trade_id': len(trades) + 1,
                'entry_time': entry_time,
                'exit_time': timestamp,
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'profit_sol': round(profit_sol, 2),
                'pnl_usd': round(pnl_usd, 3),
                'duration_hours': duration_hours,
                'exit_reason': 'Take Profit +2.00 SOL',
                'capital_after': round(capital, 2),
                'month': month_key
            })
            
            # Закрытие позиции
            position = None
            entry_price = None
            current_position = 'NONE'
            entry_price_display = 0.0
            exit_price_display = target_price
        else:
            signal = 'HOLD'
            exit_price_display = 0.0
    
    # Сохранение данных для отчёта (только важные сигналы для экономии памяти)
    if signal in ['BUY', 'SELL'] or current_position == 'LONG':
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
            'exit_price': round(exit_price_display, 2),
            'current_pnl_sol': round(current_pnl, 4),
            'capital': round(capital, 2)
        }
        
        all_signals.append(signal_record)
    
    k_prev, d_prev = k, d

# Завершение месячной статистики
for month_key, stats in monthly_stats.items():
    stats['end_capital'] = capital if month_key == max(monthly_stats.keys()) else 0
    stats['return_pct'] = (stats['profit'] / stats['start_capital']) * 100 if stats['start_capital'] > 0 else 0

# Расчёт финальной статистики
final_profit = capital - initial_capital
return_pct = (capital / initial_capital - 1) * 100
avg_holding_time = total_holding_time / len(trades) if trades else 0
annualized_return = return_pct * (365.25 / total_days)
trade_frequency = len(trades) / total_days  # сделок в день

# Дополнительная аналитика по кварталам
quarterly_stats = {}
for trade in trades:
    trade_date = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
    quarter = f"{trade_date.year}-Q{(trade_date.month-1)//3+1}"
    if quarter not in quarterly_stats:
        quarterly_stats[quarter] = {'trades': 0, 'profit': 0.0}
    quarterly_stats[quarter]['trades'] += 1
    quarterly_stats[quarter]['profit'] += trade['pnl_usd']

print()
print('📊 РЕЗУЛЬТАТЫ 1-ЧАСОВОГО БЭКТЕСТА')
print('=' * 50)
print(f'📅 Период: {data[0]["timestamp"]} → {data[-1]["timestamp"]}')
print(f'📊 Обработано баров: {len(data)} (1-часовых)')
print(f'⏰ Длительность: {total_days} дней ({total_months:.1f} месяцев)')
print(f'🔄 Всего сделок: {len(trades)}')
print(f'📊 Сигналов в записи: {len(all_signals)}')
print(f'💰 Начальный капитал: ${initial_capital:.2f}')
print(f'💰 Финальный капитал: ${capital:.2f}')
print(f'💎 Общая прибыль: ${final_profit:+.2f}')
print(f'📈 Доходность: {return_pct:+.2f}%')
print(f'📈 Годовая доходность: {annualized_return:+.2f}%')
print(f'📉 Максимальная просадка: {max_drawdown:.2f}%')

if trades:
    avg_profit = final_profit / len(trades)
    win_rate = 100.0  # Все сделки прибыльны по дизайну
    print(f'💵 Средняя прибыль за сделку: ${avg_profit:.3f}')
    print(f'⏱️ Средняя длительность сделки: {avg_holding_time:.1f} часов')
    print(f'📊 Частота торговли: {trade_frequency:.3f} сделок/день')
    print(f'🏆 Процент побед: {win_rate}%')
    print(f'🔥 Максимальная серия побед: {max_consecutive_wins}')
    
    # Топ кварталы
    best_quarter = max(quarterly_stats.items(), key=lambda x: x[1]['profit'])
    worst_quarter = min(quarterly_stats.items(), key=lambda x: x[1]['profit'])
    print(f'📈 Лучший квартал: {best_quarter[0]} (+${best_quarter[1]["profit"]:.2f}, {best_quarter[1]["trades"]} сделок)')
    print(f'📉 Худший квартал: {worst_quarter[0]} (${worst_quarter[1]["profit"]:+.2f}, {worst_quarter[1]["trades"]} сделок)')

# Создание детального Excel отчёта
print()
print('📈 СОЗДАНИЕ ОТЧЁТА ДЛЯ 1-ЧАСОВОГО ТАЙМФРЕЙМА...')

try:
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.chart import LineChart, Reference
    
    # Создание Excel файла
    output_path = 'output/excel_reports/KDJ_1H_Backtest_Report.xlsx'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Цветовая схема
    colors = {
        'HEADER': PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid"),
        'POSITIVE': PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid"),
        'NEGATIVE': PatternFill(start_color="C55A5A", end_color="C55A5A", fill_type="solid"),
        'NEUTRAL': PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid"),
        'ACCENT': PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid"),
        'TITLE': PatternFill(start_color="44546A", end_color="44546A", fill_type="solid")
    }
    
    header_font = Font(color="FFFFFF", bold=True, size=11)
    title_font = Font(color="FFFFFF", bold=True, size=14)
    bold_font = Font(bold=True, size=11)
    center_alignment = Alignment(horizontal="center", vertical="center")
    
    wb = Workbook()
    
    # Лист 1: Сводка 1-часового анализа
    summary_ws = wb.active
    summary_ws.title = "1H Summary"
    
    # Заголовок
    summary_ws.cell(row=1, column=1, value='KDJ TP-ONLY STRATEGY - 1-HOUR TIMEFRAME ANALYSIS')
    summary_ws.cell(row=1, column=1).fill = colors['TITLE']
    summary_ws.cell(row=1, column=1).font = title_font
    summary_ws.cell(row=1, column=1).alignment = center_alignment
    
    # Основная статистика
    summary_data = [
        ['', ''],  # Пустая строка
        ['TIMEFRAME COMPARISON', ''],
        ['Previous 15m Results', f'{86} trades, +$17.65 (+1.77%)'],
        ['Current 1h Results', f'{len(trades)} trades, +${final_profit:.2f} (+{return_pct:.2f}%)'],
        ['Timeframe Effect', f'{"Improved" if final_profit > 17.65 else "Reduced"} performance'],
        ['', ''],
        ['PERFORMANCE METRICS', ''],
        ['Test Period', f'{data[0]["timestamp"]} to {data[-1]["timestamp"]}'],
        ['Total Days', f'{total_days}'],
        ['Total Bars (1H)', f'{len(data):,}'],
        ['Initial Capital', f'${initial_capital:.2f}'],
        ['Final Capital', f'${capital:.2f}'],
        ['Total Profit', f'${final_profit:+.2f}'],
        ['Return %', f'{return_pct:+.2f}%'],
        ['Annualized Return', f'{annualized_return:+.2f}%'],
        ['Max Drawdown', f'{max_drawdown:.2f}%'],
        ['', ''],
        ['TRADING STATISTICS', ''],
        ['Total Trades', f'{len(trades)}'],
        ['Win Rate', f'{100.0:.1f}%'],
        ['Avg Profit/Trade', f'${avg_profit:.3f}' if trades else '$0.000'],
        ['Trade Frequency', f'{trade_frequency:.3f} trades/day'],
        ['Avg Holding Time', f'{avg_holding_time:.1f} hours' if trades else '0.0 hours'],
        ['Max Consecutive Wins', f'{max_consecutive_wins}'],
        ['', ''],
        ['QUARTERLY BREAKDOWN', ''],
        ['Best Quarter', f'{best_quarter[0]} (+${best_quarter[1]["profit"]:.2f})' if quarterly_stats else 'N/A'],
        ['Worst Quarter', f'{worst_quarter[0]} (${worst_quarter[1]["profit"]:+.2f})' if quarterly_stats else 'N/A'],
        ['', ''],
        ['STRATEGY PARAMETERS', ''],
        ['KDJ Period (ilong)', '9'],
        ['KDJ Signal (isig)', '3'],
        ['Entry Filter', 'K > D crossover (20 < K < 80)'],
        ['Exit Strategy', 'Take Profit +2.00 SOL only'],
        ['Position Size', '2% of capital'],
        ['Risk Management', 'Fixed profit target'],
    ]
    
    # Заполнение данных
    for row_idx, (label, value) in enumerate(summary_data, 2):
        summary_ws.cell(row=row_idx, column=1, value=label)
        summary_ws.cell(row=row_idx, column=2, value=value)
        
        # Форматирование заголовков
        if label and label.isupper():
            summary_ws.cell(row=row_idx, column=1).fill = colors['ACCENT']
            summary_ws.cell(row=row_idx, column=1).font = bold_font
    
    # Настройка колонок
    summary_ws.column_dimensions['A'].width = 25
    summary_ws.column_dimensions['B'].width = 35
    
    # Лист 2: Детали сделок
    if trades:
        trades_ws = wb.create_sheet(title="1H Trade Details")
        
        # Заголовки
        trade_headers = ['Trade ID', 'Entry Time', 'Exit Time', 'Entry Price', 'Exit Price', 
                        'Profit SOL', 'P&L USD', 'Duration (h)', 'Month', 'Capital After']
        
        for col_idx, header in enumerate(trade_headers, 1):
            cell = trades_ws.cell(row=1, column=col_idx, value=header)
            cell.fill = colors['HEADER']
            cell.font = header_font
            cell.alignment = center_alignment
        
        # Данные сделок
        for row_idx, trade in enumerate(trades, 2):
            trades_ws.cell(row=row_idx, column=1, value=trade['trade_id'])
            trades_ws.cell(row=row_idx, column=2, value=trade['entry_time'])
            trades_ws.cell(row=row_idx, column=3, value=trade['exit_time'])
            trades_ws.cell(row=row_idx, column=4, value=trade['entry_price'])
            trades_ws.cell(row=row_idx, column=5, value=trade['exit_price'])
            trades_ws.cell(row=row_idx, column=6, value=trade['profit_sol'])
            
            pnl_cell = trades_ws.cell(row=row_idx, column=7, value=trade['pnl_usd'])
            pnl_cell.fill = colors['POSITIVE']  # Все прибыльны
            
            trades_ws.cell(row=row_idx, column=8, value=trade['duration_hours'])
            trades_ws.cell(row=row_idx, column=9, value=trade['month'])
            trades_ws.cell(row=row_idx, column=10, value=trade['capital_after'])
        
        # Автоподгонка ширины
        for column in trades_ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 20)
            trades_ws.column_dimensions[column_letter].width = adjusted_width
    
    # Лист 3: Квартальная статистика
    quarterly_ws = wb.create_sheet(title="Quarterly Analysis")
    
    quarterly_ws.cell(row=1, column=1, value='QUARTERLY PERFORMANCE ANALYSIS')
    quarterly_ws.cell(row=1, column=1).fill = colors['TITLE']
    quarterly_ws.cell(row=1, column=1).font = title_font
    
    # Заголовки
    quarterly_headers = ['Quarter', 'Number of Trades', 'Total Profit ($)', 'Avg Profit/Trade ($)', 'Performance Rating']
    for col_idx, header in enumerate(quarterly_headers, 1):
        cell = quarterly_ws.cell(row=3, column=col_idx, value=header)
        cell.fill = colors['HEADER']
        cell.font = header_font
        cell.alignment = center_alignment
    
    # Данные по кварталам
    for row_idx, (quarter, stats) in enumerate(sorted(quarterly_stats.items()), 4):
        quarterly_ws.cell(row=row_idx, column=1, value=quarter)
        quarterly_ws.cell(row=row_idx, column=2, value=stats['trades'])
        
        profit_cell = quarterly_ws.cell(row=row_idx, column=3, value=f"{stats['profit']:.2f}")
        avg_profit = stats['profit'] / stats['trades'] if stats['trades'] > 0 else 0
        quarterly_ws.cell(row=row_idx, column=4, value=f"{avg_profit:.3f}")
        
        # Рейтинг производительности
        if stats['profit'] > 5:
            rating = "Excellent"
            color = colors['POSITIVE']
        elif stats['profit'] > 2:
            rating = "Good"
            color = colors['POSITIVE']
        elif stats['profit'] > 0:
            rating = "Fair"
            color = colors['NEUTRAL']
        else:
            rating = "Poor"
            color = colors['NEGATIVE']
        
        rating_cell = quarterly_ws.cell(row=row_idx, column=5, value=rating)
        profit_cell.fill = color
        rating_cell.fill = color
    
    # Настройка колонок
    for col in ['A', 'B', 'C', 'D', 'E']:
        quarterly_ws.column_dimensions[col].width = 18
    
    # Сохранение файла
    wb.save(output_path)
    
    print(f'✅ Excel отчёт создан: {output_path}')
    print()
    print('📊 СТРУКТУРА ОТЧЁТА:')
    print('   📋 Лист 1: 1H Summary (сводка по 1-часовому анализу)')
    print('   📋 Лист 2: 1H Trade Details (детали всех сделок)')
    print('   📋 Лист 3: Quarterly Analysis (квартальная аналитика)')
    
except ImportError as e:
    print(f'❌ Требуются библиотеки: {e}')
except Exception as e:
    print(f'❌ Ошибка создания Excel: {e}')
    import traceback
    traceback.print_exc()

print()
print('✅ 1-ЧАСОВОЙ БЭКТЕСТ ЗАВЕРШЁН!')
print(f'📊 Итог: {final_profit:+.2f} USD ({return_pct:+.2f}%) за {len(trades)} сделок')
print(f'📈 Годовая доходность: {annualized_return:+.2f}%')
print(f'⏱️ Средняя сделка: {avg_holding_time:.1f} часов')