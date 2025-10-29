#!/usr/bin/env python3
"""
KDJ TP-Only Strategy - Full Year Backtest (June-August 2025)
Generates comprehensive Excel report with yearly results
"""

import csv
import sys
import os
from datetime import datetime

print('📊 KDJ TP-ONLY STRATEGY - ГОДОВОЙ БЭКТЕСТ')
print('=' * 50)
print('🎯 Стратегия: K×D вход, только Take Profit +2.00 SOL выход')
print('📅 Период: Весь доступный год (июнь-август 2025)')
print('💡 Алгоритм: TradingView/Bybit точный')
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

# Загрузка полного года данных
print('📊 Загрузка данных за весь год...')
data = []
with open('data/test/SOLUSDT_1h.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamp = row['timestamp']
        # Загружаем весь доступный период 2025 года
        if '2025-' in timestamp:
            data.append({
                'timestamp': timestamp,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

print(f'✅ Загружено {len(data)} часовых баров')
print(f'📅 Период: {data[0]["timestamp"]} → {data[-1]["timestamp"]}')

# Инициализация стратегии
kdj = KDJTradingView(ilong=9, isig=3)
capital = 1000.0
position = None
entry_price = None
entry_time = None
exit_price = None
trades = []
all_signals = []

# Переменные для отслеживания максимального капитала и просадки
max_capital = capital
max_drawdown = 0.0

k_prev, d_prev = 50.0, 50.0

print('⚡ Запуск полного бэктеста...')

# Основной цикл бэктеста
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
    
    # Инициализация переменных для записи
    signal = 'NONE'
    current_position = 'NONE' if position is None else position
    entry_price_display = 0.0 if entry_price is None else entry_price
    exit_price_display = 0.0 if exit_price is None else exit_price
    current_pnl = 0.0
    
    # Логика торговых сигналов
    if position is None:
        # Поиск сигнала входа: K пересекает D вверх в зоне 20-80
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
                'exit_reason': 'Take Profit +2.00 SOL'
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
    
    # Сохранение данных для полного отчёта
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

# Расчёт финальной статистики
final_profit = capital - 1000.0
return_pct = (capital / 1000.0 - 1) * 100
total_signals_buy = sum(1 for s in all_signals if s['signal'] == 'BUY')
total_signals_sell = sum(1 for s in all_signals if s['signal'] == 'SELL')

print()
print('📊 РЕЗУЛЬТАТЫ ГОДОВОГО БЭКТЕСТА')
print('=' * 40)
print(f'📅 Период: {all_signals[0]["timestamp"]} → {all_signals[-1]["timestamp"]}')
print(f'📊 Обработано баров: {len(all_signals)}')
print(f'🔄 Всего сделок: {len(trades)}')
print(f'📈 BUY сигналов: {total_signals_buy}')
print(f'📉 SELL сигналов: {total_signals_sell}')
print(f'💰 Начальный капитал: $1000.00')
print(f'💰 Финальный капитал: ${capital:.2f}')
print(f'💎 Общая прибыль: ${final_profit:+.2f}')
print(f'📈 Доходность: {return_pct:+.2f}%')
print(f'📉 Максимальная просадка: {max_drawdown:.2f}%')
if trades:
    avg_profit = final_profit / len(trades)
    avg_duration = sum(t['duration_hours'] for t in trades) / len(trades)
    print(f'💵 Средняя прибыль за сделку: ${avg_profit:.3f}')
    print(f'⏱️ Средняя длительность сделки: {avg_duration:.1f} часов')

# Создание Excel отчёта
print()
print('📈 СОЗДАНИЕ EXCEL ОТЧЁТА...')

try:
    # Проверяем доступность библиотек
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    
    # Создание Excel файла
    output_path = 'output/excel_reports/KDJ_Yearly_Report.xlsx'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Цветовая схема
    colors = {
        'LONG': PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid"),    # Light Green
        'BUY': PatternFill(start_color="32CD32", end_color="32CD32", fill_type="solid"),     # Lime Green  
        'SELL': PatternFill(start_color="FF6B6B", end_color="FF6B6B", fill_type="solid"),   # Light Red
        'HOLD': PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid"),   # Gold
        'NONE': PatternFill(start_color="F0F0F0", end_color="F0F0F0", fill_type="solid"),   # Light Gray
        'HEADER': PatternFill(start_color="4682B4", end_color="4682B4", fill_type="solid")  # Steel Blue
    }
    
    header_font = Font(color="FFFFFF", bold=True)
    center_alignment = Alignment(horizontal="center")
    
    wb = Workbook()
    
    # Лист 1: Сводная статистика
    summary_ws = wb.active
    summary_ws.title = "Yearly Summary"
    
    summary_data = [
        ['KDJ TP-ONLY STRATEGY - YEARLY REPORT', ''],
        ['', ''],
        ['STRATEGY INFORMATION', ''],
        ['Strategy Name', 'KDJ TP-Only (Take Profit Only)'],
        ['Algorithm', 'TradingView/Bybit Exact KDJ'],
        ['Entry Condition', 'K crosses D upward in range 20-80'],
        ['Exit Condition', 'Take Profit +2.00 SOL only'],
        ['K×D Exit', 'DISABLED'],
        ['', ''],
        ['PARAMETERS', ''],
        ['KDJ Period (ilong)', '9'],
        ['KDJ Signal (isig)', '3'],
        ['Position Size', '2% of capital'],
        ['Profit Target', '+2.00 SOL'],
        ['', ''],
        ['PERFORMANCE RESULTS', ''],
        ['Data Period', f'{all_signals[0]["timestamp"]} → {all_signals[-1]["timestamp"]}'],
        ['Total Bars Processed', f'{len(all_signals)}'],
        ['Initial Capital', f'${1000.0:.2f}'],
        ['Final Capital', f'${capital:.2f}'],
        ['Total Profit', f'${final_profit:+.2f}'],
        ['Return %', f'{return_pct:+.2f}%'],
        ['Max Drawdown', f'{max_drawdown:.2f}%'],
        ['', ''],
        ['TRADE STATISTICS', ''],
        ['Total Trades', f'{len(trades)}'],
        ['Winning Trades', f'{len(trades)}'],
        ['Losing Trades', '0'],
        ['Win Rate', '100.0%'],
        ['BUY Signals Generated', f'{total_signals_buy}'],
        ['SELL Signals Generated', f'{total_signals_sell}'],
        ['Avg Profit per Trade', f'${avg_profit:.3f}' if trades else '$0.000'],
        ['Avg Trade Duration', f'{avg_duration:.1f} hours' if trades else '0.0 hours'],
        ['', ''],
        ['EXPORT INFO', ''],
        ['Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['File Name', 'KDJ_Yearly_Report.xlsx']
    ]
    
    # Заполнение сводки
    for row_idx, (label, value) in enumerate(summary_data, 1):
        summary_ws.cell(row=row_idx, column=1, value=label)
        summary_ws.cell(row=row_idx, column=2, value=value)
        
        # Форматирование заголовков
        if label.isupper() and ('INFORMATION' in label or 'PARAMETERS' in label or 'RESULTS' in label or 'STATISTICS' in label or 'INFO' in label):
            summary_ws.cell(row=row_idx, column=1).fill = colors['HEADER']
            summary_ws.cell(row=row_idx, column=1).font = header_font
    
    summary_ws.column_dimensions['A'].width = 25
    summary_ws.column_dimensions['B'].width = 40
    
    # Лист 2: Все торговые сигналы (сокращённый для больших данных)
    signals_ws = wb.create_sheet(title="Trading Signals")
    
    # Заголовки
    headers = list(all_signals[0].keys())
    for col_idx, header in enumerate(headers, 1):
        cell = signals_ws.cell(row=1, column=col_idx, value=header)
        cell.fill = colors['HEADER']
        cell.font = header_font
        cell.alignment = center_alignment
    
    # Данные (показываем только сигналы, не NONE/HOLD для экономии места)
    row_idx = 2
    for signal_data in all_signals:
        # Показываем все важные сигналы
        if signal_data['signal'] in ['BUY', 'SELL'] or signal_data['position'] == 'LONG':
            for col_idx, (key, value) in enumerate(signal_data.items(), 1):
                cell = signals_ws.cell(row=row_idx, column=col_idx, value=value)
                
                # Применяем цвета
                position = signal_data['position']
                signal = signal_data['signal']
                
                if position == 'LONG':
                    cell.fill = colors['LONG']
                elif signal == 'BUY':
                    cell.fill = colors['BUY']
                elif signal == 'SELL':
                    cell.fill = colors['SELL']
            
            row_idx += 1
    
    # Автоподгонка ширины
    for column in signals_ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 20)
        signals_ws.column_dimensions[column_letter].width = adjusted_width
    
    # Лист 3: Завершённые сделки
    if trades:
        trades_ws = wb.create_sheet(title="Completed Trades")
        
        # Заголовки сделок
        trade_headers = list(trades[0].keys())
        for col_idx, header in enumerate(trade_headers, 1):
            cell = trades_ws.cell(row=1, column=col_idx, value=header)
            cell.fill = colors['HEADER']
            cell.font = header_font
            cell.alignment = center_alignment
        
        # Данные сделок
        for row_idx, trade in enumerate(trades, 2):
            for col_idx, (key, value) in enumerate(trade.items(), 1):
                cell = trades_ws.cell(row=row_idx, column=col_idx, value=value)
                # Все сделки прибыльны - зелёный цвет
                cell.fill = colors['BUY']
        
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
            adjusted_width = min(max_length + 2, 25)
            trades_ws.column_dimensions[column_letter].width = adjusted_width
    
    # Сохранение файла
    wb.save(output_path)
    
    print(f'✅ Годовой Excel отчёт создан: {output_path}')
    print()
    print('📊 СТРУКТУРА ОТЧЁТА:')
    print('   📋 Лист 1: Yearly Summary (сводная статистика)')
    print('   📋 Лист 2: Trading Signals (ключевые торговые сигналы)')
    print('   📋 Лист 3: Completed Trades (детали всех сделок)')
    
except ImportError as e:
    print(f'❌ Требуются библиотеки: {e}')
    print('💡 Установка: pip install pandas openpyxl')
except Exception as e:
    print(f'❌ Ошибка создания Excel: {e}')

print()
print('✅ ГОДОВОЙ БЭКТЕСТ ЗАВЕРШЁН!')
print(f'📊 Результат: {final_profit:+.2f} USD ({return_pct:+.2f}%) за {len(trades)} сделок')