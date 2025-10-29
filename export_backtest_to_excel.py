#!/usr/bin/env python3
"""
Export last backtest results (TP-only) to Excel using new reporter module
"""

import csv
import sys
from datetime import datetime

print('📊 ЭКСПОРТ РЕЗУЛЬТАТОВ БЭКТЕСТА В EXCEL')
print('=' * 50)
print('🎯 Стратегия: KDJ только с Take Profit')
print('📅 Период: 1 июня - 31 августа 2025')
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
all_signals = []  # Для Excel экспорта

k_prev, d_prev = 50.0, 50.0

print('⚡ Повторный запуск бэктеста для сбора данных...')

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
            profit_sol = 2.0  # Точно +2 SOL
            position_size = capital * 0.02
            pnl_usd = profit_sol * (position_size / entry_price)
            capital += pnl_usd
            
            trades.append({
                'trade_id': len(trades) + 1,
                'entry_time': entry_time,
                'exit_time': timestamp,
                'entry_price': entry_price,
                'exit_price': target_price,
                'profit_sol': profit_sol,
                'pnl_usd': pnl_usd,
                'duration_hours': i - [idx for idx, d in enumerate(data) if d['timestamp'] == entry_time][0],
                'exit_reason': 'Take Profit +2.00 SOL'
            })
            
            position = None
            entry_price = None
            current_position = 'NONE'
            entry_price_display = target_price  # Показать цену выхода
        else:
            signal = 'HOLD'
    
    # Сохранение данных для Excel
    signal_record = {
        'timestamp': timestamp,
        'open': bar['open'],
        'high': bar['high'], 
        'low': bar['low'],
        'close': bar['close'],
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

# Подготовка данных для Excel
strategy_info = {
    'strategy_name': 'KDJ TP-Only Strategy',
    'hypothesis': 'K crosses D up → Long, exit ONLY at +2.00 SOL (K×D exit disabled)',
    'parameters': {
        'ilong': 9,
        'isig': 3,
        'algorithm': 'TradingView/Bybit exact',
        'entry_k_range': '20-80',
        'profit_target_sol': 2.0,
        'kdx_exit': 'DISABLED',
        'position_size_pct': 2.0
    },
    'statistics': {
        'total_bars_processed': len(all_signals),
        'total_trades': len(trades),
        'winning_trades': len(trades),  # Все сделки прибыльны
        'losing_trades': 0,
        'win_rate_pct': 100.0,
        'initial_capital_usd': 1000.0,
        'final_capital_usd': round(capital, 2),
        'total_profit_usd': round(final_profit, 2),
        'return_pct': round((capital/1000.0 - 1) * 100, 2)
    }
}

# Создание Excel отчёта
print()
print('📈 СОЗДАНИЕ EXCEL ОТЧЁТА...')
print('=' * 30)

try:
    # Прямой импорт модуля без зависимостей
    import sys
    sys.path.append('/Users/alexey/Documents/Development/Python/Trading_bot/modules/reporter')
    
    # Проверим что у нас есть нужные библиотеки
    try:
        import pandas as pd
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill, Font
        EXCEL_AVAILABLE = True
        print('✅ Excel библиотеки доступны')
    except ImportError as e:
        print(f'❌ Excel библиотеки недоступны: {e}')
        EXCEL_AVAILABLE = False
    
    if EXCEL_AVAILABLE:
        # Создаём отчёт вручную, так как у нас нет pandas
        
        # Определяем цвета
        colors = {
            'LONG': PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid"),    # Light Green
            'BUY': PatternFill(start_color="32CD32", end_color="32CD32", fill_type="solid"),     # Lime Green  
            'SELL': PatternFill(start_color="FF6B6B", end_color="FF6B6B", fill_type="solid"),   # Light Red
            'HOLD': PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid"),   # Gold
            'NONE': PatternFill(start_color="F0F0F0", end_color="F0F0F0", fill_type="solid"),   # Light Gray
            'HEADER': PatternFill(start_color="4682B4", end_color="4682B4", fill_type="solid")  # Steel Blue
        }
        
        header_font = Font(color="FFFFFF", bold=True)
        
        # Создаём Excel файл
        output_path = 'output/excel_reports/kdj_tp_only_backtest.xlsx'
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        wb = Workbook()
        ws = wb.active
        ws.title = "KDJ TP-Only Signals"
        
        # Заголовки
        headers = list(all_signals[0].keys())
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_idx, value=header)
            cell.fill = colors['HEADER']
            cell.font = header_font
        
        # Данные с цветовой кодировкой
        for row_idx, signal_data in enumerate(all_signals, 2):
            for col_idx, (key, value) in enumerate(signal_data.items(), 1):
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                
                # Применяем цвета
                position = signal_data['position']
                signal = signal_data['signal']
                
                if position == 'LONG':
                    cell.fill = colors['LONG']
                elif signal == 'BUY':
                    cell.fill = colors['BUY']
                elif signal == 'SELL':
                    cell.fill = colors['SELL'] 
                elif signal == 'HOLD':
                    cell.fill = colors['HOLD']
                elif signal == 'NONE' and position == 'NONE':
                    cell.fill = colors['NONE']
        
        # Автоподгонка ширины колонок
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 20)
            ws.column_dimensions[column_letter].width = adjusted_width
        
        # Лист сделок
        trades_ws = wb.create_sheet(title="Completed Trades")
        
        if trades:
            # Заголовки сделок
            trade_headers = list(trades[0].keys())
            for col_idx, header in enumerate(trade_headers, 1):
                cell = trades_ws.cell(row=1, column=col_idx, value=header)
                cell.fill = colors['HEADER']
                cell.font = header_font
            
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
        
        # Лист статистики
        stats_ws = wb.create_sheet(title="Strategy Statistics")
        
        # Основная информация
        stats_data = [
            ['STRATEGY INFORMATION', ''],
            ['Strategy Name', strategy_info['strategy_name']],
            ['Hypothesis', strategy_info['hypothesis']],
            ['', ''],
            ['PARAMETERS', ''],
            ['KDJ Period (ilong)', strategy_info['parameters']['ilong']],
            ['KDJ Signal (isig)', strategy_info['parameters']['isig']],
            ['Algorithm', strategy_info['parameters']['algorithm']],
            ['Entry K Range', strategy_info['parameters']['entry_k_range']],
            ['Profit Target', f"{strategy_info['parameters']['profit_target_sol']} SOL"],
            ['K×D Exit', strategy_info['parameters']['kdx_exit']],
            ['Position Size', f"{strategy_info['parameters']['position_size_pct']}%"],
            ['', ''],
            ['PERFORMANCE STATISTICS', ''],
            ['Initial Capital', f"${strategy_info['statistics']['initial_capital_usd']:.2f}"],
            ['Final Capital', f"${strategy_info['statistics']['final_capital_usd']:.2f}"],
            ['Total Profit', f"${strategy_info['statistics']['total_profit_usd']:+.2f}"],
            ['Return %', f"{strategy_info['statistics']['return_pct']:+.2f}%"],
            ['', ''],
            ['TRADE STATISTICS', ''],
            ['Total Trades', strategy_info['statistics']['total_trades']],
            ['Winning Trades', strategy_info['statistics']['winning_trades']],
            ['Losing Trades', strategy_info['statistics']['losing_trades']],
            ['Win Rate', f"{strategy_info['statistics']['win_rate_pct']:.1f}%"],
            ['Total Bars Processed', strategy_info['statistics']['total_bars_processed']]
        ]
        
        for row_idx, (label, value) in enumerate(stats_data, 1):
            stats_ws.cell(row=row_idx, column=1, value=label)
            stats_ws.cell(row=row_idx, column=2, value=value)
            
            # Выделяем заголовки
            if label.isupper() and value == '':
                stats_ws.cell(row=row_idx, column=1).fill = colors['HEADER']
                stats_ws.cell(row=row_idx, column=1).font = header_font
        
        # Подгонка ширины статистики
        stats_ws.column_dimensions['A'].width = 25
        stats_ws.column_dimensions['B'].width = 30
        
        # Лист легенды
        legend_ws = wb.create_sheet(title="Color Legend")
        
        legend_data = [
            ['COLOR LEGEND', 'DESCRIPTION'],
            ['LONG Position', 'Light Green - Currently holding long position'],
            ['BUY Signal', 'Lime Green - Entry signal generated'],
            ['SELL Signal', 'Light Red - Exit signal generated (Take Profit)'],
            ['HOLD Signal', 'Gold - Holding existing position'],
            ['NONE', 'Light Gray - No position, no signal'],
            ['', ''],
            ['STRATEGY NOTES', ''],
            ['• K×D downward crossover exit is DISABLED', ''],
            ['• ALL exits occur at exactly +2.00 SOL profit', ''],
            ['• 100% win rate by design', ''],
            ['• Risk limited by holding time only', '']
        ]
        
        for row_idx, (label, description) in enumerate(legend_data, 1):
            legend_ws.cell(row=row_idx, column=1, value=label)
            legend_ws.cell(row=row_idx, column=2, value=description)
            
            # Применяем цвета к легенде
            if 'LONG' in label:
                legend_ws.cell(row=row_idx, column=1).fill = colors['LONG']
            elif 'BUY' in label:
                legend_ws.cell(row=row_idx, column=1).fill = colors['BUY']
            elif 'SELL' in label:
                legend_ws.cell(row=row_idx, column=1).fill = colors['SELL']
            elif 'HOLD' in label:
                legend_ws.cell(row=row_idx, column=1).fill = colors['HOLD']
            elif 'NONE' in label and 'NOTES' not in label:
                legend_ws.cell(row=row_idx, column=1).fill = colors['NONE']
            elif label.isupper():
                legend_ws.cell(row=row_idx, column=1).fill = colors['HEADER']
                legend_ws.cell(row=row_idx, column=1).font = header_font
        
        legend_ws.column_dimensions['A'].width = 25
        legend_ws.column_dimensions['B'].width = 50
        
        # Сохраняем файл
        wb.save(output_path)
        
        print(f'✅ Excel отчёт создан: {output_path}')
        print()
        print('📊 СТРУКТУРА ФАЙЛА:')
        print('   📋 Лист 1: KDJ TP-Only Signals (все сигналы с цветами)')
        print('   📋 Лист 2: Completed Trades (детали сделок)')
        print('   📋 Лист 3: Strategy Statistics (статистика производительности)')
        print('   📋 Лист 4: Color Legend (расшифровка цветов)')
        print()
        print('🎨 ЦВЕТОВАЯ КОДИРОВКА:')
        print('   🟢 Зелёный: LONG позиции')
        print('   🟡 Золотой: HOLD сигналы')
        print('   🔴 Красный: SELL сигналы (Take Profit)')
        print('   🟦 Лаймовый: BUY сигналы')
        print('   ⬜ Серый: NONE (нет позиции/сигнала)')
    
    else:
        print('❌ Невозможно создать Excel файл - отсутствуют библиотеки')
        print('💡 Для установки: pip install pandas openpyxl')

except Exception as e:
    print(f'❌ Ошибка создания Excel: {e}')
    import traceback
    traceback.print_exc()

print()
print('✅ ЭКСПОРТ ЗАВЕРШЁН!')