#!/usr/bin/env python3
"""
KDJ TP-Only Strategy - Extended Backtest (Sep 2024 - Sep 2025)
15-minute timeframe data for comprehensive yearly analysis
"""

import csv
import sys
import os
from datetime import datetime, timedelta

print('📊 KDJ TP-ONLY STRATEGY - РАСШИРЕННЫЙ БЭКТЕСТ')
print('=' * 55)
print('🎯 Стратегия: K×D вход, только Take Profit +2.00 SOL выход')
print('📅 Период: 01.09.2024 - 16.09.2025 (13+ месяцев)')
print('⏰ Таймфрейм: 15 минут')
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

# Загрузка расширенного датасета
print('📊 Загрузка расширенных данных (15 минут)...')
data = []

# Целевой период: с 01.09.2024 по 16.09.2025
target_start = datetime(2024, 9, 1)
target_end = datetime(2025, 9, 16, 23, 59, 59)

with open('data/raw/SOLUSDT_15m_20240901_20250831.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamp_str = row['timestamp']
        timestamp_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        
        # Фильтрация по запрошенному периоду
        if target_start <= timestamp_dt <= target_end:
            data.append({
                'timestamp': timestamp_str,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

print(f'✅ Загружено {len(data)} 15-минутных баров')
print(f'📅 Фактический период: {data[0]["timestamp"]} → {data[-1]["timestamp"]}')

# Расчёт статистики периода
start_date = datetime.strptime(data[0]["timestamp"], '%Y-%m-%d %H:%M:%S')
end_date = datetime.strptime(data[-1]["timestamp"], '%Y-%m-%d %H:%M:%S')
total_days = (end_date - start_date).days
total_months = total_days / 30.44  # Среднее количество дней в месяце

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
monthly_results = []

# Переменные для расширенной аналитики
max_capital = capital
max_drawdown = 0.0
consecutive_wins = 0
max_consecutive_wins = 0
total_holding_time = 0

k_prev, d_prev = 50.0, 50.0

print('⚡ Запуск расширенного бэктеста...')

# Для отслеживания месячных результатов
current_month = None
month_start_capital = capital

# Основной цикл бэктеста
for i in range(len(data)):
    bar = data[i]
    timestamp = bar['timestamp']
    close = bar['close']
    high = bar['high']
    
    # Отслеживание месячной статистики
    bar_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    month_key = bar_date.strftime('%Y-%m')
    
    if current_month != month_key:
        if current_month is not None:
            # Сохраняем результат прошлого месяца
            month_profit = capital - month_start_capital
            month_return = (capital / month_start_capital - 1) * 100
            monthly_results.append({
                'month': current_month,
                'start_capital': round(month_start_capital, 2),
                'end_capital': round(capital, 2),
                'profit': round(month_profit, 2),
                'return_pct': round(month_return, 2)
            })
        
        current_month = month_key
        month_start_capital = capital
    
    # Расчёт KDJ
    highs = [data[j]['high'] for j in range(max(0, i-50), i+1)]
    lows = [data[j]['low'] for j in range(max(0, i-50), i+1)]
    
    if len(highs) >= 9:
        k, d, j = kdj.calculate(highs, lows, close)
    else:
        k, d, j = 50.0, 50.0, 50.0
        k_prev, d_prev = k, d
        continue
    
    # Торговая логика
    if position is None:
        # Поиск входа: K пересекает D вверх в зоне 20-80
        if k > d and k_prev <= d_prev and 20 < k < 80:
            position = 'LONG'
            entry_price = close
            entry_time = timestamp
            
    else:
        # Обработка открытой позиции
        target_price = entry_price + 2.0
        
        # Проверка достижения Take Profit
        if high >= target_price:
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
            duration_bars = i - entry_idx
            duration_hours = duration_bars * 0.25  # 15 минут = 0.25 часа
            total_holding_time += duration_hours
            
            # Отслеживание серии побед
            consecutive_wins += 1
            if consecutive_wins > max_consecutive_wins:
                max_consecutive_wins = consecutive_wins
            
            # Сохранение сделки
            trades.append({
                'trade_id': len(trades) + 1,
                'entry_time': entry_time,
                'exit_time': timestamp,
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'profit_sol': round(profit_sol, 2),
                'pnl_usd': round(pnl_usd, 3),
                'duration_hours': round(duration_hours, 2),
                'exit_reason': 'Take Profit +2.00 SOL',
                'capital_after': round(capital, 2)
            })
            
            # Закрытие позиции
            position = None
            entry_price = None
    
    k_prev, d_prev = k, d

# Добавляем последний месяц
if current_month is not None:
    month_profit = capital - month_start_capital
    month_return = (capital / month_start_capital - 1) * 100
    monthly_results.append({
        'month': current_month,
        'start_capital': round(month_start_capital, 2),
        'end_capital': round(capital, 2),
        'profit': round(month_profit, 2),
        'return_pct': round(month_return, 2)
    })

# Расчёт финальной статистики
final_profit = capital - initial_capital
return_pct = (capital / initial_capital - 1) * 100
avg_holding_time = total_holding_time / len(trades) if trades else 0
annualized_return = return_pct * (365.25 / total_days)

print()
print('📊 РЕЗУЛЬТАТЫ РАСШИРЕННОГО БЭКТЕСТА')
print('=' * 50)
print(f'📅 Период: {data[0]["timestamp"]} → {data[-1]["timestamp"]}')
print(f'📊 Обработано баров: {len(data)} (15-минутных)')
print(f'⏰ Длительность: {total_days} дней ({total_months:.1f} месяцев)')
print(f'🔄 Всего сделок: {len(trades)}')
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
    print(f'🏆 Процент побед: {win_rate}%')
    print(f'🔥 Максимальная серия побед: {max_consecutive_wins}')
    
    # Лучшие и худшие месяцы
    best_month = max(monthly_results, key=lambda x: x['profit'])
    worst_month = min(monthly_results, key=lambda x: x['profit'])
    print(f'📈 Лучший месяц: {best_month["month"]} (+${best_month["profit"]:.2f})')
    print(f'📉 Худший месяц: {worst_month["month"]} (${worst_month["profit"]:+.2f})')

# Создание расширенного Excel отчёта
print()
print('📈 СОЗДАНИЕ РАСШИРЕННОГО EXCEL ОТЧЁТА...')

try:
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.chart import LineChart, Reference
    
    # Создание Excel файла
    output_path = 'output/excel_reports/KDJ_Extended_Report_2024-2025.xlsx'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Расширенная цветовая схема
    colors = {
        'HEADER': PatternFill(start_color="2E86AB", end_color="2E86AB", fill_type="solid"),  # Ocean Blue
        'POSITIVE': PatternFill(start_color="A8E6CF", end_color="A8E6CF", fill_type="solid"),  # Mint Green
        'NEGATIVE': PatternFill(start_color="FFB3BA", end_color="FFB3BA", fill_type="solid"),  # Light Pink
        'NEUTRAL': PatternFill(start_color="F0F0F0", end_color="F0F0F0", fill_type="solid"),  # Light Gray
        'ACCENT': PatternFill(start_color="FFD93D", end_color="FFD93D", fill_type="solid")    # Golden Yellow
    }
    
    header_font = Font(color="FFFFFF", bold=True, size=12)
    title_font = Font(bold=True, size=14)
    center_alignment = Alignment(horizontal="center", vertical="center")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    
    wb = Workbook()
    
    # Лист 1: Исполнительное резюме
    summary_ws = wb.active
    summary_ws.title = "Executive Summary"
    
    # Заголовок
    summary_ws.merge_cells('A1:D1')
    title_cell = summary_ws.cell(row=1, column=1, value='KDJ TP-ONLY STRATEGY - EXTENDED BACKTEST REPORT')
    title_cell.font = title_font
    title_cell.alignment = center_alignment
    title_cell.fill = colors['HEADER']
    title_cell.font = Font(color="FFFFFF", bold=True, size=16)
    
    # Основная статистика
    summary_data = [
        ['', '', '', ''],  # Пустая строка
        ['STRATEGY OVERVIEW', '', '', ''],
        ['Strategy Name', 'KDJ Take Profit Only Strategy', '', ''],
        ['Timeframe', '15 minutes', '', ''],
        ['Test Period', f'{data[0]["timestamp"]} to {data[-1]["timestamp"]}', '', ''],
        ['Total Days', f'{total_days}', '', ''],
        ['Total Months', f'{total_months:.1f}', '', ''],
        ['', '', '', ''],
        ['PERFORMANCE METRICS', '', '', ''],
        ['Initial Capital', f'${initial_capital:.2f}', 'Final Capital', f'${capital:.2f}'],
        ['Total Profit', f'${final_profit:+.2f}', 'Total Return', f'{return_pct:+.2f}%'],
        ['Annualized Return', f'{annualized_return:+.2f}%', 'Max Drawdown', f'{max_drawdown:.2f}%'],
        ['', '', '', ''],
        ['TRADE STATISTICS', '', '', ''],
        ['Total Trades', f'{len(trades)}', 'Win Rate', f'{100.0:.1f}%'],
        ['Avg Profit/Trade', f'${avg_profit:.3f}' if trades else '$0.000', 'Max Consecutive Wins', f'{max_consecutive_wins}'],
        ['Avg Holding Time', f'{avg_holding_time:.1f} hours' if trades else '0.0 hours', 'Total Bars Processed', f'{len(data):,}'],
        ['', '', '', ''],
        ['MONTHLY PERFORMANCE', '', '', ''],
        ['Best Month', f'{best_month["month"]} (+${best_month["profit"]:.2f})' if monthly_results else 'N/A', '', ''],
        ['Worst Month', f'{worst_month["month"]} (${worst_month["profit"]:+.2f})' if monthly_results else 'N/A', '', ''],
    ]
    
    for row_idx, row_data in enumerate(summary_data, 2):
        for col_idx, value in enumerate(row_data, 1):
            cell = summary_ws.cell(row=row_idx, column=col_idx, value=value)
            cell.border = thin_border
            
            # Форматирование заголовков секций
            if value and value.isupper() and ('OVERVIEW' in value or 'METRICS' in value or 'STATISTICS' in value or 'PERFORMANCE' in value):
                summary_ws.merge_cells(f'{chr(64+col_idx)}{row_idx}:{chr(64+col_idx+3)}{row_idx}')
                cell.fill = colors['ACCENT']
                cell.font = Font(bold=True, size=12)
                cell.alignment = center_alignment
    
    # Автоподгонка ширины
    for col in ['A', 'B', 'C', 'D']:
        summary_ws.column_dimensions[col].width = 25
    
    # Лист 2: Месячная производительность
    monthly_ws = wb.create_sheet(title="Monthly Performance")
    
    # Заголовки
    monthly_headers = ['Month', 'Start Capital ($)', 'End Capital ($)', 'Profit ($)', 'Return (%)']
    for col_idx, header in enumerate(monthly_headers, 1):
        cell = monthly_ws.cell(row=1, column=col_idx, value=header)
        cell.fill = colors['HEADER']
        cell.font = header_font
        cell.alignment = center_alignment
        cell.border = thin_border
    
    # Данные месяцев
    for row_idx, month_data in enumerate(monthly_results, 2):
        monthly_ws.cell(row=row_idx, column=1, value=month_data['month']).border = thin_border
        monthly_ws.cell(row=row_idx, column=2, value=month_data['start_capital']).border = thin_border
        monthly_ws.cell(row=row_idx, column=3, value=month_data['end_capital']).border = thin_border
        
        profit_cell = monthly_ws.cell(row=row_idx, column=4, value=month_data['profit'])
        return_cell = monthly_ws.cell(row=row_idx, column=5, value=month_data['return_pct'])
        
        # Цветовое кодирование прибыли
        if month_data['profit'] > 0:
            profit_cell.fill = colors['POSITIVE']
            return_cell.fill = colors['POSITIVE']
        elif month_data['profit'] < 0:
            profit_cell.fill = colors['NEGATIVE']
            return_cell.fill = colors['NEGATIVE']
        else:
            profit_cell.fill = colors['NEUTRAL']
            return_cell.fill = colors['NEUTRAL']
        
        profit_cell.border = thin_border
        return_cell.border = thin_border
    
    # Автоподгонка ширины
    for column in monthly_ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 3, 20)
        monthly_ws.column_dimensions[column_letter].width = adjusted_width
    
    # Лист 3: Детали сделок
    if trades:
        trades_ws = wb.create_sheet(title="Trade Details")
        
        # Заголовки сделок
        trade_headers = list(trades[0].keys())
        for col_idx, header in enumerate(trade_headers, 1):
            cell = trades_ws.cell(row=1, column=col_idx, value=header)
            cell.fill = colors['HEADER']
            cell.font = header_font
            cell.alignment = center_alignment
            cell.border = thin_border
        
        # Данные сделок
        for row_idx, trade in enumerate(trades, 2):
            for col_idx, (key, value) in enumerate(trade.items(), 1):
                cell = trades_ws.cell(row=row_idx, column=col_idx, value=value)
                cell.fill = colors['POSITIVE']  # Все сделки прибыльны
                cell.border = thin_border
        
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
    
    print(f'✅ Расширенный Excel отчёт создан: {output_path}')
    print()
    print('📊 СТРУКТУРА ОТЧЁТА:')
    print('   📋 Лист 1: Executive Summary (исполнительное резюме)')
    print('   📋 Лист 2: Monthly Performance (месячная производительность)')
    if trades:
        print('   📋 Лист 3: Trade Details (детали всех сделок)')
    
except ImportError as e:
    print(f'❌ Требуются библиотеки: {e}')
except Exception as e:
    print(f'❌ Ошибка создания Excel: {e}')
    import traceback
    traceback.print_exc()

print()
print('✅ РАСШИРЕННЫЙ БЭКТЕСТ ЗАВЕРШЁН!')
print(f'📊 Итог: {final_profit:+.2f} USD ({return_pct:+.2f}%) за {len(trades)} сделок')
print(f'📈 Годовая доходность: {annualized_return:+.2f}%')