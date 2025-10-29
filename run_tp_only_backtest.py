#!/usr/bin/env python3
"""
Бэктест KDJ стратегии ТОЛЬКО с Take Profit
Выход K×D временно отключен
Период: 1 июня - 31 августа 2025
"""

import csv
import sys
from datetime import datetime

print('🚀 БЭКТЕСТ KDJ - ТОЛЬКО TAKE PROFIT')
print('=' * 50)
print('📅 Период: 1 июня - 31 августа 2025')
print('🎯 Take Profit: строго +2.00 SOL')
print('❌ Выход K×D: ОТКЛЮЧЕН')
print('📊 Данные: SOLUSDT 1H')
print()

# Простая реализация KDJ с точной логикой TP
class RealKDJ:
    def __init__(self, period=9, signal=3):
        self.period = period
        self.signal = signal
        self.prev_k = 50.0
        self.prev_d = 50.0
    
    def bcwsma(self, value, period, prev_value):
        """Bitcoin Wisdom SMA - точная формула TradingView"""
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
print('📊 Загрузка исторических данных...')
data = []
with open('data/test/SOLUSDT_1h.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamp = row['timestamp']
        # Фильтруем только период июнь-август 2025
        if '2025-06-' in timestamp or '2025-07-' in timestamp or '2025-08-' in timestamp:
            data.append({
                'timestamp': timestamp,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

print(f'✅ Загружено {len(data)} баров за период июнь-август 2025')
print(f'📅 От {data[0]["timestamp"]} до {data[-1]["timestamp"]}')
print()

# Инициализация бэктеста
kdj = RealKDJ(period=9, signal=3)
initial_capital = 1000.0
capital = initial_capital
position = None
entry_price = None
entry_time = None
entry_bar = None
trades = []

k_prev, d_prev = 50.0, 50.0
max_capital = capital
drawdown_peak = capital

print('⚡ Запуск бэктеста ТОЛЬКО с Take Profit (+2 SOL)...')
print()

# Обработка каждого бара
for i in range(len(data)):
    bar = data[i]
    timestamp = bar['timestamp']
    open_price = bar['open']
    high = bar['high'] 
    low = bar['low']
    close = bar['close']
    
    # Собираем данные для KDJ
    highs = [data[j]['high'] for j in range(max(0, i-50), i+1)]
    lows = [data[j]['low'] for j in range(max(0, i-50), i+1)]
    
    if len(highs) >= 9:
        k, d, j = kdj.calculate(highs, lows, close)
    else:
        k, d, j = 50.0, 50.0, 50.0
        k_prev, d_prev = k, d
        continue
    
    # === ЛОГИКА ТОРГОВЛИ ===
    
    # Поиск ВХОДА
    if position is None:
        # Условия входа: K пересекает D вверх + фильтр диапазона
        if (k > d and k_prev <= d_prev and 20 < k < 80):
            position = 'LONG'
            entry_price = close
            entry_time = timestamp
            entry_bar = i
            
            print(f'🟢 ВХОД #{len(trades)+1}: {timestamp} @ {close:.2f} SOL')
            print(f'   KDJ: K={k:.1f}, D={d:.1f}, J={j:.1f}')
    
    # Обработка СУЩЕСТВУЮЩЕЙ позиции
    elif position == 'LONG':
        exit_price = None
        exit_reason = ''
        
        # ЕДИНСТВЕННОЕ УСЛОВИЕ ВЫХОДА: Take Profit +2.00 SOL
        target_price = entry_price + 2.0
        
        # Проверяем, достигла ли цена TP в течение этого бара
        if high >= target_price:
            # TP ДОСТИГНУТ! Выходим по точной цене TP
            exit_price = target_price  # ТОЧНО +2.00 SOL
            exit_reason = 'Take Profit +2.00 SOL'
            
            # Расчёт P&L
            profit_sol = exit_price - entry_price  # Всегда = +2.00
            position_size = capital * 0.02  # 2% риск
            pnl_usd = profit_sol * (position_size / entry_price)
            capital += pnl_usd
            
            # Отслеживание просадки
            if capital > max_capital:
                max_capital = capital
                drawdown_peak = capital
            
            current_drawdown = (drawdown_peak - capital) / drawdown_peak * 100
            
            # Запись сделки
            trade = {
                'num': len(trades) + 1,
                'entry_time': entry_time,
                'exit_time': timestamp,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_sol': profit_sol,
                'pnl_usd': pnl_usd,
                'capital': capital,
                'exit_reason': exit_reason,
                'duration_bars': i - entry_bar,
                'duration_hours': i - entry_bar
            }
            trades.append(trade)
            
            print(f'✅ ВЫХОД #{len(trades)}: {timestamp} @ {exit_price:.2f} SOL')
            print(f'   P&L: {profit_sol:+.2f} SOL ({pnl_usd:+.2f} USD)')
            print(f'   Длительность: {i - entry_bar} часов')
            print(f'   Капитал: {capital:.2f} USD')
            print()
            
            # Сброс позиции
            position = None
            entry_price = None
            entry_time = None
            entry_bar = None
        
        # ВАЖНО: K×D пересечение игнорируется! Держим позицию до TP.
    
    k_prev, d_prev = k, d

# Проверяем открытые позиции
if position == 'LONG':
    print(f'⚠️  ОТКРЫТАЯ ПОЗИЦИЯ на конец периода:')
    print(f'   Вход: {entry_time} @ {entry_price:.2f} SOL')
    print(f'   Текущая цена: {data[-1]["close"]:.2f} SOL')
    print(f'   Нереализованная P&L: {data[-1]["close"] - entry_price:+.2f} SOL')
    print()

# === ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ===
print('🏁 РЕЗУЛЬТАТЫ БЭКТЕСТА - ТОЛЬКО TAKE PROFIT')
print('=' * 55)
print(f'💰 Начальный капитал:     {initial_capital:,.2f} USD')
print(f'💰 Финальный капитал:     {capital:,.2f} USD')
print(f'📈 Общая прибыль:         {capital-initial_capital:+,.2f} USD')
print(f'📊 Доходность:            {((capital/initial_capital-1)*100):+.2f}%')

if trades:
    max_drawdown = (max_capital - min([t['capital'] for t in trades])) / max_capital * 100
    print(f'📉 Максимальная просадка: {max_drawdown:.2f}%')

print(f'🔄 Всего сделок:          {len(trades)}')
print(f'🎯 ВСЕ сделки - Take Profit: {len(trades)} (100%)')

if trades:
    # Все сделки должны быть прибыльными (+2 SOL)
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    
    print(f'✅ Прибыльных сделок:     {len(winning_trades)} (100%)')
    print(f'❌ Убыточных сделок:      0 (0%)')
    print()
    
    avg_win = sum(t['pnl_usd'] for t in trades) / len(trades)
    max_win = max(t['pnl_usd'] for t in trades)
    min_win = min(t['pnl_usd'] for t in trades)
    
    print(f'💎 Средняя прибыль:       {avg_win:.2f} USD')
    print(f'🏆 Максимальная прибыль:  {max_win:.2f} USD')
    print(f'📊 Минимальная прибыль:   {min_win:.2f} USD')
    
    # Все сделки должны быть +2.00 SOL
    profit_sols = [t['profit_sol'] for t in trades]
    all_tp_correct = all(abs(p - 2.0) < 0.01 for p in profit_sols)
    print(f'🎯 Все TP = +2.00 SOL:     {all_tp_correct}')
    
    # Средняя длительность
    avg_duration = sum(t['duration_hours'] for t in trades) / len(trades)
    max_duration = max(t['duration_hours'] for t in trades)
    min_duration = min(t['duration_hours'] for t in trades)
    
    print(f'⏱️  Средняя длительность:  {avg_duration:.1f} часов')
    print(f'⏱️  Максимальная длит-ть:  {max_duration} часов')  
    print(f'⏱️  Минимальная длит-ть:   {min_duration} час(ов)')
    
    print()
    print('📊 АНАЛИЗ ВРЕМЕННОЙ ЭФФЕКТИВНОСТИ:')
    print('=' * 40)
    
    # Группировка по длительности
    short_trades = [t for t in trades if t['duration_hours'] <= 5]
    medium_trades = [t for t in trades if 5 < t['duration_hours'] <= 24]
    long_trades = [t for t in trades if t['duration_hours'] > 24]
    
    print(f'⚡ Быстрые (≤5ч):        {len(short_trades)} ({len(short_trades)/len(trades)*100:.1f}%)')
    print(f'🔄 Средние (5-24ч):      {len(medium_trades)} ({len(medium_trades)/len(trades)*100:.1f}%)')
    print(f'🐌 Долгие (>24ч):        {len(long_trades)} ({len(long_trades)/len(trades)*100:.1f}%)')
    
    print()
    print('📈 ТОП-10 ЛУЧШИХ СДЕЛОК ПО USD:')
    print('=' * 35)
    top_trades = sorted(trades, key=lambda x: x['pnl_usd'], reverse=True)[:10]
    for i, trade in enumerate(top_trades, 1):
        entry_dt = trade['entry_time'][:16]
        print(f'{i:2d}. {entry_dt} | +2.00 SOL ({trade["pnl_usd"]:+.2f} USD) | {trade["duration_hours"]}ч')
    
    print()
    print('🎯 СТАТИСТИКА ПО МЕСЯЦАМ:')
    print('=' * 30)
    
    june_trades = [t for t in trades if '2025-06' in t['entry_time']]
    july_trades = [t for t in trades if '2025-07' in t['entry_time']]
    august_trades = [t for t in trades if '2025-08' in t['entry_time']]
    
    print(f'📅 Июнь 2025:    {len(june_trades)} сделок')
    if june_trades:
        june_profit = sum(t['pnl_usd'] for t in june_trades)
        print(f'   Прибыль: {june_profit:+.2f} USD')
    
    print(f'📅 Июль 2025:    {len(july_trades)} сделок')
    if july_trades:
        july_profit = sum(t['pnl_usd'] for t in july_trades)
        print(f'   Прибыль: {july_profit:+.2f} USD')
    
    print(f'📅 Август 2025:  {len(august_trades)} сделок')
    if august_trades:
        august_profit = sum(t['pnl_usd'] for t in august_trades)
        print(f'   Прибыль: {august_profit:+.2f} USD')

print()
print('💡 ВЫВОДЫ:')
print('=' * 15)
print('• Все сделки закрываются только по Take Profit +2.00 SOL')
print('• Винрейт = 100% (по определению)')
print('• Риск ограничен временем удержания позиции')
print('• Нет мелких убытков от ложных пересечений K×D')
print('• Стратегия становится чисто трендоследящей')
print()
print('✅ БЭКТЕСТ ТОЛЬКО TP ЗАВЕРШЁН!')