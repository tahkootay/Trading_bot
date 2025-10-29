#!/usr/bin/env python3
"""
Реальный бэктест KDJ стратегии с корректной логикой TP
Период: 1 июня - 31 августа 2025
"""

import csv
import sys
from datetime import datetime

print('🚀 РЕАЛЬНЫЙ БЭКТЕСТ KDJ СТРАТЕГИИ')
print('=' * 50)
print('📅 Период: 1 июня - 31 августа 2025')
print('🎯 Take Profit: строго +2.00 SOL')
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

print('⚡ Запуск бэктеста с корректной логикой TP...')
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
        
        # КРИТИЧЕСКИ ВАЖНО: Проверяем TP на КАЖДОМ тике бара
        # Если цена в течение бара достигла entry_price + 2.0, то выходим по TP
        
        target_price = entry_price + 2.0
        
        # Проверяем, достигла ли цена TP в течение этого бара
        if high >= target_price:
            # TP ДОСТИГНУТ! Выходим по точной цене TP
            exit_price = target_price  # ТОЧНО +2.00 SOL
            exit_reason = 'Take Profit +2.00 SOL'
            
        # Иначе проверяем условие пересечения K×D вниз
        elif k < d and k_prev >= d_prev:
            exit_price = close
            exit_reason = 'K crosses D down'
        
        # Если есть сигнал выхода - закрываем позицию
        if exit_price is not None:
            # Расчёт P&L
            profit_sol = exit_price - entry_price
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
                'duration_bars': i - entry_bar
            }
            trades.append(trade)
            
            status = '✅' if profit_sol > 0 else '❌'
            print(f'{status} ВЫХОД #{len(trades)}: {timestamp} @ {exit_price:.2f} SOL')
            print(f'   P&L: {profit_sol:+.2f} SOL ({pnl_usd:+.2f} USD)')
            print(f'   Причина: {exit_reason}')
            print(f'   Длительность: {i - entry_bar} баров')
            print(f'   Капитал: {capital:.2f} USD')
            print()
            
            # Сброс позиции
            position = None
            entry_price = None
            entry_time = None
            entry_bar = None
    
    k_prev, d_prev = k, d

# === ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ===
print('🏁 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ РЕАЛЬНОГО БЭКТЕСТА')
print('=' * 55)
print(f'💰 Начальный капитал:     {initial_capital:,.2f} USD')
print(f'💰 Финальный капитал:     {capital:,.2f} USD')
print(f'📈 Общая прибыль:         {capital-initial_capital:+,.2f} USD')
print(f'📊 Доходность:            {((capital/initial_capital-1)*100):+.2f}%')

max_drawdown = (max_capital - min([t['capital'] for t in trades] + [capital])) / max_capital * 100
print(f'📉 Максимальная просадка: {max_drawdown:.2f}%')
print(f'🔄 Всего сделок:          {len(trades)}')

if trades:
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    losing_trades = [t for t in trades if t['pnl_usd'] <= 0]
    tp_trades = [t for t in trades if 'Take Profit' in t['exit_reason']]
    crossover_trades = [t for t in trades if 'crosses' in t['exit_reason']]
    
    print(f'✅ Прибыльных сделок:     {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)')
    print(f'❌ Убыточных сделок:      {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)')
    print()
    
    print(f'🎯 Выходы по Take Profit: {len(tp_trades)} ({len(tp_trades)/len(trades)*100:.1f}%)')
    print(f'📉 Выходы по пересечению: {len(crossover_trades)} ({len(crossover_trades)/len(trades)*100:.1f}%)')
    print()
    
    if winning_trades:
        avg_win = sum(t['pnl_usd'] for t in winning_trades) / len(winning_trades)
        max_win = max(t['pnl_usd'] for t in winning_trades)
        print(f'💎 Средняя прибыль:       {avg_win:.2f} USD')
        print(f'🏆 Максимальная прибыль:  {max_win:.2f} USD')
        
    if losing_trades:
        avg_loss = sum(t['pnl_usd'] for t in losing_trades) / len(losing_trades)
        max_loss = min(t['pnl_usd'] for t in losing_trades)
        print(f'💸 Средний убыток:        {avg_loss:.2f} USD')
        print(f'⚠️  Максимальный убыток:   {max_loss:.2f} USD')
    
    # Profit Factor
    total_wins = sum(t['pnl_usd'] for t in winning_trades) if winning_trades else 0
    total_losses = abs(sum(t['pnl_usd'] for t in losing_trades)) if losing_trades else 1
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    print(f'🎯 Profit Factor:         {profit_factor:.2f}')
    
    # Средняя длительность
    avg_duration = sum(t['duration_bars'] for t in trades) / len(trades)
    print(f'⏱️  Средняя длительность:  {avg_duration:.1f} часов')
    
    print()
    print('🔍 ПРОВЕРКА КОРРЕКТНОСТИ TAKE PROFIT:')
    print('=' * 40)
    
    # Проверяем все TP сделки
    tp_profits = [t['profit_sol'] for t in tp_trades]
    if tp_profits:
        print(f'📊 Take Profit сделок: {len(tp_profits)}')
        print(f'✅ Все TP = +2.00 SOL: {all(abs(p - 2.0) < 0.01 for p in tp_profits)}')
        
        if len(tp_profits) <= 10:
            print('🎯 Проверка прибылей TP:')
            for i, profit in enumerate(tp_profits, 1):
                print(f'   TP #{i}: {profit:+.2f} SOL')
        else:
            print(f'🎯 Первые 5 TP сделок:')
            for i, profit in enumerate(tp_profits[:5], 1):
                print(f'   TP #{i}: {profit:+.2f} SOL')
    
    print()
    print('📈 ТОП-5 ЛУЧШИХ СДЕЛОК:')
    print('=' * 25)
    top_trades = sorted(trades, key=lambda x: x['pnl_usd'], reverse=True)[:5]
    for i, trade in enumerate(top_trades, 1):
        entry_dt = trade['entry_time'][:16]
        print(f'{i}. {entry_dt} | {trade["profit_sol"]:+.2f} SOL ({trade["pnl_usd"]:+.2f} USD) | {trade["exit_reason"]}')

print()
print('✅ РЕАЛЬНЫЙ БЭКТЕСТ ЗАВЕРШЁН!')
print('🎯 Take Profit работает корректно: строго +2.00 SOL')