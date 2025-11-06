#!/usr/bin/env python3
"""
Запуск бэктеста модели с горизонтом 5 на октябрьских данных 2025.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.ml_training.backtester import MLBacktester

def run_october_backtest():
    """Запуск бэктеста на данных с готовыми предсказаниями."""
    
    print("🚀 Запуск бэктеста модели горизонт-5 на тестовых данных")
    
    # Используем данные с готовыми предсказаниями
    data_path = "data/processed/test_with_preds_target5.csv"
    
    # Проверяем наличие файла данных
    if not Path(data_path).exists():
        print(f"❌ Файл данных не найден: {data_path}")
        return None
    
    print(f"✅ Файл данных найден: {data_path}")
    
    # Загружаем данные
    print("📊 Загружаем данные...")
    df = pd.read_csv(data_path)
    print(f"   Загружено {len(df)} записей")
    print(f"   Период: {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}")
    
    # Данные уже содержат предсказания
    print("🔮 Используем готовые предсказания...")
    
    # Создаем копию данных с нужными колонками
    backtest_data = df.copy()
    
    # Переименовываем колонки для совместимости
    backtest_data['prediction'] = backtest_data['pred']
    backtest_data['prob_buy'] = backtest_data['prob_up']  # Вероятность роста
    backtest_data['prob_sell'] = 1 - backtest_data['prob_up']  # Вероятность падения
    
    print(f"   Предсказаний загружено: {len(backtest_data['prediction'])}")
    print(f"   Распределение: BUY={np.sum(backtest_data['prediction']==1)}, SELL={np.sum(backtest_data['prediction']==0)}")
    
    # Запускаем бэктест с разными настройками
    print("📈 Запуск бэктеста...")
    
    # Базовые настройки
    initial_balance = 1000.0
    fee = 0.001  # 0.1% комиссия
    buy_threshold = 0.6   # Покупаем если вероятность > 60%
    sell_threshold = 0.4  # Продаем если вероятность < 40%
    
    print(f"   Начальный баланс: ${initial_balance}")
    print(f"   Комиссия: {fee*100}%")
    print(f"   Пороги: BUY > {buy_threshold}, SELL < {sell_threshold}")
    
    # Простая логика бэктеста
    balance = initial_balance
    position = 0  # 0 - нет позиции, 1 - в позиции
    entry_price = 0
    trades = []
    equity_curve = []
    
    for i, row in backtest_data.iterrows():
        price = row['close']
        prob_buy = row['prob_buy']
        
        # Сигналы
        buy_signal = prob_buy > buy_threshold
        sell_signal = prob_buy < sell_threshold
        
        # Торговая логика
        if position == 0 and buy_signal:  # Входим в позицию
            entry_price = price * (1 + fee)  # С учетом комиссии
            position = 1
            trades.append({
                'timestamp': row['timestamp'],
                'action': 'BUY',
                'price': entry_price,
                'prob': prob_buy,
                'balance': balance
            })
            
        elif position == 1 and sell_signal:  # Выходим из позиции
            exit_price = price * (1 - fee)  # С учетом комиссии
            pnl = (exit_price - entry_price) / entry_price
            balance = balance * (1 + pnl)
            position = 0
            
            trades.append({
                'timestamp': row['timestamp'],
                'action': 'SELL',
                'price': exit_price,
                'prob': prob_buy,
                'balance': balance,
                'pnl_pct': pnl * 100,
                'entry_price': entry_price
            })
        
        # Обновляем кривую капитала
        if position == 1:
            current_value = balance * (price / entry_price)
        else:
            current_value = balance
            
        equity_curve.append({
            'timestamp': row['timestamp'],
            'balance': current_value,
            'price': price,
            'position': position
        })
    
    # Если остались в позиции - закрываем принудительно
    if position == 1:
        final_price = backtest_data['close'].iloc[-1]
        exit_price = final_price * (1 - fee)
        pnl = (exit_price - entry_price) / entry_price
        balance = balance * (1 + pnl)
        
        trades.append({
            'timestamp': backtest_data['timestamp'].iloc[-1],
            'action': 'SELL_FINAL',
            'price': exit_price,
            'prob': backtest_data['prob_buy'].iloc[-1],
            'balance': balance,
            'pnl_pct': pnl * 100,
            'entry_price': entry_price
        })
    
    # Анализ результатов
    print("\n📊 РЕЗУЛЬТАТЫ БЭКТЕСТА:")
    print("=" * 50)
    
    total_return = (balance - initial_balance) / initial_balance * 100
    print(f"💰 Начальный баланс: ${initial_balance:.2f}")
    print(f"💰 Финальный баланс: ${balance:.2f}")
    print(f"📈 Общая доходность: {total_return:.2f}%")
    
    if len(trades) > 0:
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] in ['SELL', 'SELL_FINAL']]
        
        print(f"🔄 Всего сделок: {len(buy_trades)} входов, {len(sell_trades)} выходов")
        
        if len(sell_trades) > 0:
            pnls = [t['pnl_pct'] for t in sell_trades if 'pnl_pct' in t]
            if pnls:
                winning_trades = [p for p in pnls if p > 0]
                losing_trades = [p for p in pnls if p <= 0]
                
                print(f"✅ Прибыльных сделок: {len(winning_trades)} ({len(winning_trades)/len(pnls)*100:.1f}%)")
                print(f"❌ Убыточных сделок: {len(losing_trades)} ({len(losing_trades)/len(pnls)*100:.1f}%)")
                print(f"📊 Средняя прибыль: {np.mean(pnls):.2f}%")
                if winning_trades:
                    print(f"📊 Средняя прибыльная: {np.mean(winning_trades):.2f}%")
                if losing_trades:
                    print(f"📊 Средняя убыточная: {np.mean(losing_trades):.2f}%")
    
    # Buy & Hold сравнение
    start_price = backtest_data['close'].iloc[0]
    end_price = backtest_data['close'].iloc[-1]
    buy_hold_return = (end_price - start_price) / start_price * 100
    
    print(f"\n📈 Buy & Hold доходность: {buy_hold_return:.2f}%")
    print(f"🎯 Превышение B&H: {total_return - buy_hold_return:.2f}%")
    
    # Сохраняем результаты
    print("\n💾 Сохранение результатов...")
    
    # DataFrame с результатами
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    
    # Сохраняем в results/
    results_dir = Path("results/horizon5_backtest")
    results_dir.mkdir(exist_ok=True)
    
    if not trades_df.empty:
        trades_df.to_csv(results_dir / "trades.csv", index=False)
        print(f"   Сделки сохранены: {results_dir / 'trades.csv'}")
    
    equity_df.to_csv(results_dir / "equity_curve.csv", index=False)
    print(f"   Кривая капитала сохранена: {results_dir / 'equity_curve.csv'}")
    
    # Статистика
    stats = {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_return_pct': total_return,
        'buy_hold_return_pct': buy_hold_return,
        'excess_return_pct': total_return - buy_hold_return,
        'total_trades': len(buy_trades),
        'winning_trades': len(winning_trades) if 'winning_trades' in locals() else 0,
        'win_rate_pct': len(winning_trades)/len(pnls)*100 if 'pnls' in locals() and pnls else 0,
        'avg_return_pct': np.mean(pnls) if 'pnls' in locals() and pnls else 0
    }
    
    with open(results_dir / "statistics.txt", 'w') as f:
        f.write("BACKTEST STATISTICS\n")
        f.write("==================\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"   Статистика сохранена: {results_dir / 'statistics.txt'}")
    
    print("\n🎉 Бэктест завершен!")
    return stats

if __name__ == "__main__":
    try:
        results = run_october_backtest()
    except Exception as e:
        print(f"❌ Ошибка при выполнении бэктеста: {e}")
        import traceback
        traceback.print_exc()