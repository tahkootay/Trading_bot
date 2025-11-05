#!/usr/bin/env python3
"""
Бэктест модели horizon_5 на октябрьских данных 2025 года.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_horizon5_backtest():
    """Запуск бэктеста модели horizon_5 на октябрьских данных 2025."""
    
    print("🚀 БЭКТЕСТ МОДЕЛИ HORIZON_5 НА ОКТЯБРЬСКИХ ДАННЫХ 2025")
    print("=" * 60)
    
    # Пути к файлам
    model_path = "models/optimized/rf_horizon_5_optimized.pkl"
    scaler_path = "models/optimized/scaler_horizon_5_optimized.pkl" 
    data_path = "data/processed/october_2025_horizon5.csv"
    
    # Проверяем наличие файлов
    files_to_check = [
        (model_path, "модель horizon_5"),
        (scaler_path, "скейлер horizon_5"),
        (data_path, "октябрьские данные 2025")
    ]
    
    for file_path, name in files_to_check:
        if not Path(file_path).exists():
            print(f"❌ Файл {name} не найден: {file_path}")
            return None
    
    print(f"✅ Все файлы найдены")
    
    # Загружаем данные
    print(f"\n📊 Загружаем октябрьские данные 2025...")
    df = pd.read_csv(data_path)
    print(f"   Записей: {len(df)}")
    print(f"   Период: {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}")
    
    # Загружаем модель и скейлер
    print(f"\n🧠 Загружаем модель horizon_5...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"   Модель: {type(model).__name__}")
    print(f"   Ожидает признаков: {model.n_features_in_}")
    
    # Подготавливаем признаки
    print(f"\n🔧 Подготавливаем признаки...")
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target_5']]
    
    print(f"   Доступно признаков: {len(feature_columns)}")
    
    if len(feature_columns) != model.n_features_in_:
        print(f"❌ Несоответствие количества признаков!")
        print(f"   Модель ожидает: {model.n_features_in_}")
        print(f"   Данные содержат: {len(feature_columns)}")
        return None
    
    # Подготавливаем данные для модели
    X = df[feature_columns].fillna(0)  # Заполняем NaN нулями
    
    # Масштабируем признаки
    print(f"🔀 Масштабируем признаки...")
    X_scaled = scaler.transform(X)
    
    # Получаем предсказания
    print(f"🔮 Генерируем предсказания...")
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Добавляем предсказания в данные
    df['prediction'] = predictions
    df['prob_up'] = probabilities[:, 1]  # Вероятность роста (класс 1)
    df['prob_down'] = probabilities[:, 0]  # Вероятность падения (класс 0)
    
    print(f"   Предсказаний сгенерировано: {len(predictions)}")
    
    # Статистика предсказаний
    pred_counts = pd.Series(predictions).value_counts()
    print(f"📊 Распределение предсказаний:")
    print(f"   BUY (1): {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(predictions)*100:.1f}%)")
    print(f"   SELL (0): {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(predictions)*100:.1f}%)")
    
    # Настройки бэктеста
    print(f"\n📈 Запуск бэктеста...")
    initial_balance = 1000.0
    fee = 0.001  # 0.1% комиссия на сделку
    buy_threshold = 0.55   # Покупаем если вероятность роста > 55%
    sell_threshold = 0.45  # Продаем если вероятность роста < 45%
    
    print(f"   Начальный капитал: ${initial_balance}")
    print(f"   Комиссия: {fee*100}%")
    print(f"   Пороги: BUY > {buy_threshold}, SELL < {sell_threshold}")
    
    # Переменные для бэктеста
    balance = initial_balance
    position = 0  # 0 - нет позиции, 1 - в лонге
    entry_price = 0
    trades = []
    equity_curve = []
    
    # Основной цикл бэктеста
    for i, row in df.iterrows():
        current_price = row['close']
        prob_up = row['prob_up']
        timestamp = row['timestamp']
        
        # Торговые сигналы
        buy_signal = prob_up > buy_threshold
        sell_signal = prob_up < sell_threshold
        
        # Логика входа в позицию
        if position == 0 and buy_signal:
            entry_price = current_price * (1 + fee)  # С учетом комиссии
            position = 1
            trades.append({
                'timestamp': timestamp,
                'action': 'BUY',
                'price': entry_price,
                'prob_up': prob_up,
                'balance': balance
            })
        
        # Логика выхода из позиции
        elif position == 1 and sell_signal:
            exit_price = current_price * (1 - fee)  # С учетом комиссии
            pnl_pct = (exit_price - entry_price) / entry_price
            balance = balance * (1 + pnl_pct)
            position = 0
            
            trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': exit_price,
                'prob_up': prob_up,
                'balance': balance,
                'pnl_pct': pnl_pct * 100,
                'entry_price': entry_price
            })
        
        # Текущая стоимость портфеля
        if position == 1:
            current_value = balance * (current_price / entry_price)
        else:
            current_value = balance
        
        equity_curve.append({
            'timestamp': timestamp,
            'balance': current_value,
            'price': current_price,
            'position': position,
            'prob_up': prob_up
        })
    
    # Принудительное закрытие позиции в конце
    if position == 1:
        final_price = df['close'].iloc[-1]
        exit_price = final_price * (1 - fee)
        pnl_pct = (exit_price - entry_price) / entry_price
        balance = balance * (1 + pnl_pct)
        
        trades.append({
            'timestamp': df['timestamp'].iloc[-1],
            'action': 'SELL_FINAL',
            'price': exit_price,
            'prob_up': df['prob_up'].iloc[-1],
            'balance': balance,
            'pnl_pct': pnl_pct * 100,
            'entry_price': entry_price
        })
    
    # Анализ результатов
    print(f"\n" + "="*60)
    print(f"📊 РЕЗУЛЬТАТЫ БЭКТЕСТА HORIZON_5 НА ОКТЯБРЕ 2025")
    print(f"="*60)
    
    final_balance = balance
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    print(f"\n💰 ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Начальный капитал: ${initial_balance:.2f}")
    print(f"   Финальный капитал: ${final_balance:.2f}")
    print(f"   Абсолютная прибыль: ${final_balance - initial_balance:.2f}")
    print(f"   Общая доходность: {total_return:.2f}%")
    
    # Статистика сделок
    if len(trades) > 0:
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] in ['SELL', 'SELL_FINAL']]
        
        print(f"\n🔄 СТАТИСТИКА СДЕЛОК:")
        print(f"   Всего сделок: {len(buy_trades)} входов, {len(sell_trades)} выходов")
        
        if len(sell_trades) > 0:
            pnls = [t['pnl_pct'] for t in sell_trades if 'pnl_pct' in t]
            if pnls:
                winning_trades = [p for p in pnls if p > 0]
                losing_trades = [p for p in pnls if p <= 0]
                
                print(f"   ✅ Прибыльных: {len(winning_trades)} ({len(winning_trades)/len(pnls)*100:.1f}%)")
                print(f"   ❌ Убыточных: {len(losing_trades)} ({len(losing_trades)/len(pnls)*100:.1f}%)")
                print(f"   📊 Средняя сделка: {np.mean(pnls):.2f}%")
                
                if winning_trades:
                    print(f"   📈 Средняя прибыльная: {np.mean(winning_trades):.2f}%")
                    print(f"   📈 Лучшая сделка: {np.max(winning_trades):.2f}%")
                    
                if losing_trades:
                    print(f"   📉 Средняя убыточная: {np.mean(losing_trades):.2f}%")
                    print(f"   📉 Худшая сделка: {np.min(losing_trades):.2f}%")
    
    # Buy & Hold сравнение
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    buy_hold_return = (end_price - start_price) / start_price * 100
    
    print(f"\n📈 СРАВНЕНИЕ С BUY & HOLD:")
    print(f"   B&H доходность: {buy_hold_return:.2f}%")
    print(f"   Превышение B&H: {total_return - buy_hold_return:.2f}%")
    
    # Точность модели
    if 'target_5' in df.columns:
        model_accuracy = (df['prediction'] == df['target_5']).mean() * 100
        print(f"\n🎯 ТОЧНОСТЬ МОДЕЛИ:")
        print(f"   Accuracy: {model_accuracy:.2f}%")
    
    # Сохранение результатов
    print(f"\n💾 Сохранение результатов...")
    results_dir = Path("results/october_2025_horizon5_backtest")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем сделки
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(results_dir / "trades.csv", index=False)
        print(f"   Сделки: {results_dir}/trades.csv")
    
    # Сохраняем кривую капитала
    equity_df = pd.DataFrame(equity_curve)
    equity_df.to_csv(results_dir / "equity_curve.csv", index=False)
    print(f"   Кривая капитала: {results_dir}/equity_curve.csv")
    
    # Сохраняем статистику
    stats = {
        'model': 'rf_horizon_5_optimized',
        'period': 'October 2025',
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_return_pct': total_return,
        'buy_hold_return_pct': buy_hold_return,
        'excess_return_pct': total_return - buy_hold_return,
        'total_trades': len(buy_trades) if 'buy_trades' in locals() else 0,
        'winning_trades': len(winning_trades) if 'winning_trades' in locals() else 0,
        'win_rate_pct': len(winning_trades)/len(pnls)*100 if 'winning_trades' in locals() and 'pnls' in locals() and pnls else 0,
        'avg_trade_pct': np.mean(pnls) if 'pnls' in locals() and pnls else 0,
        'model_accuracy_pct': model_accuracy if 'model_accuracy' in locals() else 0,
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'fee_pct': fee * 100
    }
    
    with open(results_dir / "backtest_stats.txt", 'w', encoding='utf-8') as f:
        f.write("BACKTEST RESULTS: HORIZON_5 MODEL ON OCTOBER 2025\\n")
        f.write("="*60 + "\\n\\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\\n")
    
    print(f"   Статистика: {results_dir}/backtest_stats.txt")
    
    print(f"\n🎉 Бэктест завершен!")
    print(f"📁 Результаты сохранены в: {results_dir}")
    
    return stats

if __name__ == "__main__":
    try:
        results = run_horizon5_backtest()
        if results:
            print(f"\n✅ Бэктест успешно выполнен")
        else:
            print(f"\n❌ Ошибка выполнения бэктеста")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()