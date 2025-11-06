#!/usr/bin/env python3
"""
Бэктест модели horizon_5 на полном датасете 2024 года.
Тестирование на 104,917 записях с топ-25 фичами.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import time

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_2024_full_backtest():
    """Запуск бэктеста модели horizon_5 на полном датасете 2024 года."""
    
    print("🚀 БЭКТЕСТ HORIZON_5 НА ПОЛНОМ ДАТАСЕТЕ 2024 ГОДА")
    print("=" * 65)
    
    # Пути к файлам
    model_path = "models/optimized/rf_horizon_5_optimized.pkl"
    scaler_path = "models/optimized/scaler_horizon_5_optimized.pkl" 
    data_path = "data/processed/SOLUSDT_5m_2024_top25_ready.csv"
    
    # Проверяем наличие файлов
    files_to_check = [
        (model_path, "модель horizon_5"),
        (scaler_path, "скейлер horizon_5"),
        (data_path, "датасет 2024 года")
    ]
    
    for file_path, name in files_to_check:
        if not Path(file_path).exists():
            print(f"❌ Файл {name} не найден: {file_path}")
            return None
    
    print(f"✅ Все файлы найдены")
    
    # Загружаем данные
    print(f"\n📊 Загружаем полный датасет 2024 года...")
    start_time = time.time()
    df = pd.read_csv(data_path)
    load_time = time.time() - start_time
    
    print(f"   Записей: {len(df):,}")
    print(f"   Период: {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}")
    print(f"   Время загрузки: {load_time:.1f}s")
    
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
        return None
    
    # Подготавливаем данные для модели
    print(f"🔀 Подготовка данных для модели...")
    X = df[feature_columns].fillna(0)
    
    # Масштабируем признаки (по батчам для экономии памяти)
    print(f"🔀 Масштабируем признаки...")
    batch_size = 10000
    predictions = []
    probabilities = []
    
    start_time = time.time()
    
    for i in range(0, len(X), batch_size):
        end_idx = min(i + batch_size, len(X))
        X_batch = X.iloc[i:end_idx]
        
        # Масштабируем батч
        X_scaled_batch = scaler.transform(X_batch)
        
        # Получаем предсказания
        pred_batch = model.predict(X_scaled_batch)
        prob_batch = model.predict_proba(X_scaled_batch)
        
        predictions.extend(pred_batch)
        probabilities.extend(prob_batch[:, 1])  # Вероятность роста
        
        # Прогресс
        progress = (end_idx / len(X)) * 100
        if i % (batch_size * 5) == 0:  # Показываем каждые 50k записей
            print(f"   Прогресс: {progress:.1f}% ({end_idx:,}/{len(X):,})")
    
    processing_time = time.time() - start_time
    print(f"   Время обработки: {processing_time:.1f}s")
    print(f"   Скорость: {len(X)/processing_time:.0f} записей/сек")
    
    # Добавляем предсказания в данные
    df['prediction'] = predictions
    df['prob_up'] = probabilities
    df['prob_down'] = 1 - np.array(probabilities)
    
    print(f"🔮 Предсказания сгенерированы: {len(predictions):,}")
    
    # Статистика предсказаний
    pred_counts = pd.Series(predictions).value_counts()
    print(f"📊 Распределение предсказаний:")
    print(f"   BUY (1): {pred_counts.get(1, 0):,} ({pred_counts.get(1, 0)/len(predictions)*100:.1f}%)")
    print(f"   SELL (0): {pred_counts.get(0, 0):,} ({pred_counts.get(0, 0)/len(predictions)*100:.1f}%)")
    
    # Настройки бэктеста
    print(f"\n📈 Запуск бэктеста на полном году...")
    initial_balance = 10000.0  # Увеличиваем стартовый капитал для годового теста
    fee = 0.001  # 0.1% комиссия на сделку
    buy_threshold = 0.55   # Покупаем если вероятность роста > 55%
    sell_threshold = 0.45  # Продаем если вероятность роста < 45%
    
    print(f"   Начальный капитал: ${initial_balance:,.0f}")
    print(f"   Комиссия: {fee*100}%")
    print(f"   Пороги: BUY > {buy_threshold}, SELL < {sell_threshold}")
    
    # Переменные для бэктеста
    balance = initial_balance
    position = 0  # 0 - нет позиции, 1 - в лонге
    entry_price = 0
    trades = []
    
    # Статистика для отслеживания прогресса
    total_records = len(df)
    progress_step = total_records // 20  # 20 отчетов о прогрессе
    
    print(f"   Обрабатываем {total_records:,} записей...")
    start_time = time.time()
    
    # Основной цикл бэктеста
    for i, (idx, row) in enumerate(df.iterrows()):
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
        
        # Отчет о прогрессе
        if i % progress_step == 0 and i > 0:
            progress = (i / total_records) * 100
            current_time = time.time()
            elapsed = current_time - start_time
            estimated_total = elapsed * (total_records / i)
            remaining = estimated_total - elapsed
            
            # Текущая стоимость портфеля
            if position == 1:
                current_value = balance * (current_price / entry_price)
            else:
                current_value = balance
            
            current_return = (current_value - initial_balance) / initial_balance * 100
            
            print(f"   📊 {progress:.0f}% | Доходность: {current_return:+.1f}% | "
                  f"Осталось: {remaining/60:.1f}мин | Сделок: {len([t for t in trades if t['action']=='BUY'])}")
    
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
    
    backtest_time = time.time() - start_time
    print(f"   Время бэктеста: {backtest_time:.1f}s ({total_records/backtest_time:.0f} записей/сек)")
    
    # Анализ результатов
    print(f"\n" + "="*65)
    print(f"📊 РЕЗУЛЬТАТЫ ГОДОВОГО БЭКТЕСТА HORIZON_5 НА 2024")
    print(f"="*65)
    
    final_balance = balance
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    print(f"\n💰 ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Начальный капитал: ${initial_balance:,.0f}")
    print(f"   Финальный капитал: ${final_balance:,.0f}")
    print(f"   Абсолютная прибыль: ${final_balance - initial_balance:,.0f}")
    print(f"   Общая доходность: {total_return:.2f}%")
    print(f"   Среднемесячная доходность: {total_return/12:.2f}%")
    
    # Статистика сделок
    if len(trades) > 0:
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] in ['SELL', 'SELL_FINAL']]
        
        print(f"\n🔄 СТАТИСТИКА СДЕЛОК:")
        print(f"   Всего сделок: {len(buy_trades):,} входов, {len(sell_trades):,} выходов")
        print(f"   Среднее сделок в день: {len(buy_trades)/(365):.1f}")
        print(f"   Среднее сделок в месяц: {len(buy_trades)*12/365:.0f}")
        
        if len(sell_trades) > 0:
            pnls = [t['pnl_pct'] for t in sell_trades if 'pnl_pct' in t]
            if pnls:
                winning_trades = [p for p in pnls if p > 0]
                losing_trades = [p for p in pnls if p <= 0]
                
                print(f"   ✅ Прибыльных: {len(winning_trades):,} ({len(winning_trades)/len(pnls)*100:.1f}%)")
                print(f"   ❌ Убыточных: {len(losing_trades):,} ({len(losing_trades)/len(pnls)*100:.1f}%)")
                print(f"   📊 Средняя сделка: {np.mean(pnls):+.3f}%")
                
                if winning_trades:
                    print(f"   📈 Средняя прибыльная: {np.mean(winning_trades):+.2f}%")
                    print(f"   📈 Лучшая сделка: {np.max(winning_trades):+.2f}%")
                    
                if losing_trades:
                    print(f"   📉 Средняя убыточная: {np.mean(losing_trades):+.2f}%")
                    print(f"   📉 Худшая сделка: {np.min(losing_trades):+.2f}%")
                
                # Фактор прибыли
                total_profit = sum([p for p in pnls if p > 0])
                total_loss = abs(sum([p for p in pnls if p <= 0]))
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                print(f"   💎 Фактор прибыли: {profit_factor:.2f}")
    
    # Buy & Hold сравнение
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    buy_hold_return = (end_price - start_price) / start_price * 100
    
    print(f"\n📈 СРАВНЕНИЕ С BUY & HOLD:")
    print(f"   B&H доходность: {buy_hold_return:.2f}%")
    print(f"   Превышение B&H: {total_return - buy_hold_return:+.2f}%")
    print(f"   Превышение в разах: {total_return / buy_hold_return:.1f}x")
    
    # Точность модели
    if 'target_5' in df.columns:
        model_accuracy = (df['prediction'] == df['target_5']).mean() * 100
        print(f"\n🎯 ТОЧНОСТЬ МОДЕЛИ НА 2024:")
        print(f"   Accuracy: {model_accuracy:.2f}%")
    
    # Статистика по месяцам
    print(f"\n📅 ПОМЕСЯЧНАЯ СТАТИСТИКА:")
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    monthly_trades = {}
    
    for month in range(1, 13):
        month_trades = [t for t in buy_trades 
                       if pd.to_datetime(t['timestamp']).month == month]
        monthly_trades[month] = len(month_trades)
    
    month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                   'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    
    for month, name in enumerate(month_names, 1):
        trades_count = monthly_trades.get(month, 0)
        print(f"   {name}: {trades_count:3d} сделок")
    
    # Сохранение результатов
    print(f"\n💾 Сохранение результатов...")
    results_dir = Path("results/2024_full_year_horizon5_backtest")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем сделки (только если их не слишком много)
    if len(trades) < 50000:  # Ограничение для больших файлов
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(results_dir / "trades.csv", index=False)
        print(f"   Сделки: {results_dir}/trades.csv")
    else:
        print(f"   ⚠️ Слишком много сделок ({len(trades)}), файл не сохранен")
    
    # Сохраняем статистику
    stats = {
        'model': 'rf_horizon_5_optimized',
        'period': 'Full Year 2024',
        'data_points': len(df),
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_return_pct': total_return,
        'monthly_return_pct': total_return/12,
        'buy_hold_return_pct': buy_hold_return,
        'excess_return_pct': total_return - buy_hold_return,
        'total_trades': len(buy_trades),
        'avg_trades_per_day': len(buy_trades)/365,
        'winning_trades': len(winning_trades) if 'winning_trades' in locals() else 0,
        'win_rate_pct': len(winning_trades)/len(pnls)*100 if 'winning_trades' in locals() and 'pnls' in locals() and pnls else 0,
        'avg_trade_pct': np.mean(pnls) if 'pnls' in locals() and pnls else 0,
        'profit_factor': profit_factor if 'profit_factor' in locals() else 0,
        'model_accuracy_pct': model_accuracy if 'model_accuracy' in locals() else 0,
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'fee_pct': fee * 100,
        'processing_time_sec': processing_time,
        'backtest_time_sec': backtest_time
    }
    
    with open(results_dir / "annual_backtest_stats.txt", 'w', encoding='utf-8') as f:
        f.write("ANNUAL BACKTEST RESULTS: HORIZON_5 MODEL ON 2024\\n")
        f.write("="*65 + "\\n\\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\\n")
    
    print(f"   Статистика: {results_dir}/annual_backtest_stats.txt")
    
    print(f"\n🎉 Годовой бэктест завершен!")
    print(f"📁 Результаты сохранены в: {results_dir}")
    print(f"⏱️ Общее время выполнения: {(processing_time + backtest_time)/60:.1f} минут")
    
    return stats

if __name__ == "__main__":
    try:
        print("🚀 Начинаем полный годовой бэктест 2024...")
        results = run_2024_full_backtest()
        if results:
            print(f"\n✅ Годовой бэктест успешно завершен!")
            print(f"📊 Итоговая доходность: {results['total_return_pct']:.2f}%")
        else:
            print(f"\n❌ Ошибка выполнения бэктеста")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()