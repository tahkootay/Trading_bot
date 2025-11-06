#!/usr/bin/env python3
"""
Бэктест регуляризованной модели horizon_5 на данных за август, сентябрь и октябрь 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def create_target_5(df):
    """Создает target_5 - цена через 5 периодов выше текущей?"""
    print("🎯 Создание target_5...")
    
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Target_5: 1 если цена через 5 периодов выше текущей, 0 иначе
    df['target_5'] = np.nan
    for i in range(len(df) - 5):
        current_price = df.loc[i, 'close']
        future_price = df.loc[i + 5, 'close']
        df.loc[i, 'target_5'] = 1 if future_price > current_price else 0
    
    # Удаляем последние 5 строк (нет future price)
    df = df.iloc[:-5].copy()
    
    target_counts = df['target_5'].value_counts()
    print(f"   Target_5 распределение:")
    print(f"     UP (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"     DOWN (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    
    return df

def prepare_features(df):
    """Подготавливает 25 фичей для модели horizon_5"""
    print("🔧 Подготовка 25 фичей...")
    
    # 25 фичей для модели horizon_5
    required_features = [
        'volume_sma_20', 'atr_14', 'volatility_ratio', 'bb_width', 'ema_100',
        'ema_diff_10_50', 'rsi_14', 'stoch_d', 'ema_50', 'stoch_k', 
        'bb_position', 'slope_ema_20', 'momentum_3', 'macd_line', 'bb_upper',
        'macd_histogram', 'relative_volume', 'momentum_10', 'volume_change', 'cci_20',
        'rsi_14_lag_1', 'rsi_14_lag_3', 'bb_position_lag_1', 'bb_position_lag_3', 'momentum_3_lag_1'
    ]
    
    # Проверяем наличие фичей
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        print(f"   ❌ Отсутствующие фичи: {missing_features}")
        return None, None
    
    print(f"   ✅ Все 25 фичей присутствуют")
    
    # Извлекаем фичи и target
    X = df[required_features]
    y = df['target_5']
    
    # Проверяем на NaN
    if X.isnull().any().any() or y.isnull().any():
        print("   ⚠️ Обнаружены NaN - удаляем строки с NaN")
        nan_mask = X.isnull().any(axis=1) | y.isnull()
        X = X[~nan_mask]
        y = y[~nan_mask]
        df_clean = df[~nan_mask].copy()
    else:
        df_clean = df.copy()
    
    print(f"   📊 Финальный размер: {len(X)} образцов, {len(X.columns)} фичей")
    
    return X, y, df_clean

def run_backtest(df, predictions, probabilities):
    """Запуск бэктеста с торговыми сигналами"""
    print("📈 Запуск бэктеста...")
    
    # Настройки бэктеста
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
    
    # Добавляем предсказания в данные
    df['prediction'] = predictions
    df['prob_up'] = probabilities[:, 1]  # Вероятность роста
    df['prob_down'] = probabilities[:, 0]  # Вероятность падения
    
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
    
    return balance, trades, equity_curve, df

def backtest_month(month_name, data_file, model_file, scaler_file):
    """Бэктест для конкретного месяца"""
    print(f"\n🗓️ БЭКТЕСТ ЗА {month_name.upper()}")
    print("=" * 50)
    
    # Проверяем наличие файлов
    files_to_check = [
        (data_file, f"данные за {month_name}"),
        (model_file, "модель"),
        (scaler_file, "scaler")
    ]
    
    for file_path, name in files_to_check:
        if not Path(file_path).exists():
            print(f"❌ Файл {name} не найден: {file_path}")
            return None
    
    print(f"✅ Все файлы найдены")
    
    # 1. Загружаем данные
    print(f"\n1️⃣ Загрузка данных за {month_name}...")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"   📊 Записей: {len(df):,}")
    print(f"   📅 Период: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    # 2. Создаем target_5
    df = create_target_5(df)
    
    # 3. Подготавливаем фичи
    X, y_true, df_clean = prepare_features(df)
    if X is None:
        return None
    
    # 4. Загружаем модель и scaler
    print(f"\n2️⃣ Загрузка модели...")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"   ✅ Модель загружена: {type(model).__name__}")
    
    # 5. Масштабирование и предсказания
    print(f"\n3️⃣ Генерация предсказаний...")
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Точность модели
    accuracy = accuracy_score(y_true, predictions)
    print(f"   🎯 Точность модели: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Распределение предсказаний
    pred_counts = pd.Series(predictions).value_counts()
    print(f"   📊 Распределение предсказаний:")
    print(f"      BUY (1): {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(predictions)*100:.1f}%)")
    print(f"      SELL (0): {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(predictions)*100:.1f}%)")
    
    # 6. Бэктест
    print(f"\n4️⃣ Бэктест...")
    final_balance, trades, equity_curve, df_with_signals = run_backtest(df_clean, predictions, probabilities)
    
    # 7. Анализ результатов
    print(f"\n📊 РЕЗУЛЬТАТЫ БЭКТЕСТА - {month_name.upper()}")
    print("=" * 40)
    
    initial_balance = 1000.0
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
    else:
        print(f"\n🔄 СТАТИСТИКА СДЕЛОК:")
        print(f"   ⚠️ Сделки не совершались (сигналы не сгенерированы)")
    
    # Buy & Hold сравнение
    start_price = df_clean['close'].iloc[0]
    end_price = df_clean['close'].iloc[-1]
    buy_hold_return = (end_price - start_price) / start_price * 100
    
    print(f"\n📈 СРАВНЕНИЕ С BUY & HOLD:")
    print(f"   B&H доходность: {buy_hold_return:.2f}%")
    print(f"   Превышение B&H: {total_return - buy_hold_return:.2f}%")
    
    print(f"\n🎯 ТОЧНОСТЬ МОДЕЛИ:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 8. Сохранение результатов
    print(f"\n💾 Сохранение результатов...")
    results_dir = Path(f"results/{month_name}_2025_regularized_horizon5")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем сделки
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(results_dir / "trades.csv", index=False)
        print(f"   ✅ Сделки: {results_dir}/trades.csv")
    
    # Сохраняем кривую капитала
    equity_df = pd.DataFrame(equity_curve)
    equity_df.to_csv(results_dir / "equity_curve.csv", index=False)
    print(f"   ✅ Кривая капитала: {results_dir}/equity_curve.csv")
    
    # Сохраняем предсказания
    predictions_df = df_with_signals[['timestamp', 'close', 'target_5', 'prediction', 'prob_up', 'prob_down']].copy()
    predictions_df.to_csv(results_dir / "predictions.csv", index=False)
    print(f"   ✅ Предсказания: {results_dir}/predictions.csv")
    
    # Сохраняем статистику
    stats = {
        'month': month_name,
        'model': 'rf_horizon5_regularized_conservative',
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_return_pct': total_return,
        'buy_hold_return_pct': buy_hold_return,
        'excess_return_pct': total_return - buy_hold_return,
        'total_trades': len(buy_trades) if 'buy_trades' in locals() else 0,
        'winning_trades': len(winning_trades) if 'winning_trades' in locals() else 0,
        'win_rate_pct': len(winning_trades)/len(pnls)*100 if 'winning_trades' in locals() and 'pnls' in locals() and pnls else 0,
        'avg_trade_pct': np.mean(pnls) if 'pnls' in locals() and pnls else 0,
        'model_accuracy_pct': accuracy * 100,
        'data_points': len(df_clean)
    }
    
    import json
    with open(results_dir / "backtest_stats.json", 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"   ✅ Статистика: {results_dir}/backtest_stats.json")
    print(f"📁 Результаты сохранены в: {results_dir}")
    
    return stats

def main():
    """Главная функция - запуск бэктестов для всех месяцев"""
    print("🚀 БЭКТЕСТЫ РЕГУЛЯРИЗОВАННОЙ МОДЕЛИ HORIZON_5")
    print("=" * 60)
    
    # Пути к модели
    model_file = "models/horizon5_regularized/rf_horizon5_regularized_conservative_20251103_173755.pkl"
    scaler_file = "models/horizon5_regularized/scaler_horizon5_regularized_conservative_20251103_173755.pkl"
    
    # Конфигурация месяцев для тестирования
    months_config = [
        {
            'name': 'august',
            'data_file': 'data/processed/test_aug_2025.csv'
        },
        {
            'name': 'september', 
            'data_file': 'data/processed/test_sep_2025.csv'
        },
        {
            'name': 'october',
            'data_file': 'data/processed/test_oct_2025.csv'
        }
    ]
    
    # Запускаем бэктесты
    all_results = []
    
    for month_config in months_config:
        try:
            result = backtest_month(
                month_config['name'],
                month_config['data_file'], 
                model_file,
                scaler_file
            )
            
            if result:
                all_results.append(result)
            else:
                print(f"❌ Ошибка в бэктесте для {month_config['name']}")
                
        except Exception as e:
            print(f"❌ Ошибка в {month_config['name']}: {e}")
            continue
    
    # Сводный отчет
    if all_results:
        print(f"\n📋 СВОДНЫЙ ОТЧЕТ ПО ВСЕМ МЕСЯЦАМ")
        print("=" * 60)
        
        total_return_sum = sum(r['total_return_pct'] for r in all_results)
        avg_accuracy = sum(r['model_accuracy_pct'] for r in all_results) / len(all_results)
        total_trades = sum(r['total_trades'] for r in all_results)
        
        print(f"🎯 Общие результаты:")
        print(f"   Месяцев протестировано: {len(all_results)}")
        print(f"   Суммарная доходность: {total_return_sum:.2f}%")
        print(f"   Средняя точность модели: {avg_accuracy:.2f}%")
        print(f"   Общее количество сделок: {total_trades}")
        
        print(f"\n📊 Результаты по месяцам:")
        for result in all_results:
            print(f"   {result['month'].title():10s}: {result['total_return_pct']:6.2f}% | Accuracy: {result['model_accuracy_pct']:5.2f}% | Сделок: {result['total_trades']}")
    
    print(f"\n✅ Все бэктесты завершены!")

if __name__ == "__main__":
    main()