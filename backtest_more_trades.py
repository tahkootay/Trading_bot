#!/usr/bin/env python3
"""
Бэктест с более агрессивными порогами для увеличения количества сделок
Тестируем разные пороги: 52.5%/47.5%, 52%/48%, 51%/49%
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

def run_backtest(df, predictions, probabilities, buy_threshold=0.55, sell_threshold=0.45):
    """Запуск бэктеста с заданными порогами"""
    print(f"📈 Запуск бэктеста (BUY>{buy_threshold}, SELL<{sell_threshold})...")
    
    # Настройки бэктеста
    initial_balance = 1000.0
    fee = 0.001  # 0.1% комиссия на сделку
    
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

def test_thresholds(df_clean, predictions, probabilities, accuracy):
    """Тестирует разные пороги и возвращает результаты"""
    print(f"\n🧪 ТЕСТИРОВАНИЕ РАЗНЫХ ПОРОГОВ")
    print("=" * 50)
    
    # Конфигурации порогов для тестирования
    threshold_configs = [
        {'buy': 0.55, 'sell': 0.45, 'name': 'Консервативные (55%/45%)'},
        {'buy': 0.525, 'sell': 0.475, 'name': 'Умеренные (52.5%/47.5%)'},
        {'buy': 0.52, 'sell': 0.48, 'name': 'Активные (52%/48%)'},
        {'buy': 0.51, 'sell': 0.49, 'name': 'Агрессивные (51%/49%)'},
    ]
    
    initial_balance = 1000.0
    results = []
    
    for config in threshold_configs:
        print(f"\n📊 {config['name']}")
        print("-" * 30)
        
        # Запускаем бэктест с данными порогами
        final_balance, trades, equity_curve, df_with_signals = run_backtest(
            df_clean.copy(), 
            predictions, 
            probabilities, 
            config['buy'], 
            config['sell']
        )
        
        # Подсчитываем статистику
        total_return = (final_balance - initial_balance) / initial_balance * 100
        
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] in ['SELL', 'SELL_FINAL']]
        
        win_rate = 0
        avg_trade = 0
        if sell_trades:
            pnls = [t['pnl_pct'] for t in sell_trades if 'pnl_pct' in t]
            if pnls:
                winning_trades = [p for p in pnls if p > 0]
                win_rate = len(winning_trades) / len(pnls) * 100
                avg_trade = np.mean(pnls)
        
        # Buy & Hold сравнение
        start_price = df_clean['close'].iloc[0]
        end_price = df_clean['close'].iloc[-1]
        buy_hold_return = (end_price - start_price) / start_price * 100
        
        result = {
            'name': config['name'],
            'buy_threshold': config['buy'],
            'sell_threshold': config['sell'],
            'total_return_pct': total_return,
            'buy_hold_return_pct': buy_hold_return,
            'excess_return_pct': total_return - buy_hold_return,
            'total_trades': len(buy_trades),
            'win_rate_pct': win_rate,
            'avg_trade_pct': avg_trade,
            'final_balance': final_balance,
            'trades': trades
        }
        
        results.append(result)
        
        # Выводим результаты
        print(f"   💰 Доходность: {total_return:.2f}% (vs B&H: {buy_hold_return:.2f}%)")
        print(f"   🔄 Сделок: {len(buy_trades)} входов")
        print(f"   ✅ Win rate: {win_rate:.1f}%")
        print(f"   📊 Средняя сделка: {avg_trade:.2f}%")
        print(f"   💵 Финальный баланс: ${final_balance:.2f}")
    
    return results

def backtest_month_with_thresholds(month_name, data_file, model_file, scaler_file):
    """Бэктест для конкретного месяца с разными порогами"""
    print(f"\n🗓️ ТЕСТИРОВАНИЕ ПОРОГОВ ДЛЯ {month_name.upper()}")
    print("=" * 60)
    
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
    
    # 6. Тестирование порогов
    results = test_thresholds(df_clean, predictions, probabilities, accuracy)
    
    # 7. Сводная таблица результатов
    print(f"\n📊 СВОДНАЯ ТАБЛИЦА - {month_name.upper()}")
    print("=" * 90)
    print(f"{'Пороги':<25} {'Доходность':<12} {'Сделок':<8} {'Win Rate':<10} {'Ср.сделка':<12} {'vs B&H':<10}")
    print("-" * 90)
    
    for result in results:
        print(f"{result['name']:<25} {result['total_return_pct']:>8.2f}% {result['total_trades']:>6} {result['win_rate_pct']:>8.1f}% {result['avg_trade_pct']:>9.2f}% {result['excess_return_pct']:>8.2f}%")
    
    return results

def main():
    """Главная функция - тестирование порогов для всех месяцев"""
    print("🚀 ТЕСТИРОВАНИЕ ПОРОГОВ ДЛЯ УВЕЛИЧЕНИЯ СДЕЛОК")
    print("=" * 70)
    
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
    
    # Запускаем тестирование для всех месяцев
    all_month_results = {}
    
    for month_config in months_config:
        try:
            month_results = backtest_month_with_thresholds(
                month_config['name'],
                month_config['data_file'], 
                model_file,
                scaler_file
            )
            
            if month_results:
                all_month_results[month_config['name']] = month_results
            else:
                print(f"❌ Ошибка в тестировании для {month_config['name']}")
                
        except Exception as e:
            print(f"❌ Ошибка в {month_config['name']}: {e}")
            continue
    
    # Финальный сводный отчет
    if all_month_results:
        print(f"\n📋 ФИНАЛЬНЫЙ СВОДНЫЙ ОТЧЕТ")
        print("=" * 100)
        
        # Группируем результаты по порогам
        threshold_names = ['Консервативные (55%/45%)', 'Умеренные (52.5%/47.5%)', 'Активные (52%/48%)', 'Агрессивные (51%/49%)']
        
        print(f"{'Пороги':<25} {'Общ.доходность':<15} {'Общ.сделок':<12} {'Ср.Win Rate':<12} {'Ср.сделка':<12}")
        print("-" * 100)
        
        for i, threshold_name in enumerate(threshold_names):
            total_return = 0
            total_trades = 0
            avg_win_rate = 0
            avg_trade = 0
            month_count = 0
            
            for month_name, month_results in all_month_results.items():
                if i < len(month_results):
                    result = month_results[i]
                    total_return += result['total_return_pct']
                    total_trades += result['total_trades']
                    avg_win_rate += result['win_rate_pct']
                    avg_trade += result['avg_trade_pct']
                    month_count += 1
            
            if month_count > 0:
                avg_win_rate /= month_count
                avg_trade /= month_count
                
            print(f"{threshold_name:<25} {total_return:>11.2f}% {total_trades:>9} {avg_win_rate:>9.1f}% {avg_trade:>9.2f}%")
        
        print(f"\n🎯 РЕКОМЕНДАЦИИ:")
        print("Выберите пороги исходя из ваших предпочтений:")
        print("• Консервативные: меньше сделок, но потенциально выше качество")
        print("• Агрессивные: больше сделок, но может быть больше ложных сигналов")
        
    print(f"\n✅ Тестирование порогов завершено!")

if __name__ == "__main__":
    main()