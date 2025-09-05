#!/usr/bin/env python3
"""
Финальный скрипт для стабильного тестирования бота с оптимизированным ансамблем
"""

import sys
import os
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.models.optimized_ensemble_predictor import OptimizedEnsemblePredictor

def run_production_test(sample_size: int = 200, show_progress: bool = True):
    """
    Продакшн тест стабильного бота с ансамблем.
    
    Args:
        sample_size: Количество точек данных для тестирования
        show_progress: Показывать прогресс-бары
    """
    print("🚀 ПРОДАКШН ТЕСТ СТАБИЛЬНОГО БОТА С АНСАМБЛЕМ")
    print("="*65)
    print(f"📊 Размер выборки: {sample_size} точек")
    print(f"📋 Прогресс-бары: {'Вкл' if show_progress else 'Выкл'}")
    print("="*65)
    
    total_start = time.time()
    
    # 1. Попытка загрузки реальных данных
    print("1️⃣ Поиск и загрузка данных...")
    data = load_real_data_with_fallback(sample_size)
    
    if data is None:
        print("⚠️ Реальные данные недоступны, создаем синтетические...")
        data = create_production_data(sample_size)
    
    print(f"   ✅ Загружено: {len(data)} строк данных")
    print(f"   📅 Период: {data['timestamp'].min()} - {data['timestamp'].max()}")
    
    # 2. Инициализация предсказателя
    print("2️⃣ Инициализация ML предсказателя...")
    init_start = time.time()
    
    predictor = OptimizedEnsemblePredictor(
        lazy_loading=True,  # Отложенная загрузка для быстрого старта
        show_progress=show_progress
    )
    
    init_time = time.time() - init_start
    print(f"   ⚡ Инициализация: {init_time:.3f}s")
    
    # 3. Первое предсказание (запускает загрузку)
    print("3️⃣ Первое предсказание (автозагрузка моделей)...")
    first_pred_start = time.time()
    
    test_row = data.iloc[-1:].copy()
    first_prediction = predictor.predict_ensemble(test_row)
    
    first_pred_time = time.time() - first_pred_start
    
    if first_prediction:
        print(f"   ⏱️ Время первого предсказания: {first_pred_time:.2f}s")
        
        # Получаем информацию о загрузке моделей
        model_info = predictor.get_models_info()
        model_load_time = model_info.get('total_load_time', 0)
        actual_pred_time = first_pred_time - model_load_time
        
        print(f"   🤖 Загрузка моделей: {model_load_time:.2f}s")
        print(f"   🧠 Чистое предсказание: {actual_pred_time:.3f}s")
        
        signal = "🟢 BUY" if first_prediction['final_signal'] == 1 else "🔴 SELL"
        prob = first_prediction['final_probability']
        strength = first_prediction['signal_strength']
        
        print(f"   📈 Сигнал: {signal} | Вероятность: {prob:.3f} | Сила: {strength}")
    else:
        print("   ❌ Первое предсказание неудачно!")
        return False
    
    # 4. Тест скорости (без загрузки моделей)
    print("4️⃣ Тест скорости предсказаний...")
    speed_test_size = min(50, len(data))
    
    prediction_times = []
    successful_predictions = 0
    
    for i in range(max(0, len(data) - speed_test_size), len(data)):
        pred_start = time.time()
        
        row = data.iloc[i:i+1].copy()
        pred = predictor.predict_ensemble(row)
        
        pred_time = time.time() - pred_start
        prediction_times.append(pred_time)
        
        if pred:
            successful_predictions += 1
    
    avg_speed = np.mean(prediction_times) * 1000  # миллисекунды
    success_rate = successful_predictions / speed_test_size * 100
    
    print(f"   ⚡ Средняя скорость: {avg_speed:.1f}ms")
    print(f"   ✅ Успешность: {success_rate:.1f}% ({successful_predictions}/{speed_test_size})")
    
    # 5. Проверка стабильности
    print("5️⃣ Проверка стабильности сигналов...")
    
    # Берем несколько случайных точек для проверки консистентности
    test_indices = np.random.choice(range(20, len(data)-20), size=10, replace=False)
    stability_results = []
    
    for idx in test_indices:
        row = data.iloc[idx:idx+1].copy()
        
        # Делаем 3 предсказания на одних и тех же данных
        preds = []
        for _ in range(3):
            pred = predictor.predict_ensemble(row)
            if pred:
                preds.append({
                    'signal': pred['final_signal'],
                    'probability': pred['final_probability']
                })
        
        if len(preds) == 3:
            # Проверяем консистентность
            signals = [p['signal'] for p in preds]
            probs = [p['probability'] for p in preds]
            
            signal_stable = len(set(signals)) == 1  # Все сигналы одинаковы
            prob_std = np.std(probs)  # Стандартное отклонение вероятностей
            
            stability_results.append({
                'index': idx,
                'signal_stable': signal_stable,
                'prob_std': prob_std,
                'avg_prob': np.mean(probs)
            })
    
    if stability_results:
        signal_stability = np.mean([r['signal_stable'] for r in stability_results]) * 100
        avg_prob_std = np.mean([r['prob_std'] for r in stability_results])
        
        print(f"   🎯 Стабильность сигналов: {signal_stability:.1f}%")
        print(f"   📊 Вариация вероятностей: ±{avg_prob_std:.3f}")
    
    # 6. Мини-бэктест
    print("6️⃣ Мини-бэктест (простая стратегия)...")
    
    capital = 10000
    position = 0
    position_price = 0
    trades = []
    
    # Используем последние 100 точек для мини-бэктеста
    backtest_data = data.tail(100).copy()
    
    for i, (_, row) in enumerate(backtest_data.iterrows()):
        current_price = row['close']
        
        pred = predictor.predict_ensemble(pd.DataFrame([row]))
        
        if not pred:
            continue
        
        signal = pred['final_signal']
        probability = pred['final_probability']
        
        # Простая логика торговли
        if position == 0 and signal == 1 and probability > 0.7:
            # Покупаем
            position = (capital * 0.1) / current_price  # 10% капитала
            position_price = current_price
            
        elif position > 0 and (signal == 0 or probability < 0.3):
            # Продаем
            pnl = (current_price - position_price) * position
            capital += pnl
            
            trades.append({
                'entry': position_price,
                'exit': current_price,
                'pnl': pnl,
                'return_pct': (current_price - position_price) / position_price * 100
            })
            
            position = 0
            position_price = 0
    
    if trades:
        total_return = (capital - 10000) / 10000 * 100
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100
        avg_return = np.mean([t['return_pct'] for t in trades])
        
        print(f"   💰 Сделок: {len(trades)}")
        print(f"   📈 Общая доходность: {total_return:+.2f}%")
        print(f"   🎯 Win Rate: {win_rate:.1f}%")
        print(f"   📊 Средняя доходность сделки: {avg_return:+.2f}%")
    else:
        print("   📝 Сделок не совершено (строгие условия)")
    
    # 7. Финальная оценка
    total_time = time.time() - total_start
    
    print("\n" + "="*65)
    print("📋 ИТОГОВАЯ ОЦЕНКА СТАБИЛЬНОСТИ:")
    print("="*65)
    print(f"⏱️ Общее время теста: {total_time:.2f}s")
    print(f"🚀 Время загрузки моделей: {model_load_time:.2f}s")
    print(f"⚡ Средняя скорость предсказания: {avg_speed:.1f}ms")
    print(f"✅ Успешность предсказаний: {success_rate:.1f}%")
    
    if stability_results:
        print(f"🎯 Стабильность сигналов: {signal_stability:.1f}%")
    
    # Критерии для оценки готовности к продакшн
    production_ready = (
        model_load_time < 10.0 and          # Загрузка < 10 сек
        avg_speed < 100.0 and               # Предсказание < 100ms
        success_rate > 95.0 and             # Успешность > 95%
        (not stability_results or signal_stability > 80.0)  # Стабильность > 80%
    )
    
    print("\n" + "="*65)
    if production_ready:
        print("🎉 БОТ ГОТОВ К ПРОДАКШН ИСПОЛЬЗОВАНИЮ!")
        print("✅ Все критерии стабильности выполнены")
        print("✅ Ансамбль работает корректно и быстро")
        print("✅ Предсказания стабильны и надежны")
    else:
        print("⚠️ БОТ ТРЕБУЕТ ДОПОЛНИТЕЛЬНОЙ ОПТИМИЗАЦИИ")
        
        if model_load_time >= 10.0:
            print("❌ Слишком медленная загрузка моделей")
        if avg_speed >= 100.0:
            print("❌ Слишком медленные предсказания")
        if success_rate <= 95.0:
            print("❌ Низкая успешность предсказаний")
        if stability_results and signal_stability <= 80.0:
            print("❌ Нестабильные сигналы")
    
    print("="*65)
    return production_ready


def load_real_data_with_fallback(sample_size: int) -> pd.DataFrame:
    """Попытка загрузки реальных данных с несколькими вариантами."""
    possible_files = [
        "data/bybit_futures_solusdt_5m.csv",
        "data/SOLUSDT_5m_real_*.csv",
        "data/SOLUSDT_5m.csv"
    ]
    
    for file_pattern in possible_files:
        try:
            if '*' in file_pattern:
                from glob import glob
                files = glob(file_pattern)
                if files:
                    data_file = max(files, key=lambda x: Path(x).stat().st_mtime)
                else:
                    continue
            else:
                data_file = file_pattern
                if not Path(data_file).exists():
                    continue
            
            data = pd.read_csv(data_file)
            
            if 'timestamp' not in data.columns:
                data['timestamp'] = pd.date_range(start='2024-08-01', periods=len(data), freq='5min')
            else:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Берем последние sample_size строк
            if len(data) > sample_size:
                data = data.tail(sample_size).copy()
            
            return prepare_data_features(data)
            
        except Exception as e:
            continue
    
    return None


def create_production_data(sample_size: int) -> pd.DataFrame:
    """Создание продакшн-качества синтетических данных."""
    np.random.seed(42)
    
    # Генерация реалистичных OHLCV данных для SOL/USDT
    dates = pd.date_range(start='2024-08-20', periods=sample_size, freq='5min')
    
    data = []
    current_price = 150.0  # Начальная цена SOL
    
    for i in range(sample_size):
        # Реалистичные изменения цены (волатильность SOL)
        change_pct = np.random.normal(0, 0.015)  # 1.5% стандартное отклонение
        current_price *= (1 + change_pct)
        
        # OHLC с реалистичными спредами
        high = current_price * (1 + abs(np.random.normal(0, 0.005)))
        low = current_price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = current_price * (1 + np.random.normal(0, 0.002))
        
        # Объем коррелирует с волатильностью
        volatility = abs(change_pct)
        base_volume = 1000000
        volume = base_volume * (1 + volatility * 10) * np.random.uniform(0.5, 2.0)
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    return prepare_data_features(df)


def prepare_data_features(df: pd.DataFrame) -> pd.DataFrame:
    """Подготовка всех технических индикаторов."""
    # MA
    df['MA5'] = df['close'].rolling(5, min_periods=1).mean()
    df['MA10'] = df['close'].rolling(10, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(20, min_periods=1).mean()
    
    # EMA
    df['EMA12'] = df['close'].ewm(span=12, min_periods=1).mean()
    df['EMA26'] = df['close'].ewm(span=26, min_periods=1).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    bb_std = df['close'].rolling(20, min_periods=1).std()
    df['BB_hband'] = df['MA20'] + (bb_std * 2)
    df['BB_lband'] = df['MA20'] - (bb_std * 2)
    df['BB_width'] = df['BB_hband'] - df['BB_lband']
    df['BB_position'] = (df['close'] - df['BB_lband']) / (df['BB_width'] + 1e-8)
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
    df['vol_change'] = df['volume'].pct_change().fillna(0)
    
    # Price indicators
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['price_change_5'] = df['close'].pct_change(5).fillna(0)
    df['volatility'] = df['close'].rolling(14, min_periods=1).std() / (df['close'].rolling(14, min_periods=1).mean() + 1e-8)
    
    # Price position indicators
    df['high_low_pct'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    high_low_range = df['high'] - df['low'] + 1e-8
    df['close_to_high'] = (df['close'] - df['low']) / high_low_range
    df['close_to_low'] = (df['high'] - df['close']) / high_low_range
    
    # Timeframe
    df['timeframe_minutes'] = 5.0
    
    # Заполняем NaN
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df


def main():
    """Главная функция с аргументами командной строки."""
    parser = argparse.ArgumentParser(description='Production stability test for trading bot ensemble')
    parser.add_argument('--samples', type=int, default=200, help='Number of data samples to test')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bars')
    
    args = parser.parse_args()
    
    try:
        success = run_production_test(
            sample_size=args.samples,
            show_progress=not args.no_progress
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Тест прерван пользователем")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()