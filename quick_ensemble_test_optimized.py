#!/usr/bin/env python3
"""
Быстрый тест оптимизированного ансамбля с малым объемом данных
"""

import sys
import os
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.models.optimized_ensemble_predictor import OptimizedEnsemblePredictor, ProgressBar

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 100) -> pd.DataFrame:
    """Создание небольшого набора тестовых данных."""
    print(f"📊 Создание {n_samples} строк тестовых данных...")
    
    np.random.seed(42)
    
    # Генерация базовых OHLCV данных
    dates = pd.date_range(start='2024-08-20', periods=n_samples, freq='5min')
    base_price = 150
    
    data = []
    current_price = base_price
    
    for i in range(n_samples):
        # Случайное изменение цены
        change = np.random.normal(0, 0.5)
        current_price += change
        
        high = current_price + abs(np.random.normal(0, 0.3))
        low = current_price - abs(np.random.normal(0, 0.3))
        volume = np.random.uniform(500000, 2000000)
        
        data.append({
            'timestamp': dates[i],
            'open': current_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # Добавляем все необходимые технические индикаторы
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
    df['BB_hband'] = df['MA20'] + (df['close'].rolling(20, min_periods=1).std() * 2)
    df['BB_lband'] = df['MA20'] - (df['close'].rolling(20, min_periods=1).std() * 2)
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
    df['close_to_high'] = (df['close'] - df['low']) / ((df['high'] - df['low']) + 1e-8)
    df['close_to_low'] = (df['high'] - df['close']) / ((df['high'] - df['low']) + 1e-8)
    
    # Timeframe
    df['timeframe_minutes'] = 5.0
    
    # Заполняем NaN
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    print(f"✅ Создано {len(df)} строк с полным набором признаков")
    return df

def run_quick_test():
    """Быстрый тест оптимизированного ансамбля."""
    print("🚀 БЫСТРЫЙ ТЕСТ ОПТИМИЗИРОВАННОГО АНСАМБЛЯ")
    print("="*60)
    
    total_start = time.time()
    
    # 1. Создание тестовых данных
    print("1️⃣ Создание тестовых данных...")
    data = create_test_data(100)  # Всего 100 строк
    
    # 2. Инициализация предсказателя
    print("2️⃣ Инициализация ML предсказателя...")
    init_start = time.time()
    predictor = OptimizedEnsemblePredictor(lazy_loading=True, show_progress=True)
    init_time = time.time() - init_start
    print(f"   Время инициализации: {init_time:.3f}s")
    
    # 3. Первое предсказание (запускает загрузку моделей)
    print("3️⃣ Первое предсказание (загрузка моделей)...")
    first_pred_start = time.time()
    
    test_row = data.iloc[-1:].copy()  # Последняя строка
    first_prediction = predictor.predict_ensemble(test_row)
    
    first_pred_time = time.time() - first_pred_start
    print(f"   Время первого предсказания: {first_pred_time:.2f}s")
    
    if first_prediction:
        signal = "BUY" if first_prediction['final_signal'] == 1 else "HOLD/SELL"
        prob = first_prediction['final_probability']
        strength = first_prediction['signal_strength']
        
        print(f"   📈 Результат: {signal} | Вероятность: {prob:.3f} | Сила: {strength}")
        
        base_preds = first_prediction.get('base_predictions', {})
        if base_preds:
            print("   🤖 Базовые модели:")
            for model, pred in base_preds.items():
                print(f"      {model}: {pred:.3f}")
    else:
        print("   ❌ Первое предсказание не удалось")
        return
    
    # 4. Массовое тестирование (небольшое)
    print("4️⃣ Массовое тестирование (20 предсказаний)...")
    mass_start = time.time()
    
    predictions = []
    prediction_times = []
    
    progress = ProgressBar(20, "🔄 Предсказания")
    
    for i in range(max(0, len(data)-20), len(data)):
        pred_start = time.time()
        
        row = data.iloc[i:i+1].copy()
        pred = predictor.predict_ensemble(row)
        
        pred_time = time.time() - pred_start
        prediction_times.append(pred_time)
        
        if pred:
            predictions.append({
                'index': i,
                'signal': pred['final_signal'],
                'probability': pred['final_probability'],
                'strength': pred['signal_strength']
            })
        
        progress.update(1)
    
    mass_time = time.time() - mass_start
    avg_pred_time = np.mean(prediction_times)
    
    print(f"   Время массового теста: {mass_time:.2f}s")
    print(f"   Среднее время предсказания: {avg_pred_time*1000:.1f}ms")
    print(f"   Успешных предсказаний: {len(predictions)}/20")
    
    # 5. Анализ результатов
    print("5️⃣ Анализ результатов...")
    
    if predictions:
        buy_signals = sum(1 for p in predictions if p['signal'] == 1)
        strong_signals = sum(1 for p in predictions if p['strength'] == 'STRONG')
        avg_probability = np.mean([p['probability'] for p in predictions])
        
        print(f"   BUY сигналов: {buy_signals}/20 ({buy_signals/20*100:.1f}%)")
        print(f"   Сильных сигналов: {strong_signals}/20 ({strong_signals/20*100:.1f}%)")
        print(f"   Средняя вероятность: {avg_probability:.3f}")
        
        # Показываем несколько примеров
        print("   📋 Примеры предсказаний:")
        for pred in predictions[-5:]:
            signal_text = "BUY" if pred['signal'] == 1 else "SELL"
            print(f"      [{pred['index']:3d}] {signal_text} | {pred['probability']:.3f} | {pred['strength']}")
    
    # 6. Финальная статистика
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   Общее время теста: {total_time:.2f}s")
    print(f"   Время инициализации: {init_time:.3f}s ({init_time/total_time*100:.1f}%)")
    print(f"   Время первого предсказания: {first_pred_time:.2f}s ({first_pred_time/total_time*100:.1f}%)")
    print(f"   Время массовых предсказаний: {mass_time:.2f}s ({mass_time/total_time*100:.1f}%)")
    
    # Получаем информацию о моделях
    if hasattr(predictor, 'get_models_info'):
        model_info = predictor.get_models_info()
        model_load_time = model_info.get('total_load_time', 0)
        print(f"   Время загрузки ML моделей: {model_load_time:.2f}s")
        
        load_times = model_info.get('load_times', {})
        if load_times:
            print("   ⚡ Время загрузки по моделям:")
            for model, load_time in load_times.items():
                print(f"      {model}: {load_time:.3f}s")
    
    print("="*60)
    
    # 7. Заключение
    if avg_pred_time < 0.1 and len(predictions) >= 18:  # 90% успешных предсказаний
        print("🎉 ТЕСТ ПРОЙДЕН УСПЕШНО!")
        print("✅ Модели загружаются и работают корректно")
        print("✅ Время предсказания приемлемо")
        print("✅ Ансамбль стабилен")
    elif len(predictions) >= 15:  # 75% успешных
        print("⚠️ ТЕСТ ПРОЙДЕН С ЗАМЕЧАНИЯМИ")
        if avg_pred_time >= 0.1:
            print("  ⚠️ Предсказания медленные")
        print("  ✅ Ансамбль в основном работает")
    else:
        print("❌ ТЕСТ НЕ ПРОЙДЕН")
        print("  ❌ Много ошибок в предсказаниях")
        print("  💡 Требуется дополнительная отладка")

def main():
    """Главная функция."""
    try:
        run_quick_test()
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()