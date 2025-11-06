#!/usr/bin/env python3
"""
Подготовка датасета 2024 года с топ-25 фичами для тестирования модели horizon_5.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def prepare_2024_top25_dataset():
    """Подготовка датасета 2024 года с топ-25 фичами."""
    
    print("🗓️ ПОДГОТОВКА ДАТАСЕТА 2024 ГОДА С ТОП-25 ФИЧАМИ")
    print("="*60)
    
    # Загружаем данные 2024 года с индикаторами
    input_file = "data/raw/SOLUSDT_5m_2024_advanced_indicators.csv"
    
    if not Path(input_file).exists():
        print(f"❌ Файл не найден: {input_file}")
        return None
    
    print(f"📊 Загружаем данные 2024 года...")
    df = pd.read_csv(input_file)
    print(f"   Исходных записей: {len(df):,}")
    print(f"   Колонок: {len(df.columns)}")
    print(f"   Период: {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}")
    
    # Топ-25 фичей из анализа важности (те же, что использует модель horizon_5)
    top_25_features = [
        # Топ-10 самых важных
        'volume_sma_20',     # Volume indicator
        'atr_14',            # Volatility
        'volatility_ratio',  # Volatility ratio
        'bb_width',          # Bollinger width
        'ema_100',           # Long-term trend
        'ema_diff_10_50',    # MA difference
        'rsi_14',            # Main oscillator
        'stoch_d',           # Stochastic
        'ema_50',            # Medium trend
        'stoch_k',           # Stochastic
        
        # Следующие 15 лучших
        'bb_position',       # BB position
        'slope_ema_20',      # EMA slope
        'momentum_3',        # Short momentum
        'macd_line',         # MACD
        'bb_upper',          # BB upper
        'macd_histogram',    # MACD histogram
        'relative_volume',   # Relative volume
        'momentum_10',       # Medium momentum
        'volume_change',     # Volume change
        'cci_20',            # CCI
        
        # Лаговые признаки (добавим их позже)
        'rsi_14_lag_1',      # RSI lag 1
        'rsi_14_lag_3',      # RSI lag 3
        'bb_position_lag_1', # BB position lag 1
        'bb_position_lag_3', # BB position lag 3
        'momentum_3_lag_1'   # Momentum lag 1
    ]
    
    print(f"\n🔧 Создание топ-25 фичей...")
    
    # Проверяем, какие базовые фичи есть в данных
    available_base_features = []
    missing_base_features = []
    
    # Базовые фичи (без лагов)
    base_features = [feat for feat in top_25_features if '_lag_' not in feat]
    
    for feature in base_features:
        if feature in df.columns:
            available_base_features.append(feature)
        else:
            missing_base_features.append(feature)
    
    print(f"   ✅ Доступных базовых фичей: {len(available_base_features)}/{len(base_features)}")
    print(f"   ❌ Отсутствующих фичей: {len(missing_base_features)}")
    
    if missing_base_features:
        print("   Отсутствующие фичи:")
        for feat in missing_base_features:
            print(f"      - {feat}")
    
    # Создаем недостающие фичи
    print(f"\n🔧 Создание недостающих фичей...")
    
    # volatility_ratio (если нет)
    if 'volatility_ratio' not in df.columns and 'atr_14' in df.columns:
        df['volatility_ratio'] = df['atr_14'] / df['close'] 
        print(f"   ✅ Создана volatility_ratio")
    
    # relative_volume (если нет)
    if 'relative_volume' not in df.columns and 'volume_sma_20' in df.columns:
        df['relative_volume'] = df['volume'] / df['volume_sma_20']
        print(f"   ✅ Создана relative_volume")
    
    # Создаем лаговые признаки
    print(f"\n🔄 Создание лаговых признаков...")
    
    lag_features = {
        'rsi_14_lag_1': ('rsi_14', 1),
        'rsi_14_lag_3': ('rsi_14', 3),
        'bb_position_lag_1': ('bb_position', 1),
        'bb_position_lag_3': ('bb_position', 3),
        'momentum_3_lag_1': ('momentum_3', 1)
    }
    
    for lag_feat, (base_feat, lag) in lag_features.items():
        if base_feat in df.columns:
            df[lag_feat] = df[base_feat].shift(lag)
            print(f"   ✅ Создан {lag_feat}")
        else:
            print(f"   ❌ Не удалось создать {lag_feat} - нет {base_feat}")
    
    # Создаем target_5 (цена через 5 периодов выше текущей)
    print(f"\n🎯 Создание target_5...")
    df['target_5'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # Финальная проверка доступности всех топ-25 фичей
    print(f"\n📋 Проверка финальной доступности фичей...")
    final_available = []
    final_missing = []
    
    for feature in top_25_features:
        if feature in df.columns:
            final_available.append(feature)
        else:
            final_missing.append(feature)
    
    print(f"   ✅ Доступно: {len(final_available)}/25 фичей")
    print(f"   ❌ Отсутствует: {len(final_missing)} фичей")
    
    if final_missing:
        print("   Отсутствующие фичи:")
        for feat in final_missing:
            print(f"      - {feat}")
    
    # Подготавливаем финальный датасет
    print(f"\n💾 Подготовка финального датасета...")
    
    # Берем только нужные колонки
    final_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + final_available + ['target_5']
    
    # Создаем финальный датафрейм
    final_df = df[final_columns].copy()
    
    # Убираем строки с NaN в target_5 и лаговых фичах
    print(f"   Записей до очистки: {len(final_df):,}")
    
    # Удаляем последние 5 строк (где target_5 = NaN)
    final_df = final_df.dropna(subset=['target_5'])
    
    # Удаляем первые строки где есть NaN в лаговых фичах
    final_df = final_df.dropna()
    
    print(f"   Записей после очистки: {len(final_df):,}")
    
    # Статистика по target_5
    target_counts = final_df['target_5'].value_counts()
    print(f"\n📊 Распределение target_5:")
    print(f"   UP (1): {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(final_df)*100:.1f}%)")
    print(f"   DOWN (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(final_df)*100:.1f}%)")
    
    # Проверяем, что у нас ровно 25 фичей
    feature_cols = [col for col in final_df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target_5']]
    print(f"\n🔢 Финальная проверка фичей:")
    print(f"   Количество фичей: {len(feature_cols)}")
    
    if len(feature_cols) == 25:
        print(f"   ✅ Идеально! Ровно 25 фичей")
    else:
        print(f"   ⚠️ Ожидалось 25, получено {len(feature_cols)}")
    
    print(f"\n📋 Список фичей:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"   {i:2d}. {feat}")
    
    # Сохраняем результат
    output_file = "data/processed/SOLUSDT_5m_2024_top25_ready.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Результат сохранен:")
    print(f"   Файл: {output_file}")
    print(f"   Записей: {len(final_df):,}")
    print(f"   Фичей: {len(feature_cols)}")
    print(f"   Период: {final_df['timestamp'].iloc[0]} - {final_df['timestamp'].iloc[-1]}")
    
    # Краткая статистика по рынку
    start_price = final_df['close'].iloc[0]
    end_price = final_df['close'].iloc[-1]
    annual_return = (end_price / start_price - 1) * 100
    
    print(f"\n📈 Рыночная статистика 2024:")
    print(f"   Начальная цена: ${start_price:.2f}")
    print(f"   Конечная цена: ${end_price:.2f}")
    print(f"   Доходность: {annual_return:.1f}%")
    print(f"   Мин: ${final_df['low'].min():.2f}")
    print(f"   Макс: ${final_df['high'].max():.2f}")
    
    print(f"\n✅ Датасет 2024 года готов для тестирования модели horizon_5!")
    
    return output_file

if __name__ == "__main__":
    try:
        result = prepare_2024_top25_dataset()
        if result:
            print(f"\n🎉 Успешно подготовлен: {result}")
        else:
            print(f"\n❌ Ошибка подготовки датасета")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()