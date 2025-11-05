#!/usr/bin/env python3
"""
Подготовка октябрьских данных для бэктеста модели horizon_5.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def prepare_october_data_for_horizon5():
    """Подготовка октябрьских данных в формате для модели horizon_5."""
    
    print("🗓️ Подготовка октябрьских данных для модели horizon_5")
    
    # Ищем октябрьские данные с индикаторами
    october_files = [
        "data/raw/SOLUSDT_5m_20251001_20251031_advanced_indicators.csv",
        "data/raw/SOLUSDT_5m_20251022_20251029_advanced_indicators.csv"
    ]
    
    input_file = None
    for file_path in october_files:
        if Path(file_path).exists():
            input_file = file_path
            break
    
    if not input_file:
        print(f"❌ Октябрьские данные с индикаторами не найдены")
        print(f"   Искали: {october_files}")
        return None
    
    print(f"📊 Используем файл: {input_file}")
    
    # Загружаем данные
    df = pd.read_csv(input_file)
    print(f"   Загружено записей: {len(df)}")
    print(f"   Колонок: {len(df.columns)}")
    
    # Список признаков, которые ожидает модель (из metadata_v1.json)
    expected_features = [
        "sma_5", "sma_10", "sma_20", "ema_5", "ema_10", "ema_20", "ema_50", "ema_100",
        "slope_ema_20", "ema_diff_10_50", "rsi_14", "stoch_k", "stoch_d", 
        "macd_line", "macd_signal", "macd_histogram", "cci_20", "atr_14",
        "high_low_range", "body_to_range", "volatility_ratio",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
        "bb_touch_upper", "bb_touch_lower", "volume_change", "volume_sma_20",
        "relative_volume", "candle_type", "candle_ratio", "momentum_3", "momentum_10",
        "rsi_14_lag_1", "rsi_14_lag_2", "rsi_14_lag_3",
        "macd_line_lag_1", "macd_line_lag_2", "macd_line_lag_3",
        "ema_20_lag_1", "ema_20_lag_2", "ema_20_lag_3",
        "bb_position_lag_1", "bb_position_lag_2", "bb_position_lag_3",
        "momentum_3_lag_1", "momentum_3_lag_2", "momentum_3_lag_3"
    ]
    
    print(f"🔍 Ожидаемых признаков: {len(expected_features)}")
    
    # Проверяем, какие признаки есть в данных
    available_features = [col for col in expected_features if col in df.columns]
    missing_features = [col for col in expected_features if col not in df.columns]
    
    print(f"✅ Доступных признаков: {len(available_features)}")
    print(f"❌ Отсутствующих признаков: {len(missing_features)}")
    
    if missing_features:
        print("   Отсутствующие признаки:")
        for feature in missing_features[:10]:  # Показываем первые 10
            print(f"   - {feature}")
        if len(missing_features) > 10:
            print(f"   ... и еще {len(missing_features) - 10}")
    
    # Если слишком много отсутствующих признаков, используем альтернативный подход
    if len(missing_features) > 10:
        print("\n⚠️  Слишком много отсутствующих признаков")
        print("🔄 Используем доступные данные и создаем мини-бэктест")
        
        # Создадим упрощенную версию с доступными признаками
        # Используем данные, которые точно есть
        basic_features = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if 'rsi_14' in df.columns:
            basic_features.append('rsi_14')
        if 'ema_20' in df.columns:
            basic_features.append('ema_20')
        if 'ema_50' in df.columns:
            basic_features.append('ema_50')
            
        # Создаем упрощенный датасет
        simple_df = df[basic_features].copy()
        
        # Создаем простую цель: цена через 5 периодов выше текущей
        simple_df['target_5'] = (simple_df['close'].shift(-5) > simple_df['close']).astype(int)
        
        # Убираем NaN
        simple_df = simple_df.dropna()
        
        output_file = "data/processed/october_2025_simple_horizon5.csv"
        simple_df.to_csv(output_file, index=False)
        
        print(f"💾 Сохранен упрощенный датасет: {output_file}")
        print(f"   Записей: {len(simple_df)}")
        
        return output_file
    
    # Если большинство признаков доступны, создаем полный датасет
    feature_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume'] + available_features].copy()
    
    # Заполняем отсутствующие признаки нулями или средними значениями
    for feature in missing_features:
        if 'lag' in feature:
            # Для лаговых признаков используем базовый признак
            base_feature = feature.replace('_lag_1', '').replace('_lag_2', '').replace('_lag_3', '')
            if base_feature in df.columns:
                feature_df[feature] = df[base_feature]
            else:
                feature_df[feature] = 0
        else:
            feature_df[feature] = 0
    
    # Создаем цель для горизонта 5
    feature_df['target_5'] = (feature_df['close'].shift(-5) > feature_df['close']).astype(int)
    
    # Убираем строки с NaN
    feature_df = feature_df.dropna()
    
    # Сохраняем в нужном порядке
    final_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + expected_features + ['target_5']
    final_df = feature_df[final_columns]
    
    output_file = "data/processed/october_2025_horizon5_ready.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"💾 Сохранен полный датасет: {output_file}")
    print(f"   Записей: {len(final_df)}")
    print(f"   Признаков: {len(expected_features)}")
    
    # Статистика по цели
    target_stats = final_df['target_5'].value_counts()
    print(f"📊 Распределение цели:")
    print(f"   UP (1): {target_stats.get(1, 0)} ({target_stats.get(1, 0)/len(final_df)*100:.1f}%)")
    print(f"   DOWN (0): {target_stats.get(0, 0)} ({target_stats.get(0, 0)/len(final_df)*100:.1f}%)")
    
    return output_file

if __name__ == "__main__":
    try:
        result = prepare_october_data_for_horizon5()
        if result:
            print(f"\n✅ Данные готовы для бэктеста: {result}")
        else:
            print("\n❌ Не удалось подготовить данные")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()