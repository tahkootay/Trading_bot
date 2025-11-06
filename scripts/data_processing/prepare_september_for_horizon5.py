#!/usr/bin/env python3
"""
Подготовка сентябрьских данных для бэктеста модели horizon_5.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def prepare_september_data_for_horizon5():
    """Подготовка сентябрьских данных в формате для модели horizon_5."""
    
    print("🗓️ Подготовка сентябрьских данных для модели horizon_5")
    
    # Файл с данными до сентября
    input_file = "data/raw/SOLUSDT_5m_20250301_20250930_advanced_indicators.csv"
    
    if not Path(input_file).exists():
        print(f"❌ Файл с данными не найден: {input_file}")
        return None
    
    print(f"📊 Используем файл: {input_file}")
    
    # Загружаем данные
    df = pd.read_csv(input_file)
    print(f"   Загружено записей: {len(df)}")
    print(f"   Колонок: {len(df.columns)}")
    
    # Конвертируем timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   Период: {df['timestamp'].min()} - {df['timestamp'].max()}")
    
    # Фильтруем только сентябрьские данные
    september_mask = (df['timestamp'] >= pd.Timestamp('2025-09-01')) & (df['timestamp'] < pd.Timestamp('2025-10-01'))
    df_september = df[september_mask].copy()
    
    print(f"   Сентябрьских записей: {len(df_september)}")
    
    if len(df_september) == 0:
        print("❌ Нет данных за сентябрь 2025")
        return None
    
    # Список признаков, которые ожидает модель horizon_5 (из october_2025_horizon5.csv)
    expected_features = [
        "volume_sma_20", "atr_14", "volatility_ratio", "bb_width", "ema_100",
        "ema_diff_10_50", "rsi_14", "stoch_d", "ema_50", "stoch_k", 
        "bb_position", "slope_ema_20", "momentum_3", "macd_line", "bb_upper",
        "macd_histogram", "relative_volume", "momentum_10", "volume_change", "cci_20",
        "rsi_14_lag_1", "rsi_14_lag_3", "bb_position_lag_1", "bb_position_lag_3", "momentum_3_lag_1"
    ]
    
    print(f"\n🔧 Проверяем наличие признаков...")
    missing_features = []
    available_features = []
    
    for feature in expected_features:
        if feature in df_september.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
    
    print(f"   ✅ Доступно: {len(available_features)}/{len(expected_features)} признаков")
    
    if missing_features:
        print(f"   ❌ Отсутствуют признаки: {missing_features}")
        print(f"   🔧 Добавляем недостающие lag-признаки...")
        
        # Добавляем lag-признаки
        lag_features_map = {
            'rsi_14_lag_1': ('rsi_14', 1),
            'rsi_14_lag_3': ('rsi_14', 3),
            'bb_position_lag_1': ('bb_position', 1),
            'bb_position_lag_3': ('bb_position', 3),
            'momentum_3_lag_1': ('momentum_3', 1)
        }
        
        for lag_feature, (base_feature, lag_periods) in lag_features_map.items():
            if lag_feature in missing_features and base_feature in df_september.columns:
                df_september[lag_feature] = df_september[base_feature].shift(lag_periods)
                available_features.append(lag_feature)
                missing_features.remove(lag_feature)
                print(f"     ✅ Добавлен {lag_feature}")
        
        print(f"   📊 Итого доступно: {len(available_features)}/{len(expected_features)} признаков")
    
    # Создаем target_5 (цена через 5 периодов выше текущей?)
    print(f"\n🎯 Создаем target_5...")
    df_september = df_september.sort_values('timestamp').reset_index(drop=True)
    
    # Target_5: 1 если цена через 5 периодов выше текущей, 0 иначе
    df_september['target_5'] = 0
    for i in range(len(df_september) - 5):
        current_price = df_september.loc[i, 'close']
        future_price = df_september.loc[i + 5, 'close']
        df_september.loc[i, 'target_5'] = 1 if future_price > current_price else 0
    
    # Удаляем последние 5 строк (нет future price)
    df_september = df_september.iloc[:-5].copy()
    
    target_counts = df_september['target_5'].value_counts()
    print(f"   Target_5 распределение:")
    print(f"     UP (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df_september)*100:.1f}%)")
    print(f"     DOWN (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df_september)*100:.1f}%)")
    
    # Создаем итоговый датасет с нужными колонками
    print(f"\n📋 Создаем итоговый датасет...")
    
    # Основные колонки
    base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Используем только доступные признаки
    final_columns = base_columns + available_features + ['target_5']
    
    df_final = df_september[final_columns].copy()
    
    print(f"   Финальный датасет: {len(df_final)} записей, {len(df_final.columns)} колонок")
    print(f"   Период: {df_final['timestamp'].min()} - {df_final['timestamp'].max()}")
    
    # Заполняем NaN значения
    numeric_columns = df_final.select_dtypes(include=[np.number]).columns
    df_final[numeric_columns] = df_final[numeric_columns].fillna(0)
    
    # Сохраняем результат
    output_file = "data/processed/september_2025_horizon5.csv"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    df_final.to_csv(output_file, index=False)
    print(f"\n💾 Сохранено: {output_file}")
    
    print(f"\n✅ Подготовка сентябрьских данных завершена!")
    print(f"📁 Файл готов для бэктеста: {output_file}")
    
    return output_file

if __name__ == "__main__":
    try:
        result = prepare_september_data_for_horizon5()
        if result:
            print(f"\n🎉 Успешно подготовлены данные: {result}")
        else:
            print(f"\n❌ Ошибка подготовки данных")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()