#!/usr/bin/env python3
"""
Подготовка октябрьских данных 2025 для бэктеста модели horizon_5.
Создание target_5 из существующих данных.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def prepare_october_horizon5():
    """Подготовка октябрьских данных 2025 с target_5."""
    
    print("🗓️ Подготовка октябрьских данных 2025 для модели horizon_5")
    
    # Загружаем оптимизированные октябрьские данные
    input_file = "data/processed/SOLUSDT_5m_october2025_optimized.csv"
    
    if not Path(input_file).exists():
        print(f"❌ Файл не найден: {input_file}")
        return None
    
    df = pd.read_csv(input_file)
    print(f"📊 Загружено: {len(df)} записей")
    print(f"   Период: {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}")
    
    # Создаем target_5 (цена через 5 периодов выше текущей)
    print("🎯 Создание target_5...")
    df['target_5'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # Удаляем старый target_10
    if 'target_10' in df.columns:
        df = df.drop('target_10', axis=1)
    
    # Убираем последние 5 строк с NaN в target_5
    df = df.dropna(subset=['target_5'])
    
    print(f"   Записей после очистки: {len(df)}")
    
    # Статистика по target_5
    target_counts = df['target_5'].value_counts()
    print(f"📊 Распределение target_5:")
    print(f"   UP (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"   DOWN (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    
    # Проверяем признаки
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target_5']]
    print(f"🔢 Признаков для модели: {len(feature_cols)}")
    
    if len(feature_cols) != 25:
        print(f"⚠️ Ожидалось 25 признаков, получено {len(feature_cols)}")
        return None
    
    # Сохраняем подготовленные данные
    output_file = "data/processed/october_2025_horizon5.csv"
    df.to_csv(output_file, index=False)
    
    print(f"💾 Данные сохранены: {output_file}")
    print(f"✅ Готово для бэктеста модели horizon_5")
    
    return output_file

if __name__ == "__main__":
    result = prepare_october_horizon5()
    if result:
        print(f"\n🎉 Данные готовы: {result}")
    else:
        print("\n❌ Ошибка подготовки данных")