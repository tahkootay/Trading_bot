#!/usr/bin/env python3
"""
Подготовка октябрьских данных для тестирования модели.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def prepare_october_data():
    """Подготовка оптимизированного датасета для октября."""
    
    print("🗓️  ПОДГОТОВКА ОКТЯБРЬСКИХ ДАННЫХ")
    print("=" * 45)
    
    # Загружаем данные с индикаторами
    input_file = "data/raw/SOLUSDT_5m_20251001_20251031_advanced.csv"
    
    if not Path(input_file).exists():
        print(f"❌ Файл не найден: {input_file}")
        return None
    
    df = pd.read_csv(input_file)
    print(f"📊 Загружено данных: {len(df)} строк")
    
    # Топ-25 фичей из нашего анализа
    top_25_features = [
        'volume_sma_20', 'atr_14', 'volatility_ratio', 'bb_width', 'ema_100',
        'ema_diff_10_50', 'rsi_14', 'stoch_d', 'ema_50', 'stoch_k',
        'bb_position', 'slope_ema_20', 'momentum_3', 'macd_line', 'bb_upper',
        'macd_histogram', 'relative_volume', 'momentum_10', 'volume_change', 'cci_20'
    ]
    
    # Создаём лаг-фичи
    print("🔧 Создание лаг-фичей...")
    for feature in ['rsi_14', 'bb_position', 'momentum_3']:
        if feature in df.columns:
            if feature == 'rsi_14':
                df['rsi_14_lag_1'] = df['rsi_14'].shift(1)
                df['rsi_14_lag_3'] = df['rsi_14'].shift(3)
            elif feature == 'bb_position':
                df['bb_position_lag_1'] = df['bb_position'].shift(1)
                df['bb_position_lag_3'] = df['bb_position'].shift(3)
            elif feature == 'momentum_3':
                df['momentum_3_lag_1'] = df['momentum_3'].shift(1)
    
    # Все топ-25 фичи включая лаги
    lag_features = ['rsi_14_lag_1', 'rsi_14_lag_3', 'bb_position_lag_1', 'bb_position_lag_3', 'momentum_3_lag_1']
    all_top_25 = top_25_features + lag_features
    
    # Базовые колонки
    base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Фильтруем только доступные фичи
    available_features = [f for f in all_top_25 if f in df.columns]
    missing_features = [f for f in all_top_25 if f not in df.columns]
    
    if missing_features:
        print(f"⚠️  Недостающие фичи: {missing_features}")
    
    print(f"✅ Доступные фичи: {len(available_features)}/25")
    
    # Создаём оптимизированный датасет
    columns_to_keep = base_cols + available_features
    october_optimized = df[columns_to_keep].copy()
    
    # Удаляем NaN
    october_optimized = october_optimized.dropna()
    
    # Добавляем target для тестирования (горизонт 10)
    october_optimized['target_10'] = (october_optimized['close'].shift(-10) > october_optimized['close']).astype(int)
    october_optimized = october_optimized.dropna()
    
    # Сохраняем
    output_file = "data/processed/SOLUSDT_5m_october2025_real.csv"
    october_optimized.to_csv(output_file, index=False)
    
    print(f"✅ Оптимизированный датасет создан: {output_file}")
    print(f"📈 Финальные фичи: {len(available_features)}")
    print(f"📊 Финальные строки: {len(october_optimized)}")
    print(f"📅 Период: {october_optimized['timestamp'].min()} - {october_optimized['timestamp'].max()}")
    
    return output_file

def main():
    prepare_october_data()

if __name__ == "__main__":
    main()