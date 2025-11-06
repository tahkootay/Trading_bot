#!/usr/bin/env python3
"""
Clean Data Splitting Script

Создает чистое хронологическое разделение данных без пересечений:
- Train: 2025-01-01 → 2025-06-30 (6 месяцев)
- Validation: 2025-07-01 → 2025-08-31 (2 месяца)  
- Test: 2025-09-01 → 2025-10-31 (2 месяца)

Особенности:
- Убирает NaN значения из начала данных
- Создает целевые переменные для разных горизонтов
- Обеспечивает нулевые пересечения между наборами
- Сохраняет метаданные для воспроизводимости
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def create_target_variables(df, horizons=[3, 5, 7]):
    """Создает целевые переменные для разных горизонтов предсказания"""
    print(f"🎯 Создание целевых переменных для горизонтов: {horizons}")
    
    for horizon in horizons:
        target_col = f'target_h{horizon}'
        df[target_col] = (df['close'].shift(-horizon) > df['close']).astype(int)
        
        # Статистика распределения
        target_counts = df[target_col].value_counts()
        if len(target_counts) == 2:
            up_ratio = target_counts.get(1, 0) / len(df[target_col].dropna()) * 100
            print(f"  Горизонт {horizon}: {up_ratio:.1f}% UP, {100-up_ratio:.1f}% DOWN")
    
    return df

def clean_and_prepare_data(file_path):
    """Очистка и подготовка данных для разделения"""
    print("📂 Загрузка и очистка исходных данных...")
    
    # Загрузка
    df = pd.read_csv(file_path)
    print(f"📊 Исходный размер: {df.shape}")
    
    # Конвертация timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Удаление строк с NaN (первые 99 строк по анализу)
    df_clean = df.dropna().copy()
    removed_rows = len(df) - len(df_clean)
    print(f"🧹 Удалено строк с NaN: {removed_rows}")
    print(f"✅ Размер после очистки: {df_clean.shape}")
    
    # Сортировка по времени для гарантии хронологического порядка
    df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
    
    # Создание целевых переменных
    df_clean = create_target_variables(df_clean, horizons=[3, 5, 7])
    
    return df_clean

def split_data_chronologically(df):
    """Хронологическое разделение данных без пересечений"""
    print("\n⏱️ Хронологическое разделение данных...")
    
    # Точные временные границы
    train_start = '2025-01-01 00:00:00'
    train_end = '2025-06-30 23:59:59'
    
    val_start = '2025-07-01 00:00:00'
    val_end = '2025-08-31 23:59:59'
    
    test_start = '2025-09-01 00:00:00'
    test_end = '2025-10-31 23:59:59'
    
    # Разделение по временным границам
    train_df = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)].copy()
    val_df = df[(df['timestamp'] >= val_start) & (df['timestamp'] <= val_end)].copy()
    test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)].copy()
    
    print(f"📅 Train период: {train_start} → {train_end}")
    print(f"   Размер: {len(train_df):,} записей")
    print(f"   Реальный период: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
    
    print(f"📅 Validation период: {val_start} → {val_end}")
    print(f"   Размер: {len(val_df):,} записей")
    print(f"   Реальный период: {val_df['timestamp'].min()} → {val_df['timestamp'].max()}")
    
    print(f"📅 Test период: {test_start} → {test_end}")
    print(f"   Размер: {len(test_df):,} записей")
    print(f"   Реальный период: {test_df['timestamp'].min()} → {test_df['timestamp'].max()}")
    
    # Проверка на пересечения
    overlap_check = check_for_overlaps(train_df, val_df, test_df)
    if overlap_check:
        print("✅ Пересечения между наборами отсутствуют")
    else:
        raise ValueError("❌ Обнаружены пересечения между наборами!")
    
    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df,
        'boundaries': {
            'train': [train_start, train_end],
            'validation': [val_start, val_end],
            'test': [test_start, test_end]
        }
    }

def check_for_overlaps(train_df, val_df, test_df):
    """Проверка отсутствия пересечений между наборами данных"""
    train_max = train_df['timestamp'].max()
    val_min = val_df['timestamp'].min()
    val_max = val_df['timestamp'].max()
    test_min = test_df['timestamp'].min()
    
    # Проверка: train должен заканчиваться до val
    gap1 = val_min > train_max
    # Проверка: val должен заканчиваться до test  
    gap2 = test_min > val_max
    
    print(f"🔍 Проверка пересечений:")
    print(f"   Train конец → Val начало: {train_max} → {val_min} ({'✅' if gap1 else '❌'})")
    print(f"   Val конец → Test начало: {val_max} → {test_min} ({'✅' if gap2 else '❌'})")
    
    return gap1 and gap2

def validate_target_variables(splits_data, horizons=[3, 5, 7]):
    """Валидация целевых переменных"""
    print(f"\n🎯 Валидация целевых переменных...")
    
    for split_name, split_df in splits_data.items():
        if split_name == 'boundaries':
            continue
            
        print(f"\n📊 {split_name.upper()} SET:")
        
        for horizon in horizons:
            target_col = f'target_h{horizon}'
            
            if target_col not in split_df.columns:
                print(f"   ❌ {target_col}: отсутствует")
                continue
            
            # Удаляем строки где нельзя создать target (последние N строк)
            valid_targets = split_df[target_col].dropna()
            total_rows = len(split_df)
            valid_rows = len(valid_targets)
            lost_rows = total_rows - valid_rows
            
            if valid_rows == 0:
                print(f"   ❌ {target_col}: нет валидных значений")
                continue
            
            # Распределение классов
            up_count = (valid_targets == 1).sum()
            down_count = (valid_targets == 0).sum()
            up_ratio = up_count / valid_rows * 100
            
            print(f"   ✅ {target_col}: {valid_rows:,} валидных из {total_rows:,} ({lost_rows} потеряно)")
            print(f"      UP: {up_count:,} ({up_ratio:.1f}%), DOWN: {down_count:,} ({100-up_ratio:.1f}%)")

def save_clean_splits(splits_data, base_dir="data/processed"):
    """Сохранение чистых разделов данных"""
    print(f"\n💾 Сохранение чистых разделов в {base_dir}...")
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Определение индикаторов для LSTM
    lstm_features = [
        'rsi_14', 'rsi_14_lag_1', 'rsi_14_lag_3',
        'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100',
        'sma_5', 'sma_10', 'sma_20',
        'macd_line', 'macd', 'macd_signal', 'macd_histogram',
        'bb_position', 'bb_width', 'bb_position_lag_1', 'bb_position_lag_3',
        'atr_14', 'volatility_ratio',
        'momentum_3', 'momentum_10', 'momentum_3_lag_1',
        'stoch_k', 'stoch_d',
        'relative_volume', 'volume_change',
        'candle_ratio', 'body_to_range',
        'cci_20', 'slope_ema_20', 'ema_diff_10_50'
    ]
    
    saved_files = {}
    
    for split_name, split_df in splits_data.items():
        if split_name == 'boundaries':
            continue
        
        # Базовые колонки + индикаторы + целевые переменные
        target_cols = [col for col in split_df.columns if col.startswith('target_h')]
        base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Все нужные колонки
        all_cols = base_cols + lstm_features + target_cols
        
        # Фильтрация доступных колонок
        available_cols = [col for col in all_cols if col in split_df.columns]
        
        # Создание чистой выборки
        clean_df = split_df[available_cols].copy()
        
        # Сохранение
        filename = f"{split_name}_2025_clean.csv"
        filepath = os.path.join(base_dir, filename)
        clean_df.to_csv(filepath, index=False)
        
        saved_files[split_name] = {
            'filepath': filepath,
            'rows': len(clean_df),
            'columns': len(clean_df.columns),
            'period': [str(clean_df['timestamp'].min()), str(clean_df['timestamp'].max())]
        }
        
        print(f"✅ {split_name}: {len(clean_df):,} x {len(clean_df.columns)} → {filename}")
    
    return saved_files

def create_metadata(splits_data, saved_files, original_file_path):
    """Создание метаданных разделения"""
    print("\n📋 Создание метаданных...")
    
    # Сбор статистики
    metadata = {
        'created_at': datetime.now().isoformat(),
        'source_file': original_file_path,
        'split_method': 'chronological_no_overlap',
        'horizons': [3, 5, 7],
        'lstm_features_count': 33,
        'boundaries': splits_data['boundaries'],
        'splits': {}
    }
    
    for split_name, file_info in saved_files.items():
        split_df = splits_data[split_name]
        
        # Статистика целевых переменных
        target_stats = {}
        for horizon in [3, 5, 7]:
            target_col = f'target_h{horizon}'
            if target_col in split_df.columns:
                valid_targets = split_df[target_col].dropna()
                if len(valid_targets) > 0:
                    up_ratio = (valid_targets == 1).mean()
                    target_stats[f'h{horizon}'] = {
                        'total_samples': len(valid_targets),
                        'up_ratio': round(up_ratio, 4),
                        'down_ratio': round(1 - up_ratio, 4)
                    }
        
        metadata['splits'][split_name] = {
            'file_path': file_info['filepath'],
            'rows': file_info['rows'],
            'columns': file_info['columns'],
            'period': file_info['period'],
            'target_statistics': target_stats
        }
    
    # Сохранение метаданных
    metadata_path = "data/processed/split_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Метаданные сохранены: {metadata_path}")
    
    return metadata

def main():
    """Основная функция создания чистых разделов"""
    print("🚀 Создание чистых разделов данных для LSTM обучения")
    print("="*60)
    
    # Путь к исходным данным
    source_file = "data/raw/SOLUSDT_5m_20250101_20251031_advanced_indicators.csv"
    
    if not os.path.exists(source_file):
        print(f"❌ Исходный файл не найден: {source_file}")
        return
    
    try:
        # 1. Очистка и подготовка данных
        df_clean = clean_and_prepare_data(source_file)
        
        # 2. Хронологическое разделение
        splits_data = split_data_chronologically(df_clean)
        
        # 3. Валидация целевых переменных  
        validate_target_variables(splits_data)
        
        # 4. Сохранение чистых файлов
        saved_files = save_clean_splits(splits_data)
        
        # 5. Создание метаданных
        metadata = create_metadata(splits_data, saved_files, source_file)
        
        print("\n" + "="*60)
        print("✅ УСПЕШНО! Чистые разделы данных созданы:")
        print("="*60)
        
        for split_name, file_info in saved_files.items():
            print(f"📁 {split_name.upper():>10}: {file_info['rows']:>6,} записей → {os.path.basename(file_info['filepath'])}")
        
        print(f"\n🎯 Готово к обучению LSTM с горизонтами: 3, 5, 7 баров")
        print(f"📄 Метаданные: split_metadata.json")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        raise

if __name__ == "__main__":
    main()