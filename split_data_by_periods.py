#!/usr/bin/env python3
"""
Разделение данных на периоды для обучения и тестирования:
- Train: январь–июнь 2025
- Validation: июль 2025  
- Test: август, сентябрь, октябрь 2025 (отдельные файлы)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def split_data_by_periods():
    """Разделение данных на тренировочные, валидационные и тестовые периоды."""
    
    print("📊 РАЗДЕЛЕНИЕ ДАННЫХ ПО ПЕРИОДАМ 2025 ГОДА")
    print("=" * 60)
    
    # Загружаем исходный файл
    input_file = "data/raw/SOLUSDT_5m_20250101_20251031_advanced_indicators.csv"
    
    if not Path(input_file).exists():
        print(f"❌ Файл не найден: {input_file}")
        return None
    
    print(f"📁 Загружаем файл: {input_file}")
    df = pd.read_csv(input_file)
    
    # Конвертируем timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"📊 Исходные данные:")
    print(f"   Записей: {len(df):,}")
    print(f"   Колонок: {len(df.columns)}")
    print(f"   Период: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print()
    
    # Создаем директорию для обработанных данных
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Определяем временные границы
    periods = {
        'train': {
            'start': pd.Timestamp('2025-01-01'),
            'end': pd.Timestamp('2025-06-30 23:59:59'),
            'name': 'Train (Jan-Jun 2025)',
            'filename': 'train_jan_jun_2025.csv'
        },
        'validation': {
            'start': pd.Timestamp('2025-07-01'),
            'end': pd.Timestamp('2025-07-31 23:59:59'),
            'name': 'Validation (Jul 2025)',
            'filename': 'validation_jul_2025.csv'
        },
        'test_aug': {
            'start': pd.Timestamp('2025-08-01'),
            'end': pd.Timestamp('2025-08-31 23:59:59'),
            'name': 'Test August 2025',
            'filename': 'test_aug_2025.csv'
        },
        'test_sep': {
            'start': pd.Timestamp('2025-09-01'),
            'end': pd.Timestamp('2025-09-30 23:59:59'),
            'name': 'Test September 2025',
            'filename': 'test_sep_2025.csv'
        },
        'test_oct': {
            'start': pd.Timestamp('2025-10-01'),
            'end': pd.Timestamp('2025-10-31 23:59:59'),
            'name': 'Test October 2025',
            'filename': 'test_oct_2025.csv'
        }
    }
    
    # Разделяем и сохраняем данные
    saved_files = {}
    total_saved_records = 0
    
    for period_key, period_info in periods.items():
        print(f"📅 Обрабатываем {period_info['name']}...")
        
        # Фильтруем данные по периоду
        mask = (df['timestamp'] >= period_info['start']) & (df['timestamp'] <= period_info['end'])
        period_df = df[mask].copy()
        
        if len(period_df) == 0:
            print(f"   ⚠️ Нет данных за период {period_info['name']}")
            continue
        
        # Информация о периоде
        period_start = period_df['timestamp'].min()
        period_end = period_df['timestamp'].max()
        period_records = len(period_df)
        
        print(f"   📊 Записей: {period_records:,}")
        print(f"   📅 Фактический период: {period_start} → {period_end}")
        
        # Проверяем наличие всех 25 фичей
        expected_features = [
            'volume_sma_20', 'atr_14', 'volatility_ratio', 'bb_width', 'ema_100',
            'ema_diff_10_50', 'rsi_14', 'stoch_d', 'ema_50', 'stoch_k', 
            'bb_position', 'slope_ema_20', 'momentum_3', 'macd_line', 'bb_upper',
            'macd_histogram', 'relative_volume', 'momentum_10', 'volume_change', 'cci_20',
            'rsi_14_lag_1', 'rsi_14_lag_3', 'bb_position_lag_1', 'bb_position_lag_3', 'momentum_3_lag_1'
        ]
        
        missing_features = [f for f in expected_features if f not in period_df.columns]
        if missing_features:
            print(f"   ⚠️ Отсутствующие фичи: {missing_features}")
        else:
            print(f"   ✅ Все 25 фичей присутствуют")
        
        # Сохраняем файл
        output_file = output_dir / period_info['filename']
        period_df.to_csv(output_file, index=False)
        
        print(f"   💾 Сохранено: {output_file}")
        print(f"   📏 Размер файла: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        print()
        
        saved_files[period_key] = {
            'filename': str(output_file),
            'records': period_records,
            'start': period_start,
            'end': period_end,
            'size_mb': output_file.stat().st_size / 1024 / 1024
        }
        
        total_saved_records += period_records
    
    # Итоговая статистика
    print("📋 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"📊 Всего обработано записей: {total_saved_records:,} из {len(df):,}")
    print(f"📊 Покрытие: {total_saved_records/len(df)*100:.1f}%")
    print()
    
    for period_key, file_info in saved_files.items():
        period_name = periods[period_key]['name']
        print(f"📁 {period_name}:")
        print(f"   Файл: {file_info['filename']}")
        print(f"   Записей: {file_info['records']:,}")
        print(f"   Период: {file_info['start']} → {file_info['end']}")
        print(f"   Размер: {file_info['size_mb']:.1f} MB")
        print()
    
    print("✅ Разделение данных завершено успешно!")
    print("🎯 Файлы готовы для:")
    print("   • Обучения моделей (train)")
    print("   • Валидации гиперпараметров (validation)")
    print("   • Тестирования на разных месяцах (test_aug, test_sep, test_oct)")
    
    return saved_files

if __name__ == "__main__":
    try:
        result = split_data_by_periods()
        if result:
            print(f"\n🎉 Успешно создано {len(result)} файлов")
        else:
            print(f"\n❌ Ошибка при разделении данных")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()