#!/usr/bin/env python3
"""
Data Quality Analysis Script

Анализирует исходные данные SOLUSDT и выявляет проблемы качества:
- NaN значения
- Inf/-inf значения  
- Временные пропуски
- Статистика по индикаторам
- Рекомендации по очистке
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def analyze_data_quality(file_path):
    """Комплексный анализ качества данных"""
    print("🔍 Анализ качества данных SOLUSDT...")
    print(f"📁 Файл: {file_path}")
    
    # Загрузка данных
    df = pd.read_csv(file_path)
    print(f"📊 Общий размер данных: {df.shape}")
    print(f"📅 Период: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    
    # Конвертация timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Анализ NaN значений
    print("\n" + "="*50)
    print("📋 АНАЛИЗ NaN ЗНАЧЕНИЙ")
    print("="*50)
    
    nan_summary = df.isnull().sum()
    nan_cols = nan_summary[nan_summary > 0].sort_values(ascending=False)
    
    if len(nan_cols) > 0:
        print("❌ Найдены NaN значения:")
        for col, count in nan_cols.items():
            percentage = (count / len(df)) * 100
            print(f"  {col}: {count:,} ({percentage:.2f}%)")
        
        # Найти первую полную строку
        complete_rows = df.dropna()
        if len(complete_rows) > 0:
            first_complete_idx = df.index[df.isnull().sum(axis=1) == 0][0]
            first_complete_time = df.loc[first_complete_idx, 'timestamp']
            print(f"\n✅ Первая полная строка: индекс {first_complete_idx}, время {first_complete_time}")
            print(f"📉 Потеря данных: {first_complete_idx:,} строк ({(first_complete_idx/len(df))*100:.2f}%)")
        else:
            print("❌ Нет полностью заполненных строк!")
            return None
    else:
        print("✅ NaN значения не найдены")
    
    # 2. Анализ Inf значений
    print("\n" + "="*50)
    print("📋 АНАЛИЗ INF ЗНАЧЕНИЙ")
    print("="*50)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_found = False
    
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_found = True
            print(f"❌ {col}: {inf_count} inf значений")
    
    if not inf_found:
        print("✅ Inf значения не найдены")
    
    # 3. Анализ важных индикаторов для LSTM
    print("\n" + "="*50)
    print("📋 АНАЛИЗ ИНДИКАТОРОВ ДЛЯ LSTM")
    print("="*50)
    
    # Список индикаторов из LSTM модели
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
    
    available_features = [f for f in lstm_features if f in df.columns]
    missing_features = [f for f in lstm_features if f not in df.columns]
    
    print(f"✅ Доступных индикаторов: {len(available_features)}/{len(lstm_features)}")
    if missing_features:
        print("❌ Отсутствующие индикаторы:")
        for feat in missing_features:
            print(f"  - {feat}")
    
    # 4. Временной анализ
    print("\n" + "="*50)
    print("📋 ВРЕМЕННОЙ АНАЛИЗ")
    print("="*50)
    
    time_diff = df['timestamp'].diff()
    expected_diff = pd.Timedelta('5 minutes')
    gaps = time_diff[time_diff != expected_diff]
    
    if len(gaps) > 0:
        print(f"⚠️ Найдены временные пропуски: {len(gaps)}")
        print("Первые 5 пропусков:")
        for i, (idx, gap) in enumerate(gaps.head().items()):
            if i < 5:
                timestamp = df.loc[idx, 'timestamp']
                print(f"  {timestamp}: пропуск {gap}")
    else:
        print("✅ Временные пропуски не найдены")
    
    # 5. Статистика по периодам для разделения
    print("\n" + "="*50)
    print("📋 СТАТИСТИКА ПО ПЕРИОДАМ")
    print("="*50)
    
    # Определить границы периодов
    train_start = '2025-01-01'
    train_end = '2025-06-30'
    val_start = '2025-07-01' 
    val_end = '2025-08-31'
    test_start = '2025-09-01'
    test_end = '2025-10-31'
    
    train_data = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)]
    val_data = df[(df['timestamp'] >= val_start) & (df['timestamp'] <= val_end)]
    test_data = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)]
    
    print(f"🎯 Train период ({train_start} → {train_end}): {len(train_data):,} записей")
    print(f"🎯 Validation период ({val_start} → {val_end}): {len(val_data):,} записей")
    print(f"🎯 Test период ({test_start} → {test_end}): {len(test_data):,} записей")
    print(f"📊 Всего обработано: {len(train_data) + len(val_data) + len(test_data):,} из {len(df):,} записей")
    
    # 6. Рекомендации
    print("\n" + "="*50)
    print("📋 РЕКОМЕНДАЦИИ")
    print("="*50)
    
    recommendations = []
    
    if len(nan_cols) > 0:
        recommendations.append(f"❗ Удалить первые {first_complete_idx} строк с NaN значениями")
    
    if inf_found:
        recommendations.append("❗ Обработать Inf значения (заменить на NaN и интерполировать)")
        
    if len(gaps) > 0:
        recommendations.append("❗ Проанализировать временные пропуски")
    
    if len(missing_features) > 0:
        recommendations.append(f"❗ Добавить {len(missing_features)} отсутствующих индикаторов")
    
    if len(recommendations) == 0:
        print("✅ Данные готовы к использованию!")
    else:
        print("Необходимые действия:")
        for rec in recommendations:
            print(f"  {rec}")
    
    # 7. Сводная информация для разделения
    result_info = {
        'total_records': len(df),
        'first_complete_row': first_complete_idx if len(nan_cols) > 0 else 0,
        'available_features': available_features,
        'missing_features': missing_features,
        'train_records': len(train_data),
        'val_records': len(val_data), 
        'test_records': len(test_data),
        'time_gaps': len(gaps),
        'has_nan': len(nan_cols) > 0,
        'has_inf': inf_found
    }
    
    return result_info

def main():
    """Основная функция анализа"""
    data_path = "data/raw/SOLUSDT_5m_20250101_20251031_advanced_indicators.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Файл данных не найден: {data_path}")
        return
    
    # Анализ данных
    info = analyze_data_quality(data_path)
    
    if info:
        # Сохранение результатов
        os.makedirs("data/processed", exist_ok=True)
        
        report_content = f"""Отчет о качестве данных SOLUSDT
Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

СВОДНАЯ ИНФОРМАЦИЯ:
==================
Общее количество записей: {info['total_records']:,}
Первая полная строка: {info['first_complete_row']:,}
Доступных индикаторов: {len(info['available_features'])}/33
Записей для train: {info['train_records']:,}
Записей для validation: {info['val_records']:,}
Записей для test: {info['test_records']:,}

ПРОБЛЕМЫ:
=========
NaN значения: {'Есть' if info['has_nan'] else 'Нет'}
Inf значения: {'Есть' if info['has_inf'] else 'Нет'}  
Временные пропуски: {info['time_gaps']}

ДОСТУПНЫЕ ИНДИКАТОРЫ:
====================
{chr(10).join(f'✅ {feat}' for feat in info['available_features'])}

ОТСУТСТВУЮЩИЕ ИНДИКАТОРЫ:
=========================
{chr(10).join(f'❌ {feat}' for feat in info['missing_features']) if info['missing_features'] else 'Нет'}
"""
        
        with open("data/processed/data_quality_report.txt", "w", encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n💾 Отчет сохранен: data/processed/data_quality_report.txt")

if __name__ == "__main__":
    main()