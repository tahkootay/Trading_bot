#!/usr/bin/env python3
"""
LSTM Compatibility Test Script

Тестирует совместимость подготовленных данных с LSTM:
- Проверка создания последовательностей
- Тест нормализации данных
- Валидация размерностей
- Проверка целевых переменных
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import os

def load_clean_data():
    """Загрузка чистых данных"""
    print("📂 Загрузка чистых данных...")
    
    # Загрузка метаданных
    with open("data/processed/split_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Загрузка файлов
    train_df = pd.read_csv("data/processed/train_2025_clean.csv")
    val_df = pd.read_csv("data/processed/validation_2025_clean.csv") 
    test_df = pd.read_csv("data/processed/test_2025_clean.csv")
    
    print(f"✅ Train: {train_df.shape}")
    print(f"✅ Validation: {val_df.shape}")
    print(f"✅ Test: {test_df.shape}")
    
    return train_df, val_df, test_df, metadata

def test_feature_availability(train_df):
    """Тест доступности LSTM признаков"""
    print("\n🔍 Проверка доступности LSTM признаков...")
    
    # Список необходимых признаков
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
    
    available_features = [f for f in lstm_features if f in train_df.columns]
    missing_features = [f for f in lstm_features if f not in train_df.columns]
    
    print(f"✅ Доступных признаков: {len(available_features)}/{len(lstm_features)}")
    
    if missing_features:
        print("❌ Отсутствующие признаки:")
        for feat in missing_features:
            print(f"   - {feat}")
        return False, available_features
    else:
        print("✅ Все необходимые признаки присутствуют")
        return True, available_features

def test_target_variables(train_df, val_df, test_df, horizons=[3, 5, 7]):
    """Тест целевых переменных"""
    print(f"\n🎯 Проверка целевых переменных для горизонтов {horizons}...")
    
    all_good = True
    
    for horizon in horizons:
        target_col = f'target_h{horizon}'
        print(f"\n📊 Горизонт {horizon}:")
        
        for split_name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if target_col not in df.columns:
                print(f"   ❌ {split_name}: отсутствует {target_col}")
                all_good = False
                continue
                
            # Статистика
            target_series = df[target_col]
            valid_count = target_series.notna().sum()
            nan_count = target_series.isna().sum()
            
            if valid_count == 0:
                print(f"   ❌ {split_name}: нет валидных значений в {target_col}")
                all_good = False
                continue
                
            # Распределение классов
            up_count = (target_series == 1).sum()
            down_count = (target_series == 0).sum()
            up_ratio = up_count / valid_count * 100
            
            print(f"   ✅ {split_name}: {valid_count:,} валидных, {nan_count} NaN")
            print(f"      UP: {up_count:,} ({up_ratio:.1f}%), DOWN: {down_count:,} ({100-up_ratio:.1f}%)")
            
            # Проверка балансировки
            if abs(up_ratio - 50) > 10:  # Больше 60% или меньше 40%
                print(f"   ⚠️  {split_name}: дисбаланс классов!")
    
    return all_good

def create_sequences(data, targets, window_size=30):
    """Создание последовательностей для LSTM"""
    X, y = [], []
    
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(targets[i])
    
    return np.array(X), np.array(y)

def test_sequence_creation(train_df, available_features, window_sizes=[20, 30, 60]):
    """Тест создания последовательностей"""
    print(f"\n🔄 Тест создания последовательностей для window_size: {window_sizes}...")
    
    # Подготовка данных
    feature_data = train_df[available_features].values
    target_data = train_df['target_h3'].values
    
    all_good = True
    results = {}
    
    for window_size in window_sizes:
        try:
            print(f"\n📏 Window size: {window_size}")
            
            # Создание последовательностей
            X, y = create_sequences(feature_data, target_data, window_size)
            
            print(f"   ✅ Входные данные: {feature_data.shape}")
            print(f"   ✅ Последовательности X: {X.shape}")
            print(f"   ✅ Целевые значения y: {y.shape}")
            
            # Проверки
            expected_samples = len(feature_data) - window_size
            if len(X) != expected_samples:
                print(f"   ❌ Неверное количество последовательностей: {len(X)} != {expected_samples}")
                all_good = False
                continue
                
            if X.shape != (expected_samples, window_size, len(available_features)):
                print(f"   ❌ Неверная размерность X: {X.shape}")
                all_good = False
                continue
                
            if y.shape != (expected_samples,):
                print(f"   ❌ Неверная размерность y: {y.shape}")
                all_good = False
                continue
                
            # Проверка на NaN/inf
            nan_count = np.isnan(X).sum()
            inf_count = np.isinf(X).sum()
            
            if nan_count > 0:
                print(f"   ⚠️  Найдено {nan_count} NaN значений в последовательностях")
            if inf_count > 0:
                print(f"   ⚠️  Найдено {inf_count} Inf значений в последовательностях")
            
            results[window_size] = {
                'X_shape': X.shape,
                'y_shape': y.shape,
                'nan_count': int(nan_count),
                'inf_count': int(inf_count)
            }
            
            print(f"   ✅ Window size {window_size} совместим с LSTM")
            
        except Exception as e:
            print(f"   ❌ Ошибка для window_size {window_size}: {e}")
            all_good = False
    
    return all_good, results

def test_normalization(train_df, val_df, test_df, available_features):
    """Тест нормализации данных"""
    print(f"\n📊 Тест нормализации данных...")
    
    try:
        # Подготовка данных
        train_features = train_df[available_features].values
        val_features = val_df[available_features].values
        test_features = test_df[available_features].values
        
        print(f"✅ Train features: {train_features.shape}")
        print(f"✅ Val features: {val_features.shape}")
        print(f"✅ Test features: {test_features.shape}")
        
        # Инициализация скейлера
        scaler = StandardScaler()
        
        # Обучение ТОЛЬКО на train данных
        scaler.fit(train_features)
        print("✅ Скейлер обучен на train данных")
        
        # Нормализация всех наборов
        train_scaled = scaler.transform(train_features)
        val_scaled = scaler.transform(val_features) 
        test_scaled = scaler.transform(test_features)
        
        print("✅ Нормализация применена ко всем наборам")
        
        # Проверки результатов нормализации
        print("\n📈 Статистика нормализованных данных:")
        
        # Train должен иметь mean≈0, std≈1
        train_mean = np.mean(train_scaled, axis=0)
        train_std = np.std(train_scaled, axis=0)
        
        mean_check = np.abs(train_mean).max() < 0.01  # Среднее близко к 0
        std_check = np.abs(train_std - 1).max() < 0.01  # Стандартное отклонение близко к 1
        
        print(f"   Train mean (max abs): {np.abs(train_mean).max():.6f} ({'✅' if mean_check else '❌'})")
        print(f"   Train std (max dev from 1): {np.abs(train_std - 1).max():.6f} ({'✅' if std_check else '❌'})")
        
        # Проверка на NaN/Inf после нормализации
        for name, data in [('Train', train_scaled), ('Val', val_scaled), ('Test', test_scaled)]:
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            
            status = "✅" if nan_count == 0 and inf_count == 0 else "❌"
            print(f"   {name}: {nan_count} NaN, {inf_count} Inf {status}")
        
        normalization_ok = mean_check and std_check
        return normalization_ok, (train_scaled, val_scaled, test_scaled)
        
    except Exception as e:
        print(f"❌ Ошибка нормализации: {e}")
        return False, None

def test_full_pipeline(train_df, val_df, available_features, horizon=3, window_size=30):
    """Полный тест пайплайна подготовки данных для LSTM"""
    print(f"\n🚀 Полный тест пайплайна (horizon={horizon}, window={window_size})...")
    
    try:
        # 1. Подготовка данных
        target_col = f'target_h{horizon}'
        
        train_features = train_df[available_features].values
        val_features = val_df[available_features].values
        
        train_targets = train_df[target_col].values
        val_targets = val_df[target_col].values
        
        print(f"✅ Данные подготовлены: {len(available_features)} признаков")
        
        # 2. Нормализация
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        val_scaled = scaler.transform(val_features)
        
        print("✅ Нормализация выполнена")
        
        # 3. Создание последовательностей
        X_train, y_train = create_sequences(train_scaled, train_targets, window_size)
        X_val, y_val = create_sequences(val_scaled, val_targets, window_size)
        
        print(f"✅ Последовательности созданы:")
        print(f"   Train: X{X_train.shape}, y{y_train.shape}")
        print(f"   Val: X{X_val.shape}, y{y_val.shape}")
        
        # 4. Финальные проверки
        input_shape = (window_size, len(available_features))
        print(f"✅ Input shape для LSTM: {input_shape}")
        
        # Проверка балансировки
        train_up_ratio = np.mean(y_train) * 100
        val_up_ratio = np.mean(y_val) * 100
        
        print(f"✅ Балансировка:")
        print(f"   Train UP: {train_up_ratio:.1f}%")
        print(f"   Val UP: {val_up_ratio:.1f}%")
        
        return True, {
            'input_shape': input_shape,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'train_up_ratio': train_up_ratio,
            'val_up_ratio': val_up_ratio
        }
        
    except Exception as e:
        print(f"❌ Ошибка полного пайплайна: {e}")
        return False, None

def main():
    """Основная функция тестирования совместимости"""
    print("🧪 Тестирование совместимости данных с LSTM")
    print("="*60)
    
    try:
        # 1. Загрузка данных
        train_df, val_df, test_df, metadata = load_clean_data()
        
        # 2. Тест признаков
        features_ok, available_features = test_feature_availability(train_df)
        
        # 3. Тест целевых переменных
        targets_ok = test_target_variables(train_df, val_df, test_df)
        
        # 4. Тест создания последовательностей
        sequences_ok, sequence_results = test_sequence_creation(train_df, available_features)
        
        # 5. Тест нормализации
        normalization_ok, scaled_data = test_normalization(train_df, val_df, test_df, available_features)
        
        # 6. Полный пайплайн тест
        pipeline_ok, pipeline_results = test_full_pipeline(train_df, val_df, available_features)
        
        # Финальная оценка
        print("\n" + "="*60)
        print("🏁 ИТОГОВАЯ ОЦЕНКА СОВМЕСТИМОСТИ")
        print("="*60)
        
        all_tests = [features_ok, targets_ok, sequences_ok, normalization_ok, pipeline_ok]
        test_names = ['Признаки', 'Целевые переменные', 'Последовательности', 'Нормализация', 'Полный пайплайн']
        
        for name, result in zip(test_names, all_tests):
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{name:20}: {status}")
        
        overall_status = all(all_tests)
        
        print(f"\n🎯 ОБЩИЙ СТАТУС: {'✅ ГОТОВО К ОБУЧЕНИЮ LSTM' if overall_status else '❌ ТРЕБУЕТ ДОРАБОТКИ'}")
        
        if overall_status:
            print("\n🚀 Рекомендации для обучения:")
            print(f"   • Признаков: {len(available_features)}")
            print(f"   • Input shape: {pipeline_results['input_shape']}")
            print(f"   • Train samples: {pipeline_results['train_samples']:,}")
            print(f"   • Val samples: {pipeline_results['val_samples']:,}")
            print(f"   • Балансировка: Train {pipeline_results['train_up_ratio']:.1f}%, Val {pipeline_results['val_up_ratio']:.1f}%")
        
        return overall_status
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)