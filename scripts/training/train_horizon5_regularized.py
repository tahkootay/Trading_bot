#!/usr/bin/env python3
"""
Обучение модели horizon_5 с улучшенной регуляризацией для борьбы с переобучением
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def create_target_5(df):
    """Создает target_5 - цена через 5 периодов выше текущей?"""
    print("🎯 Создание target_5...")
    
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Target_5: 1 если цена через 5 периодов выше текущей, 0 иначе
    df['target_5'] = 0
    for i in range(len(df) - 5):
        current_price = df.loc[i, 'close']
        future_price = df.loc[i + 5, 'close']
        df.loc[i, 'target_5'] = 1 if future_price > current_price else 0
    
    # Удаляем последние 5 строк (нет future price)
    df = df.iloc[:-5].copy()
    
    target_counts = df['target_5'].value_counts()
    print(f"   Target_5 распределение:")
    print(f"     UP (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"     DOWN (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    
    return df

def prepare_features(df):
    """Подготавливает 25 фичей для модели horizon_5"""
    print("🔧 Подготовка 25 фичей...")
    
    # 25 фичей для модели horizon_5
    required_features = [
        'volume_sma_20', 'atr_14', 'volatility_ratio', 'bb_width', 'ema_100',
        'ema_diff_10_50', 'rsi_14', 'stoch_d', 'ema_50', 'stoch_k', 
        'bb_position', 'slope_ema_20', 'momentum_3', 'macd_line', 'bb_upper',
        'macd_histogram', 'relative_volume', 'momentum_10', 'volume_change', 'cci_20',
        'rsi_14_lag_1', 'rsi_14_lag_3', 'bb_position_lag_1', 'bb_position_lag_3', 'momentum_3_lag_1'
    ]
    
    # Проверяем наличие фичей
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        print(f"   ❌ Отсутствующие фичи: {missing_features}")
        return None, None
    
    print(f"   ✅ Все 25 фичей присутствуют")
    
    # Извлекаем фичи и target
    X = df[required_features]
    y = df['target_5']
    
    # Проверяем на NaN
    if X.isnull().any().any():
        print("   ⚠️ Обнаружены NaN в фичах - удаляем строки с NaN")
        nan_mask = X.isnull().any(axis=1) | y.isnull()
        X = X[~nan_mask]
        y = y[~nan_mask]
    
    print(f"   📊 Финальный размер: {len(X)} образцов, {len(X.columns)} фичей")
    
    return X, y

def train_regularized_horizon5():
    """Обучение регуляризованной модели horizon_5"""
    print("🚀 ОБУЧЕНИЕ РЕГУЛЯРИЗОВАННОЙ МОДЕЛИ HORIZON_5")
    print("=" * 60)
    
    # 1. Загружаем обучающие данные
    print("1️⃣ Загрузка тренировочных данных...")
    train_file = "data/processed/train_jan_jun_2025_clean.csv"
    
    if not Path(train_file).exists():
        print(f"❌ Файл не найден: {train_file}")
        return None
    
    df_train = pd.read_csv(train_file)
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    
    print(f"   📊 Загружено: {len(df_train):,} записей")
    print(f"   📅 Период: {df_train['timestamp'].min()} → {df_train['timestamp'].max()}")
    
    # 2. Создаем target_5
    df_train = create_target_5(df_train)
    
    # 3. Подготавливаем фичи
    X_train, y_train = prepare_features(df_train)
    if X_train is None:
        return None
    
    # 4. Загружаем валидационные данные
    print("\n2️⃣ Загрузка валидационных данных...")
    val_file = "data/processed/validation_jul_2025.csv"
    
    if not Path(val_file).exists():
        print(f"❌ Файл не найден: {val_file}")
        return None
    
    df_val = pd.read_csv(val_file)
    df_val['timestamp'] = pd.to_datetime(df_val['timestamp'])
    df_val = create_target_5(df_val)
    X_val, y_val = prepare_features(df_val)
    
    if X_val is None:
        return None
    
    print(f"   📊 Валидация: {len(X_val):,} записей")
    
    # 5. Масштабирование данных
    print("\n3️⃣ Масштабирование данных...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"   ✅ Масштабирование завершено")
    
    # 6. Обучение нескольких моделей с разными параметрами
    print("\n4️⃣ Обучение моделей с разной степенью регуляризации...")
    
    models_config = [
        {
            'name': 'Conservative',
            'params': {
                'n_estimators': 50,
                'max_depth': 5,
                'min_samples_split': 100,
                'min_samples_leaf': 50,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            }
        },
        {
            'name': 'Moderate',
            'params': {
                'n_estimators': 100,
                'max_depth': 8,
                'min_samples_split': 50,
                'min_samples_leaf': 25,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            }
        },
        {
            'name': 'Balanced',
            'params': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 30,
                'min_samples_leaf': 15,
                'max_features': 0.7,
                'random_state': 42,
                'n_jobs': -1
            }
        }
    ]
    
    best_model = None
    best_score = 0
    best_config = None
    results = []
    
    for config in models_config:
        print(f"\n   🔧 Обучение {config['name']} модели...")
        print(f"      Параметры: {config['params']}")
        
        model = RandomForestClassifier(**config['params'])
        
        start_time = datetime.now()
        model.fit(X_train_scaled, y_train)
        train_time = (datetime.now() - start_time).total_seconds()
        
        # Оценка
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        overfitting = train_acc - val_acc
        
        print(f"      Train Accuracy: {train_acc:.4f}")
        print(f"      Val Accuracy:   {val_acc:.4f}")
        print(f"      Переобучение:   {overfitting:.4f}")
        print(f"      Время обучения: {train_time:.1f} сек")
        
        # Логирование в требуемом формате
        print(f"      Accuracy (train): {train_acc:.4f} Accuracy (val): {val_acc:.4f}")
        
        if overfitting > 0.05:
            print(f"      ⚠️ ПРЕДУПРЕЖДЕНИЕ: train accuracy > val accuracy + 5%")
        else:
            print(f"      ✅ Переобучение в норме")
        
        # Сохраняем результаты
        result = {
            'name': config['name'],
            'model': model,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'overfitting': overfitting,
            'train_time': train_time,
            'params': config['params']
        }
        results.append(result)
        
        # Выбираем лучшую модель (баланс val_acc и переобучения)
        score = val_acc - max(0, overfitting - 0.05) * 2  # Штраф за переобучение
        if score > best_score:
            best_score = score
            best_model = model
            best_config = result
    
    print(f"\n5️⃣ Лучшая модель: {best_config['name']}")
    print(f"   Train Accuracy: {best_config['train_acc']:.4f}")
    print(f"   Val Accuracy:   {best_config['val_acc']:.4f}")
    print(f"   Переобучение:   {best_config['overfitting']:.4f}")
    
    # Статистический анализ лучшей модели
    print(f"\n📊 Статистический анализ лучшей модели:")
    print(f"   Среднее значение признаков (train): {X_train.mean().mean():.4f}")
    print(f"   Среднее значение target (train): {y_train.mean():.4f}")
    print(f"   Среднее значение признаков (val): {X_val.mean().mean():.4f}")
    print(f"   Среднее значение target (val): {y_val.mean():.4f}")
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 ТОП-10 важных признаков:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    # 6. Сохранение лучшей модели
    print("\n6️⃣ Сохранение лучшей модели...")
    
    models_dir = Path("models/horizon5_regularized")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"regularized_{best_config['name'].lower()}_{timestamp}"
    
    model_file = models_dir / f"rf_horizon5_{version}.pkl"
    scaler_file = models_dir / f"scaler_horizon5_{version}.pkl"
    importance_file = models_dir / f"feature_importance_{version}.csv"
    
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    feature_importance.to_csv(importance_file, index=False)
    
    print(f"   ✅ Модель сохранена: {model_file}")
    print(f"   ✅ Scaler сохранен: {scaler_file}")
    print(f"   ✅ Важность признаков: {importance_file}")
    
    # Сохранение детального отчета
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    report_file = reports_dir / f"horizon5_regularized_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("ОТЧЁТ ОБ ОБУЧЕНИИ РЕГУЛЯРИЗОВАННОЙ МОДЕЛИ HORIZON_5\\n")
        f.write("=" * 60 + "\\n\\n")
        f.write(f"Время обучения: {timestamp}\\n")
        f.write(f"Данные: {train_file}\\n\\n")
        
        f.write("РЕЗУЛЬТАТЫ ВСЕХ МОДЕЛЕЙ:\\n")
        f.write("-" * 40 + "\\n")
        for result in results:
            f.write(f"\\n{result['name']} модель:\\n")
            f.write(f"  Train Accuracy: {result['train_acc']:.4f}\\n")
            f.write(f"  Val Accuracy: {result['val_acc']:.4f}\\n")
            f.write(f"  Переобучение: {result['overfitting']:.4f}\\n")
            f.write(f"  Параметры: {result['params']}\\n")
            if result['overfitting'] > 0.05:
                f.write(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Переобучение!\\n")
        
        f.write(f"\\n\\nЛУЧШАЯ МОДЕЛЬ: {best_config['name']}\\n")
        f.write(f"Итоговая Val Accuracy: {best_config['val_acc']:.4f}\\n")
        f.write(f"Итоговое переобучение: {best_config['overfitting']:.4f}\\n")
    
    print(f"   ✅ Отчёт сохранен: {report_file}")
    
    print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"📊 Лучшая модель: {best_config['name']}")
    print(f"📊 Val Accuracy: {best_config['val_acc']:.4f}")
    print(f"📊 Переобучение: {best_config['overfitting']:.4f}")
    
    return {
        'best_model_file': str(model_file),
        'best_scaler_file': str(scaler_file),
        'best_config': best_config,
        'all_results': results
    }

if __name__ == "__main__":
    try:
        result = train_regularized_horizon5()
        if result:
            print(f"\n✅ Регуляризованная модель обучена успешно!")
        else:
            print(f"\n❌ Ошибка при обучении модели")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()