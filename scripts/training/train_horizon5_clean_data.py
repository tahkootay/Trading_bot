#!/usr/bin/env python3
"""
Обучение модели horizon_5 на очищенных данных train_jan_jun_2025_clean.csv
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

def train_horizon5_model():
    """Обучение модели horizon_5"""
    print("🚀 ОБУЧЕНИЕ МОДЕЛИ HORIZON_5")
    print("=" * 50)
    
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
    
    # 6. Обучение модели
    print("\n4️⃣ Обучение Random Forest модели...")
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"   ⚙️ Параметры модели:")
    print(f"      n_estimators: {model.n_estimators}")
    print(f"      max_depth: {model.max_depth}")
    print(f"      min_samples_split: {model.min_samples_split}")
    print(f"      min_samples_leaf: {model.min_samples_leaf}")
    
    start_time = datetime.now()
    model.fit(X_train_scaled, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    
    print(f"   ✅ Модель обучена за {train_time:.1f} секунд")
    
    # 7. Оценка модели
    print("\n5️⃣ Оценка качества модели...")
    
    # Предсказания
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    # Точности
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    # Логирование в требуемом формате
    print(f"Accuracy (train): {train_acc:.4f} Accuracy (val): {val_acc:.4f}")
    
    print(f"\n📊 Результаты:")
    print(f"   Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Val Accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    overfitting = train_acc - val_acc
    print(f"   Переобучение:   {overfitting:.4f}")
    
    if overfitting > 0.05:
        print("   ⚠️ ПРЕДУПРЕЖДЕНИЕ: train accuracy > val accuracy + 5%")
    else:
        print("   ✅ Переобучение в норме")
    
    # Статистический анализ
    print(f"\n📊 Статистический анализ:")
    print(f"   Среднее значение признаков (train): {X_train.mean().mean():.4f}")
    print(f"   Среднее значение target (train): {y_train.mean():.4f}")
    print(f"   Среднее значение признаков (val): {X_val.mean().mean():.4f}")
    print(f"   Среднее значение target (val): {y_val.mean():.4f}")
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 ТОП-10 важных признаков:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    # 8. Сохранение модели
    print("\n6️⃣ Сохранение модели...")
    
    # Создаем директории
    models_dir = Path("models/horizon5_jan_jun")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"jan_jun_{timestamp}"
    
    # Сохраняем модель
    model_file = models_dir / f"rf_horizon5_{version}.pkl"
    scaler_file = models_dir / f"scaler_horizon5_{version}.pkl"
    importance_file = models_dir / f"feature_importance_{version}.csv"
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    feature_importance.to_csv(importance_file, index=False)
    
    print(f"   ✅ Модель сохранена: {model_file}")
    print(f"   ✅ Scaler сохранен: {scaler_file}")
    print(f"   ✅ Важность признаков: {importance_file}")
    
    # Сохранение отчета
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    report_file = reports_dir / f"horizon5_training_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("ОТЧЁТ ОБ ОБУЧЕНИИ МОДЕЛИ HORIZON_5\\n")
        f.write("=" * 50 + "\\n\\n")
        f.write(f"Время обучения: {timestamp}\\n")
        f.write(f"Данные: {train_file}\\n")
        f.write(f"Период обучения: {df_train['timestamp'].min()} - {df_train['timestamp'].max()}\\n\\n")
        
        f.write("МЕТРИКИ ТОЧНОСТИ:\\n")
        f.write(f"Train Accuracy: {train_acc:.4f}\\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\\n")
        f.write(f"Переобучение: {overfitting:.4f}\\n\\n")
        
        if overfitting > 0.05:
            f.write("⚠️ ПРЕДУПРЕЖДЕНИЕ: Обнаружено переобучение!\\n\\n")
        
        f.write("СТАТИСТИКИ ДАННЫХ:\\n")
        f.write(f"Образцов для обучения: {len(X_train)}\\n")
        f.write(f"Образцов для валидации: {len(X_val)}\\n")
        f.write(f"Признаков: {len(X_train.columns)}\\n")
        f.write(f"Время обучения: {train_time:.1f} сек\\n")
    
    print(f"   ✅ Отчёт сохранен: {report_file}")
    
    print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    
    return {
        'model_file': str(model_file),
        'scaler_file': str(scaler_file),
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'overfitting': overfitting,
        'training_time': train_time
    }

if __name__ == "__main__":
    try:
        result = train_horizon5_model()
        if result:
            print(f"\n✅ Модель обучена успешно!")
            print(f"📊 Train Accuracy: {result['train_accuracy']:.4f}")
            print(f"📊 Val Accuracy: {result['val_accuracy']:.4f}")
            print(f"📊 Переобучение: {result['overfitting']:.4f}")
        else:
            print(f"\n❌ Ошибка при обучении модели")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()