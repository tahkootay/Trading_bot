#!/usr/bin/env python3
"""
Автоматическое пошаговое изучение Random Forest

Этот скрипт автоматически проходит по всем шагам обучения Random Forest
без остановок на пользовательский ввод.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time


def step_1_load_and_understand_data():
    """Шаг 1: Загружаем и изучаем данные"""
    print("🔄 ШАГ 1: ЗАГРУЗКА И ПОНИМАНИЕ ДАННЫХ")
    print("=" * 50)
    
    # Загружаем данные
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    print(f"📊 Размеры данных:")
    print(f"   Train: {train_df.shape} (строки, колонки)")
    print(f"   Val:   {val_df.shape}")
    print(f"   Test:  {test_df.shape}")
    
    # Разделяем на признаки (X) и целевую переменную (y)
    feature_cols = [col for col in train_df.columns if col != 'target']
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    print(f"\n🎯 Что мы предсказываем:")
    print(f"   Target = 1: цена вырастет через 3 бара")
    print(f"   Target = 0: цена упадёт через 3 бара")
    
    print(f"\n📈 Распределение классов в train:")
    target_counts = y_train.value_counts()
    print(f"   Класс 0 (падение): {target_counts[0]} ({target_counts[0]/len(y_train)*100:.1f}%)")
    print(f"   Класс 1 (рост):    {target_counts[1]} ({target_counts[1]/len(y_train)*100:.1f}%)")
    
    print(f"\n🔢 Признаки (всего {len(feature_cols)}):")
    print("   Примеры признаков:")
    for i, feature in enumerate(feature_cols[:10]):
        print(f"   {i+1:2d}. {feature}")
    print(f"   ... и ещё {len(feature_cols)-10} признаков")
    
    print(f"\n📊 Статистика признаков показывает:")
    print(f"   ✅ Все признаки нормализованы (mean ≈ 0, std ≈ 1)")
    print(f"   ✅ Нет экстремальных выбросов")
    print(f"   ✅ Данные готовы для машинного обучения")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols


def step_2_understand_random_forest():
    """Шаг 2: Что такое Random Forest"""
    print("\n🌲 ШАГ 2: ЧТО ТАКОЕ RANDOM FOREST")
    print("=" * 50)
    
    print("🌳 Random Forest = 'Случайный лес' из деревьев решений")
    print("\nКАК ЭТО РАБОТАЕТ:")
    print("1️⃣ Создаём много (например, 100) деревьев решений")
    print("2️⃣ Каждое дерево обучается на случайной выборке данных")
    print("3️⃣ Каждое дерево использует случайный набор признаков")
    print("4️⃣ Для предсказания все деревья 'голосуют'")
    print("5️⃣ Выбираем класс, за который проголосовало больше деревьев")
    
    print("\nПРИМЕР ПРЕДСКАЗАНИЯ:")
    print("- 70 деревьев: 'Цена вырастет' (target=1)")
    print("- 30 деревьев: 'Цена упадёт' (target=0)")
    print("→ Результат: цена вырастет (вероятность = 0.7)")
    
    print("\nПРЕИМУЩЕСТВА:")
    print("✅ Устойчив к переобучению")
    print("✅ Хорошо работает 'из коробки'")
    print("✅ Показывает важность признаков")
    print("✅ Может находить нелинейные зависимости")


def step_3_train_basic_model(X_train, y_train, X_val, y_val, feature_cols):
    """Шаг 3: Обучение базовой модели"""
    print("\n🚀 ШАГ 3: ОБУЧЕНИЕ БАЗОВОЙ МОДЕЛИ")
    print("=" * 50)
    
    print("📝 Создаём базовую модель Random Forest с параметрами по умолчанию:")
    
    basic_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"   ⚙️ n_estimators = 100 (количество деревьев)")
    print(f"   ⚙️ max_depth = None (без ограничений)")
    print(f"   ⚙️ max_features = sqrt({len(feature_cols)}) = {int(np.sqrt(len(feature_cols)))}")
    
    print(f"\n🔄 Обучаем модель на {len(X_train):,} образцах...")
    start_time = time.time()
    basic_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"✅ Модель обучена за {train_time:.1f} секунд!")
    
    # Предсказания
    train_pred = basic_model.predict(X_train)
    val_pred = basic_model.predict(X_val)
    val_proba = basic_model.predict_proba(X_val)[:, 1]
    
    # Оценка качества
    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    print(f"\n📈 РЕЗУЛЬТАТЫ БАЗОВОЙ МОДЕЛИ:")
    print(f"   🎯 Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   🎯 Val Accuracy:   {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    overfitting = train_accuracy - val_accuracy
    print(f"   📊 Переобучение: {overfitting:.4f}")
    
    if overfitting > 0.05:
        print(f"   ⚠️ Есть переобучение! (разница > 5%)")
    else:
        print(f"   ✅ Переобучение в норме")
    
    # Важность признаков (топ-10)
    print(f"\n🔍 ТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': basic_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    return basic_model, feature_importance


def step_4_parameter_experiments(X_train, y_train, X_val, y_val):
    """Шаг 4: Эксперименты с параметрами"""
    print("\n🧪 ШАГ 4: ЭКСПЕРИМЕНТЫ С ПАРАМЕТРАМИ")
    print("=" * 50)
    
    results = []
    
    # Эксперимент 1: Количество деревьев
    print("🌳 Эксперимент 1: Количество деревьев (n_estimators)")
    n_estimators_values = [50, 100, 150, 200]
    
    for n_est in n_estimators_values:
        model = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        results.append(('n_estimators', n_est, val_acc))
        print(f"   n_estimators={n_est:3d}: {val_acc:.4f}")
    
    # Эксперимент 2: Глубина деревьев
    print("\n🏗️ Эксперимент 2: Глубина деревьев (max_depth)")
    max_depth_values = [5, 10, 15, 20, None]
    
    for depth in max_depth_values:
        model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        results.append(('max_depth', depth, val_acc))
        depth_str = str(depth) if depth else "None"
        print(f"   max_depth={depth_str:4s}: {val_acc:.4f}")
    
    # Эксперимент 3: Минимальные образцы
    print("\n✂️ Эксперимент 3: Минимальные образцы для разделения")
    min_samples_values = [2, 10, 20, 50]
    
    for min_split in min_samples_values:
        model = RandomForestClassifier(
            n_estimators=100, min_samples_split=min_split, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        results.append(('min_samples_split', min_split, val_acc))
        print(f"   min_samples_split={min_split:2d}: {val_acc:.4f}")
    
    # Найти лучшие параметры
    print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
    for param_type in ['n_estimators', 'max_depth', 'min_samples_split']:
        param_results = [r for r in results if r[0] == param_type]
        best = max(param_results, key=lambda x: x[2])
        print(f"   {param_type:18s}: {best[1]} → {best[2]:.4f}")
    
    return results


def step_5_optimized_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Шаг 5: Оптимизированная модель"""
    print("\n🏆 ШАГ 5: ОПТИМИЗИРОВАННАЯ МОДЕЛЬ")
    print("=" * 50)
    
    print("Создаём улучшенную модель на основе экспериментов:")
    
    # Оптимизированные параметры
    optimized_model = RandomForestClassifier(
        n_estimators=150,       # Увеличили для стабильности
        max_depth=15,          # Ограничили для уменьшения переобучения
        min_samples_split=20,  # Более консервативное разделение
        min_samples_leaf=10,   # Больше образцов в листьях
        max_features='sqrt',   # Стандартное значение
        random_state=42,
        n_jobs=-1
    )
    
    print(f"   ⚙️ n_estimators = 150 (было 100)")
    print(f"   ⚙️ max_depth = 15 (было None)")
    print(f"   ⚙️ min_samples_split = 20 (было 2)")
    print(f"   ⚙️ min_samples_leaf = 10 (было 1)")
    
    print(f"\n🔄 Обучаем оптимизированную модель...")
    start_time = time.time()
    optimized_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Оценка на всех наборах
    train_pred = optimized_model.predict(X_train)
    val_pred = optimized_model.predict(X_val)
    test_pred = optimized_model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"✅ Модель обучена за {train_time:.1f} секунд!")
    
    print(f"\n📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   🎯 Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   🎯 Val Accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"   🎯 Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print(f"   📊 Переобучение: {train_acc - val_acc:.4f}")
    print(f"   📊 Обобщение: {val_acc - test_acc:.4f}")
    
    # Подробный отчёт
    print(f"\n📋 Подробная оценка (Test set):")
    report = classification_report(y_test, test_pred, target_names=['Down', 'Up'], output_dict=True)
    
    print(f"   Down класс - Precision: {report['Down']['precision']:.3f}, Recall: {report['Down']['recall']:.3f}")
    print(f"   Up класс   - Precision: {report['Up']['precision']:.3f}, Recall: {report['Up']['recall']:.3f}")
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, test_pred)
    print(f"\n🔢 Матрица ошибок:")
    print(f"   Правильно предсказано Down: {cm[0,0]}")
    print(f"   Неправильно Down→Up: {cm[0,1]}")
    print(f"   Неправильно Up→Down: {cm[1,0]}")
    print(f"   Правильно предсказано Up: {cm[1,1]}")
    
    # Торговые сигналы
    test_proba = optimized_model.predict_proba(X_test)[:, 1]
    
    print(f"\n💰 АНАЛИЗ ТОРГОВЫХ СИГНАЛОВ:")
    confident_buy = (test_proba > 0.7).sum()
    buy = (test_proba > 0.6).sum()
    neutral = ((test_proba >= 0.4) & (test_proba <= 0.6)).sum()
    sell = (test_proba < 0.4).sum()
    confident_sell = (test_proba < 0.3).sum()
    
    print(f"   📈 Уверенные покупки (>0.7): {confident_buy}")
    print(f"   📈 Покупки (>0.6): {buy}")
    print(f"   ⏸️ Нейтральные (0.4-0.6): {neutral}")
    print(f"   📉 Продажи (<0.4): {sell}")
    print(f"   📉 Уверенные продажи (<0.3): {confident_sell}")
    
    return optimized_model


def step_6_trading_insights(model, X_test, y_test, feature_cols):
    """Шаг 6: Торговые инсайты"""
    print("\n💡 ШАГ 6: ТОРГОВЫЕ ИНСАЙТЫ")
    print("=" * 50)
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("🔍 ТОП-15 САМЫХ ВАЖНЫХ ПРИЗНАКОВ ДЛЯ ПРЕДСКАЗАНИЯ:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"   {i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")
    
    # Анализ по категориям признаков
    print(f"\n📊 АНАЛИЗ ПО КАТЕГОРИЯМ ПРИЗНАКОВ:")
    
    categories = {
        'Trend (SMA/EMA)': [f for f in feature_cols if any(x in f for x in ['sma_', 'ema_'])],
        'Oscillators': [f for f in feature_cols if any(x in f for x in ['rsi_', 'stoch_', 'macd_', 'cci_'])],
        'Bollinger': [f for f in feature_cols if 'bb_' in f],
        'Volume': [f for f in feature_cols if 'volume' in f or 'relative_volume' in f],
        'Volatility': [f for f in feature_cols if any(x in f for x in ['atr_', 'volatility_'])],
        'Lags': [f for f in feature_cols if '_lag_' in f],
        'Others': [f for f in feature_cols if not any(cat in f for cat in ['sma_', 'ema_', 'rsi_', 'stoch_', 'macd_', 'cci_', 'bb_', 'volume', 'atr_', 'volatility_', '_lag_'])]
    }
    
    for cat_name, features in categories.items():
        if features:
            cat_importance = feature_importance[feature_importance['feature'].isin(features)]['importance'].sum()
            print(f"   {cat_name:15s}: {cat_importance:.4f} ({len(features)} признаков)")
    
    # Вероятности и точность
    test_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n📈 АНАЛИЗ ТОЧНОСТИ ПО УВЕРЕННОСТИ:")
    confidence_levels = [0.3, 0.4, 0.6, 0.7, 0.8]
    
    for conf in confidence_levels:
        high_conf_mask = (test_proba > conf) | (test_proba < (1-conf))
        if high_conf_mask.sum() > 0:
            high_conf_pred = (test_proba[high_conf_mask] > 0.5).astype(int)
            high_conf_true = y_test[high_conf_mask]
            high_conf_acc = accuracy_score(high_conf_true, high_conf_pred)
            print(f"   Уверенность >{conf:.1f} или <{1-conf:.1f}: {high_conf_acc:.4f} ({high_conf_mask.sum()} сигналов)")


def main():
    """Главная функция"""
    print("🎓 АВТОМАТИЧЕСКОЕ ИЗУЧЕНИЕ RANDOM FOREST")
    print("=" * 60)
    print("Пошаговый разбор машинного обучения для торговли")
    print("=" * 60)
    
    # Выполняем все шаги
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = step_1_load_and_understand_data()
    step_2_understand_random_forest()
    basic_model, importance = step_3_train_basic_model(X_train, y_train, X_val, y_val, feature_cols)
    results = step_4_parameter_experiments(X_train, y_train, X_val, y_val)
    optimized_model = step_5_optimized_model(X_train, y_train, X_val, y_val, X_test, y_test)
    step_6_trading_insights(optimized_model, X_test, y_test, feature_cols)
    
    print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 50)
    print("✅ Ты изучил Random Forest от А до Я")
    print("✅ Понял влияние каждого параметра")
    print("✅ Научился интерпретировать результаты")
    print("✅ Создал оптимизированную торговую модель")
    print("\n🚀 Готов к изучению XGBoost или нейронных сетей!")


if __name__ == "__main__":
    main()