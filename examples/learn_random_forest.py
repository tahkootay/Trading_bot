#!/usr/bin/env python3
"""
Пошаговое изучение Random Forest для торговых данных

Этот скрипт проходит по всем шагам обучения Random Forest:
1. Загрузка и понимание данных
2. Что такое Random Forest и как он работает
3. Параметры модели и их влияние
4. Обучение и интерпретация результатов
5. Тюнинг гиперпараметров
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any


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
    
    print(f"\n📊 Статистика признаков (первые 5):")
    print(X_train.iloc[:, :5].describe().round(3))
    
    input("\n⏸️ Нажмите Enter для продолжения...")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols


def step_2_understand_random_forest():
    """Шаг 2: Что такое Random Forest"""
    print("\n🌲 ШАГ 2: ЧТО ТАКОЕ RANDOM FOREST")
    print("=" * 50)
    
    print("""
🌳 Random Forest = "Случайный лес" из деревьев решений

КАК ЭТО РАБОТАЕТ:
1️⃣ Создаём много (например, 100) деревьев решений
2️⃣ Каждое дерево обучается на случайной выборке данных
3️⃣ Каждое дерево использует случайный набор признаков
4️⃣ Для предсказания все деревья "голосуют"
5️⃣ Выбираем класс, за который проголосовало больше деревьев

ПРИМЕР:
- Дерево 1: "Цена вырастет" (target=1)
- Дерево 2: "Цена упадёт"  (target=0)
- Дерево 3: "Цена вырастет" (target=1)
- ...
- Дерево 100: "Цена вырастет" (target=1)

Результат: 70 деревьев сказали "вырастет", 30 - "упадёт"
→ Предсказание: цена вырастет (target=1)

ПОЧЕМУ ЭТО ХОРОШО:
✅ Устойчив к переобучению (overfitting)
✅ Хорошо работает "из коробки" 
✅ Показывает важность признаков
✅ Может находить нелинейные зависимости
✅ Устойчив к выбросам и шуму

НЕДОСТАТКИ:
❌ Сложно интерпретировать отдельное предсказание
❌ Может быть медленным на больших данных
❌ Склонен к переобучению при очень глубоких деревьях
    """)
    
    input("⏸️ Нажмите Enter для продолжения...")


def step_3_key_parameters():
    """Шаг 3: Ключевые параметры Random Forest"""
    print("\n⚙️ ШАГ 3: КЛЮЧЕВЫЕ ПАРАМЕТРЫ RANDOM FOREST")
    print("=" * 50)
    
    print("""
🎛️ ОСНОВНЫЕ ПАРАМЕТРЫ ДЛЯ НАСТРОЙКИ:

1️⃣ n_estimators (количество деревьев)
   • По умолчанию: 100
   • Больше деревьев = более стабильные предсказания
   • Но дольше обучение и предсказание
   • Для торговли: попробуй 50-200

2️⃣ max_depth (максимальная глубина дерева)
   • По умолчанию: None (без ограничений)
   • Меньше глубина = меньше переобучение
   • Для торговли: попробуй 5-15

3️⃣ min_samples_split (мин. образцов для разделения)
   • По умолчанию: 2
   • Больше значение = проще деревья
   • Для торговли: попробуй 10-50

4️⃣ min_samples_leaf (мин. образцов в листе)
   • По умолчанию: 1
   • Больше значение = более консервативные предсказания
   • Для торговли: попробуй 5-20

5️⃣ max_features (количество признаков для каждого дерева)
   • По умолчанию: 'sqrt' (корень из общего количества)
   • 'sqrt': хорошо для классификации
   • Число: точное количество признаков
   • Для торговли: попробуй 'sqrt', 'log2' или число

6️⃣ random_state (воспроизводимость)
   • Устанавливаем для одинаковых результатов
   • Для торговли: любое число (например, 42)

🎯 СТРАТЕГИЯ ПОДБОРА ПАРАМЕТРОВ:
1. Начни с базовых параметров
2. Увеличивай n_estimators до стабилизации accuracy
3. Настраивай max_depth для баланса точность/переобучение
4. Подбирай min_samples_split и min_samples_leaf
5. Экспериментируй с max_features
    """)
    
    input("⏸️ Нажмите Enter для продолжения...")


def step_4_train_basic_model(X_train, y_train, X_val, y_val, feature_cols):
    """Шаг 4: Обучение базовой модели"""
    print("\n🚀 ШАГ 4: ОБУЧЕНИЕ БАЗОВОЙ МОДЕЛИ")
    print("=" * 50)
    
    print("📝 Создаём базовую модель Random Forest:")
    
    # Базовая модель с параметрами по умолчанию
    basic_model = RandomForestClassifier(
        n_estimators=100,        # 100 деревьев
        random_state=42,         # для воспроизводимости
        n_jobs=-1               # используем все процессоры
    )
    
    print(f"   ⚙️ n_estimators = 100 (количество деревьев)")
    print(f"   ⚙️ max_depth = None (без ограничений)")
    print(f"   ⚙️ min_samples_split = 2")
    print(f"   ⚙️ min_samples_leaf = 1")
    print(f"   ⚙️ max_features = 'sqrt' = {int(np.sqrt(len(feature_cols)))}")
    
    print(f"\n🔄 Обучаем модель на {len(X_train)} образцах...")
    basic_model.fit(X_train, y_train)
    
    print(f"✅ Модель обучена!")
    
    # Предсказания
    print(f"\n🔮 Делаем предсказания...")
    train_pred = basic_model.predict(X_train)
    val_pred = basic_model.predict(X_val)
    
    # Вероятности (важно для торговли!)
    train_proba = basic_model.predict_proba(X_train)[:, 1]  # Вероятность класса 1
    val_proba = basic_model.predict_proba(X_val)[:, 1]
    
    print(f"   📊 Предсказания: 0 или 1")
    print(f"   📊 Вероятности: от 0.0 до 1.0")
    
    # Оценка качества
    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    print(f"\n📈 РЕЗУЛЬТАТЫ:")
    print(f"   🎯 Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   🎯 Val Accuracy:   {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # Проверяем переобучение
    overfitting = train_accuracy - val_accuracy
    print(f"   📊 Разница (переобучение): {overfitting:.4f}")
    
    if overfitting > 0.05:
        print(f"   ⚠️ Возможно переобучение! (разница > 5%)")
    else:
        print(f"   ✅ Переобучение в норме")
    
    # Распределение вероятностей
    print(f"\n📊 Распределение вероятностей (val):")
    print(f"   Мин: {val_proba.min():.3f}")
    print(f"   Макс: {val_proba.max():.3f}")
    print(f"   Медиана: {np.median(val_proba):.3f}")
    
    # Важность признаков (топ-10)
    print(f"\n🔍 ТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': basic_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    input("\n⏸️ Нажмите Enter для продолжения...")
    
    return basic_model, feature_importance


def step_5_parameter_experiments(X_train, y_train, X_val, y_val, feature_cols):
    """Шаг 5: Эксперименты с параметрами"""
    print("\n🧪 ШАГ 5: ЭКСПЕРИМЕНТЫ С ПАРАМЕТРАМИ")
    print("=" * 50)
    
    results = []
    
    print("Тестируем разные комбинации параметров...")
    print("(Это может занять несколько минут)\n")
    
    # Эксперимент 1: Количество деревьев
    print("🌳 Эксперимент 1: Количество деревьев (n_estimators)")
    n_estimators_values = [50, 100, 150, 200]
    
    for n_est in n_estimators_values:
        model = RandomForestClassifier(
            n_estimators=n_est,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        
        results.append({
            'experiment': 'n_estimators',
            'parameter': f'n_estimators={n_est}',
            'value': n_est,
            'val_accuracy': val_acc
        })
        
        print(f"   n_estimators={n_est:3d}: {val_acc:.4f}")
    
    # Эксперимент 2: Глубина деревьев
    print("\n🏗️ Эксперимент 2: Глубина деревьев (max_depth)")
    max_depth_values = [5, 10, 15, 20, None]
    
    for depth in max_depth_values:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        
        results.append({
            'experiment': 'max_depth',
            'parameter': f'max_depth={depth}',
            'value': depth if depth else 999,
            'val_accuracy': val_acc
        })
        
        depth_str = str(depth) if depth else "None"
        print(f"   max_depth={depth_str:4s}: {val_acc:.4f}")
    
    # Эксперимент 3: Минимальные образцы для разделения
    print("\n✂️ Эксперимент 3: Минимальные образцы для разделения (min_samples_split)")
    min_samples_split_values = [2, 10, 20, 50]
    
    for min_split in min_samples_split_values:
        model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=min_split,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        
        results.append({
            'experiment': 'min_samples_split',
            'parameter': f'min_samples_split={min_split}',
            'value': min_split,
            'val_accuracy': val_acc
        })
        
        print(f"   min_samples_split={min_split:2d}: {val_acc:.4f}")
    
    # Находим лучшие параметры для каждого эксперимента
    print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
    results_df = pd.DataFrame(results)
    
    for exp in ['n_estimators', 'max_depth', 'min_samples_split']:
        exp_results = results_df[results_df['experiment'] == exp]
        best = exp_results.loc[exp_results['val_accuracy'].idxmax()]
        print(f"   {exp:18s}: {best['parameter']:20s} → {best['val_accuracy']:.4f}")
    
    input("\n⏸️ Нажмите Enter для продолжения...")
    
    return results_df


def step_6_best_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """Шаг 6: Создание лучшей модели"""
    print("\n🏆 ШАГ 6: СОЗДАНИЕ ЛУЧШЕЙ МОДЕЛИ")
    print("=" * 50)
    
    print("На основе экспериментов создаём оптимизированную модель:")
    
    # Оптимизированные параметры (можешь изменить на основе экспериментов)
    best_model = RandomForestClassifier(
        n_estimators=150,        # Больше деревьев для стабильности
        max_depth=10,           # Ограничиваем глубину против переобучения
        min_samples_split=20,   # Более консервативное разделение
        min_samples_leaf=10,    # Больше образцов в листьях
        max_features='sqrt',    # Стандартное значение для классификации
        random_state=42,
        n_jobs=-1
    )
    
    print(f"   ⚙️ n_estimators = 150")
    print(f"   ⚙️ max_depth = 10")
    print(f"   ⚙️ min_samples_split = 20")
    print(f"   ⚙️ min_samples_leaf = 10")
    print(f"   ⚙️ max_features = 'sqrt'")
    
    print(f"\n🔄 Обучаем оптимизированную модель...")
    best_model.fit(X_train, y_train)
    
    # Оценка на всех наборах
    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   🎯 Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   🎯 Val Accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"   🎯 Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print(f"\n📋 Подробный отчёт (Test set):")
    print(classification_report(y_test, test_pred, 
                              target_names=['Down (0)', 'Up (1)']))
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, test_pred)
    print(f"\n🔢 Матрица ошибок (Test set):")
    print(f"                 Предсказано")
    print(f"                 Down    Up")
    print(f"   Реально Down  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"   Реально Up    {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Торговые сигналы
    test_proba = best_model.predict_proba(X_test)[:, 1]
    
    print(f"\n💰 ТОРГОВЫЕ СИГНАЛЫ:")
    strong_buy = (test_proba > 0.7).sum()
    buy = (test_proba > 0.6).sum()
    sell = (test_proba < 0.4).sum()
    strong_sell = (test_proba < 0.3).sum()
    
    print(f"   📈 Strong Buy (prob > 0.7): {strong_buy}")
    print(f"   📈 Buy (prob > 0.6):        {buy}")
    print(f"   📉 Sell (prob < 0.4):       {sell}")
    print(f"   📉 Strong Sell (prob < 0.3): {strong_sell}")
    
    return best_model


def main():
    """Главная функция - пошаговое изучение Random Forest"""
    print("🎓 ИЗУЧАЕМ RANDOM FOREST ПОШАГОВО")
    print("=" * 60)
    print("Этот скрипт поможет понять каждый шаг машинного обучения")
    print("от загрузки данных до создания торговой модели.")
    print("=" * 60)
    
    # Шаг 1: Загрузка данных
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = step_1_load_and_understand_data()
    
    # Шаг 2: Теория Random Forest
    step_2_understand_random_forest()
    
    # Шаг 3: Параметры
    step_3_key_parameters()
    
    # Шаг 4: Базовая модель
    basic_model, importance = step_4_train_basic_model(X_train, y_train, X_val, y_val, feature_cols)
    
    # Шаг 5: Эксперименты
    results = step_5_parameter_experiments(X_train, y_train, X_val, y_val, feature_cols)
    
    # Шаг 6: Лучшая модель
    best_model = step_6_best_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
    
    print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 50)
    print("Теперь ты понимаешь:")
    print("✅ Как работает Random Forest")
    print("✅ Какие параметры можно настраивать")
    print("✅ Как интерпретировать результаты")
    print("✅ Как использовать модель для торговли")
    print("\nСледующий шаг: изучить XGBoost или Neural Networks!")


if __name__ == "__main__":
    main()