# 🗃️ Система хранения ML моделей

Эта документация объясняет, где и как хранятся обученные модели машинного обучения в торговой системе.

## 📂 Структура директории моделей

```
models/
├── random_forest/           # Модели Random Forest
│   ├── random_forest_v1.pkl
│   ├── random_forest_v2.pkl
│   └── ...
├── xgboost/                # Модели XGBoost
│   ├── xgboost_v1.pkl
│   └── ...
├── scalers/                # Нормализаторы данных
│   ├── scaler_v1.pkl
│   ├── scaler_v2.pkl
│   └── ...
└── metadata/               # Метаданные и важность признаков
    ├── metadata_v1.json
    ├── metadata_v2.json
    ├── feature_importance_v1.csv
    ├── feature_importance_v2.csv
    └── ...
```

## 📦 Состав сохранённой модели

Каждая версия модели состоит из **4 файлов**:

### 1. 🤖 Файл модели (`random_forest_v1.pkl`)
- **Содержит**: Полностью обученную модель со всеми параметрами
- **Формат**: Pickle (бинарный)
- **Размер**: 1-50 MB (зависит от сложности)

### 2. ⚖️ Файл нормализатора (`scaler_v1.pkl`)
- **Содержит**: StandardScaler для преобразования новых данных
- **Назначение**: Нормализация признаков перед предсказанием
- **Формат**: Pickle (бинарный)

### 3. 📋 Файл метаданных (`metadata_v1.json`)
- **Содержит**: Полную информацию о модели
- **Формат**: JSON (текстовый)
- **Включает**:
  ```json
  {
    "model_info": {
      "model_type": "random_forest",
      "version": "v1",
      "timestamp": "20251030_190220",
      "sklearn_version": "1.3.0",
      "python_version": "3.11.0"
    },
    "performance": {
      "train_accuracy": 0.823,
      "val_accuracy": 0.778,
      "test_accuracy": 0.765,
      "overfitting": 0.045
    },
    "hyperparameters": {
      "n_estimators": 150,
      "max_depth": 15,
      "min_samples_split": 20,
      "random_state": 42
    },
    "training": {
      "train_samples": 15000,
      "val_samples": 3000,
      "test_samples": 3000,
      "prediction_horizon": 3,
      "symbol": "SOLUSDT",
      "timeframe": "5m"
    },
    "features": {
      "count": 50,
      "columns": ["sma_20", "rsi_14", "bb_width", ...]
    }
  }
  ```

### 4. 📊 Файл важности признаков (`feature_importance_v1.csv`)
- **Содержит**: Важность каждого признака для модели
- **Формат**: CSV (текстовый)
- **Пример**:
  ```csv
  feature,importance
  sma_20,0.250
  rsi_14,0.220
  bb_width,0.180
  volume_sma,0.150
  ema_50,0.120
  ```

## 🔧 Как использовать систему

### 1. Сохранение модели
```python
from modules.data_collector.model_manager import ModelManager

manager = ModelManager()

# Сохраняем обученную модель
files = manager.save_model(
    model=trained_model,
    scaler=fitted_scaler,
    model_type="random_forest",
    version="v1",
    feature_columns=feature_list,
    performance_metrics=metrics,
    hyperparameters=params,
    training_info=info,
    feature_importance=importance_df
)
```

### 2. Загрузка модели
```python
# Загружаем модель для предсказаний
model, scaler, metadata = manager.load_model("v1")

print(f"Точность: {metadata['performance']['test_accuracy']}")
print(f"Признаков: {metadata['features']['count']}")
```

### 3. Предсказания на новых данных
```python
# Делаем предсказания
predictions = manager.predict(
    data=new_market_data,
    version="v1",
    return_probabilities=True
)

print(f"Предсказания: {predictions['predictions']}")
print(f"Вероятности: {predictions['probability_up']}")
```

### 4. Торговые сигналы
```python
# Генерируем торговые сигналы
signals = manager.get_trading_signals(
    data=market_data,
    version="v1",
    buy_threshold=0.6,
    sell_threshold=0.4
)

print(signals[['signal', 'probability_up', 'signal_strength']])
```

## 📋 Управление версиями моделей

### Просмотр всех моделей
```python
models_df = manager.list_models()
print(models_df)
```

Результат:
```
version model_type   timestamp    test_accuracy  val_accuracy  features_count
v3      random_forest 20251030_150000  0.782         0.775         55
v2      random_forest 20251030_120000  0.765         0.778         50  
v1      random_forest 20251030_100000  0.756         0.761         45
```

### Сравнение моделей
```python
comparison = manager.compare_models(["v1", "v2", "v3"])
print(comparison)
```

## 🎯 Практические примеры

### Пример 1: Быстрая проверка модели
```bash
# Запустить демонстрацию сохранения
python3 demo_model_storage.py

# Запустить демонстрацию использования
python3 demo_model_usage.py
```

### Пример 2: Реальное обучение и сохранение
```python
# 1. Подготовить данные
from modules.data_collector.ml_data_prep import MLDataPreparator
prep = MLDataPreparator()
train, val, test = prep.prepare_ml_data("data/SOLUSDT_5m_indicators.csv")

# 2. Обучить модель
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
model.fit(train[0], train[1])

# 3. Сохранить модель
manager = ModelManager()
manager.save_model(model, scaler, "random_forest", "v1", ...)
```

## 💡 Лучшие практики

### ✅ Рекомендации
1. **Версионирование**: Используйте понятные версии (v1, v2, v1.1)
2. **Метаданные**: Всегда сохраняйте полную информацию о модели
3. **Тестирование**: Проверяйте загруженную модель на тестовых данных
4. **Бэкапы**: Регулярно создавайте копии лучших моделей
5. **Документация**: Ведите журнал изменений между версиями

### ⚠️ Важные моменты
1. **Признаки**: Новые данные должны иметь те же признаки, что и при обучении
2. **Нормализация**: Всегда применяйте тот же scaler, что использовался при обучении
3. **Версии**: Следите за совместимостью версий библиотек (sklearn, Python)
4. **Размер**: Большие модели медленнее загружаются и занимают больше памяти

## 🚀 Интеграция с торговым ботом

```python
class TradingBot:
    def __init__(self):
        self.model_manager = ModelManager()
        # Загружаем лучшую модель при старте
        self.model, self.scaler, self.metadata = self.model_manager.load_model("v3")
    
    def get_trading_signal(self, current_data):
        # Получаем торговый сигнал от модели
        signals = self.model_manager.get_trading_signals(
            data=current_data,
            version="v3",
            buy_threshold=0.65,
            sell_threshold=0.35
        )
        return signals.iloc[-1]  # Последний сигнал
    
    def should_trade(self, signal):
        # Торгуем только при сильных сигналах
        return signal['signal_strength'] == 'STRONG'
```

## 🔍 Troubleshooting

### Проблема: "FileNotFoundError: metadata not found"
**Решение**: Убедитесь, что модель была сохранена с правильной версией

### Проблема: "Missing features: ['feature_name']"
**Решение**: Добавьте отсутствующие признаки в данные или пересчитайте индикаторы

### Проблема: "Model performance degraded"
**Решение**: Модель могла устареть, обучите новую на свежих данных

---

## 📞 Поддержка

Если у вас возникли вопросы по системе хранения моделей:
1. Проверьте примеры в `demo_model_storage.py` и `demo_model_usage.py`
2. Изучите код в `modules/data_collector/model_manager.py`
3. Убедитесь, что все зависимости установлены