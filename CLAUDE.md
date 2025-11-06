# Trading Bot - Структура Проекта

Система для сбора криптовалютных данных, расчета технических индикаторов и машинного обучения для предсказания цены.

## 📁 Структура Проекта

### src/ - Исходный код модулей
```
src/
├── data_collection/    # Сбор данных с Bybit API
├── indicators/         # Технические индикаторы (RSI, MACD, etc)
├── models/            # ML модели для предсказания цены
│   ├── lstm/          # LSTM модели на Keras/JAX
│   └── random_forest/ # Random Forest модели
├── backtesting/       # Бэктестирование торговых стратегий
└── utils/             # Общие функции и утилиты
```

### saved_models/ - Сохраненные модели
```
saved_models/
├── lstm/              # .keras файлы LSTM моделей
├── random_forest/     # .pkl файлы Random Forest моделей
└── metadata/          # Метаданные моделей
```

### scripts/ - Исполняемые скрипты
```
scripts/
├── training/          # Скрипты обучения моделей
├── backtesting/       # Скрипты бэктестирования  
├── data_processing/   # Обработка и подготовка данных
├── analysis/          # Анализ результатов
└── utils/             # Вспомогательные скрипты
```

### data/ - Данные
```
data/
├── raw/              # Исходные OHLCV данные
└── processed/        # Обработанные данные с индикаторами
```

### results/ - Результаты экспериментов
```
results/
├── backtests/        # Результаты бэктестов
├── reports/          # Отчеты и анализ
└── visualizations/   # Графики и визуализации
```

## 🎯 Основные Компоненты

### 1. Сбор Данных
- **Модуль**: `src/data_collection/`
- **Функция**: Получение исторических данных OHLCV с Bybit
- **Формат**: 5-минутные свечи SOLUSDT
- **Команда**: `python -m src.data_collection --symbol SOLUSDT --timeframe 5m`

### 2. Технические Индикаторы
- **Модуль**: `src/indicators/` (выделен из data_collection)
- **Функция**: Расчет RSI, MACD, Bollinger Bands, ATR, etc
- **Всего**: 33+ индикатора включая лаги

### 3. LSTM Модели
- **Модуль**: `src/models/lstm/`
- **Backend**: JAX/Keras для высокой производительности
- **Архитектура**: LSTM(64) → LSTM(32) → Dense → Sigmoid
- **Параметры**: Настраиваемые window_size и horizon

### 4. Random Forest Модели  
- **Модуль**: `src/models/random_forest/`
- **Библиотека**: scikit-learn
- **Особенности**: Feature importance analysis

### 5. Бэктестирование
- **Модуль**: `src/backtesting/`
- **Функция**: Симуляция торговли на исторических данных
- **Метрики**: Доходность, точность, Sharpe ratio

## ⚙️ Настраиваемые Параметры

### Горизонт Предсказания
- **Параметр**: `--horizon N`
- **Значения**: 3, 5, 7, 10 баров (по умолчанию 3)
- **Описание**: На сколько баров вперед предсказывать цену

### Размер Окна
- **Параметр**: `--window N` 
- **Значения**: 20, 30, 60 баров (по умолчанию 30)
- **Описание**: Сколько исторических баров использовать

### Тип Модели
- **Параметр**: `--model TYPE`
- **Значения**: `lstm`, `rf` (Random Forest)
- **Описание**: Какую модель использовать

## 🚀 Основные Команды

### Сбор Данных
```bash
# Сбор данных за неделю
python -m src.data_collection --symbol SOLUSDT --timeframe 5m --period week

# Добавление индикаторов к существующим данным
python -m src.indicators data/raw/SOLUSDT_5m.csv data/processed/SOLUSDT_5m_indicators.csv
```

### Обучение Моделей
```bash
# LSTM с горизонтом 5 баров
python scripts/training/train_lstm.py --horizon 5 --window 30

# Random Forest с горизонтом 3 бара  
python scripts/training/train_rf.py --horizon 3 --window 20
```

### Бэктестирование
```bash
# Бэктест LSTM модели
python scripts/backtesting/run_backtest.py --model lstm --horizon 3

# Бэктест с кастомными параметрами
python scripts/backtesting/run_backtest.py --model rf --horizon 5 --period "2025-08-01:2025-10-31"
```

### Анализ Результатов
```bash
# Анализ важности признаков
python scripts/analysis/analyze_feature_importance.py --model rf

# Сравнение всех моделей
python scripts/analysis/compare_all_models.py
```

## 📊 Данные для Обучения

### Инструмент
- **Пара**: SOLUSDT (Solana/USDT)
- **Источник**: Bybit API
- **Таймфрейм**: 5 минут

### Период
- **Основные данные**: Январь-Октябрь 2025
- **Обучение**: 70% (Янв-Июль)  
- **Валидация**: 15% (Август)
- **Тест**: 15% (Сен-Окт)

### Признаки
- **Количество**: 33 технических индикатора
- **Категории**: RSI, MACD, Bollinger Bands, Volume, Momentum
- **Лаги**: Включены лаги на 1 и 3 бара для key индикаторов

## 🎛️ Конфигурация

### Файлы настроек
- `config/default.yaml` - Основные настройки
- `config/indicators.yaml` - Параметры индикаторов
- `config/models.yaml` - Настройки моделей

### Переменные окружения
```bash
export BYBIT_API_KEY="your_api_key"
export BYBIT_API_SECRET="your_api_secret"
```

## 🔧 Разработка

### Добавление новой модели
1. Создать папку в `src/models/new_model/`
2. Реализовать интерфейсы train(), predict(), save(), load()
3. Добавить скрипты в `scripts/training/`
4. Обновить `scripts/backtesting/`

### Добавление нового индикатора
1. Реализовать функцию в `src/indicators/`
2. Добавить в список признаков модели
3. Обновить конфигурационные файлы

## 📈 Текущие Результаты

### LSTM Модель (Horizon 3)
- **Точность**: ~52.1% (тест)
- **Доходность**: -10.20% (Авг-Окт 2025)
- **Статус**: Требует доработки

### Проблемы
- Нестабильность между периодами
- Переторговля (слишком много сделок)
- Низкая генерализация

### Следующие шаги
- Улучшить feature engineering
- Добавить attention mechanisms
- Реализовать ensemble подходы
- Оптимизировать risk management

## 🛠️ Инструменты

### Основные библиотеки
- **ML**: Keras/JAX, scikit-learn
- **Данные**: pandas, numpy
- **API**: ccxt (для Bybit)
- **Визуализация**: matplotlib, seaborn

### Требования
- Python 3.8+
- Минимум 8GB RAM для LSTM
- CUDA для ускорения (опционально)

## 📝 Примечания

- Все параметры настраиваемы через CLI
- Горизонт предсказания не зафиксирован на 3 барах
- Модульная архитектура для независимой разработки
- Сохранение обратной совместимости с существующими моделями