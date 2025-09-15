# 📈 Modular Trading System

Модульная торговая система для технического анализа и алгоритмической торговли.

## 🏗️ Архитектура

Четыре независимых модуля, взаимодействующих через файловые форматы (CSV, JSON):

```
modules/data_collector/  → CSV/JSON → modules/indicators/ → CSV → modules/extremes_analyzer/ → CSV
                              ↓                                         ↓
                        modules/backtester/ ←──────────────────→ modules/reporter/ → HTML
                              ↓
                        modules/trading_bot/ (placeholder)
```

## 🚀 Быстрый старт

### Сбор данных
```bash
# Данные за неделю (5m)
python -m modules.data_collector --symbol SOLUSDT --timeframe 5m --period week

# Данные за август 2025 (1h)
python -m modules.data_collector --symbol SOLUSDT --timeframe 1h --start 2025-08-01 --end 2025-08-31
```

### Расчет индикаторов
```bash
# Все индикаторы
python -m modules.indicators --input data/raw/SOLUSDT_1h_20250801_20250831.csv --output data/processed/SOLUSDT_indicators.csv

# Конкретные индикаторы
python -m modules.indicators --input data/raw/SOLUSDT_1h_20250801_20250831.csv --indicators RSI,MACD,BB --output data/processed/SOLUSDT_custom.csv
```

### Анализ экстремумов и обогащение данных
```bash
# Обогащение файла с индикаторами
python -m modules.extremes_analyzer --enrich-indicators --data data/processed/SOLUSDT_indicators.csv --symbol SOLUSDT --timeframe 1h

# Стандартный анализ экстремумов
python -m modules.extremes_analyzer --data data/raw/SOLUSDT_1h.csv --symbol SOLUSDT --timeframe 1h

# 🆕 Статистический анализ (Версия 2.0)
python -m modules.extremes_analyzer --statistical-analysis --data data/processed/SOLUSDT_enriched.csv --symbol SOLUSDT --timeframe 1h

# Статистический анализ с генерацией PDF/HTML отчетов
python -m modules.extremes_analyzer --statistical-analysis --data data/processed/SOLUSDT_enriched.csv --report-format both

# Анализ с фильтром по таймфрейму
python -m modules.extremes_analyzer --statistical-analysis --data data/processed/SOLUSDT_enriched.csv --timeframe-filter 1h
```

### Бэктестинг
```bash
python -m modules.backtester --strategy examples/strategies/bb_touch_strategy.py --data data/processed/SOLUSDT_enriched.csv
```

### Генерация отчетов
```bash
python -m modules.reporter --results output/backtests/strategy_results.json
```

## 📁 Структура проекта

```
📦 Trading_bot/
├── 📁 modules/                    # Основные модули
│   ├── 📁 data_collector/         # Модуль 1: Сбор данных
│   ├── 📁 indicators/             # Модуль расчета индикаторов  
│   ├── 📁 extremes_analyzer/      # Модуль анализа экстремумов
│   ├── 📁 backtester/             # Модуль 2: Бэктестинг
│   ├── 📁 reporter/               # Модуль 3: Отчеты
│   └── 📁 trading_bot/            # Модуль 4: Торговый бот (заглушка)
├── 📁 examples/strategies/        # Примеры стратегий
├── 📁 data/                       # Данные
│   ├── 📁 raw/                    # Исходные данные
│   └── 📁 processed/              # Обработанные данные
├── 📁 output/                     # Результаты
│   ├── 📁 backtests/              # Результаты бэктестов
│   ├── 📁 extremes/               # Результаты анализа экстремумов
│   └── 📁 examples/               # Примеры графиков
├── 📁 docs/                       # Документация
├── 📁 tests/                      # Тесты
├── 📁 scripts/                    # Вспомогательные скрипты
├── 📁 config/                     # Конфигурации
└── 📁 archive/                    # Архив устаревших файлов
```

## 🔧 Возможности

### 📊 Модуль сбора данных
- Сбор исторических данных с Bybit
- Поддержка множественных таймфреймов
- Автоматическое выравнивание временных меток
- Форматы вывода: CSV, JSON, Parquet

### 📈 Модуль индикаторов
- **Скользящие средние:** SMA, EMA, WMA, DEMA, TEMA, HMA, VWMA
- **Осцилляторы:** RSI, MACD, Stochastic, Williams %R, CCI
- **Волатильность:** ATR, Bollinger Bands, Keltner Channels, Donchian Channels
- **Объем:** OBV, VWAP, A/D Line, Chaikin Money Flow

### 🎯 Модуль анализа экстремумов (v2.0)
- **Обнаружение ценовых экстремумов** с настраиваемыми параметрами
- **Анализ экстремумов индикаторов:** RSI, Bollinger Bands, MACD
- **🆕 Статистический анализ:**
  - **Корреляционный анализ** с интерактивными heatmap
  - **Распределения экстремумов** и индикаторов
  - **Автоматические отчеты** в HTML/PDF формате
  - **Фильтры по таймфреймам**
- **Дополнительная аналитика:**
  - Моментум и изменения цены
  - Анализ объемов и всплесков активности
  - Волатильность и спреды
  - Уровни поддержки/сопротивления
  - Близость к ключевым уровням

### 🔄 Модуль бэктестинга
- Симуляция торговых стратегий
- Детальная статистика производительности
- Анализ рисков и просадок
- Поддержка множественных стратегий

### 📋 Модуль отчетности
- HTML отчеты с интерактивными графиками
- Визуализация результатов бэктестинга
- Анализ экстремумов с графиками
- Экспорт в различные форматы

## 📈 Пример обогащенных данных

После обработки модулем анализа экстремумов файл содержит:

**Исходные данные:** timestamp, open, high, low, close, volume, symbol, timeframe  
**Индикаторы:** rsi, sma, ema, macd_line, macd_signal, macd_histogram, atr, bb_upper, bb_middle, bb_lower  
**Экстремумы цены:** is_extreme, extreme_type, extreme_strength, extreme_price_change  
**Экстремумы индикаторов:** rsi_extreme, bb_extreme, macd_signal_change  
**Аналитические данные:** price_change_1h, volume_spike, high_low_spread, near_resistance, near_support  

## 🎯 Шаблон стратегии

```python
from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal

class MyStrategy(StrategyBase):
    def _initialize(self):
        self.parameters = {'period': 20, 'position_size_pct': 0.1}
        self.state = {'history': []}
    
    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        # Логика стратегии с доступом к обогащенным данным
        if data.is_extreme and data.extreme_type == 'min' and data.rsi_oversold:
            return TradeSignal(signal=Signal.BUY)
        elif data.bb_breakout_upper and data.volume_spike:
            return TradeSignal(signal=Signal.SELL)
        return TradeSignal(signal=Signal.HOLD)
```

## 🛠️ Требования

- Python 3.8+
- pandas, numpy, matplotlib
- pybit (для сбора данных)
- plotly (для интерактивных графиков)

## 📝 Установка

```bash
# Клонирование репозитория
git clone <repository_url>
cd Trading_bot

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Установка зависимостей
pip install -r requirements.txt
```

## 📋 Правила разработки

⚠️ **КРИТИЧЕСКИ ВАЖНО:**
1. **НЕ используйте импорты между модулями**
2. **Используйте файловую коммуникацию (JSON/CSV)**
3. **Каждый модуль работает автономно**
4. **Тестируйте модули независимо**

## 📚 Документация

- [Руководство по модулям](docs/modules_integration.md)
- [Структура стратегий](docs/strategy_structure.md) 
- [Архитектура системы](docs/NEW_MODULAR_ARCHITECTURE.md)
- [Технические спецификации экстремумов](docs/technical_spec_extremes.md)

## 🆕 Статистический анализ v2.0

### 📊 Новые возможности:

**Корреляционный анализ:**
- Матрица корреляций между всеми числовыми переменными
- Интерактивные heatmap с plotly
- Статические heatmap с matplotlib/seaborn
- Автоматическое выявление сильных корреляций (|r| > 0.5)

**Распределения данных:**
- Распределения силы экстремумов
- RSI на экстремумах vs обычных точках
- Boxplot сравнения индикаторов
- Pie chart типов экстремумов

**Автоматические отчеты:**
- HTML отчеты с встроенными графиками
- PDF отчеты с таблицами и статистикой
- Интерактивные элементы и навигация
- Адаптивный дизайн для мобильных устройств

**Фильтрация данных:**
- Фильтры по таймфреймам
- Настраиваемые параметры анализа
- Гибкие критерии отбора данных

### 📈 Выходные файлы:

```
📦 output/extremes/
├── 📁 statistical_analysis/
│   ├── correlation_heatmap.png          # Статическая heatmap
│   ├── correlation_heatmap_interactive.html  # Интерактивная heatmap
│   ├── 📁 distributions/
│   │   ├── extremes_distributions.png   # Распределения экстремумов
│   │   └── indicators_boxplots.png      # Boxplot индикаторов
│   └── statistical_analysis_YYYYMMDD_HHMMSS.json  # JSON с результатами
└── 📁 reports/
    ├── statistical_report_YYYYMMDD_HHMMSS.html    # HTML отчет
    └── statistical_report_YYYYMMDD_HHMMSS.pdf     # PDF отчет
```

## 🏷️ Версия

**v2.0.0** - Модульная архитектура с расширенным статистическим анализом