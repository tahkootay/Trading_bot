# Взаимодействие модулей торговой системы

## Архитектура системы

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Collector │ => │   Indicators    │ => │    Strategy     │ => │   Backtester    │
│                 │    │                 │    │                 │    │                 │
│ SOLUSDT_5m.csv  │    │ RSI, EMA, BB    │    │ Trading Logic   │    │ Results.json    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Поток данных

### 1. Сбор данных (`modules/data_collector`)
```bash
python -m modules.data_collector --symbol SOLUSDT --timeframe 5m --period week
```
**Выходной формат:** `data/raw/SOLUSDT_5m.csv`
```csv
timestamp,open,high,low,close,volume,symbol
2024-01-01 00:00:00,150.1,151.5,149.8,150.7,2500.0,SOLUSDT
```

### 2. Технические индикаторы (`modules/indicators`)

#### Доступные индикаторы:
- **Скользящие средние:** SMA, EMA, WMA, DEMA, TEMA, HMA, VWMA
- **Осцилляторы:** RSI, MACD, Stochastic, Williams %R, CCI, Momentum, ROC  
- **Волатильность:** ATR, Bollinger Bands, Keltner Channels, Donchian Channels
- **Объём:** OBV, VWAP, A/D Line, Chaikin Money Flow, Volume Oscillator

#### Использование в стратегии:
```python
from modules.indicators import RSI, EMA, BollingerBands, SMA

# Инициализация в _initialize()
self.indicators = {
    'ema_fast': EMA(period=9),
    'ema_slow': EMA(period=21),
    'rsi': RSI(period=21),
    'bb': BollingerBands(period=20, std_dev=2.0),
    'volume_ma': SMA(period=20)
}

# Обновление в update_indicators()
def update_indicators(self, data: MarketData):
    self.state['ema_fast_val'] = self.indicators['ema_fast'].update(data.close)
    self.state['rsi_val'] = self.indicators['rsi'].update(data.close)
    self.state['bb_values'] = self.indicators['bb'].update(data.close)
```

### 3. Стратегия (`examples/strategies/`)

#### Базовый класс стратегии:
```python
from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import RSI, EMA, BollingerBands

class MyStrategy(StrategyBase):
    def _initialize(self):
        # Параметры стратегии
        self.parameters = {...}
        
        # Инициализация индикаторов
        self.indicators = {...}
        
        # Состояние стратегии
        self.state = {...}
    
    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        # 1. Обновить индикаторы
        self.update_indicators(data)
        
        # 2. Проверить готовность индикаторов
        if not self._indicators_ready():
            return TradeSignal(signal=Signal.HOLD)
        
        # 3. Генерировать сигналы
        return self._generate_entry_signal(data, position)
```

### 4. Бэктестер (`modules/backtester`)

```bash
python -m modules.backtester --strategy ./examples/strategies/sol_usdt_strategy.py --data ./data/raw/SOLUSDT_5m.csv
```

#### Процесс выполнения:
1. Загружает данные из CSV
2. Инициализирует стратегию  
3. Для каждого бара:
   - Создаёт `MarketData`
   - Вызывает `strategy.on_bar()`
   - Обрабатывает `TradeSignal`
   - Обновляет позиции и P&L
4. Форматирует и сохраняет результаты

**Выходной формат:** JSON с метриками и сделками

## Пример: SOL/USDT стратегия

### Конфигурация индикаторов:
```python
self.indicators = {
    'ema_fast': EMA(period=9),      # Быстрая EMA
    'ema_slow': EMA(period=21),     # Медленная EMA  
    'rsi': RSI(period=21),          # RSI с периодом 21
    'bb': BollingerBands(           # Полосы Боллинджера
        period=20, 
        std_dev=2.0
    ),
    'volume_ma': SMA(period=20)     # Средний объём
}
```

### Условия входа LONG:
```python
long_conditions = [
    current_price <= self.state['bb_values']['lower'],  # Цена у нижней полосы
    self.state['rsi_val'] <= 25,                        # RSI перепродано
    ema_ratio > (1 - 0.002),                           # EMA близко к пересечению
    volume_spike                                        # Повышенный объём
]
```

### Условия входа SHORT:
```python
short_conditions = [
    current_price >= self.state['bb_values']['upper'],  # Цена у верхней полосы
    self.state['rsi_val'] >= 75,                        # RSI перекуплено
    ema_ratio < (1 + 0.002),                           # EMA близко к пересечению
    volume_spike                                        # Повышенный объём
]
```

### Риск-менеджмент:
- **Стоп-лосс:** 1.5% от цены входа
- **Тейк-профит:** 2% (первый уровень), 3.5% (второй уровень)
- **Трейлинг стоп:** Активируется после 1.5% прибыли
- **Лимит времени:** Максимум 4 часа в позиции
- **Аварийный выход:** При объёме > 3x средний + неблагоприятное движение

## Команды для запуска

### 1. Сбор данных
```bash
python -m modules.data_collector --symbol SOLUSDT --timeframe 5m --period week
```

### 2. Бэктест стратегии
```bash
python -m modules.backtester --strategy ./examples/strategies/sol_usdt_strategy.py --data ./data/raw/SOLUSDT_5m.csv
```

### 3. Генерация отчёта  
```bash
python -m modules.reporter --results ./output/backtests/results.json
```

## Ключевые преимущества архитектуры

### ✅ Модульность
- Каждый модуль работает независимо
- Стандартизированные интерфейсы
- Легко заменить или обновить модуль

### ✅ Переиспользование кода
- Индикаторы доступны всем стратегиям
- Единый интерфейс бэктестера
- Стандартный формат данных

### ✅ Тестирование
- Каждый модуль можно тестировать отдельно
- Интеграционные тесты проверяют взаимодействие
- Воспроизводимые результаты

### ✅ Масштабируемость
- Легко добавить новые индикаторы
- Простое создание новых стратегий
- Поддержка различных источников данных

## Типичные проблемы и решения

### Проблема: Индикаторы не готовы
```python
def _indicators_ready(self) -> bool:
    return (
        self.indicators['ema_fast'].is_ready() and
        self.indicators['ema_slow'].is_ready() and
        self.indicators['rsi'].is_ready() and
        self.indicators['bb'].is_ready()
    )
```

### Проблема: Отсутствие данных
```python
# Проверка валидности данных
if data.close <= 0 or data.volume <= 0:
    return TradeSignal(signal=Signal.HOLD)
```

### Проблема: Переход между позициями
```python
# Правильная проверка направления позиции
if position.direction == "LONG":
    # Логика для длинной позиции
elif position.direction == "SHORT":
    # Логика для короткой позиции
else:  # position.direction == "NONE"
    # Логика для входа в позицию
```

## Дальнейшее развитие

1. **Добавление новых индикаторов** → `modules/indicators/`
2. **Создание новых стратегий** → `examples/strategies/`
3. **Интеграция с реальными биржами** → `modules/trading_bot/`
4. **Улучшение отчётности** → `modules/reporter/`
5. **Веб-интерфейс** → Отдельный веб-модуль

Система готова к использованию и дальнейшему расширению! 🚀