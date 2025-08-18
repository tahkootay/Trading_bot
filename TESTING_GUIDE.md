# 🧪 Руководство по тестированию торгового бота

## 📋 Содержание
- [Быстрый старт](#быстрый-старт)
- [Работа с блоками данных](#работа-с-блоками-данных)
- [Тестирование стратегий](#тестирование-стратегий)
- [Анализ результатов](#анализ-результатов)
- [Отладка и диагностика](#отладка-и-диагностика)
- [Команды и параметры](#команды-и-параметры)

---

## 🚀 Быстрый старт

### 1. Создание блоков данных
```bash
# Создать все предустановленные блоки
python scripts/manage_data_blocks.py create-predefined

# Проверить созданные блоки
python scripts/manage_data_blocks.py list-blocks
```

### 2. Базовый тест стратегии
```bash
# Тест на одном дне (быстро)
python scripts/enhanced_backtest.py --block-id august_12_single_day

# Тест на полной неделе
python scripts/enhanced_backtest.py --block-id august_10_17_full --save-results
```

### 3. Тест улучшенной стратегии
```bash
# Агрессивная momentum стратегия
python scripts/final_aggressive_strategy.py
```

---

## 📦 Работа с блоками данных

### Предустановленные блоки

#### `august_10_17_full` - Полная неделя
- **Период**: 10-17 августа 2025 (7 дней)
- **Тип**: Mixed (смешанные условия)
- **Применение**: Комплексное тестирование стратегии
- **Размер**: ~2304 свечей (5m)

```bash
# Информация о блоке
python scripts/manage_data_blocks.py info august_10_17_full

# Тест на блоке
python scripts/enhanced_backtest.py --block-id august_10_17_full
```

#### `august_10_13_trend` - Трендовый период
- **Период**: 10-13 августа 2025 (4 дня)
- **Тип**: Trend (трендовый)
- **Применение**: Тестирование трендовых стратегий

```bash
python scripts/enhanced_backtest.py --block-id august_10_13_trend
```

#### `august_14_17_volatile` - Волатильный период
- **Период**: 14-17 августа 2025 (4 дня)
- **Тип**: Volatile (волатильный)
- **Применение**: Стресс-тестирование риск-менеджмента

```bash
python scripts/enhanced_backtest.py --block-id august_14_17_volatile
```

#### `august_12_single_day` - Один день
- **Период**: 12 августа 2025 (24 часа)
- **Тип**: Mixed (смешанный)
- **Применение**: Быстрая отладка и тестирование
- **Особенности**: Движение цены +8.7% ($173→$193)

```bash
python scripts/enhanced_backtest.py --block-id august_12_single_day
```

### Управление блоками

#### Создание блоков
```bash
# Создать все предустановленные блоки
python scripts/manage_data_blocks.py create-predefined

# Создать кастомный блок
python scripts/manage_data_blocks.py create \
    my_test_block \
    "Мой тестовый блок" \
    "Описание блока" \
    --start "2025-08-10 00:00:00" \
    --end "2025-08-11 23:59:59" \
    --type mixed \
    --timeframes "1m,5m,15m,1h"
```

#### Просмотр блоков
```bash
# Список всех блоков
python scripts/manage_data_blocks.py list-blocks

# Фильтр по типу
python scripts/manage_data_blocks.py list-blocks --type trend
python scripts/manage_data_blocks.py list-blocks --type volatile

# Детальная информация
python scripts/manage_data_blocks.py info august_10_17_full
```

#### Загрузка и проверка
```bash
# Загрузить данные блока
python scripts/manage_data_blocks.py load august_12_single_day

# Загрузить только определенные timeframes
python scripts/manage_data_blocks.py load august_10_17_full -t 5m -t 1h

# Проверить целостность
python scripts/manage_data_blocks.py verify august_12_single_day
```

#### Очистка
```bash
# Удалить блок (с подтверждением)
python scripts/manage_data_blocks.py delete my_test_block

# Очистить поврежденные блоки
python scripts/manage_data_blocks.py cleanup
```

---

## 🎯 Тестирование стратегий

### Базовая стратегия

#### Консервативные настройки (по умолчанию)
```bash
# Стандартный тест
python scripts/enhanced_backtest.py --block-id august_12_single_day

# С сохранением результатов
python scripts/enhanced_backtest.py --block-id august_10_17_full --save-results
```

**Параметры по умолчанию:**
- `min_confidence: 0.15` (15%)
- `min_volume_ratio: 1.2` (1.2x)
- `min_adx: 20.0`

### Улучшенные стратегии

#### Тест с пониженными порогами
```bash
python scripts/test_improved_strategy.py
```

**Параметры:**
- `min_confidence: 0.10` (10%)
- `min_volume_ratio: 1.0` (1.0x)
- `min_adx: 15.0`

#### Агрессивная momentum стратегия
```bash
python scripts/final_aggressive_strategy.py
```

**Параметры:**
- `min_confidence: 0.03` (3%)
- `min_volume_ratio: 0.3` (0.3x)
- `immediate_entry_threshold: 0.006` (0.6%)
- Очень тугие стопы, широкие цели

### Сравнительное тестирование

#### Тест всех стратегий на одном блоке
```bash
echo "=== Консервативная стратегия ==="
python scripts/enhanced_backtest.py --block-id august_12_single_day

echo "=== Улучшенная стратегия ==="
python scripts/test_improved_strategy.py

echo "=== Агрессивная стратегия ==="
python scripts/final_aggressive_strategy.py
```

#### Тест одной стратегии на разных блоках
```bash
for block in august_12_single_day august_10_13_trend august_14_17_volatile; do
    echo "=== Тест на блоке: $block ==="
    python scripts/final_aggressive_strategy.py
    # Заменить блок в коде или добавить параметр командной строки
done
```

---

## 📊 Анализ результатов

### Типы результатов

#### 1. Консольный вывод
```
📊 FINAL AGGRESSIVE STRATEGY RESULTS
--------------------------------------------------
✅ Total trades: 19
📶 Signals generated: 217
💰 Total P&L: $1.38
📈 Final balance: $10001.38
📊 Return: +0.01%
🎯 Win rate: 15.8% (3/19)
```

#### 2. JSON файлы результатов
**Расположение**: `Output/`
- `backtest_results_SOLUSDT_YYYYMMDD_HHMMSS.json`
- `improved_backtest_results_YYYYMMDD_HHMMSS.json`
- `final_aggressive_results_YYYYMMDD_HHMMSS.json`

#### 3. HTML отчеты
```bash
# Генерация детального HTML отчета
python scripts/generate_detailed_report.py
```

**Файл**: `Output/SOL_USDT_Backtest_Report_YYYYMMDD_HHMMSS.html`

### Ключевые метрики

#### Результативность
- **Total trades** - общее количество сделок
- **Win rate** - процент прибыльных сделок
- **Total P&L** - общая прибыль/убыток
- **Total return** - процентная доходность

#### Эффективность
- **Signals generated** - количество сгенерированных сигналов
- **Market capture** - процент от движения рынка
- **Max drawdown** - максимальная просадка

#### Анализ сделок
- **Average win/loss** - средняя прибыль/убыток
- **Profit factor** - отношение прибыли к убыткам
- **Sharpe ratio** - риск-скорректированная доходность

---

## 🔍 Отладка и диагностика

### Анализ сигналов

#### Диагностика отсутствия сделок
```bash
# Анализ причин отсутствия сигналов
python scripts/analyze_signals.py
```

**Выводы:**
- Количество отфильтрованных сигналов по каждому критерию
- Потенциальные сигналы с их характеристиками
- Рекомендации по улучшению параметров

#### Отладка агрессивной стратегии
```bash
# Детальная диагностика новой стратегии
python scripts/debug_aggressive_strategy.py
```

**Анализирует:**
- Immediate entry conditions
- Basic filter failures
- Confidence thresholds
- Biggest price movements

### Проверка данных

#### Валидация блоков
```bash
# Проверить все блоки
python scripts/manage_data_blocks.py cleanup

# Проверить конкретный блок
python scripts/manage_data_blocks.py verify august_12_single_day
```

#### Анализ данных
```bash
# Загрузить и показать статистику
python scripts/manage_data_blocks.py load august_12_single_day
```

### Логирование

#### Просмотр логов
```bash
# Логи сохраняются автоматически во время выполнения
# Структурированное логирование в JSON формате
# Уровни: DEBUG, INFO, WARNING, ERROR
```

---

## ⚙️ Команды и параметры

### Основные скрипты

#### `enhanced_backtest.py`
```bash
python scripts/enhanced_backtest.py [OPTIONS]

# Опции:
--block-id BLOCK_ID     # Использовать фиксированный блок данных
--save-results          # Сохранить результаты в JSON
--balance AMOUNT        # Начальный баланс (по умолчанию: 10000)
--symbol SYMBOL         # Торговый символ (по умолчанию: SOLUSDT)
--days DAYS            # Количество дней для тестирования
```

#### `manage_data_blocks.py`
```bash
python scripts/manage_data_blocks.py COMMAND [OPTIONS]

# Команды:
create-predefined                    # Создать предустановленные блоки
list-blocks [--type TYPE] [--symbol SYMBOL]  # Список блоков
info BLOCK_ID                       # Информация о блоке
load BLOCK_ID [-t TIMEFRAME]        # Загрузить блок
verify BLOCK_ID                     # Проверить целостность
delete BLOCK_ID                     # Удалить блок
cleanup                             # Очистить поврежденные блоки
create BLOCK_ID NAME DESC [OPTIONS] # Создать кастомный блок
```

#### Специализированные тесты
```bash
# Улучшенная стратегия
python scripts/test_improved_strategy.py

# Агрессивная momentum стратегия  
python scripts/final_aggressive_strategy.py

# Анализ сигналов
python scripts/analyze_signals.py

# Отладка агрессивной стратегии
python scripts/debug_aggressive_strategy.py
```

### Переменные окружения

```bash
# Опциональные настройки
export LOG_LEVEL=INFO              # Уровень логирования
export PAPER_TRADING=true          # Режим paper trading
export MAX_POSITION_SIZE=0.02      # Максимальный размер позиции
```

---

## 🎯 Рекомендованные сценарии тестирования

### 1. Первичная проверка
```bash
# Быстрая проверка работоспособности
python scripts/manage_data_blocks.py create-predefined
python scripts/enhanced_backtest.py --block-id august_12_single_day
```

### 2. Разработка стратегии
```bash
# Итеративное тестирование
python scripts/analyze_signals.py                    # Анализ проблем
python scripts/debug_aggressive_strategy.py          # Отладка решения
python scripts/final_aggressive_strategy.py          # Тест решения
```

### 3. Комплексная оценка
```bash
# Тест на разных типах рынка
python scripts/enhanced_backtest.py --block-id august_10_13_trend     # Тренд
python scripts/enhanced_backtest.py --block-id august_14_17_volatile  # Волатильность
python scripts/enhanced_backtest.py --block-id august_10_17_full      # Смешанный
```

### 4. Производственная готовность
```bash
# Финальная проверка перед деплоем
python scripts/final_aggressive_strategy.py          # Основной тест
python scripts/generate_detailed_report.py           # Детальный отчет
python scripts/manage_data_blocks.py verify-all      # Проверка данных
```

---

## 🚨 Важные замечания

### Безопасность
- ✅ Все тесты выполняются на исторических данных
- ✅ Реальные API ключи не используются
- ✅ Никаких реальных сделок не совершается

### Производительность
- 📊 Тест на одном дне: ~20-30 секунд
- 📊 Тест на неделе: ~2-3 минуты
- 📊 Генерация отчета: ~1-2 минуты

### Требования к данным
- 📁 Исходные файлы: `data/SOLUSDT_*_real_*.csv`
- 📦 Блоки данных: `data/blocks/`
- 📄 Результаты: `Output/`

### Устранение неполадок

#### Ошибка "Block not found"
```bash
python scripts/manage_data_blocks.py create-predefined
```

#### Ошибка "No source file found"
```bash
# Проверить наличие исходных файлов
ls data/SOLUSDT_*_real_*.csv
```

#### Ошибка "No trades executed"
```bash
# Использовать агрессивную стратегию
python scripts/final_aggressive_strategy.py
```

---

## 📚 Дополнительные ресурсы

- 📖 [CLAUDE.md](CLAUDE.md) - Основная документация проекта
- 📋 [Описание торгового бота](Output/Описание_торгового_бота_SOL_USDT.md) - Детальное описание системы
- 📘 [Руководство по блокам данных](Output/Руководство_по_фиксированным_блокам_данных.md) - Полное руководство по блокам

---

**🎉 Успешного тестирования!**