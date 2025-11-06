# Установка зависимостей для LSTM обучения

## Быстрый старт

Данные уже подготовлены и готовы к обучению. Остается только установить ML библиотеки.

## Вариант 1: PyTorch (рекомендуемый)

PyTorch версия LSTM модели автоматически установит зависимости при первом запуске:

```bash
python3 src/models/lstm/torch_lstm_model.py
```

## Вариант 2: TensorFlow/Keras

### Через conda (рекомендуется)
```bash
# Установка miniforge если еще не установлена
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh -o miniforge.sh
bash miniforge.sh -b -p $HOME/miniforge

# Активация и создание окружения
source $HOME/miniforge/bin/activate
conda create -n lstm-env python=3.11 tensorflow pandas scikit-learn matplotlib seaborn -y
conda activate lstm-env

# Запуск обучения
python src/models/lstm/real_lstm_model.py
```

### Через pip с Python 3.11
```bash
# Установка Python 3.11 через pyenv
curl https://pyenv.run | bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
pyenv install 3.11.10
pyenv global 3.11.10

# Установка зависимостей
pip install tensorflow pandas scikit-learn matplotlib seaborn

# Запуск обучения
python src/models/lstm/real_lstm_model.py
```

## Готовые данные

Все файлы уже подготовлены:
- ✅ `data/processed/train_2025_clean.csv` - тренировочные данные
- ✅ `data/processed/validation_2025_clean.csv` - валидационные данные  
- ✅ `data/processed/test_2025_clean.csv` - тестовые данные
- ✅ 33 LSTM признака валидированы
- ✅ Целевые переменные для горизонтов 3, 5, 7 созданы
- ✅ Нет пересечений между наборами данных

## Запуск через CLI

```bash
# LSTM с горизонтом 3
python main.py train --model lstm --horizon 3 --window 30 --epochs 30

# LSTM с горизонтом 5  
python main.py train --model lstm --horizon 5 --window 30 --epochs 50

# LSTM с горизонтом 7
python main.py train --model lstm --horizon 7 --window 60 --epochs 30
```

## Проблемы с установкой?

### Python 3.13 не поддерживается TensorFlow
- Используйте Python 3.11 или 3.12
- Или используйте PyTorch версию (автоматическая установка)

### Проблемы с permissions
```bash
# Добавьте --user или --break-system-packages
pip install --user tensorflow
# или
pip install --break-system-packages tensorflow
```

### macOS Apple Silicon (M1/M2)
```bash
# Используйте conda для лучшей совместимости
conda install -c apple tensorflow-deps
pip install tensorflow-macos tensorflow-metal
```

## Что будет создано при обучении

После обучения появятся:
- `saved_models/real_lstm_horizon{N}.keras` - обученная модель
- `saved_models/scaler_real_lstm_h{N}.pkl` - нормализатор данных
- `reports/lstm_training_history_h{N}.png` - график обучения
- `reports/real_lstm_results_h{N}.txt` - отчет с метриками
- `data/processed/real_lstm_predictions_h{N}.csv` - предсказания

## Проверка готовности

```bash
# Тест совместимости (без ML библиотек)
python scripts/data_processing/test_lstm_compatibility.py
```

Результат должен показать: **✅ ГОТОВО К ОБУЧЕНИЮ LSTM**