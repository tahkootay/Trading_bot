#!/usr/bin/env python3
"""
Быстрое обучение ML моделей с использованием доступных данных
"""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

# Устанавливаем переменную среды для логирования
import os
os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent)

def main():
    """Быстрое обучение с оптимальными настройками"""
    from scripts.train_ml_models import main as train_main
    
    # Запуск с настройками по умолчанию
    # Используется доступные данные из папки data/
    import click
    
    ctx = click.Context(train_main)
    ctx.params = {
        'data_dir': 'data',
        'models_dir': 'models', 
        'forward_periods': 12,  # 1 hour prediction on 5min timeframe
        'model_version': None,  # Auto timestamp
        'min_samples': 500      # Reduced minimum for available data
    }
    
    train_main.main(standalone_mode=False, **ctx.params)

if __name__ == "__main__":
    print("🚀 Быстрое обучение ML моделей...")
    print("📂 Используемые данные: data/")
    print("💾 Сохранение моделей: models/")
    print("⏰ Горизонт прогноза: 1 час (12 периодов по 5 мин)")
    print()
    
    main()