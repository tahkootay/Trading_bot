#!/usr/bin/env python3
"""
Диагностический скрипт для отладки проблем инициализации ансамбля моделей
"""

import sys
import os
import time
import logging
from pathlib import Path

# Настройка логирования с детальным выводом
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_ensemble.log')
    ]
)

logger = logging.getLogger(__name__)

def test_model_files():
    """Тест доступности файлов моделей."""
    logger.info("🔍 Проверка файлов моделей...")
    
    models_dir = Path("models/ensemble_live")
    logger.info(f"Проверка директории: {models_dir}")
    
    if not models_dir.exists():
        logger.error(f"❌ Директория не найдена: {models_dir}")
        return False
    
    # Проверяем ссылку latest
    latest_link = models_dir / "latest"
    if latest_link.exists():
        if latest_link.is_symlink():
            target = latest_link.resolve()
            logger.info(f"✅ Ссылка latest указывает на: {target}")
            models_path = target
        else:
            logger.info(f"✅ latest - это директория: {latest_link}")
            models_path = latest_link
    else:
        logger.warning("⚠️ Ссылка latest не найдена, ищем последнюю директорию...")
        subdirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name != "latest"]
        if subdirs:
            subdirs.sort(key=lambda x: x.name, reverse=True)
            models_path = subdirs[0]
            logger.info(f"✅ Используем директорию: {models_path}")
        else:
            logger.error("❌ Нет поддиректорий с моделями!")
            return False
    
    # Проверяем файлы моделей
    expected_files = [
        'random_forest_intraday.joblib',
        'lightgbm_intraday.joblib', 
        'xgboost_intraday.joblib',
        'catboost_intraday.joblib',
        'meta_intraday.joblib',
        'scaler.joblib',
        'feature_names.joblib'
    ]
    
    missing_files = []
    for filename in expected_files:
        file_path = models_path / filename
        if file_path.exists():
            size = file_path.stat().st_size
            logger.info(f"✅ {filename} - {size} bytes")
        else:
            missing_files.append(filename)
            logger.error(f"❌ Отсутствует: {filename}")
    
    if missing_files:
        logger.error(f"❌ Отсутствуют файлы: {missing_files}")
        return False
    
    logger.info("✅ Все файлы моделей найдены")
    return True

def test_model_loading():
    """Тест загрузки отдельных моделей."""
    logger.info("🔄 Тестирование загрузки отдельных моделей...")
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        import joblib
        logger.info("✅ joblib импортирован")
    except Exception as e:
        logger.error(f"❌ Ошибка импорта joblib: {e}")
        return False
    
    models_dir = Path("models/ensemble_live/latest")
    model_files = {
        'random_forest': 'random_forest_intraday.joblib',
        'lightgbm': 'lightgbm_intraday.joblib',
        'xgboost': 'xgboost_intraday.joblib',
        'catboost': 'catboost_intraday.joblib'
    }
    
    loaded_models = {}
    
    for model_name, filename in model_files.items():
        logger.info(f"🔄 Загружаем {model_name}...")
        file_path = models_dir / filename
        
        try:
            start_time = time.time()
            model = joblib.load(file_path)
            load_time = time.time() - start_time
            
            logger.info(f"✅ {model_name} загружена за {load_time:.2f}s")
            logger.info(f"   Тип: {type(model)}")
            
            # Проверяем базовые атрибуты
            if hasattr(model, 'n_features_in_'):
                logger.info(f"   Признаков: {model.n_features_in_}")
            if hasattr(model, 'classes_'):
                logger.info(f"   Классы: {model.classes_}")
                
            loaded_models[model_name] = model
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    # Тест метамодели
    logger.info("🔄 Загружаем метамодель...")
    try:
        start_time = time.time()
        meta_model = joblib.load(models_dir / "meta_intraday.joblib")
        load_time = time.time() - start_time
        
        logger.info(f"✅ Метамодель загружена за {load_time:.2f}s")
        logger.info(f"   Тип: {type(meta_model)}")
        
        if hasattr(meta_model, 'n_features_in_'):
            logger.info(f"   Признаков: {meta_model.n_features_in_}")
        if hasattr(meta_model, 'classes_'):
            logger.info(f"   Классы: {meta_model.classes_}")
            
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки метамодели: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Тест скейлера и признаков
    logger.info("🔄 Загружаем скейлер и признаки...")
    try:
        scaler = joblib.load(models_dir / "scaler.joblib")
        logger.info(f"✅ Скейлер загружен: {type(scaler)}")
        
        feature_names = joblib.load(models_dir / "feature_names.joblib")
        logger.info(f"✅ Признаки загружены: {len(feature_names)} признаков")
        logger.info(f"   Первые 5: {feature_names[:5]}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки скейлера/признаков: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    logger.info("✅ Все модели загружены успешно!")
    return True

def test_ensemble_predictor_step_by_step():
    """Тест пошаговой инициализации EnsemblePredictor."""
    logger.info("🔄 Тестирование EnsemblePredictor пошагово...")
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        logger.info("🔄 Импортируем EnsemblePredictor...")
        from src.models.ensemble_predictor import EnsemblePredictor
        logger.info("✅ EnsemblePredictor импортирован")
        
        # Создаем экземпляр БЕЗ автозагрузки
        logger.info("🔄 Создаем экземпляр предсказателя...")
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        
        # Инициализируем атрибуты вручную
        logger.info("🔄 Инициализируем атрибуты...")
        predictor.models_dir = Path("models/ensemble_live")
        predictor.use_latest = True
        predictor.base_models = {}
        predictor.meta_model = None
        predictor.scaler = None
        predictor.feature_names = []
        predictor.models_loaded = False
        predictor.models_info = {}
        predictor.logger = logging.getLogger("EnsemblePredictor")
        
        logger.info("✅ Атрибуты инициализированы")
        
        # Тестируем поиск директории
        logger.info("🔄 Тестируем поиск директории моделей...")
        models_path = predictor._find_models_directory()
        if models_path:
            logger.info(f"✅ Директория найдена: {models_path}")
        else:
            logger.error("❌ Директория не найдена")
            return False
        
        # Тестируем загрузку моделей по частям
        logger.info("🔄 Тестируем загрузку моделей по частям...")
        success = predictor.load_models()
        
        if success:
            logger.info("✅ Модели загружены успешно!")
            logger.info(f"   Базовые модели: {list(predictor.base_models.keys())}")
            logger.info(f"   Метамодель: {'Да' if predictor.meta_model else 'Нет'}")
            logger.info(f"   Скейлер: {'Да' if predictor.scaler else 'Нет'}")
            logger.info(f"   Признаков: {len(predictor.feature_names)}")
            logger.info(f"   Готов: {predictor.is_ready()}")
        else:
            logger.error("❌ Ошибка загрузки моделей")
            return False
        
    except Exception as e:
        logger.error(f"❌ Ошибка в тестировании EnsemblePredictor: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    logger.info("✅ EnsemblePredictor протестирован успешно!")
    return True

def test_memory_usage():
    """Тест использования памяти при загрузке."""
    logger.info("🔄 Тестирование использования памяти...")
    
    try:
        import psutil
        process = psutil.Process()
        
        # Начальное использование памяти
        memory_start = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"📊 Начальное использование памяти: {memory_start:.1f} MB")
        
        # Загружаем модели
        sys.path.insert(0, str(Path(__file__).parent))
        from src.models.ensemble_predictor import EnsemblePredictor
        
        logger.info("🔄 Создаем EnsemblePredictor...")
        predictor = EnsemblePredictor()
        
        # Финальное использование памяти
        memory_end = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_end - memory_start
        
        logger.info(f"📊 Финальное использование памяти: {memory_end:.1f} MB")
        logger.info(f"📊 Прирост памяти: {memory_delta:.1f} MB")
        
        if memory_delta > 1000:  # Больше 1GB
            logger.warning(f"⚠️ Большой расход памяти: {memory_delta:.1f} MB")
        
    except ImportError:
        logger.warning("⚠️ psutil не установлен, пропускаем тест памяти")
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования памяти: {e}")

def main():
    """Основная функция диагностики."""
    logger.info("🚀 Запуск диагностики инициализации ансамбля")
    logger.info("="*60)
    
    tests = [
        ("Проверка файлов моделей", test_model_files),
        ("Тест загрузки моделей", test_model_loading),
        ("Тест EnsemblePredictor", test_ensemble_predictor_step_by_step),
        ("Тест использования памяти", test_memory_usage),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info("-" * 60)
        logger.info(f"🧪 {test_name}")
        logger.info("-" * 60)
        
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            results[test_name] = {
                'success': result,
                'duration': end_time - start_time
            }
            
            if result:
                logger.info(f"✅ {test_name} - УСПЕШНО ({end_time - start_time:.2f}s)")
            else:
                logger.error(f"❌ {test_name} - ПРОВАЛЕН ({end_time - start_time:.2f}s)")
                
        except Exception as e:
            results[test_name] = {
                'success': False,
                'duration': 0,
                'error': str(e)
            }
            logger.error(f"❌ {test_name} - ИСКЛЮЧЕНИЕ: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Итоговый отчет
    logger.info("="*60)
    logger.info("📋 ИТОГОВЫЙ ОТЧЕТ")
    logger.info("="*60)
    
    for test_name, result in results.items():
        status = "✅ УСПЕШНО" if result['success'] else "❌ ПРОВАЛЕН"
        duration = result.get('duration', 0)
        logger.info(f"{test_name}: {status} ({duration:.2f}s)")
        if 'error' in result:
            logger.info(f"   Ошибка: {result['error']}")
    
    # Общий результат
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if r['success'])
    
    logger.info("-" * 60)
    logger.info(f"📊 Пройдено: {successful_tests}/{total_tests} тестов")
    
    if successful_tests == total_tests:
        logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        logger.error("❌ ЕСТЬ ПРОВАЛЬНЫЕ ТЕСТЫ!")
        logger.info("\n💡 РЕКОМЕНДАЦИИ:")
        logger.info("1. Проверьте наличие всех файлов моделей")
        logger.info("2. Убедитесь, что модели совместимы с текущей версией библиотек")
        logger.info("3. Проверьте доступность памяти")
        logger.info("4. Рассмотрите возможность отложенной загрузки моделей")

if __name__ == "__main__":
    main()