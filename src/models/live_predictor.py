#!/usr/bin/env python3
"""
Live предсказатель для торгового бота
Объединяет генерацию признаков и ансамбль моделей согласно спецификации
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Callable
import logging
from datetime import datetime, timedelta
import time

from ..feature_engine.live_features import LiveFeatureGenerator, FeatureValidator
from .ensemble_predictor import EnsemblePredictor, ModelPerformanceTracker


class LiveTradingPredictor:
    """
    Основной класс для live предсказаний в торговле.
    Интегрирует генерацию признаков и ансамбль моделей.
    """
    
    def __init__(self, 
                 models_dir: str = "models/ensemble_live",
                 min_confidence: float = 0.6,
                 prediction_cooldown: int = 30):
        """
        Инициализация live предсказателя.
        
        Args:
            models_dir: Директория с моделями
            min_confidence: Минимальная уверенность для сигнала
            prediction_cooldown: Время в секундах между предсказаниями
        """
        # Компоненты
        self.feature_generator = LiveFeatureGenerator(window_size=100)
        self.ensemble_predictor = EnsemblePredictor(models_dir=models_dir)
        self.performance_tracker = ModelPerformanceTracker()
        
        # Параметры
        self.min_confidence = min_confidence
        self.prediction_cooldown = prediction_cooldown
        
        # Состояние
        self.last_prediction_time = None
        self.last_prediction = None
        self.prediction_history = []
        self.is_ready = False
        
        # Статистика
        self.stats = {
            'total_predictions': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'strong_signals': 0,
            'weak_signals': 0,
            'errors': 0
        }
        
        # Колбэки
        self.on_prediction_callback: Optional[Callable] = None
        self.on_signal_callback: Optional[Callable] = None
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
        # Проверка готовности компонентов
        self._check_readiness()
    
    def _check_readiness(self):
        """Проверка готовности всех компонентов."""
        if self.ensemble_predictor.is_ready():
            self.is_ready = True
            self.logger.info("🎯 Live predictor is ready!")
        else:
            self.is_ready = False
            self.logger.error("❌ Live predictor not ready - models not loaded")
    
    def set_prediction_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Установка колбэка для всех предсказаний."""
        self.on_prediction_callback = callback
    
    def set_signal_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Установка колбэка только для торговых сигналов."""
        self.on_signal_callback = callback
    
    def _should_make_prediction(self) -> bool:
        """Проверка, нужно ли делать новое предсказание."""
        if not self.is_ready:
            return False
        
        if self.last_prediction_time is None:
            return True
        
        time_since_last = time.time() - self.last_prediction_time
        return time_since_last >= self.prediction_cooldown
    
    def process_new_data(self, kline_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Обработка новых данных и генерация предсказания.
        
        Args:
            kline_df: DataFrame с OHLCV данными
            
        Returns:
            Результат предсказания или None
        """
        if not self._should_make_prediction():
            return None
        
        try:
            # 1. Генерируем признаки
            features_df = self.feature_generator.update_data(kline_df)
            if features_df is None:
                self.logger.debug("Insufficient data for feature generation")
                return None
            
            # 2. Валидируем признаки
            validation_result = FeatureValidator.validate_features(features_df)
            if not validation_result['valid']:
                self.logger.warning(f"Feature validation failed: {validation_result['issues']}")
                self.stats['errors'] += 1
                return None
            
            # 3. Получаем предсказание от ансамбля
            prediction = self.ensemble_predictor.predict_ensemble(features_df)
            if prediction is None:
                self.logger.error("Failed to get ensemble prediction")
                self.stats['errors'] += 1
                return None
            
            # 4. Добавляем контекстную информацию
            latest_candle = kline_df.iloc[-1] if len(kline_df) > 0 else {}
            feature_summary = self.feature_generator.get_feature_summary()
            
            result = {
                'timestamp': datetime.now(),
                'price': latest_candle.get('close', 0),
                'prediction': prediction,
                'feature_summary': feature_summary,
                'validation': validation_result,
                'is_trading_signal': self._is_trading_signal(prediction)
            }
            
            # 5. Обновляем статистику
            self._update_stats(result)
            
            # 6. Сохраняем результат
            self.last_prediction = result
            self.last_prediction_time = time.time()
            self.prediction_history.append(result)
            
            # Ограничиваем историю
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            # 7. Вызываем колбэки
            if self.on_prediction_callback:
                try:
                    self.on_prediction_callback(result)
                except Exception as e:
                    self.logger.error(f"Error in prediction callback: {e}")
            
            if result['is_trading_signal'] and self.on_signal_callback:
                try:
                    self.on_signal_callback(result)
                except Exception as e:
                    self.logger.error(f"Error in signal callback: {e}")
            
            # 8. Логируем результат
            self._log_prediction(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing new data: {e}")
            self.stats['errors'] += 1
            return None
    
    def _is_trading_signal(self, prediction: Dict[str, Any]) -> bool:
        """Определяет, является ли предсказание торговым сигналом."""
        final_prob = prediction.get('final_probability', 0.5)
        
        # Сигнал BUY если вероятность > min_confidence
        # Сигнал SELL если вероятность < (1 - min_confidence)
        return (
            final_prob >= self.min_confidence or 
            final_prob <= (1 - self.min_confidence)
        )
    
    def _update_stats(self, result: Dict[str, Any]):
        """Обновление статистики."""
        self.stats['total_predictions'] += 1
        
        prediction = result['prediction']
        final_signal = prediction.get('final_signal', 0)
        strength = prediction.get('signal_strength', 'WEAK')
        
        if final_signal == 1:
            self.stats['buy_signals'] += 1
        else:
            self.stats['sell_signals'] += 1
        
        if strength == 'STRONG':
            self.stats['strong_signals'] += 1
        else:
            self.stats['weak_signals'] += 1
    
    def _log_prediction(self, result: Dict[str, Any]):
        """Логирование предсказания."""
        prediction = result['prediction']
        price = result.get('price', 0)
        
        signal = "BUY" if prediction['final_signal'] == 1 else "SELL/HOLD"
        prob = prediction['final_probability']
        strength = prediction['signal_strength']
        is_signal = result['is_trading_signal']
        
        log_msg = (
            f"🎯 Prediction: {signal} | "
            f"Price: {price:.4f} | "
            f"Prob: {prob:.3f} | "
            f"Strength: {strength} | "
            f"Trading Signal: {'✅' if is_signal else '❌'}"
        )
        
        if is_signal:
            self.logger.info(log_msg)
        else:
            self.logger.debug(log_msg)
    
    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Получение последнего предсказания."""
        return self.last_prediction
    
    def get_trading_signals_history(self, limit: int = 50) -> list:
        """Получение истории торговых сигналов."""
        trading_signals = [
            pred for pred in self.prediction_history[-limit:]
            if pred.get('is_trading_signal', False)
        ]
        return trading_signals
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики работы предсказателя."""
        stats = self.stats.copy()
        
        # Добавляем производительность
        if self.stats['total_predictions'] > 0:
            stats['buy_signal_rate'] = self.stats['buy_signals'] / self.stats['total_predictions']
            stats['strong_signal_rate'] = self.stats['strong_signals'] / self.stats['total_predictions']
            stats['error_rate'] = self.stats['errors'] / self.stats['total_predictions']
        
        # Добавляем информацию о компонентах
        stats['is_ready'] = self.is_ready
        stats['models_info'] = self.ensemble_predictor.get_models_info()
        stats['last_prediction_time'] = self.last_prediction_time
        
        return stats
    
    def reset_stats(self):
        """Сброс статистики."""
        self.stats = {
            'total_predictions': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'strong_signals': 0,
            'weak_signals': 0,
            'errors': 0
        }
        self.logger.info("Statistics reset")
    
    def get_prediction_summary(self, result: Optional[Dict[str, Any]] = None) -> str:
        """Получение краткой сводки по предсказанию."""
        if result is None:
            result = self.last_prediction
        
        if result is None:
            return "No predictions available"
        
        prediction = result['prediction']
        return self.ensemble_predictor.get_prediction_summary(prediction)


class AsyncLivePredictor:
    """Асинхронная обертка для live предсказателя."""
    
    def __init__(self, predictor: LiveTradingPredictor):
        self.predictor = predictor
        self.logger = logging.getLogger(__name__)
    
    async def process_data_async(self, kline_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Асинхронная обработка данных."""
        loop = asyncio.get_event_loop()
        
        try:
            # Запускаем в executor чтобы не блокировать event loop
            result = await loop.run_in_executor(
                None, self.predictor.process_new_data, kline_df
            )
            return result
        except Exception as e:
            self.logger.error(f"Error in async prediction: {e}")
            return None


# Пример интеграции с WebSocket
async def websocket_prediction_example():
    """Пример интеграции предсказателя с WebSocket."""
    from ..data_collector.websocket_client import BybitWebSocketClient
    
    # Создаем предсказатель
    predictor = LiveTradingPredictor()
    
    if not predictor.is_ready:
        print("❌ Predictor not ready")
        return
    
    # Колбэки для обработки сигналов
    def on_prediction(result):
        summary = predictor.get_prediction_summary(result)
        print(f"📊 Prediction: {summary}")
    
    def on_trading_signal(result):
        summary = predictor.get_prediction_summary(result)
        print(f"🚨 TRADING SIGNAL: {summary}")
    
    predictor.set_prediction_callback(on_prediction)
    predictor.set_signal_callback(on_trading_signal)
    
    # Колбэк для обработки новых данных от WebSocket
    def on_kline_data(df: pd.DataFrame):
        result = predictor.process_new_data(df)
        # result может быть None если предсказание не делалось
    
    # Создаем WebSocket клиент
    ws_client = BybitWebSocketClient(symbol="SOLUSDT", timeframe="5")
    ws_client.set_kline_callback(on_kline_data)
    
    try:
        print("🚀 Starting live prediction with WebSocket...")
        await ws_client.connect()
    except KeyboardInterrupt:
        print("🛑 Stopping...")
    finally:
        await ws_client.disconnect()
        
        # Печатаем статистику
        stats = predictor.get_stats()
        print(f"\n📈 Final Stats: {stats}")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Запуск примера
    asyncio.run(websocket_prediction_example())