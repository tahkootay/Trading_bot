#!/usr/bin/env python3
"""
Тестирование бэктестирования с оптимизированным ансамблем на малых данных
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedEnsemblePredictor:
    """Оптимизированная версия с отложенной загрузкой для бэктеста."""
    
    def __init__(self, models_dir: str = "models/ensemble_live", lazy_loading: bool = True):
        self.models_dir = Path(models_dir)
        self.lazy_loading = lazy_loading
        self.base_models = {}
        self.meta_model = None
        self.scaler = None
        self.feature_names = []
        self.models_loaded = False
        self.logger = logging.getLogger("OptimizedEnsemble")
        
        if not lazy_loading:
            self.load_models()
        else:
            self.logger.info("✅ Predictor создан с отложенной загрузкой")
    
    def _find_models_directory(self):
        """Поиск директории с моделями."""
        if not self.models_dir.exists():
            return None
        
        latest_link = self.models_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                return latest_link.resolve()
            elif latest_link.is_dir():
                return latest_link
        
        subdirs = [d for d in self.models_dir.iterdir() if d.is_dir() and d.name != "latest"]
        if subdirs:
            subdirs.sort(key=lambda x: x.name, reverse=True)
            return subdirs[0]
        
        return None
    
    def load_models(self):
        """Загрузка моделей."""
        if self.models_loaded:
            return True
        
        try:
            import joblib
            models_path = self._find_models_directory()
            
            if models_path is None:
                self.logger.error("Директория моделей не найдена")
                return False
            
            # Загружаем модели
            model_files = {
                'random_forest': 'random_forest_intraday.joblib',
                'lightgbm': 'lightgbm_intraday.joblib',
                'xgboost': 'xgboost_intraday.joblib',
                'catboost': 'catboost_intraday.joblib'
            }
            
            for model_name, filename in model_files.items():
                model_path = models_path / filename
                if model_path.exists():
                    self.base_models[model_name] = joblib.load(model_path)
            
            # Метамодель
            meta_path = models_path / "meta_intraday.joblib"
            if meta_path.exists():
                self.meta_model = joblib.load(meta_path)
            
            # Скейлер и признаки
            scaler_path = models_path / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            features_path = models_path / "feature_names.joblib"
            if features_path.exists():
                self.feature_names = joblib.load(features_path)
            
            self.models_loaded = True
            self.logger.info("✅ Все модели загружены")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки: {e}")
            return False
    
    def predict_proba(self, features):
        """Предсказание вероятности для совместимости с бэктестом."""
        if not self.models_loaded:
            if not self.load_models():
                return np.array([[0.5, 0.5]])  # Нейтральное предсказание
        
        try:
            # Подготовка признаков
            if isinstance(features, pd.DataFrame):
                # Исключаем нецелевые колонки
                exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'confirm']
                feature_cols = [col for col in features.columns if col not in exclude_cols]
                features_array = features[feature_cols].values
            else:
                features_array = features
            
            if len(features_array.shape) == 1:
                features_array = features_array.reshape(1, -1)
            
            # Скейлинг
            if self.scaler is not None:
                features_array = self.scaler.transform(features_array)
            
            # Базовые предсказания
            base_predictions = []
            expected_models = ['random_forest', 'lightgbm', 'xgboost', 'catboost']
            
            for model_name in expected_models:
                if model_name in self.base_models:
                    model = self.base_models[model_name]
                    pred_proba = model.predict_proba(features_array)
                    if len(pred_proba.shape) == 2 and pred_proba.shape[1] >= 2:
                        probability = pred_proba[0, 1]
                    else:
                        probability = pred_proba[0]
                    base_predictions.append(probability)
                else:
                    base_predictions.append(0.5)
            
            # Метамодель
            if self.meta_model is not None:
                meta_features_array = np.array([base_predictions])
                final_proba = self.meta_model.predict_proba(meta_features_array)
                return final_proba
            else:
                # Простое усреднение если нет метамодели
                avg_prob = np.mean(base_predictions)
                return np.array([[1-avg_prob, avg_prob]])
            
        except Exception as e:
            self.logger.error(f"Ошибка предсказания: {e}")
            return np.array([[0.5, 0.5]])
    
    def predict(self, features):
        """Предсказание класса."""
        proba = self.predict_proba(features)
        return 1 if proba[0, 1] > 0.5 else 0


def create_test_data():
    """Создание синтетических данных для тестирования."""
    logger.info("📊 Создаем синтетические данные...")
    
    # Создаем 1 день 5-минутных данных (288 свечей)
    dates = pd.date_range(start='2024-08-23', periods=288, freq='5T')
    
    # Синтетические цены (случайное блуждание)
    np.random.seed(42)
    prices = 150 + np.cumsum(np.random.randn(288) * 0.5)  # SOL около $150
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.random.rand(288) * 2,
        'low': prices - np.random.rand(288) * 2,
        'close': prices,
        'volume': np.random.rand(288) * 1000000 + 500000
    })
    
    # Корректируем high/low
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    logger.info(f"✅ Создано {len(data)} свечей данных")
    return data


def add_technical_indicators(data):
    """Добавление технических индикаторов."""
    logger.info("📈 Добавляем технические индикаторы...")
    
    # Скользящие средние
    data['MA5'] = data['close'].rolling(5).mean()
    data['MA10'] = data['close'].rolling(10).mean()
    data['MA20'] = data['close'].rolling(20).mean()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['close'].ewm(span=12).mean()
    exp2 = data['close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    data['BB_middle'] = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
    data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
    data['BB_position'] = (data['close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
    
    # ATR
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['ATR'] = true_range.rolling(14).mean()
    data['ATR_ratio'] = data['ATR'] / data['close']
    
    # Volume indicators
    data['volume_sma'] = data['volume'].rolling(20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_sma']
    
    # Price vs VWAP
    data['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
    data['price_vs_vwap'] = (data['close'] - data['vwap']) / data['vwap']
    
    # Volume change
    data['vol_change'] = data['volume'].pct_change()
    
    # Дополнительные признаки для полного набора (23 признака)
    data['price_change'] = data['close'].pct_change()
    data['high_low_ratio'] = (data['high'] - data['low']) / data['close']
    data['open_close_ratio'] = (data['close'] - data['open']) / data['open']
    
    # Убираем NaN
    data = data.dropna()
    
    logger.info(f"✅ Добавлены индикаторы, осталось {len(data)} строк")
    return data


def simple_backtest(data, predictor):
    """Простой бэктест с ансамблем."""
    logger.info("🔬 Запуск простого бэктеста...")
    
    balance = 10000.0
    position = None
    trades = []
    
    # Используем только последние 100 свечей для ускорения
    test_data = data.tail(100)
    
    for i in range(50, len(test_data)):  # Начинаем с 50 для индикаторов
        current_row = test_data.iloc[i]
        current_price = current_row['close']
        
        # Подготовка признаков
        features_df = pd.DataFrame([current_row])
        
        try:
            # Получение предсказания
            prediction_proba = predictor.predict_proba(features_df)[0, 1]
            prediction = 1 if prediction_proba > 0.55 else 0  # Порог 55%
            
            # Логика торговли
            if position is None and prediction == 1:  # Покупаем
                position = {
                    'entry_price': current_price,
                    'entry_time': current_row['timestamp'],
                    'type': 'BUY'
                }
                
            elif position is not None and (prediction == 0 or i == len(test_data) - 1):  # Продаем
                exit_price = current_price
                pnl = exit_price - position['entry_price']
                pnl_pct = (pnl / position['entry_price']) * 100
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_row['timestamp'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'prediction_proba': prediction_proba
                })
                
                balance += pnl
                position = None
                
        except Exception as e:
            logger.warning(f"Ошибка на строке {i}: {e}")
            continue
    
    # Результаты
    if trades:
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = winning_trades / total_trades
        
        logger.info("="*50)
        logger.info("📋 РЕЗУЛЬТАТЫ БЭКТЕСТА:")
        logger.info(f"  Начальный баланс: ${10000:.2f}")
        logger.info(f"  Финальный баланс: ${balance:.2f}")
        logger.info(f"  Общий PnL: ${total_pnl:.2f}")
        logger.info(f"  Всего сделок: {total_trades}")
        logger.info(f"  Прибыльных сделок: {winning_trades}")
        logger.info(f"  Винрейт: {win_rate:.1%}")
        logger.info("="*50)
        
        # Показываем последние сделки
        logger.info("📊 Последние 3 сделки:")
        for trade in trades[-3:]:
            logger.info(f"  {trade['entry_time'].strftime('%H:%M')} → {trade['exit_time'].strftime('%H:%M')}: "
                       f"${trade['pnl']:+.2f} ({trade['pnl_pct']:+.1f}%) "
                       f"Prob: {trade['prediction_proba']:.3f}")
    else:
        logger.info("❌ Сделки не были выполнены")


def main():
    """Главная функция."""
    logger.info("🎯 Тестирование ансамбля на синтетических данных")
    logger.info("="*60)
    
    try:
        # 1. Создание предсказателя
        logger.info("1️⃣ Создаем оптимизированный предсказатель...")
        start_time = time.time()
        predictor = OptimizedEnsemblePredictor(lazy_loading=True)
        creation_time = time.time() - start_time
        logger.info(f"   Создание: {creation_time:.4f}s")
        
        # 2. Создание данных
        logger.info("2️⃣ Создаем тестовые данные...")
        data = create_test_data()
        data = add_technical_indicators(data)
        
        # 3. Первое предсказание (триггерит загрузку моделей)
        logger.info("3️⃣ Тестируем первое предсказание (загрузка моделей)...")
        start_time = time.time()
        test_features = data.iloc[-1:][['MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal', 
                                       'BB_position', 'ATR_ratio', 'volume_ratio', 'price_vs_vwap',
                                       'vol_change', 'price_change', 'high_low_ratio', 'open_close_ratio']]
        
        first_pred = predictor.predict_proba(test_features)
        first_pred_time = time.time() - start_time
        logger.info(f"   Первое предсказание: {first_pred_time:.2f}s")
        logger.info(f"   Результат: {first_pred[0]}")
        
        # 4. Последующие предсказания
        logger.info("4️⃣ Тестируем последующие предсказания...")
        start_time = time.time()
        for _ in range(10):
            predictor.predict_proba(test_features)
        subsequent_time = time.time() - start_time
        logger.info(f"   10 предсказаний: {subsequent_time:.3f}s ({subsequent_time/10:.3f}s каждое)")
        
        # 5. Запуск бэктеста
        logger.info("5️⃣ Запускаем бэктест...")
        simple_backtest(data, predictor)
        
        logger.info("✅ Все тесты завершены успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()