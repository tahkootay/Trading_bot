#!/usr/bin/env python3
"""
Модуль генерации признаков для live торговли
Согласно спецификации: MA5/10/20, RSI, MACD, Bollinger Bands, Volume change
"""

import pandas as pd
import numpy as np
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from typing import Optional, Dict, Any
import logging


class LiveFeatureGenerator:
    """Генератор признаков для live торговли в реальном времени."""
    
    def __init__(self, window_size: int = 100):
        """
        Инициализация генератора признаков.
        
        Args:
            window_size: Размер скользящего окна для хранения данных
        """
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        
        # Параметры индикаторов (согласно спецификации)
        self.rsi_window = 14
        self.bb_window = 20
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Буфер для хранения исходных данных
        self.data_buffer = pd.DataFrame()
        
        # Последние рассчитанные признаки (для отладки)
        self.last_features = None
        
    def update_data(self, new_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Обновление данных и генерация признаков.
        
        Args:
            new_df: DataFrame с новыми OHLCV данными
            
        Returns:
            DataFrame с признаками или None если недостаточно данных
        """
        try:
            # Обновляем буфер данных
            self.data_buffer = new_df.copy()
            
            # Проверяем достаточность данных
            if len(self.data_buffer) < max(self.bb_window, self.macd_slow, 20):
                self.logger.debug(f"Insufficient data: {len(self.data_buffer)} rows")
                return None
            
            # Генерируем признаки
            features_df = self._generate_features(self.data_buffer)
            
            if features_df is not None and len(features_df) > 0:
                self.last_features = features_df.iloc[-1:].copy()
                return features_df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error updating features: {e}")
            return None
    
    def _generate_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Генерация всех признаков согласно спецификации."""
        try:
            df_features = df.copy().sort_values('timestamp').reset_index(drop=True)
            
            # 1. Moving Averages (согласно спецификации)
            df_features['MA5'] = df_features['close'].rolling(5).mean()
            df_features['MA10'] = df_features['close'].rolling(10).mean() 
            df_features['MA20'] = df_features['close'].rolling(20).mean()
            
            # 2. RSI
            rsi_indicator = RSIIndicator(df_features['close'], window=self.rsi_window)
            df_features['RSI'] = rsi_indicator.rsi()
            
            # 3. MACD (согласно спецификации)
            macd = MACD(df_features['close'], 
                       window_fast=self.macd_fast, 
                       window_slow=self.macd_slow, 
                       window_sign=self.macd_signal)
            df_features['MACD'] = macd.macd()
            df_features['MACD_signal'] = macd.macd_signal()
            df_features['MACD_diff'] = df_features['MACD'] - df_features['MACD_signal']
            
            # 4. Bollinger Bands (согласно спецификации)
            bb = BollingerBands(df_features['close'], window=self.bb_window)
            df_features['BB_hband'] = bb.bollinger_hband()
            df_features['BB_lband'] = bb.bollinger_lband()
            df_features['BB_width'] = bb.bollinger_wband()
            df_features['BB_position'] = (df_features['close'] - df_features['BB_lband']) / (df_features['BB_hband'] - df_features['BB_lband'])
            
            # 5. Volume change (согласно спецификации)
            df_features['vol_change'] = df_features['volume'].pct_change()
            df_features['volume_sma'] = df_features['volume'].rolling(20).mean()
            df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma']
            
            # 6. Дополнительные технические признаки (как в train_ensemble_models.py)
            df_features['price_change'] = df_features['close'].pct_change()
            df_features['price_change_5'] = df_features['close'].pct_change(5)
            df_features['volatility'] = df_features['close'].pct_change().rolling(20).std()
            
            # 7. EMA индикаторы
            df_features['EMA12'] = EMAIndicator(df_features['close'], window=12).ema_indicator()
            df_features['EMA26'] = EMAIndicator(df_features['close'], window=26).ema_indicator()
            
            # 8. High/Low диапазоны
            df_features['high_low_pct'] = (df_features['high'] - df_features['low']) / df_features['close']
            df_features['close_to_high'] = (df_features['high'] - df_features['close']) / df_features['close']
            df_features['close_to_low'] = (df_features['close'] - df_features['low']) / df_features['close']
            
            # Удаляем NaN значения
            df_features = df_features.dropna()
            
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            return None
    
    def get_feature_columns(self) -> list:
        """Получение списка колонок с признаками."""
        return [
            'MA5', 'MA10', 'MA20',
            'RSI', 
            'MACD', 'MACD_signal', 'MACD_diff',
            'BB_hband', 'BB_lband', 'BB_width', 'BB_position',
            'vol_change', 'volume_sma', 'volume_ratio',
            'price_change', 'price_change_5', 'volatility',
            'EMA12', 'EMA26',
            'high_low_pct', 'close_to_high', 'close_to_low'
        ]
    
    def get_latest_features(self, exclude_ohlcv: bool = True) -> Optional[pd.DataFrame]:
        """
        Получение последних рассчитанных признаков.
        
        Args:
            exclude_ohlcv: Исключить OHLCV колонки
            
        Returns:
            DataFrame с последними признаками или None
        """
        if self.last_features is None:
            return None
        
        if exclude_ohlcv:
            feature_cols = self.get_feature_columns()
            # Добавляем timestamp для контекста
            cols_to_include = ['timestamp'] + [col for col in feature_cols if col in self.last_features.columns]
            return self.last_features[cols_to_include].copy()
        
        return self.last_features.copy()
    
    def prepare_for_model(self, features_df: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """
        Подготовка признаков для модели ML.
        
        Args:
            features_df: DataFrame с признаками (если None, используются последние)
            
        Returns:
            numpy array готовых для модели признаков или None
        """
        if features_df is None:
            features_df = self.get_latest_features(exclude_ohlcv=True)
        
        if features_df is None:
            return None
        
        # Получаем только признаки (без timestamp)
        feature_cols = self.get_feature_columns()
        available_cols = [col for col in feature_cols if col in features_df.columns]
        
        if not available_cols:
            return None
        
        # Берем последнюю строку и преобразуем в numpy array
        features_array = features_df[available_cols].iloc[-1:].values
        
        # Проверяем на NaN
        if np.isnan(features_array).any():
            self.logger.warning("NaN values found in features")
            return None
        
        return features_array
    
    def get_feature_summary(self) -> Optional[Dict[str, Any]]:
        """Получение сводки по последним признакам."""
        if self.last_features is None:
            return None
        
        latest = self.last_features.iloc[-1]
        
        return {
            'timestamp': latest['timestamp'],
            'price': latest['close'] if 'close' in latest else None,
            'MA5': latest.get('MA5'),
            'MA10': latest.get('MA10'),
            'MA20': latest.get('MA20'),
            'RSI': latest.get('RSI'),
            'MACD_signal': 'BUY' if latest.get('MACD_diff', 0) > 0 else 'SELL',
            'BB_position': latest.get('BB_position'),
            'volume_ratio': latest.get('volume_ratio'),
            'volatility': latest.get('volatility')
        }


class FeatureValidator:
    """Валидатор для проверки качества признаков."""
    
    @staticmethod
    def validate_features(features_df: pd.DataFrame) -> Dict[str, Any]:
        """Валидация признаков."""
        validation_result = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        # Проверка на NaN
        nan_counts = features_df.isnull().sum()
        if nan_counts.sum() > 0:
            validation_result['valid'] = False
            validation_result['issues'].append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Проверка на inf
        inf_counts = np.isinf(features_df.select_dtypes(include=[np.number])).sum()
        if inf_counts.sum() > 0:
            validation_result['valid'] = False
            validation_result['issues'].append(f"Inf values found: {inf_counts[inf_counts > 0].to_dict()}")
        
        # Статистики
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        validation_result['stats'] = {
            'shape': features_df.shape,
            'numeric_columns': len(numeric_cols),
            'data_range': {
                col: {'min': features_df[col].min(), 'max': features_df[col].max()}
                for col in numeric_cols[:5]  # Показываем только первые 5
            }
        }
        
        return validation_result


# Пример использования
def example_usage():
    """Демонстрация использования генератора признаков."""
    import datetime
    
    # Создаем тестовые данные
    dates = pd.date_range(start='2025-01-01', periods=50, freq='5T')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 105, 50),
        'high': np.random.uniform(105, 110, 50),
        'low': np.random.uniform(95, 100, 50),
        'close': np.random.uniform(100, 105, 50),
        'volume': np.random.uniform(1000, 5000, 50)
    })
    
    # Создаем генератор признаков
    generator = LiveFeatureGenerator()
    
    # Обновляем данные
    features = generator.update_data(test_data)
    
    if features is not None:
        print(f"Generated features: {features.shape}")
        print(f"Feature columns: {generator.get_feature_columns()}")
        
        # Получаем последние признаки
        latest = generator.get_latest_features()
        if latest is not None:
            print(f"Latest features shape: {latest.shape}")
        
        # Подготовка для модели
        model_features = generator.prepare_for_model()
        if model_features is not None:
            print(f"Model features shape: {model_features.shape}")
        
        # Сводка
        summary = generator.get_feature_summary()
        print(f"Feature summary: {summary}")
        
        # Валидация
        validation = FeatureValidator.validate_features(features)
        print(f"Validation result: {validation}")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    example_usage()