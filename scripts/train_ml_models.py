#!/usr/bin/env python3
"""
Скрипт для обучения ML моделей торгового бота SOL/USDT

Этапы:
1. Загрузка и подготовка исторических данных
2. Расчет технических индикаторов и признаков
3. Создание целевых переменных (таргетов)
4. Обучение ансамбля моделей (XGBoost, LightGBM, CatBoost)
5. Сохранение обученных моделей
"""

import asyncio
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import click
import json
from typing import Dict, List, Tuple, Optional

# Подавление предупреждений
warnings.filterwarnings('ignore')

# Добавление src в путь
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

try:
    from src.feature_engine.technical_indicators import TechnicalIndicatorCalculator
    from src.models.ml_models import MLModelPredictor
    from src.utils.logger import setup_logging, TradingLogger
    from src.utils.types import TimeFrame
except ImportError:
    # Fallback для совместимости с Python 3.9
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.feature_engine.technical_indicators import TechnicalIndicatorCalculator
    from src.models.ml_models import MLModelPredictor
    from src.utils.logger import setup_logging, TradingLogger
    from src.utils.types import TimeFrame

class MLModelTrainer:
    """Класс для обучения ML моделей торгового бота."""
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = TradingLogger("ml_trainer")
        self.indicator_calc = TechnicalIndicatorCalculator()
        
    def load_historical_data(self, data_dir: str = "data") -> pd.DataFrame:
        """Загрузка исторических данных."""
        data_path = Path(data_dir)
        
        # Приоритет данных: 90-дневные Bybit futures -> enhanced -> реальные -> блоки -> testnet
        data_files = [
            # 90-дневные Bybit futures данные (наивысший приоритет)
            "data_bybit_futures_90d/SOLUSDT_5m_90d_bybit_futures.csv",
            "data_bybit_futures_90d/SOLUSDT_1m_90d_bybit_futures.csv",
            "data_bybit_futures_90d/SOLUSDT_15m_90d_bybit_futures.csv",
            # 90-дневные enhanced данные 
            "SOLUSDT_5m_90d_enhanced.csv",
            "SOLUSDT_5m_real_90d.csv",
            # Реальные данные (предпочтительно)
            "SOLUSDT_5m_real_7d.csv",
            "SOLUSDT_5m_real_2025-08-10_to_2025-08-17.csv",
            # Блоки данных
            "blocks/data/august_10_17_full/SOLUSDT_5m_august_10_17_full.csv",
            "blocks/data/august_12_single_day/SOLUSDT_5m_august_12_single_day.csv",
            # Testnet как резерв
            "SOLUSDT_5m_testnet.csv"
        ]
        
        dfs = []
        for file_name in data_files:
            file_path = data_path / file_name
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')
                    elif 'open_time' in df.columns:
                        df['open_time'] = pd.to_datetime(df['open_time'])
                        df = df.set_index('open_time')
                    
                    # Проверка необходимых колонок
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_cols):
                        dfs.append(df)
                        self.logger.log_system_event(
                            event_type="data_loaded",
                            component="ml_trainer",
                            status="success",
                            details={"file": file_name, "rows": len(df)}
                        )
                    else:
                        self.logger.log_error(
                            error_type="missing_columns",
                            component="ml_trainer", 
                            error_message=f"Missing required columns in {file_name}"
                        )
                        
                except Exception as e:
                    self.logger.log_error(
                        error_type="data_load_failed",
                        component="ml_trainer",
                        error_message=str(e),
                        details={"file": file_name}
                    )
        
        if not dfs:
            raise ValueError("Не найдены подходящие файлы данных")
        
        # Объединение всех данных
        combined_df = pd.concat(dfs, axis=0)
        combined_df = combined_df.sort_index()
        
        # Удаление дубликатов
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        self.logger.log_system_event(
            event_type="data_combined",
            component="ml_trainer",
            status="success",
            details={
                "total_rows": len(combined_df),
                "date_range": f"{combined_df.index[0]} to {combined_df.index[-1]}",
                "files_used": len(dfs)
            }
        )
        
        return combined_df
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет технических индикаторов и признаков."""
        try:
            # Базовые технические индикаторы
            features_df = df.copy()
            
            # Moving Averages
            features_df['ema_8'] = df['close'].ewm(span=8).mean()
            features_df['ema_13'] = df['close'].ewm(span=13).mean()
            features_df['ema_21'] = df['close'].ewm(span=21).mean()
            features_df['ema_34'] = df['close'].ewm(span=34).mean()
            features_df['ema_55'] = df['close'].ewm(span=55).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            features_df['macd'] = exp1 - exp2
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            features_df['macd_hist'] = features_df['macd'] - features_df['macd_signal']
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            features_df['atr'] = true_range.rolling(window=14).mean()
            
            # ADX
            features_df['adx'] = self._calculate_adx(df)
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = df['close'].rolling(window=bb_period).mean()
            bb_std_dev = df['close'].rolling(window=bb_period).std()
            features_df['bb_upper'] = bb_middle + (bb_std_dev * bb_std)
            features_df['bb_lower'] = bb_middle - (bb_std_dev * bb_std)
            features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / bb_middle
            
            # VWAP
            features_df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            # Volume features
            features_df['volume_sma'] = df['volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = df['volume'] / features_df['volume_sma']
            
            # Price features
            features_df['price_change_1'] = df['close'].pct_change(1)
            features_df['price_change_5'] = df['close'].pct_change(5)
            features_df['price_change_15'] = df['close'].pct_change(15)
            
            # Volatility features
            features_df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            # Momentum features
            features_df['momentum'] = df['close'] / df['close'].shift(10) - 1
            
            # Support/Resistance features (упрощенно)
            features_df['high_20'] = df['high'].rolling(window=20).max()
            features_df['low_20'] = df['low'].rolling(window=20).min()
            features_df['distance_to_high'] = (features_df['high_20'] - df['close']) / df['close']
            features_df['distance_to_low'] = (df['close'] - features_df['low_20']) / df['close']
            
            # Trend features
            features_df['ema_trend'] = (features_df['ema_8'] > features_df['ema_21']).astype(int)
            features_df['price_vs_vwap'] = (df['close'] > features_df['vwap']).astype(int)
            
            self.logger.log_system_event(
                event_type="features_calculated",
                component="ml_trainer",
                status="success",
                details={"features_count": len(features_df.columns)}
            )
            
            return features_df
            
        except Exception as e:
            self.logger.log_error(
                error_type="feature_calculation_failed",
                component="ml_trainer",
                error_message=str(e)
            )
            raise
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет ADX (Average Directional Index)."""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # True Range
            tr1 = high - low
            tr2 = np.abs(high - close.shift())
            tr3 = np.abs(low - close.shift())
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Directional Movement
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # Smoothed values
            atr = true_range.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            # ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return adx
            
        except Exception:
            return pd.Series(25.0, index=df.index)  # Fallback value
    
    def create_targets(self, df: pd.DataFrame, forward_periods: int = 12) -> pd.DataFrame:
        """Создание целевых переменных для обучения."""
        target_df = df.copy()
        
        # Будущие цены для определения направления движения
        future_close = df['close'].shift(-forward_periods)
        current_close = df['close']
        
        # Расчет будущего изменения цены в %
        future_return = (future_close - current_close) / current_close
        
        # Пороги для классификации движений
        # Адаптированы под волатильность SOL/USDT (~2-5% движения)
        up_threshold = 0.015   # 1.5% рост
        down_threshold = -0.015  # 1.5% падение
        
        # Создание таргетов
        target_df['target'] = 0  # Flat/sideways
        target_df.loc[future_return > up_threshold, 'target'] = 1   # Up
        target_df.loc[future_return < down_threshold, 'target'] = 2  # Down
        
        # Дополнительные целевые переменные для регрессии
        target_df['future_return'] = future_return
        target_df['future_volatility'] = df['close'].pct_change().rolling(forward_periods).std().shift(-forward_periods)
        
        # Удаление последних строк где нет будущих данных
        target_df = target_df.iloc[:-forward_periods]
        
        # Статистика таргетов
        target_counts = target_df['target'].value_counts()
        self.logger.log_system_event(
            event_type="targets_created",
            component="ml_trainer", 
            status="success",
            details={
                "target_distribution": target_counts.to_dict(),
                "forward_periods": forward_periods,
                "up_threshold": up_threshold,
                "down_threshold": down_threshold
            }
        )
        
        return target_df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Подготовка данных для обучения."""
        # Признаки для обучения
        feature_columns = [
            # Moving averages
            'ema_8', 'ema_13', 'ema_21', 'ema_34', 'ema_55',
            # Technical indicators  
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr', 'adx',
            # Bollinger bands
            'bb_width',
            # Volume
            'volume_ratio', 
            # Price changes
            'price_change_1', 'price_change_5', 'price_change_15',
            # Other features
            'volatility', 'momentum', 'distance_to_high', 'distance_to_low',
            'ema_trend', 'price_vs_vwap'
        ]
        
        # Фильтрация существующих колонок
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Подготовка X и y
        X = df[available_features].copy()
        y = df['target'].copy()
        
        # Удаление строк с NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Замена inf на NaN и затем на 0
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        self.logger.log_system_event(
            event_type="training_data_prepared",
            component="ml_trainer",
            status="success", 
            details={
                "features_count": len(available_features),
                "samples_count": len(X),
                "target_distribution": y.value_counts().to_dict()
            }
        )
        
        return X, y
    
    async def train_models(self, X: pd.DataFrame, y: pd.Series, model_version: str = None) -> bool:
        """Обучение ML моделей."""
        try:
            if model_version is None:
                model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Создание директории для моделей
            model_dir = self.output_dir / model_version
            model_dir.mkdir(exist_ok=True)
            
            # Создание экземпляра MLModelPredictor
            ml_predictor = MLModelPredictor(models_path=self.output_dir)
            
            # Подготовка DataFrame для обучения
            training_data = X.copy()
            training_data['target'] = y
            
            # Обучение моделей
            success = ml_predictor.train_models(
                training_data=training_data,
                target_column='target',
                model_version=model_version
            )
            
            if success:
                self.logger.log_system_event(
                    event_type="models_trained",
                    component="ml_trainer",
                    status="success",
                    details={
                        "model_version": model_version,
                        "training_samples": len(X),
                        "features": X.columns.tolist()
                    }
                )
                
                # Создание symbolic link на latest версию
                latest_link = self.output_dir / "latest"
                if latest_link.exists():
                    latest_link.unlink()
                latest_link.symlink_to(model_version)
                
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.log_error(
                error_type="model_training_failed",
                component="ml_trainer", 
                error_message=str(e)
            )
            return False


@click.command()
@click.option('--data-dir', '-d', default='data', help='Directory with historical data')
@click.option('--models-dir', '-m', default='models', help='Output directory for trained models')
@click.option('--forward-periods', '-f', default=12, help='Forward periods for target creation (5min * 12 = 1 hour)')
@click.option('--model-version', '-v', default=None, help='Model version (default: timestamp)')
@click.option('--min-samples', default=1000, help='Minimum samples required for training')
def main(data_dir: str, models_dir: str, forward_periods: int, model_version: str, min_samples: int):
    """Обучение ML моделей для торгового бота SOL/USDT."""
    
    # Настройка логирования
    setup_logging("INFO")
    logger = TradingLogger("ml_trainer_main")
    
    logger.log_system_event(
        event_type="training_started",
        component="ml_trainer_main",
        status="starting",
        details={
            "data_dir": data_dir,
            "models_dir": models_dir,
            "forward_periods": forward_periods,
            "min_samples": min_samples
        }
    )
    
    async def run_training():
        try:
            # Создание тренера
            trainer = MLModelTrainer(output_dir=models_dir)
            
            # 1. Загрузка данных
            print("📊 Загрузка исторических данных...")
            df = trainer.load_historical_data(data_dir)
            
            if len(df) < min_samples:
                raise ValueError(f"Недостаточно данных: {len(df)} < {min_samples}")
            
            print(f"✅ Загружено {len(df)} записей с {df.index[0]} по {df.index[-1]}")
            
            # 2. Расчет признаков
            print("🔧 Расчет технических индикаторов...")
            features_df = trainer.calculate_features(df)
            
            # 3. Создание таргетов
            print("🎯 Создание целевых переменных...")
            target_df = trainer.create_targets(features_df, forward_periods)
            
            # 4. Подготовка данных
            print("📋 Подготовка данных для обучения...")
            X, y = trainer.prepare_training_data(target_df)
            
            if len(X) < min_samples:
                raise ValueError(f"После обработки недостаточно данных: {len(X)} < {min_samples}")
            
            print(f"📈 Подготовлено {len(X)} образцов с {len(X.columns)} признаками")
            print(f"📊 Распределение классов: {y.value_counts().to_dict()}")
            
            # 5. Обучение моделей
            print("🤖 Обучение ML моделей...")
            success = await trainer.train_models(X, y, model_version)
            
            if success:
                print(f"✅ Модели успешно обучены и сохранены в {models_dir}")
                print(f"🔗 Создана ссылка на latest версию")
                
                # Информация об использовании
                print("\n📚 Для использования обученных моделей:")
                print(f"1. Убедитесь, что папка {models_dir} доступна для торгового бота")
                print("2. Перезапустите торгового бота - он автоматически загрузит обученные модели")
                print("3. Проверьте логи на сообщения о загрузке моделей")
                
            else:
                print("❌ Ошибка при обучении моделей")
                return False
            
            return True
            
        except Exception as e:
            logger.log_error(
                error_type="training_failed",
                component="ml_trainer_main",
                error_message=str(e)
            )
            print(f"❌ Ошибка: {str(e)}")
            return False
    
    # Запуск обучения
    success = asyncio.run(run_training())
    
    if success:
        print("\n🎉 Обучение ML моделей завершено успешно!")
        sys.exit(0)
    else:
        print("\n💥 Обучение завершилось с ошибками")
        sys.exit(1)


if __name__ == "__main__":
    main()