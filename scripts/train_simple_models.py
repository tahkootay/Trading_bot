#!/usr/bin/env python3
"""
Простое обучение ML моделей без XGBoost (только scikit-learn)
"""

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
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

class SimpleMLTrainer:
    """Простой тренер ML моделей на scikit-learn"""
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self, data_dir: str = "data") -> pd.DataFrame:
        """Загрузка всех доступных данных"""
        data_path = Path(data_dir)
        
        # Ищем файлы с данными (5-минутный таймфрейм)
        data_files = []
        for pattern in ['*5m*.csv', '*_real_*.csv', '*_august_*.csv']:
            data_files.extend(data_path.glob(pattern))
            data_files.extend(data_path.glob(f"blocks/data/*/{pattern}"))
        
        print(f"📊 Найдено {len(data_files)} файлов данных")
        
        dfs = []
        for file_path in data_files:
            if '5m' in file_path.name:  # Фокусируемся на 5-минутных данных
                try:
                    df = pd.read_csv(file_path)
                    
                    # Проверяем наличие необходимых колонок
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_cols):
                        # Настройка индекса времени
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df = df.set_index('timestamp')
                        
                        dfs.append(df)
                        print(f"✅ Загружен: {file_path.name} ({len(df)} строк)")
                
                except Exception as e:
                    print(f"❌ Ошибка загрузки {file_path.name}: {e}")
        
        if not dfs:
            raise ValueError("Не найдено подходящих данных для обучения")
        
        # Объединяем и очищаем данные
        combined_df = pd.concat(dfs, axis=0)
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        print(f"📈 Объединено: {len(combined_df)} записей")
        print(f"📅 Период: {combined_df.index[0]} - {combined_df.index[-1]}")
        
        return combined_df
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет простых технических индикаторов"""
        print("🔧 Расчет технических индикаторов...")
        
        features_df = df.copy()
        
        # Простые скользящие средние
        for period in [5, 10, 20, 50]:
            features_df[f'sma_{period}'] = df['close'].rolling(period).mean()
            features_df[f'price_vs_sma_{period}'] = (df['close'] - features_df[f'sma_{period}']) / features_df[f'sma_{period}']
        
        # RSI (упрощенный)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        features_df['rsi'] = 100 - (100 / (1 + rs))
        
        # Простой MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features_df['macd'] = ema_12 - ema_26
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features_df['bb_upper'] = sma_20 + (std_20 * 2)
        features_df['bb_lower'] = sma_20 - (std_20 * 2)
        features_df['bb_position'] = (df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # Волатильность
        features_df['volatility'] = df['close'].pct_change().rolling(20).std()
        
        # Изменения цены
        for period in [1, 5, 10, 20]:
            features_df[f'price_change_{period}'] = df['close'].pct_change(period)
        
        # Объемные индикаторы
        features_df['volume_sma'] = df['volume'].rolling(20).mean()
        features_df['volume_ratio'] = df['volume'] / features_df['volume_sma']
        
        # Максимумы и минимумы
        features_df['high_20'] = df['high'].rolling(20).max()
        features_df['low_20'] = df['low'].rolling(20).min()
        features_df['price_position'] = (df['close'] - features_df['low_20']) / (features_df['high_20'] - features_df['low_20'])
        
        print(f"✅ Рассчитано {len(features_df.columns)} признаков")
        return features_df
    
    def create_targets(self, df: pd.DataFrame, forward_periods: int = 12) -> pd.DataFrame:
        """Создание целевых переменных"""
        print(f"🎯 Создание целевых переменных (прогноз на {forward_periods} периодов)...")
        
        target_df = df.copy()
        
        # Будущая цена
        future_price = df['close'].shift(-forward_periods)
        current_price = df['close']
        
        # Процентное изменение
        price_change = (future_price - current_price) / current_price
        
        # Классификация с адаптивными порогами
        # Используем процентили для адаптации к волатильности
        up_threshold = price_change.quantile(0.75)    # Топ 25% движений
        down_threshold = price_change.quantile(0.25)  # Низ 25% движений
        
        print(f"📊 Пороги классификации: UP > {up_threshold:.3%}, DOWN < {down_threshold:.3%}")
        
        # Создание таргетов
        target_df['target'] = 1  # Flat (по умолчанию)
        target_df.loc[price_change > up_threshold, 'target'] = 2   # Up
        target_df.loc[price_change < down_threshold, 'target'] = 0  # Down
        
        # Удаляем последние строки без будущих данных
        target_df = target_df.iloc[:-forward_periods]
        
        # Статистика
        target_counts = target_df['target'].value_counts().sort_index()
        print(f"📈 Распределение классов:")
        print(f"  DOWN (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(target_df)*100:.1f}%)")
        print(f"  FLAT (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(target_df)*100:.1f}%)")
        print(f"  UP (2):   {target_counts.get(2, 0)} ({target_counts.get(2, 0)/len(target_df)*100:.1f}%)")
        
        return target_df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Подготовка данных для обучения"""
        print("📋 Подготовка данных для обучения...")
        
        # Выбор признаков (исключаем исходные OHLCV и служебные)
        feature_cols = [col for col in df.columns if col not in 
                       ['open', 'high', 'low', 'close', 'volume', 'turnover', 'target']]
        
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        # Удаление строк с NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Замена бесконечности и оставшихся NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"✅ Подготовлено: {len(X)} образцов с {len(X.columns)} признаками")
        print(f"📋 Признаки: {', '.join(X.columns[:5])}{'...' if len(X.columns) > 5 else ''}")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, model_version: str = None):
        """Обучение ансамбля простых моделей"""
        print("🤖 Обучение ML моделей...")
        
        if model_version is None:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Разделение на обучение/тест
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Разделение данных:")
        print(f"  Обучение: {len(X_train)} образцов")
        print(f"  Тест: {len(X_test)} образцов")
        
        # Масштабирование признаков
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Модели для обучения
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=10,
                random_state=42
            )
        }
        
        trained_models = {}
        model_scores = {}
        
        # Обучение каждой модели
        for model_name, model in models.items():
            print(f"🔄 Обучение {model_name}...")
            
            # Обучение
            if 'forest' in model_name or 'tree' in model_name:
                model.fit(X_train, y_train)  # Tree-based models не нуждаются в масштабировании
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Оценка
            accuracy = accuracy_score(y_test, y_pred)
            
            trained_models[model_name] = model
            model_scores[model_name] = accuracy
            
            print(f"  ✅ {model_name}: точность = {accuracy:.3f}")
        
        # Сохранение моделей
        model_dir = self.output_dir / model_version
        model_dir.mkdir(exist_ok=True)
        
        for model_name, model in trained_models.items():
            model_path = model_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
        
        # Сохранение scaler и метаданных
        joblib.dump(scaler, model_dir / "scaler.joblib")
        joblib.dump(X.columns.tolist(), model_dir / "feature_names.joblib")
        
        # Метаданные
        metadata = {
            "model_version": model_version,
            "training_date": datetime.now().isoformat(),
            "samples_trained": len(X_train),
            "samples_tested": len(X_test),
            "features_count": len(X.columns),
            "feature_names": X.columns.tolist(),
            "model_scores": model_scores,
            "ensemble_weights": {
                "random_forest": 0.4,
                "gradient_boosting": 0.4,
                "decision_tree": 0.2
            }
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Создание символической ссылки на latest
        latest_link = self.output_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(model_version)
        
        print(f"💾 Модели сохранены в: {model_dir}")
        print(f"🔗 Создана ссылка: {latest_link}")
        
        # Показать лучшие признаки (для Random Forest)
        if 'random_forest' in trained_models:
            feature_importance = trained_models['random_forest'].feature_importances_
            feature_names = X.columns
            
            # Топ-10 признаков
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"🏆 Топ-10 важных признаков:")
            for i, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return model_scores

@click.command()
@click.option('--data-dir', '-d', default='data', help='Директория с данными')
@click.option('--models-dir', '-m', default='models', help='Директория для моделей')
@click.option('--forward-periods', '-f', default=12, help='Периодов вперед для прогноза')
@click.option('--min-samples', default=500, help='Минимум образцов для обучения')
def main(data_dir: str, models_dir: str, forward_periods: int, min_samples: int):
    """Простое обучение ML моделей на scikit-learn"""
    
    print("🚀 Простое обучение ML моделей (без XGBoost)")
    print("=" * 50)
    
    try:
        trainer = SimpleMLTrainer(output_dir=models_dir)
        
        # 1. Загрузка данных
        df = trainer.load_data(data_dir)
        
        if len(df) < min_samples:
            print(f"⚠️  Внимание: данных мало ({len(df)} < {min_samples}), но продолжаем...")
        
        # 2. Расчет признаков
        features_df = trainer.calculate_features(df)
        
        # 3. Создание таргетов
        target_df = trainer.create_targets(features_df, forward_periods)
        
        # 4. Подготовка данных
        X, y = trainer.prepare_training_data(target_df)
        
        if len(X) < min_samples // 2:
            print(f"❌ После обработки слишком мало данных: {len(X)}")
            return False
        
        # 5. Обучение
        scores = trainer.train_models(X, y)
        
        # Результат
        avg_score = np.mean(list(scores.values()))
        print("=" * 50)
        print(f"🎉 Обучение завершено!")
        print(f"📊 Средняя точность: {avg_score:.3f}")
        
        if avg_score > 0.4:
            print("✅ Модели готовы к использованию!")
            print("🔄 Перезапустите торгового бота для загрузки новых моделей")
        else:
            print("⚠️  Качество моделей низкое, но они сохранены для тестирования")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)