#!/usr/bin/env python3
"""
Обучение ансамбля моделей на всех имеющихся таймфреймах за 90 дней
Более качественное обучение с использованием множественных источников данных
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# ML Models
from sklearn.ensemble import RandomForestClassifier
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ Warning: LightGBM not available, will use GradientBoosting instead")

import xgboost as xgb
import catboost as cb
from sklearn.ensemble import GradientBoostingClassifier

# Technical Analysis
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

warnings.filterwarnings('ignore')


class MultiTimeframeEnsembleTrainer:
    """Тренер ансамбля с использованием множественных таймфреймов."""
    
    def __init__(self, output_dir: str = "models/ensemble_live"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Базовые модели
        self.base_models = {}
        
        # Random Forest
        self.base_models['random_forest'] = RandomForestClassifier(
            n_estimators=300,  # Увеличено для лучшего качества
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM или GradientBoosting
        if LIGHTGBM_AVAILABLE:
            self.base_models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.05,  # Уменьшено для стабильности
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            self.base_models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        
        # XGBoost
        self.base_models['xgboost'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        # CatBoost
        self.base_models['catboost'] = cb.CatBoostClassifier(
            iterations=300,
            depth=10,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False
        )
        
        # Метамодель
        self.meta_model = LogisticRegression(
            C=1.0,
            random_state=42,
            max_iter=2000
        )
        
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def find_90d_data_files(self) -> dict:
        """Поиск всех файлов с 90-дневными данными."""
        data_dir = Path("data/bybit_futures")
        
        timeframe_files = {}
        
        for file_path in data_dir.glob("*90d_bybit_futures.csv"):
            # Извлекаем таймфрейм из названия файла
            filename = file_path.name
            parts = filename.split("_")
            if len(parts) >= 2:
                timeframe = parts[1]  # например, "5m", "1h", "1d"
                timeframe_files[timeframe] = file_path
        
        print(f"🔍 Найдено {len(timeframe_files)} файлов данных:")
        for tf, path in timeframe_files.items():
            print(f"   {tf}: {path.name}")
        
        return timeframe_files
    
    def load_and_combine_data(self, timeframe_files: dict) -> pd.DataFrame:
        """Загрузка и объединение данных из разных таймфреймов."""
        combined_data = []
        
        for timeframe, file_path in timeframe_files.items():
            try:
                print(f"📊 Загружаем данные {timeframe}...")
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['timeframe'] = timeframe
                
                print(f"   ✅ {timeframe}: {len(df):,} записей")
                combined_data.append(df)
                
            except Exception as e:
                print(f"   ❌ Ошибка загрузки {timeframe}: {e}")
                continue
        
        if not combined_data:
            raise ValueError("Не удалось загрузить ни одного файла данных")
        
        # Объединяем все данные
        all_data = pd.concat(combined_data, ignore_index=True)
        all_data = all_data.sort_values(['timeframe', 'timestamp']).reset_index(drop=True)
        
        print(f"\n📈 Общий датасет: {len(all_data):,} записей")
        print(f"📅 Период: {all_data['timestamp'].min()} → {all_data['timestamp'].max()}")
        print(f"⏰ Таймфреймы: {sorted(all_data['timeframe'].unique())}")
        
        return all_data
    
    def generate_features_multi_tf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация признаков для мультитаймфрейм данных.
        """
        print("🔧 Генерация признаков по таймфреймам...")
        
        processed_groups = []
        
        # Обрабатываем каждый таймфрейм отдельно
        for timeframe in df['timeframe'].unique():
            tf_data = df[df['timeframe'] == timeframe].copy()
            tf_data = tf_data.sort_values('timestamp').reset_index(drop=True)
            
            print(f"   Обработка {timeframe}: {len(tf_data):,} записей...")
            
            # Генерируем технические индикаторы
            tf_features = self._generate_single_tf_features(tf_data, timeframe)
            
            if tf_features is not None and len(tf_features) > 0:
                processed_groups.append(tf_features)
                print(f"   ✅ {timeframe}: {len(tf_features):,} записей с признаками")
        
        if not processed_groups:
            raise ValueError("Не удалось сгенерировать признаки ни для одного таймфрейма")
        
        # Объединяем все обработанные данные
        all_features = pd.concat(processed_groups, ignore_index=True)
        all_features = all_features.sort_values(['timeframe', 'timestamp']).reset_index(drop=True)
        
        print(f"✅ Итого признаков: {len(all_features):,} записей")
        
        return all_features
    
    def _generate_single_tf_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Генерация признаков для одного таймфрейма."""
        try:
            df_features = df.copy()
            
            # Базовые признаки
            df_features['MA5'] = df_features['close'].rolling(5).mean()
            df_features['MA10'] = df_features['close'].rolling(10).mean()
            df_features['MA20'] = df_features['close'].rolling(20).mean()
            
            # RSI
            rsi = RSIIndicator(df_features['close'], window=14)
            df_features['RSI'] = rsi.rsi()
            
            # MACD
            macd = MACD(df_features['close'])
            df_features['MACD'] = macd.macd()
            df_features['MACD_signal'] = macd.macd_signal()
            df_features['MACD_diff'] = df_features['MACD'] - df_features['MACD_signal']
            
            # Bollinger Bands
            bb = BollingerBands(df_features['close'], window=20)
            df_features['BB_hband'] = bb.bollinger_hband()
            df_features['BB_lband'] = bb.bollinger_lband()
            df_features['BB_width'] = bb.bollinger_wband()
            df_features['BB_position'] = (df_features['close'] - df_features['BB_lband']) / (df_features['BB_hband'] - df_features['BB_lband'])
            
            # Volume признаки
            df_features['vol_change'] = df_features['volume'].pct_change()
            df_features['volume_sma'] = df_features['volume'].rolling(20).mean()
            df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma']
            
            # Price признаки
            df_features['price_change'] = df_features['close'].pct_change()
            df_features['price_change_5'] = df_features['close'].pct_change(5)
            df_features['volatility'] = df_features['close'].pct_change().rolling(20).std()
            
            # EMA
            df_features['EMA12'] = EMAIndicator(df_features['close'], window=12).ema_indicator()
            df_features['EMA26'] = EMAIndicator(df_features['close'], window=26).ema_indicator()
            
            # High/Low диапазоны
            df_features['high_low_pct'] = (df_features['high'] - df_features['low']) / df_features['close']
            df_features['close_to_high'] = (df_features['high'] - df_features['close']) / df_features['close']
            df_features['close_to_low'] = (df_features['close'] - df_features['low']) / df_features['close']
            
            # Добавляем таймфрейм как категориальную переменную
            timeframe_mapping = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30, '60m': 60, '1h': 60,
                '120m': 120, '240m': 240, '720m': 720, 'D': 1440
            }
            df_features['timeframe_minutes'] = timeframe_mapping.get(timeframe, 5)
            
            # Удаляем NaN
            df_features = df_features.dropna()
            
            return df_features
            
        except Exception as e:
            print(f"      ❌ Ошибка генерации признаков для {timeframe}: {e}")
            return None
    
    def create_targets_multi_tf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание таргетов с учетом таймфреймов."""
        print("🎯 Создание таргетов...")
        
        # Разные горизонты прогноза для разных таймфреймов
        horizon_mapping = {
            '1m': 15,    # 15 минут вперед
            '5m': 6,     # 30 минут вперед  
            '15m': 4,    # 1 час вперед
            '30m': 4,    # 2 часа вперед
            '60m': 4,    # 4 часа вперед
            '120m': 4,   # 8 часов вперед
            '240m': 3,   # 12 часов вперед
            '720m': 2,   # 1 день вперед
            'D': 2       # 2 дня вперед
        }
        
        # Разные пороги для разных таймфреймов
        threshold_mapping = {
            '1m': 0.001,   # 0.1%
            '5m': 0.002,   # 0.2%
            '15m': 0.005,  # 0.5%
            '30m': 0.008,  # 0.8%
            '60m': 0.01,   # 1.0%
            '120m': 0.015, # 1.5%
            '240m': 0.02,  # 2.0%
            '720m': 0.03,  # 3.0%
            'D': 0.05      # 5.0%
        }
        
        processed_groups = []
        
        for timeframe in df['timeframe'].unique():
            tf_data = df[df['timeframe'] == timeframe].copy()
            
            future_bars = horizon_mapping.get(timeframe, 5)
            threshold = threshold_mapping.get(timeframe, 0.002)
            
            # Создаем таргет для этого таймфрейма
            tf_data['future_return'] = tf_data['close'].shift(-future_bars) / tf_data['close'] - 1
            tf_data['target'] = (tf_data['future_return'] > threshold).astype(int)
            
            # Удаляем строки без таргетов
            tf_data = tf_data[tf_data['future_return'].notna()].copy()
            
            if len(tf_data) > 0:
                processed_groups.append(tf_data)
                
                target_dist = tf_data['target'].value_counts().sort_index()
                print(f"   {timeframe}: {len(tf_data):,} записей | Threshold: {threshold:.1%} | BUY: {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(tf_data)*100:.1f}%)")
        
        # Объединяем все данные
        all_targets = pd.concat(processed_groups, ignore_index=True)
        
        total_dist = all_targets['target'].value_counts().sort_index()
        print(f"\n📊 Общее распределение таргетов:")
        for target, count in total_dist.items():
            print(f"   {target}: {count:,} ({count/len(all_targets)*100:.1f}%)")
        
        return all_targets
    
    def train_ensemble_pipeline(self):
        """Полный пайплайн обучения ансамбля."""
        print("🚀 Запуск обучения на мультитаймфрейм данных...")
        print("=" * 60)
        
        # 1. Найти и загрузить все данные
        timeframe_files = self.find_90d_data_files()
        if not timeframe_files:
            raise ValueError("Не найдено файлов с данными")
        
        combined_data = self.load_and_combine_data(timeframe_files)
        
        # 2. Генерация признаков
        features_data = self.generate_features_multi_tf(combined_data)
        
        # 3. Создание таргетов
        targets_data = self.create_targets_multi_tf(features_data)
        
        # 4. Подготовка данных для ML
        feature_cols = [col for col in targets_data.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 
                                     'volume', 'target', 'future_return', 'timeframe']]
        
        X = targets_data[feature_cols]
        y = targets_data['target']
        
        print(f"\n📋 Итоговые признаки: {len(feature_cols)}")
        print(f"📊 Размер датасета: {len(X):,} записей")
        self.feature_names = feature_cols
        
        # 5. Нормализация
        print("🔧 Нормализация признаков...")
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # 6. Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Разделение: Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # 7. Обучение базовых моделей
        trained_models, base_predictions = self.train_base_models(X_train, y_train)
        
        # 8. Обучение метамодели
        meta_model = self.train_meta_model(base_predictions, y_train)
        
        # 9. Оценка ансамбля
        ensemble_preds, ensemble_probs, test_base_preds = self.evaluate_ensemble(
            X_test, y_test, trained_models
        )
        
        # 10. Сохранение моделей
        model_dir = self.save_models(trained_models, feature_cols)
        
        print("\n" + "=" * 60)
        print("🎉 Обучение мультитаймфрейм ансамбля завершено!")
        print(f"📁 Модели сохранены в: {model_dir}")
        print(f"🎯 Финальная точность: {accuracy_score(y_test, ensemble_preds):.4f}")
        print(f"📊 Размер обучающей выборки: {len(X_train):,} записей")
        print(f"🔢 Количество таймфреймов: {len(timeframe_files)}")
        
        return {
            'model_dir': model_dir,
            'accuracy': accuracy_score(y_test, ensemble_preds),
            'feature_names': feature_cols,
            'timeframes_used': list(timeframe_files.keys()),
            'total_samples': len(X)
        }
    
    def train_base_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
        """Обучение базовых моделей."""
        print("\n🤖 Обучение базовых моделей...")
        
        trained_models = {}
        base_predictions = {}
        
        # Используем кросс-валидацию для более надежной оценки
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.base_models.items():
            print(f"   Обучение {name}...")
            
            # Обучение
            model.fit(X_train, y_train)
            
            # Кросс-валидация
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            print(f"   ✅ {name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Сохранение
            trained_models[name] = model
            
            # Предсказания для метамодели
            base_predictions[name] = model.predict_proba(X_train)[:, 1]
        
        return trained_models, base_predictions
    
    def train_meta_model(self, base_predictions: dict, y_train: pd.Series):
        """Обучение метамодели."""
        print("\n🎯 Обучение метамодели (Logistic Regression)...")
        
        # Создание фичей для метамодели
        meta_features = pd.DataFrame(base_predictions)
        
        # Кросс-валидация метамодели
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.meta_model, meta_features, y_train, cv=cv, scoring='accuracy')
        
        # Обучение
        self.meta_model.fit(meta_features, y_train)
        
        print(f"   ✅ Meta-model: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self.meta_model
    
    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series, trained_models: dict):
        """Оценка ансамбля."""
        print("\n📊 Оценка ансамбля...")
        
        # Предсказания базовых моделей
        test_base_predictions = {}
        for name, model in trained_models.items():
            test_base_predictions[name] = model.predict_proba(X_test)[:, 1]
        
        # Предсказание метамодели
        meta_test_features = pd.DataFrame(test_base_predictions)
        ensemble_predictions = self.meta_model.predict(meta_test_features)
        ensemble_probabilities = self.meta_model.predict_proba(meta_test_features)[:, 1]
        
        # Метрики
        accuracy = accuracy_score(y_test, ensemble_predictions)
        print(f"🎯 Точность ансамбля: {accuracy:.4f}")
        
        print("\n📋 Детальный отчет:")
        print(classification_report(y_test, ensemble_predictions))
        
        return ensemble_predictions, ensemble_probabilities, test_base_predictions
    
    def save_models(self, trained_models: dict, feature_names: list):
        """Сохранение моделей."""
        print(f"\n💾 Сохранение моделей...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.output_dir / f"{timestamp}_multi_tf"
        model_dir.mkdir(exist_ok=True)
        
        # Сохранение базовых моделей
        for name, model in trained_models.items():
            joblib.dump(model, model_dir / f"{name}_intraday.joblib")
            print(f"   ✅ Сохранен {name}")
        
        # Сохранение метамодели и компонентов
        joblib.dump(self.meta_model, model_dir / "meta_intraday.joblib")
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        joblib.dump(feature_names, model_dir / "feature_names.joblib")
        print(f"   ✅ Сохранены meta_model, scaler, feature_names")
        
        # Обновление ссылки latest
        latest_link = self.output_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(f"{timestamp}_multi_tf")
        
        print(f"🎉 Все модели сохранены в: {model_dir}")
        
        return model_dir


def main():
    """Основная функция."""
    trainer = MultiTimeframeEnsembleTrainer()
    
    try:
        results = trainer.train_ensemble_pipeline()
        
        print(f"\n🎯 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   📁 Директория моделей: {results['model_dir']}")
        print(f"   🎯 Точность: {results['accuracy']:.4f}")
        print(f"   🔢 Признаков: {len(results['feature_names'])}")
        print(f"   ⏰ Таймфреймов: {len(results['timeframes_used'])} ({', '.join(results['timeframes_used'])})")
        print(f"   📊 Общий размер датасета: {results['total_samples']:,} записей")
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    else:
        print("\n❌ ОБУЧЕНИЕ ЗАВЕРШИЛОСЬ С ОШИБКОЙ!")