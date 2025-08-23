#!/usr/bin/env python3
"""
Оптимизированное обучение ансамбля на всех таймфреймах за 90 дней
С выборкой данных для ускорения процесса
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# ML Models
from sklearn.ensemble import RandomForestClassifier
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import xgboost as xgb
import catboost as cb

# Technical Analysis
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

warnings.filterwarnings('ignore')


class OptimizedEnsembleTrainer:
    """Оптимизированный тренер ансамбля."""
    
    def __init__(self, max_samples_per_tf: int = 10000, output_dir: str = "models/ensemble_live"):
        self.max_samples_per_tf = max_samples_per_tf
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Оптимизированные модели (меньше параметров для скорости)
        self.base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,  # Уменьшено
                max_depth=8,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'catboost': cb.CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
        }
        
        # Добавляем LightGBM если доступен
        if LIGHTGBM_AVAILABLE:
            self.base_models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        self.meta_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def load_optimized_data(self) -> pd.DataFrame:
        """Загрузка данных с выборкой для оптимизации."""
        print("📊 Загрузка данных с оптимизацией...")
        
        data_dir = Path("data/bybit_futures")
        combined_data = []
        
        # Приоритет таймфреймов (более важные = больше данных)
        priority_mapping = {
            '5m': 1.0,    # 100% данных
            '15m': 0.8,   # 80% данных  
            '1m': 0.3,    # 30% данных (слишком много)
            '60m': 0.7,   # 70% данных
            '240m': 1.0,  # 100% данных
            'D': 1.0,     # 100% данных
            '120m': 1.0,  # 100% данных
            '720m': 1.0   # 100% данных
        }
        
        total_loaded = 0
        
        for file_path in sorted(data_dir.glob("*90d_bybit_futures.csv")):
            timeframe = file_path.name.split("_")[1]
            
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                original_count = len(df)
                
                # Применяем выборку если нужно
                sample_ratio = priority_mapping.get(timeframe, 0.5)
                max_samples = int(self.max_samples_per_tf * sample_ratio)
                
                if len(df) > max_samples:
                    # Стратифицированная выборка по времени
                    df = df.sample(n=max_samples, random_state=42).sort_values('timestamp')
                
                df['timeframe'] = timeframe
                combined_data.append(df)
                
                total_loaded += len(df)
                print(f"   {timeframe}: {len(df):,}/{original_count:,} записей ({len(df)/original_count*100:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ Ошибка {timeframe}: {e}")
        
        if not combined_data:
            raise ValueError("Нет данных для обучения")
        
        all_data = pd.concat(combined_data, ignore_index=True)
        all_data = all_data.sort_values(['timeframe', 'timestamp']).reset_index(drop=True)
        
        print(f"\n✅ Загружено: {total_loaded:,} записей")
        print(f"📅 Период: {all_data['timestamp'].min()} → {all_data['timestamp'].max()}")
        print(f"⏰ Таймфреймы: {sorted(all_data['timeframe'].unique())}")
        
        return all_data
    
    def generate_features_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Быстрая генерация признаков."""
        print("⚡ Быстрая генерация признаков...")
        
        processed_groups = []
        
        for timeframe in df['timeframe'].unique():
            tf_data = df[df['timeframe'] == timeframe].copy()
            tf_data = tf_data.sort_values('timestamp').reset_index(drop=True)
            
            # Базовые индикаторы
            tf_data['MA5'] = tf_data['close'].rolling(5).mean()
            tf_data['MA10'] = tf_data['close'].rolling(10).mean()
            tf_data['MA20'] = tf_data['close'].rolling(20).mean()
            
            # RSI
            tf_data['RSI'] = RSIIndicator(tf_data['close'], window=14).rsi()
            
            # MACD
            macd = MACD(tf_data['close'])
            tf_data['MACD'] = macd.macd()
            tf_data['MACD_signal'] = macd.macd_signal()
            tf_data['MACD_diff'] = tf_data['MACD'] - tf_data['MACD_signal']
            
            # Bollinger Bands
            bb = BollingerBands(tf_data['close'], window=20)
            tf_data['BB_hband'] = bb.bollinger_hband()
            tf_data['BB_lband'] = bb.bollinger_lband()
            tf_data['BB_width'] = bb.bollinger_wband()
            tf_data['BB_position'] = (tf_data['close'] - tf_data['BB_lband']) / (tf_data['BB_hband'] - tf_data['BB_lband'])
            
            # Volume и Price
            tf_data['vol_change'] = tf_data['volume'].pct_change()
            tf_data['volume_sma'] = tf_data['volume'].rolling(20).mean()
            tf_data['volume_ratio'] = tf_data['volume'] / tf_data['volume_sma']
            tf_data['price_change'] = tf_data['close'].pct_change()
            tf_data['price_change_5'] = tf_data['close'].pct_change(5)
            tf_data['volatility'] = tf_data['close'].pct_change().rolling(20).std()
            
            # EMA
            tf_data['EMA12'] = EMAIndicator(tf_data['close'], window=12).ema_indicator()
            tf_data['EMA26'] = EMAIndicator(tf_data['close'], window=26).ema_indicator()
            
            # High/Low
            tf_data['high_low_pct'] = (tf_data['high'] - tf_data['low']) / tf_data['close']
            tf_data['close_to_high'] = (tf_data['high'] - tf_data['close']) / tf_data['close']
            tf_data['close_to_low'] = (tf_data['close'] - tf_data['low']) / tf_data['close']
            
            # Timeframe encoding
            timeframe_mapping = {'1m': 1, '5m': 5, '15m': 15, '60m': 60, '120m': 120, '240m': 240, '720m': 720, 'D': 1440}
            tf_data['timeframe_minutes'] = timeframe_mapping.get(timeframe, 5)
            
            # Убираем NaN
            tf_data = tf_data.dropna()
            
            if len(tf_data) > 0:
                processed_groups.append(tf_data)
                print(f"   {timeframe}: {len(tf_data):,} записей с признаками")
        
        all_features = pd.concat(processed_groups, ignore_index=True)
        print(f"✅ Итого признаков: {len(all_features):,} записей")
        
        return all_features
    
    def create_targets_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Быстрое создание таргетов."""
        print("🎯 Создание таргетов...")
        
        # Единые параметры для всех таймфреймов (упрощение)
        future_bars = 5
        threshold = 0.005  # 0.5%
        
        processed_groups = []
        
        for timeframe in df['timeframe'].unique():
            tf_data = df[df['timeframe'] == timeframe].copy()
            
            tf_data['future_return'] = tf_data['close'].shift(-future_bars) / tf_data['close'] - 1
            tf_data['target'] = (tf_data['future_return'] > threshold).astype(int)
            tf_data = tf_data[tf_data['future_return'].notna()].copy()
            
            if len(tf_data) > 0:
                processed_groups.append(tf_data)
                
                target_dist = tf_data['target'].value_counts().sort_index()
                buy_pct = target_dist.get(1, 0) / len(tf_data) * 100
                print(f"   {timeframe}: {len(tf_data):,} записей | BUY: {target_dist.get(1, 0):,} ({buy_pct:.1f}%)")
        
        all_targets = pd.concat(processed_groups, ignore_index=True)
        
        total_dist = all_targets['target'].value_counts().sort_index()
        print(f"\n📊 Общее распределение:")
        for target, count in total_dist.items():
            print(f"   {target}: {count:,} ({count/len(all_targets)*100:.1f}%)")
        
        return all_targets
    
    def train_fast_pipeline(self):
        """Быстрый пайплайн обучения."""
        print("🚀 БЫСТРОЕ ОБУЧЕНИЕ АНСАМБЛЯ НА ВСЕХ ТАЙМФРЕЙМАХ")
        print("=" * 60)
        
        # 1. Загрузка данных
        combined_data = self.load_optimized_data()
        
        # 2. Генерация признаков
        features_data = self.generate_features_fast(combined_data)
        
        # 3. Создание таргетов
        targets_data = self.create_targets_fast(features_data)
        
        # 4. Подготовка для ML
        feature_cols = [col for col in targets_data.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 
                                     'volume', 'target', 'future_return', 'timeframe']]
        
        X = targets_data[feature_cols]
        y = targets_data['target']
        
        print(f"\n📋 Признаков: {len(feature_cols)}")
        print(f"📊 Датасет: {len(X):,} записей")
        self.feature_names = feature_cols
        
        # 5. Нормализация
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # 6. Разделение
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # 7. Обучение базовых моделей
        print("\n🤖 Обучение базовых моделей...")
        trained_models = {}
        base_predictions = {}
        
        for name, model in self.base_models.items():
            print(f"   Обучение {name}...")
            model.fit(X_train, y_train)
            
            # Оценка на тесте
            test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            print(f"   ✅ {name}: {test_acc:.4f}")
            
            trained_models[name] = model
            base_predictions[name] = model.predict_proba(X_train)[:, 1]
        
        # 8. Обучение метамодели
        print("\n🎯 Обучение метамодели...")
        meta_features = pd.DataFrame(base_predictions)
        self.meta_model.fit(meta_features, y_train)
        
        # 9. Оценка ансамбля
        test_base_predictions = {}
        for name, model in trained_models.items():
            test_base_predictions[name] = model.predict_proba(X_test)[:, 1]
        
        meta_test_features = pd.DataFrame(test_base_predictions)
        ensemble_predictions = self.meta_model.predict(meta_test_features)
        ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
        
        print(f"\n🎯 РЕЗУЛЬТАТ АНСАМБЛЯ: {ensemble_accuracy:.4f}")
        print("\n📋 Детальный отчет:")
        print(classification_report(y_test, ensemble_predictions))
        
        # 10. Сохранение
        model_dir = self.save_models(trained_models, feature_cols)
        
        print("\n" + "=" * 60)
        print("🎉 БЫСТРОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"📁 Модели: {model_dir}")
        print(f"🎯 Точность: {ensemble_accuracy:.4f}")
        print(f"📊 Обучено на: {len(X):,} записей")
        
        return {
            'model_dir': model_dir,
            'accuracy': ensemble_accuracy,
            'total_samples': len(X),
            'feature_count': len(feature_cols)
        }
    
    def save_models(self, trained_models: dict, feature_names: list):
        """Сохранение моделей."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.output_dir / f"{timestamp}_multi_tf_optimized"
        model_dir.mkdir(exist_ok=True)
        
        # Сохранение всех компонентов
        for name, model in trained_models.items():
            joblib.dump(model, model_dir / f"{name}_intraday.joblib")
        
        joblib.dump(self.meta_model, model_dir / "meta_intraday.joblib")
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        joblib.dump(feature_names, model_dir / "feature_names.joblib")
        
        # Обновление latest
        latest_link = self.output_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(f"{timestamp}_multi_tf_optimized")
        
        print(f"💾 Модели сохранены в: {model_dir}")
        return model_dir


def main():
    """Основная функция."""
    print("⚡ ЗАПУСК ОПТИМИЗИРОВАННОГО ОБУЧЕНИЯ")
    
    trainer = OptimizedEnsembleTrainer(max_samples_per_tf=15000)
    
    try:
        results = trainer.train_fast_pipeline()
        
        print(f"\n🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   🎯 Точность: {results['accuracy']:.4f}")
        print(f"   📊 Размер обучения: {results['total_samples']:,} записей")
        print(f"   🔢 Признаков: {results['feature_count']}")
        print(f"   📁 Модели: {results['model_dir'].name}")
        
        return True
        
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    else:
        print("\n❌ ОБУЧЕНИЕ ЗАВЕРШИЛОСЬ С ОШИБКОЙ!")