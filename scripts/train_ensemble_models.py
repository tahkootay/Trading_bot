#!/usr/bin/env python3
"""
Обучение ансамбля из 4 базовых моделей + метамодель для live-торговли
Согласно спецификации: RF, LightGBM, XGBoost, CatBoost + Logistic Regression
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
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
# from ta.volume import VolumeSMAIndicator  # Not available in current ta version

warnings.filterwarnings('ignore')

class EnsembleModelTrainer:
    """Тренер ансамбля из 4 базовых моделей + метамодель."""
    
    def __init__(self, output_dir: str = "models/ensemble_live"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Базовые модели
        self.base_models = {}
        
        # Random Forest - всегда доступен
        self.base_models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM - с fallback на GradientBoosting
        if LIGHTGBM_AVAILABLE:
            self.base_models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            print("⚠️  Using GradientBoostingClassifier instead of LightGBM")
            self.base_models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        
        # XGBoost
        self.base_models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        # CatBoost
        self.base_models['catboost'] = cb.CatBoostClassifier(
            iterations=200,
            depth=8,
            learning_rate=0.1,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False
        )
        
        # Метамодель
        self.meta_model = LogisticRegression(
            C=1.0,
            random_state=42,
            max_iter=1000
        )
        
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def generate_features_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация фичей для обучения согласно спецификации.
        Включает: MA5/10/20, RSI, MACD, Bollinger Bands, Volume change
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print("🔧 Generating features...")
        
        # Moving Averages (как в спецификации)
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA10'] = df['close'].rolling(10).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        
        # RSI
        rsi = RSIIndicator(df['close'], window=14)
        df['RSI'] = rsi.rsi()
        
        # MACD
        macd = MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=20)
        df['BB_hband'] = bb.bollinger_hband()
        df['BB_lband'] = bb.bollinger_lband()
        df['BB_width'] = bb.bollinger_wband()
        df['BB_position'] = (df['close'] - df['BB_lband']) / (df['BB_hband'] - df['BB_lband'])
        
        # Volume change
        df['vol_change'] = df['volume'].pct_change()
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Additional technical features
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        
        # EMA indicators 
        df['EMA12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['EMA26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # High/Low ranges
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_to_high'] = (df['high'] - df['close']) / df['close']
        df['close_to_low'] = (df['close'] - df['low']) / df['close']
        
        # Drop NaN values
        df = df.dropna()
        
        print(f"✅ Generated {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} features")
        return df
    
    def create_targets(self, df: pd.DataFrame, future_bars: int = 5, threshold: float = 0.002) -> pd.DataFrame:
        """
        Создание бинарных таргетов для классификации.
        1 = BUY (цена вырастет больше чем на threshold за future_bars свечей)
        0 = NO SIGNAL (цена не изменится значительно)
        """
        df = df.copy()
        
        # Будущая доходность
        df['future_return'] = df['close'].shift(-future_bars) / df['close'] - 1
        
        # Бинарный таргет: 1 если доходность > threshold, иначе 0
        df['target'] = (df['future_return'] > threshold).astype(int)
        
        # Удаляем строки без таргетов
        df = df[df['future_return'].notna()].copy()
        
        target_distribution = df['target'].value_counts().sort_index()
        print(f"📊 Target distribution:")
        for target, count in target_distribution.items():
            print(f"   {target}: {count:,} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def train_base_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Обучение базовых моделей."""
        print("\n🤖 Training base models...")
        
        trained_models = {}
        base_predictions = {}
        
        for name, model in self.base_models.items():
            print(f"   Training {name}...")
            
            # Обучение
            model.fit(X_train, y_train)
            
            # Валидация
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            print(f"   ✅ {name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Сохранение модели
            trained_models[name] = model
            
            # Получение предсказаний для метамодели (на обучающей выборке)
            base_predictions[name] = model.predict_proba(X_train)[:, 1]
        
        return trained_models, base_predictions
    
    def train_meta_model(self, base_predictions: dict, y_train: pd.Series):
        """Обучение метамодели на предсказаниях базовых моделей."""
        print("\n🎯 Training meta-model (Logistic Regression)...")
        
        # Создание фичей для метамодели из предсказаний базовых моделей
        meta_features = pd.DataFrame(base_predictions)
        
        # Обучение метамодели
        self.meta_model.fit(meta_features, y_train)
        
        # Валидация метамодели
        cv_scores = cross_val_score(self.meta_model, meta_features, y_train, cv=3, scoring='accuracy')
        print(f"   ✅ Meta-model: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self.meta_model
    
    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series, trained_models: dict):
        """Оценка качества ансамбля."""
        print("\n📊 Evaluating ensemble performance...")
        
        # Предсказания базовых моделей на тестовой выборке
        test_base_predictions = {}
        for name, model in trained_models.items():
            test_base_predictions[name] = model.predict_proba(X_test)[:, 1]
        
        # Предсказание метамодели
        meta_test_features = pd.DataFrame(test_base_predictions)
        ensemble_predictions = self.meta_model.predict(meta_test_features)
        ensemble_probabilities = self.meta_model.predict_proba(meta_test_features)[:, 1]
        
        # Метрики
        accuracy = accuracy_score(y_test, ensemble_predictions)
        print(f"🎯 Ensemble Accuracy: {accuracy:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, ensemble_predictions))
        
        return ensemble_predictions, ensemble_probabilities, test_base_predictions
    
    def save_models(self, trained_models: dict, feature_names: list):
        """Сохранение всех моделей."""
        print(f"\n💾 Saving models to {self.output_dir}...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.output_dir / timestamp
        model_dir.mkdir(exist_ok=True)
        
        # Сохранение базовых моделей
        for name, model in trained_models.items():
            joblib.dump(model, model_dir / f"{name}_intraday.joblib")
            print(f"   ✅ Saved {name}")
        
        # Сохранение метамодели
        joblib.dump(self.meta_model, model_dir / "meta_intraday.joblib")
        print(f"   ✅ Saved meta_model")
        
        # Сохранение скейлера и фичей
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        joblib.dump(feature_names, model_dir / "feature_names.joblib")
        print(f"   ✅ Saved scaler and feature_names")
        
        # Создание ссылки на последнюю версию
        latest_link = self.output_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(timestamp)
        
        print(f"🎉 All models saved successfully!")
        return model_dir
    
    def train_full_pipeline(self, data_file: str):
        """Полный пайплайн обучения ансамбля."""
        print("🚀 Starting ensemble training pipeline...")
        print("=" * 60)
        
        # Загрузка данных
        print(f"📊 Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"   ✅ Loaded {len(df):,} records")
        print(f"   📅 Period: {df['timestamp'].min()} → {df['timestamp'].max()}")
        
        # Генерация фичей
        df_features = self.generate_features_training(df)
        
        # Создание таргетов
        df_targets = self.create_targets(df_features)
        
        # Подготовка данных
        feature_cols = [col for col in df_targets.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 
                                     'volume', 'target', 'future_return']]
        
        X = df_targets[feature_cols]
        y = df_targets['target']
        
        print(f"\n📋 Features: {len(feature_cols)}")
        self.feature_names = feature_cols
        
        # Нормализация фичей
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Обучение базовых моделей
        trained_models, base_predictions = self.train_base_models(X_train, y_train)
        
        # Обучение метамодели
        meta_model = self.train_meta_model(base_predictions, y_train)
        
        # Оценка ансамбля
        ensemble_preds, ensemble_probs, test_base_preds = self.evaluate_ensemble(
            X_test, y_test, trained_models
        )
        
        # Сохранение моделей
        model_dir = self.save_models(trained_models, feature_cols)
        
        print("\n" + "=" * 60)
        print("🎉 Ensemble training completed successfully!")
        print(f"📁 Models saved to: {model_dir}")
        
        return {
            'model_dir': model_dir,
            'feature_names': feature_cols,
            'accuracy': accuracy_score(y_test, ensemble_preds),
            'base_models': trained_models,
            'meta_model': meta_model
        }

def main():
    """Основная функция для обучения ансамбля."""
    
    # Поиск данных для обучения
    data_files = [
        "data/bybit_futures/SOLUSDT_5m_90d_bybit_futures.csv",
        "data/enhanced/SOLUSDT_5m_90d_enhanced.csv"
    ]
    
    data_file = None
    for file_path in data_files:
        if Path(file_path).exists():
            data_file = file_path
            break
    
    if not data_file:
        print("❌ No training data found!")
        print("   Expected files:")
        for file_path in data_files:
            print(f"     {file_path}")
        return
    
    # Создание тренера и запуск обучения
    trainer = EnsembleModelTrainer()
    results = trainer.train_full_pipeline(data_file)
    
    print(f"\n🎯 Final Results:")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   Feature count: {len(results['feature_names'])}")
    print(f"   Model directory: {results['model_dir']}")

if __name__ == "__main__":
    main()