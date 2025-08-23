#!/usr/bin/env python3
"""
ПОЛНОЕ ПРОДУКЦИОННОЕ ОБУЧЕНИЕ АНСАМБЛЯ ML-МОДЕЛЕЙ
=====================================================

Обучение на ВСЕХ имеющихся данных за 90 дней без ограничений:
- Все 168K+ записей из 8 таймфреймов
- Максимальные параметры моделей для лучшего качества
- Расширенная кросс-валидация
- Детальная аналитика и метрики
- Продукционное качество моделей

ВРЕМЯ ВЫПОЛНЕНИЯ: ~15-30 минут
ТРЕБОВАНИЯ ОЗУ: ~2-4 ГБ
"""

import sys
import warnings
import time
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# ML и валидация
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.linear_model import LogisticRegression

# Базовые ML модели
from sklearn.ensemble import RandomForestClassifier
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available")

import xgboost as xgb
import catboost as cb

# Технический анализ
import ta
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

warnings.filterwarnings('ignore')


class ProductionEnsembleTrainer:
    """Продукционный тренер ансамбля с максимальным качеством."""
    
    def __init__(self, output_dir: str = "models/ensemble_live"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # МАКСИМАЛЬНЫЕ параметры моделей для продукционного качества
        print("🔧 Инициализация продукционных моделей...")
        
        # Random Forest - увеличенные параметры
        self.base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=500,      # Увеличено с 100
                max_depth=15,          # Увеличено с 8
                min_samples_split=5,   # Уменьшено для большей детализации
                min_samples_leaf=2,    # Уменьшено для большей детализации
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,        # Out-of-bag scoring
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        }
        
        # XGBoost - продукционные параметры
        self.base_models['xgboost'] = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,      # Уменьшено для стабильности
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=3,      # Для дисбаланса классов
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=1
        )
        
        # CatBoost - продукционные параметры
        self.base_models['catboost'] = cb.CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            border_count=128,
            feature_border_type='GreedyLogSum',
            bootstrap_type='Bayesian',
            bagging_temperature=1,
            random_state=42,
            verbose=100,  # Каждые 100 итераций
            early_stopping_rounds=50
        )
        
        # LightGBM если доступен
        if LIGHTGBM_AVAILABLE:
            self.base_models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=3,
                random_state=42,
                n_jobs=-1,
                verbose=100,
                early_stopping_rounds=50
            )
        
        # Продвинутая метамодель
        self.meta_model = LogisticRegression(
            C=0.1,                    # Регуляризация
            penalty='elasticnet',     # Elastic Net регуляризация
            l1_ratio=0.5,            # Баланс L1/L2
            solver='saga',           # Solver для elastic net
            max_iter=5000,           # Увеличено для сходимости
            random_state=42,
            n_jobs=-1
        )
        
        # Скалер
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Метрики и статистики
        self.training_stats = {}
        self.feature_importance = {}
        
        print(f"✅ Инициализированы {len(self.base_models)} базовых модели")
        print(f"📊 Ожидаемое время обучения: 15-30 минут")
    
    def load_full_dataset(self) -> pd.DataFrame:
        """Загрузка ВСЕХ доступных данных без ограничений."""
        print("\n" + "="*60)
        print("📊 ЗАГРУЗКА ПОЛНОГО ДАТАСЕТА (БЕЗ ОГРАНИЧЕНИЙ)")
        print("="*60)
        
        data_dir = Path("data/bybit_futures")
        combined_data = []
        total_original = 0
        
        # Загружаем ВСЕ файлы полностью
        for file_path in sorted(data_dir.glob("*90d_bybit_futures.csv")):
            timeframe = file_path.name.split("_")[1]
            
            try:
                start_time = time.time()
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['timeframe'] = timeframe
                
                combined_data.append(df)
                total_original += len(df)
                
                load_time = time.time() - start_time
                print(f"   ✅ {timeframe:>4s}: {len(df):>7,} записей ({load_time:.1f}s)")
                
            except Exception as e:
                print(f"   ❌ {timeframe}: Ошибка - {e}")
                continue
        
        if not combined_data:
            raise ValueError("Не удалось загрузить данные")
        
        # Объединяем все данные
        print("\n🔄 Объединение всех данных...")
        start_time = time.time()
        
        all_data = pd.concat(combined_data, ignore_index=True)
        all_data = all_data.sort_values(['timeframe', 'timestamp']).reset_index(drop=True)
        
        combine_time = time.time() - start_time
        
        print(f"✅ ПОЛНЫЙ ДАТАСЕТ ЗАГРУЖЕН:")
        print(f"   📊 Всего записей: {len(all_data):,}")
        print(f"   📅 Период: {all_data['timestamp'].min()} → {all_data['timestamp'].max()}")
        print(f"   ⏰ Таймфреймы: {sorted(all_data['timeframe'].unique())}")
        print(f"   ⏱️  Время объединения: {combine_time:.1f}s")
        print(f"   💾 Размер в памяти: ~{all_data.memory_usage(deep=True).sum() / 1024**2:.0f} МБ")
        
        return all_data
    
    def generate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Генерация расширенного набора технических признаков."""
        print("\n" + "="*60)  
        print("🔧 ГЕНЕРАЦИЯ РАСШИРЕННЫХ ТЕХНИЧЕСКИХ ПРИЗНАКОВ")
        print("="*60)
        
        processed_groups = []
        total_features_generated = 0
        
        for timeframe in sorted(df['timeframe'].unique()):
            tf_data = df[df['timeframe'] == timeframe].copy()
            tf_data = tf_data.sort_values('timestamp').reset_index(drop=True)
            
            print(f"\n📈 Обработка {timeframe} ({len(tf_data):,} записей)...")
            start_time = time.time()
            
            # Базовые Moving Averages
            tf_data['MA5'] = tf_data['close'].rolling(5).mean()
            tf_data['MA10'] = tf_data['close'].rolling(10).mean()
            tf_data['MA20'] = tf_data['close'].rolling(20).mean()
            tf_data['MA50'] = tf_data['close'].rolling(50).mean()
            tf_data['MA100'] = tf_data['close'].rolling(100).mean()
            
            # Exponential Moving Averages
            tf_data['EMA8'] = EMAIndicator(tf_data['close'], window=8).ema_indicator()
            tf_data['EMA12'] = EMAIndicator(tf_data['close'], window=12).ema_indicator()
            tf_data['EMA26'] = EMAIndicator(tf_data['close'], window=26).ema_indicator()
            tf_data['EMA50'] = EMAIndicator(tf_data['close'], window=50).ema_indicator()
            
            # MA кроссоверы
            tf_data['MA5_MA20_cross'] = (tf_data['MA5'] > tf_data['MA20']).astype(int)
            tf_data['EMA12_EMA26_cross'] = (tf_data['EMA12'] > tf_data['EMA26']).astype(int)
            
            # RSI семейство
            tf_data['RSI14'] = RSIIndicator(tf_data['close'], window=14).rsi()
            tf_data['RSI21'] = RSIIndicator(tf_data['close'], window=21).rsi()
            tf_data['RSI_overbought'] = (tf_data['RSI14'] > 70).astype(int)
            tf_data['RSI_oversold'] = (tf_data['RSI14'] < 30).astype(int)
            
            # MACD расширенный
            macd = MACD(tf_data['close'])
            tf_data['MACD'] = macd.macd()
            tf_data['MACD_signal'] = macd.macd_signal()
            tf_data['MACD_diff'] = tf_data['MACD'] - tf_data['MACD_signal']
            tf_data['MACD_histogram'] = macd.macd_diff()
            tf_data['MACD_bullish'] = (tf_data['MACD_diff'] > 0).astype(int)
            
            # Bollinger Bands расширенный
            bb = BollingerBands(tf_data['close'], window=20)
            tf_data['BB_hband'] = bb.bollinger_hband()
            tf_data['BB_lband'] = bb.bollinger_lband()
            tf_data['BB_width'] = bb.bollinger_wband()
            tf_data['BB_position'] = (tf_data['close'] - tf_data['BB_lband']) / (tf_data['BB_hband'] - tf_data['BB_lband'])
            tf_data['BB_squeeze'] = (tf_data['BB_width'] < tf_data['BB_width'].rolling(20).mean() * 0.8).astype(int)
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(tf_data['high'], tf_data['low'], tf_data['close'])
            tf_data['Stoch_K'] = stoch.stoch()
            tf_data['Stoch_D'] = stoch.stoch_signal()
            tf_data['Stoch_overbought'] = (tf_data['Stoch_K'] > 80).astype(int)
            tf_data['Stoch_oversold'] = (tf_data['Stoch_K'] < 20).astype(int)
            
            # Williams %R
            tf_data['Williams_R'] = WilliamsRIndicator(tf_data['high'], tf_data['low'], tf_data['close']).williams_r()
            
            # ADX (Average Directional Index)
            try:
                adx = ADXIndicator(tf_data['high'], tf_data['low'], tf_data['close'])
                tf_data['ADX'] = adx.adx()
                tf_data['ADX_strong_trend'] = (tf_data['ADX'] > 25).astype(int)
            except:
                tf_data['ADX'] = 0
                tf_data['ADX_strong_trend'] = 0
            
            # Average True Range (волатильность)
            tf_data['ATR'] = AverageTrueRange(tf_data['high'], tf_data['low'], tf_data['close']).average_true_range()
            tf_data['ATR_normalized'] = tf_data['ATR'] / tf_data['close']
            
            # Volume индикаторы
            tf_data['volume_sma5'] = tf_data['volume'].rolling(5).mean()
            tf_data['volume_sma20'] = tf_data['volume'].rolling(20).mean()
            tf_data['volume_ratio'] = tf_data['volume'] / tf_data['volume_sma20']
            tf_data['vol_change'] = tf_data['volume'].pct_change()
            tf_data['volume_spike'] = (tf_data['volume'] > tf_data['volume_sma20'] * 2).astype(int)
            
            # On Balance Volume
            try:
                tf_data['OBV'] = OnBalanceVolumeIndicator(tf_data['close'], tf_data['volume']).on_balance_volume()
                tf_data['OBV_sma'] = tf_data['OBV'].rolling(20).mean()
                tf_data['OBV_divergence'] = (tf_data['OBV'] > tf_data['OBV_sma']).astype(int)
            except:
                tf_data['OBV'] = 0
                tf_data['OBV_sma'] = 0
                tf_data['OBV_divergence'] = 0
            
            # Price action индикаторы
            tf_data['price_change'] = tf_data['close'].pct_change()
            tf_data['price_change_5'] = tf_data['close'].pct_change(5)
            tf_data['price_change_10'] = tf_data['close'].pct_change(10)
            tf_data['price_change_20'] = tf_data['close'].pct_change(20)
            
            # Volatility
            tf_data['volatility_5'] = tf_data['close'].pct_change().rolling(5).std()
            tf_data['volatility_20'] = tf_data['close'].pct_change().rolling(20).std()
            tf_data['volatility_50'] = tf_data['close'].pct_change().rolling(50).std()
            
            # High/Low анализ
            tf_data['high_low_pct'] = (tf_data['high'] - tf_data['low']) / tf_data['close']
            tf_data['close_to_high'] = (tf_data['high'] - tf_data['close']) / tf_data['close']
            tf_data['close_to_low'] = (tf_data['close'] - tf_data['low']) / tf_data['close']
            tf_data['upper_shadow'] = (tf_data['high'] - tf_data[['open', 'close']].max(axis=1)) / tf_data['close']
            tf_data['lower_shadow'] = (tf_data[['open', 'close']].min(axis=1) - tf_data['low']) / tf_data['close']
            
            # Gap анализ
            tf_data['gap'] = (tf_data['open'] - tf_data['close'].shift(1)) / tf_data['close'].shift(1)
            tf_data['gap_up'] = (tf_data['gap'] > 0.001).astype(int)  # >0.1% gap
            tf_data['gap_down'] = (tf_data['gap'] < -0.001).astype(int)
            
            # Support/Resistance levels
            tf_data['close_above_MA20'] = (tf_data['close'] > tf_data['MA20']).astype(int)
            tf_data['close_above_MA50'] = (tf_data['close'] > tf_data['MA50']).astype(int)
            tf_data['price_momentum'] = tf_data['close'] / tf_data['close'].shift(10) - 1
            
            # Timeframe encoding
            timeframe_mapping = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30, '60m': 60, '1h': 60,
                '120m': 120, '2h': 120, '240m': 240, '4h': 240,
                '720m': 720, '12h': 720, 'D': 1440, '1d': 1440
            }
            tf_data['timeframe_minutes'] = timeframe_mapping.get(timeframe, 5)
            
            # Удаляем NaN
            original_len = len(tf_data)
            tf_data = tf_data.dropna()
            dropped = original_len - len(tf_data)
            
            processing_time = time.time() - start_time
            
            if len(tf_data) > 0:
                processed_groups.append(tf_data)
                feature_count = len([col for col in tf_data.columns 
                                   if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'timeframe']])
                total_features_generated += feature_count
                
                print(f"   ✅ {len(tf_data):>6,} записей | {feature_count:>2d} признаков | "
                      f"Удалено NaN: {dropped:>4,} | Время: {processing_time:.1f}s")
            else:
                print(f"   ⚠️  Все записи удалены после обработки NaN")
        
        # Объединяем обработанные данные
        print(f"\n🔄 Объединение обработанных данных...")
        start_time = time.time()
        
        all_features = pd.concat(processed_groups, ignore_index=True)
        all_features = all_features.sort_values(['timeframe', 'timestamp']).reset_index(drop=True)
        
        combine_time = time.time() - start_time
        
        print(f"✅ ГЕНЕРАЦИЯ ПРИЗНАКОВ ЗАВЕРШЕНА:")
        print(f"   📊 Итого записей: {len(all_features):,}")
        print(f"   🔢 Среднее признаков: {total_features_generated // len(processed_groups)}")
        print(f"   ⏱️  Время объединения: {combine_time:.1f}s")
        print(f"   💾 Размер в памяти: ~{all_features.memory_usage(deep=True).sum() / 1024**2:.0f} МБ")
        
        return all_features
    
    def create_advanced_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание продвинутых таргетов с учетом специфики таймфреймов."""
        print("\n" + "="*60)
        print("🎯 СОЗДАНИЕ ПРОДВИНУТЫХ ТАРГЕТОВ")
        print("="*60)
        
        # Параметры для каждого таймфрейма (оптимизированные)
        timeframe_params = {
            '1m': {'horizon': 30, 'threshold': 0.002},   # 30 минут, 0.2%
            '5m': {'horizon': 12, 'threshold': 0.005},   # 1 час, 0.5%
            '15m': {'horizon': 8,  'threshold': 0.008},  # 2 часа, 0.8%
            '60m': {'horizon': 6,  'threshold': 0.012},  # 6 часов, 1.2%
            '120m': {'horizon': 4, 'threshold': 0.015},  # 8 часов, 1.5%
            '240m': {'horizon': 3, 'threshold': 0.020},  # 12 часов, 2.0%
            '720m': {'horizon': 2, 'threshold': 0.030},  # 1 день, 3.0%
            'D': {'horizon': 2,    'threshold': 0.050}   # 2 дня, 5.0%
        }
        
        processed_groups = []
        total_buy_signals = 0
        total_records = 0
        
        for timeframe in sorted(df['timeframe'].unique()):
            tf_data = df[df['timeframe'] == timeframe].copy()
            
            params = timeframe_params.get(timeframe, {'horizon': 5, 'threshold': 0.005})
            horizon = params['horizon']
            threshold = params['threshold']
            
            print(f"📈 {timeframe:>4s}: горизонт={horizon:>2d} свечей, порог={threshold:.1%}")
            
            # Создаем таргеты
            tf_data['future_return'] = tf_data['close'].shift(-horizon) / tf_data['close'] - 1
            tf_data['target'] = (tf_data['future_return'] > threshold).astype(int)
            
            # Удаляем записи без таргетов
            tf_data = tf_data[tf_data['future_return'].notna()].copy()
            
            if len(tf_data) > 0:
                processed_groups.append(tf_data)
                
                target_dist = tf_data['target'].value_counts().sort_index()
                buy_count = target_dist.get(1, 0)
                buy_pct = buy_count / len(tf_data) * 100
                
                total_buy_signals += buy_count
                total_records += len(tf_data)
                
                print(f"        {len(tf_data):>7,} записей | BUY: {buy_count:>5,} ({buy_pct:>5.1f}%)")
        
        # Объединяем все данные
        all_targets = pd.concat(processed_groups, ignore_index=True)
        
        final_dist = all_targets['target'].value_counts().sort_index()
        print(f"\n📊 ИТОГОВОЕ РАСПРЕДЕЛЕНИЕ ТАРГЕТОВ:")
        print(f"   0 (HOLD/SELL): {final_dist.get(0, 0):>7,} ({final_dist.get(0, 0)/len(all_targets)*100:>5.1f}%)")
        print(f"   1 (BUY):       {final_dist.get(1, 0):>7,} ({final_dist.get(1, 0)/len(all_targets)*100:>5.1f}%)")
        print(f"   📊 Общий баланс классов: {final_dist.get(1, 0) / final_dist.get(0, 1):.3f}")
        
        return all_targets
    
    def train_production_models(self, X_train, X_test, y_train, y_test):
        """Продукционное обучение моделей с детальным логированием."""
        print("\n" + "="*60)
        print("🤖 ПРОДУКЦИОННОЕ ОБУЧЕНИЕ БАЗОВЫХ МОДЕЛЕЙ")
        print("="*60)
        
        trained_models = {}
        base_predictions = {}
        model_scores = {}
        
        # Расширенная кросс-валидация
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.base_models.items():
            print(f"\n🔧 Обучение {name.upper()}...")
            start_time = time.time()
            
            # Обучение модели
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Предсказания для валидации
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Детальные метрики
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            # Кросс-валидация
            cv_start = time.time()
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_time = time.time() - cv_start
            
            # Сохранение результатов
            model_scores[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_time': train_time,
                'cv_time': cv_time
            }
            
            trained_models[name] = model
            base_predictions[name] = model.predict_proba(X_train)[:, 1]
            
            print(f"   ✅ Точность: {accuracy:.4f}")
            print(f"   📊 Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            print(f"   🎯 AUC-ROC: {auc:.4f}")
            print(f"   🔄 CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"   ⏱️  Время обучения: {train_time:.1f}s | CV: {cv_time:.1f}s")
            
            # Важность признаков для tree-based моделей
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(self.feature_names, model.feature_importances_))
        
        self.training_stats['base_models'] = model_scores
        return trained_models, base_predictions
    
    def train_advanced_meta_model(self, base_predictions, y_train):
        """Обучение продвинутой метамодели."""
        print(f"\n🎯 ОБУЧЕНИЕ ПРОДВИНУТОЙ МЕТАМОДЕЛИ...")
        
        meta_features = pd.DataFrame(base_predictions)
        
        # Добавляем полиномиальные признаки для метамодели
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        meta_features_poly = poly.fit_transform(meta_features)
        
        start_time = time.time()
        
        # Обучение с кросс-валидацией
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.meta_model, meta_features_poly, y_train, cv=cv, scoring='accuracy')
        
        # Финальное обучение
        self.meta_model.fit(meta_features_poly, y_train)
        
        train_time = time.time() - start_time
        
        print(f"   ✅ Meta CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   ⏱️  Время: {train_time:.1f}s")
        print(f"   🔢 Признаков метамодели: {meta_features_poly.shape[1]}")
        
        self.training_stats['meta_model'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_time': train_time,
            'n_features': meta_features_poly.shape[1]
        }
        
        # Сохраняем poly transformer
        self.poly_transformer = poly
        
        return self.meta_model
    
    def evaluate_production_ensemble(self, X_test, y_test, trained_models):
        """Продукционная оценка ансамбля."""
        print(f"\n📊 ДЕТАЛЬНАЯ ОЦЕНКА АНСАМБЛЯ...")
        
        # Предсказания базовых моделей
        test_base_predictions = {}
        for name, model in trained_models.items():
            test_base_predictions[name] = model.predict_proba(X_test)[:, 1]
        
        # Подготовка данных для метамодели
        meta_test_features = pd.DataFrame(test_base_predictions)
        meta_test_features_poly = self.poly_transformer.transform(meta_test_features)
        
        # Предсказания ансамбля
        ensemble_predictions = self.meta_model.predict(meta_test_features_poly)
        ensemble_probabilities = self.meta_model.predict_proba(meta_test_features_poly)[:, 1]
        
        # Детальные метрики
        accuracy = accuracy_score(y_test, ensemble_predictions)
        precision = precision_score(y_test, ensemble_predictions)
        recall = recall_score(y_test, ensemble_predictions)
        f1 = f1_score(y_test, ensemble_predictions)
        auc = roc_auc_score(y_test, ensemble_probabilities)
        
        print(f"🎯 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ АНСАМБЛЯ:")
        print(f"   📊 Accuracy:  {accuracy:.4f}")
        print(f"   📈 Precision: {precision:.4f}")
        print(f"   📉 Recall:    {recall:.4f}")
        print(f"   ⚖️  F1-Score:  {f1:.4f}")
        print(f"   🎲 AUC-ROC:   {auc:.4f}")
        
        print(f"\n📋 ДЕТАЛЬНЫЙ ОТЧЕТ:")
        print(classification_report(y_test, ensemble_predictions, digits=4))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, ensemble_predictions)
        print(f"\n🎭 CONFUSION MATRIX:")
        print(f"   True Neg: {cm[0,0]:>6,} | False Pos: {cm[0,1]:>6,}")
        print(f"   False Neg: {cm[1,0]:>5,} | True Pos:  {cm[1,1]:>6,}")
        
        self.training_stats['ensemble'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'confusion_matrix': cm.tolist()
        }
        
        return ensemble_predictions, ensemble_probabilities, test_base_predictions
    
    def save_production_models(self, trained_models, feature_names):
        """Сохранение продукционных моделей."""
        print(f"\n💾 СОХРАНЕНИЕ ПРОДУКЦИОННЫХ МОДЕЛЕЙ...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.output_dir / f"{timestamp}_production_full"
        model_dir.mkdir(exist_ok=True)
        
        # Сохранение базовых моделей
        for name, model in trained_models.items():
            joblib.dump(model, model_dir / f"{name}_intraday.joblib")
            print(f"   ✅ {name}")
        
        # Сохранение метамодели и компонентов
        joblib.dump(self.meta_model, model_dir / "meta_intraday.joblib")
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        joblib.dump(feature_names, model_dir / "feature_names.joblib")
        joblib.dump(self.poly_transformer, model_dir / "poly_transformer.joblib")
        
        # Сохранение статистик обучения
        joblib.dump(self.training_stats, model_dir / "training_stats.joblib")
        joblib.dump(self.feature_importance, model_dir / "feature_importance.joblib")
        
        print(f"   ✅ meta_model, scaler, feature_names")
        print(f"   ✅ poly_transformer, training_stats, feature_importance")
        
        # Обновление latest
        latest_link = self.output_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(f"{timestamp}_production_full")
        
        print(f"🎉 Модели сохранены: {model_dir}")
        print(f"🔗 Latest обновлен: {latest_link}")
        
        return model_dir
    
    def run_full_production_pipeline(self):
        """Запуск полного продукционного пайплайна."""
        print("\n" + "="*80)
        print("🚀 ПОЛНОЕ ПРОДУКЦИОННОЕ ОБУЧЕНИЕ АНСАМБЛЯ")
        print("   Без ограничений • Максимальное качество • Все данные")
        print("="*80)
        
        total_start_time = time.time()
        
        # 1. Загрузка полного датасета
        combined_data = self.load_full_dataset()
        
        # 2. Генерация расширенных признаков
        features_data = self.generate_advanced_features(combined_data)
        
        # 3. Создание продвинутых таргетов
        targets_data = self.create_advanced_targets(features_data)
        
        # 4. Подготовка данных для ML
        feature_cols = [col for col in targets_data.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 
                                     'volume', 'target', 'future_return', 'timeframe']]
        
        X = targets_data[feature_cols]
        y = targets_data['target']
        
        print(f"\n📋 ФИНАЛЬНЫЕ ДАННЫЕ ДЛЯ ОБУЧЕНИЯ:")
        print(f"   🔢 Признаков: {len(feature_cols)}")
        print(f"   📊 Записей: {len(X):,}")
        print(f"   💾 Размер: ~{X.memory_usage(deep=True).sum() / 1024**2:.0f} МБ")
        
        self.feature_names = feature_cols
        
        # 5. Нормализация
        print(f"\n🔧 Нормализация данных...")
        normalize_start = time.time()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        normalize_time = time.time() - normalize_start
        print(f"   ✅ Нормализация завершена ({normalize_time:.1f}s)")
        
        # 6. Разделение на train/test
        print(f"\n📊 Разделение на train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42, stratify=y
        )
        
        print(f"   📈 Train: {len(X_train):,} записей ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   📉 Test:  {len(X_test):,} записей ({len(X_test)/len(X)*100:.1f}%)")
        
        # 7. Обучение базовых моделей
        trained_models, base_predictions = self.train_production_models(
            X_train, X_test, y_train, y_test
        )
        
        # 8. Обучение метамодели
        meta_model = self.train_advanced_meta_model(base_predictions, y_train)
        
        # 9. Оценка ансамбля
        ensemble_preds, ensemble_probs, test_base_preds = self.evaluate_production_ensemble(
            X_test, y_test, trained_models
        )
        
        # 10. Сохранение моделей
        model_dir = self.save_production_models(trained_models, feature_cols)
        
        # Финальная статистика
        total_time = time.time() - total_start_time
        
        print("\n" + "="*80)
        print("🎉 ПОЛНОЕ ПРОДУКЦИОННОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("="*80)
        print(f"⏱️  ОБЩЕЕ ВРЕМЯ: {total_time//60:.0f}м {total_time%60:.0f}с")
        print(f"📁 МОДЕЛИ: {model_dir.name}")
        print(f"🎯 ТОЧНОСТЬ: {self.training_stats['ensemble']['accuracy']:.4f}")
        print(f"🔢 ПРИЗНАКОВ: {len(feature_cols)}")
        print(f"📊 ДАННЫХ: {len(X):,} записей")
        print(f"🎲 AUC-ROC: {self.training_stats['ensemble']['auc_roc']:.4f}")
        print("="*80)
        
        return {
            'model_dir': model_dir,
            'accuracy': self.training_stats['ensemble']['accuracy'],
            'auc_roc': self.training_stats['ensemble']['auc_roc'],
            'total_samples': len(X),
            'feature_count': len(feature_cols),
            'training_time': total_time,
            'stats': self.training_stats
        }


def main():
    """Основная функция запуска."""
    print("🔥 ЗАПУСК ПОЛНОГО ПРОДУКЦИОННОГО ОБУЧЕНИЯ")
    print("   ⚠️  ВНИМАНИЕ: Процесс может занять 15-30 минут")
    print("   💾 Требования: ~2-4 ГБ ОЗУ")
    print()
    
    try:
        trainer = ProductionEnsembleTrainer()
        results = trainer.run_full_production_pipeline()
        
        print(f"\n🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   🎯 Точность: {results['accuracy']:.4f}")
        print(f"   🎲 AUC-ROC: {results['auc_roc']:.4f}")
        print(f"   📊 Обучено на: {results['total_samples']:,} записях")
        print(f"   🔢 Признаков: {results['feature_count']}")
        print(f"   ⏱️  Время: {results['training_time']//60:.0f}м {results['training_time']%60:.0f}с")
        print(f"   📁 Директория: {results['model_dir'].name}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 ПРОДУКЦИОННОЕ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print(f"🚀 Новые модели готовы к live торговле!")
    else:
        print(f"\n💥 ОБУЧЕНИЕ ЗАВЕРШИЛОСЬ С ОШИБКОЙ!")
        sys.exit(1)