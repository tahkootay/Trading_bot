#!/usr/bin/env python3
"""
Бэктестирование ансамбля на данных 10-17 августа с прогресс-баром
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedEnsembleBacktester:
    """Оптимизированный бэктестер с ансамблем и отложенной загрузкой."""
    
    def __init__(self, models_dir: str = "models/ensemble_live"):
        self.models_dir = Path(models_dir)
        
        # Модели - загружаем по требованию
        self.base_models = {}
        self.meta_model = None
        self.scaler = None
        self.feature_names = []
        self.models_loaded = False
        
        # Параметры бэктеста
        self.initial_balance = 10000.0
        self.commission_rate = 0.001  # 0.1%
        self.slippage = 0.0005  # 0.05%
        
        # Состояние бэктеста
        self.balance = self.initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = []
        
        # Статистика
        self.processed_candles = 0
        self.signals_generated = 0
        
        logger.info(f"✅ Бэктестер создан (баланс: ${self.initial_balance:,.2f})")
    
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
        """Загрузка моделей при первом использовании."""
        if self.models_loaded:
            return True
        
        logger.info("🔄 Загрузка моделей ансамбля...")
        start_time = time.time()
        
        try:
            import joblib
            models_path = self._find_models_directory()
            
            if models_path is None:
                logger.error("❌ Директория моделей не найдена")
                return False
            
            logger.info(f"📁 Загрузка из: {models_path.name}")
            
            # Прогресс-бар для загрузки моделей
            model_files = {
                'random_forest': 'random_forest_intraday.joblib',
                'lightgbm': 'lightgbm_intraday.joblib',
                'xgboost': 'xgboost_intraday.joblib',
                'catboost': 'catboost_intraday.joblib'
            }
            
            with tqdm(total=len(model_files) + 3, desc="Загрузка моделей", unit="файл") as pbar:
                # Загружаем базовые модели
                for model_name, filename in model_files.items():
                    model_path = models_path / filename
                    if model_path.exists():
                        self.base_models[model_name] = joblib.load(model_path)
                        pbar.set_postfix({"Модель": model_name})
                    pbar.update(1)
                
                # Метамодель
                meta_path = models_path / "meta_intraday.joblib"
                if meta_path.exists():
                    self.meta_model = joblib.load(meta_path)
                    pbar.set_postfix({"Модель": "meta"})
                pbar.update(1)
                
                # Скейлер
                scaler_path = models_path / "scaler.joblib"
                if scaler_path.exists():
                    self.scaler = joblib.load(scaler_path)
                    pbar.set_postfix({"Модель": "scaler"})
                pbar.update(1)
                
                # Признаки
                features_path = models_path / "feature_names.joblib"
                if features_path.exists():
                    self.feature_names = joblib.load(features_path)
                    pbar.set_postfix({"Модель": f"{len(self.feature_names)} признаков"})
                pbar.update(1)
            
            if not self.base_models or self.meta_model is None:
                logger.error("❌ Критические модели не загружены")
                return False
            
            self.models_loaded = True
            load_time = time.time() - start_time
            logger.info(f"🎉 Модели загружены за {load_time:.2f}s ({len(self.base_models)} базовых + мета)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки моделей: {e}")
            return False
    
    def prepare_features(self, row):
        """Подготовка признаков из строки данных."""
        try:
            if not self.feature_names:
                return None
            
            # Создаем массив признаков в правильном порядке
            features_array = []
            for feature_name in self.feature_names:
                if feature_name in row:
                    value = row[feature_name]
                    # Проверяем на NaN/Inf
                    if pd.isna(value) or np.isinf(value):
                        value = 0.0
                    features_array.append(float(value))
                else:
                    # Заполняем отсутствующие признаки разумными значениями
                    if 'MA' in feature_name or 'EMA' in feature_name:
                        features_array.append(float(row.get('close', 150.0)))
                    elif 'RSI' in feature_name:
                        features_array.append(50.0)
                    elif 'BB_position' in feature_name:
                        features_array.append(0.5)
                    elif 'volume' in feature_name:
                        features_array.append(1.0)
                    elif 'timeframe' in feature_name:
                        features_array.append(5.0)
                    else:
                        features_array.append(0.0)
            
            features_array = np.array(features_array).reshape(1, -1)
            
            # Скейлинг
            if self.scaler is not None:
                features_array = self.scaler.transform(features_array)
            
            return features_array
            
        except Exception as e:
            logger.warning(f"Ошибка подготовки признаков: {e}")
            return None
    
    def predict_signal(self, features_array):
        """Предсказание сигнала через ансамбль."""
        if features_array is None:
            return None
        
        try:
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
                    base_predictions.append(float(probability))
                else:
                    base_predictions.append(0.5)
            
            # Метамодель
            if self.meta_model is not None and len(base_predictions) == 4:
                meta_features_array = np.array([base_predictions])
                final_proba = self.meta_model.predict_proba(meta_features_array)
                return {
                    'probability': float(final_proba[0, 1]),
                    'signal': 1 if final_proba[0, 1] > 0.55 else 0,
                    'confidence': abs(final_proba[0, 1] - 0.5) * 2,
                    'base_predictions': base_predictions
                }
            else:
                # Простое усреднение
                avg_prob = np.mean(base_predictions)
                return {
                    'probability': float(avg_prob),
                    'signal': 1 if avg_prob > 0.55 else 0,
                    'confidence': abs(avg_prob - 0.5) * 2,
                    'base_predictions': base_predictions
                }
                
        except Exception as e:
            logger.warning(f"Ошибка предсказания: {e}")
            return None
    
    def process_signal(self, timestamp, price, signal_data):
        """Обработка торгового сигнала."""
        if signal_data is None:
            return
        
        signal = signal_data['signal']
        probability = signal_data['probability']
        
        # Логика входа в позицию
        if self.position is None and signal == 1 and probability > 0.6:
            # Покупаем
            entry_price = price * (1 + self.slippage)  # Слиппаж
            position_size = (self.balance * 0.02) / entry_price  # 2% от баланса
            commission = position_size * entry_price * self.commission_rate
            
            self.position = {
                'type': 'BUY',
                'entry_time': timestamp,
                'entry_price': entry_price,
                'size': position_size,
                'commission_paid': commission,
                'signal_data': signal_data
            }
            
            self.balance -= commission
            self.signals_generated += 1
            
        # Логика выхода из позиции
        elif self.position is not None:
            should_exit = False
            exit_reason = ""
            
            # Выход по сигналу (вероятность < 40%)
            if signal == 0 or probability < 0.4:
                should_exit = True
                exit_reason = "signal"
            
            # Выход по времени (более 4 часов)
            elif (timestamp - self.position['entry_time']).total_seconds() > 4 * 3600:
                should_exit = True
                exit_reason = "time"
            
            if should_exit:
                exit_price = price * (1 - self.slippage)  # Слиппаж
                pnl_gross = (exit_price - self.position['entry_price']) * self.position['size']
                exit_commission = self.position['size'] * exit_price * self.commission_rate
                pnl_net = pnl_gross - self.position['commission_paid'] - exit_commission
                
                trade = {
                    'entry_time': self.position['entry_time'],
                    'exit_time': timestamp,
                    'entry_price': self.position['entry_price'],
                    'exit_price': exit_price,
                    'size': self.position['size'],
                    'pnl_gross': pnl_gross,
                    'pnl_net': pnl_net,
                    'commission_total': self.position['commission_paid'] + exit_commission,
                    'exit_reason': exit_reason,
                    'duration_hours': (timestamp - self.position['entry_time']).total_seconds() / 3600,
                    'entry_probability': self.position['signal_data']['probability']
                }
                
                self.trades.append(trade)
                self.balance += pnl_net
                self.position = None
    
    def record_equity(self, timestamp):
        """Записать состояние счета."""
        position_value = 0.0
        if self.position:
            current_price = self.current_price  # Устанавливается в основном цикле
            position_value = self.position['size'] * current_price
        
        total_equity = self.balance + position_value
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': self.balance,
            'position_value': position_value,
            'total_equity': total_equity
        })
    
    def load_data_for_period(self, start_date: str, end_date: str):
        """Загрузка данных за период."""
        logger.info(f"📊 Поиск данных за период {start_date} - {end_date}")
        
        # Пробуем найти подходящий блок данных
        try:
            from src.utils.data_blocks import DataBlockManager
            
            manager = DataBlockManager()
            blocks = manager.list_blocks()
            
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            suitable_blocks = []
            for block_id, block_info in blocks.items():
                if (block_info.start_time <= start_dt and 
                    block_info.end_time >= end_dt):
                    suitable_blocks.append((block_id, block_info))
            
            if suitable_blocks:
                # Берем первый подходящий блок
                block_id, block_info = suitable_blocks[0]
                logger.info(f"✅ Найден блок: {block_id}")
                logger.info(f"   Период: {block_info.start_time} - {block_info.end_time}")
                
                block_data = manager.load_block(block_id)
                if "5m" in block_data:
                    data = block_data["5m"].copy()
                    data.set_index('timestamp', inplace=True)
                    
                    # Фильтруем по нужному периоду
                    mask = (data.index >= start_dt) & (data.index <= end_dt)
                    data = data[mask]
                    
                    logger.info(f"✅ Загружено {len(data)} свечей 5m из блока")
                    return data
            
        except Exception as e:
            logger.warning(f"Не удалось загрузить из блоков: {e}")
        
        # Пробуем загрузить из файлов
        logger.info("🔄 Поиск в файлах данных...")
        data_dir = Path("data")
        
        # Ищем файлы с реальными данными за нужный период
        pattern = "SOLUSDT_5m_real_*.csv"
        files = list(data_dir.glob(pattern))
        
        if not files:
            logger.error("❌ Не найдены файлы с реальными данными")
            return None
        
        # Берем самый свежий файл
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        logger.info(f"📁 Загрузка из файла: {latest_file.name}")
        
        try:
            data = pd.read_csv(latest_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
            
            # Фильтруем по периоду
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            mask = (data.index >= start_dt) & (data.index <= end_dt)
            data = data[mask]
            
            if len(data) == 0:
                logger.error(f"❌ Нет данных за период {start_date} - {end_date}")
                return None
            
            logger.info(f"✅ Загружено {len(data)} свечей за период")
            return data
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки файла: {e}")
            return None
    
    def add_technical_indicators(self, data):
        """Добавление технических индикаторов в правильном порядке."""
        logger.info("📈 Генерация технических индикаторов...")
        
        try:
            # Ожидаемые признаки модели
            expected_features = [
                'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 
                'BB_hband', 'BB_lband', 'BB_width', 'BB_position', 'vol_change', 
                'volume_sma', 'volume_ratio', 'price_change', 'price_change_5', 
                'volatility', 'EMA12', 'EMA26', 'high_low_pct', 'close_to_high', 
                'close_to_low', 'timeframe_minutes'
            ]
            
            with tqdm(total=len(expected_features), desc="Генерация признаков", unit="индикатор") as pbar:
                
                # Moving Averages
                data['MA5'] = data['close'].rolling(5, min_periods=1).mean()
                pbar.update(1)
                data['MA10'] = data['close'].rolling(10, min_periods=1).mean()
                pbar.update(1)
                data['MA20'] = data['close'].rolling(20, min_periods=1).mean()
                pbar.update(1)
                
                # EMA
                data['EMA12'] = data['close'].ewm(span=12, min_periods=1).mean()
                pbar.update(1)
                data['EMA26'] = data['close'].ewm(span=26, min_periods=1).mean()
                pbar.update(1)
                
                # RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                rs = gain / (loss + 1e-8)
                data['RSI'] = 100 - (100 / (1 + rs))
                pbar.update(1)
                
                # MACD
                data['MACD'] = data['EMA12'] - data['EMA26']
                pbar.update(1)
                data['MACD_signal'] = data['MACD'].ewm(span=9, min_periods=1).mean()
                pbar.update(1)
                data['MACD_diff'] = data['MACD'] - data['MACD_signal']
                pbar.update(1)
                
                # Bollinger Bands
                bb_std = data['close'].rolling(20, min_periods=1).std()
                data['BB_hband'] = data['MA20'] + (bb_std * 2)
                pbar.update(1)
                data['BB_lband'] = data['MA20'] - (bb_std * 2)
                pbar.update(1)
                data['BB_width'] = data['BB_hband'] - data['BB_lband']
                pbar.update(1)
                data['BB_position'] = (data['close'] - data['BB_lband']) / (data['BB_width'] + 1e-8)
                pbar.update(1)
                
                # Volume indicators
                data['volume_sma'] = data['volume'].rolling(20, min_periods=1).mean()
                pbar.update(1)
                data['volume_ratio'] = data['volume'] / (data['volume_sma'] + 1e-8)
                pbar.update(1)
                data['vol_change'] = data['volume'].pct_change().fillna(0)
                pbar.update(1)
                
                # Price changes
                data['price_change'] = data['close'].pct_change().fillna(0)
                pbar.update(1)
                data['price_change_5'] = data['close'].pct_change(5).fillna(0)
                pbar.update(1)
                
                # Volatility
                data['volatility'] = data['close'].rolling(14, min_periods=1).std() / (data['close'].rolling(14, min_periods=1).mean() + 1e-8)
                pbar.update(1)
                
                # Price position indicators
                data['high_low_pct'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
                pbar.update(1)
                data['close_to_high'] = (data['close'] - data['low']) / ((data['high'] - data['low']) + 1e-8)
                pbar.update(1)
                data['close_to_low'] = (data['high'] - data['close']) / ((data['high'] - data['low']) + 1e-8)
                pbar.update(1)
                
                # Timeframe
                data['timeframe_minutes'] = 5.0
                pbar.update(1)
            
            # Заполняем NaN
            data = data.fillna(method='ffill').fillna(0)
            
            logger.info(f"✅ Добавлены все {len(expected_features)} признаков")
            return data
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации индикаторов: {e}")
            return None
    
    def run_backtest(self, data):
        """Запуск бэктестирования."""
        logger.info("🚀 Запуск бэктестирования...")
        
        # Загружаем модели при первом использовании
        if not self.load_models():
            logger.error("❌ Не удалось загрузить модели")
            return None
        
        start_time = time.time()
        
        # Пропускаем первые 50 свечей для стабилизации индикаторов
        start_idx = 50
        total_candles = len(data) - start_idx
        
        logger.info(f"📊 Обработка {total_candles} свечей (пропускаем первые {start_idx})")
        
        # Основной цикл с прогресс-баром
        with tqdm(total=total_candles, desc="Бэктестирование", unit="свечи") as pbar:
            
            for i in range(start_idx, len(data)):
                row = data.iloc[i]
                timestamp = data.index[i]
                self.current_price = row['close']
                
                # Подготавливаем признаки
                features_array = self.prepare_features(row)
                
                # Получаем сигнал
                signal_data = self.predict_signal(features_array)
                
                # Обрабатываем сигнал
                self.process_signal(timestamp, self.current_price, signal_data)
                
                # Записываем эквити
                self.record_equity(timestamp)
                
                self.processed_candles += 1
                
                # Обновляем прогресс-бар
                if self.processed_candles % 100 == 0:
                    pbar.set_postfix({
                        "Баланс": f"${self.balance:.0f}",
                        "Сделок": len(self.trades),
                        "Сигналов": self.signals_generated,
                        "Позиция": "Есть" if self.position else "Нет"
                    })
                
                pbar.update(1)
        
        # Закрываем оставшуюся позицию
        if self.position:
            final_price = data.iloc[-1]['close']
            final_timestamp = data.index[-1]
            
            exit_price = final_price * (1 - self.slippage)
            pnl_gross = (exit_price - self.position['entry_price']) * self.position['size']
            exit_commission = self.position['size'] * exit_price * self.commission_rate
            pnl_net = pnl_gross - self.position['commission_paid'] - exit_commission
            
            self.trades.append({
                'entry_time': self.position['entry_time'],
                'exit_time': final_timestamp,
                'entry_price': self.position['entry_price'],
                'exit_price': exit_price,
                'size': self.position['size'],
                'pnl_gross': pnl_gross,
                'pnl_net': pnl_net,
                'commission_total': self.position['commission_paid'] + exit_commission,
                'exit_reason': 'backtest_end',
                'duration_hours': (final_timestamp - self.position['entry_time']).total_seconds() / 3600,
                'entry_probability': self.position['signal_data']['probability']
            })
            
            self.balance += pnl_net
            self.position = None
        
        backtest_time = time.time() - start_time
        logger.info(f"⏱️ Бэктестирование завершено за {backtest_time:.2f}s")
        
        return self.calculate_results()
    
    def calculate_results(self):
        """Расчет результатов бэктестирования."""
        if not self.trades:
            return {"error": "Нет сделок"}
        
        # Базовые метрики
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl_net'] > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades
        total_pnl = sum(t['pnl_net'] for t in self.trades)
        total_commission = sum(t['commission_total'] for t in self.trades)
        
        final_balance = self.balance
        total_return_pct = (final_balance / self.initial_balance - 1) * 100
        
        # Дополнительные метрики
        if winning_trades > 0:
            avg_win = np.mean([t['pnl_net'] for t in self.trades if t['pnl_net'] > 0])
        else:
            avg_win = 0
        
        if losing_trades > 0:
            avg_loss = abs(np.mean([t['pnl_net'] for t in self.trades if t['pnl_net'] <= 0]))
        else:
            avg_loss = 0
        
        profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if avg_loss > 0 else 0
        
        # Дюрация
        avg_duration = np.mean([t['duration_hours'] for t in self.trades])
        
        # Эквити кривая для максимальной просадки
        equities = [eq['total_equity'] for eq in self.equity_curve]
        if equities:
            peak = self.initial_balance
            max_drawdown = 0
            for equity in equities:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            max_drawdown = 0
        
        return {
            'period': f"{self.equity_curve[0]['timestamp'].date()} - {self.equity_curve[-1]['timestamp'].date()}",
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'total_return_pct': total_return_pct,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_duration_hours': avg_duration,
            'max_drawdown': max_drawdown,
            'total_commission': total_commission,
            'signals_generated': self.signals_generated,
            'processed_candles': self.processed_candles,
            'trades': self.trades  # Для детального анализа
        }
    
    def print_results(self, results):
        """Вывод результатов бэктестирования."""
        if "error" in results:
            print(f"❌ {results['error']}")
            return
        
        print("\n" + "="*80)
        print("           РЕЗУЛЬТАТЫ БЭКТЕСТИРОВАНИЯ (10-17 АВГУСТА)")
        print("="*80)
        
        print(f"📅 Период:              {results['period']}")
        print(f"📊 Обработано свечей:   {results['processed_candles']:,}")
        print(f"🎯 Генерировано сигналов: {results['signals_generated']}")
        
        print("\n💰 ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   Начальный баланс:    ${results['initial_balance']:,.2f}")
        print(f"   Финальный баланс:    ${results['final_balance']:,.2f}")
        print(f"   Общий доход:         {results['total_return_pct']:+.2f}%")
        print(f"   P&L:                 ${results['total_pnl']:+,.2f}")
        print(f"   Комиссии:            ${results['total_commission']:,.2f}")
        
        print("\n📈 ТОРГОВАЯ СТАТИСТИКА:")
        print(f"   Всего сделок:        {results['total_trades']}")
        print(f"   Прибыльных:          {results['winning_trades']}")
        print(f"   Убыточных:           {results['losing_trades']}")
        print(f"   Винрейт:             {results['win_rate']:.1%}")
        print(f"   Средняя прибыль:     ${results['avg_win']:,.2f}")
        print(f"   Средний убыток:      ${results['avg_loss']:,.2f}")
        print(f"   Профит-фактор:       {results['profit_factor']:.2f}")
        
        print("\n⏱️ ВРЕМЕННАЯ СТАТИСТИКА:")
        print(f"   Средняя длительность: {results['avg_duration_hours']:.1f} часов")
        print(f"   Максимальная просадка: {results['max_drawdown']:.1%}")
        
        # Показываем последние сделки
        if results['trades']:
            print(f"\n📋 ПОСЛЕДНИЕ 5 СДЕЛОК:")
            print("-" * 80)
            recent_trades = results['trades'][-5:]
            for i, trade in enumerate(recent_trades, 1):
                entry_time = trade['entry_time'].strftime('%m-%d %H:%M')
                exit_time = trade['exit_time'].strftime('%m-%d %H:%M')
                pnl_str = f"${trade['pnl_net']:+.2f}"
                duration_str = f"{trade['duration_hours']:.1f}h"
                prob_str = f"{trade['entry_probability']:.3f}"
                
                print(f"   {i}. {entry_time}→{exit_time} | "
                      f"{pnl_str:>8} | {duration_str:>6} | "
                      f"Prob:{prob_str} | {trade['exit_reason']}")
        
        print("\n" + "="*80)
        
        # Оценка результатов
        if results['total_return_pct'] > 10 and results['win_rate'] > 0.5:
            print("🎉 ОТЛИЧНЫЕ РЕЗУЛЬТАТЫ!")
        elif results['total_return_pct'] > 0 and results['win_rate'] > 0.4:
            print("✅ ПОЛОЖИТЕЛЬНЫЕ РЕЗУЛЬТАТЫ")
        elif results['total_return_pct'] > -5:
            print("⚠️ НЕЙТРАЛЬНЫЕ РЕЗУЛЬТАТЫ")
        else:
            print("❌ ТРЕБУЕТСЯ ОПТИМИЗАЦИЯ")
        
        print("="*80)


def main():
    """Главная функция."""
    print("🎯 БЭКТЕСТИРОВАНИЕ АНСАМБЛЯ НА ДАННЫХ 10-17 АВГУСТА")
    print("="*80)
    
    try:
        # Параметры периода (используем 2025 год, так как данные за этот год)
        start_date = "2025-08-10T00:00:00"
        end_date = "2025-08-17T23:59:59"
        
        # Создаем бэктестер
        backtester = OptimizedEnsembleBacktester()
        
        # Загружаем данные
        data = backtester.load_data_for_period(start_date, end_date)
        if data is None:
            logger.error("❌ Не удалось загрузить данные")
            return
        
        # Добавляем индикаторы
        data = backtester.add_technical_indicators(data)
        if data is None:
            logger.error("❌ Ошибка генерации индикаторов")
            return
        
        # Запускаем бэктест
        results = backtester.run_backtest(data)
        if results is None:
            logger.error("❌ Ошибка бэктестирования")
            return
        
        # Выводим результаты
        backtester.print_results(results)
        
        # Сохраняем результаты
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"backtest_results_aug10_17_{timestamp}.json"
        
        try:
            import json
            with open(results_file, 'w') as f:
                # Убираем trades из сохранения (слишком много данных)
                save_results = results.copy()
                if 'trades' in save_results:
                    save_results['trades_count'] = len(save_results.pop('trades'))
                json.dump(save_results, f, indent=2, default=str)
            print(f"\n💾 Результаты сохранены: {results_file}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить результаты: {e}")
        
        print("\n✅ Бэктестирование завершено успешно!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Бэктестирование прервано пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()