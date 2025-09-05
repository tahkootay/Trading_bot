#!/usr/bin/env python3
"""
Оптимизированный бэктест с ML ансамблем, прогресс-барами и быстрой инициализацией
"""

import sys
import os
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импорты проекта
from src.models.optimized_ensemble_predictor import OptimizedEnsemblePredictor, ProgressBar

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedBacktester:
    """
    Оптимизированный бэктестер с ML ансамблем, прогресс-барами и быстрой инициализацией.
    """
    
    def __init__(self, 
                 symbol: str = "SOLUSDT",
                 initial_capital: float = 10000.0,
                 position_size: float = 0.1,
                 show_progress: bool = True,
                 commission_rate: float = 0.001,
                 slippage: float = 0.0005):
        """
        Инициализация оптимизированного бэктестера.
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.show_progress = show_progress
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # Компоненты
        self.predictor = None
        
        # Результаты
        self.results = {}
        self.trades = []
        self.equity_curve = []
        
        # Статистика производительности
        self.performance_stats = {
            'model_load_time': 0,
            'data_prep_time': 0,
            'backtest_time': 0,
            'predictions_made': 0,
            'avg_prediction_time': 0
        }
        
    def prepare_features_fast(self, data: pd.DataFrame) -> pd.DataFrame:
        """Быстрая подготовка технических индикаторов."""
        logger.info("📊 Быстрая подготовка технических индикаторов...")
        
        prep_start = time.time()
        
        # Копируем данные
        df = data.copy()
        
        if self.show_progress:
            progress = ProgressBar(6, "📈 Расчет индикаторов")
        
        # 1. Moving Averages (векторизованные операции)
        df['MA5'] = df['close'].rolling(5, min_periods=1).mean()
        df['MA10'] = df['close'].rolling(10, min_periods=1).mean()  
        df['MA20'] = df['close'].rolling(20, min_periods=1).mean()
        df['EMA12'] = df['close'].ewm(span=12, min_periods=1).mean()
        df['EMA26'] = df['close'].ewm(span=26, min_periods=1).mean()
        
        if self.show_progress:
            progress.update(1)
        
        # 2. RSI (оптимизированный)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        if self.show_progress:
            progress.update(1)
        
        # 3. MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        if self.show_progress:
            progress.update(1)
        
        # 4. Bollinger Bands
        bb_std = df['close'].rolling(20, min_periods=1).std()
        df['BB_hband'] = df['MA20'] + (bb_std * 2)
        df['BB_lband'] = df['MA20'] - (bb_std * 2)
        df['BB_width'] = df['BB_hband'] - df['BB_lband']
        df['BB_position'] = (df['close'] - df['BB_lband']) / (df['BB_width'] + 1e-8)
        
        if self.show_progress:
            progress.update(1)
        
        # 5. Volume indicators
        df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
        df['vol_change'] = df['volume'].pct_change().fillna(0)
        
        if self.show_progress:
            progress.update(1)
        
        # 6. Price indicators
        df['price_change'] = df['close'].pct_change().fillna(0)
        df['price_change_5'] = df['close'].pct_change(5).fillna(0)
        
        # Volatility
        price_std = df['close'].rolling(14, min_periods=1).std()
        price_mean = df['close'].rolling(14, min_periods=1).mean()
        df['volatility'] = price_std / (price_mean + 1e-8)
        
        # Price position indicators
        df['high_low_pct'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
        high_low_range = df['high'] - df['low'] + 1e-8
        df['close_to_high'] = (df['close'] - df['low']) / high_low_range
        df['close_to_low'] = (df['high'] - df['close']) / high_low_range
        
        # Timeframe
        df['timeframe_minutes'] = 5.0
        
        if self.show_progress:
            progress.update(1)
        
        # Заполняем NaN более эффективно
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        
        prep_time = time.time() - prep_start
        self.performance_stats['data_prep_time'] = prep_time
        
        logger.info(f"✅ Индикаторы рассчитаны за {prep_time:.2f}s: {len(df)} строк")
        return df
    
    def run_optimized_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Запуск оптимизированного бэктеста."""
        logger.info("🚀 Запуск оптимизированного ML бэктеста...")
        backtest_start = time.time()
        
        # 1. Подготовка данных с таймингом
        prepared_data = self.prepare_features_fast(data)
        
        # 2. Инициализация ML предсказателя с таймингом
        logger.info("🤖 Создание оптимизированного ML предсказателя...")
        model_start = time.time()
        
        self.predictor = OptimizedEnsemblePredictor(
            lazy_loading=True,  # Отложенная загрузка
            show_progress=self.show_progress
        )
        
        self.performance_stats['model_load_time'] = time.time() - model_start
        
        # 3. Инициализация торговых переменных
        capital = self.initial_capital
        position = 0  # Размер текущей позиции
        position_price = 0  # Цена входа
        trades = []
        equity_curve = []
        
        # Статистика предсказаний
        prediction_times = []
        predictions_made = 0
        
        # 4. Основной цикл бэктеста с прогресс-баром
        total_rows = len(prepared_data)
        start_idx = 50  # Пропускаем первые строки для стабилизации индикаторов
        
        if self.show_progress:
            backtest_progress = ProgressBar(total_rows - start_idx, "💹 ML Backtest")
        
        logger.info(f"💹 Обработка {total_rows - start_idx} свечей с ML предсказаниями...")
        
        for i in range(start_idx, total_rows):
            current_row = prepared_data.iloc[i]
            current_price = current_row['close']
            current_time = current_row.get('timestamp', i)
            
            # Получаем ML предсказание с таймингом
            pred_start = time.time()
            
            # Создаем DataFrame для предсказания
            prediction_data = pd.DataFrame([current_row])
            
            try:
                prediction = self.predictor.predict_ensemble(prediction_data)
                pred_time = time.time() - pred_start
                prediction_times.append(pred_time)
                predictions_made += 1
                
                if prediction is None:
                    if self.show_progress:
                        backtest_progress.update(1)
                    continue
                
                signal = prediction['final_signal']
                probability = prediction['final_probability']
                strength = prediction['signal_strength']
                
            except Exception as e:
                logger.warning(f"Ошибка предсказания на индексе {i}: {e}")
                if self.show_progress:
                    backtest_progress.update(1)
                continue
            
            # Торговая логика
            if position == 0:  # Нет позиции
                # Условия открытия позиции
                if signal == 1 and probability > 0.65 and strength == 'STRONG':
                    # Открываем лонг позицию
                    position_value = capital * self.position_size
                    commission = position_value * self.commission_rate
                    entry_price = current_price * (1 + self.slippage)
                    
                    position = position_value / entry_price
                    position_price = entry_price
                    capital -= commission
                    
                    trades.append({
                        'type': 'BUY',
                        'time': current_time,
                        'price': entry_price,
                        'size': position,
                        'probability': probability,
                        'strength': strength,
                        'capital_before': capital + commission,
                        'commission': commission
                    })
                    
            else:  # Есть позиция
                # Условия закрытия позиции
                should_close = False
                close_reason = ''
                
                # Стоп-лосс (5%)
                if current_price <= position_price * 0.95:
                    should_close = True
                    close_reason = 'stop_loss'
                # Тейк-профит (10%)
                elif current_price >= position_price * 1.1:
                    should_close = True
                    close_reason = 'take_profit'
                # Сигнал на выход или слабая уверенность
                elif signal == 0 or probability < 0.4:
                    should_close = True
                    close_reason = 'signal_exit'
                
                if should_close:
                    exit_price = current_price * (1 - self.slippage)
                    position_value = position * exit_price
                    commission = position_value * self.commission_rate
                    
                    gross_pnl = (exit_price - position_price) * position
                    net_pnl = gross_pnl - commission
                    
                    capital += position_value - commission
                    
                    trades.append({
                        'type': 'SELL',
                        'time': current_time,
                        'price': exit_price,
                        'size': position,
                        'pnl_gross': gross_pnl,
                        'pnl_net': net_pnl,
                        'capital_after': capital,
                        'commission': commission,
                        'close_reason': close_reason
                    })
                    
                    position = 0
                    position_price = 0
            
            # Обновляем equity curve
            if position > 0:
                unrealized_pnl = (current_price - position_price) * position
                total_equity = capital + (position * current_price)
            else:
                total_equity = capital
                unrealized_pnl = 0
                
            equity_curve.append({
                'time': current_time,
                'equity': total_equity,
                'price': current_price,
                'position': position,
                'unrealized_pnl': unrealized_pnl
            })
            
            if self.show_progress:
                backtest_progress.update(1)
        
        # 5. Закрываем оставшуюся позицию
        if position > 0:
            final_price = prepared_data.iloc[-1]['close']
            final_time = prepared_data.iloc[-1].get('timestamp', len(prepared_data))
            
            exit_price = final_price * (1 - self.slippage)
            position_value = position * exit_price
            commission = position_value * self.commission_rate
            
            gross_pnl = (exit_price - position_price) * position
            net_pnl = gross_pnl - commission
            
            capital += position_value - commission
            
            trades.append({
                'type': 'SELL',
                'time': final_time,
                'price': exit_price,
                'size': position,
                'pnl_gross': gross_pnl,
                'pnl_net': net_pnl,
                'capital_after': capital,
                'commission': commission,
                'close_reason': 'backtest_end'
            })
        
        # 6. Расчет финальной статистики производительности
        total_backtest_time = time.time() - backtest_start
        self.performance_stats['backtest_time'] = total_backtest_time
        self.performance_stats['predictions_made'] = predictions_made
        self.performance_stats['avg_prediction_time'] = np.mean(prediction_times) if prediction_times else 0
        
        # 7. Расчет результатов
        self.trades = trades
        self.equity_curve = equity_curve
        results = self.calculate_comprehensive_metrics(equity_curve, trades, capital)
        results['performance_stats'] = self.performance_stats
        
        self.results = results
        
        logger.info(f"✅ Бэктест завершен за {total_backtest_time:.2f}s!")
        return results
    
    def calculate_comprehensive_metrics(self, equity_curve: List[Dict], trades: List[Dict], final_capital: float) -> Dict[str, Any]:
        """Расчет комплексных метрик производительности."""
        logger.info("📊 Расчет метрик производительности...")
        
        if not trades:
            return {'error': 'No trades executed', 'final_capital': final_capital}
        
        # Фильтруем только сделки покупки/продажи с P&L
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL' and 'pnl_net' in t]
        
        if not sell_trades:
            return {'error': 'No completed trades', 'final_capital': final_capital}
        
        # Базовые метрики
        total_trades = len(buy_trades)
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
        # P&L анализ
        pnls = [t['pnl_net'] for t in sell_trades]
        total_pnl = sum(pnls)
        
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / len(sell_trades) * 100
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf')
        
        # Equity curve анализ
        if len(equity_curve) > 1:
            equity_values = [point['equity'] for point in equity_curve]
            equity_series = pd.Series(equity_values)
            
            # Максимальная просадка
            rolling_max = equity_series.cummax()
            drawdown = (rolling_max - equity_series) / rolling_max
            max_drawdown = drawdown.max() * 100
            
            # Sharpe ratio (на основе дневной доходности)
            returns = equity_series.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(288)  # 5-min * 288 = 1 день
            else:
                sharpe_ratio = 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0
        
        # Временной анализ (для торгов с timestamp)
        durations = []
        for buy_trade, sell_trade in zip(buy_trades, sell_trades):
            if 'time' in buy_trade and 'time' in sell_trade:
                try:
                    duration = sell_trade['time'] - buy_trade['time']
                    if hasattr(duration, 'total_seconds'):
                        durations.append(duration.total_seconds() / 3600)  # часы
                    else:
                        durations.append(float(duration))  # предполагаем часы
                except:
                    pass
        
        avg_trade_duration = np.mean(durations) if durations else 0
        
        # Комиссии
        total_commission = sum(t.get('commission', 0) for t in trades)
        
        # ML специфические метрики
        high_confidence_trades = len([t for t in buy_trades if t.get('probability', 0) > 0.8])
        strong_signal_trades = len([t for t in buy_trades if t.get('strength') == 'STRONG'])
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'completed_trades': len(sell_trades),
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'avg_trade_duration_hours': avg_trade_duration,
            'total_commission': total_commission,
            'commission_pct_of_pnl': abs(total_commission / total_pnl * 100) if total_pnl != 0 else 0,
            'high_confidence_trades': high_confidence_trades,
            'strong_signal_trades': strong_signal_trades,
            'best_trade': max(pnls),
            'worst_trade': min(pnls),
            'equity_curve': equity_curve,
            'trades_detail': trades
        }
    
    def print_comprehensive_results(self):
        """Печать комплексных результатов с ML и производительными метриками."""
        if not self.results:
            logger.error("❌ Нет результатов для отображения")
            return
        
        print("\n" + "="*80)
        print("🤖 РЕЗУЛЬТАТЫ ОПТИМИЗИРОВАННОГО ML БЭКТЕСТА")
        print("="*80)
        
        if "error" in self.results:
            print(f"❌ {self.results['error']}")
            return
        
        # Производительность системы
        perf = self.results.get('performance_stats', {})
        print("⚡ ПРОИЗВОДИТЕЛЬНОСТЬ СИСТЕМЫ:")
        print(f"   Загрузка ML моделей: {perf.get('model_load_time', 0):.2f}s")
        print(f"   Подготовка данных: {perf.get('data_prep_time', 0):.2f}s")
        print(f"   Время бэктеста: {perf.get('backtest_time', 0):.2f}s")
        print(f"   Предсказаний сделано: {perf.get('predictions_made', 0)}")
        print(f"   Среднее время предсказания: {perf.get('avg_prediction_time', 0)*1000:.1f}ms")
        
        # Финансовые результаты
        print(f"\n💰 ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   Начальный капитал: ${self.results['initial_capital']:,.2f}")
        print(f"   Финальный капитал: ${self.results['final_capital']:,.2f}")
        print(f"   Общая доходность: {self.results['total_return_pct']:+.2f}%")
        print(f"   Общий P&L: ${self.results['total_pnl']:+,.2f}")
        print(f"   Лучшая сделка: ${self.results['best_trade']:+.2f}")
        print(f"   Худшая сделка: ${self.results['worst_trade']:+.2f}")
        
        # Торговая статистика
        print(f"\n📊 ТОРГОВАЯ СТАТИСТИКА:")
        print(f"   Всего позиций: {self.results['total_trades']}")
        print(f"   Завершенных сделок: {self.results['completed_trades']}")
        print(f"   Выигрышных: {self.results['winning_trades']}")
        print(f"   Проигрышных: {self.results['losing_trades']}")
        print(f"   Win Rate: {self.results['win_rate_pct']:.1f}%")
        print(f"   Profit Factor: {self.results['profit_factor']:.2f}")
        
        # ML специфические метрики
        print(f"\n🧠 ML МЕТРИКИ:")
        print(f"   Сделки с высокой уверенностью (>80%): {self.results['high_confidence_trades']}")
        print(f"   Сильные сигналы: {self.results['strong_signal_trades']}")
        print(f"   Среднее время удержания: {self.results['avg_trade_duration_hours']:.1f} часов")
        
        # Риск-метрики
        print(f"\n⚠️ РИСК-МЕТРИКИ:")
        print(f"   Максимальная просадка: {self.results['max_drawdown_pct']:.2f}%")
        print(f"   Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"   Средний выигрыш: ${self.results['avg_win']:.2f}")
        print(f"   Средний проигрыш: ${self.results['avg_loss']:.2f}")
        
        # Затраты
        print(f"\n💸 ЗАТРАТЫ:")
        print(f"   Общие комиссии: ${self.results['total_commission']:.2f}")
        print(f"   Комиссии от P&L: {self.results['commission_pct_of_pnl']:.1f}%")
        
        # Последние сделки
        if self.trades:
            print(f"\n📝 ПОСЛЕДНИЕ 5 СДЕЛОК:")
            trades_to_show = []
            for i in range(0, len(self.trades), 2):  # Берем пары BUY-SELL
                if i + 1 < len(self.trades):
                    buy_trade = self.trades[i]
                    sell_trade = self.trades[i + 1]
                    if buy_trade['type'] == 'BUY' and sell_trade['type'] == 'SELL':
                        trades_to_show.append((buy_trade, sell_trade))
            
            for buy_trade, sell_trade in trades_to_show[-5:]:
                prob = buy_trade.get('probability', 0)
                pnl = sell_trade.get('pnl_net', 0)
                reason = sell_trade.get('close_reason', 'unknown')
                pnl_color = "🟢" if pnl > 0 else "🔴"
                
                print(f"   BUY ${buy_trade['price']:.4f} (prob: {prob:.3f}) → "
                      f"SELL ${sell_trade['price']:.4f} {pnl_color} ${pnl:+.2f} [{reason}]")
        
        print("="*80)
        
        # Оценка качества стратегии
        if (self.results['win_rate_pct'] > 50 and 
            self.results['profit_factor'] > 1.2 and 
            self.results['max_drawdown_pct'] < 15):
            print("🎉 ОТЛИЧНАЯ СТРАТЕГИЯ - достигает ключевые цели!")
        elif (self.results['win_rate_pct'] > 40 and 
              self.results['profit_factor'] > 1.0):
            print("✅ ХОРОШАЯ СТРАТЕГИЯ - показывает потенциал")
        else:
            print("⚠️ ТРЕБУЕТ ОПТИМИЗАЦИИ - необходимы улучшения")


def load_data_with_fallback(symbol: str, days: int = 7) -> Optional[pd.DataFrame]:
    """Загрузка данных с несколькими попытками."""
    logger.info(f"📥 Загрузка данных {symbol} за последние {days} дней...")
    
    # Возможные пути к файлам данных
    possible_files = [
        f"data/bybit_futures_{symbol.lower()}_5m.csv",
        f"data/{symbol}_5m_*.csv",
        f"data/{symbol.upper()}_5m.csv",
        f"data/SOLUSDT_5m.csv"  # fallback
    ]
    
    data_file = None
    for file_pattern in possible_files:
        if '*' in file_pattern:
            # Паттерн с wildcard
            from glob import glob
            files = glob(file_pattern)
            if files:
                # Берем самый новый файл
                data_file = max(files, key=lambda x: Path(x).stat().st_mtime)
                break
        else:
            # Конкретный файл
            if Path(file_pattern).exists():
                data_file = file_pattern
                break
    
    if not data_file:
        logger.error("❌ Не найдено файлов с данными!")
        logger.info("💡 Попробуйте запустить: python scripts/collect_data.py --symbol SOLUSDT --days 30")
        return None
    
    logger.info(f"📂 Загрузка из: {data_file}")
    
    try:
        data = pd.read_csv(data_file)
        
        # Обработка колонки времени
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        else:
            # Если нет timestamp, создаем индекс времени
            data['timestamp'] = pd.date_range(start='2024-08-01', periods=len(data), freq='5min')
        
        # Берем последние N дней
        if days and len(data) > days * 288:  # 288 = 5-min свечей в день
            data = data.tail(days * 288).copy()
        
        # Сортируем по времени
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✅ Загружено {len(data)} свечей")
        logger.info(f"   Период: {data['timestamp'].min()} - {data['timestamp'].max()}")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки данных: {e}")
        return None


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description='Optimized ML Ensemble Backtest')
    parser.add_argument('--symbol', default='SOLUSDT', help='Trading symbol')
    parser.add_argument('--days', type=int, default=7, help='Number of days to test')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--position-size', type=float, default=0.1, help='Position size (fraction of capital)')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage rate')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bars')
    
    args = parser.parse_args()
    
    print("🤖 Оптимизированный ML Ансамбль Бэктест")
    print("="*50)
    print(f"Символ: {args.symbol}")
    print(f"Дни: {args.days}")
    print(f"Капитал: ${args.capital:,.2f}")
    print(f"Размер позиции: {args.position_size:.1%}")
    print("="*50)
    
    # Загрузка данных
    data = load_data_with_fallback(args.symbol, args.days)
    if data is None:
        return
    
    # Создание бэктестера
    backtester = OptimizedBacktester(
        symbol=args.symbol,
        initial_capital=args.capital,
        position_size=args.position_size,
        show_progress=not args.no_progress,
        commission_rate=args.commission,
        slippage=args.slippage
    )
    
    # Запуск бэктеста
    start_time = time.time()
    try:
        results = backtester.run_optimized_backtest(data)
        total_time = time.time() - start_time
        
        # Показ результатов
        backtester.print_comprehensive_results()
        
        print(f"\n⏱️ Общее время выполнения: {total_time:.2f}s")
        
        # Дополнительная информация о производительности
        perf = results.get('performance_stats', {})
        if perf:
            overhead = (perf.get('model_load_time', 0) + perf.get('data_prep_time', 0)) / total_time * 100
            print(f"   Накладные расходы (загрузка+подготовка): {overhead:.1f}%")
            
            predictions_per_sec = perf.get('predictions_made', 0) / perf.get('backtest_time', 1)
            print(f"   Скорость ML предсказаний: {predictions_per_sec:.0f} предсказаний/сек")
        
    except Exception as e:
        logger.error(f"❌ Ошибка выполнения бэктеста: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()