#!/usr/bin/env python3
"""
ML Backtester - тестирование ML модели на исторических данных с эмуляцией торговли.

Показывает какой результат получился бы при реальной торговле по предсказаниям модели.
"""

import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MLBacktester:
    """Бэктестер для ML моделей."""
    
    def __init__(self, model_path: str, data_path: str, initial_balance: float = 1000.0, 
                 fee: float = 0.001, buy_threshold: float = 0.6, sell_threshold: float = 0.4):
        """
        Инициализация бэктестера.
        
        Args:
            model_path: Путь к обученной модели
            data_path: Путь к тестовым данным
            initial_balance: Начальный баланс
            fee: Комиссия за сделку (доля от оборота)
            buy_threshold: Порог вероятности для покупки
            sell_threshold: Порог вероятности для продажи
        """
        self.model_path = model_path
        self.data_path = data_path
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.fee = fee
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        # Состояние позиции
        self.position = None  # 'long' или None
        self.entry_price = 0.0
        self.entry_time = None
        self.position_size = 0.0
        
        # История
        self.trades = []
        self.balance_history = []
        self.equity_curve = []
        
        # Загрузка модели и данных
        self.model = None
        self.scaler = None
        self.data = None
        
    def load_model(self):
        """Загрузка обученной модели."""
        print(f"📥 Loading model: {self.model_path}")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Попытка загрузить скалер
        scaler_path = self.model_path.replace('.pkl', '_scaler.pkl')
        if not Path(scaler_path).exists():
            # Альтернативные пути для скалера
            possible_scaler_paths = [
                self.model_path.replace('rf_', 'scaler_'),
                self.model_path.replace('models/', 'models/scaler_'),
                'models/scaler.pkl',
                'models/optimized/scaler_horizon_10_optimized.pkl'
            ]
            
            for path in possible_scaler_paths:
                if Path(path).exists():
                    scaler_path = path
                    break
        
        if Path(scaler_path).exists():
            print(f"📥 Loading scaler: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            print("⚠️  No scaler found - using raw features")
        
        print(f"✅ Model loaded successfully")
    
    def load_data(self):
        """Загрузка тестовых данных."""
        print(f"📊 Loading test data: {self.data_path}")
        
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.data = pd.read_csv(self.data_path)
        
        # Проверка обязательных колонок
        required_cols = ['close']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"✅ Loaded {len(self.data)} rows of test data")
        print(f"📈 Price range: {self.data['close'].min():.2f} - {self.data['close'].max():.2f}")
        
        # Добавляем временные метки если их нет
        if 'timestamp' not in self.data.columns:
            self.data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(self.data), freq='5T')
        
        return self.data
    
    def prepare_features(self, row_idx: int) -> np.ndarray:
        """Подготовка фичей для модели."""
        
        # Исключаем служебные колонки
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Добавляем target колонки если есть
        target_cols = [col for col in self.data.columns if col.startswith('target')]
        exclude_cols.extend(target_cols)
        
        # Получаем фичи
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("No feature columns found in data")
        
        features = self.data.iloc[row_idx][feature_cols].values.reshape(1, -1)
        
        # Применяем скалер если есть
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        return features
    
    def get_signal(self, row_idx: int) -> str:
        """Получение торгового сигнала от модели."""
        
        features = self.prepare_features(row_idx)
        
        # Получаем вероятности
        try:
            probabilities = self.model.predict_proba(features)[0]
            prob_up = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        except Exception as e:
            print(f"⚠️  Error getting prediction: {e}")
            return 'hold'
        
        # Принимаем решение
        if prob_up > self.buy_threshold:
            return 'buy'
        elif prob_up < self.sell_threshold:
            return 'sell'
        else:
            return 'hold'
    
    def open_position(self, price: float, timestamp, signal: str):
        """Открытие позиции."""
        
        if signal == 'buy' and self.position is None:
            # Открываем лонг позицию
            self.position = 'long'
            self.entry_price = price
            self.entry_time = timestamp
            
            # Рассчитываем размер позиции (всем балансом минус комиссия)
            self.position_size = self.current_balance / price * (1 - self.fee)
            
            print(f"📈 OPEN LONG at {price:.4f} | Balance: {self.current_balance:.2f}")
            
    def close_position(self, price: float, timestamp, signal: str):
        """Закрытие позиции."""
        
        if self.position == 'long' and signal == 'sell':
            # Закрываем лонг позицию
            exit_price = price
            
            # Рассчитываем прибыль
            gross_profit = self.position_size * exit_price
            fee_amount = gross_profit * self.fee
            net_profit = gross_profit - fee_amount
            
            profit_pct = (net_profit - self.current_balance) / self.current_balance
            
            # Обновляем баланс
            self.current_balance = net_profit
            
            # Записываем сделку
            trade = {
                'entry_time': self.entry_time,
                'exit_time': timestamp,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'position_size': self.position_size,
                'gross_profit': gross_profit,
                'fee_amount': fee_amount,
                'net_profit': net_profit,
                'profit_pct': profit_pct,
                'profit_abs': net_profit - self.current_balance + (net_profit - self.current_balance),
                'duration_bars': len(self.balance_history) - len([t for t in self.trades])
            }
            
            self.trades.append(trade)
            
            print(f"📉 CLOSE LONG at {price:.4f} | Profit: {profit_pct:+.2%} | Balance: {self.current_balance:.2f}")
            
            # Сбрасываем позицию
            self.position = None
            self.entry_price = 0.0
            self.entry_time = None
            self.position_size = 0.0
    
    def update_equity_curve(self, price: float, timestamp):
        """Обновление кривой баланса."""
        
        # Текущая стоимость портфеля
        if self.position == 'long':
            # Учитываем текущую стоимость открытой позиции
            current_value = self.position_size * price * (1 - self.fee)
        else:
            current_value = self.current_balance
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': self.current_balance,
            'equity': current_value,
            'price': price
        })
    
    def run_backtest(self):
        """Запуск бэктеста."""
        
        print(f"\n🚀 STARTING BACKTEST")
        print("=" * 50)
        print(f"💰 Initial balance: ${self.initial_balance:,.2f}")
        print(f"📊 Test period: {len(self.data)} bars")
        print(f"🎯 Buy threshold: {self.buy_threshold}")
        print(f"🎯 Sell threshold: {self.sell_threshold}")
        print(f"💸 Fee: {self.fee:.1%}")
        
        # Итерируемся по данным
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            price = row['close']
            timestamp = row['timestamp'] if 'timestamp' in row else i
            
            # Получаем сигнал
            signal = self.get_signal(i)
            
            # Выполняем торговые действия
            if signal == 'buy':
                self.open_position(price, timestamp, signal)
            elif signal == 'sell':
                self.close_position(price, timestamp, signal)
            
            # Обновляем кривую баланса
            self.update_equity_curve(price, timestamp)
            
            # Прогресс (реже выводим для ускорения)
            if i % 5000 == 0 or i == len(self.data) - 1:
                progress = (i + 1) / len(self.data) * 100
                print(f"📊 Progress: {progress:.1f}% | Trades: {len(self.trades)} | Balance: ${self.current_balance:.2f}")
        
        # Закрываем открытую позицию в конце
        if self.position == 'long':
            final_price = self.data.iloc[-1]['close']
            final_timestamp = self.data.iloc[-1]['timestamp'] if 'timestamp' in self.data.columns else len(self.data) - 1
            self.close_position(final_price, final_timestamp, 'sell')
        
        print(f"\n✅ Backtest completed!")
    
    def calculate_statistics(self) -> Dict:
        """Расчёт статистики бэктеста."""
        
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Базовая статистика
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['profit_pct'] > 0]
        losing_trades = [t for t in self.trades if t['profit_pct'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Прибыли и убытки
        profits = [t['profit_pct'] for t in self.trades]
        total_return = (self.current_balance / self.initial_balance) - 1
        
        # Максимальная просадка
        equity_values = [point['equity'] for point in self.equity_curve]
        if equity_values:
            running_max = np.maximum.accumulate(equity_values)
            drawdowns = (np.array(equity_values) - running_max) / running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        else:
            max_drawdown = 0
        
        # Лучшая и худшая сделка
        best_trade = max(profits) if profits else 0
        worst_trade = min(profits) if profits else 0
        
        # Средние значения
        avg_profit = np.mean(profits) if profits else 0
        avg_win = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_pct'] for t in losing_trades]) if losing_trades else 0
        
        # Sharpe ratio (упрощённый)
        if len(profits) > 1:
            sharpe_ratio = np.mean(profits) / np.std(profits) * np.sqrt(252) if np.std(profits) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Buy & Hold сравнение
        start_price = self.data.iloc[0]['close']
        end_price = self.data.iloc[-1]['close']
        buy_hold_return = (end_price / start_price) - 1
        
        stats = {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'buy_hold_return': buy_hold_return,
            'vs_buy_hold': total_return - buy_hold_return
        }
        
        return stats
    
    def print_statistics(self):
        """Вывод статистики в консоль."""
        
        stats = self.calculate_statistics()
        
        if 'error' in stats:
            print(f"❌ {stats['error']}")
            return
        
        print(f"\n📊 BACKTEST RESULTS")
        print("=" * 50)
        print(f"💰 Initial balance:  ${stats['initial_balance']:,.2f}")
        print(f"📈 Final balance:    ${stats['final_balance']:,.2f}")
        print(f"🎯 Total return:     {stats['total_return']:+.2%}")
        print(f"📊 Total trades:     {stats['total_trades']}")
        print(f"✅ Winning trades:   {stats['winning_trades']} ({stats['win_rate']:.1%})")
        print(f"❌ Losing trades:    {stats['losing_trades']}")
        print(f"📈 Average profit:   {stats['avg_profit']:+.2%}")
        print(f"🟢 Average win:      {stats['avg_win']:+.2%}")
        print(f"🔴 Average loss:     {stats['avg_loss']:+.2%}")
        print(f"🚀 Best trade:       {stats['best_trade']:+.2%}")
        print(f"💀 Worst trade:      {stats['worst_trade']:+.2%}")
        print(f"⚠️  Max drawdown:    {stats['max_drawdown']:+.2%}")
        print(f"📉 Sharpe ratio:     {stats['sharpe_ratio']:.2f}")
        
        print(f"\n🔄 COMPARISON")
        print("-" * 30)
        print(f"📊 Buy & Hold:       {stats['buy_hold_return']:+.2%}")
        print(f"🤖 ML Strategy:      {stats['total_return']:+.2%}")
        print(f"🎯 Difference:       {stats['vs_buy_hold']:+.2%}")
        
        if stats['vs_buy_hold'] > 0:
            print("✅ ML strategy outperformed Buy & Hold!")
        else:
            print("❌ ML strategy underperformed Buy & Hold")
    
    def save_results(self, output_dir: str = "results"):
        """Сохранение результатов."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Сохраняем сделки
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = output_path / "backtest_trades.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"💾 Trades saved: {trades_file}")
        
        # Сохраняем кривую баланса
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_file = output_path / "backtest_equity_curve.csv"
            equity_df.to_csv(equity_file, index=False)
            print(f"💾 Equity curve saved: {equity_file}")
        
        # Сохраняем статистику
        stats = self.calculate_statistics()
        if 'error' not in stats:
            stats_file = output_path / "backtest_statistics.txt"
            with open(stats_file, 'w') as f:
                f.write("BACKTEST STATISTICS\n")
                f.write("=" * 30 + "\n\n")
                
                for key, value in stats.items():
                    if isinstance(value, float):
                        if 'return' in key or 'rate' in key or 'profit' in key or 'drawdown' in key:
                            f.write(f"{key}: {value:+.2%}\n")
                        else:
                            f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            print(f"💾 Statistics saved: {stats_file}")

def main():
    """Главная функция."""
    
    parser = argparse.ArgumentParser(description='ML Backtester - тестирование ML модели на исторических данных')
    
    parser.add_argument('--model', type=str, required=True, help='Путь к обученной модели (.pkl)')
    parser.add_argument('--data', type=str, required=True, help='Путь к тестовым данным (.csv)')
    parser.add_argument('--initial_balance', type=float, default=1000.0, help='Начальный баланс (default: 1000)')
    parser.add_argument('--fee', type=float, default=0.001, help='Комиссия за сделку (default: 0.001 = 0.1%)')
    parser.add_argument('--buy_threshold', type=float, default=0.6, help='Порог вероятности для покупки (default: 0.6)')
    parser.add_argument('--sell_threshold', type=float, default=0.4, help='Порог вероятности для продажи (default: 0.4)')
    parser.add_argument('--output_dir', type=str, default='results', help='Папка для сохранения результатов')
    
    args = parser.parse_args()
    
    print("🤖 ML BACKTESTER")
    print("=" * 40)
    
    try:
        # Создаём бэктестер
        backtester = MLBacktester(
            model_path=args.model,
            data_path=args.data,
            initial_balance=args.initial_balance,
            fee=args.fee,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold
        )
        
        # Загружаем модель и данные
        backtester.load_model()
        backtester.load_data()
        
        # Запускаем бэктест
        backtester.run_backtest()
        
        # Выводим статистику
        backtester.print_statistics()
        
        # Сохраняем результаты
        backtester.save_results(args.output_dir)
        
        print(f"\n🎉 Backtest completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())