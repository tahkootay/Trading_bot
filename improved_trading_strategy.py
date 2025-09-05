#!/usr/bin/env python3
"""
Улучшенная торговая стратегия с риск-менеджментом
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.models.optimized_ensemble_predictor import OptimizedEnsemblePredictor

class ImprovedTradingStrategy:
    """Улучшенная торговая стратегия с риск-менеджментом."""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.position_price = 0
        self.trades = []
        self.equity_curve = []
        
        # Параметры риск-менеджмента
        self.max_position_size = 0.10  # Максимум 10% капитала на сделку
        self.stop_loss_pct = 0.05      # Стоп-лосс 5%
        self.take_profit_pct = 0.15    # Тейк-профит 15%
        self.min_probability = 0.75    # Минимальная вероятность для входа
        self.max_daily_loss = 0.03     # Максимум 3% потерь в день
        self.trail_stop_pct = 0.03     # Трейлинг стоп 3%
        
        # Фильтры сигналов
        self.min_signal_strength = 'STRONG'
        self.require_trend_alignment = True
        self.consecutive_losses_limit = 3
        
        # Состояние стратегии
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.highest_price_in_position = 0
        self.current_date = None
    
    def should_enter_position(self, prediction, current_price, technical_data):
        """Проверяет, стоит ли входить в позицию."""
        if not prediction:
            return False
        
        # Базовые условия
        if prediction['final_signal'] != 1:
            return False
        if prediction['final_probability'] < self.min_probability:
            return False
        if prediction['signal_strength'] != self.min_signal_strength:
            return False
        
        # Риск-менеджмент
        if self.daily_pnl < -self.initial_capital * self.max_daily_loss:
            return False  # Достигнут дневной лимит потерь
        
        if self.consecutive_losses >= self.consecutive_losses_limit:
            return False  # Слишком много подряд убыточных сделок
        
        # Проверка тренда (если включена)
        if self.require_trend_alignment:
            if not self._is_trend_aligned(technical_data, signal=1):
                return False
        
        # Проверка доступного капитала
        position_value = self.capital * self.max_position_size
        if position_value < 100:  # Минимальная сумма для сделки
            return False
        
        return True
    
    def should_exit_position(self, prediction, current_price, technical_data):
        """Проверяет, стоит ли выйти из позиции."""
        if self.position == 0:
            return False, None
        
        current_pnl_pct = (current_price - self.position_price) / self.position_price
        
        # Стоп-лосс
        if current_pnl_pct <= -self.stop_loss_pct:
            return True, "STOP_LOSS"
        
        # Тейк-профит
        if current_pnl_pct >= self.take_profit_pct:
            return True, "TAKE_PROFIT"
        
        # Трейлинг стоп
        if current_price > self.highest_price_in_position:
            self.highest_price_in_position = current_price
        
        trail_stop_price = self.highest_price_in_position * (1 - self.trail_stop_pct)
        if current_price <= trail_stop_price:
            return True, "TRAILING_STOP"
        
        # ML сигнал на выход
        if prediction and prediction['final_signal'] == 0:
            if prediction['final_probability'] < 0.4:
                return True, "ML_EXIT"
        
        return False, None
    
    def _is_trend_aligned(self, technical_data, signal):
        """Проверяет, совпадает ли сигнал с трендом."""
        if 'MA20' not in technical_data or 'MA50' not in technical_data:
            return True  # Если нет данных о тренде, разрешаем сделку
        
        # Восходящий тренд: MA20 > MA50
        uptrend = technical_data['MA20'] > technical_data.get('MA50', technical_data['MA20'])
        
        if signal == 1:  # BUY сигнал
            return uptrend
        else:  # SELL сигнал
            return not uptrend
    
    def execute_trade(self, action, current_price, current_time, prediction=None, exit_reason=None):
        """Исполняет сделку."""
        if action == 'BUY' and self.position == 0:
            # Открытие позиции
            position_value = self.capital * self.max_position_size
            self.position = position_value / current_price
            self.position_price = current_price
            self.highest_price_in_position = current_price
            
            commission = position_value * 0.001
            self.capital -= commission
            
            trade = {
                'type': 'BUY',
                'timestamp': current_time,
                'price': current_price,
                'size': self.position,
                'probability': prediction['final_probability'] if prediction else 0,
                'strength': prediction['signal_strength'] if prediction else 'UNKNOWN',
                'commission': commission,
                'base_predictions': prediction.get('base_predictions', {}) if prediction else {}
            }
            self.trades.append(trade)
            
        elif action == 'SELL' and self.position > 0:
            # Закрытие позиции
            exit_value = self.position * current_price
            commission = exit_value * 0.001
            pnl = (current_price - self.position_price) * self.position - commission
            
            self.capital += pnl
            self.daily_pnl += pnl
            
            # Обновляем счетчик убыточных сделок
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            trade = {
                'type': 'SELL',
                'timestamp': current_time,
                'price': current_price,
                'size': self.position,
                'pnl': pnl,
                'commission': commission,
                'return_pct': (current_price - self.position_price) / self.position_price * 100,
                'exit_reason': exit_reason or 'ML_EXIT'
            }
            self.trades.append(trade)
            
            self.position = 0
            self.position_price = 0
            self.highest_price_in_position = 0
    
    def update_equity(self, current_price, current_time):
        """Обновляет кривую эквити."""
        # Сброс дневного P&L в новый день
        if self.current_date != current_time.date():
            self.daily_pnl = 0
            self.current_date = current_time.date()
        
        if self.position > 0:
            unrealized_pnl = (current_price - self.position_price) * self.position
            total_equity = self.capital + (self.position * current_price)
        else:
            total_equity = self.capital
        
        self.equity_curve.append({
            'timestamp': current_time,
            'equity': total_equity,
            'price': current_price,
            'position': self.position,
            'daily_pnl': self.daily_pnl
        })
    
    def get_performance_metrics(self):
        """Возвращает метрики производительности."""
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        
        if not sell_trades:
            return {
                'total_return': 0,
                'win_rate': 0,
                'total_trades': len(buy_trades),
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        pnls = [t['pnl'] for t in sell_trades if 'pnl' in t]
        winning_trades = len([p for p in pnls if p > 0])
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        win_rate = winning_trades / len(pnls) * 100 if pnls else 0
        
        avg_win = np.mean([p for p in pnls if p > 0]) if winning_trades > 0 else 0
        avg_loss = abs(np.mean([p for p in pnls if p <= 0])) if len(pnls) > winning_trades else 0
        
        profit_factor = (avg_win * winning_trades) / (avg_loss * (len(pnls) - winning_trades)) if avg_loss > 0 else float('inf')
        
        # Максимальная просадка
        equity_values = [e['equity'] for e in self.equity_curve]
        if equity_values:
            equity_array = np.array(equity_values)
            running_max = np.maximum.accumulate(equity_array)
            drawdowns = (equity_array - running_max) / running_max * 100
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        else:
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': len(buy_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'final_capital': self.capital
        }

def prepare_features_for_prediction(df):
    """Подготовка признаков для ML предсказания с дополнительными индикаторами."""
    # Базовые MA
    df['MA5'] = df['close'].rolling(5, min_periods=1).mean()
    df['MA10'] = df['close'].rolling(10, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['close'].rolling(50, min_periods=1).mean()  # Дополнительная MA для тренда
    
    # EMA
    df['EMA12'] = df['close'].ewm(span=12, min_periods=1).mean()
    df['EMA26'] = df['close'].ewm(span=26, min_periods=1).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    bb_std = df['close'].rolling(20, min_periods=1).std()
    df['BB_hband'] = df['MA20'] + (bb_std * 2)
    df['BB_lband'] = df['MA20'] - (bb_std * 2)
    df['BB_width'] = df['BB_hband'] - df['BB_lband']
    df['BB_position'] = (df['close'] - df['BB_lband']) / (df['BB_width'] + 1e-8)
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
    df['vol_change'] = df['volume'].pct_change().fillna(0)
    
    # Price indicators
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['price_change_5'] = df['close'].pct_change(5).fillna(0)
    df['volatility'] = df['close'].rolling(14, min_periods=1).std() / (df['close'].rolling(14, min_periods=1).mean() + 1e-8)
    
    # Price position indicators
    df['high_low_pct'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    high_low_range = df['high'] - df['low'] + 1e-8
    df['close_to_high'] = (df['close'] - df['low']) / high_low_range
    df['close_to_low'] = (df['high'] - df['close']) / high_low_range
    
    # Timeframe
    df['timeframe_minutes'] = 5.0
    
    # Заполняем NaN
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def run_improved_backtest():
    """Запускает улучшенный бэктест."""
    print("🚀 Запуск улучшенного бэктеста с риск-менеджментом")
    print("=" * 60)
    
    # Инициализация
    predictor = OptimizedEnsemblePredictor(lazy_loading=True, show_progress=True)
    strategy = ImprovedTradingStrategy(initial_capital=10000)
    
    # Генерация тестовых данных
    np.random.seed(42)
    dates = pd.date_range(start='2024-08-11 00:00', end='2024-08-17 23:55', freq='5min')
    
    data = []
    current_price = 148.50
    
    for i, timestamp in enumerate(dates):
        change_pct = np.random.normal(0, 0.012)
        current_price *= (1 + change_pct)
        
        high = current_price * (1 + abs(np.random.normal(0, 0.003)))
        low = current_price * (1 - abs(np.random.normal(0, 0.003)))
        open_price = current_price * (1 + np.random.normal(0, 0.001))
        volume = np.random.uniform(800000, 2500000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df = prepare_features_for_prediction(df)
    
    print(f"📊 Данные подготовлены: {len(df)} свечей")
    print("💹 Запуск торговли с улучшенной стратегией...")
    
    # Торговый цикл
    test_indices = range(50, len(df), 10)  # Каждая 10-я свеча
    
    for i in test_indices:
        row = df.iloc[i:i+1].copy()
        current_price = row['close'].iloc[0]
        current_time = row['timestamp'].iloc[0]
        
        # Получаем ML предсказание
        try:
            prediction = predictor.predict_ensemble(row)
        except:
            prediction = None
        
        # Техническая данные для фильтров
        technical_data = {
            'MA20': row['MA20'].iloc[0],
            'MA50': row['MA50'].iloc[0],
            'RSI': row['RSI'].iloc[0],
            'MACD_diff': row['MACD_diff'].iloc[0]
        }
        
        # Проверка на выход из позиции
        should_exit, exit_reason = strategy.should_exit_position(prediction, current_price, technical_data)
        if should_exit:
            strategy.execute_trade('SELL', current_price, current_time, exit_reason=exit_reason)
        
        # Проверка на вход в позицию
        if strategy.position == 0 and strategy.should_enter_position(prediction, current_price, technical_data):
            strategy.execute_trade('BUY', current_price, current_time, prediction)
        
        # Обновление эквити
        strategy.update_equity(current_price, current_time)
    
    # Закрываем позицию если открыта
    if strategy.position > 0:
        last_price = df['close'].iloc[-1]
        last_time = df['timestamp'].iloc[-1]
        strategy.execute_trade('SELL', last_price, last_time, exit_reason='END_OF_PERIOD')
    
    # Результаты
    metrics = strategy.get_performance_metrics()
    
    print("\n📈 РЕЗУЛЬТАТЫ УЛУЧШЕННОЙ СТРАТЕГИИ:")
    print("=" * 50)
    print(f"💰 Общая доходность: {metrics['total_return']:+.2f}%")
    print(f"💰 Финальный капитал: ${metrics['final_capital']:,.2f}")
    print(f"🎯 Win Rate: {metrics['win_rate']:.1f}%")
    print(f"📊 Всего сделок: {metrics['total_trades']}")
    print(f"⚡ Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"📉 Максимальная просадка: {metrics['max_drawdown']:.2f}%")
    print(f"🔴 Подряд убыточных: {metrics['consecutive_losses']}")
    
    # Анализ сделок
    buy_trades = [t for t in strategy.trades if t['type'] == 'BUY']
    sell_trades = [t for t in strategy.trades if t['type'] == 'SELL']
    
    if sell_trades:
        exit_reasons = {}
        for trade in sell_trades:
            reason = trade.get('exit_reason', 'UNKNOWN')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        print("\n🚪 Причины выходов:")
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count}")
    
    return {
        'strategy': strategy,
        'metrics': metrics,
        'predictor': predictor
    }

if __name__ == "__main__":
    results = run_improved_backtest()