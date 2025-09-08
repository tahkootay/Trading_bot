"""
Агрессивная версия Bollinger Bands Mean Reversion Strategy

Более мягкие условия входа для увеличения количества сделок:
- Снижены пороги RSI
- Уменьшены требования к объёму
- Разрешены контртрендовые входы
- Сокращена минимальная дистанция от середины
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from enum import Enum

from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal


class PositionState(Enum):
    """Состояния позиции"""
    IDLE = "IDLE"
    ENTRY_PENDING_LONG = "ENTRY_PENDING_LONG"
    ENTRY_PENDING_SHORT = "ENTRY_PENDING_SHORT"
    ACTIVE_LONG = "ACTIVE_LONG"
    ACTIVE_SHORT = "ACTIVE_SHORT"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    CLOSED_PROFIT = "CLOSED_PROFIT"
    CLOSED_LOSS = "CLOSED_LOSS"


class AggressiveBollingerStrategy(StrategyBase):
    """
    Агрессивная версия стратегии отскоков от полос Боллинджера
    """
    
    def _initialize(self):
        """Инициализация с более агрессивными параметрами"""
        
        self.parameters = {
            # Символ и таймфреймы
            'symbol': 'SOLUSDT',
            'tf_entry': '5m',
            'tf_context': '15m',
            'tf_day': '1h',
            
            # Bollinger Bands
            'bb_period': 20,
            'bb_std': 2.0,
            
            # Трендовые фильтры (более мягкие)
            'ema_trend_short': 50,
            'ema_trend_long': 100,
            
            # RSI фильтры (АГРЕССИВНЫЕ)
            'rsi_period': 14,
            'rsi_lower': 45,      # Увеличено с 35 - больше сигналов
            'rsi_upper': 55,      # Уменьшено с 65 - больше сигналов
            
            # Объёмные фильтры (МЯГКИЕ)
            'volume_spike_mult': 1.0,   # Снижено с 1.25 - без требований к объёму
            'volume_avg_period': 20,
            
            # Дистанции и входы (АГРЕССИВНЫЕ)
            'min_distance_from_middle': 0.1,  # Снижено с 0.3 - ближе к середине
            'entry_offset': 0.01,             # Уменьшено с 0.02
            
            # ATR и стопы
            'atr_period': 14,
            'stop_atr_mult': 0.8,     # Снижено с 1.0 - более близкие стопы
            'fixed_buffer': 0.15,     # Снижено с 0.2
            
            # Управление позицией
            'partial_ratio': 0.5,
            'breakeven_buffer': 0.05,  # Снижено с 0.1
            'trailing_atr_mult': 0.6,  # Снижено с 0.8
            
            # Risk management (АГРЕССИВНЫЙ)
            'max_risk_per_trade_pct': 0.8,  # Увеличено с 0.5
            'max_trades_per_day': 15,       # Увеличено с 10
            'max_consecutive_losses': 4,    # Увеличено с 3
            'cooldown_hours': 1,            # Снижено с 2
            'entry_timeout_bars': 3,        # Увеличено с 2
            
            # Размер позиции
            'position_size_pct': 0.1,
            'contrarian_size_mult': 0.8,    # Увеличено с 0.5
            
            # Дополнительные фильтры (МЯГКИЕ)
            'allow_contrarian': True,
            'require_candle_confirmation': False,  # Отключено
            'min_bb_width': 0.2,              # Снижено с 0.5
        }
        
        # Состояние стратегии (аналогично базовой)
        self.state = {
            'history': [],
            'indicators': {},
            'position_state': PositionState.IDLE,
            'entry_price': None,
            'stop_price': None,
            'partial_filled': False,
            'position_data': None,
            'trades_today': 0,
            'consecutive_losses': 0,
            'last_trade_day': None,
            'circuit_breaker_until': None,
            'daily_pnl': 0.0,
            'entry_pending_since': None,
            'entry_timeout_count': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
        }

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Основная логика (упрощённая версия)"""
        
        # Обновляем историю и индикаторы
        self._update_history(data)
        self._update_indicators()
        
        # Проверяем достаточность данных
        if not self._has_sufficient_data():
            return TradeSignal(signal=Signal.HOLD)
        
        # Управление дневными лимитами
        self._reset_daily_counters(data.timestamp)
        
        # Проверяем circuit breaker
        if self._is_circuit_breaker_active(data.timestamp):
            return TradeSignal(signal=Signal.HOLD, reason="Circuit breaker active")
        
        # Проверяем дневные лимиты
        if self.state['trades_today'] >= self.parameters['max_trades_per_day']:
            return TradeSignal(signal=Signal.HOLD, reason="Daily trade limit reached")
        
        # Основная логика
        if position.quantity == 0:
            return self._check_entry_conditions(data)
        else:
            return self._manage_position(data, position)

    def _update_history(self, data: MarketData):
        """Обновление истории данных"""
        bar_data = {
            'timestamp': data.timestamp,
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'close': data.close,
            'volume': data.volume if hasattr(data, 'volume') else 0
        }
        self.state['history'].append(bar_data)
        
        # Ограничиваем размер истории
        if len(self.state['history']) > 300:
            self.state['history'] = self.state['history'][-300:]

    def _update_indicators(self):
        """Обновление индикаторов (упрощённая версия)"""
        if len(self.state['history']) < self.parameters['bb_period']:
            return
        
        df = pd.DataFrame(self.state['history'])
        
        # Bollinger Bands
        bb_period = self.parameters['bb_period']
        bb_std = self.parameters['bb_std']
        
        df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * bb_std)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * bb_std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # EMA (упрощённо)
        df['ema_short'] = df['close'].ewm(span=20).mean()  # Используем короткие периоды
        df['ema_long'] = df['close'].ewm(span=40).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR (упрощённый)
        df['high_low'] = df['high'] - df['low']
        df['atr'] = df['high_low'].rolling(window=self.parameters['atr_period']).mean()
        
        # Средний объём
        df['avg_volume'] = df['volume'].rolling(window=self.parameters['volume_avg_period']).mean()
        
        # Сохраняем актуальные индикаторы
        if len(df) > 0:
            last_row = df.iloc[-1]
            self.state['indicators'] = {
                'bb_mid': last_row.get('bb_mid', 0),
                'bb_upper': last_row.get('bb_upper', 0),
                'bb_lower': last_row.get('bb_lower', 0),
                'bb_width': last_row.get('bb_width', 0),
                'ema_short': last_row.get('ema_short', 0),
                'ema_long': last_row.get('ema_long', 0),
                'rsi': last_row.get('rsi', 50),
                'atr': last_row.get('atr', 0.5),
                'avg_volume': last_row.get('avg_volume', 1),
                'current_price': last_row['close'],
                'current_volume': last_row['volume']
            }

    def _has_sufficient_data(self) -> bool:
        """Проверка достаточности данных"""
        min_required = max(self.parameters['bb_period'], self.parameters['rsi_period'])
        return len(self.state['history']) >= min_required

    def _reset_daily_counters(self, timestamp: datetime):
        """Сброс дневных счётчиков"""
        current_day = timestamp.date()
        if self.state['last_trade_day'] != current_day:
            self.state['trades_today'] = 0
            self.state['daily_pnl'] = 0.0
            self.state['last_trade_day'] = current_day

    def _is_circuit_breaker_active(self, timestamp: datetime) -> bool:
        """Проверка circuit breaker"""
        if self.state['circuit_breaker_until']:
            return timestamp < self.state['circuit_breaker_until']
        return False

    def _check_entry_conditions(self, data: MarketData) -> TradeSignal:
        """Агрессивная проверка условий входа"""
        
        indicators = self.state['indicators']
        current_price = indicators['current_price']
        
        # Проверяем минимальную ширину полос (мягче)
        if indicators['bb_width'] < self.parameters['min_bb_width']:
            return TradeSignal(signal=Signal.HOLD, reason="BB width too narrow")
        
        # АГРЕССИВНЫЕ условия для лонга
        long_conditions = (
            self._price_near_bb_lower(current_price, indicators['bb_lower']) and
            indicators['rsi'] < self.parameters['rsi_lower'] and
            abs(current_price - indicators['bb_mid']) >= self.parameters['min_distance_from_middle']
        )
        
        if long_conditions:
            entry_price = current_price
            stop_price = indicators['bb_lower'] - self.parameters['fixed_buffer']
            
            self.state['position_data'] = {
                'direction': 'LONG',
                'entry_price': entry_price,
                'stop_price': stop_price,
                'bb_mid': indicators['bb_mid'],
                'bb_upper': indicators['bb_upper'],
            }
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"Aggressive BB long: price={current_price:.4f}, RSI={indicators['rsi']:.1f}"
            )
        
        # АГРЕССИВНЫЕ условия для шорта
        short_conditions = (
            self._price_near_bb_upper(current_price, indicators['bb_upper']) and
            indicators['rsi'] > self.parameters['rsi_upper'] and
            abs(current_price - indicators['bb_mid']) >= self.parameters['min_distance_from_middle']
        )
        
        if short_conditions:
            entry_price = current_price
            stop_price = indicators['bb_upper'] + self.parameters['fixed_buffer']
            
            self.state['position_data'] = {
                'direction': 'SHORT',
                'entry_price': entry_price,
                'stop_price': stop_price,
                'bb_mid': indicators['bb_mid'],
                'bb_lower': indicators['bb_lower'],
            }
            
            return TradeSignal(
                signal=Signal.SELL,
                reason=f"Aggressive BB short: price={current_price:.4f}, RSI={indicators['rsi']:.1f}"
            )
        
        return TradeSignal(signal=Signal.HOLD)

    def _price_near_bb_lower(self, current_price: float, bb_lower: float) -> bool:
        """Более мягкая проверка близости к нижней границе"""
        tolerance = bb_lower * 0.01  # 1% допуск (было 0.2%)
        return current_price <= bb_lower + tolerance

    def _price_near_bb_upper(self, current_price: float, bb_upper: float) -> bool:
        """Более мягкая проверка близости к верхней границе"""
        tolerance = bb_upper * 0.01  # 1% допуск (было 0.2%)
        return current_price >= bb_upper - tolerance

    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Упрощённое управление позицией"""
        
        if not self.state['position_data']:
            return TradeSignal(signal=Signal.HOLD)
        
        current_price = self.state['indicators']['current_price']
        position_data = self.state['position_data']
        direction = position_data['direction']
        
        # Проверяем стоп-лосс
        if self._check_stop_loss(current_price, position_data):
            self._record_trade_result('LOSS')
            return TradeSignal(
                signal=Signal.SELL if direction == 'LONG' else Signal.BUY,
                reason=f"Stop loss: {current_price:.4f}"
            )
        
        # Частичное закрытие на средней линии
        if not self.state['partial_filled']:
            if self._check_partial_target(current_price, position_data):
                self.state['partial_filled'] = True
                return TradeSignal(
                    signal=Signal.SELL if direction == 'LONG' else Signal.BUY,
                    reason=f"Partial target (mid): {current_price:.4f}"
                )
        
        # Полное закрытие на противоположной границе
        else:
            if self._check_full_target(current_price, position_data):
                self._record_trade_result('WIN')
                return TradeSignal(
                    signal=Signal.SELL if direction == 'LONG' else Signal.BUY,
                    reason=f"Full target: {current_price:.4f}"
                )
        
        return TradeSignal(signal=Signal.HOLD)

    def _check_stop_loss(self, current_price: float, position_data: Dict[str, Any]) -> bool:
        """Проверка стоп-лосса"""
        stop_price = position_data['stop_price']
        direction = position_data['direction']
        
        if direction == 'LONG':
            return current_price <= stop_price
        else:
            return current_price >= stop_price

    def _check_partial_target(self, current_price: float, position_data: Dict[str, Any]) -> bool:
        """Проверка частичной цели"""
        bb_mid = position_data['bb_mid']
        direction = position_data['direction']
        
        if direction == 'LONG':
            return current_price >= bb_mid
        else:
            return current_price <= bb_mid

    def _check_full_target(self, current_price: float, position_data: Dict[str, Any]) -> bool:
        """Проверка полной цели"""
        direction = position_data['direction']
        
        if direction == 'LONG':
            return current_price >= position_data['bb_upper']
        else:
            return current_price <= position_data['bb_lower']

    def _record_trade_result(self, result: str):
        """Запись результата сделки"""
        self.state['total_trades'] += 1
        self.state['trades_today'] += 1
        
        if result == 'WIN':
            self.state['winning_trades'] += 1
            self.state['consecutive_losses'] = 0
        else:
            self.state['consecutive_losses'] += 1
            
            # Circuit breaker
            if self.state['consecutive_losses'] >= self.parameters['max_consecutive_losses']:
                self.state['circuit_breaker_until'] = (
                    datetime.now() + timedelta(hours=self.parameters['cooldown_hours'])
                )
                self.state['consecutive_losses'] = 0
        
        # Сброс состояния позиции
        self.state['position_state'] = PositionState.CLOSED_PROFIT if result == 'WIN' else PositionState.CLOSED_LOSS
        self.state['position_data'] = None
        self.state['partial_filled'] = False