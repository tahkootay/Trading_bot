"""
Bollinger Bands Mean Reversion Strategy

Стратегия маятниковых отскоков от границ полос Боллинджера с частичной фиксацией
на средней линии и опциональным расширением до противоположной границы.

Основные принципы:
1. Входы от крайних границ BB при подтверждении тренда/RSI/объёма
2. Частичная фиксация на средней линии (50% позиции)
3. Продолжение до противоположной границы при наличии импульса
4. Строгий risk management и фильтры качества
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


class BollingerMeanReversionStrategy(StrategyBase):
    """
    Bollinger Bands Mean Reversion Strategy
    Ловит отскоки от границ полос Боллинджера к средней линии
    """
    
    def _initialize(self):
        """Инициализация параметров стратегии"""
        
        # Основные параметры торговли
        self.parameters = {
            # Символ и таймфреймы
            'symbol': 'SOLUSDT',
            'tf_entry': '5m',
            'tf_context': '15m',  # для трендового фильтра
            'tf_day': '1h',       # дневной фильтр
            
            # Bollinger Bands
            'bb_period': 20,      # период SMA для BB
            'bb_std': 2.0,        # количество стандартных отклонений
            
            # Трендовые фильтры
            'ema_trend_short': 50,   # на 15m контексте
            'ema_trend_long': 100,   # на 15m контексте
            
            # RSI фильтры
            'rsi_period': 14,
            'rsi_lower': 35,      # порог для лонгов
            'rsi_upper': 65,      # порог для шортов
            
            # Объёмные фильтры
            'volume_spike_mult': 1.25,  # объём >= 1.25 * avg для подтверждения
            'volume_avg_period': 20,
            
            # Дистанции и входы
            'min_distance_from_middle': 0.3,  # мин. расстояние от mid в USDT
            'entry_offset': 0.02,             # отступ от границы BB для лимит-ордера
            
            # ATR и стопы
            'atr_period': 14,
            'stop_atr_mult': 1.0,     # стоп = ATR * mult
            'fixed_buffer': 0.2,      # фиксированный буфер для стопа (USDT)
            
            # Управление позицией
            'partial_ratio': 0.5,     # доля позиции для частичного закрытия
            'breakeven_buffer': 0.1,  # буфер для перевода в безубыток
            'trailing_atr_mult': 0.8, # trailing stop после partial fill
            
            # Risk management
            'max_risk_per_trade_pct': 0.5,  # макс риск на сделку
            'max_trades_per_day': 10,
            'max_consecutive_losses': 3,
            'cooldown_hours': 2,      # часов пауза после circuit breaker
            'entry_timeout_bars': 2,  # таймаут на исполнение лимит-ордера
            
            # Размер позиции
            'position_size_pct': 0.1,  # базовый размер позиции
            'contrarian_size_mult': 0.5,  # множитель для контртрендовых входов
            
            # Дополнительные фильтры
            'allow_contrarian': True,    # разрешить контртрендовые входы
            'require_candle_confirmation': True,  # требовать свечные паттерны
            'min_bb_width': 0.5,        # минимальная ширина BB в USDT
        }
        
        # Состояние стратегии
        self.state = {
            # История данных
            'history': [],
            'indicators': {},
            
            # Состояние позиции
            'position_state': PositionState.IDLE,
            'entry_price': None,
            'stop_price': None,
            'partial_filled': False,
            'position_data': None,
            
            # Управление рисками
            'trades_today': 0,
            'consecutive_losses': 0,
            'last_trade_day': None,
            'circuit_breaker_until': None,
            'daily_pnl': 0.0,
            
            # Таймауты и ожидание
            'entry_pending_since': None,
            'entry_timeout_count': 0,
            
            # Статистика
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
        }

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """
        Основная логика обработки новой свечи
        """
        
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
        
        # Основная логика в зависимости от состояния
        if position.quantity == 0:  # Нет позиции
            return self._check_entry_conditions(data)
        else:  # Есть позиция
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
        
        # Ограничиваем размер истории (сохраняем 500 свечей)
        max_history = 500
        if len(self.state['history']) > max_history:
            self.state['history'] = self.state['history'][-max_history:]

    def _update_indicators(self):
        """Обновление всех технических индикаторов"""
        
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
        
        # EMA для трендового фильтра (упрощённо, на том же TF)
        df['ema_short'] = df['close'].ewm(span=self.parameters['ema_trend_short']).mean()
        df['ema_long'] = df['close'].ewm(span=self.parameters['ema_trend_long']).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=self.parameters['atr_period']).mean()
        
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
                'atr': last_row.get('atr', 0),
                'avg_volume': last_row.get('avg_volume', 0),
                'current_price': last_row['close'],
                'current_volume': last_row['volume']
            }

    def _has_sufficient_data(self) -> bool:
        """Проверка достаточности данных для расчётов"""
        min_required = max(
            self.parameters['bb_period'],
            self.parameters['ema_trend_long'],
            self.parameters['rsi_period'],
            self.parameters['atr_period'],
            self.parameters['volume_avg_period']
        )
        return len(self.state['history']) >= min_required

    def _reset_daily_counters(self, timestamp: datetime):
        """Сброс дневных счётчиков"""
        current_day = timestamp.date()
        if self.state['last_trade_day'] != current_day:
            self.state['trades_today'] = 0
            self.state['daily_pnl'] = 0.0
            self.state['last_trade_day'] = current_day

    def _is_circuit_breaker_active(self, timestamp: datetime) -> bool:
        """Проверка активности circuit breaker"""
        if self.state['circuit_breaker_until']:
            return timestamp < self.state['circuit_breaker_until']
        return False

    def _check_entry_conditions(self, data: MarketData) -> TradeSignal:
        """Проверка условий для входа в позицию"""
        
        indicators = self.state['indicators']
        
        # Проверяем минимальную ширину полос
        if indicators['bb_width'] < self.parameters['min_bb_width']:
            return TradeSignal(signal=Signal.HOLD, reason="BB width too narrow")
        
        # Проверяем условия для лонга (отскок от нижней границы)
        long_signal = self._check_long_entry_conditions(data, indicators)
        if long_signal:
            return long_signal
        
        # Проверяем условия для шорта (отскок от верхней границы)
        short_signal = self._check_short_entry_conditions(data, indicators)
        if short_signal:
            return short_signal
        
        return TradeSignal(signal=Signal.HOLD)

    def _check_long_entry_conditions(self, data: MarketData, indicators: Dict[str, float]) -> Optional[TradeSignal]:
        """Проверка условий для лонг-позиции"""
        
        current_price = indicators['current_price']
        bb_lower = indicators['bb_lower']
        bb_mid = indicators['bb_mid']
        
        # 1. Цена должна коснуться или опуститься ниже нижней границы BB
        if not self._price_touched_bb_lower(current_price, bb_lower):
            return None
        
        # 2. Проверяем расстояние до средней линии
        distance_to_mid = abs(current_price - bb_mid)
        if distance_to_mid < self.parameters['min_distance_from_middle']:
            return None
        
        # 3. Трендовый фильтр
        trend_ok = self._check_trend_filter_long(indicators)
        if not trend_ok and not self.parameters['allow_contrarian']:
            return None
        
        # 4. RSI фильтр
        if indicators['rsi'] > self.parameters['rsi_lower']:
            return None
        
        # 5. Объёмный фильтр
        volume_ok = self._check_volume_confirmation(indicators)
        if not volume_ok:
            return None
        
        # 6. Свечное подтверждение (опционально)
        if self.parameters['require_candle_confirmation']:
            candle_ok = self._check_bullish_candle_pattern()
            if not candle_ok:
                return None
        
        # Все условия выполнены - генерируем сигнал
        entry_price = max(current_price, bb_lower + self.parameters['entry_offset'])
        stop_price = self._calculate_stop_price(entry_price, bb_lower, indicators['atr'], 'LONG')
        
        # Определяем размер позиции
        position_size = self._calculate_position_size(entry_price, stop_price, trend_ok)
        
        if position_size <= 0:
            return None
        
        # Сохраняем данные для управления позицией
        self.state['position_data'] = {
            'direction': 'LONG',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'bb_mid': bb_mid,
            'bb_upper': indicators['bb_upper'],
            'position_size': position_size
        }
        
        self.state['position_state'] = PositionState.ENTRY_PENDING_LONG
        self.state['entry_pending_since'] = data.timestamp
        
        return TradeSignal(
            signal=Signal.BUY,
            reason=f"BB bounce long: price={current_price:.4f}, bb_lower={bb_lower:.4f}, "
                   f"RSI={indicators['rsi']:.1f}, vol_mult={indicators['current_volume']/indicators['avg_volume']:.2f}"
        )

    def _check_short_entry_conditions(self, data: MarketData, indicators: Dict[str, float]) -> Optional[TradeSignal]:
        """Проверка условий для шорт-позиции"""
        
        current_price = indicators['current_price']
        bb_upper = indicators['bb_upper']
        bb_mid = indicators['bb_mid']
        
        # 1. Цена должна коснуться или подняться выше верхней границы BB
        if not self._price_touched_bb_upper(current_price, bb_upper):
            return None
        
        # 2. Проверяем расстояние до средней линии
        distance_to_mid = abs(current_price - bb_mid)
        if distance_to_mid < self.parameters['min_distance_from_middle']:
            return None
        
        # 3. Трендовый фильтр
        trend_ok = self._check_trend_filter_short(indicators)
        if not trend_ok and not self.parameters['allow_contrarian']:
            return None
        
        # 4. RSI фильтр
        if indicators['rsi'] < self.parameters['rsi_upper']:
            return None
        
        # 5. Объёмный фильтр
        volume_ok = self._check_volume_confirmation(indicators)
        if not volume_ok:
            return None
        
        # 6. Свечное подтверждение (опционально)
        if self.parameters['require_candle_confirmation']:
            candle_ok = self._check_bearish_candle_pattern()
            if not candle_ok:
                return None
        
        # Все условия выполнены - генерируем сигнал
        entry_price = min(current_price, bb_upper - self.parameters['entry_offset'])
        stop_price = self._calculate_stop_price(entry_price, bb_upper, indicators['atr'], 'SHORT')
        
        # Определяем размер позиции
        position_size = self._calculate_position_size(entry_price, stop_price, trend_ok)
        
        if position_size <= 0:
            return None
        
        # Сохраняем данные для управления позицией
        self.state['position_data'] = {
            'direction': 'SHORT',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'bb_mid': bb_mid,
            'bb_lower': indicators['bb_lower'],
            'position_size': position_size
        }
        
        self.state['position_state'] = PositionState.ENTRY_PENDING_SHORT
        self.state['entry_pending_since'] = data.timestamp
        
        return TradeSignal(
            signal=Signal.SELL,
            reason=f"BB bounce short: price={current_price:.4f}, bb_upper={bb_upper:.4f}, "
                   f"RSI={indicators['rsi']:.1f}, vol_mult={indicators['current_volume']/indicators['avg_volume']:.2f}"
        )

    def _price_touched_bb_lower(self, current_price: float, bb_lower: float) -> bool:
        """Проверка касания нижней границы BB"""
        # Проверяем, что цена близка к нижней границе или ниже
        tolerance = bb_lower * 0.002  # 0.2% допуск
        return current_price <= bb_lower + tolerance

    def _price_touched_bb_upper(self, current_price: float, bb_upper: float) -> bool:
        """Проверка касания верхней границы BB"""
        # Проверяем, что цена близка к верхней границе или выше
        tolerance = bb_upper * 0.002  # 0.2% допуск
        return current_price >= bb_upper - tolerance

    def _check_trend_filter_long(self, indicators: Dict[str, float]) -> bool:
        """Трендовый фильтр для лонгов"""
        return indicators['ema_short'] > indicators['ema_long']

    def _check_trend_filter_short(self, indicators: Dict[str, float]) -> bool:
        """Трендовый фильтр для шортов"""
        return indicators['ema_short'] < indicators['ema_long']

    def _check_volume_confirmation(self, indicators: Dict[str, float]) -> bool:
        """Проверка объёмного подтверждения"""
        required_volume = indicators['avg_volume'] * self.parameters['volume_spike_mult']
        return indicators['current_volume'] >= required_volume

    def _check_bullish_candle_pattern(self) -> bool:
        """Проверка бычьего свечного паттерна"""
        if len(self.state['history']) < 2:
            return True  # Не требуем, если недостаточно данных
        
        current = self.state['history'][-1]
        prev = self.state['history'][-2]
        
        # Простые паттерны: hammer, bullish engulfing, или просто бычья свеча
        is_bullish_candle = current['close'] > current['open']
        is_hammer = (current['close'] - current['low']) > 2 * abs(current['close'] - current['open'])
        is_engulfing = (current['open'] < prev['close'] and 
                       current['close'] > prev['open'] and
                       current['close'] > prev['high'])
        
        return is_bullish_candle or is_hammer or is_engulfing

    def _check_bearish_candle_pattern(self) -> bool:
        """Проверка медвежьего свечного паттерна"""
        if len(self.state['history']) < 2:
            return True  # Не требуем, если недостаточно данных
        
        current = self.state['history'][-1]
        prev = self.state['history'][-2]
        
        # Простые паттерны: shooting star, bearish engulfing, или просто медвежья свеча
        is_bearish_candle = current['close'] < current['open']
        is_shooting_star = (current['high'] - current['close']) > 2 * abs(current['close'] - current['open'])
        is_engulfing = (current['open'] > prev['close'] and 
                       current['close'] < prev['open'] and
                       current['close'] < prev['low'])
        
        return is_bearish_candle or is_shooting_star or is_engulfing

    def _calculate_stop_price(self, entry_price: float, bb_boundary: float, atr: float, direction: str) -> float:
        """Расчёт цены стоп-лосса"""
        
        atr_buffer = atr * self.parameters['stop_atr_mult']
        fixed_buffer = self.parameters['fixed_buffer']
        
        buffer = max(atr_buffer, fixed_buffer)
        
        if direction == 'LONG':
            return min(bb_boundary - buffer, entry_price - buffer)
        else:  # SHORT
            return max(bb_boundary + buffer, entry_price + buffer)

    def _calculate_position_size(self, entry_price: float, stop_price: float, trend_ok: bool) -> float:
        """Расчёт размера позиции на основе риска"""
        
        # Базовый размер позиции
        base_capital = 10000  # Условный капитал для расчёта
        risk_amount = base_capital * self.parameters['max_risk_per_trade_pct'] / 100
        
        # Расстояние до стопа
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return 0
        
        # Размер позиции исходя из риска
        position_value = risk_amount / (stop_distance / entry_price)
        position_size = position_value / entry_price
        
        # Корректировка для контртрендовых входов
        if not trend_ok and self.parameters['allow_contrarian']:
            position_size *= self.parameters['contrarian_size_mult']
        
        return position_size

    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Управление открытой позицией"""
        
        if not self.state['position_data']:
            return TradeSignal(signal=Signal.HOLD)
        
        current_price = self.state['indicators']['current_price']
        position_data = self.state['position_data']
        direction = position_data['direction']
        
        # Расчёт текущего P&L
        if direction == 'LONG':
            pnl = (current_price - position_data['entry_price']) * abs(position.quantity)
        else:
            pnl = (position_data['entry_price'] - current_price) * abs(position.quantity)
        
        # Проверяем стоп-лосс
        stop_hit = self._check_stop_loss(current_price, position_data)
        if stop_hit:
            self._record_trade_result('LOSS', pnl)
            return TradeSignal(
                signal=Signal.SELL if direction == 'LONG' else Signal.BUY,
                reason=f"Stop loss hit: {current_price:.4f} vs stop {position_data['stop_price']:.4f}"
            )
        
        # Частичное закрытие на средней линии
        if not self.state['partial_filled']:
            partial_target_hit = self._check_partial_target(current_price, position_data)
            if partial_target_hit:
                self.state['partial_filled'] = True
                # Переносим стоп в безубыток
                self._move_stop_to_breakeven(position_data)
                
                return TradeSignal(
                    signal=Signal.SELL if direction == 'LONG' else Signal.BUY,
                    reason=f"Partial target (BB mid) reached: {current_price:.4f}"
                )
        
        # Полное закрытие на противоположной границе
        else:  # После частичного закрытия
            full_target_hit = self._check_full_target(current_price, position_data)
            if full_target_hit:
                self._record_trade_result('WIN', pnl)
                return TradeSignal(
                    signal=Signal.SELL if direction == 'LONG' else Signal.BUY,
                    reason=f"Full target (opposite BB) reached: {current_price:.4f}"
                )
            
            # Trailing stop после частичного закрытия
            trailing_stop_hit = self._check_trailing_stop(current_price, position_data)
            if trailing_stop_hit:
                self._record_trade_result('WIN', pnl)  # Считаем профитной, т.к. часть уже закрыта
                return TradeSignal(
                    signal=Signal.SELL if direction == 'LONG' else Signal.BUY,
                    reason=f"Trailing stop hit: {current_price:.4f}"
                )
        
        return TradeSignal(signal=Signal.HOLD)

    def _check_stop_loss(self, current_price: float, position_data: Dict[str, Any]) -> bool:
        """Проверка срабатывания стоп-лосса"""
        stop_price = position_data['stop_price']
        direction = position_data['direction']
        
        if direction == 'LONG':
            return current_price <= stop_price
        else:
            return current_price >= stop_price

    def _check_partial_target(self, current_price: float, position_data: Dict[str, Any]) -> bool:
        """Проверка достижения частичной цели (средняя линия BB)"""
        bb_mid = position_data['bb_mid']
        direction = position_data['direction']
        
        if direction == 'LONG':
            return current_price >= bb_mid
        else:
            return current_price <= bb_mid

    def _check_full_target(self, current_price: float, position_data: Dict[str, Any]) -> bool:
        """Проверка достижения полной цели (противоположная граница BB)"""
        direction = position_data['direction']
        
        if direction == 'LONG':
            target = position_data['bb_upper']
            return current_price >= target
        else:
            target = position_data['bb_lower']
            return current_price <= target

    def _check_trailing_stop(self, current_price: float, position_data: Dict[str, Any]) -> bool:
        """Проверка trailing stop после частичного закрытия"""
        # Упрощённый trailing stop на основе ATR
        atr = self.state['indicators']['atr']
        trailing_distance = atr * self.parameters['trailing_atr_mult']
        direction = position_data['direction']
        entry_price = position_data['entry_price']
        
        # Устанавливаем trailing stop относительно входа с буфером
        if direction == 'LONG':
            trailing_stop = entry_price + self.parameters['breakeven_buffer'] - trailing_distance
            return current_price <= trailing_stop
        else:
            trailing_stop = entry_price - self.parameters['breakeven_buffer'] + trailing_distance
            return current_price >= trailing_stop

    def _move_stop_to_breakeven(self, position_data: Dict[str, Any]):
        """Перенос стопа в безубыток после частичного закрытия"""
        entry_price = position_data['entry_price']
        direction = position_data['direction']
        buffer = self.parameters['breakeven_buffer']
        
        if direction == 'LONG':
            position_data['stop_price'] = entry_price + buffer
        else:
            position_data['stop_price'] = entry_price - buffer

    def _record_trade_result(self, result: str, pnl: float):
        """Запись результата сделки"""
        
        self.state['total_trades'] += 1
        self.state['trades_today'] += 1
        self.state['daily_pnl'] += pnl
        self.state['total_profit'] += pnl
        
        if result == 'WIN':
            self.state['winning_trades'] += 1
            self.state['consecutive_losses'] = 0
        else:
            self.state['consecutive_losses'] += 1
            
            # Проверяем circuit breaker
            if self.state['consecutive_losses'] >= self.parameters['max_consecutive_losses']:
                cooldown_hours = self.parameters['cooldown_hours']
                self.state['circuit_breaker_until'] = (
                    datetime.now() + timedelta(hours=cooldown_hours)
                )
                self.state['consecutive_losses'] = 0
        
        # Сброс состояния позиции
        self.state['position_state'] = PositionState.CLOSED_PROFIT if result == 'WIN' else PositionState.CLOSED_LOSS
        self.state['position_data'] = None
        self.state['partial_filled'] = False