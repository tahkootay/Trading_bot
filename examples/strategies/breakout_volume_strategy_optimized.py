"""
Optimized Breakout Volume Strategy

Откорректированная версия стратегии с более реалистичными параметрами
на основе анализа рыночных условий.

Основные изменения:
- Снижены требования к объёму (1.3 -> 1.1)
- Расширен диапазон флэтов (0.5-1.5 -> 0.3-2.5 USDT)
- Улучшена логика трендовых фильтров
- Добавлена адаптивность к волатильности
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal


class OptimizedBreakoutVolumeStrategy(StrategyBase):
    """
    Оптимизированная стратегия пробоя флэта с подтверждением объёмом
    """
    
    def _initialize(self):
        """Инициализация оптимизированных параметров стратегии"""
        
        # Торговые параметры (откорректированы по результатам анализа)
        self.parameters = {
            # Таймфреймы
            'timeframe_entry': '5m',
            'timeframe_context': '15m', 
            'timeframe_day': '1h',
            
            # Индикаторы
            'ema_fast': 20,         # EMA быстрая для 5m
            'ema_slow': 50,         # EMA медленная для 5m
            'vwap_lookback': 'session',  # VWAP за сессию
            'volume_avg_period': 20,     # Период среднего объёма
            
            # Параметры пробоя (ОПТИМИЗИРОВАНЫ)
            'breakout_min_range': 0.3,   # Уменьшено с 0.5 - больше возможностей
            'breakout_max_range': 2.5,   # Увеличено с 1.5 - захватываем крупные флэты
            'breakout_volume_mult': 1.1, # Снижено с 1.3 - более реалистично
            'flat_periods': 12,          # Количество свечей для определения флэта
            'max_body_pct': 0.4,         # Увеличено с 0.3 - больше гибкости
            
            # Управление позицией
            'target_move': 1.5,          # Снижено с 2.0 - более реалистично
            'partial_target': 1.0,       # Снижено с 1.5
            'stop_buffer': 0.15,         # Снижено с 0.2
            'entry_offset': 0.03,        # Снижено с 0.05
            'breakeven_buffer': 0.08,    # Снижено с 0.1
            'partial_close_pct': 0.5,    # Доля позиции для частичного закрытия
            
            # Риск-менеджмент
            'max_risk_per_trade_pct': 0.5,  # % риска от депозита
            'max_trades_per_day': 8,        # Лимит сделок в день
            'max_leverage': 5,              # Максимальное плечо
            'min_liquidity_hour': 50000,    # Снижено с 100k - более доступно
            'circuit_breaker_losses': 3,    # Количество убытков подряд для остановки
            'cooldown_hours': 12,           # Снижено с 24 часов
            
            # Размер позиции  
            'position_size_pct': 0.1,      # Базовый размер позиции от депозита
            
            # Адаптивность к волатильности
            'volatility_lookback': 24,     # Часы для расчёта волатильности
            'high_vol_multiplier': 1.2,    # Множитель для высокой волатильности
            'low_vol_multiplier': 0.8      # Множитель для низкой волатильности
        }
        
        # Состояние стратегии
        self.state = {
            'history': [],              # История цен и индикаторов
            'current_position': None,   # Текущая позиция
            'trades_today': 0,          # Количество сделок за день
            'consecutive_losses': 0,    # Подряд идущие убытки
            'last_trade_day': None,     # Последний день торговли
            'circuit_breaker_until': None,  # Время окончания circuit breaker
            'position_state': 'IDLE',   # IDLE, ENTRY_PENDING, ACTIVE, PARTIAL_TAKEN
            'flat_data': None,          # Данные о текущем флэте
            'entry_pending': None,      # Данные об отложенном входе
            'partial_filled': False,    # Флаг частичного закрытия
            'volatility': None          # Текущая волатильность
        }

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """
        Обработка новой свечи с оптимизированной логикой
        """
        
        # Обновляем историю данных
        self._update_history(data)
        
        # Проверяем достаточность данных
        min_required = max(
            self.parameters['ema_slow'], 
            self.parameters['volume_avg_period'],
            self.parameters['flat_periods'],
            self.parameters['volatility_lookback']
        )
        
        if len(self.state['history']) < min_required:
            return TradeSignal(signal=Signal.HOLD)
        
        # Сброс дневного счётчика сделок
        self._reset_daily_trades(data.timestamp)
        
        # Проверка circuit breaker
        if self._is_circuit_breaker_active(data.timestamp):
            return TradeSignal(signal=Signal.HOLD)
        
        # Проверка лимита сделок за день
        if self.state['trades_today'] >= self.parameters['max_trades_per_day']:
            return TradeSignal(signal=Signal.HOLD)
        
        # Обновляем индикаторы и волатильность
        self._update_indicators()
        self._update_volatility()
        
        # Логика в зависимости от состояния позиции
        if position.quantity == 0:  # Нет позиции
            return self._check_entry_conditions(data, position)
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
        
        # Ограничиваем размер истории
        max_history = 576  # 48 часов * 12 свечей
        if len(self.state['history']) > max_history:
            self.state['history'] = self.state['history'][-max_history:]

    def _reset_daily_trades(self, timestamp: datetime):
        """Сброс дневного счётчика сделок"""
        
        current_day = timestamp.date()
        if self.state['last_trade_day'] != current_day:
            self.state['trades_today'] = 0
            self.state['last_trade_day'] = current_day

    def _is_circuit_breaker_active(self, timestamp: datetime) -> bool:
        """Проверка активности circuit breaker"""
        
        if self.state['circuit_breaker_until']:
            return timestamp < self.state['circuit_breaker_until']
        return False

    def _update_indicators(self):
        """Обновление технических индикаторов"""
        
        df = pd.DataFrame(self.state['history'])
        
        # EMA
        df['ema_fast'] = df['close'].ewm(span=self.parameters['ema_fast']).mean()
        df['ema_slow'] = df['close'].ewm(span=self.parameters['ema_slow']).mean()
        
        # Средний объём
        df['avg_volume'] = df['volume'].rolling(
            window=self.parameters['volume_avg_period']
        ).mean()
        
        # VWAP (упрощённая версия - за последние 24 часа)
        vwap_period = min(288, len(df))  # 24 часа или доступные данные
        if vwap_period > 0:
            recent_data = df.tail(vwap_period)
            vwap = (recent_data['close'] * recent_data['volume']).sum() / recent_data['volume'].sum()
            df.loc[df.index[-1], 'vwap'] = vwap
        
        # Обновляем историю с индикаторами
        self.state['history'] = df.to_dict('records')

    def _update_volatility(self):
        """Обновление адаптивной волатильности"""
        
        lookback = self.parameters['volatility_lookback']
        if len(self.state['history']) < lookback:
            self.state['volatility'] = None
            return
        
        # Расчёт волатильности на основе истинного диапазона
        recent_bars = self.state['history'][-lookback:]
        ranges = []
        
        for i in range(1, len(recent_bars)):
            prev_close = recent_bars[i-1]['close']
            curr_high = recent_bars[i]['high']
            curr_low = recent_bars[i]['low']
            
            # Истинный диапазон
            true_range = max(
                curr_high - curr_low,
                abs(curr_high - prev_close),
                abs(curr_low - prev_close)
            )
            ranges.append(true_range)
        
        if ranges:
            self.state['volatility'] = np.mean(ranges)

    def _get_adaptive_parameters(self) -> Dict[str, float]:
        """Получение адаптивных параметров на основе волатильности"""
        
        if self.state['volatility'] is None:
            return {
                'volume_mult': self.parameters['breakout_volume_mult'],
                'min_range': self.parameters['breakout_min_range'],
                'max_range': self.parameters['breakout_max_range']
            }
        
        # Средняя волатильность для SOL/USDT ~0.5-1.0
        high_vol_threshold = 1.0
        low_vol_threshold = 0.3
        
        if self.state['volatility'] > high_vol_threshold:
            # Высокая волатильность - ужесточаем условия
            return {
                'volume_mult': self.parameters['breakout_volume_mult'] * self.parameters['high_vol_multiplier'],
                'min_range': self.parameters['breakout_min_range'] * 1.2,
                'max_range': self.parameters['breakout_max_range'] * 1.5
            }
        elif self.state['volatility'] < low_vol_threshold:
            # Низкая волатильность - смягчаем условия
            return {
                'volume_mult': self.parameters['breakout_volume_mult'] * self.parameters['low_vol_multiplier'],
                'min_range': self.parameters['breakout_min_range'] * 0.7,
                'max_range': self.parameters['breakout_max_range'] * 0.8
            }
        else:
            # Нормальная волатильность
            return {
                'volume_mult': self.parameters['breakout_volume_mult'],
                'min_range': self.parameters['breakout_min_range'],
                'max_range': self.parameters['breakout_max_range']
            }

    def _check_entry_conditions(self, data: MarketData, position: Position) -> TradeSignal:
        """Проверка условий для входа в позицию с адаптивными параметрами"""
        
        # Получаем адаптивные параметры
        adaptive_params = self._get_adaptive_parameters()
        
        # Определяем флэт с адаптивными параметрами
        flat_data = self._detect_flat(adaptive_params)
        if not flat_data:
            return TradeSignal(signal=Signal.HOLD)
        
        self.state['flat_data'] = flat_data
        
        # Проверяем пробой с адаптивными параметрами
        breakout_direction = self._detect_breakout(data, flat_data, adaptive_params)
        if not breakout_direction:
            return TradeSignal(signal=Signal.HOLD)
        
        # Проверяем трендовые фильтры (более мягкие)
        if not self._pass_trend_filter(breakout_direction):
            return TradeSignal(signal=Signal.HOLD)
        
        # Рассчитываем параметры входа
        entry_params = self._calculate_entry_parameters(
            data, flat_data, breakout_direction
        )
        
        if not entry_params:
            return TradeSignal(signal=Signal.HOLD)
        
        # Генерируем сигнал входа
        signal_type = Signal.BUY if breakout_direction == 'LONG' else Signal.SELL
        
        # Обновляем состояние
        self.state['position_state'] = 'ACTIVE'
        self.state['trades_today'] += 1
        self.state['partial_filled'] = False
        
        return TradeSignal(
            signal=signal_type,
            entry_reason=f"Optimized Breakout {breakout_direction.lower()} - "
                        f"Range: {flat_data['range']:.3f}, "
                        f"Volume: {data.volume:.0f} "
                        f"(avg: {self.state['history'][-1].get('avg_volume', 0):.0f}), "
                        f"Volatility: {self.state['volatility']:.3f if self.state['volatility'] else 'N/A'}"
        )

    def _detect_flat(self, adaptive_params: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Определение зоны флэта с адаптивными параметрами"""
        
        n_periods = self.parameters['flat_periods']
        if len(self.state['history']) < n_periods:
            return None
        
        # Берём последние N свечей
        recent_bars = self.state['history'][-n_periods:]
        
        # Определяем границы флэта
        highs = [bar['high'] for bar in recent_bars]
        lows = [bar['low'] for bar in recent_bars]
        
        high_flat = max(highs)
        low_flat = min(lows)
        range_flat = high_flat - low_flat
        
        # Проверяем диапазон флэта с адаптивными параметрами
        if not (adaptive_params['min_range'] <= range_flat <= adaptive_params['max_range']):
            return None
        
        # Проверяем, что внутри флэта не более заданного % импульсных свечей
        big_body_count = 0
        max_body_size = range_flat * 0.5
        
        for bar in recent_bars:
            body_size = abs(bar['close'] - bar['open'])
            if body_size > max_body_size:
                big_body_count += 1
        
        big_body_pct = big_body_count / len(recent_bars)
        if big_body_pct > self.parameters['max_body_pct']:
            return None
        
        return {
            'resistance': high_flat,
            'support': low_flat,
            'range': range_flat,
            'periods': n_periods
        }

    def _detect_breakout(self, data: MarketData, flat_data: Dict[str, float], 
                        adaptive_params: Dict[str, float]) -> Optional[str]:
        """Определение пробоя флэта с адаптивными параметрами"""
        
        current_bar = self.state['history'][-1]
        avg_volume = current_bar.get('avg_volume', 0)
        
        if avg_volume == 0:
            return None
        
        # Минимальный объём для подтверждения пробоя (адаптивный)
        min_volume = avg_volume * adaptive_params['volume_mult']
        
        if data.volume < min_volume:
            return None
        
        # Проверяем пробой вверх
        if data.close > flat_data['resistance']:
            return 'LONG'
        
        # Проверяем пробой вниз
        if data.close < flat_data['support']:
            return 'SHORT'
        
        return None

    def _pass_trend_filter(self, direction: str) -> bool:
        """Более мягкая проверка соответствия направления тренду"""
        
        current_bar = self.state['history'][-1]
        
        # EMA фильтр
        ema_fast = current_bar.get('ema_fast', 0)
        ema_slow = current_bar.get('ema_slow', 0)
        
        if ema_fast == 0 or ema_slow == 0:
            return True  # Разрешаем, если нет данных
        
        # VWAP фильтр  
        vwap = current_bar.get('vwap', 0)
        current_price = current_bar['close']
        
        if direction == 'LONG':
            # Для лонгов: хотя бы одно условие из трёх
            ema_bullish = ema_fast >= ema_slow * 0.998  # Практически равно или больше
            vwap_bullish = current_price >= vwap * 0.999 if vwap > 0 else True
            price_momentum = current_price > current_bar['open']  # Растущая свеча
            
            return ema_bullish or vwap_bullish or price_momentum
            
        else:  # SHORT
            # Для шортов: хотя бы одно условие из трёх
            ema_bearish = ema_fast <= ema_slow * 1.002  # Практически равно или меньше
            vwap_bearish = current_price <= vwap * 1.001 if vwap > 0 else True
            price_momentum = current_price < current_bar['open']  # Падающая свеча
            
            return ema_bearish or vwap_bearish or price_momentum

    def _calculate_entry_parameters(self, data: MarketData, flat_data: Dict[str, float], 
                                  direction: str) -> Optional[Dict[str, Any]]:
        """Расчёт оптимизированных параметров входа в позицию"""
        
        # Цена входа с отступом от уровня пробоя
        if direction == 'LONG':
            entry_price = flat_data['resistance'] + self.parameters['entry_offset']
            stop_price = flat_data['support'] - self.parameters['stop_buffer']
        else:
            entry_price = flat_data['support'] - self.parameters['entry_offset']
            stop_price = flat_data['resistance'] + self.parameters['stop_buffer']
        
        # Расстояние до стопа
        stop_distance = abs(entry_price - stop_price)
        
        if stop_distance <= 0:
            return None
        
        return {
            'entry_price': entry_price,
            'stop_price': stop_price,
            'stop_distance': stop_distance,
            'direction': direction
        }

    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Оптимизированное управление открытой позицией"""
        
        if not self.state['flat_data']:
            return TradeSignal(signal=Signal.HOLD)
        
        current_price = data.close
        entry_price = position.entry_price
        direction = 'LONG' if position.quantity > 0 else 'SHORT'
        
        # Рассчитываем текущий P&L
        if direction == 'LONG':
            unrealized_pnl = (current_price - entry_price) * abs(position.quantity)
        else:
            unrealized_pnl = (entry_price - current_price) * abs(position.quantity)
        
        # Проверяем условия для частичного закрытия
        if not self.state['partial_filled']:
            partial_target_usd = self.parameters['partial_target']
            
            if unrealized_pnl >= partial_target_usd:
                self.state['partial_filled'] = True
                return TradeSignal(
                    signal=Signal.SELL if direction == 'LONG' else Signal.BUY,
                    size_pct=self.parameters['partial_close_pct'],
                    exit_reason=f"Partial profit target reached: {unrealized_pnl:.2f} USDT"
                )
        
        # Проверяем условия для полного закрытия (целевая прибыль)
        target_move_usd = self.parameters['target_move']
        
        if unrealized_pnl >= target_move_usd:
            self._record_trade_result('WIN')
            return TradeSignal(
                signal=Signal.SELL if direction == 'LONG' else Signal.BUY,
                exit_reason=f"Target profit reached: {unrealized_pnl:.2f} USDT"
            )
        
        # Проверяем стоп-лосс
        flat_data = self.state['flat_data']
        
        if direction == 'LONG':
            stop_level = flat_data['support'] - self.parameters['stop_buffer']
            # После частичного закрытия переносим стоп в безубыток
            if self.state['partial_filled']:
                stop_level = max(stop_level, entry_price + self.parameters['breakeven_buffer'])
            
            if current_price <= stop_level:
                self._record_trade_result('LOSS')
                return TradeSignal(
                    signal=Signal.SELL,
                    exit_reason=f"Stop loss hit at {current_price:.4f}"
                )
                
        else:  # SHORT
            stop_level = flat_data['resistance'] + self.parameters['stop_buffer']
            # После частичного закрытия переносим стоп в безубыток  
            if self.state['partial_filled']:
                stop_level = min(stop_level, entry_price - self.parameters['breakeven_buffer'])
            
            if current_price >= stop_level:
                self._record_trade_result('LOSS')
                return TradeSignal(
                    signal=Signal.BUY,
                    exit_reason=f"Stop loss hit at {current_price:.4f}"
                )
        
        return TradeSignal(signal=Signal.HOLD)

    def _record_trade_result(self, result: str):
        """Запись результата сделки"""
        
        if result == 'LOSS':
            self.state['consecutive_losses'] += 1
            
            # Проверяем circuit breaker
            if (self.state['consecutive_losses'] >= 
                self.parameters['circuit_breaker_losses']):
                
                # Активируем circuit breaker
                cooldown_hours = self.parameters['cooldown_hours']
                self.state['circuit_breaker_until'] = (
                    datetime.now() + timedelta(hours=cooldown_hours)
                )
                self.state['consecutive_losses'] = 0
                
        else:  # WIN
            self.state['consecutive_losses'] = 0
        
        # Сброс состояния позиции
        self.state['position_state'] = 'IDLE'
        self.state['flat_data'] = None
        self.state['partial_filled'] = False