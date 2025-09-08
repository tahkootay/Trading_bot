"""
Агрессивная Breakout Strategy - максимально упрощенная версия для демонстрации

Убираем все ограничения чтобы показать работоспособность концепции.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal


class AggressiveBreakoutStrategy(StrategyBase):
    """
    Максимально агрессивная версия пробойной стратегии
    """
    
    def _initialize(self):
        """Минимальные требования для генерации сигналов"""
        
        self.parameters = {
            # Детекция флэта - очень мягкие условия
            'flat_periods': 6,           # Всего 6 свечей (30 минут)
            'breakout_min_range': 0.1,   # Минимум 0.1 USDT
            'breakout_max_range': 10.0,  # Максимум 10 USDT (почти без ограничений)
            'max_body_pct': 0.8,         # 80% свечей могут быть импульсными
            
            # Объём - почти без фильтра
            'volume_avg_period': 10,     # Короткий период
            'breakout_volume_mult': 1.0, # Без требований к объёму
            
            # Цели и стопы
            'target_move': 0.8,          # Цель всего 0.8 USDT
            'stop_buffer': 0.05,         # Стоп очень близко
            'entry_offset': 0.01,        # Минимальный отступ
            
            # Размер позиции
            'position_size_pct': 0.1     # 10%
        }
        
        self.state = {
            'history': [],
            'flat_data': None
        }

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Упрощенная логика обработки"""
        
        # Обновляем историю
        bar_data = {
            'timestamp': data.timestamp,
            'open': data.open,
            'high': data.high, 
            'low': data.low,
            'close': data.close,
            'volume': data.volume if hasattr(data, 'volume') else 0
        }
        self.state['history'].append(bar_data)
        
        # Ограничиваем историю
        if len(self.state['history']) > 100:
            self.state['history'] = self.state['history'][-100:]
        
        # Нужна минимальная история
        if len(self.state['history']) < self.parameters['flat_periods']:
            return TradeSignal(signal=Signal.HOLD)
        
        # Если нет позиции - ищем вход
        if position.quantity == 0:
            return self._check_entry(data, position)
        else:
            return self._manage_position(data, position)

    def _check_entry(self, data: MarketData, position: Position) -> TradeSignal:
        """Поиск входа с минимальными условиями"""
        
        # Определяем флэт
        flat_data = self._detect_simple_flat()
        if not flat_data:
            return TradeSignal(signal=Signal.HOLD)
        
        self.state['flat_data'] = flat_data
        
        # Проверяем пробой - БЕЗ требований к объёму
        if data.close > flat_data['resistance']:
            return TradeSignal(
                signal=Signal.BUY,
                entry_reason=f"Simple breakout UP - range {flat_data['range']:.3f}"
            )
        elif data.close < flat_data['support']:
            return TradeSignal(
                signal=Signal.SELL, 
                entry_reason=f"Simple breakout DOWN - range {flat_data['range']:.3f}"
            )
        
        return TradeSignal(signal=Signal.HOLD)

    def _detect_simple_flat(self) -> Optional[Dict[str, float]]:
        """Максимально простое определение флэта"""
        
        n_periods = self.parameters['flat_periods']
        recent_bars = self.state['history'][-n_periods:]
        
        # Границы флэта
        highs = [bar['high'] for bar in recent_bars]
        lows = [bar['low'] for bar in recent_bars]
        
        high_flat = max(highs)
        low_flat = min(lows)
        range_flat = high_flat - low_flat
        
        # Проверяем только диапазон
        if not (self.parameters['breakout_min_range'] <= range_flat <= 
                self.parameters['breakout_max_range']):
            return None
        
        return {
            'resistance': high_flat,
            'support': low_flat,
            'range': range_flat
        }

    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Простое управление позицией"""
        
        if not self.state['flat_data']:
            return TradeSignal(signal=Signal.HOLD)
        
        current_price = data.close
        entry_price = position.entry_price
        direction = 'LONG' if position.quantity > 0 else 'SHORT'
        
        # P&L в USDT
        if direction == 'LONG':
            pnl = (current_price - entry_price) * abs(position.quantity)
        else:
            pnl = (entry_price - current_price) * abs(position.quantity)
        
        # Простая цель
        if pnl >= self.parameters['target_move']:
            return TradeSignal(
                signal=Signal.SELL if direction == 'LONG' else Signal.BUY,
                exit_reason=f"Target reached: {pnl:.2f} USDT"
            )
        
        # Простой стоп
        flat_data = self.state['flat_data']
        
        if direction == 'LONG':
            stop_level = flat_data['support'] - self.parameters['stop_buffer']
            if current_price <= stop_level:
                return TradeSignal(
                    signal=Signal.SELL,
                    exit_reason=f"Stop hit: {current_price:.4f}"
                )
        else:
            stop_level = flat_data['resistance'] + self.parameters['stop_buffer']
            if current_price >= stop_level:
                return TradeSignal(
                    signal=Signal.BUY,
                    exit_reason=f"Stop hit: {current_price:.4f}"
                )
        
        return TradeSignal(signal=Signal.HOLD)