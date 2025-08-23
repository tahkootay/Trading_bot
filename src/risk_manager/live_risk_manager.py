#!/usr/bin/env python3
"""
Live риск-менеджер для торгового бота
Согласно спецификации: stop-loss, max exposure, задержки между сделками
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json


class RiskLevel(str, Enum):
    """Уровни риска."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(str, Enum):
    """Действия риск-менеджмента."""
    ALLOW = "allow"
    REDUCE_SIZE = "reduce_size"
    BLOCK = "block"
    FORCE_CLOSE = "force_close"


@dataclass
class RiskLimits:
    """Лимиты риска."""
    max_position_size_usd: float = 1000.0  # Максимальный размер позиции в USD
    max_total_exposure_usd: float = 5000.0  # Максимальная общая экспозиция
    max_daily_loss_usd: float = 500.0  # Максимальная дневная потеря
    max_drawdown_pct: float = 0.05  # Максимальная просадка (5%)
    min_trade_interval_sec: int = 300  # Минимальный интервал между сделками (5 мин)
    max_trades_per_hour: int = 10  # Максимум сделок в час
    max_trades_per_day: int = 50  # Максимум сделок в день
    stop_loss_pct: float = 0.02  # Stop-loss в процентах (2%)
    take_profit_pct: float = 0.04  # Take-profit в процентах (4%)
    max_correlation_exposure: float = 0.7  # Максимальная корреляция между позициями


@dataclass
class TradeRecord:
    """Запись о сделке для отслеживания."""
    timestamp: datetime
    symbol: str
    side: str
    size: float
    price: float
    pnl: float = 0.0
    closed: bool = False


@dataclass
class PositionRisk:
    """Риск позиции."""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    risk_level: RiskLevel
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    time_in_position: timedelta = field(default_factory=lambda: timedelta())
    
    
class LiveRiskManager:
    """
    Live риск-менеджер для контроля рисков в торговле.
    Согласно спецификации с stop-loss, max exposure и задержками.
    """
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None, 
                 initial_balance: float = 10000.0):
        """
        Инициализация риск-менеджера.
        
        Args:
            risk_limits: Лимиты риска
            initial_balance: Начальный баланс
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Состояние
        self.active_positions: Dict[str, PositionRisk] = {}
        self.trade_history: List[TradeRecord] = []
        self.last_trade_time: Optional[datetime] = None
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.hourly_trade_count = 0
        self.last_hour_reset = datetime.now().replace(minute=0, second=0, microsecond=0)
        self.last_day_reset = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Блокировки
        self.trading_blocked = False
        self.block_reason = ""
        self.emergency_stop = False
        
        # Колбэки
        self.on_risk_alert = None
        self.on_stop_loss = None
        self.on_take_profit = None
        self.on_emergency_stop = None
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🛡️ Risk Manager initialized")
        self._log_risk_limits()
    
    def _log_risk_limits(self):
        """Логирование текущих лимитов риска."""
        limits = {
            'max_position_size_usd': self.risk_limits.max_position_size_usd,
            'max_total_exposure_usd': self.risk_limits.max_total_exposure_usd,
            'max_daily_loss_usd': self.risk_limits.max_daily_loss_usd,
            'stop_loss_pct': self.risk_limits.stop_loss_pct * 100,
            'min_trade_interval_sec': self.risk_limits.min_trade_interval_sec
        }
        self.logger.info(f"📊 Risk limits: {limits}")
    
    def validate_new_trade(self, symbol: str, side: str, size: float, 
                          price: float) -> Tuple[RiskAction, str]:
        """
        Валидация новой сделки согласно риск-лимитам.
        
        Args:
            symbol: Символ
            side: Направление ('buy' или 'sell')
            size: Размер позиции
            price: Цена входа
            
        Returns:
            (RiskAction, reason) - действие и причина
        """
        try:
            # Проверка экстренной остановки
            if self.emergency_stop:
                return RiskAction.BLOCK, "Emergency stop activated"
            
            # Проверка общей блокировки
            if self.trading_blocked:
                return RiskAction.BLOCK, f"Trading blocked: {self.block_reason}"
            
            # Обновляем счетчики времени
            self._update_time_counters()
            
            # 1. Проверка временных ограничений
            time_check = self._check_time_limits()
            if time_check[0] != RiskAction.ALLOW:
                return time_check
            
            # 2. Проверка лимитов на количество сделок
            count_check = self._check_trade_count_limits()
            if count_check[0] != RiskAction.ALLOW:
                return count_check
            
            # 3. Проверка размера позиции
            position_value = size * price
            if position_value > self.risk_limits.max_position_size_usd:
                # Предлагаем уменьшить размер
                suggested_size = self.risk_limits.max_position_size_usd / price
                return RiskAction.REDUCE_SIZE, f"Position too large, suggested: {suggested_size:.4f}"
            
            # 4. Проверка общей экспозиции
            current_exposure = self._calculate_total_exposure()
            new_exposure = current_exposure + position_value
            
            if new_exposure > self.risk_limits.max_total_exposure_usd:
                return RiskAction.BLOCK, f"Total exposure limit exceeded: {new_exposure:.2f} > {self.risk_limits.max_total_exposure_usd:.2f}"
            
            # 5. Проверка дневной потери
            if self.daily_pnl < -self.risk_limits.max_daily_loss_usd:
                return RiskAction.BLOCK, f"Daily loss limit exceeded: {self.daily_pnl:.2f}"
            
            # 6. Проверка максимальной просадки
            drawdown_check = self._check_drawdown()
            if drawdown_check[0] != RiskAction.ALLOW:
                return drawdown_check
            
            # 7. Проверка корреляции (если есть другие позиции)
            correlation_check = self._check_correlation_risk(symbol, position_value)
            if correlation_check[0] != RiskAction.ALLOW:
                return correlation_check
            
            return RiskAction.ALLOW, "Trade approved"
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {e}")
            return RiskAction.BLOCK, f"Validation error: {str(e)}"
    
    def _update_time_counters(self):
        """Обновление временных счетчиков."""
        now = datetime.now()
        
        # Сброс почасового счетчика
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        if current_hour > self.last_hour_reset:
            self.hourly_trade_count = 0
            self.last_hour_reset = current_hour
        
        # Сброс дневного счетчика
        current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if current_day > self.last_day_reset:
            self.daily_trade_count = 0
            self.daily_pnl = 0.0
            self.last_day_reset = current_day
            self.logger.info("📅 Daily counters reset")
    
    def _check_time_limits(self) -> Tuple[RiskAction, str]:
        """Проверка временных ограничений между сделками."""
        if self.last_trade_time is None:
            return RiskAction.ALLOW, "First trade"
        
        time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
        
        if time_since_last < self.risk_limits.min_trade_interval_sec:
            remaining = self.risk_limits.min_trade_interval_sec - time_since_last
            return RiskAction.BLOCK, f"Min interval not met, wait {remaining:.0f}s"
        
        return RiskAction.ALLOW, "Time limit OK"
    
    def _check_trade_count_limits(self) -> Tuple[RiskAction, str]:
        """Проверка лимитов на количество сделок."""
        if self.hourly_trade_count >= self.risk_limits.max_trades_per_hour:
            return RiskAction.BLOCK, f"Hourly trade limit exceeded: {self.hourly_trade_count}"
        
        if self.daily_trade_count >= self.risk_limits.max_trades_per_day:
            return RiskAction.BLOCK, f"Daily trade limit exceeded: {self.daily_trade_count}"
        
        return RiskAction.ALLOW, "Count limits OK"
    
    def _calculate_total_exposure(self) -> float:
        """Расчет общей экспозиции."""
        total = 0.0
        for position in self.active_positions.values():
            total += abs(position.size * position.current_price)
        return total
    
    def _check_drawdown(self) -> Tuple[RiskAction, str]:
        """Проверка максимальной просадки."""
        if self.current_balance <= 0:
            return RiskAction.FORCE_CLOSE, "Zero balance"
        
        drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
        
        if drawdown > self.risk_limits.max_drawdown_pct:
            return RiskAction.FORCE_CLOSE, f"Max drawdown exceeded: {drawdown:.2%}"
        
        return RiskAction.ALLOW, "Drawdown OK"
    
    def _check_correlation_risk(self, symbol: str, position_value: float) -> Tuple[RiskAction, str]:
        """Проверка риска корреляции между позициями."""
        if not self.active_positions:
            return RiskAction.ALLOW, "No existing positions"
        
        # Упрощенная проверка: если есть позиции в том же активе
        base_asset = symbol.split('/')[0] if '/' in symbol else symbol[:3]
        
        correlated_exposure = 0.0
        for pos_symbol, position in self.active_positions.items():
            pos_base = pos_symbol.split('/')[0] if '/' in pos_symbol else pos_symbol[:3]
            if pos_base == base_asset:
                correlated_exposure += abs(position.size * position.current_price)
        
        total_correlated = correlated_exposure + position_value
        correlation_ratio = total_correlated / self.current_balance
        
        if correlation_ratio > self.risk_limits.max_correlation_exposure:
            return RiskAction.REDUCE_SIZE, f"High correlation risk: {correlation_ratio:.2%}"
        
        return RiskAction.ALLOW, "Correlation OK"
    
    def add_position(self, symbol: str, side: str, size: float, entry_price: float) -> bool:
        """
        Добавление новой позиции в отслеживание.
        
        Args:
            symbol: Символ
            side: Направление
            size: Размер позиции
            entry_price: Цена входа
            
        Returns:
            bool - успешность добавления
        """
        try:
            # Рассчитываем stop-loss и take-profit
            if side == 'buy':
                stop_loss = entry_price * (1 - self.risk_limits.stop_loss_pct)
                take_profit = entry_price * (1 + self.risk_limits.take_profit_pct)
            else:
                stop_loss = entry_price * (1 + self.risk_limits.stop_loss_pct)
                take_profit = entry_price * (1 - self.risk_limits.take_profit_pct)
            
            position_risk = PositionRisk(
                symbol=symbol,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                unrealized_pnl=0.0,
                risk_level=RiskLevel.LOW,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit
            )
            
            self.active_positions[symbol] = position_risk
            
            # Обновляем счетчики
            self.last_trade_time = datetime.now()
            self.hourly_trade_count += 1
            self.daily_trade_count += 1
            
            # Добавляем запись о сделке
            trade_record = TradeRecord(
                timestamp=datetime.now(),
                symbol=symbol,
                side=side,
                size=size,
                price=entry_price
            )
            self.trade_history.append(trade_record)
            
            self.logger.info(f"📈 Added position: {symbol} {side} {size} @ {entry_price} | SL: {stop_loss:.4f} | TP: {take_profit:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            return False
    
    def update_position_prices(self, price_updates: Dict[str, float]):
        """
        Обновление цен позиций и проверка stop-loss/take-profit.
        
        Args:
            price_updates: Словарь {symbol: current_price}
        """
        positions_to_close = []
        
        for symbol, current_price in price_updates.items():
            if symbol not in self.active_positions:
                continue
            
            position = self.active_positions[symbol]
            position.current_price = current_price
            
            # Обновляем unrealized PnL
            if position.size > 0:  # Long position
                position.unrealized_pnl = position.size * (current_price - position.entry_price)
            else:  # Short position
                position.unrealized_pnl = position.size * (position.entry_price - current_price)
            
            # Проверяем stop-loss
            stop_loss_hit = False
            take_profit_hit = False
            
            if position.size > 0:  # Long position
                if current_price <= position.stop_loss_price:
                    stop_loss_hit = True
                elif current_price >= position.take_profit_price:
                    take_profit_hit = True
            else:  # Short position
                if current_price >= position.stop_loss_price:
                    stop_loss_hit = True
                elif current_price <= position.take_profit_price:
                    take_profit_hit = True
            
            # Обновляем уровень риска
            position.risk_level = self._calculate_position_risk_level(position)
            
            # Отмечаем позиции для закрытия
            if stop_loss_hit:
                positions_to_close.append((symbol, "stop_loss"))
                self.logger.warning(f"🛑 STOP LOSS hit for {symbol} at {current_price}")
            elif take_profit_hit:
                positions_to_close.append((symbol, "take_profit"))
                self.logger.info(f"🎯 TAKE PROFIT hit for {symbol} at {current_price}")
        
        # Вызываем колбэки для закрытия позиций
        for symbol, reason in positions_to_close:
            self._trigger_position_close(symbol, reason)
    
    def _calculate_position_risk_level(self, position: PositionRisk) -> RiskLevel:
        """Расчет уровня риска позиции."""
        if position.entry_price == 0:
            return RiskLevel.LOW
        
        # Процентное изменение цены от входа
        price_change_pct = abs(position.current_price - position.entry_price) / position.entry_price
        
        if price_change_pct < 0.01:  # < 1%
            return RiskLevel.LOW
        elif price_change_pct < 0.03:  # < 3%
            return RiskLevel.MEDIUM
        elif price_change_pct < 0.05:  # < 5%
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _trigger_position_close(self, symbol: str, reason: str):
        """Инициирование закрытия позиции."""
        position = self.active_positions.get(symbol)
        if not position:
            return
        
        if reason == "stop_loss" and self.on_stop_loss:
            self.on_stop_loss(symbol, position)
        elif reason == "take_profit" and self.on_take_profit:
            self.on_take_profit(symbol, position)
    
    def close_position(self, symbol: str, close_price: float, reason: str = "manual"):
        """
        Закрытие позиции.
        
        Args:
            symbol: Символ позиции
            close_price: Цена закрытия
            reason: Причина закрытия
        """
        if symbol not in self.active_positions:
            self.logger.warning(f"Position {symbol} not found for closing")
            return
        
        position = self.active_positions[symbol]
        
        # Рассчитываем финальный PnL
        if position.size > 0:  # Long position
            pnl = position.size * (close_price - position.entry_price)
        else:  # Short position
            pnl = position.size * (position.entry_price - close_price)
        
        # Обновляем баланс и PnL
        self.current_balance += pnl
        self.daily_pnl += pnl
        
        # Обновляем запись в истории
        for trade in reversed(self.trade_history):
            if trade.symbol == symbol and not trade.closed:
                trade.pnl = pnl
                trade.closed = True
                break
        
        # Удаляем позицию
        del self.active_positions[symbol]
        
        self.logger.info(f"📉 Closed position: {symbol} | PnL: {pnl:.2f} | Reason: {reason} | Balance: {self.current_balance:.2f}")
        
        # Проверяем критические условия
        self._check_emergency_conditions()
    
    def _check_emergency_conditions(self):
        """Проверка условий для экстренной остановки."""
        # Проверка максимальной дневной потери
        if self.daily_pnl <= -self.risk_limits.max_daily_loss_usd:
            self._trigger_emergency_stop(f"Daily loss limit hit: {self.daily_pnl:.2f}")
        
        # Проверка максимальной просадки
        drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
        if drawdown >= self.risk_limits.max_drawdown_pct:
            self._trigger_emergency_stop(f"Max drawdown hit: {drawdown:.2%}")
    
    def _trigger_emergency_stop(self, reason: str):
        """Инициирование экстренной остановки."""
        self.emergency_stop = True
        self.trading_blocked = True
        self.block_reason = f"Emergency stop: {reason}"
        
        self.logger.critical(f"🚨 EMERGENCY STOP: {reason}")
        
        if self.on_emergency_stop:
            self.on_emergency_stop(reason, self.active_positions.copy())
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Получение сводки по рискам."""
        total_exposure = self._calculate_total_exposure()
        drawdown = (self.initial_balance - self.current_balance) / self.initial_balance if self.initial_balance > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'trading_status': {
                'blocked': self.trading_blocked,
                'emergency_stop': self.emergency_stop,
                'block_reason': self.block_reason
            },
            'balances': {
                'initial': self.initial_balance,
                'current': self.current_balance,
                'drawdown_pct': drawdown * 100
            },
            'daily_stats': {
                'pnl': self.daily_pnl,
                'trade_count': self.daily_trade_count,
                'hourly_count': self.hourly_trade_count
            },
            'positions': {
                'count': len(self.active_positions),
                'total_exposure': total_exposure,
                'symbols': list(self.active_positions.keys())
            },
            'risk_levels': {
                level.value: len([p for p in self.active_positions.values() if p.risk_level == level])
                for level in RiskLevel
            }
        }
    
    def get_position_details(self) -> List[Dict[str, Any]]:
        """Получение детальной информации по позициям."""
        details = []
        
        for position in self.active_positions.values():
            details.append({
                'symbol': position.symbol,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'risk_level': position.risk_level.value,
                'stop_loss': position.stop_loss_price,
                'take_profit': position.take_profit_price,
                'pnl_pct': (position.unrealized_pnl / abs(position.size * position.entry_price)) * 100 if position.entry_price != 0 else 0
            })
        
        return details
    
    def set_callbacks(self, **callbacks):
        """Установка колбэков для различных событий."""
        if 'on_risk_alert' in callbacks:
            self.on_risk_alert = callbacks['on_risk_alert']
        if 'on_stop_loss' in callbacks:
            self.on_stop_loss = callbacks['on_stop_loss']
        if 'on_take_profit' in callbacks:
            self.on_take_profit = callbacks['on_take_profit']
        if 'on_emergency_stop' in callbacks:
            self.on_emergency_stop = callbacks['on_emergency_stop']
    
    def reset_emergency_stop(self, reason: str = "manual_reset"):
        """Сброс экстренной остановки (осторожно!)."""
        self.emergency_stop = False
        self.trading_blocked = False
        self.block_reason = ""
        self.logger.warning(f"⚠️ Emergency stop reset: {reason}")
    
    def adjust_risk_limits(self, new_limits: RiskLimits):
        """Обновление лимитов риска."""
        old_limits = self.risk_limits
        self.risk_limits = new_limits
        
        self.logger.info(f"🔧 Risk limits updated")
        self._log_risk_limits()
        
        # Пересчитываем stop-loss/take-profit для активных позиций
        for position in self.active_positions.values():
            if position.size > 0:  # Long
                position.stop_loss_price = position.entry_price * (1 - new_limits.stop_loss_pct)
                position.take_profit_price = position.entry_price * (1 + new_limits.take_profit_pct)
            else:  # Short
                position.stop_loss_price = position.entry_price * (1 + new_limits.stop_loss_pct)
                position.take_profit_price = position.entry_price * (1 - new_limits.take_profit_pct)


# Пример использования
def example_usage():
    """Демонстрация использования риск-менеджера."""
    
    # Создаем риск-менеджер с кастомными лимитами
    risk_limits = RiskLimits(
        max_position_size_usd=500.0,
        max_total_exposure_usd=2000.0,
        max_daily_loss_usd=200.0,
        stop_loss_pct=0.015,  # 1.5%
        take_profit_pct=0.03,  # 3%
        min_trade_interval_sec=180  # 3 минуты
    )
    
    risk_manager = LiveRiskManager(risk_limits, initial_balance=5000.0)
    
    # Установка колбэков
    def on_stop_loss(symbol, position):
        print(f"🛑 STOP LOSS: Close {symbol} at {position.current_price}")
    
    def on_take_profit(symbol, position):
        print(f"🎯 TAKE PROFIT: Close {symbol} at {position.current_price}")
    
    def on_emergency_stop(reason, positions):
        print(f"🚨 EMERGENCY: {reason} | Open positions: {len(positions)}")
    
    risk_manager.set_callbacks(
        on_stop_loss=on_stop_loss,
        on_take_profit=on_take_profit,
        on_emergency_stop=on_emergency_stop
    )
    
    # Тестирование валидации сделок
    print("\n=== Testing Trade Validation ===")
    
    # Первая сделка - должна пройти
    action, reason = risk_manager.validate_new_trade("SOLUSDT", "buy", 2.0, 150.0)
    print(f"Trade 1: {action} - {reason}")
    
    if action == RiskAction.ALLOW:
        risk_manager.add_position("SOLUSDT", "buy", 2.0, 150.0)
    
    # Попытка сделки через 1 секунду - должна блокироваться
    action, reason = risk_manager.validate_new_trade("BTCUSDT", "buy", 0.01, 45000.0)
    print(f"Trade 2 (immediate): {action} - {reason}")
    
    # Обновление цены и проверка stop-loss
    print("\n=== Testing Price Updates ===")
    risk_manager.update_position_prices({"SOLUSDT": 147.0})  # -2% от входа
    
    # Статистика
    print("\n=== Risk Summary ===")
    summary = risk_manager.get_risk_summary()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    example_usage()