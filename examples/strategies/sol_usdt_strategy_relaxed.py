from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import RSI, EMA, BollingerBands, SMA
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import numpy as np


class SolUsdtStrategyRelaxed(StrategyBase):
    """
    Relaxed version of SOL/USDT strategy with more achievable conditions.
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        # Basic parameters
        self.parameters = {
            'trading_pair': 'SOL/USDT',
            'position_size_pct': 0.02,  # 2% of total deposit
            'max_positions': 1,
            'max_trades_per_day': 8,
            
            # Technical indicators - RELAXED
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 21,
            'rsi_overbought': 65,  # 75 → 65 (более реалистично)
            'rsi_oversold': 35,    # 25 → 35 (более реалистично) 
            'bb_period': 20,
            'bb_deviation': 2.0,
            'volume_ma_period': 20,
            
            # Entry conditions - RELAXED
            'ema_crossover_threshold': 0.005,  # 0.2% → 0.5% (более широкий диапазон)
            'volume_spike_multiplier': 1.3,    # 1.5 → 1.3 (легче достичь)
            
            # Risk management
            'stop_loss_pct': 0.015,     # 1.5%
            'take_profit_1_pct': 0.02,  # 2%
            'take_profit_2_pct': 0.035, # 3.5%
            'breakeven_profit_pct': 0.015,
            'trailing_stop_offset': 0.005,
            
            # Time and volatility filters
            'max_position_hours': 4,
            'min_volatility_24h': 0.03,
            'emergency_volume_multiplier': 3.0,
            'emergency_price_move_pct': 0.005,
        }
        
        # Initialize indicators using indicators module
        self.indicators = {
            'ema_fast': EMA(period=self.parameters['ema_fast']),
            'ema_slow': EMA(period=self.parameters['ema_slow']),
            'rsi': RSI(period=self.parameters['rsi_period']),
            'bb': BollingerBands(period=self.parameters['bb_period'], std_dev=self.parameters['bb_deviation']),
            'volume_ma': SMA(period=self.parameters['volume_ma_period'])
        }
        
        # Strategy state
        self.state = {
            'trades_today': 0,
            'last_trade_date': None,
            'consecutive_losses': 0,
            'daily_pnl': 0.0,
            'position_entry_time': None,
            'tp1_hit': False,
            'tp2_hit': False,
            'trailing_stop_active': False,
            
            # Current indicator values
            'ema_fast_val': 0.0,
            'ema_slow_val': 0.0,
            'rsi_val': 50.0,
            'bb_values': {'upper': 0.0, 'middle': 0.0, 'lower': 0.0},
            'volume_ma_val': 0.0,
        }

    def update_indicators(self, data: MarketData) -> None:
        """Update technical indicators with new market data."""
        # Update all indicators
        self.state['ema_fast_val'] = self.indicators['ema_fast'].update(data.close)
        self.state['ema_slow_val'] = self.indicators['ema_slow'].update(data.close)
        self.state['rsi_val'] = self.indicators['rsi'].update(data.close)
        self.state['bb_values'] = self.indicators['bb'].update(data.close)
        self.state['volume_ma_val'] = self.indicators['volume_ma'].update(data.volume)
        
        # Handle None values for indicators that aren't ready yet
        if self.state['ema_fast_val'] is None:
            self.state['ema_fast_val'] = data.close
        if self.state['ema_slow_val'] is None:
            self.state['ema_slow_val'] = data.close
        if self.state['rsi_val'] is None:
            self.state['rsi_val'] = 50.0
        if self.state['bb_values'] is None:
            self.state['bb_values'] = {'upper': data.close, 'middle': data.close, 'lower': data.close}
        if self.state['volume_ma_val'] is None:
            self.state['volume_ma_val'] = data.volume

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Process new market data and generate trading signal."""
        # Update indicators first
        self.update_indicators(data)
        
        # Reset daily counters if new day
        self._check_new_day(data.timestamp)
        
        # Check if indicators are ready
        if not self._indicators_ready():
            return TradeSignal(signal=Signal.HOLD)
        
        # Check if we should trade based on filters
        if not self._should_trade(data, position):
            return TradeSignal(signal=Signal.HOLD)
        
        # Position management for open positions
        if position.direction != "NONE":
            return self._manage_position(data, position)
        
        # Entry signal generation
        return self._generate_entry_signal(data, position)
    
    def _indicators_ready(self) -> bool:
        """Check if all indicators are ready and have valid values."""
        return (
            self.indicators['ema_fast'].is_ready() and
            self.indicators['ema_slow'].is_ready() and
            self.indicators['rsi'].is_ready() and
            self.indicators['bb'].is_ready() and
            self.indicators['volume_ma'].is_ready()
        )
    
    def _check_new_day(self, timestamp: datetime):
        """Reset daily counters if new trading day."""
        current_date = timestamp.date()
        if self.state['last_trade_date'] != current_date:
            self.state['trades_today'] = 0
            self.state['daily_pnl'] = 0.0
            self.state['last_trade_date'] = current_date
    
    def _should_trade(self, data: MarketData, position: Position) -> bool:
        """Check if trading is allowed based on various filters."""
        # Max positions check
        if position.direction != "NONE" and self.parameters['max_positions'] <= 1:
            return False
        
        # Daily trade limit
        if self.state['trades_today'] >= self.parameters['max_trades_per_day']:
            return False
        
        # Daily loss limit (5% of deposit)
        if self.state['daily_pnl'] <= -0.05:
            return False
        
        # Consecutive losses pause (2 hours)
        if self.state['consecutive_losses'] >= 3:
            return False
        
        # Volume filter - RELAXED
        if data.volume < self.state['volume_ma_val'] * 0.7:  # 0.8 → 0.7
            return False
        
        return True
    
    def _generate_entry_signal(self, data: MarketData, position: Position) -> TradeSignal:
        """Generate entry signals based on strategy conditions."""
        current_price = data.close
        current_volume = data.volume
        
        # Volume spike condition - RELAXED
        volume_spike = current_volume > self.state['volume_ma_val'] * self.parameters['volume_spike_multiplier']
        
        # EMA crossover proximity - RELAXED
        ema_ratio_for_long = self.state['ema_fast_val'] / self.state['ema_slow_val']
        ema_ratio_for_short = self.state['ema_fast_val'] / self.state['ema_slow_val']
        
        # LONG entry conditions - RELAXED
        long_conditions = [
            current_price <= self.state['bb_values']['lower'] * 1.005,  # Небольшая толерантность +0.5%
            self.state['rsi_val'] <= self.parameters['rsi_oversold'],   # 35 вместо 25
            ema_ratio_for_long > (1 - self.parameters['ema_crossover_threshold']),  # 0.5% вместо 0.2%
            volume_spike or current_volume > self.state['volume_ma_val'] * 1.1  # Альтернатива: просто выше среднего
        ]
        
        # SHORT entry conditions - RELAXED
        short_conditions = [
            current_price >= self.state['bb_values']['upper'] * 0.995,  # Небольшая толерантность -0.5%
            self.state['rsi_val'] >= self.parameters['rsi_overbought'], # 65 вместо 75
            ema_ratio_for_short < (1 + self.parameters['ema_crossover_threshold']),  # 0.5% вместо 0.2%
            volume_spike or current_volume > self.state['volume_ma_val'] * 1.1  # Альтернатива: просто выше среднего
        ]
        
        if all(long_conditions):
            self._record_trade_entry(data.timestamp)
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"RELAXED LONG: BB={self.state['bb_values']['lower']:.2f}, RSI={self.state['rsi_val']:.1f}, EMA_ratio={ema_ratio_for_long:.4f}, Vol_ratio={current_volume/self.state['volume_ma_val']:.2f}",
                stop_loss=current_price * (1 - self.parameters['stop_loss_pct']),
                take_profit=current_price * (1 + self.parameters['take_profit_1_pct'])
            )
        
        if all(short_conditions):
            self._record_trade_entry(data.timestamp)
            return TradeSignal(
                signal=Signal.SELL,
                reason=f"RELAXED SHORT: BB={self.state['bb_values']['upper']:.2f}, RSI={self.state['rsi_val']:.1f}, EMA_ratio={ema_ratio_for_short:.4f}, Vol_ratio={current_volume/self.state['volume_ma_val']:.2f}",
                stop_loss=current_price * (1 + self.parameters['stop_loss_pct']),
                take_profit=current_price * (1 - self.parameters['take_profit_1_pct'])
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Manage open positions with trailing stops and exits."""
        # Упрощённое управление позициями
        return TradeSignal(signal=Signal.HOLD)
    
    def _record_trade_entry(self, timestamp: datetime):
        """Record trade entry for tracking."""
        self.state['trades_today'] += 1
        self.state['position_entry_time'] = timestamp
        self.state['tp1_hit'] = False
        self.state['tp2_hit'] = False
        self.state['trailing_stop_active'] = False
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Calculate position size for trade."""
        return capital * self.parameters['position_size_pct']