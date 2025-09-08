from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import RSI, EMA, BollingerBands, SMA
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import numpy as np


class SolUsdtStrategy(StrategyBase):
    """
    SOL/USDT trading strategy based on Bollinger Bands, RSI, EMA crossovers and volume analysis.
    
    Entry conditions:
    LONG: Price <= BB Lower, RSI <= 25, EMA9 close to crossing EMA21 upward, high volume
    SHORT: Price >= BB Upper, RSI >= 75, EMA9 close to crossing EMA21 downward, high volume
    
    Risk management: 1.5% stop loss, multiple take profits, trailing stops
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        # Basic parameters
        self.parameters = {
            'trading_pair': 'SOL/USDT',
            'position_size_pct': 0.02,  # 2% of total deposit
            'max_positions': 1,
            'max_trades_per_day': 8,
            
            # Technical indicators
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 21,
            'rsi_overbought': 75,
            'rsi_oversold': 25,
            'bb_period': 20,
            'bb_deviation': 2.0,
            'volume_ma_period': 20,
            
            # Entry conditions
            'ema_crossover_threshold': 0.002,  # 0.2% threshold for EMA proximity
            'volume_spike_multiplier': 1.5,
            
            # Risk management
            'stop_loss_pct': 0.015,     # 1.5%
            'take_profit_1_pct': 0.02,  # 2%
            'take_profit_2_pct': 0.035, # 3.5%
            'breakeven_profit_pct': 0.015,  # Move to breakeven after 1.5% profit
            'trailing_stop_offset': 0.005,  # 0.5% above entry for trailing
            
            # Time and volatility filters
            'max_position_hours': 4,
            'min_volatility_24h': 0.03,  # 3%
            'emergency_volume_multiplier': 3.0,
            'emergency_price_move_pct': 0.005,  # 0.5%
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
        
        # Volume filter
        if data.volume < self.state['volume_ma_val'] * 0.8:
            return False
        
        # Time-based filters would be implemented here
        # For backtesting, we'll skip complex time filters
        
        return True
    
    def _generate_entry_signal(self, data: MarketData, position: Position) -> TradeSignal:
        """Generate entry signals based on strategy conditions."""
        current_price = data.close
        current_volume = data.volume
        
        # Volume spike condition
        volume_spike = current_volume > self.state['volume_ma_val'] * self.parameters['volume_spike_multiplier']
        
        # EMA crossover proximity
        ema_ratio_for_long = self.state['ema_fast_val'] / self.state['ema_slow_val']
        ema_ratio_for_short = self.state['ema_fast_val'] / self.state['ema_slow_val']
        
        # LONG entry conditions
        long_conditions = [
            current_price <= self.state['bb_values']['lower'],
            self.state['rsi_val'] <= self.parameters['rsi_oversold'],
            ema_ratio_for_long > (1 - self.parameters['ema_crossover_threshold']),
            volume_spike
        ]
        
        # SHORT entry conditions  
        short_conditions = [
            current_price >= self.state['bb_values']['upper'],
            self.state['rsi_val'] >= self.parameters['rsi_overbought'],
            ema_ratio_for_short < (1 + self.parameters['ema_crossover_threshold']),
            volume_spike
        ]
        
        if all(long_conditions):
            self._record_trade_entry(data.timestamp)
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"LONG: BB={self.state['bb_values']['lower']:.2f}, RSI={self.state['rsi_val']:.1f}, EMA_ratio={ema_ratio_for_long:.4f}",
                stop_loss=current_price * (1 - self.parameters['stop_loss_pct']),
                take_profit=current_price * (1 + self.parameters['take_profit_1_pct'])
            )
        
        if all(short_conditions):
            self._record_trade_entry(data.timestamp)
            return TradeSignal(
                signal=Signal.SELL,
                reason=f"SHORT: BB={self.state['bb_values']['upper']:.2f}, RSI={self.state['rsi_val']:.1f}, EMA_ratio={ema_ratio_for_short:.4f}",
                stop_loss=current_price * (1 + self.parameters['stop_loss_pct']),
                take_profit=current_price * (1 - self.parameters['take_profit_1_pct'])
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Manage open positions with trailing stops and exits."""
        current_price = data.close
        
        # Check for emergency exit
        if self._should_emergency_exit(data, position):
            self.state['position_entry_time'] = None
            return TradeSignal(
                signal=Signal.CLOSE_LONG if position.direction == "LONG" else Signal.CLOSE_SHORT,
                reason="Emergency exit - high volume adverse move"
            )
        
        # Time-based exit (max 4 hours)
        if self._should_time_exit(data.timestamp):
            self.state['position_entry_time'] = None
            return TradeSignal(
                signal=Signal.CLOSE_LONG if position.direction == "LONG" else Signal.CLOSE_SHORT,
                reason="Time exit - max position duration reached"
            )
        
        # Trailing stop logic
        profit_pct = (current_price - position.entry_price) / position.entry_price
        if position.direction == "SHORT":  # Short position
            profit_pct = -profit_pct
        
        # Activate trailing stop after breakeven profit
        if profit_pct >= self.parameters['breakeven_profit_pct'] and not self.state['trailing_stop_active']:
            self.state['trailing_stop_active'] = True
            # Move stop to breakeven + small profit
            new_stop = position.entry_price * (1 + self.parameters['trailing_stop_offset'])
            if position.direction == "SHORT":  # Short position
                new_stop = position.entry_price * (1 - self.parameters['trailing_stop_offset'])
            
            return TradeSignal(
                signal=Signal.HOLD,
                reason=f"Trailing stop activated at breakeven+{self.parameters['trailing_stop_offset']:.1%}",
                stop_loss=new_stop
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _should_emergency_exit(self, data: MarketData, position: Position) -> bool:
        """Check if emergency exit conditions are met."""
        current_volume = data.volume
        volume_spike = current_volume > self.state['volume_ma_val'] * self.parameters['emergency_volume_multiplier']
        
        if not volume_spike:
            return False
        
        # Check for adverse P&L movement
        emergency_loss_threshold = self.parameters['emergency_price_move_pct'] * position.entry_price
        
        if position.direction == "LONG" and position.unrealized_pnl < -emergency_loss_threshold:
            return True
        elif position.direction == "SHORT" and position.unrealized_pnl < -emergency_loss_threshold:
            return True
        
        return False
    
    def _should_time_exit(self, timestamp: datetime) -> bool:
        """Check if position should be closed due to time limit."""
        if self.state['position_entry_time'] is None:
            return False
        
        time_in_position = timestamp - self.state['position_entry_time']
        max_time = timedelta(hours=self.parameters['max_position_hours'])
        
        return time_in_position >= max_time
    
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
    
    def get_stop_loss(self, data: MarketData, entry_price: float, direction: str) -> Optional[float]:
        """Calculate stop loss price."""
        if direction == "LONG":
            return entry_price * (1 - self.parameters['stop_loss_pct'])
        elif direction == "SHORT":
            return entry_price * (1 + self.parameters['stop_loss_pct'])
        return None
    
    def get_take_profit(self, data: MarketData, entry_price: float, direction: str) -> Optional[float]:
        """Calculate take profit price."""
        if direction == "LONG":
            return entry_price * (1 + self.parameters['take_profit_1_pct'])
        elif direction == "SHORT":
            return entry_price * (1 - self.parameters['take_profit_1_pct'])
        return None