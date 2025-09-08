from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import RSI, EMA, BollingerBands, SMA
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import numpy as np


class SolUsdtStrategyDebug(StrategyBase):
    """
    Debug version of SOL/USDT strategy with detailed logging.
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
            
            # Debug counters
            'bar_count': 0,
            'debug_logs': []
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
        self.state['bar_count'] += 1
        
        # Update indicators first
        self.update_indicators(data)
        
        # Reset daily counters if new day
        self._check_new_day(data.timestamp)
        
        # Debug log every 100 bars
        if self.state['bar_count'] % 100 == 0 or self.state['bar_count'] <= 50:
            self._debug_log(data, position)
        
        # Check if indicators are ready
        if not self._indicators_ready():
            if self.state['bar_count'] <= 30:  # Log first 30 bars
                print(f"Bar {self.state['bar_count']}: Indicators not ready yet")
            return TradeSignal(signal=Signal.HOLD)
        
        # Check if we should trade based on filters
        if not self._should_trade(data, position):
            if self.state['bar_count'] % 200 == 0:  # Log every 200 bars
                self._debug_should_trade(data, position)
            return TradeSignal(signal=Signal.HOLD)
        
        # Position management for open positions
        if position.direction != "NONE":
            return self._manage_position(data, position)
        
        # Entry signal generation with detailed logging
        signal = self._generate_entry_signal_debug(data, position)
        
        if signal.signal != Signal.HOLD:
            print(f"🚨 SIGNAL GENERATED: {signal.signal.value} at bar {self.state['bar_count']}")
            print(f"   Price: ${data.close:.2f}")
            print(f"   Reason: {signal.reason}")
        
        return signal
    
    def _debug_log(self, data: MarketData, position: Position):
        """Log debug information."""
        print(f"\n=== DEBUG BAR {self.state['bar_count']} ===")
        print(f"Time: {data.timestamp}")
        print(f"Price: ${data.close:.2f}")
        print(f"Volume: {data.volume:.0f}")
        
        if self._indicators_ready():
            print("INDICATORS:")
            print(f"  EMA Fast(9): {self.state['ema_fast_val']:.2f}")
            print(f"  EMA Slow(21): {self.state['ema_slow_val']:.2f}")
            print(f"  RSI(21): {self.state['rsi_val']:.1f}")
            print(f"  BB Upper: {self.state['bb_values']['upper']:.2f}")
            print(f"  BB Middle: {self.state['bb_values']['middle']:.2f}")
            print(f"  BB Lower: {self.state['bb_values']['lower']:.2f}")
            print(f"  Volume MA: {self.state['volume_ma_val']:.0f}")
            
            # Check entry conditions
            self._debug_entry_conditions(data)
        else:
            ready_status = {
                'ema_fast': self.indicators['ema_fast'].is_ready(),
                'ema_slow': self.indicators['ema_slow'].is_ready(),
                'rsi': self.indicators['rsi'].is_ready(),
                'bb': self.indicators['bb'].is_ready(),
                'volume_ma': self.indicators['volume_ma'].is_ready(),
            }
            print(f"INDICATORS NOT READY: {ready_status}")
    
    def _debug_entry_conditions(self, data: MarketData):
        """Debug entry conditions."""
        current_price = data.close
        current_volume = data.volume
        
        # Volume spike condition
        volume_spike = current_volume > self.state['volume_ma_val'] * self.parameters['volume_spike_multiplier']
        volume_ratio = current_volume / self.state['volume_ma_val'] if self.state['volume_ma_val'] > 0 else 0
        
        # EMA crossover proximity
        ema_ratio = self.state['ema_fast_val'] / self.state['ema_slow_val']
        ema_crossover_threshold = self.parameters['ema_crossover_threshold']
        
        # LONG entry conditions
        long_bb = current_price <= self.state['bb_values']['lower']
        long_rsi = self.state['rsi_val'] <= self.parameters['rsi_oversold']
        long_ema = ema_ratio > (1 - ema_crossover_threshold)
        
        # SHORT entry conditions  
        short_bb = current_price >= self.state['bb_values']['upper']
        short_rsi = self.state['rsi_val'] >= self.parameters['rsi_overbought']
        short_ema = ema_ratio < (1 + ema_crossover_threshold)
        
        print("ENTRY CONDITIONS:")
        print(f"  Volume spike: {volume_spike} (ratio: {volume_ratio:.2f})")
        print(f"  EMA ratio: {ema_ratio:.6f}")
        print()
        print(f"LONG CONDITIONS:")
        print(f"  Price <= BB Lower: {long_bb} ({current_price:.2f} <= {self.state['bb_values']['lower']:.2f})")
        print(f"  RSI <= 25: {long_rsi} ({self.state['rsi_val']:.1f} <= {self.parameters['rsi_oversold']})")
        print(f"  EMA close to cross: {long_ema} ({ema_ratio:.6f} > {1-ema_crossover_threshold:.6f})")
        print(f"  Volume spike: {volume_spike}")
        print(f"  ALL LONG: {all([long_bb, long_rsi, long_ema, volume_spike])}")
        print()
        print(f"SHORT CONDITIONS:")
        print(f"  Price >= BB Upper: {short_bb} ({current_price:.2f} >= {self.state['bb_values']['upper']:.2f})")
        print(f"  RSI >= 75: {short_rsi} ({self.state['rsi_val']:.1f} >= {self.parameters['rsi_overbought']})")
        print(f"  EMA close to cross: {short_ema} ({ema_ratio:.6f} < {1+ema_crossover_threshold:.6f})")
        print(f"  Volume spike: {volume_spike}")
        print(f"  ALL SHORT: {all([short_bb, short_rsi, short_ema, volume_spike])}")
    
    def _debug_should_trade(self, data: MarketData, position: Position):
        """Debug should_trade filters."""
        print(f"\n=== TRADE FILTERS (Bar {self.state['bar_count']}) ===")
        
        # Max positions check
        pos_check = position.direction == "NONE" or self.parameters['max_positions'] > 1
        print(f"Position check: {pos_check} (direction: {position.direction})")
        
        # Daily trade limit
        trade_limit_check = self.state['trades_today'] < self.parameters['max_trades_per_day']
        print(f"Trade limit: {trade_limit_check} ({self.state['trades_today']}/{self.parameters['max_trades_per_day']})")
        
        # Daily loss limit
        loss_limit_check = self.state['daily_pnl'] > -0.05
        print(f"Loss limit: {loss_limit_check} (daily P&L: {self.state['daily_pnl']:.4f})")
        
        # Consecutive losses
        consec_check = self.state['consecutive_losses'] < 3
        print(f"Consecutive losses: {consec_check} ({self.state['consecutive_losses']}/3)")
        
        # Volume filter
        volume_check = data.volume >= self.state['volume_ma_val'] * 0.8
        volume_ratio = data.volume / self.state['volume_ma_val'] if self.state['volume_ma_val'] > 0 else 0
        print(f"Volume filter: {volume_check} (ratio: {volume_ratio:.2f})")
        
        overall = all([pos_check, trade_limit_check, loss_limit_check, consec_check, volume_check])
        print(f"OVERALL TRADE ALLOWED: {overall}")
    
    def _generate_entry_signal_debug(self, data: MarketData, position: Position) -> TradeSignal:
        """Generate entry signals with debug info."""
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
        
        # Log when close to signal
        long_score = sum(long_conditions)
        short_score = sum(short_conditions)
        
        if long_score >= 3 or short_score >= 3:
            print(f"\n🔥 CLOSE TO SIGNAL (Bar {self.state['bar_count']})")
            print(f"LONG score: {long_score}/4, SHORT score: {short_score}/4")
            self._debug_entry_conditions(data)
        
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
    
    # Copy other methods from original strategy
    def _indicators_ready(self) -> bool:
        return (
            self.indicators['ema_fast'].is_ready() and
            self.indicators['ema_slow'].is_ready() and
            self.indicators['rsi'].is_ready() and
            self.indicators['bb'].is_ready() and
            self.indicators['volume_ma'].is_ready()
        )
    
    def _check_new_day(self, timestamp: datetime):
        current_date = timestamp.date()
        if self.state['last_trade_date'] != current_date:
            self.state['trades_today'] = 0
            self.state['daily_pnl'] = 0.0
            self.state['last_trade_date'] = current_date
    
    def _should_trade(self, data: MarketData, position: Position) -> bool:
        if position.direction != "NONE" and self.parameters['max_positions'] <= 1:
            return False
        if self.state['trades_today'] >= self.parameters['max_trades_per_day']:
            return False
        if self.state['daily_pnl'] <= -0.05:
            return False
        if self.state['consecutive_losses'] >= 3:
            return False
        if data.volume < self.state['volume_ma_val'] * 0.8:
            return False
        return True
    
    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        return TradeSignal(signal=Signal.HOLD)
    
    def _record_trade_entry(self, timestamp: datetime):
        self.state['trades_today'] += 1
        self.state['position_entry_time'] = timestamp
        self.state['tp1_hit'] = False
        self.state['tp2_hit'] = False
        self.state['trailing_stop_active'] = False
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        return capital * self.parameters['position_size_pct']