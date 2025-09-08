from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import EMA
from typing import Optional


class UltraSimpleStrategy(StrategyBase):
    """
    Ultra simple strategy that will definitely generate trades.
    Just trades on EMA crossovers without any additional filters.
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            'ema_fast': 5,     # Very fast
            'ema_slow': 10,    # Very slow  
            'position_size_pct': 0.01,
        }
        
        # Initialize indicators
        self.indicators = {
            'ema_fast': EMA(period=self.parameters['ema_fast']),
            'ema_slow': EMA(period=self.parameters['ema_slow']),
        }
        
        # Strategy state
        self.state = {
            'ema_fast_val': 0.0,
            'ema_slow_val': 0.0,
            'prev_ema_fast': 0.0,
            'prev_ema_slow': 0.0,
            'trades_count': 0,
            'last_signal_bar': -10,  # Track to avoid multiple signals
        }

    def update_indicators(self, data: MarketData) -> None:
        """Update technical indicators with new market data."""
        # Store previous values
        self.state['prev_ema_fast'] = self.state['ema_fast_val'] or data.close
        self.state['prev_ema_slow'] = self.state['ema_slow_val'] or data.close
        
        # Update indicators
        new_fast = self.indicators['ema_fast'].update(data.close)
        new_slow = self.indicators['ema_slow'].update(data.close)
        
        # Handle None values
        self.state['ema_fast_val'] = new_fast if new_fast is not None else data.close
        self.state['ema_slow_val'] = new_slow if new_slow is not None else data.close

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Process new market data and generate trading signal."""
        self.state['bar_count'] = getattr(self.state, 'bar_count', 0) + 1
        
        # Update indicators first
        self.update_indicators(data)
        
        # Check if indicators are ready
        if not self._indicators_ready():
            if self.state['bar_count'] <= 15:
                print(f"Bar {self.state['bar_count']}: Waiting for indicators to be ready")
            return TradeSignal(signal=Signal.HOLD)
        
        # Close existing position on opposite crossover
        if position.direction != "NONE":
            return self._manage_position(data, position)
        
        # Only allow one signal every 5 bars to avoid noise
        if self.state['bar_count'] - self.state['last_signal_bar'] < 5:
            return TradeSignal(signal=Signal.HOLD)
        
        # Entry signal generation - pure MA crossover
        return self._generate_entry_signal(data)
    
    def _indicators_ready(self) -> bool:
        """Check if all indicators are ready."""
        return (
            self.indicators['ema_fast'].is_ready() and
            self.indicators['ema_slow'].is_ready()
        )
    
    def _generate_entry_signal(self, data: MarketData) -> TradeSignal:
        """Generate entry signals based on pure MA crossover."""
        
        # Golden Cross: Fast MA crosses above Slow MA
        golden_cross = (
            self.state['ema_fast_val'] > self.state['ema_slow_val'] and
            self.state['prev_ema_fast'] <= self.state['prev_ema_slow']
        )
        
        # Death Cross: Fast MA crosses below Slow MA
        death_cross = (
            self.state['ema_fast_val'] < self.state['ema_slow_val'] and
            self.state['prev_ema_fast'] >= self.state['prev_ema_slow']
        )
        
        if golden_cross:
            self.state['trades_count'] += 1
            self.state['last_signal_bar'] = self.state['bar_count']
            print(f"🟢 LONG signal at bar {self.state['bar_count']}: Price=${data.close:.2f}")
            print(f"   EMA Fast: {self.state['ema_fast_val']:.2f} > EMA Slow: {self.state['ema_slow_val']:.2f}")
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"Golden Cross: EMA{self.parameters['ema_fast']}={self.state['ema_fast_val']:.2f} > EMA{self.parameters['ema_slow']}={self.state['ema_slow_val']:.2f}",
                stop_loss=data.close * 0.98,    # 2% stop loss
                take_profit=data.close * 1.03   # 3% take profit
            )
        
        if death_cross:
            self.state['trades_count'] += 1
            self.state['last_signal_bar'] = self.state['bar_count']
            print(f"🔴 SHORT signal at bar {self.state['bar_count']}: Price=${data.close:.2f}")
            print(f"   EMA Fast: {self.state['ema_fast_val']:.2f} < EMA Slow: {self.state['ema_slow_val']:.2f}")
            
            return TradeSignal(
                signal=Signal.SELL,
                reason=f"Death Cross: EMA{self.parameters['ema_fast']}={self.state['ema_fast_val']:.2f} < EMA{self.parameters['ema_slow']}={self.state['ema_slow_val']:.2f}",
                stop_loss=data.close * 1.02,    # 2% stop loss  
                take_profit=data.close * 0.97   # 3% take profit
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Simple position management - close on opposite crossover."""
        
        # Close LONG on death cross
        if (position.direction == "LONG" and 
            self.state['ema_fast_val'] < self.state['ema_slow_val'] and
            self.state['prev_ema_fast'] >= self.state['prev_ema_slow']):
            
            print(f"🔴 Closing LONG at bar {self.state['bar_count']}: Death Cross")
            return TradeSignal(
                signal=Signal.CLOSE_LONG,
                reason="Exit LONG: Death Cross detected"
            )
        
        # Close SHORT on golden cross
        if (position.direction == "SHORT" and 
            self.state['ema_fast_val'] > self.state['ema_slow_val'] and
            self.state['prev_ema_fast'] <= self.state['prev_ema_slow']):
            
            print(f"🟢 Closing SHORT at bar {self.state['bar_count']}: Golden Cross")
            return TradeSignal(
                signal=Signal.CLOSE_SHORT,
                reason="Exit SHORT: Golden Cross detected"
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Calculate position size for trade."""
        return capital * self.parameters['position_size_pct']