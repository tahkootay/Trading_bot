# Strategy File Structure for Backtesting

## Overview
This document describes the required structure and format for strategy files used with the backtesting module (`modules/backtester`).

## Base Strategy Class

All strategies must inherit from `StrategyBase` and implement the required methods:

```python
from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal

class MyStrategy(StrategyBase):
    def _initialize(self):
        """Initialize strategy parameters and state"""
        pass
    
    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Process new market data and return trading signal"""
        pass
```

## Required Components

### 1. Class Definition
```python
class MyStrategy(StrategyBase):
    """
    Brief strategy description.
    
    Args:
        Any constructor parameters if needed
    """
```

### 2. _initialize() Method
```python
def _initialize(self):
    """Initialize strategy parameters and internal state."""
    # Strategy parameters (configurable values)
    self.parameters = {
        'period': 20,                    # Lookback period for indicators
        'position_size_pct': 0.1,        # Position size as % of portfolio
        'stop_loss_pct': 0.02,           # Stop loss percentage
        'take_profit_pct': 0.04,         # Take profit percentage
        # Add other parameters as needed
    }
    
    # Strategy state (internal variables)
    self.state = {
        'history': [],                   # Price history buffer
        'indicators': {},                # Calculated indicators
        'last_signal': Signal.HOLD,      # Previous signal
        'entry_price': 0.0,              # Entry price tracking
        # Add other state variables as needed
    }
```

### 3. on_bar() Method
```python
def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
    """
    Process new market data bar and generate trading signal.
    
    Args:
        data (MarketData): Current market data with OHLCV
        position (Position): Current position information
        
    Returns:
        TradeSignal: Trading signal with action and optional parameters
    """
    # Update internal state
    self._update_state(data)
    
    # Calculate indicators
    indicators = self._calculate_indicators()
    
    # Generate signal based on strategy logic
    signal = self._generate_signal(data, position, indicators)
    
    return TradeSignal(
        signal=signal,
        quantity=self._calculate_quantity(data),
        stop_loss=self._calculate_stop_loss(data),
        take_profit=self._calculate_take_profit(data)
    )
```

## Data Types Reference

### MarketData
```python
@dataclass
class MarketData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
```

### Position
```python
@dataclass
class Position:
    size: float          # Current position size (positive=long, negative=short, 0=flat)
    entry_price: float   # Average entry price
    unrealized_pnl: float # Current unrealized P&L
    realized_pnl: float  # Total realized P&L
```

### TradeSignal
```python
@dataclass
class TradeSignal:
    signal: Signal                    # BUY, SELL, or HOLD
    quantity: Optional[float] = None  # Trade quantity (optional)
    stop_loss: Optional[float] = None # Stop loss price (optional)
    take_profit: Optional[float] = None # Take profit price (optional)
```

### Signal Enum
```python
class Signal(Enum):
    BUY = "BUY"     # Open long position or close short
    SELL = "SELL"   # Open short position or close long  
    HOLD = "HOLD"   # No action
```

## Complete Strategy Example

```python
from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from typing import List
import numpy as np

class MovingAverageCrossover(StrategyBase):
    """
    Simple moving average crossover strategy.
    
    Generates BUY signal when fast MA crosses above slow MA.
    Generates SELL signal when fast MA crosses below slow MA.
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            'fast_period': 10,
            'slow_period': 20,
            'position_size_pct': 0.1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04
        }
        
        self.state = {
            'price_history': [],
            'fast_ma': 0.0,
            'slow_ma': 0.0,
            'prev_fast_ma': 0.0,
            'prev_slow_ma': 0.0
        }
    
    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Process new bar and generate trading signal."""
        # Update price history
        self.state['price_history'].append(data.close)
        
        # Keep only required history
        max_period = max(self.parameters['fast_period'], self.parameters['slow_period'])
        if len(self.state['price_history']) > max_period:
            self.state['price_history'] = self.state['price_history'][-max_period:]
        
        # Calculate moving averages
        if len(self.state['price_history']) >= self.parameters['slow_period']:
            self.state['prev_fast_ma'] = self.state['fast_ma']
            self.state['prev_slow_ma'] = self.state['slow_ma']
            
            fast_prices = self.state['price_history'][-self.parameters['fast_period']:]
            slow_prices = self.state['price_history'][-self.parameters['slow_period']:]
            
            self.state['fast_ma'] = np.mean(fast_prices)
            self.state['slow_ma'] = np.mean(slow_prices)
            
            # Generate signals
            signal = self._generate_signal(data, position)
            
            return TradeSignal(
                signal=signal,
                quantity=self._calculate_position_size(data),
                stop_loss=self._calculate_stop_loss(data, signal),
                take_profit=self._calculate_take_profit(data, signal)
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _generate_signal(self, data: MarketData, position: Position) -> Signal:
        """Generate trading signal based on MA crossover."""
        fast_ma = self.state['fast_ma']
        slow_ma = self.state['slow_ma']
        prev_fast_ma = self.state['prev_fast_ma']
        prev_slow_ma = self.state['prev_slow_ma']
        
        # Bullish crossover (fast MA crosses above slow MA)
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            return Signal.BUY
        
        # Bearish crossover (fast MA crosses below slow MA)
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            return Signal.SELL
        
        return Signal.HOLD
    
    def _calculate_position_size(self, data: MarketData) -> float:
        """Calculate position size based on percentage of portfolio."""
        # This would typically use portfolio value from backtester
        # For now, return relative size
        return self.parameters['position_size_pct']
    
    def _calculate_stop_loss(self, data: MarketData, signal: Signal) -> float:
        """Calculate stop loss price."""
        if signal == Signal.BUY:
            return data.close * (1 - self.parameters['stop_loss_pct'])
        elif signal == Signal.SELL:
            return data.close * (1 + self.parameters['stop_loss_pct'])
        return None
    
    def _calculate_take_profit(self, data: MarketData, signal: Signal) -> float:
        """Calculate take profit price."""
        if signal == Signal.BUY:
            return data.close * (1 + self.parameters['take_profit_pct'])
        elif signal == Signal.SELL:
            return data.close * (1 - self.parameters['take_profit_pct'])
        return None
```

## File Placement

Strategy files should be placed in:
```
examples/strategies/
├── my_strategy.py
├── moving_average_crossover.py
└── mean_reversion.py
```

## Usage with Backtester

```bash
# Run backtest with strategy file
python -m modules.backtester --strategy ./examples/strategies/my_strategy.py --data ./data/raw/SOLUSDT_5m.csv
```

## Best Practices

1. **Keep strategies focused**: One main logic per strategy class
2. **Use descriptive parameter names**: Make parameters self-explanatory  
3. **Implement proper risk management**: Always include stop loss logic
4. **Handle edge cases**: Check for insufficient data, division by zero, etc.
5. **Document your logic**: Add comments explaining the strategy rationale
6. **Test incrementally**: Start with simple logic, add complexity gradually
7. **Use type hints**: All methods should have proper type annotations
8. **Follow naming conventions**: Use descriptive class and method names

## Common Patterns

### Indicator Calculation
```python
def _calculate_sma(self, prices: List[float], period: int) -> float:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return 0.0
    return sum(prices[-period:]) / period

def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
    """Calculate Relative Strength Index."""
    # RSI implementation
    pass
```

### State Management
```python
def _update_state(self, data: MarketData):
    """Update internal strategy state."""
    self.state['price_history'].append(data.close)
    # Keep only necessary history to avoid memory issues
    if len(self.state['price_history']) > 100:
        self.state['price_history'] = self.state['price_history'][-50:]
```

### Signal Filtering
```python
def _should_trade(self, signal: Signal, position: Position) -> bool:
    """Apply additional filters to trading signals."""
    # Don't trade if already in position in same direction
    if signal == Signal.BUY and position.size > 0:
        return False
    if signal == Signal.SELL and position.size < 0:
        return False
    
    # Add more filters as needed
    return True
```