from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import RSI, EMA, SMA
from typing import List, Optional, Dict
from datetime import datetime, timedelta


class SimpleDemoStrategy(StrategyBase):
    """
    Simple demo strategy that will actually generate trades.
    Uses simple moving average crossover with RSI filter.
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            'ema_fast': 5,     # Очень быстрые MA для частых сигналов
            'ema_slow': 15,    
            'rsi_period': 14,
            'rsi_oversold': 45,    # Очень мягкие условия RSI
            'rsi_overbought': 55,
            'volume_ma_period': 10,
            'position_size_pct': 0.01,  # 1% риск
            'stop_loss_pct': 0.02,      # 2% стоп
            'take_profit_pct': 0.03,    # 3% тейк
        }
        
        # Initialize indicators
        self.indicators = {
            'ema_fast': EMA(period=self.parameters['ema_fast']),
            'ema_slow': EMA(period=self.parameters['ema_slow']),
            'rsi': RSI(period=self.parameters['rsi_period']),
            'volume_ma': SMA(period=self.parameters['volume_ma_period'])
        }
        
        # Strategy state
        self.state = {
            'ema_fast_val': 0.0,
            'ema_slow_val': 0.0,
            'prev_ema_fast': 0.0,
            'prev_ema_slow': 0.0,
            'rsi_val': 50.0,
            'volume_ma_val': 0.0,
            'trades_count': 0,
            'signals_generated': []
        }

    def update_indicators(self, data: MarketData) -> None:
        """Update technical indicators with new market data."""
        # Store previous values
        self.state['prev_ema_fast'] = self.state['ema_fast_val'] or data.close
        self.state['prev_ema_slow'] = self.state['ema_slow_val'] or data.close
        
        # Update indicators
        self.state['ema_fast_val'] = self.indicators['ema_fast'].update(data.close)
        self.state['ema_slow_val'] = self.indicators['ema_slow'].update(data.close)
        self.state['rsi_val'] = self.indicators['rsi'].update(data.close)
        self.state['volume_ma_val'] = self.indicators['volume_ma'].update(data.volume)
        
        # Handle None values
        if self.state['ema_fast_val'] is None:
            self.state['ema_fast_val'] = data.close
        if self.state['ema_slow_val'] is None:
            self.state['ema_slow_val'] = data.close
        if self.state['rsi_val'] is None:
            self.state['rsi_val'] = 50.0
        if self.state['volume_ma_val'] is None:
            self.state['volume_ma_val'] = data.volume

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Process new market data and generate trading signal."""
        # Update indicators first
        self.update_indicators(data)
        
        # Check if indicators are ready
        if not self._indicators_ready():
            return TradeSignal(signal=Signal.HOLD)
        
        # Position management for open positions
        if position.direction != "NONE":
            return self._manage_position(data, position)
        
        # Entry signal generation
        return self._generate_entry_signal(data, position)
    
    def _indicators_ready(self) -> bool:
        """Check if all indicators are ready."""
        return (
            self.indicators['ema_fast'].is_ready() and
            self.indicators['ema_slow'].is_ready() and
            self.indicators['rsi'].is_ready() and
            self.indicators['volume_ma'].is_ready()
        )
    
    def _generate_entry_signal(self, data: MarketData, position: Position) -> TradeSignal:
        """Generate entry signals based on MA crossover + RSI."""
        
        # MA Crossover detection
        golden_cross = (
            self.state['ema_fast_val'] > self.state['ema_slow_val'] and
            self.state['prev_ema_fast'] <= self.state['prev_ema_slow']
        )
        
        death_cross = (
            self.state['ema_fast_val'] < self.state['ema_slow_val'] and
            self.state['prev_ema_fast'] >= self.state['prev_ema_slow']
        )
        
        # Volume filter (простой)
        volume_ok = data.volume > self.state['volume_ma_val'] * 0.8
        
        # LONG signal: Golden cross + RSI not overbought + volume
        if (golden_cross and 
            self.state['rsi_val'] < self.parameters['rsi_overbought'] and 
            volume_ok):
            
            self.state['trades_count'] += 1
            self.state['signals_generated'].append({
                'type': 'LONG',
                'time': data.timestamp,
                'price': data.close,
                'rsi': self.state['rsi_val'],
                'ema_fast': self.state['ema_fast_val'],
                'ema_slow': self.state['ema_slow_val']
            })
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"Golden Cross: EMA({self.parameters['ema_fast']})={self.state['ema_fast_val']:.2f} > EMA({self.parameters['ema_slow']})={self.state['ema_slow_val']:.2f}, RSI={self.state['rsi_val']:.1f}",
                stop_loss=data.close * (1 - self.parameters['stop_loss_pct']),
                take_profit=data.close * (1 + self.parameters['take_profit_pct'])
            )
        
        # SHORT signal: Death cross + RSI not oversold + volume  
        if (death_cross and 
            self.state['rsi_val'] > self.parameters['rsi_oversold'] and 
            volume_ok):
            
            self.state['trades_count'] += 1
            self.state['signals_generated'].append({
                'type': 'SHORT',
                'time': data.timestamp,
                'price': data.close,
                'rsi': self.state['rsi_val'],
                'ema_fast': self.state['ema_fast_val'],
                'ema_slow': self.state['ema_slow_val']
            })
            
            return TradeSignal(
                signal=Signal.SELL,
                reason=f"Death Cross: EMA({self.parameters['ema_fast']})={self.state['ema_fast_val']:.2f} < EMA({self.parameters['ema_slow']})={self.state['ema_slow_val']:.2f}, RSI={self.state['rsi_val']:.1f}",
                stop_loss=data.close * (1 + self.parameters['stop_loss_pct']),
                take_profit=data.close * (1 - self.parameters['take_profit_pct'])
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Simple position management."""
        # Простое правило: закрыть позицию при противоположном кроссовере
        
        # Обратный crossover для LONG позиций
        if (position.direction == "LONG" and 
            self.state['ema_fast_val'] < self.state['ema_slow_val'] and
            self.state['prev_ema_fast'] >= self.state['prev_ema_slow']):
            
            return TradeSignal(
                signal=Signal.CLOSE_LONG,
                reason=f"Exit LONG: Death Cross detected"
            )
        
        # Обратный crossover для SHORT позиций
        if (position.direction == "SHORT" and 
            self.state['ema_fast_val'] > self.state['ema_slow_val'] and
            self.state['prev_ema_fast'] <= self.state['prev_ema_slow']):
            
            return TradeSignal(
                signal=Signal.CLOSE_SHORT,
                reason=f"Exit SHORT: Golden Cross detected"
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Calculate position size for trade."""
        return capital * self.parameters['position_size_pct']