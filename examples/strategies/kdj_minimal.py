from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import StochasticOscillator
from typing import List, Optional, Dict


class KDJMinimal(StrategyBase):
    """
    Minimal KDJ strategy to test basic functionality
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            'k_period': 14,
            'd_period': 3,
            'smooth_k': 1,
            'position_size_pct': 0.01,
        }
        
        self.kdj_indicator = StochasticOscillator(
            k_period=self.parameters['k_period'],
            d_period=self.parameters['d_period'],
            smooth_k=self.parameters['smooth_k']
        )
        
        self.state = {
            'k_value': 50.0,
            'd_value': 50.0,
            'prev_k': 50.0,
            'prev_d': 50.0,
            'ohlc_history': [],
            'trade_count': 0
        }

    def update_indicators(self, data: MarketData) -> None:
        """Update indicators."""
        self.state['prev_k'] = self.state['k_value']
        self.state['prev_d'] = self.state['d_value']
        
        ohlc_bar = {
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'close': data.close,
            'volume': data.volume
        }
        self.state['ohlc_history'].append(ohlc_bar)
        
        # Keep history manageable
        if len(self.state['ohlc_history']) > 30:
            self.state['ohlc_history'] = self.state['ohlc_history'][-30:]
        
        if len(self.state['ohlc_history']) >= self.parameters['k_period']:
            kdj_result = self.kdj_indicator.calculate_ohlc(self.state['ohlc_history'])
            self.state['k_value'] = kdj_result['k']
            self.state['d_value'] = kdj_result['d']

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Main strategy logic - simplified."""
        self.update_indicators(data)
        
        if len(self.state['ohlc_history']) < self.parameters['k_period']:
            return TradeSignal(signal=Signal.HOLD)
        
        # If in position, just hold (no complex exit logic for now)
        if position.direction != "NONE":
            return TradeSignal(signal=Signal.HOLD)
        
        # Simple entry: K crosses above D
        k_crosses_up = (
            self.state['k_value'] > self.state['d_value'] and
            self.state['prev_k'] <= self.state['prev_d']
        )
        
        # Additional filter: K not too high (avoid overbought)
        if k_crosses_up and self.state['k_value'] < 80:
            self.state['trade_count'] += 1
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"KDJ: K={self.state['k_value']:.1f} crosses D={self.state['d_value']:.1f} up",
                take_profit=data.close * 1.02,  # 2% take profit
                stop_loss=data.close * 0.98     # 2% stop loss
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Position sizing."""
        return capital * self.parameters['position_size_pct']