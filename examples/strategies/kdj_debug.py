from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import StochasticOscillator


class KDJDebug(StrategyBase):
    """Debug version - minimal KDJ with very simple logic"""
    
    def _initialize(self):
        self.parameters = {
            'k_period': 14,
            'd_period': 3,
            'smooth_k': 1,
            'position_size_pct': 0.01,
        }
        
        self.kdj = StochasticOscillator(
            k_period=self.parameters['k_period'],
            d_period=self.parameters['d_period'],
            smooth_k=self.parameters['smooth_k']
        )
        
        self.state = {
            'ohlc_history': [],
            'trade_count': 0,
            'position_opened': False
        }

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        # Build OHLC history
        ohlc = {
            'open': data.open, 'high': data.high, 
            'low': data.low, 'close': data.close, 'volume': data.volume
        }
        self.state['ohlc_history'].append(ohlc)
        
        # Keep only needed history
        if len(self.state['ohlc_history']) > 20:
            self.state['ohlc_history'] = self.state['ohlc_history'][-20:]
        
        # Wait for enough data
        if len(self.state['ohlc_history']) < self.parameters['k_period']:
            return TradeSignal(signal=Signal.HOLD)
        
        # Calculate KDJ
        kdj_result = self.kdj.calculate_ohlc(self.state['ohlc_history'])
        k_value = kdj_result['k']
        d_value = kdj_result['d']
        
        # Very simple logic: buy once when K > D and K < 70
        if (position.direction == "NONE" and 
            not self.state['position_opened'] and 
            k_value > d_value and k_value < 70):
            
            self.state['position_opened'] = True
            self.state['trade_count'] += 1
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"KDJ Debug: K={k_value:.1f} > D={d_value:.1f}",
                take_profit=data.close * 1.015,  # 1.5% profit target
                stop_loss=data.close * 0.985     # 1.5% stop loss
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        return capital * self.parameters['position_size_pct']