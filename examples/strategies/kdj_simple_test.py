from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import StochasticOscillator
from typing import List, Optional, Dict


class KDJSimpleTest(StrategyBase):
    """
    Simplified KDJ test for validation
    Same logic but with more conservative parameters and debugging
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            'k_period': 14,         # Standard Stochastic period
            'd_period': 3,          # D smoothing period
            'smooth_k': 1,          # No additional K smoothing for simplicity
            'max_k_entry': 70,      # More conservative entry (avoid overbought)
            'min_k_entry': 30,      # Avoid oversold entries too
            'profit_target_sol': 1.5,  # Smaller profit target
            'position_size_pct': 0.02,  # 2% position size
            'cooldown_bars': 5,     # Bars to wait after trade
        }
        
        self.kdj_indicator = StochasticOscillator(
            k_period=self.parameters['k_period'],
            d_period=self.parameters['d_period'],
            smooth_k=self.parameters['smooth_k']
        )
        
        self.state = {
            'k_value': 50.0,
            'd_value': 50.0,
            'j_value': 50.0,
            'prev_k': 50.0,
            'prev_d': 50.0,
            'ohlc_history': [],
            'entry_price': 0.0,
            'bars_since_trade': 0,
            'trade_count': 0,
            'debug_signals': []
        }

    def calculate_j_value(self, k: float, d: float) -> float:
        """J = 3*K - 2*D"""
        return 3.0 * k - 2.0 * d

    def update_indicators(self, data: MarketData) -> None:
        """Update indicators and track bar count."""
        self.state['prev_k'] = self.state['k_value']
        self.state['prev_d'] = self.state['d_value']
        self.state['bars_since_trade'] += 1
        
        ohlc_bar = {
            'open': data.open, 'high': data.high, 'low': data.low, 
            'close': data.close, 'volume': data.volume
        }
        self.state['ohlc_history'].append(ohlc_bar)
        
        # Keep reasonable history
        if len(self.state['ohlc_history']) > 50:
            self.state['ohlc_history'] = self.state['ohlc_history'][-50:]
        
        if len(self.state['ohlc_history']) >= self.parameters['k_period']:
            kdj_result = self.kdj_indicator.calculate_ohlc(self.state['ohlc_history'])
            self.state['k_value'] = kdj_result['k']
            self.state['d_value'] = kdj_result['d']
            self.state['j_value'] = self.calculate_j_value(
                self.state['k_value'], self.state['d_value']
            )

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Main strategy logic."""
        self.update_indicators(data)
        
        if not self._indicators_ready():
            return TradeSignal(signal=Signal.HOLD)
        
        if position.direction != "NONE":
            return self._manage_position(data, position)
        
        return self._generate_entry_signal(data, position)
    
    def _indicators_ready(self) -> bool:
        """Check if we have enough data."""
        return len(self.state['ohlc_history']) >= self.parameters['k_period']
    
    def _generate_entry_signal(self, data: MarketData, position: Position) -> TradeSignal:
        """Generate entry signals with additional filters."""
        
        # Check cooldown period
        if self.state['bars_since_trade'] < self.parameters['cooldown_bars']:
            return TradeSignal(signal=Signal.HOLD)
        
        # K crosses D upward
        k_crosses_up = (
            self.state['k_value'] > self.state['d_value'] and
            self.state['prev_k'] <= self.state['prev_d']
        )
        
        # Additional filters
        k_in_range = (
            self.parameters['min_k_entry'] < self.state['k_value'] < self.parameters['max_k_entry']
        )
        
        # Price momentum filter (simple)
        if len(self.state['ohlc_history']) >= 2:
            price_momentum = data.close > self.state['ohlc_history'][-2]['close']
        else:
            price_momentum = True
        
        if k_crosses_up and k_in_range and price_momentum:
            self.state['entry_price'] = data.close
            self.state['trade_count'] += 1
            self.state['bars_since_trade'] = 0
            
            self.state['debug_signals'].append({
                'action': 'ENTRY',
                'time': data.timestamp,
                'price': data.close,
                'k': self.state['k_value'],
                'd': self.state['d_value'],
                'j': self.state['j_value']
            })
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"KDJ Entry: K={self.state['k_value']:.1f} > D={self.state['d_value']:.1f}, J={self.state['j_value']:.1f}",
                take_profit=data.close + self.parameters['profit_target_sol']
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Manage open positions."""
        
        if position.direction == "LONG":
            # Profit target reached
            price_gain = data.close - self.state['entry_price']
            if price_gain >= self.parameters['profit_target_sol']:
                self.state['debug_signals'].append({
                    'action': 'EXIT_PROFIT',
                    'time': data.timestamp,
                    'price': data.close,
                    'gain': price_gain,
                    'k': self.state['k_value'],
                    'd': self.state['d_value']
                })
                
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"Profit target: +{price_gain:.2f} SOL"
                )
            
            # K crosses D downward (exit signal)
            k_crosses_down = (
                self.state['k_value'] < self.state['d_value'] and
                self.state['prev_k'] >= self.state['prev_d']
            )
            
            if k_crosses_down:
                self.state['debug_signals'].append({
                    'action': 'EXIT_CROSS',
                    'time': data.timestamp,
                    'price': data.close,
                    'gain': price_gain,
                    'k': self.state['k_value'],
                    'd': self.state['d_value']
                })
                
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"KDJ Exit: K={self.state['k_value']:.1f} < D={self.state['d_value']:.1f}"
                )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Position sizing."""
        return capital * self.parameters['position_size_pct']
    
    def get_strategy_info(self) -> Dict:
        """Strategy info for analysis."""
        return {
            'strategy_name': 'KDJ Simple Test',
            'parameters': self.parameters,
            'total_trades': self.state['trade_count'],
            'debug_signals_count': len(self.state['debug_signals']),
            'last_kdj': {
                'K': self.state['k_value'],
                'D': self.state['d_value'],
                'J': self.state['j_value']
            }
        }