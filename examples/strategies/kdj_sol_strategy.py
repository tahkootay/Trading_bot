from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import StochasticOscillator
from typing import List, Optional, Dict


class KDJSolStrategy(StrategyBase):
    """
    KDJ Strategy for SOL Trading
    
    Entry: K crosses D upward + K < 80 (avoid overbought)
    Exit: Price moves up 2 SOL OR K crosses D downward
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            'k_period': 14,         # Standard period for %K
            'd_period': 3,          # D smoothing period
            'smooth_k': 1,          # No additional K smoothing
            'max_k_entry': 80,      # Max K for entry (avoid overbought)
            'min_k_entry': 20,      # Min K for entry (avoid oversold)
            'profit_target_sol': 2.0,  # Profit target: 2 SOL
            'position_size_pct': 0.02,  # 2% position size
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
            'entry_price': None,
            'trade_count': 0,
            'signals_log': []
        }

    def calculate_j_value(self, k: float, d: float) -> float:
        """Calculate J line: J = 3*K - 2*D"""
        return 3.0 * k - 2.0 * d

    def update_indicators(self, data: MarketData) -> None:
        """Update KDJ indicators."""
        # Store previous values for crossover detection
        self.state['prev_k'] = self.state['k_value']
        self.state['prev_d'] = self.state['d_value']
        
        # Add current OHLC data
        ohlc_bar = {
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'close': data.close,
            'volume': data.volume
        }
        self.state['ohlc_history'].append(ohlc_bar)
        
        # Keep reasonable history
        max_history = self.parameters['k_period'] + 10
        if len(self.state['ohlc_history']) > max_history:
            self.state['ohlc_history'] = self.state['ohlc_history'][-max_history:]
        
        # Calculate KDJ if we have enough data
        if len(self.state['ohlc_history']) >= self.parameters['k_period']:
            kdj_result = self.kdj_indicator.calculate_ohlc(self.state['ohlc_history'])
            self.state['k_value'] = kdj_result['k']
            self.state['d_value'] = kdj_result['d']
            self.state['j_value'] = self.calculate_j_value(
                self.state['k_value'], 
                self.state['d_value']
            )

    def is_k_crosses_d_upward(self) -> bool:
        """Check if K line crosses D line upward."""
        return (
            self.state['k_value'] > self.state['d_value'] and
            self.state['prev_k'] <= self.state['prev_d']
        )
    
    def is_k_crosses_d_downward(self) -> bool:
        """Check if K line crosses D line downward."""
        return (
            self.state['k_value'] < self.state['d_value'] and
            self.state['prev_k'] >= self.state['prev_d']
        )

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Main strategy logic."""
        self.update_indicators(data)
        
        # Wait for indicators to be ready
        if len(self.state['ohlc_history']) < self.parameters['k_period']:
            return TradeSignal(signal=Signal.HOLD)
        
        # Handle existing position
        if position.direction == "LONG":
            return self._handle_long_position(data, position)
        
        # Look for entry signals
        return self._look_for_entry(data, position)
    
    def _look_for_entry(self, data: MarketData, position: Position) -> TradeSignal:
        """Look for entry signals."""
        # Entry condition: K crosses D upward + K in acceptable range
        if (self.is_k_crosses_d_upward() and 
            self.parameters['min_k_entry'] < self.state['k_value'] < self.parameters['max_k_entry']):
            
            # Record entry price for profit target calculation
            self.state['entry_price'] = data.close
            self.state['trade_count'] += 1
            
            # Log entry signal
            self.state['signals_log'].append({
                'action': 'ENTRY',
                'timestamp': data.timestamp,
                'price': data.close,
                'k': self.state['k_value'],
                'd': self.state['d_value'],
                'j': self.state['j_value']
            })
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"KDJ Entry: K={self.state['k_value']:.1f} > D={self.state['d_value']:.1f}, J={self.state['j_value']:.1f}"
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _handle_long_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Handle existing long position."""
        if self.state['entry_price'] is None:
            # Safety: if we don't have entry price, close position
            return TradeSignal(signal=Signal.CLOSE_LONG, reason="No entry price recorded")
        
        current_profit = data.close - self.state['entry_price']
        
        # Exit condition 1: Profit target reached (2 SOL)
        if current_profit >= self.parameters['profit_target_sol']:
            self.state['signals_log'].append({
                'action': 'EXIT_PROFIT',
                'timestamp': data.timestamp,
                'price': data.close,
                'profit_sol': current_profit,
                'k': self.state['k_value'],
                'd': self.state['d_value']
            })
            
            # Reset entry price
            self.state['entry_price'] = None
            
            return TradeSignal(
                signal=Signal.CLOSE_LONG,
                reason=f"Profit Target: +{current_profit:.2f} SOL reached"
            )
        
        # Exit condition 2: K crosses D downward
        if self.is_k_crosses_d_downward():
            self.state['signals_log'].append({
                'action': 'EXIT_CROSSOVER',
                'timestamp': data.timestamp,
                'price': data.close,
                'profit_sol': current_profit,
                'k': self.state['k_value'],
                'd': self.state['d_value']
            })
            
            # Reset entry price
            self.state['entry_price'] = None
            
            return TradeSignal(
                signal=Signal.CLOSE_LONG,
                reason=f"KDJ Exit: K={self.state['k_value']:.1f} crosses D={self.state['d_value']:.1f} down, P/L: {current_profit:+.2f} SOL"
            )
        
        # Hold position
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Calculate position size."""
        return capital * self.parameters['position_size_pct']
    
    def get_strategy_info(self) -> Dict:
        """Return strategy information."""
        return {
            'strategy_name': 'KDJ SOL Strategy',
            'description': 'K crosses D up -> Long, exit at +2 SOL or K crosses D down',
            'parameters': self.parameters,
            'total_trades': self.state['trade_count'],
            'signals_count': len(self.state['signals_log']),
            'current_kdj': {
                'K': self.state['k_value'],
                'D': self.state['d_value'],
                'J': self.state['j_value']
            }
        }