from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import StochasticOscillator
from typing import List, Optional, Dict
from datetime import datetime


class KDJStrategy(StrategyBase):
    """
    KDJ Strategy Implementation
    
    Hypothesis: При пересечении K и D вверх — открываем лонг; 
    закрываем при движении цены на 2 SOL или при пересечении K и D вниз.
    
    Entry Signal: K crosses D upward, K < 80 (avoid overbought zone)
    Exit Signal: Price moves up 2 SOL from entry OR K crosses D downward
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            'k_period': 14,         # Standard Stochastic period
            'd_period': 3,          # D smoothing period  
            'smooth_k': 1,          # No additional K smoothing
            'max_k_entry': 75,      # Max K value for entry (avoid overbought)
            'min_k_entry': 25,      # Min K value for entry (avoid oversold)
            'profit_target_sol': 1.5,  # Profit target in SOL
            'position_size_pct': 0.02,  # 2% position size
            'stop_loss_pct': 0.03,      # 3% stop loss
        }
        
        # Initialize KDJ indicator (using StochasticOscillator)
        self.kdj_indicator = StochasticOscillator(
            k_period=self.parameters['k_period'],
            d_period=self.parameters['d_period'],
            smooth_k=self.parameters['smooth_k']
        )
        
        # Strategy state
        self.state = {
            'k_value': 50.0,
            'd_value': 50.0, 
            'j_value': 50.0,
            'prev_k': 50.0,
            'prev_d': 50.0,
            'ohlc_history': [],
            'entry_price': 0.0,
            'trades_count': 0,
            'signals_generated': []
        }

    def calculate_j_value(self, k: float, d: float) -> float:
        """Calculate J value using formula: J = 3*K - 2*D"""
        return 3.0 * k - 2.0 * d

    def update_indicators(self, data: MarketData) -> None:
        """Update KDJ indicators with new market data."""
        # Store previous values for crossover detection
        self.state['prev_k'] = self.state['k_value']
        self.state['prev_d'] = self.state['d_value']
        
        # Add current OHLC to history
        ohlc_bar = {
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'close': data.close,
            'volume': data.volume
        }
        self.state['ohlc_history'].append(ohlc_bar)
        
        # Keep only necessary history
        max_history = max(self.parameters['k_period'] * 2, 50)
        if len(self.state['ohlc_history']) > max_history:
            self.state['ohlc_history'] = self.state['ohlc_history'][-max_history:]
        
        # Calculate KDJ values
        if len(self.state['ohlc_history']) >= self.parameters['k_period']:
            kdj_result = self.kdj_indicator.calculate_ohlc(self.state['ohlc_history'])
            self.state['k_value'] = kdj_result['k']
            self.state['d_value'] = kdj_result['d']
            self.state['j_value'] = self.calculate_j_value(
                self.state['k_value'], 
                self.state['d_value']
            )

    def is_k_crosses_d_upward(self) -> bool:
        """Detect K crosses D upward."""
        return (
            self.state['k_value'] > self.state['d_value'] and
            self.state['prev_k'] <= self.state['prev_d']
        )
    
    def is_k_crosses_d_downward(self) -> bool:
        """Detect K crosses D downward."""
        return (
            self.state['k_value'] < self.state['d_value'] and
            self.state['prev_k'] >= self.state['prev_d']
        )

    def is_profit_target_reached(self, data: MarketData, position: Position) -> bool:
        """Check if profit target of 2 SOL is reached."""
        if position.direction != "LONG" or self.state['entry_price'] == 0.0:
            return False
        
        price_diff = data.close - self.state['entry_price']
        return price_diff >= self.parameters['profit_target_sol']

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
        """Check if indicators have enough data."""
        return (
            len(self.state['ohlc_history']) >= self.parameters['k_period'] and
            self.state['k_value'] != 50.0 and  # Default value check
            self.state['d_value'] != 50.0
        )
    
    def _generate_entry_signal(self, data: MarketData, position: Position) -> TradeSignal:
        """Generate entry signals based on KDJ crossover."""
        
        # LONG signal: K crosses D upward + K in acceptable range
        if (self.is_k_crosses_d_upward() and 
            self.parameters['min_k_entry'] < self.state['k_value'] < self.parameters['max_k_entry']):
            
            # Store entry price for profit target calculation
            self.state['entry_price'] = data.close
            self.state['trades_count'] += 1
            
            # Log signal details
            self.state['signals_generated'].append({
                'type': 'LONG_ENTRY',
                'time': data.timestamp,
                'price': data.close,
                'k_value': self.state['k_value'],
                'd_value': self.state['d_value'],
                'j_value': self.state['j_value']
            })
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"KDJ Long Entry: K={self.state['k_value']:.1f} crosses D={self.state['d_value']:.1f} upward, J={self.state['j_value']:.1f}",
                stop_loss=data.close * (1 - self.parameters['stop_loss_pct']),
                take_profit=data.close + self.parameters['profit_target_sol']
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Manage open positions based on exit conditions."""
        
        if position.direction == "LONG":
            # Exit condition 1: Profit target reached (2 SOL)
            if self.is_profit_target_reached(data, position):
                profit_sol = data.close - self.state['entry_price']
                
                self.state['signals_generated'].append({
                    'type': 'LONG_EXIT_PROFIT',
                    'time': data.timestamp,
                    'price': data.close,
                    'profit_sol': profit_sol,
                    'k_value': self.state['k_value'],
                    'd_value': self.state['d_value']
                })
                
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"Profit Target: +{profit_sol:.2f} SOL reached"
                )
            
            # Exit condition 2: K crosses D downward
            if self.is_k_crosses_d_downward():
                profit_sol = data.close - self.state['entry_price']
                
                self.state['signals_generated'].append({
                    'type': 'LONG_EXIT_CROSSOVER',
                    'time': data.timestamp,
                    'price': data.close,
                    'profit_sol': profit_sol,
                    'k_value': self.state['k_value'],
                    'd_value': self.state['d_value']
                })
                
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"KDJ Exit: K={self.state['k_value']:.1f} crosses D={self.state['d_value']:.1f} downward"
                )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Calculate position size for trade."""
        return capital * self.parameters['position_size_pct']
    
    def get_strategy_info(self) -> Dict:
        """Return strategy information for reporting."""
        return {
            'strategy_name': 'KDJ Strategy',
            'hypothesis': 'K crosses D upward -> LONG, exit at +2 SOL or K crosses D downward',
            'parameters': self.parameters,
            'total_signals': len(self.state['signals_generated']),
            'total_trades': self.state['trades_count'],
            'current_kdj': {
                'K': self.state['k_value'],
                'D': self.state['d_value'], 
                'J': self.state['j_value']
            }
        }