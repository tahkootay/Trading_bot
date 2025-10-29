from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators.kdj_tradingview import KDJTradingView
from typing import List, Optional, Dict


class KDJFinal(StrategyBase):
    """
    KDJ Strategy - Final Implementation
    
    Hypothesis Test: "При пересечении K и D вверх — открываем лонг; 
    закрываем при движении цены на 2 SOL или при пересечении K и D вниз."
    
    Entry: K crosses D upward + K < 80 (avoid overbought)
    Exit: Price moves up 2 SOL OR K crosses D downward
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            # KDJ Parameters
            'ilong': 9,             # Period for highest/lowest 
            'isig': 3,              # Signal period for smoothing
            
            # Entry Filters
            'max_k_entry': 80,      # Max K value for entry (avoid overbought)
            'min_k_entry': 20,      # Min K value for entry (avoid oversold)
            
            # Exit Conditions
            'profit_target_sol': 2.0,  # Profit target: 2 SOL
            
            # Risk Management
            'position_size_pct': 0.02,  # 2% of capital per trade
        }
        
        # Initialize KDJ indicator (exact TradingView/Bybit algorithm)
        self.kdj_indicator = KDJTradingView(
            ilong=self.parameters['ilong'],
            isig=self.parameters['isig']
        )
        
        # Strategy state
        self.state = {
            'k_current': 50.0,
            'd_current': 50.0,
            'j_current': 50.0,
            'k_previous': 50.0,
            'd_previous': 50.0,
            'ohlc_data': [],
            'entry_price': None,
            'trade_count': 0,
            'trade_log': []
        }

    def calculate_j_line(self, k: float, d: float) -> float:
        """Calculate J line: J = 3*K - 2*D"""
        return 3.0 * k - 2.0 * d

    def update_kdj_indicators(self, data: MarketData) -> None:
        """Update KDJ indicators with new market data."""
        # Store previous values for crossover detection
        self.state['k_previous'] = self.state['k_current']
        self.state['d_previous'] = self.state['d_current']
        
        # Add new OHLC data
        new_bar = {
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'close': data.close,
            'volume': data.volume
        }
        self.state['ohlc_data'].append(new_bar)
        
        # Keep only necessary history (period + buffer)
        max_history = self.parameters['ilong'] + 5
        if len(self.state['ohlc_data']) > max_history:
            self.state['ohlc_data'] = self.state['ohlc_data'][-max_history:]
        
        # Calculate KDJ values if we have enough data
        if len(self.state['ohlc_data']) >= self.parameters['ilong']:
            kdj_values = self.kdj_indicator.calculate_ohlc(self.state['ohlc_data'])
            self.state['k_current'] = kdj_values['k']
            self.state['d_current'] = kdj_values['d']
            self.state['j_current'] = kdj_values['j']  # J уже вычислен в TradingView алгоритме

    def detect_k_crosses_d_upward(self) -> bool:
        """Detect when K line crosses D line from below to above."""
        return (
            self.state['k_current'] > self.state['d_current'] and
            self.state['k_previous'] <= self.state['d_previous']
        )
    
    def detect_k_crosses_d_downward(self) -> bool:
        """Detect when K line crosses D line from above to below."""
        return (
            self.state['k_current'] < self.state['d_current'] and
            self.state['k_previous'] >= self.state['d_previous']
        )

    def is_profit_target_reached(self, current_price: float) -> bool:
        """Check if 2 SOL profit target is reached."""
        if self.state['entry_price'] is None:
            return False
        profit = current_price - self.state['entry_price']
        return profit >= self.parameters['profit_target_sol']

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Process each bar and generate trading signals."""
        # Update indicators first
        self.update_kdj_indicators(data)
        
        # Wait for indicators to have enough data
        if len(self.state['ohlc_data']) < self.parameters['ilong']:
            return TradeSignal(signal=Signal.HOLD)
        
        # Handle existing position
        if position.direction == "LONG":
            return self._handle_existing_position(data)
        
        # Look for new entry opportunities
        return self._look_for_entry_signal(data)
    
    def _look_for_entry_signal(self, data: MarketData) -> TradeSignal:
        """Look for entry signals based on KDJ crossover."""
        
        # Entry condition: K crosses D upward + K in acceptable range
        k_crosses_up = self.detect_k_crosses_d_upward()
        k_in_range = (
            self.parameters['min_k_entry'] < self.state['k_current'] < self.parameters['max_k_entry']
        )
        
        if k_crosses_up and k_in_range:
            # Record trade details
            self.state['entry_price'] = data.close
            self.state['trade_count'] += 1
            
            # Log entry signal
            self.state['trade_log'].append({
                'action': 'ENTRY',
                'timestamp': str(data.timestamp),
                'price': data.close,
                'k_value': self.state['k_current'],
                'd_value': self.state['d_current'],
                'j_value': self.state['j_current'],
                'trade_number': self.state['trade_count']
            })
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"KDJ Entry #{self.state['trade_count']}: K={self.state['k_current']:.1f} crosses D={self.state['d_current']:.1f} up, J={self.state['j_current']:.1f}"
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _handle_existing_position(self, data: MarketData) -> TradeSignal:
        """Handle existing long position - check exit conditions."""
        
        if self.state['entry_price'] is None:
            # Safety check - should not happen
            return TradeSignal(signal=Signal.CLOSE_LONG, reason="Missing entry price - safety close")
        
        current_profit_sol = data.close - self.state['entry_price']
        
        # Exit Condition 1: Profit target of 2 SOL reached
        if self.is_profit_target_reached(data.close):
            self.state['trade_log'].append({
                'action': 'EXIT_PROFIT',
                'timestamp': str(data.timestamp),
                'price': data.close,
                'profit_sol': current_profit_sol,
                'k_value': self.state['k_current'],
                'd_value': self.state['d_current']
            })
            
            # Reset entry price
            self.state['entry_price'] = None
            
            return TradeSignal(
                signal=Signal.CLOSE_LONG,
                reason=f"Profit Target Reached: +{current_profit_sol:.2f} SOL"
            )
        
        # Exit Condition 2: K crosses D downward
        if self.detect_k_crosses_d_downward():
            self.state['trade_log'].append({
                'action': 'EXIT_CROSSOVER',
                'timestamp': str(data.timestamp),
                'price': data.close,
                'profit_sol': current_profit_sol,
                'k_value': self.state['k_current'],
                'd_value': self.state['d_current']
            })
            
            # Reset entry price
            self.state['entry_price'] = None
            
            return TradeSignal(
                signal=Signal.CLOSE_LONG,
                reason=f"KDJ Exit: K={self.state['k_current']:.1f} crosses D={self.state['d_current']:.1f} down, P/L: {current_profit_sol:+.2f} SOL"
            )
        
        # Hold position
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Calculate position size based on capital percentage."""
        return capital * self.parameters['position_size_pct']
    
    def get_strategy_info(self) -> Dict:
        """Return strategy information for analysis."""
        return {
            'strategy_name': 'KDJ Final Strategy',
            'hypothesis': 'K crosses D up -> Long, exit at +2 SOL or K crosses D down',
            'parameters': self.parameters,
            'statistics': {
                'total_trades': self.state['trade_count'],
                'signals_logged': len(self.state['trade_log'])
            },
            'current_indicators': {
                'K': round(self.state['k_current'], 2),
                'D': round(self.state['d_current'], 2),
                'J': round(self.state['j_current'], 2)
            },
            'position_status': {
                'entry_price': self.state['entry_price'],
                'in_position': self.state['entry_price'] is not None
            }
        }