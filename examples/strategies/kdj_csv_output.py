from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators.kdj_tradingview import KDJTradingView
from typing import List, Optional, Dict
import pandas as pd
import os
# Excel dependencies are now handled by the reporter module


class KDJCsvOutput(StrategyBase):
    """
    KDJ Strategy with CSV output containing all signals and indicator values
    
    Output CSV columns:
    - Original OHLCV data
    - K, D, J values
    - buy_signal (1/0)
    - sell_signal (1/0)
    - in_position (1/0)
    - entry_price
    - current_pnl_sol
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            # KDJ Parameters (exact TradingView/Bybit algorithm)
            'ilong': 9,             # Period for highest/lowest (default: 9)
            'isig': 3,              # Signal period for smoothing (default: 3)
            
            # Entry Filters
            'max_k_entry': 80,      # Max K value for entry
            'min_k_entry': 20,      # Min K value for entry
            
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
            'exit_price': None,
            'trade_count': 0,
            
            # CSV output data
            'csv_data': []
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
        
        # Keep only necessary history
        max_history = self.parameters['ilong'] + 10
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

    def calculate_current_pnl_sol(self, current_price: float) -> float:
        """Calculate current P&L in SOL."""
        if self.state['entry_price'] is None:
            return 0.0
        return current_price - self.state['entry_price']

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Process each bar and generate trading signals."""
        # Update indicators first
        self.update_kdj_indicators(data)
        
        # Initialize signal flags  
        signal_action = "NONE"
        
        # Current position status
        position_status = position.direction if position.direction != "NONE" else "NONE"
        current_pnl_sol = self.calculate_current_pnl_sol(data.close) if position_status == "LONG" else 0.0
        
        # Wait for indicators to have enough data
        if len(self.state['ohlc_data']) < self.parameters['ilong']:
            signal = TradeSignal(signal=Signal.HOLD)
        else:
            # Handle existing position
            if position.direction == "LONG":
                signal = self._handle_existing_position(data)
                if signal.signal == Signal.CLOSE_LONG:
                    signal_action = "SELL"
                    self.state['exit_price'] = data.close
                else:
                    signal_action = "HOLD"  # Holding existing position
            else:
                # Look for new entry opportunities
                signal = self._look_for_entry_signal(data)
                if signal.signal == Signal.BUY:
                    signal_action = "BUY"
                    self.state['exit_price'] = None  # Reset exit price for new trade
                else:
                    signal_action = "NONE"  # No position, no signal
        
        # Record data for CSV output
        csv_row = {
            'timestamp': data.timestamp,
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'close': data.close,
            'volume': data.volume,
            'k_value': round(self.state['k_current'], 2),
            'd_value': round(self.state['d_current'], 2),
            'j_value': round(self.state['j_current'], 2),
            'signal': signal_action,
            'position': position_status,
            'entry_price': self.state['entry_price'] if self.state['entry_price'] else 0.0,
            'exit_price': self.state['exit_price'] if self.state['exit_price'] else 0.0,
            'current_pnl_sol': round(current_pnl_sol, 4)
        }
        
        self.state['csv_data'].append(csv_row)
        
        return signal
    
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
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"KDJ Entry #{self.state['trade_count']}: K={self.state['k_current']:.1f} crosses D={self.state['d_current']:.1f} up"
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _handle_existing_position(self, data: MarketData) -> TradeSignal:
        """Handle existing long position - check exit conditions."""
        
        if self.state['entry_price'] is None:
            # Safety check
            return TradeSignal(signal=Signal.CLOSE_LONG, reason="Missing entry price - safety close")
        
        current_profit_sol = data.close - self.state['entry_price']
        
        # Exit Condition 1: Profit target of 2 SOL reached
        if self.is_profit_target_reached(data.close):
            # Reset entry price
            self.state['entry_price'] = None
            
            return TradeSignal(
                signal=Signal.CLOSE_LONG,
                reason=f"Profit Target Reached: +{current_profit_sol:.2f} SOL"
            )
        
        # Exit Condition 2: K crosses D downward
        if self.detect_k_crosses_d_downward():
            # Reset entry price
            self.state['entry_price'] = None
            
            return TradeSignal(
                signal=Signal.CLOSE_LONG,
                reason=f"KDJ Exit: K crosses D down, P/L: {current_profit_sol:+.2f} SOL"
            )
        
        # Hold position
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Calculate position size based on capital percentage."""
        return capital * self.parameters['position_size_pct']
    
    def save_csv_output(self, output_path: str = None) -> str:
        """Save CSV with all data and signals."""
        if not self.state['csv_data']:
            return "No data to save"
        
        # Create DataFrame
        df = pd.DataFrame(self.state['csv_data'])
        
        # Set default output path if not provided
        if output_path is None:
            output_dir = "output/csv_reports"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/kdj_strategy_signals.csv"
        else:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def save_excel_output(self, output_path: str = None) -> str:
        """Save Excel with color-coded positions and signals using reporter module."""
        if not self.state['csv_data']:
            return "No data to save"
        
        # Set default output path if not provided
        if output_path is None:
            output_dir = "output/excel_reports"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/kdj_strategy_signals.xlsx"
        
        # Import Excel exporter from reporter module
        try:
            from modules.reporter.excel_exporter import ExcelExporter
        except ImportError:
            print("❌ Excel exporter not available")
            return "Excel exporter not available"
        
        # Create exporter and export data
        exporter = ExcelExporter()
        
        # Check if Excel libraries are available
        if not exporter.check_availability():
            print("❌ Excel export not available. Install openpyxl: pip install openpyxl")
            return "Excel library not available"
        
        # Get strategy info for legend
        strategy_info = self.get_strategy_info()
        
        try:
            return exporter.export_strategy_signals(
                data=self.state['csv_data'],
                output_path=output_path,
                strategy_info=strategy_info
            )
        except Exception as e:
            print(f"❌ Excel export failed: {e}")
            return f"Excel export failed: {e}"
    
    def get_strategy_info(self) -> Dict:
        """Return strategy information for analysis."""
        return {
            'strategy_name': 'KDJ CSV Output Strategy',
            'hypothesis': 'K crosses D up -> Long, exit at +2 SOL or K crosses D down',
            'parameters': self.parameters,
            'statistics': {
                'total_trades': self.state['trade_count'],
                'total_bars_processed': len(self.state['csv_data'])
            },
            'output_columns': [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'k_value', 'd_value', 'j_value', 
                'signal', 'position', 
                'entry_price', 'exit_price', 'current_pnl_sol'
            ],
            'color_coding': {
                'LONG': 'Light Green - Currently holding long position',
                'SHORT': 'Light Salmon - Currently holding short position',  
                'BUY': 'Lime Green - Entry signal generated',
                'SELL': 'Light Red - Exit signal generated',
                'HOLD': 'Gold - Holding existing position',
                'NONE': 'Light Gray - No position, no signal'
            }
        }