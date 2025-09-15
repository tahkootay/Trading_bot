#!/usr/bin/env python3
"""
Script to run detailed MACD+RSI strategy and export comprehensive data.
"""

import sys
import os
sys.path.append('/Users/alexey/Documents/Development/Python/Trading_bot')

from examples.strategies.macd_rsi_detailed_strategy import MacdRsiDetailedStrategy
from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
import pandas as pd
import numpy as np
from datetime import datetime


class SimplePosition:
    """Simple position class for backtesting."""
    
    def __init__(self):
        self.direction = "NONE"
        self.entry_price = 0.0
        self.size = 0.0


class DetailedBacktester:
    """Simple backtester for detailed analysis."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = None
        self.trades = []
        
    def run_backtest(self, strategy: StrategyBase, data_file: str):
        """Run backtest with detailed logging."""
        
        # Load data
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"📊 Loading {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Initialize strategy
        strategy._initialize()
        
        # Initialize position
        current_position = SimplePosition()
        
        # Process each bar
        for idx, row in df.iterrows():
            # Create MarketData
            market_data = MarketData(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            
            # Update timestamp in strategy log
            if hasattr(strategy, 'detailed_log') and strategy.detailed_log:
                if len(strategy.detailed_log) > idx:
                    strategy.detailed_log[idx]['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            # Get signal from strategy
            signal = strategy.on_bar(market_data, current_position)
            
            # Execute signal
            if signal and signal.signal != Signal.HOLD:
                self._execute_signal(signal, market_data, current_position, strategy)
        
        # Export results
        log_file = strategy.export_detailed_log()
        
        return {
            'final_capital': self.current_capital,
            'return_pct': (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            'trades': len(self.trades),
            'log_file': log_file
        }
    
    def _execute_signal(self, signal: TradeSignal, data: MarketData, position: SimplePosition, strategy):
        """Execute trading signal."""
        
        if signal.signal == Signal.BUY and position.direction == "NONE":
            # Open long position
            position.direction = "LONG"
            position.entry_price = data.close
            position.size = self.current_capital * 0.05  # 5% position size
            
            self.trades.append({
                'type': 'OPEN_LONG',
                'price': data.close,
                'timestamp': data.timestamp,
                'size': position.size
            })
            
        elif signal.signal == Signal.SELL and position.direction == "NONE":
            # Open short position
            position.direction = "SHORT"
            position.entry_price = data.close
            position.size = self.current_capital * 0.05
            
            self.trades.append({
                'type': 'OPEN_SHORT',
                'price': data.close,
                'timestamp': data.timestamp,
                'size': position.size
            })
            
        elif signal.signal == Signal.CLOSE_LONG and position.direction == "LONG":
            # Close long position
            pnl = (data.close - position.entry_price) / position.entry_price * position.size
            self.current_capital += pnl
            
            self.trades.append({
                'type': 'CLOSE_LONG',
                'price': data.close,
                'timestamp': data.timestamp,
                'pnl': pnl
            })
            
            position.direction = "NONE"
            position.entry_price = 0
            position.size = 0
            
        elif signal.signal == Signal.CLOSE_SHORT and position.direction == "SHORT":
            # Close short position
            pnl = (position.entry_price - data.close) / position.entry_price * position.size
            self.current_capital += pnl
            
            self.trades.append({
                'type': 'CLOSE_SHORT',
                'price': data.close,
                'timestamp': data.timestamp,
                'pnl': pnl
            })
            
            position.direction = "NONE"
            position.entry_price = 0
            position.size = 0


def main():
    """Main execution function."""
    
    print("🚀 Starting detailed MACD+RSI backtest...")
    
    # Initialize strategy and backtester
    strategy = MacdRsiDetailedStrategy()
    backtester = DetailedBacktester()
    
    # Run backtest
    data_file = 'data/raw/SOLUSDT_15m_20250601_20250831.csv'
    results = backtester.run_backtest(strategy, data_file)
    
    print(f"\n📈 Backtest Results:")
    print(f"   Final Capital: ${results['final_capital']:,.2f}")
    print(f"   Total Return: {results['return_pct']:+.2f}%")
    print(f"   Total Trades: {results['trades']}")
    print(f"   Log File: {results['log_file']}")
    
    # Load and display sample of detailed log
    df = pd.read_csv(results['log_file'])
    
    print(f"\n📊 Detailed Log Summary:")
    print(f"   Total bars: {len(df)}")
    print(f"   Indicators ready: {df['indicators_ready'].sum()}")
    print(f"   Buy signals: {(df['signal_type'] == 'BUY').sum()}")
    print(f"   Sell signals: {(df['signal_type'] == 'SELL').sum()}")
    print(f"   Close long signals: {(df['signal_type'] == 'CLOSE_LONG').sum()}")
    print(f"   Close short signals: {(df['signal_type'] == 'CLOSE_SHORT').sum()}")
    
    # Show sample of data with trading signals
    print(f"\n📋 Sample Trading Signals:")
    trading_signals = df[df['signal_type'].isin(['BUY', 'SELL', 'CLOSE_LONG', 'CLOSE_SHORT'])]
    if len(trading_signals) > 0:
        print(trading_signals[['bar', 'timestamp', 'close', 'signal_type', 'macd_line', 'rsi', 'ema50']].head(10).to_string(index=False))
    
    print(f"\n✅ Analysis complete! Detailed data available in: {results['log_file']}")


if __name__ == "__main__":
    main()