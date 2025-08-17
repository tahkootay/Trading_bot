#!/usr/bin/env python3
"""Run enhanced backtest with testnet data."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

class SimpleBacktestEngine:
    """Simplified backtesting engine for testnet data."""
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage: float = 0.0005,  # 0.05%
    ):
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # Backtest state
        self.current_balance = initial_balance
        self.positions: List[Dict] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_balance
        
        # Simple strategy parameters (adjusted for testnet)
        self.strategy_params = {
            'rsi_oversold': 40,    # Less extreme RSI levels
            'rsi_overbought': 60,
            'sma_short': 5,        # Shorter periods for more signals
            'sma_long': 10,
            'position_size_pct': 0.02,  # 2% of balance
            'max_position_time_hours': 3,
        }
    
    def load_testnet_data(self, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """Load testnet data from CSV files."""
        data = {}
        data_path = Path(data_dir)
        
        timeframe_files = {
            "1m": "SOLUSDT_1m_testnet.csv",
            "5m": "SOLUSDT_5m_testnet.csv",
            "15m": "SOLUSDT_15m_testnet.csv",
            "1h": "SOLUSDT_1h_testnet.csv",
        }
        
        for tf, filename in timeframe_files.items():
            filepath = data_path / filename
            if filepath.exists():
                print(f"📊 Loading {tf} data from {filename}")
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Add simple technical indicators
                df = self._add_indicators(df)
                
                data[tf] = df
                print(f"  ✅ Loaded {len(df)} candles for {tf}")
            else:
                print(f"  ❌ No data file found: {filename}")
        
        return data
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple technical indicators."""
        # Simple Moving Averages (adjusted for testnet)
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        
        # RSI (simplified)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Price change
        df['price_change'] = df['close'].pct_change()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        primary_timeframe: str = "5m",
    ) -> Dict[str, Any]:
        """Run simplified backtesting simulation."""
        
        print(f"🔬 Running backtest with {primary_timeframe} data")
        
        if primary_timeframe not in data:
            raise ValueError(f"Primary timeframe {primary_timeframe} not available")
        
        df = data[primary_timeframe]
        
        print(f"📅 Period: {df.index[0]} → {df.index[-1]}")
        print(f"📊 Total candles: {len(df)}")
        print()
        
        # Ensure we have enough data for indicators
        min_periods = 20
        if len(df) < min_periods:
            raise ValueError(f"Not enough data. Need at least {min_periods} candles, got {len(df)}")
        
        # Process each candle
        for i in range(min_periods, len(df)):
            if i % 50 == 0:
                print(f"📈 Processing candle {i}/{len(df)} ({i/len(df)*100:.1f}%)")
            
            current_time = df.index[i]
            current_candle = df.iloc[i]
            current_price = current_candle['close']
            
            # Generate signal using simple strategy
            signal = self._generate_simple_signal(current_candle, df.iloc[:i+1])
            
            if signal:
                # Process signal
                self._process_signal(signal, current_time, current_price)
            
            # Update existing positions
            self._update_positions(current_time, current_price)
            
            # Record equity
            self._record_equity(current_time)
        
        # Close any remaining positions
        self._close_all_positions(df.index[-1], df.iloc[-1]['close'])
        
        # Calculate results
        results = self._calculate_results()
        
        return results
    
    def _generate_simple_signal(self, current_candle: pd.Series, historical_df: pd.DataFrame) -> Optional[Dict]:
        """Generate simple trading signal."""
        
        # Skip if we don't have enough data
        if len(historical_df) < 20:
            return None
        
        # Get indicators
        rsi = current_candle['rsi']
        sma_5 = current_candle['sma_5']
        sma_10 = current_candle['sma_10']
        current_price = current_candle['close']
        volume_ratio = current_candle['volume_ratio']
        price_change = current_candle['price_change']
        
        # Skip if indicators are NaN
        if pd.isna(rsi) or pd.isna(sma_5) or pd.isna(sma_10):
            return None
        
        # Simple strategy: RSI + Moving Average crossover + price movement
        signal_type = None
        confidence = 0.0
        
        # Bullish signal (more relaxed for testnet)
        if (rsi < self.strategy_params['rsi_oversold'] and 
            sma_5 > sma_10 and 
            current_price > sma_5 and
            abs(price_change) > 0.005):  # At least 0.5% price movement
            signal_type = "BUY"
            confidence = 0.6
        
        # Bearish signal (more relaxed for testnet)
        elif (rsi > self.strategy_params['rsi_overbought'] and 
              sma_5 < sma_10 and 
              current_price < sma_5 and
              abs(price_change) > 0.005):  # At least 0.5% price movement
            signal_type = "SELL"
            confidence = 0.6
        
        # Alternative: momentum-based signals for low-volume environment
        elif abs(price_change) > 0.01:  # 1% price movement
            if price_change > 0 and current_price > sma_5:
                signal_type = "BUY"
                confidence = 0.4
            elif price_change < 0 and current_price < sma_5:
                signal_type = "SELL"
                confidence = 0.4
        
        if signal_type:
            return {
                'type': signal_type,
                'confidence': confidence,
                'price': current_price,
                'rsi': rsi,
                'sma_signal': sma_5 - sma_10,
            }
        
        return None
    
    def _process_signal(self, signal: Dict, current_time: datetime, current_price: float) -> None:
        """Process trading signal."""
        
        # Check if we already have a position
        if self.positions:
            return  # Skip if already in position
        
        # Calculate position size
        position_value = self.current_balance * self.strategy_params['position_size_pct']
        quantity = position_value / current_price
        
        # Apply slippage
        if signal['type'] == "BUY":
            entry_price = current_price * (1 + self.slippage)
        else:
            entry_price = current_price * (1 - self.slippage)
        
        # Calculate commission
        commission = position_value * self.commission_rate
        
        # Create position
        position = {
            'signal_type': signal['type'],
            'entry_time': current_time,
            'entry_price': entry_price,
            'quantity': quantity,
            'commission_paid': commission,
            'unrealized_pnl': 0.0,
            'confidence': signal['confidence'],
            'signal_data': signal,
        }
        
        self.positions.append(position)
        self.current_balance -= commission
    
    def _update_positions(self, current_time: datetime, current_price: float) -> None:
        """Update existing positions."""
        
        positions_to_close = []
        
        for i, position in enumerate(self.positions):
            # Calculate unrealized PnL
            if position['signal_type'] == "BUY":
                pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - current_price) * position['quantity']
            
            position['unrealized_pnl'] = pnl
            
            # Simple exit rules
            should_close = False
            close_reason = ""
            
            # Time stop
            hours_elapsed = (current_time - position['entry_time']).total_seconds() / 3600
            if hours_elapsed > self.strategy_params['max_position_time_hours']:
                should_close = True
                close_reason = "time_stop"
            
            # Simple profit/loss rules
            pnl_pct = pnl / (position['quantity'] * position['entry_price'])
            if pnl_pct > 0.03:  # 3% profit
                should_close = True
                close_reason = "take_profit"
            elif pnl_pct < -0.02:  # 2% loss
                should_close = True
                close_reason = "stop_loss"
            
            if should_close:
                positions_to_close.append((i, close_reason, current_time, current_price))
        
        # Close positions
        for i, close_reason, close_time, close_price in reversed(positions_to_close):
            self._close_position(i, close_reason, close_time, close_price)
    
    def _close_position(
        self,
        position_index: int,
        close_reason: str,
        close_time: datetime,
        close_price: float,
    ) -> None:
        """Close a position."""
        
        position = self.positions[position_index]
        
        # Apply slippage
        if position['signal_type'] == "BUY":
            exit_price = close_price * (1 - self.slippage)
        else:
            exit_price = close_price * (1 + self.slippage)
        
        # Calculate final PnL
        if position['signal_type'] == "BUY":
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        # Calculate commission
        exit_commission = position['quantity'] * exit_price * self.commission_rate
        net_pnl = pnl - position['commission_paid'] - exit_commission
        
        # Update balance
        position_value = position['quantity'] * exit_price
        self.current_balance += position_value - exit_commission
        
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': close_time,
            'signal_type': position['signal_type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'gross_pnl': pnl,
            'commission': position['commission_paid'] + exit_commission,
            'net_pnl': net_pnl,
            'close_reason': close_reason,
            'duration': close_time - position['entry_time'],
            'confidence': position['confidence'],
        }
        
        self.trades.append(trade)
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1
        
        self.total_pnl += net_pnl
        
        # Remove position
        del self.positions[position_index]
    
    def _close_all_positions(self, final_time: datetime, final_price: float) -> None:
        """Close all remaining positions at the end."""
        while self.positions:
            self._close_position(0, "backtest_end", final_time, final_price)
    
    def _record_equity(self, timestamp: datetime) -> None:
        """Record current equity."""
        
        # Calculate total unrealized PnL
        unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions)
        
        # Calculate position value
        position_value = sum(
            pos['quantity'] * pos['entry_price'] for pos in self.positions
        )
        
        total_equity = self.current_balance + position_value + unrealized_pnl
        
        # Update peak equity and drawdown
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
        
        current_drawdown = (self.peak_equity - total_equity) / self.peak_equity if self.peak_equity > 0 else 0
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'total_equity': total_equity,
            'drawdown': current_drawdown,
        })
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest results."""
        
        if not self.trades:
            return {"error": "No trades executed"}
        
        # Basic statistics
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        winning_trades_data = [t for t in self.trades if t['net_pnl'] > 0]
        losing_trades_data = [t for t in self.trades if t['net_pnl'] <= 0]
        
        avg_win = np.mean([t['net_pnl'] for t in winning_trades_data]) if winning_trades_data else 0
        avg_loss = abs(np.mean([t['net_pnl'] for t in losing_trades_data])) if losing_trades_data else 0
        
        profit_factor = (avg_win * len(winning_trades_data)) / (avg_loss * len(losing_trades_data)) if avg_loss > 0 else 0
        
        final_balance = self.current_balance
        total_return = (final_balance / self.initial_balance - 1) * 100
        
        # Duration statistics
        durations = [(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in self.trades]  # hours
        avg_duration = np.mean(durations) if durations else 0
        
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'total_return_pct': total_return,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.total_trades - self.winning_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': self.max_drawdown,
            'avg_trade_duration_hours': avg_duration,
            'total_commission': sum(t['commission'] for t in self.trades),
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted backtest results."""
        
        print("\\n" + "="*60)
        print("           TESTNET BACKTEST RESULTS (SOL/USDT)")
        print("="*60)
        
        if "error" in results:
            print(f"❌ {results['error']}")
            return
        
        # Performance Summary
        print("💰 PERFORMANCE SUMMARY")
        print("-" * 30)
        print(f"Initial Balance:      ${results['initial_balance']:,.2f}")
        print(f"Final Balance:        ${results['final_balance']:,.2f}")
        print(f"Total Return:         {results['total_return_pct']:+.2f}%")
        print(f"Total P&L:            ${results['total_pnl']:+,.2f}")
        print()
        
        # Trading Statistics
        print("📊 TRADING STATISTICS")
        print("-" * 30)
        print(f"Total Trades:         {results['total_trades']}")
        print(f"Winning Trades:       {results['winning_trades']}")
        print(f"Losing Trades:        {results['losing_trades']}")
        print(f"Win Rate:             {results['win_rate']:.1%}")
        print()
        
        # Risk Metrics
        print("⚠️  RISK METRICS")
        print("-" * 30)
        print(f"Max Drawdown:         {results['max_drawdown']:.1%}")
        print(f"Profit Factor:        {results['profit_factor']:.2f}")
        print(f"Average Win:          ${results['avg_win']:+.2f}")
        print(f"Average Loss:         ${results['avg_loss']:+.2f}")
        print(f"Avg Trade Duration:   {results['avg_trade_duration_hours']:.1f} hours")
        print()
        
        # Costs
        print("💸 COSTS")
        print("-" * 30)
        print(f"Total Commission:     ${results['total_commission']:,.2f}")
        print(f"Commission % of P&L:  {abs(results['total_commission']/results['total_pnl']*100) if results['total_pnl'] != 0 else 0:.1f}%")
        
        print("\\n" + "="*60)


def main():
    """Run testnet backtest."""
    
    print("🔬 Testnet Backtest for SOL/USDT")
    print("💰 Initial Balance: $10,000.00")
    print("💸 Commission Rate: 0.100%")
    print()
    
    # Initialize backtest engine
    engine = SimpleBacktestEngine(
        initial_balance=10000.0,
        commission_rate=0.001,
    )
    
    # Load data
    print("📊 Loading testnet data...")
    try:
        data = engine.load_testnet_data()
        
        if not data:
            print("❌ No testnet data found. Please run collect_testnet_data.py first")
            return
        
        print(f"✅ Loaded data for {len(data)} timeframes")
        print()
        
        # Run backtest
        print("🚀 Running backtest...")
        results = engine.run_backtest(data, primary_timeframe="5m")
        
        # Print results
        engine.print_results(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"testnet_backtest_results_{timestamp}.json"
        
        # Add trade details
        detailed_results = results.copy()
        detailed_results['trades'] = [
            {
                **trade,
                'entry_time': trade['entry_time'].isoformat(),
                'exit_time': trade['exit_time'].isoformat(),
                'duration': str(trade['duration']),
            } for trade in engine.trades
        ]
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\\n💾 Results saved to: {results_file}")
        
        # Show sample trades
        if engine.trades:
            print(f"\\n📋 Recent Trades (Last 5 of {len(engine.trades)}):") 
            print("-" * 80)
            print(f"{'Date':>12} {'Type':>4} {'Entry':>8} {'Exit':>8} {'P&L':>8} {'Duration':>8} {'Reason':>12}")
            print("-" * 80)
            
            for trade in engine.trades[-5:]:
                duration = trade['duration'].total_seconds() / 3600
                print(f"{trade['entry_time'].strftime('%m-%d %H:%M'):>12} "
                      f"{trade['signal_type']:>4} "
                      f"${trade['entry_price']:>7.2f} "
                      f"${trade['exit_price']:>7.2f} "
                      f"${trade['net_pnl']:>+7.2f} "
                      f"{duration:>7.1f}h "
                      f"{trade['close_reason']:>12}")
    
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()