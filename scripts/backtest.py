#!/usr/bin/env python3
"""Backtesting script for trading strategies."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.feature_engine.technical_indicators import FeatureEngine, TechnicalIndicatorCalculator
try:
    from src.models.ml_models import MLModelPredictor
except ImportError:
    from src.models.ml_models_mock import MLModelPredictor
from src.signal_generator.signal_generator import TradingSignalGenerator
from src.risk_manager.risk_manager import RiskManager
from src.utils.types import TimeFrame, Signal, SignalType
from src.utils.logger import setup_logging, TradingLogger


class BacktestEngine:
    """Backtesting engine for trading strategies."""
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage: float = 0.0005,  # 0.05%
    ):
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        self.logger = TradingLogger("backtest")
        
        # Initialize components
        self.feature_engine = FeatureEngine()
        self.indicator_calc = TechnicalIndicatorCalculator()
        
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
    
    def load_historical_data(self, symbol: str, days: int = 30) -> Dict[TimeFrame, pd.DataFrame]:
        """Load historical data for backtesting."""
        # In a real implementation, this would load from exchange API or database
        # For now, we'll create mock data
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate mock OHLCV data
        timeframes = [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.H1]
        data = {}
        
        for tf in timeframes:
            # Calculate number of candles based on timeframe
            if tf == TimeFrame.M1:
                freq = "1min"
                periods = days * 24 * 60
            elif tf == TimeFrame.M5:
                freq = "5min"
                periods = days * 24 * 12
            elif tf == TimeFrame.M15:
                freq = "15min"
                periods = days * 24 * 4
            else:  # 1H
                freq = "1H"
                periods = days * 24
            
            # Create date range
            date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
            
            # Generate realistic price data (SOL around $100-150)
            base_price = 120.0
            price_data = []
            current_price = base_price
            
            for i in range(len(date_range)):
                # Random walk with trend
                change = np.random.normal(0, 0.002)  # 0.2% volatility
                current_price *= (1 + change)
                
                # Create OHLC
                high = current_price * (1 + abs(np.random.normal(0, 0.001)))
                low = current_price * (1 - abs(np.random.normal(0, 0.001)))
                open_price = current_price * (1 + np.random.normal(0, 0.0005))
                close = current_price
                volume = np.random.uniform(1000, 10000)
                
                price_data.append({
                    'timestamp': date_range[i],
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                })
            
            df = pd.DataFrame(price_data)
            df.set_index('timestamp', inplace=True)
            data[tf] = df
        
        return data
    
    def run_backtest(
        self,
        symbol: str,
        data: Dict[TimeFrame, pd.DataFrame],
        primary_timeframe: TimeFrame = TimeFrame.M5,
    ) -> Dict[str, Any]:
        """Run backtesting simulation."""
        
        self.logger.log_system_event(
            event_type="backtest_start",
            component="backtest",
            status="starting",
            details={"symbol": symbol, "data_points": len(data[primary_timeframe])},
        )
        
        # Get primary timeframe data
        df = data[primary_timeframe]
        
        # Process each candle
        for i in range(100, len(df)):  # Start from index 100 to have enough history
            current_time = df.index[i]
            current_price = df.iloc[i]['close']
            
            # Create historical data slice
            historical_data = {
                symbol: {
                    tf: data[tf].iloc[:i+1] for tf in data.keys()
                }
            }
            
            # Generate features
            features = self.feature_engine.generate_features(
                market_data=historical_data,
                symbol=symbol,
                primary_timeframe=primary_timeframe,
            )
            
            if not features:
                continue
            
            # Create mock signal based on simple strategy
            signal = self._generate_mock_signal(
                symbol=symbol,
                current_price=current_price,
                features=features,
                timestamp=current_time,
            )
            
            if signal:
                # Process signal
                self._process_signal(signal, current_time, current_price)
            
            # Update existing positions
            self._update_positions(current_time, current_price)
            
            # Record equity
            self._record_equity(current_time)
        
        # Calculate final results
        results = self._calculate_results()
        
        self.logger.log_system_event(
            event_type="backtest_complete",
            component="backtest",
            status="completed",
            details=results,
        )
        
        return results
    
    def _generate_mock_signal(
        self,
        symbol: str,
        current_price: float,
        features: Dict[str, float],
        timestamp: datetime,
    ) -> Optional[Signal]:
        """Generate mock trading signal based on simple strategy."""
        
        # Simple strategy: EMA crossover + RSI
        ema_8 = features.get('ema_8', current_price)
        ema_21 = features.get('ema_21', current_price)
        rsi = features.get('rsi', 50)
        volume_ratio = features.get('volume_ratio', 1.0)
        
        # Mock confidence calculation
        confidence = 0.0
        signal_type = None
        
        # Bullish conditions
        if (ema_8 > ema_21 and 
            rsi > 30 and rsi < 70 and 
            volume_ratio > 1.2 and
            current_price > ema_8):
            
            signal_type = SignalType.BUY
            confidence = min(0.8, (ema_8 / ema_21 - 1) * 100 + volume_ratio * 0.1)
        
        # Bearish conditions
        elif (ema_8 < ema_21 and 
              rsi > 30 and rsi < 70 and 
              volume_ratio > 1.2 and
              current_price < ema_8):
            
            signal_type = SignalType.SELL
            confidence = min(0.8, (ema_21 / ema_8 - 1) * 100 + volume_ratio * 0.1)
        
        if signal_type and confidence > 0.15:
            # Calculate stop loss and take profit
            atr = features.get('atr_ratio', 0.02) * current_price
            
            if signal_type == SignalType.BUY:
                stop_loss = current_price - (atr * 1.2)
                take_profit = current_price + (atr * 2.0)
            else:
                stop_loss = current_price + (atr * 1.2)
                take_profit = current_price - (atr * 2.0)
            
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                timestamp=timestamp,
                confidence=confidence,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"EMA crossover strategy: confidence {confidence:.3f}",
                features=features,
            )
        
        return None
    
    def _process_signal(self, signal: Signal, current_time: datetime, current_price: float) -> None:
        """Process trading signal."""
        
        # Check if we already have a position
        if self.positions:
            return  # Skip if already in position
        
        # Calculate position size (2% of balance)
        position_size = self.current_balance * 0.02
        quantity = position_size / current_price
        
        # Apply slippage
        if signal.signal_type == SignalType.BUY:
            entry_price = current_price * (1 + self.slippage)
        else:
            entry_price = current_price * (1 - self.slippage)
        
        # Calculate commission
        commission = position_size * self.commission_rate
        
        # Create position
        position = {
            'signal_type': signal.signal_type,
            'entry_time': current_time,
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'commission_paid': commission,
            'unrealized_pnl': 0.0,
        }
        
        self.positions.append(position)
        self.current_balance -= commission
        
        self.logger.log_signal(
            symbol=signal.symbol,
            signal_type=signal.signal_type.value,
            confidence=signal.confidence,
            price=entry_price,
            reasoning=signal.reasoning,
        )
    
    def _update_positions(self, current_time: datetime, current_price: float) -> None:
        """Update existing positions."""
        
        positions_to_close = []
        
        for i, position in enumerate(self.positions):
            # Calculate unrealized PnL
            if position['signal_type'] == SignalType.BUY:
                pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - current_price) * position['quantity']
            
            position['unrealized_pnl'] = pnl
            
            # Check exit conditions
            should_close = False
            close_reason = ""
            
            # Stop loss
            if position['signal_type'] == SignalType.BUY and current_price <= position['stop_loss']:
                should_close = True
                close_reason = "stop_loss"
            elif position['signal_type'] == SignalType.SELL and current_price >= position['stop_loss']:
                should_close = True
                close_reason = "stop_loss"
            
            # Take profit
            elif position['signal_type'] == SignalType.BUY and current_price >= position['take_profit']:
                should_close = True
                close_reason = "take_profit"
            elif position['signal_type'] == SignalType.SELL and current_price <= position['take_profit']:
                should_close = True
                close_reason = "take_profit"
            
            # Time stop (4 hours)
            elif (current_time - position['entry_time']).total_seconds() > 4 * 3600:
                should_close = True
                close_reason = "time_stop"
            
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
        if position['signal_type'] == SignalType.BUY:
            exit_price = close_price * (1 - self.slippage)
        else:
            exit_price = close_price * (1 + self.slippage)
        
        # Calculate final PnL
        if position['signal_type'] == SignalType.BUY:
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        # Calculate commission
        exit_commission = position['quantity'] * exit_price * self.commission_rate
        net_pnl = pnl - position['commission_paid'] - exit_commission
        
        # Update balance
        self.current_balance += net_pnl + (position['quantity'] * position['entry_price'])
        
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
        }
        
        self.trades.append(trade)
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1
        
        self.total_pnl += net_pnl
        
        # Remove position
        del self.positions[position_index]
        
        self.logger.log_position(
            symbol="SOLUSDT",
            side="LONG" if position['signal_type'] == SignalType.BUY else "SHORT",
            size=position['quantity'],
            entry_price=position['entry_price'],
            current_price=exit_price,
            pnl=net_pnl,
            action="CLOSED",
        )
    
    def _record_equity(self, timestamp: datetime) -> None:
        """Record current equity."""
        
        # Calculate total unrealized PnL
        unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions)
        total_equity = self.current_balance + unrealized_pnl
        
        # Update peak equity and drawdown
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
        
        current_drawdown = (self.peak_equity - total_equity) / self.peak_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': self.current_balance,
            'unrealized_pnl': unrealized_pnl,
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
        
        # Risk metrics
        returns = [t['net_pnl'] / self.initial_balance for t in self.trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Duration statistics
        durations = [(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in self.trades]  # hours
        avg_duration = np.mean(durations)
        
        final_balance = self.current_balance
        total_return = (final_balance / self.initial_balance - 1) * 100
        
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
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_duration_hours': avg_duration,
            'total_commission': sum(t['commission'] for t in self.trades),
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted backtest results."""
        
        print("\n" + "="*60)
        print("           BACKTEST RESULTS")
        print("="*60)
        
        if "error" in results:
            print(f"❌ {results['error']}")
            return
        
        print(f"💰 Initial Balance:     ${results['initial_balance']:,.2f}")
        print(f"💰 Final Balance:       ${results['final_balance']:,.2f}")
        print(f"📈 Total Return:        {results['total_return_pct']:+.2f}%")
        print(f"💵 Total P&L:           ${results['total_pnl']:+,.2f}")
        print()
        
        print(f"📊 Total Trades:        {results['total_trades']}")
        print(f"✅ Winning Trades:      {results['winning_trades']}")
        print(f"❌ Losing Trades:       {results['losing_trades']}")
        print(f"🎯 Win Rate:            {results['win_rate']:.1%}")
        print()
        
        print(f"💹 Profit Factor:       {results['profit_factor']:.2f}")
        print(f"📈 Average Win:         ${results['avg_win']:+.2f}")
        print(f"📉 Average Loss:        ${results['avg_loss']:+.2f}")
        print(f"⚠️  Max Drawdown:       {results['max_drawdown']:.1%}")
        print(f"📊 Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print()
        
        print(f"⏱️  Avg Trade Duration:  {results['avg_trade_duration_hours']:.1f} hours")
        print(f"💸 Total Commission:    ${results['total_commission']:,.2f}")
        
        print("\n" + "="*60)
        
        # Performance assessment
        if results['win_rate'] >= 0.55 and results['profit_factor'] >= 1.3:
            print("✅ GOOD PERFORMANCE - Strategy meets target criteria")
        elif results['win_rate'] >= 0.50 and results['profit_factor'] >= 1.0:
            print("⚠️  MODERATE PERFORMANCE - Consider optimization")
        else:
            print("❌ POOR PERFORMANCE - Strategy needs significant improvement")


@click.command()
@click.option("--symbol", default="SOLUSDT", help="Trading symbol")
@click.option("--days", default=30, help="Number of days to backtest")
@click.option("--balance", default=10000.0, help="Initial balance")
@click.option("--commission", default=0.001, help="Commission rate")
@click.option("--config", help="Config file path")
def main(symbol: str, days: int, balance: float, commission: float, config: str):
    """Run backtesting for trading strategy."""
    
    # Setup logging
    setup_logging(log_level="INFO", log_format="text")
    
    print(f"🔬 Starting Backtest for {symbol}")
    print(f"📅 Period: {days} days")
    print(f"💰 Initial Balance: ${balance:,.2f}")
    print(f"💸 Commission Rate: {commission:.3%}")
    print()
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_balance=balance,
        commission_rate=commission,
    )
    
    # Load historical data
    print("📊 Loading historical data...")
    data = engine.load_historical_data(symbol, days)
    print(f"✅ Loaded {len(data[TimeFrame.M5])} candles")
    
    # Run backtest
    print("🚀 Running backtest...")
    results = engine.run_backtest(symbol, data)
    
    # Print results
    engine.print_results(results)
    
    # Show sample trades
    if engine.trades:
        print("\n📋 Sample Trades (Last 5):")
        print("-" * 80)
        for trade in engine.trades[-5:]:
            duration = trade['duration'].total_seconds() / 3600
            print(f"{trade['entry_time'].strftime('%Y-%m-%d %H:%M')} | "
                  f"{trade['signal_type'].value:4} | "
                  f"${trade['entry_price']:7.2f} → ${trade['exit_price']:7.2f} | "
                  f"PnL: ${trade['net_pnl']:+7.2f} | "
                  f"{duration:4.1f}h | {trade['close_reason']}")


if __name__ == "__main__":
    main()