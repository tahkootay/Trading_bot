#!/usr/bin/env python3
"""Enhanced backtesting script using real Bybit data."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import click
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.feature_engine.technical_indicators import FeatureEngine, TechnicalIndicatorCalculator
from src.utils.types import TimeFrame, Signal, SignalType
from src.utils.logger import setup_logging, TradingLogger


class EnhancedBacktestEngine:
    """Enhanced backtesting engine using real market data."""
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage: float = 0.0005,  # 0.05%
    ):
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        self.logger = TradingLogger("enhanced_backtest")
        
        # Initialize components
        self.feature_engine = FeatureEngine()
        self.indicator_calc = TechnicalIndicatorCalculator()
        
        # Backtest state
        self.current_balance = initial_balance
        self.positions: List[Dict] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.signals_generated: List[Dict] = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_balance
        
        # Strategy parameters (SOL-optimized)
        self.strategy_params = {
            'min_confidence': 0.15,
            'min_volume_ratio': 1.2,
            'min_adx': 20.0,
            'atr_sl_multiplier': 1.2,
            'atr_tp_multiplier': 2.0,
            'max_position_time_hours': 4,
            'position_size_pct': 0.02,  # 2% of balance
        }
    
    def load_data_from_files(self, symbol: str, data_dir: str = "data") -> Dict[TimeFrame, pd.DataFrame]:
        """Load data from CSV files."""
        data = {}
        data_path = Path(data_dir)
        
        timeframe_map = {
            "1m": TimeFrame.M1,
            "5m": TimeFrame.M5,
            "15m": TimeFrame.M15,
            "1h": TimeFrame.H1,
        }
        
        for tf_str, tf_enum in timeframe_map.items():
            # First try to find real data files
            real_pattern = f"{symbol}_{tf_str}_real_*.csv"
            real_files = list(data_path.glob(real_pattern))
            
            if real_files:
                # Use the most recent real data file
                latest_file = max(real_files, key=lambda x: x.stat().st_mtime)
                print(f"📊 Loading {tf_str} REAL data from {latest_file.name}")
                
                df = pd.read_csv(latest_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                data[tf_enum] = df
                print(f"  ✅ Loaded {len(df)} REAL candles for {tf_str}")
            else:
                # Fallback to testnet data
                file_pattern = f"{symbol}_{tf_str}_*.csv"
                files = list(data_path.glob(file_pattern))
                
                if files:
                    # Use the most recent file
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    print(f"📊 Loading {tf_str} testnet data from {latest_file.name}")
                    
                    df = pd.read_csv(latest_file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    
                    data[tf_enum] = df
                    print(f"  ✅ Loaded {len(df)} testnet candles for {tf_str}")
                else:
                    print(f"  ❌ No data file found for {tf_str}")
        
        return data
    
    def run_backtest(
        self,
        symbol: str,
        data: Dict[TimeFrame, pd.DataFrame],
        primary_timeframe: TimeFrame = TimeFrame.M5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Run enhanced backtesting simulation."""
        
        self.logger.log_system_event(
            event_type="enhanced_backtest_start",
            component="enhanced_backtest",
            status="starting",
            details={"symbol": symbol, "timeframes": list(data.keys())},
        )
        
        # Get primary timeframe data
        if primary_timeframe not in data:
            raise ValueError(f"Primary timeframe {primary_timeframe.value} not available")
        
        df_primary = data[primary_timeframe]
        
        # Filter date range if specified
        if start_date:
            df_primary = df_primary[df_primary.index >= start_date]
        if end_date:
            df_primary = df_primary[df_primary.index <= end_date]
        
        print(f"🔬 Backtesting on {len(df_primary)} candles ({primary_timeframe.value})")
        print(f"📅 Period: {df_primary.index[0]} → {df_primary.index[-1]}")
        print()
        
        # Ensure we have enough data for indicators
        min_periods = 100
        if len(df_primary) < min_periods:
            raise ValueError(f"Not enough data. Need at least {min_periods} candles, got {len(df_primary)}")
        
        # Process each candle
        for i in range(min_periods, len(df_primary)):
            if i % 500 == 0:
                print(f"📈 Processing candle {i}/{len(df_primary)} ({i/len(df_primary)*100:.1f}%)")
            
            current_time = df_primary.index[i]
            current_candle = df_primary.iloc[i]
            current_price = current_candle['close']
            
            # Create historical data slice up to current point
            historical_data = {
                symbol: {}
            }
            
            for tf, tf_data in data.items():
                # Get data up to current time
                historical_slice = tf_data[tf_data.index <= current_time]
                if len(historical_slice) > 0:
                    historical_data[symbol][tf] = historical_slice
            
            # Generate features
            try:
                features = self.feature_engine.generate_features(
                    market_data=historical_data,
                    symbol=symbol,
                    primary_timeframe=primary_timeframe,
                )
            except Exception as e:
                # Skip this candle if feature generation fails
                continue
            
            if not features:
                continue
            
            # Generate signal using enhanced strategy
            signal = self._generate_enhanced_signal(
                symbol=symbol,
                current_price=current_price,
                current_candle=current_candle,
                features=features,
                timestamp=current_time,
            )
            
            if signal:
                # Record signal for analysis
                self.signals_generated.append({
                    'timestamp': current_time,
                    'signal': signal,
                    'price': current_price,
                    'features': features.copy(),
                })
                
                # Process signal
                self._process_signal(signal, current_time, current_price)
            
            # Update existing positions
            self._update_positions(current_time, current_price)
            
            # Record equity
            self._record_equity(current_time)
        
        # Close any remaining positions
        self._close_all_positions(df_primary.index[-1], df_primary.iloc[-1]['close'])
        
        # Calculate final results
        results = self._calculate_enhanced_results()
        
        self.logger.log_system_event(
            event_type="enhanced_backtest_complete",
            component="enhanced_backtest",
            status="completed",
            details=results,
        )
        
        return results
    
    def _generate_enhanced_signal(
        self,
        symbol: str,
        current_price: float,
        current_candle: pd.Series,
        features: Dict[str, float],
        timestamp: datetime,
    ) -> Optional[Signal]:
        """Generate enhanced trading signal using multiple criteria."""
        
        # Get key features
        ema_8 = features.get('ema_8', current_price)
        ema_21 = features.get('ema_21', current_price)
        ema_55 = features.get('ema_55', current_price)
        rsi = features.get('rsi', 50)
        adx = features.get('adx', 0)
        macd = features.get('macd', 0)
        macd_signal = features.get('macd_signal', 0)
        volume_ratio = features.get('volume_ratio', 1.0)
        price_vs_vwap = features.get('price_vs_vwap', 0)
        atr_ratio = features.get('atr_ratio', 0.02)
        bb_position = features.get('bb_position', 0.5)
        
        # Check minimum requirements
        if adx < self.strategy_params['min_adx']:
            return None
        
        if volume_ratio < self.strategy_params['min_volume_ratio']:
            return None
        
        # Calculate confidence score
        confidence = 0.0
        signal_type = None
        
        # Bullish conditions
        bullish_score = 0.0
        bullish_conditions = [
            ema_8 > ema_21,  # Short EMA above medium
            ema_21 > ema_55,  # Medium EMA above long
            current_price > ema_8,  # Price above short EMA
            rsi > 40 and rsi < 75,  # RSI in reasonable range
            macd > macd_signal,  # MACD bullish
            price_vs_vwap > -0.01,  # Price near or above VWAP
            volume_ratio > 1.2,  # Good volume
            bb_position > 0.2 and bb_position < 0.8,  # Not extreme BB position
        ]
        
        bullish_score = sum(bullish_conditions) / len(bullish_conditions)
        
        # Bearish conditions
        bearish_score = 0.0
        bearish_conditions = [
            ema_8 < ema_21,  # Short EMA below medium
            ema_21 < ema_55,  # Medium EMA below long
            current_price < ema_8,  # Price below short EMA
            rsi > 25 and rsi < 60,  # RSI in reasonable range
            macd < macd_signal,  # MACD bearish
            price_vs_vwap < 0.01,  # Price near or below VWAP
            volume_ratio > 1.2,  # Good volume
            bb_position > 0.2 and bb_position < 0.8,  # Not extreme BB position
        ]
        
        bearish_score = sum(bearish_conditions) / len(bearish_conditions)
        
        # Determine signal
        if bullish_score > 0.6 and bullish_score > bearish_score + 0.2:
            signal_type = SignalType.BUY
            confidence = bullish_score
        elif bearish_score > 0.6 and bearish_score > bullish_score + 0.2:
            signal_type = SignalType.SELL
            confidence = bearish_score
        
        # Check minimum confidence
        if not signal_type or confidence < self.strategy_params['min_confidence']:
            return None
        
        # Calculate stop loss and take profit
        atr = atr_ratio * current_price
        
        if signal_type == SignalType.BUY:
            stop_loss = current_price - (atr * self.strategy_params['atr_sl_multiplier'])
            take_profit = current_price + (atr * self.strategy_params['atr_tp_multiplier'])
        else:
            stop_loss = current_price + (atr * self.strategy_params['atr_sl_multiplier'])
            take_profit = current_price - (atr * self.strategy_params['atr_tp_multiplier'])
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            timestamp=timestamp,
            confidence=confidence,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"Enhanced strategy: {signal_type.value} confidence {confidence:.3f}, "
                     f"Bullish: {bullish_score:.2f}, Bearish: {bearish_score:.2f}",
            features=features,
        )
    
    def _process_signal(self, signal: Signal, current_time: datetime, current_price: float) -> None:
        """Process trading signal."""
        
        # Check if we already have a position
        if self.positions:
            return  # Skip if already in position
        
        # Calculate position size
        position_value = self.current_balance * self.strategy_params['position_size_pct']
        quantity = position_value / current_price
        
        # Apply slippage
        if signal.signal_type == SignalType.BUY:
            entry_price = current_price * (1 + self.slippage)
        else:
            entry_price = current_price * (1 - self.slippage)
        
        # Calculate commission
        commission = position_value * self.commission_rate
        
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
            'confidence': signal.confidence,
            'reasoning': signal.reasoning,
        }
        
        self.positions.append(position)
        self.current_balance -= commission
    
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
            
            # Time stop
            elif (current_time - position['entry_time']).total_seconds() > self.strategy_params['max_position_time_hours'] * 3600:
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
            'reasoning': position['reasoning'],
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
        
        current_drawdown = (self.peak_equity - total_equity) / self.peak_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': self.current_balance,
            'position_value': position_value,
            'unrealized_pnl': unrealized_pnl,
            'total_equity': total_equity,
            'drawdown': current_drawdown,
        })
    
    def _calculate_enhanced_results(self) -> Dict[str, Any]:
        """Calculate enhanced backtest results."""
        
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
        
        # Enhanced metrics
        if returns:
            negative_returns = [r for r in returns if r < 0]
            sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(252) if negative_returns and np.std(negative_returns) > 0 else 0
        else:
            sortino_ratio = 0
        
        calmar_ratio = (np.mean(returns) * 252) / self.max_drawdown if self.max_drawdown > 0 else 0
        
        # Duration statistics
        durations = [(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in self.trades]  # hours
        avg_duration = np.mean(durations)
        
        # Movement capture analysis (SOL-specific)
        movements_captured = []
        for trade in self.trades:
            if trade['signal_type'] == SignalType.BUY:
                movement = trade['exit_price'] - trade['entry_price']
            else:
                movement = trade['entry_price'] - trade['exit_price']
            movements_captured.append(movement)
        
        movements_over_2 = len([m for m in movements_captured if m >= 2.0])
        avg_movement = np.mean(movements_captured) if movements_captured else 0
        
        final_balance = self.current_balance
        total_return = (final_balance / self.initial_balance - 1) * 100
        
        # Confidence analysis
        high_conf_trades = [t for t in self.trades if t['confidence'] >= 0.7]
        high_conf_win_rate = len([t for t in high_conf_trades if t['net_pnl'] > 0]) / len(high_conf_trades) if high_conf_trades else 0
        
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
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'avg_trade_duration_hours': avg_duration,
            'total_commission': sum(t['commission'] for t in self.trades),
            'signals_generated': len(self.signals_generated),
            'signal_to_trade_ratio': self.total_trades / len(self.signals_generated) if self.signals_generated else 0,
            'movements_over_2_usd': movements_over_2,
            'avg_movement_captured': avg_movement,
            'high_confidence_trades': len(high_conf_trades),
            'high_confidence_win_rate': high_conf_win_rate,
        }
        
        return results
    
    def print_enhanced_results(self, results: Dict[str, Any]) -> None:
        """Print formatted enhanced backtest results."""
        
        print("\n" + "="*70)
        print("           ENHANCED BACKTEST RESULTS (SOL/USDT)")
        print("="*70)
        
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
        print(f"Signals Generated:    {results['signals_generated']}")
        print(f"Signal→Trade Ratio:   {results['signal_to_trade_ratio']:.1%}")
        print()
        
        # Risk Metrics
        print("⚠️  RISK METRICS")
        print("-" * 30)
        print(f"Max Drawdown:         {results['max_drawdown']:.1%}")
        print(f"Profit Factor:        {results['profit_factor']:.2f}")
        print(f"Sharpe Ratio:         {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:        {results['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:         {results['calmar_ratio']:.2f}")
        print()
        
        # SOL-Specific Metrics
        print("🎯 SOL-SPECIFIC METRICS")
        print("-" * 30)
        print(f"Average Win:          ${results['avg_win']:+.2f}")
        print(f"Average Loss:         ${results['avg_loss']:+.2f}")
        print(f"Avg Movement Captured: ${results['avg_movement_captured']:+.2f}")
        print(f"Movements ≥$2:        {results['movements_over_2_usd']}")
        print(f"Avg Trade Duration:   {results['avg_trade_duration_hours']:.1f} hours")
        print()
        
        # Confidence Analysis
        print("🧠 CONFIDENCE ANALYSIS")
        print("-" * 30)
        print(f"High Confidence Trades: {results['high_confidence_trades']}")
        print(f"High Conf Win Rate:     {results['high_confidence_win_rate']:.1%}")
        print()
        
        # Costs
        print("💸 COSTS")
        print("-" * 30)
        print(f"Total Commission:     ${results['total_commission']:,.2f}")
        print(f"Commission % of P&L:  {abs(results['total_commission']/results['total_pnl']*100) if results['total_pnl'] != 0 else 0:.1f}%")
        
        print("\n" + "="*70)
        
        # Performance Assessment
        meets_targets = (
            results['win_rate'] >= 0.55 and
            results['profit_factor'] >= 1.3 and
            results['max_drawdown'] <= 0.10 and
            results['movements_over_2_usd'] >= results['total_trades'] * 0.3  # 30% of trades capture $2+
        )
        
        if meets_targets:
            print("✅ EXCELLENT - Strategy meets all target criteria!")
        elif results['win_rate'] >= 0.50 and results['profit_factor'] >= 1.0:
            print("⚠️  MODERATE - Strategy shows promise but needs optimization")
        else:
            print("❌ NEEDS WORK - Strategy requires significant improvement")
        
        print("="*70)


@click.command()
@click.option("--symbol", default="SOLUSDT", help="Trading symbol")
@click.option("--data-dir", default="data", help="Data directory")
@click.option("--balance", default=10000.0, help="Initial balance")
@click.option("--commission", default=0.001, help="Commission rate")
@click.option("--days", help="Limit backtest to recent N days")
@click.option("--save-results", is_flag=True, help="Save detailed results to file")
def main(symbol: str, data_dir: str, balance: float, commission: float, days: Optional[int], save_results: bool):
    """Run enhanced backtesting using real Bybit data."""
    
    # Setup logging
    setup_logging(log_level="INFO", log_format="text")
    
    print(f"🔬 Enhanced Backtest for {symbol}")
    print(f"💰 Initial Balance: ${balance:,.2f}")
    print(f"💸 Commission Rate: {commission:.3%}")
    print(f"📁 Data Directory: {data_dir}")
    print()
    
    # Initialize backtest engine
    engine = EnhancedBacktestEngine(
        initial_balance=balance,
        commission_rate=commission,
    )
    
    # Load data
    print("📊 Loading market data...")
    try:
        data = engine.load_data_from_files(symbol, data_dir)
        
        if not data:
            print("❌ No data files found. Please run data collection first:")
            print("   python scripts/collect_data.py --symbol SOLUSDT --days 30")
            return
        
        print(f"✅ Loaded data for {len(data)} timeframes")
        
        # Set date range if specified
        start_date = None
        if days:
            latest_data = max(data.values(), key=len)
            end_date = latest_data.index[-1]
            start_date = end_date - timedelta(days=int(days))
            print(f"📅 Limiting to last {days} days: {start_date.date()} → {end_date.date()}")
        
        # Run backtest
        print("🚀 Running enhanced backtest...")
        results = engine.run_backtest(
            symbol=symbol, 
            data=data,
            start_date=start_date,
        )
        
        # Print results
        engine.print_enhanced_results(results)
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure Output directory exists
            Path("../Output").mkdir(exist_ok=True)
            results_file = f"../Output/backtest_results_{symbol}_{timestamp}.json"
            
            # Add trade details
            detailed_results = results.copy()
            detailed_results['trades'] = engine.trades
            detailed_results['equity_curve'] = engine.equity_curve
            detailed_results['signals'] = [
                {
                    'timestamp': s['timestamp'].isoformat(),
                    'signal_type': s['signal'].signal_type.value,
                    'confidence': s['signal'].confidence,
                    'price': s['price'],
                } for s in engine.signals_generated
            ]
            
            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {results_file}")
        
        # Show sample trades
        if engine.trades:
            print(f"\n📋 Sample Trades (Last 5 of {len(engine.trades)}):")
            print("-" * 100)
            print(f"{'Date':>16} {'Type':>4} {'Entry':>8} {'Exit':>8} {'P&L':>8} {'Duration':>8} {'Reason':>12} {'Conf':>5}")
            print("-" * 100)
            
            for trade in engine.trades[-5:]:
                duration = trade['duration'].total_seconds() / 3600
                print(f"{trade['entry_time'].strftime('%m-%d %H:%M'):>16} "
                      f"{trade['signal_type'].value:>4} "
                      f"${trade['entry_price']:>7.2f} "
                      f"${trade['exit_price']:>7.2f} "
                      f"${trade['net_pnl']:>+7.2f} "
                      f"{duration:>7.1f}h "
                      f"{trade['close_reason']:>12} "
                      f"{trade['confidence']:>4.2f}")
    
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()