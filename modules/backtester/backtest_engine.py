"""
Core backtesting engine that executes strategies on historical data.
"""

import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd

from .strategy_base import StrategyBase, BacktestConfig, MarketData, Position, Signal, TradeSignal


@dataclass
class Trade:
    """Completed trade record."""
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float
    commission: float
    duration_minutes: int
    entry_reason: str
    exit_reason: str


@dataclass
class EquityPoint:
    """Point in equity curve."""
    timestamp: datetime
    equity: float
    drawdown: float


@dataclass
class DailyReturn:
    """Daily return data."""
    date: datetime
    return_pct: float
    cumulative_return: float


@dataclass
class BacktestState:
    """Current state of backtest."""
    current_time: datetime
    capital: float
    equity: float
    max_equity: float
    position: Position
    trades: List[Trade]
    equity_curve: List[EquityPoint]
    pending_orders: List[Dict[str, Any]]


class BacktestEngine:
    """Core backtesting engine."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.state: Optional[BacktestState] = None
        self.strategy: Optional[StrategyBase] = None
    
    async def run_backtest(
        self,
        strategy: StrategyBase,
        data: pd.DataFrame
    ) -> 'BacktestResults':
        """
        Run backtest with given strategy and data.
        
        Args:
            strategy: Strategy instance to test
            data: Historical price data
            
        Returns:
            BacktestResults with all metrics and trades
        """
        
        # Initialize
        self.strategy = strategy
        self.strategy.reset()
        
        self._initialize_state(data.iloc[0]['timestamp'])
        
        # Process each bar
        for idx, row in data.iterrows():
            market_data = MarketData(
                timestamp=row['timestamp'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                symbol=row.get('symbol', '')
            )
            
            await self._process_bar(market_data)
        
        # Close any open position
        if self.state.position.direction != "NONE":
            await self._close_position(data.iloc[-1], "End of backtest")
        
        # Calculate final results
        results = self._calculate_results(
            start_date=data.iloc[0]['timestamp'],
            end_date=data.iloc[-1]['timestamp']
        )
        
        return results
    
    def _initialize_state(self, start_time: datetime) -> None:
        """Initialize backtest state."""
        
        self.state = BacktestState(
            current_time=start_time,
            capital=self.config.initial_capital,
            equity=self.config.initial_capital,
            max_equity=self.config.initial_capital,
            position=Position(
                direction="NONE",
                quantity=0.0,
                entry_price=0.0,
                entry_time=start_time,
                unrealized_pnl=0.0,
                duration_minutes=0
            ),
            trades=[],
            equity_curve=[],
            pending_orders=[]
        )
    
    async def _process_bar(self, data: MarketData) -> None:
        """Process single market data bar."""
        
        self.state.current_time = data.timestamp
        
        # Update position PnL
        self._update_position_pnl(data)
        
        # Update position duration
        if self.state.position.direction != "NONE":
            duration = (data.timestamp - self.state.position.entry_time).total_seconds() / 60
            self.state.position.duration_minutes = int(duration)
        
        # Get strategy signal
        signal = self.strategy.on_bar(data, self.state.position)
        
        # Execute signal
        await self._execute_signal(data, signal)
        
        # Record equity curve point
        drawdown = (self.state.max_equity - self.state.equity) / self.state.max_equity * 100
        if self.state.equity > self.state.max_equity:
            self.state.max_equity = self.state.equity
            drawdown = 0.0
        
        self.state.equity_curve.append(EquityPoint(
            timestamp=data.timestamp,
            equity=self.state.equity,
            drawdown=drawdown
        ))
    
    def _update_position_pnl(self, data: MarketData) -> None:
        """Update unrealized PnL for open position."""
        
        if self.state.position.direction == "NONE":
            self.state.position.unrealized_pnl = 0.0
            self.state.equity = self.state.capital
        else:
            if self.state.position.direction == "LONG":
                pnl = (data.close - self.state.position.entry_price) * self.state.position.quantity
            else:  # SHORT
                pnl = (self.state.position.entry_price - data.close) * self.state.position.quantity
            
            self.state.position.unrealized_pnl = pnl
            self.state.equity = self.state.capital + pnl
    
    async def _execute_signal(self, data: MarketData, signal: TradeSignal) -> None:
        """Execute trading signal."""
        
        if signal.signal == Signal.HOLD:
            return
        
        elif signal.signal == Signal.BUY:
            if self.state.position.direction == "NONE":
                await self._open_position(data, "LONG", signal)
        
        elif signal.signal == Signal.SELL:
            if self.state.position.direction == "NONE":
                await self._open_position(data, "SHORT", signal)
        
        elif signal.signal == Signal.CLOSE_LONG:
            if self.state.position.direction == "LONG":
                await self._close_position(data, signal.reason or "Signal close")
        
        elif signal.signal == Signal.CLOSE_SHORT:
            if self.state.position.direction == "SHORT":
                await self._close_position(data, signal.reason or "Signal close")
    
    async def _open_position(self, data: MarketData, direction: str, signal: TradeSignal) -> None:
        """Open new position."""
        
        # Calculate position size
        if signal.quantity > 0:
            quantity = signal.quantity
        else:
            quantity = self.strategy.get_position_size(data, signal.signal, self.state.capital)
        
        # Apply position size limits
        max_quantity = self.state.capital * self.config.max_position_size
        quantity = min(quantity, max_quantity)
        
        if quantity <= 0:
            return
        
        # Calculate entry price (with slippage)
        entry_price = data.close * (1 + self.config.slippage_rate)
        if direction == "SHORT":
            entry_price = data.close * (1 - self.config.slippage_rate)
        
        # Calculate number of shares/contracts from dollar amount
        num_shares = quantity / entry_price
        
        # Calculate commission based on dollar value
        commission = quantity * self.config.commission_rate
        
        # Check if we have enough capital
        required_capital = quantity + commission
        if required_capital > self.state.capital:
            return  # Not enough capital
        
        # Open position
        self.state.position = Position(
            direction=direction,
            quantity=num_shares,  # Number of shares/contracts
            entry_price=entry_price,
            entry_time=data.timestamp,
            unrealized_pnl=0.0,
            duration_minutes=0
        )
        
        # Update capital
        self.state.capital -= commission
        
        # Notify strategy
        self.strategy.on_trade_opened(
            data=data,
            entry_price=entry_price,
            quantity=self.state.position.quantity,
            direction=direction
        )
    
    async def _close_position(self, data: MarketData, reason: str) -> None:
        """Close current position."""
        
        if self.state.position.direction == "NONE":
            return
        
        # Calculate exit price (with slippage)
        exit_price = data.close * (1 - self.config.slippage_rate)
        if self.state.position.direction == "SHORT":
            exit_price = data.close * (1 + self.config.slippage_rate)
        
        # Calculate PnL
        if self.state.position.direction == "LONG":
            pnl = (exit_price - self.state.position.entry_price) * self.state.position.quantity
        else:  # SHORT
            pnl = (self.state.position.entry_price - exit_price) * self.state.position.quantity
        
        # Calculate commission
        position_value = self.state.position.quantity * exit_price
        commission = position_value * self.config.commission_rate
        
        # Net PnL after commission
        net_pnl = pnl - commission
        
        # Update capital
        self.state.capital += net_pnl
        
        # Record trade
        duration = int((data.timestamp - self.state.position.entry_time).total_seconds() / 60)
        
        trade = Trade(
            entry_time=self.state.position.entry_time,
            exit_time=data.timestamp,
            direction=self.state.position.direction,
            entry_price=self.state.position.entry_price,
            exit_price=exit_price,
            quantity=self.state.position.quantity,
            pnl=net_pnl,
            commission=commission,
            duration_minutes=duration,
            entry_reason="Strategy signal",
            exit_reason=reason
        )
        
        self.state.trades.append(trade)
        
        # Notify strategy
        self.strategy.on_trade_closed(
            data=data,
            exit_price=exit_price,
            pnl=net_pnl,
            direction=self.state.position.direction
        )
        
        # Clear position
        self.state.position = Position(
            direction="NONE",
            quantity=0.0,
            entry_price=0.0,
            entry_time=data.timestamp,
            unrealized_pnl=0.0,
            duration_minutes=0
        )
    
    def _calculate_results(self, start_date: datetime, end_date: datetime) -> 'BacktestResults':
        """Calculate final backtest results and metrics."""
        
        from .result_formatter import BacktestResults
        
        # Calculate daily returns
        daily_returns = self._calculate_daily_returns()
        
        # Performance metrics
        total_return_pct = ((self.state.capital - self.config.initial_capital) / 
                           self.config.initial_capital * 100)
        
        # Trade statistics
        trades = self.state.trades
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_trade_pnl = sum(t.pnl for t in trades) / len(trades) if trades else 0
        best_trade = max(t.pnl for t in trades) if trades else 0
        worst_trade = min(t.pnl for t in trades) if trades else 0
        
        # Risk metrics
        returns_series = [dr.return_pct for dr in daily_returns]
        avg_return = sum(returns_series) / len(returns_series) if returns_series else 0
        
        # Standard deviation of returns
        if len(returns_series) > 1:
            variance = sum((r - avg_return) ** 2 for r in returns_series) / (len(returns_series) - 1)
            std_dev = variance ** 0.5
            sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Sortino ratio (downside deviation)
        negative_returns = [r for r in returns_series if r < 0]
        if negative_returns:
            downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
            downside_deviation = downside_variance ** 0.5
            sortino_ratio = avg_return / downside_deviation if downside_deviation > 0 else 0
        else:
            sortino_ratio = 0
        
        # Profit factor
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Maximum drawdown
        max_drawdown = max(ep.drawdown for ep in self.state.equity_curve) if self.state.equity_curve else 0
        
        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in trades:
            if trade.pnl < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        performance_metrics = {
            'total_return_pct': total_return_pct,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'best_trade_pnl': best_trade,
            'worst_trade_pnl': worst_trade,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'max_consecutive_losses': max_consecutive_losses,
            'total_commission': sum(t.commission for t in trades),
            'avg_trade_duration_minutes': sum(t.duration_minutes for t in trades) / len(trades) if trades else 0
        }
        
        return BacktestResults(
            strategy_name=self.strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.config.initial_capital,
            final_capital=self.state.capital,
            performance_metrics=performance_metrics,
            trades=trades,
            equity_curve=self.state.equity_curve,
            daily_returns=daily_returns
        )
    
    def _calculate_daily_returns(self) -> List[DailyReturn]:
        """Calculate daily returns from equity curve."""
        
        if not self.state.equity_curve:
            return []
        
        daily_returns = []
        
        # Group equity points by date
        daily_equity = {}
        for point in self.state.equity_curve:
            date = point.timestamp.date()
            if date not in daily_equity:
                daily_equity[date] = []
            daily_equity[date].append(point.equity)
        
        # Calculate daily returns
        dates = sorted(daily_equity.keys())
        prev_equity = self.config.initial_capital
        cumulative_return = 0.0
        
        for date in dates:
            final_equity = daily_equity[date][-1]  # End of day equity
            
            daily_return = ((final_equity - prev_equity) / prev_equity * 100) if prev_equity > 0 else 0
            cumulative_return += daily_return
            
            daily_returns.append(DailyReturn(
                date=datetime.combine(date, datetime.min.time()),
                return_pct=daily_return,
                cumulative_return=cumulative_return
            ))
            
            prev_equity = final_equity
        
        return daily_returns