"""
Result formatting and validation for backtest outputs.
Ensures consistent output format for report generation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from .backtest_engine import Trade, EquityPoint, DailyReturn


@dataclass
class BacktestResults:
    """Complete backtest results structure."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    performance_metrics: Dict[str, float]
    trades: List[Trade]
    equity_curve: List[EquityPoint]
    daily_returns: List[DailyReturn]


class ResultFormatter:
    """Formats and validates backtest results."""
    
    def format_results(self, results: BacktestResults) -> BacktestResults:
        """Format and validate backtest results."""
        
        # Validate results
        self._validate_results(results)
        
        # Round numeric values for cleaner output
        results.performance_metrics = self._round_metrics(results.performance_metrics)
        
        return results
    
    def _validate_results(self, results: BacktestResults) -> None:
        """Validate backtest results for consistency."""
        
        errors = []
        
        # Basic validation
        if results.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        if results.start_date >= results.end_date:
            errors.append("Start date must be before end date")
        
        # Trade validation
        for i, trade in enumerate(results.trades):
            if trade.entry_time >= trade.exit_time:
                errors.append(f"Trade {i}: entry time must be before exit time")
            
            if trade.quantity <= 0:
                errors.append(f"Trade {i}: quantity must be positive")
            
            if trade.entry_price <= 0 or trade.exit_price <= 0:
                errors.append(f"Trade {i}: prices must be positive")
        
        # Equity curve validation
        if len(results.equity_curve) == 0:
            errors.append("Equity curve cannot be empty")
        
        prev_timestamp = None
        for i, point in enumerate(results.equity_curve):
            if prev_timestamp and point.timestamp <= prev_timestamp:
                errors.append(f"Equity curve point {i}: timestamps must be ascending")
            
            if point.equity < 0:
                errors.append(f"Equity curve point {i}: equity cannot be negative")
            
            if point.drawdown < 0:
                errors.append(f"Equity curve point {i}: drawdown cannot be negative")
            
            prev_timestamp = point.timestamp
        
        if errors:
            raise ValueError(f"Backtest validation failed: {'; '.join(errors)}")
    
    def _round_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Round metrics to appropriate decimal places."""
        
        rounded = {}
        
        for key, value in metrics.items():
            if 'pct' in key or 'rate' in key or 'ratio' in key:
                # Percentages and ratios to 2 decimal places
                rounded[key] = round(value, 2)
            elif 'pnl' in key or 'profit' in key or 'loss' in key or 'commission' in key:
                # Money values to 2 decimal places
                rounded[key] = round(value, 2)
            elif 'trades' in key or 'consecutive' in key:
                # Count values as integers
                rounded[key] = int(value)
            elif 'minutes' in key:
                # Duration to 1 decimal place
                rounded[key] = round(value, 1)
            else:
                # Default to 4 decimal places
                rounded[key] = round(value, 4)
        
        return rounded
    
    def to_dict(self, results: BacktestResults) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        
        return {
            'strategy_info': {
                'name': results.strategy_name,
                'start_date': results.start_date.isoformat(),
                'end_date': results.end_date.isoformat(),
                'duration_days': (results.end_date - results.start_date).days,
                'initial_capital': results.initial_capital,
                'final_capital': results.final_capital
            },
            'performance_metrics': results.performance_metrics,
            'trade_summary': {
                'total_trades': len(results.trades),
                'trades': [
                    {
                        'entry_time': trade.entry_time.isoformat(),
                        'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                        'direction': trade.direction,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'quantity': trade.quantity,
                        'pnl': trade.pnl,
                        'commission': trade.commission,
                        'duration_minutes': trade.duration_minutes,
                        'entry_reason': trade.entry_reason,
                        'exit_reason': trade.exit_reason
                    }
                    for trade in results.trades
                ]
            },
            'equity_curve': [
                {
                    'timestamp': point.timestamp.isoformat(),
                    'equity': point.equity,
                    'drawdown': point.drawdown
                }
                for point in results.equity_curve
            ],
            'daily_returns': [
                {
                    'date': point.date.isoformat(),
                    'return_pct': point.return_pct,
                    'cumulative_return': point.cumulative_return
                }
                for point in results.daily_returns
            ]
        }
    
    def save_json(self, results: BacktestResults, file_path: str) -> None:
        """Save results to JSON file."""
        
        results_dict = self.to_dict(results)
        
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
    
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """Load results from JSON file."""
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def generate_summary_text(self, results: BacktestResults) -> str:
        """Generate text summary of backtest results."""
        
        metrics = results.performance_metrics
        duration_days = (results.end_date - results.start_date).days
        
        summary = f"""
BACKTEST SUMMARY: {results.strategy_name}
{'=' * 50}

PERIOD: {results.start_date.date()} to {results.end_date.date()} ({duration_days} days)

CAPITAL:
  Initial: ${results.initial_capital:,.2f}
  Final: ${results.final_capital:,.2f}
  Total Return: {metrics['total_return_pct']:.2f}%
  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%

TRADES:
  Total: {metrics['total_trades']}
  Winning: {metrics['winning_trades']} ({metrics['win_rate_pct']:.2f}%)
  Losing: {metrics['losing_trades']}
  Average P&L: ${metrics['avg_trade_pnl']:.2f}
  Best Trade: ${metrics['best_trade_pnl']:.2f}
  Worst Trade: ${metrics['worst_trade_pnl']:.2f}
  Avg Duration: {metrics['avg_trade_duration_minutes']:.1f} minutes

RISK METRICS:
  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
  Sortino Ratio: {metrics['sortino_ratio']:.3f}
  Profit Factor: {metrics['profit_factor']:.3f}
  Max Consecutive Losses: {metrics['max_consecutive_losses']}

COSTS:
  Total Commission: ${metrics['total_commission']:.2f}
"""
        
        return summary
    
    def generate_trade_analysis(self, results: BacktestResults) -> Dict[str, Any]:
        """Generate detailed trade analysis."""
        
        if not results.trades:
            return {'message': 'No trades executed'}
        
        trades = results.trades
        
        # Trade duration analysis
        durations = [t.duration_minutes for t in trades]
        duration_stats = {
            'min_minutes': min(durations),
            'max_minutes': max(durations),
            'avg_minutes': sum(durations) / len(durations),
            'median_minutes': sorted(durations)[len(durations) // 2]
        }
        
        # PnL distribution
        pnls = [t.pnl for t in trades]
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]
        
        pnl_stats = {
            'total_pnl': sum(pnls),
            'avg_winner': sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0,
            'avg_loser': sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0,
            'largest_winner': max(pnls) if pnls else 0,
            'largest_loser': min(pnls) if pnls else 0
        }
        
        # Monthly performance
        monthly_pnl = {}
        for trade in trades:
            month_key = trade.entry_time.strftime('%Y-%m')
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += trade.pnl
        
        return {
            'trade_count': len(trades),
            'duration_stats': duration_stats,
            'pnl_stats': pnl_stats,
            'monthly_performance': monthly_pnl,
            'first_trade': trades[0].entry_time.isoformat(),
            'last_trade': trades[-1].entry_time.isoformat()
        }
    
    def validate_for_reporting(self, results: BacktestResults) -> Dict[str, Any]:
        """Validate results are suitable for report generation."""
        
        validation_report = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        # Check for minimum data requirements
        if len(results.trades) == 0:
            validation_report['warnings'].append("No trades executed - strategy may be too conservative")
        
        if len(results.equity_curve) < 10:
            validation_report['warnings'].append("Very short equity curve - consider longer backtest period")
        
        # Check for concerning metrics
        metrics = results.performance_metrics
        
        if metrics.get('max_drawdown_pct', 0) > 50:
            validation_report['warnings'].append(f"High maximum drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        if metrics.get('win_rate_pct', 0) < 30:
            validation_report['warnings'].append(f"Low win rate: {metrics['win_rate_pct']:.2f}%")
        
        if metrics.get('sharpe_ratio', 0) < 0:
            validation_report['warnings'].append(f"Negative Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
        
        # Calculate statistics for reporting
        validation_report['stats'] = {
            'data_points': len(results.equity_curve),
            'trade_frequency': len(results.trades) / max(1, (results.end_date - results.start_date).days) * 30,  # trades per month
            'total_return_pct': metrics.get('total_return_pct', 0),
            'has_sufficient_data': len(results.equity_curve) >= 30 and len(results.trades) >= 5
        }
        
        return validation_report