"""Comprehensive risk management system for trading bot."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from ..utils.types import (
    Signal, Position, Account, RiskMetrics, PerformanceMetrics,
    SignalType, Side, PositionSide
)
from ..utils.logger import TradingLogger


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskAction(str, Enum):
    """Risk management actions."""
    ALLOW = "ALLOW"
    REDUCE_SIZE = "REDUCE_SIZE"
    PAUSE_TRADING = "PAUSE_TRADING"
    CLOSE_POSITIONS = "CLOSE_POSITIONS"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class RiskEvent:
    """Risk event data structure."""
    timestamp: datetime
    risk_type: str
    risk_level: RiskLevel
    symbol: str
    description: str
    current_value: float
    threshold: float
    action_taken: RiskAction
    details: Dict[str, Any]


@dataclass
class PositionRisk:
    """Position-specific risk metrics."""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    risk_amount: float
    time_in_position: timedelta
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]


class RiskManager:
    """Advanced risk management system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = TradingLogger("risk_manager")
        
        # Risk limits (aligned with algorithm)
        self.max_daily_loss = config.get("max_daily_loss", 0.03)  # 3%
        self.max_drawdown = config.get("max_drawdown", 0.10)  # 10%
        self.max_position_size = config.get("max_position_size", 0.02)  # 2%
        self.max_consecutive_losses = config.get("max_consecutive_losses", 3)
        self.max_correlation = config.get("max_correlation", 0.8)
        self.max_positions = config.get("max_positions", 3)
        self.pause_after_losses_hours = config.get("pause_after_losses_hours", 2)
        
        # Algorithm-specific emergency stops
        self.extreme_1m_move = config.get("extreme_1m_move", 0.05)  # 5% in 1 minute
        self.black_swan_dd = config.get("black_swan_dd", 0.15)  # 15% drawdown
        self.max_slippage_pct = config.get("max_slippage_pct", 0.001)  # 0.1%
        
        # Drawdown management levels
        self.dd_levels = config.get("dd_levels", {
            0.05: {'position_mult': 0.5, 'conf_boost': 0},
            0.08: {'position_mult': 0.3, 'conf_boost': 0.10},
            0.10: {'position_mult': 0, 'conf_boost': None}
        })
        
        # Position sizing
        self.position_sizing_method = config.get("position_sizing_method", "kelly")
        self.kelly_fraction = config.get("kelly_fraction", 0.25)
        
        # State tracking
        self.daily_pnl = 0.0
        self.max_daily_profit = 0.0
        self.consecutive_losses = 0
        self.last_loss_time: Optional[datetime] = None
        self.total_risk_exposure = 0.0
        self.is_trading_paused = False
        self.pause_until: Optional[datetime] = None
        
        # Position tracking
        self.current_positions: Dict[str, Position] = {}
        self.position_risks: Dict[str, PositionRisk] = {}
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.risk_events: List[RiskEvent] = []
        self.daily_stats: Dict[str, Dict] = {}
        
        # Account tracking
        self.initial_balance = 0.0
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.account_high_water_mark = 0.0
    
    async def validate_signal(
        self,
        signal: Signal,
        account: Account,
        current_positions: List[Position],
    ) -> Tuple[bool, Optional[RiskAction], Optional[str]]:
        """Validate trading signal against risk parameters."""
        try:
            # Update internal state
            self._update_account_state(account)
            self._update_positions(current_positions)
            
            # Check if trading is paused
            if self.is_trading_paused:
                if self.pause_until and datetime.now() < self.pause_until:
                    return False, RiskAction.PAUSE_TRADING, "Trading paused due to risk limits"
                else:
                    self.is_trading_paused = False
                    self.pause_until = None
            
            # Check daily loss limit
            if not self._check_daily_loss_limit():
                return False, RiskAction.PAUSE_TRADING, "Daily loss limit exceeded"
            
            # Check maximum drawdown
            if not self._check_max_drawdown():
                return False, RiskAction.EMERGENCY_STOP, "Maximum drawdown exceeded"
            
            # Check consecutive losses
            if not self._check_consecutive_losses():
                return False, RiskAction.PAUSE_TRADING, "Too many consecutive losses"
            
            # Check position limits
            if not self._check_position_limits(signal):
                return False, RiskAction.ALLOW, "Maximum positions reached"
            
            # Check correlation limits
            if not self._check_correlation_limits(signal):
                return False, RiskAction.ALLOW, "Correlation limit exceeded"
            
            # Check position size
            suggested_size = self.calculate_position_size(signal, account)
            if suggested_size <= 0:
                return False, RiskAction.ALLOW, "Position size too small"
            
            # All checks passed
            return True, RiskAction.ALLOW, "Risk checks passed"
        
        except Exception as e:
            self.logger.log_error(
                error_type="risk_validation_failed",
                component="risk_manager",
                error_message=str(e),
                details={"symbol": signal.symbol},
            )
            return False, RiskAction.PAUSE_TRADING, f"Risk validation error: {str(e)}"
    
    def calculate_position_size(
        self,
        signal: Signal,
        account: Account,
        risk_per_trade: Optional[float] = None,
    ) -> float:
        """Calculate optimal position size based on risk parameters."""
        try:
            if risk_per_trade is None:
                risk_per_trade = self.max_position_size
            
            available_balance = account.available_balance
            
            if self.position_sizing_method == "fixed":
                return available_balance * risk_per_trade
            
            elif self.position_sizing_method == "kelly":
                return self._kelly_position_size(signal, account, risk_per_trade)
            
            elif self.position_sizing_method == "volatility":
                return self._volatility_position_size(signal, account, risk_per_trade)
            
            else:
                # Default to fixed percentage
                return available_balance * risk_per_trade
        
        except Exception as e:
            self.logger.log_error(
                error_type="position_sizing_failed",
                component="risk_manager",
                error_message=str(e),
            )
            return 0.0
    
    def _kelly_position_size(
        self,
        signal: Signal,
        account: Account,
        max_risk: float,
    ) -> float:
        """Calculate position size using Kelly Criterion."""
        try:
            # Get historical performance for this type of signal
            win_rate, avg_win, avg_loss = self._get_signal_performance(signal)
            
            if win_rate <= 0 or avg_win <= 0 or avg_loss <= 0:
                # Fallback to conservative sizing
                return account.available_balance * (max_risk * 0.5)
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds received (avg_win/avg_loss), p = win_rate, q = loss_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety multiplier and cap
            safe_kelly = kelly_fraction * self.kelly_fraction
            safe_kelly = max(0, min(safe_kelly, max_risk))
            
            return account.available_balance * safe_kelly
        
        except Exception as e:
            self.logger.log_error(
                error_type="kelly_calculation_failed",
                component="risk_manager",
                error_message=str(e),
            )
            return account.available_balance * (max_risk * 0.5)
    
    def _volatility_position_size(
        self,
        signal: Signal,
        account: Account,
        max_risk: float,
    ) -> float:
        """Calculate position size based on volatility."""
        try:
            # Get ATR or similar volatility measure from signal features
            atr_ratio = signal.features.get('atr_ratio', 0.02)
            
            # Adjust position size inversely to volatility
            base_vol = 0.02  # 2% baseline volatility
            vol_adjustment = base_vol / atr_ratio if atr_ratio > 0 else 1.0
            vol_adjustment = max(0.5, min(vol_adjustment, 2.0))  # Cap between 0.5x and 2x
            
            adjusted_risk = max_risk * vol_adjustment
            adjusted_risk = min(adjusted_risk, max_risk)  # Don't exceed max
            
            return account.available_balance * adjusted_risk
        
        except Exception as e:
            self.logger.log_error(
                error_type="volatility_sizing_failed",
                component="risk_manager",
                error_message=str(e),
            )
            return account.available_balance * max_risk
    
    def _get_signal_performance(self, signal: Signal) -> Tuple[float, float, float]:
        """Get historical performance for similar signals."""
        try:
            # Filter similar trades from history
            similar_trades = [
                trade for trade in self.trade_history
                if (
                    trade.get("symbol") == signal.symbol and
                    trade.get("signal_type") == signal.signal_type.value and
                    trade.get("confidence", 0) >= signal.confidence * 0.8
                )
            ]
            
            if len(similar_trades) < 10:
                # Not enough data, use overall performance
                similar_trades = self.trade_history[-50:]  # Last 50 trades
            
            if not similar_trades:
                return 0.55, 1.5, 1.0  # Default conservative estimates
            
            wins = [trade for trade in similar_trades if trade.get("pnl", 0) > 0]
            losses = [trade for trade in similar_trades if trade.get("pnl", 0) < 0]
            
            if not wins or not losses:
                return 0.55, 1.5, 1.0
            
            win_rate = len(wins) / len(similar_trades)
            avg_win = np.mean([trade["pnl"] for trade in wins])
            avg_loss = abs(np.mean([trade["pnl"] for trade in losses]))
            
            return win_rate, avg_win, avg_loss
        
        except Exception:
            return 0.55, 1.5, 1.0  # Safe defaults
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded."""
        if self.daily_pnl <= -abs(self.max_daily_loss * self.current_balance):
            self._trigger_risk_event(
                risk_type="daily_loss_limit",
                risk_level=RiskLevel.CRITICAL,
                symbol="PORTFOLIO",
                description="Daily loss limit exceeded",
                current_value=abs(self.daily_pnl / self.current_balance),
                threshold=self.max_daily_loss,
                action=RiskAction.PAUSE_TRADING,
            )
            return False
        return True
    
    def _check_max_drawdown(self) -> bool:
        """Check if maximum drawdown is exceeded."""
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            
            if current_drawdown >= self.max_drawdown:
                self._trigger_risk_event(
                    risk_type="max_drawdown",
                    risk_level=RiskLevel.CRITICAL,
                    symbol="PORTFOLIO",
                    description="Maximum drawdown exceeded",
                    current_value=current_drawdown,
                    threshold=self.max_drawdown,
                    action=RiskAction.EMERGENCY_STOP,
                )
                return False
        return True
    
    def _check_consecutive_losses(self) -> bool:
        """Check consecutive losses limit."""
        if self.consecutive_losses >= self.max_consecutive_losses:
            # Check if pause period has passed
            if self.last_loss_time:
                time_since_loss = datetime.now() - self.last_loss_time
                required_pause = timedelta(hours=self.pause_after_losses_hours)
                
                if time_since_loss < required_pause:
                    self._trigger_risk_event(
                        risk_type="consecutive_losses",
                        risk_level=RiskLevel.HIGH,
                        symbol="PORTFOLIO",
                        description="Too many consecutive losses",
                        current_value=self.consecutive_losses,
                        threshold=self.max_consecutive_losses,
                        action=RiskAction.PAUSE_TRADING,
                    )
                    return False
                else:
                    # Reset consecutive losses after pause
                    self.consecutive_losses = 0
        return True
    
    def _check_position_limits(self, signal: Signal) -> bool:
        """Check position count limits."""
        current_position_count = len(self.current_positions)
        
        # Check if already have position in this symbol
        if signal.symbol in self.current_positions:
            return True  # Allow position management
        
        if current_position_count >= self.max_positions:
            self._trigger_risk_event(
                risk_type="max_positions",
                risk_level=RiskLevel.MEDIUM,
                symbol=signal.symbol,
                description="Maximum position count reached",
                current_value=current_position_count,
                threshold=self.max_positions,
                action=RiskAction.ALLOW,
            )
            return False
        
        return True
    
    def _check_correlation_limits(self, signal: Signal) -> bool:
        """Check position correlation limits."""
        try:
            if len(self.current_positions) < 2:
                return True  # Not enough positions to check correlation
            
            # Simple correlation check based on direction
            # In production, would use actual price correlation
            same_direction_count = 0
            
            for position in self.current_positions.values():
                signal_direction = signal.signal_type in [SignalType.BUY]
                position_direction = position.side == PositionSide.LONG
                
                if signal_direction == position_direction:
                    same_direction_count += 1
            
            correlation_ratio = same_direction_count / len(self.current_positions)
            
            if correlation_ratio > self.max_correlation:
                self._trigger_risk_event(
                    risk_type="high_correlation",
                    risk_level=RiskLevel.MEDIUM,
                    symbol=signal.symbol,
                    description="High correlation between positions",
                    current_value=correlation_ratio,
                    threshold=self.max_correlation,
                    action=RiskAction.ALLOW,
                )
                return False
            
            return True
        
        except Exception:
            return True  # Default to allow if check fails
    
    def _update_account_state(self, account: Account) -> None:
        """Update account state tracking."""
        self.current_balance = account.total_balance
        
        if self.initial_balance == 0:
            self.initial_balance = account.total_balance
            self.peak_balance = account.total_balance
            self.account_high_water_mark = account.total_balance
        
        # Update peak balance
        if account.total_balance > self.peak_balance:
            self.peak_balance = account.total_balance
        
        # Update daily PnL (simplified - in production would track from market open)
        today = datetime.now().date()
        if today not in self.daily_stats:
            self.daily_stats[today] = {
                "starting_balance": self.current_balance,
                "trades": [],
                "pnl": 0.0,
            }
        
        # Calculate daily PnL
        starting_balance = self.daily_stats[today]["starting_balance"]
        self.daily_pnl = account.total_balance - starting_balance
        
        # Update high water mark
        if account.total_balance > self.account_high_water_mark:
            self.account_high_water_mark = account.total_balance
    
    def _update_positions(self, positions: List[Position]) -> None:
        """Update position tracking."""
        # Convert to dict for easier access
        self.current_positions = {pos.symbol: pos for pos in positions}
        
        # Update position risks
        for position in positions:
            self.position_risks[position.symbol] = PositionRisk(
                symbol=position.symbol,
                position_size=position.size,
                entry_price=position.entry_price,
                current_price=position.mark_price,
                unrealized_pnl=position.unrealized_pnl,
                risk_amount=abs(position.unrealized_pnl) if position.unrealized_pnl < 0 else 0,
                time_in_position=datetime.now() - position.created_at,
                stop_loss=None,  # Would be populated from order data
                take_profit=None,  # Would be populated from order data
                risk_reward_ratio=None,  # Would be calculated
            )
        
        # Calculate total risk exposure
        self.total_risk_exposure = sum(
            abs(pos.unrealized_pnl) for pos in positions
            if pos.unrealized_pnl < 0
        )
    
    def _trigger_risk_event(
        self,
        risk_type: str,
        risk_level: RiskLevel,
        symbol: str,
        description: str,
        current_value: float,
        threshold: float,
        action: RiskAction,
    ) -> None:
        """Trigger a risk event."""
        event = RiskEvent(
            timestamp=datetime.now(),
            risk_type=risk_type,
            risk_level=risk_level,
            symbol=symbol,
            description=description,
            current_value=current_value,
            threshold=threshold,
            action_taken=action,
            details={
                "daily_pnl": self.daily_pnl,
                "current_balance": self.current_balance,
                "consecutive_losses": self.consecutive_losses,
                "position_count": len(self.current_positions),
                "total_exposure": self.total_risk_exposure,
            },
        )
        
        self.risk_events.append(event)
        
        # Apply action
        if action == RiskAction.PAUSE_TRADING:
            self.is_trading_paused = True
            self.pause_until = datetime.now() + timedelta(hours=self.pause_after_losses_hours)
        
        # Log event
        self.logger.log_risk_event(
            event_type=risk_type,
            symbol=symbol,
            risk_level=risk_level.value,
            details=event.details,
            action_taken=action.value,
        )
    
    def record_trade_result(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        signal_confidence: float,
        trade_duration: timedelta,
    ) -> None:
        """Record trade result for performance tracking."""
        try:
            trade_record = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "pnl": pnl,
                "signal_confidence": signal_confidence,
                "trade_duration": trade_duration,
                "is_win": pnl > 0,
            }
            
            self.trade_history.append(trade_record)
            
            # Update consecutive losses
            if pnl <= 0:
                self.consecutive_losses += 1
                self.last_loss_time = datetime.now()
            else:
                self.consecutive_losses = 0
            
            # Update daily stats
            today = datetime.now().date()
            if today in self.daily_stats:
                self.daily_stats[today]["trades"].append(trade_record)
                self.daily_stats[today]["pnl"] += pnl
            
            # Keep only last 1000 trades
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
            
            self.logger.log_performance(
                period="trade",
                total_trades=1,
                win_rate=1.0 if pnl > 0 else 0.0,
                profit_factor=max(pnl, 0.01) / max(abs(min(pnl, 0)), 0.01),
                pnl=pnl,
                max_drawdown=self._calculate_current_drawdown(),
            )
        
        except Exception as e:
            self.logger.log_error(
                error_type="trade_recording_failed",
                component="risk_manager",
                error_message=str(e),
            )
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        if self.account_high_water_mark > 0:
            return (self.account_high_water_mark - self.current_balance) / self.account_high_water_mark
        return 0.0
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk metrics."""
        try:
            # Calculate performance metrics
            recent_trades = self.trade_history[-100:] if self.trade_history else []
            
            win_rate = 0.0
            profit_factor = 1.0
            sharpe_ratio = 0.0
            
            if recent_trades:
                wins = [t for t in recent_trades if t["pnl"] > 0]
                win_rate = len(wins) / len(recent_trades)
                
                total_wins = sum(t["pnl"] for t in wins)
                total_losses = abs(sum(t["pnl"] for t in recent_trades if t["pnl"] <= 0))
                
                if total_losses > 0:
                    profit_factor = total_wins / total_losses
                
                # Calculate Sharpe ratio (simplified)
                returns = [t["pnl"] / self.current_balance for t in recent_trades]
                if len(returns) > 1:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    if std_return > 0:
                        sharpe_ratio = avg_return / std_return * np.sqrt(252)  # Annualized
            
            # VaR calculation (simplified 95% VaR)
            var_95 = 0.0
            if recent_trades:
                pnl_values = [t["pnl"] for t in recent_trades]
                var_95 = abs(np.percentile(pnl_values, 5)) if pnl_values else 0.0
            
            return RiskMetrics(
                timestamp=datetime.now(),
                account_balance=self.current_balance,
                total_exposure=self.total_risk_exposure,
                unrealized_pnl=sum(pos.unrealized_pnl for pos in self.current_positions.values()),
                daily_pnl=self.daily_pnl,
                max_drawdown=self._calculate_current_drawdown(),
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                correlation_matrix={},  # Would calculate actual correlations
                position_count=len(self.current_positions),
                max_position_size=self.max_position_size,
            )
        
        except Exception as e:
            self.logger.log_error(
                error_type="risk_metrics_calculation_failed",
                component="risk_manager",
                error_message=str(e),
            )
            # Return default metrics
            return RiskMetrics(
                timestamp=datetime.now(),
                account_balance=self.current_balance,
                total_exposure=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=1.0,
                sharpe_ratio=0.0,
                var_95=0.0,
                correlation_matrix={},
                position_count=0,
                max_position_size=self.max_position_size,
            )
    
    def get_recent_risk_events(self, hours: int = 24) -> List[RiskEvent]:
        """Get recent risk events."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.risk_events
            if event.timestamp >= cutoff_time
        ]
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at market open)."""
        self.daily_pnl = 0.0
        self.max_daily_profit = 0.0
        today = datetime.now().date()
        
        if today not in self.daily_stats:
            self.daily_stats[today] = {
                "starting_balance": self.current_balance,
                "trades": [],
                "pnl": 0.0,
            }
    
    def emergency_stop(self) -> RiskAction:
        """Trigger emergency stop procedures."""
        self.is_trading_paused = True
        self.pause_until = None  # Indefinite pause until manual reset
        
        self._trigger_risk_event(
            risk_type="emergency_stop",
            risk_level=RiskLevel.CRITICAL,
            symbol="PORTFOLIO",
            description="Emergency stop triggered",
            current_value=1.0,
            threshold=1.0,
            action=RiskAction.EMERGENCY_STOP,
        )
        
        return RiskAction.CLOSE_POSITIONS
    
    def resume_trading(self) -> None:
        """Resume trading after manual review."""
        self.is_trading_paused = False
        self.pause_until = None
        
        self.logger.log_system_event(
            event_type="trading_resumed",
            component="risk_manager",
            status="resumed",
        )
    
    def check_emergency_conditions(self, market_data: Dict[str, Any]) -> Optional[RiskAction]:
        """
        Check for emergency conditions requiring immediate action
        Aligned with algorithm specification
        """
        # Check extreme 1-minute moves
        extreme_move = self._check_extreme_price_move(market_data)
        if extreme_move:
            return extreme_move
        
        # Check black swan drawdown
        black_swan = self._check_black_swan_drawdown()
        if black_swan:
            return black_swan
        
        # Check excessive slippage
        slippage_check = self._check_excessive_slippage(market_data)
        if slippage_check:
            return slippage_check
        
        return None
    
    def _check_extreme_price_move(self, market_data: Dict[str, Any]) -> Optional[RiskAction]:
        """Check for 5%+ move in 1 minute"""
        price_data = market_data.get('price_1m', [])
        if len(price_data) < 2:
            return None
        
        current_price = price_data[-1]
        prev_price = price_data[-2]
        
        price_change = abs(current_price - prev_price) / prev_price
        
        if price_change >= self.extreme_1m_move:
            self._create_risk_event(
                risk_type="extreme_price_move",
                risk_level=RiskLevel.CRITICAL,
                symbol="SOLUSDT",
                description=f"Extreme 1m price move: {price_change:.2%}",
                current_value=price_change,
                threshold=self.extreme_1m_move,
                action=RiskAction.EMERGENCY_STOP
            )
            return RiskAction.EMERGENCY_STOP
        
        return None
    
    def _check_black_swan_drawdown(self) -> Optional[RiskAction]:
        """Check for black swan drawdown (15%+)"""
        current_dd = self._calculate_current_drawdown()
        
        if current_dd >= self.black_swan_dd:
            self._create_risk_event(
                risk_type="black_swan_drawdown",
                risk_level=RiskLevel.CRITICAL,
                symbol="ALL",
                description=f"Black swan drawdown: {current_dd:.2%}",
                current_value=current_dd,
                threshold=self.black_swan_dd,
                action=RiskAction.EMERGENCY_STOP
            )
            return RiskAction.EMERGENCY_STOP
        
        return None
    
    def _check_excessive_slippage(self, market_data: Dict[str, Any]) -> Optional[RiskAction]:
        """Check for excessive slippage indicating liquidity crisis"""
        recent_executions = market_data.get('recent_executions', [])
        if not recent_executions:
            return None
        
        # Check last 5 executions
        for execution in recent_executions[-5:]:
            expected_price = execution.get('expected_price', 0)
            actual_price = execution.get('actual_price', 0)
            
            if expected_price > 0:
                slippage = abs(actual_price - expected_price) / expected_price
                
                if slippage >= self.max_slippage_pct:
                    self._create_risk_event(
                        risk_type="excessive_slippage",
                        risk_level=RiskLevel.HIGH,
                        symbol=execution.get('symbol', 'SOLUSDT'),
                        description=f"Excessive slippage: {slippage:.3%}",
                        current_value=slippage,
                        threshold=self.max_slippage_pct,
                        action=RiskAction.PAUSE_TRADING
                    )
                    return RiskAction.PAUSE_TRADING
        
        return None
    
    def get_drawdown_adjustments(self, current_dd: float) -> Dict[str, float]:
        """
        Get position sizing adjustments based on current drawdown
        As per algorithm specification
        """
        for dd_threshold in sorted(self.dd_levels.keys(), reverse=True):
            if current_dd >= dd_threshold:
                return self.dd_levels[dd_threshold]
        
        return {'position_mult': 1.0, 'conf_boost': 0}
    
    def check_regime_risk_limits(self, regime: str, setup_type: str) -> bool:
        """
        Check if setup type is appropriate for current market regime
        """
        regime_restrictions = {
            'volatile_choppy': ['mean_reversion'],  # Only mean reversion in choppy markets
            'low_volatility_range': ['breakout', 'momentum'],  # No range trading in low vol
        }
        
        if regime in regime_restrictions:
            allowed_setups = regime_restrictions[regime]
            return setup_type in allowed_setups
        
        return True  # Allow all setups in other regimes
    
    def _create_risk_event(
        self,
        risk_type: str,
        risk_level: RiskLevel,
        symbol: str,
        description: str,
        current_value: float,
        threshold: float,
        action: RiskAction
    ) -> None:
        """Create and log a risk event"""
        event = RiskEvent(
            timestamp=datetime.now(),
            risk_type=risk_type,
            risk_level=risk_level,
            symbol=symbol,
            description=description,
            current_value=current_value,
            threshold=threshold,
            action_taken=action,
            details={
                "daily_pnl": self.daily_pnl,
                "current_balance": self.current_balance,
                "consecutive_losses": self.consecutive_losses,
                "position_count": len(self.current_positions),
                "total_exposure": self.total_risk_exposure,
            }
        )
        
        self.risk_events.append(event)
        
        # Apply action
        if action == RiskAction.PAUSE_TRADING:
            self.is_trading_paused = True
            self.pause_until = datetime.now() + timedelta(hours=self.pause_after_losses_hours)
        elif action == RiskAction.EMERGENCY_STOP:
            self.is_trading_paused = True
            self.pause_until = None  # Indefinite pause
        
        # Log event
        self.logger.log_risk_event(
            event_type=risk_type,
            symbol=symbol,
            risk_level=risk_level.value,
            details=event.details,
            action_taken=action.value,
        )