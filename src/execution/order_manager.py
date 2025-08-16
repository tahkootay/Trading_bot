"""Advanced order execution and management system."""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from ..data_collector.bybit_client import BybitHTTPClient
from ..utils.types import (
    Order, Position, Signal, Side, OrderType, OrderStatus, SignalType
)
from ..utils.logger import TradingLogger


class ExecutionStrategy(str, Enum):
    """Order execution strategies."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"
    ICEBERG = "ICEBERG"
    SMART = "SMART"


@dataclass
class OrderRequest:
    """Order request data structure."""
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    reduce_only: bool = False
    post_only: bool = False
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SMART
    max_slippage: float = 0.001  # 0.1%
    timeout_seconds: int = 30


@dataclass
class TradeExecution:
    """Trade execution result."""
    order_id: str
    symbol: str
    side: Side
    quantity: float
    executed_quantity: float
    avg_price: float
    slippage: float
    execution_time: timedelta
    commission: float
    status: OrderStatus
    error_message: Optional[str] = None


class OrderManager:
    """Advanced order management and execution system."""
    
    def __init__(
        self,
        exchange_client: BybitHTTPClient,
        config: Dict[str, Any],
    ):
        self.exchange_client = exchange_client
        self.config = config
        self.logger = TradingLogger("order_manager")
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.execution_history: List[TradeExecution] = []
        
        # Position tracking
        self.current_positions: Dict[str, Position] = {}
        
        # Execution statistics
        self.total_executions = 0
        self.total_slippage = 0.0
        self.avg_execution_time = 0.0
        
        # Callbacks
        self.order_callbacks: List[Callable] = []
        self.execution_callbacks: List[Callable] = []
        
        # Configuration
        self.default_slippage_limit = config.get("max_slippage", 0.002)  # 0.2%
        self.default_timeout = config.get("order_timeout", 30)
        self.min_order_size = config.get("min_order_size", 10.0)  # $10 minimum
        
        # Smart execution parameters
        self.liquidity_threshold = config.get("liquidity_threshold", 1000.0)
        self.iceberg_chunk_size = config.get("iceberg_chunk_size", 0.1)  # 10% chunks
        self.twap_intervals = config.get("twap_intervals", 5)
    
    async def execute_signal(
        self,
        signal: Signal,
        position_size: float,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.SMART,
    ) -> Optional[TradeExecution]:
        """Execute trading signal with optimal strategy."""
        try:
            # Determine order parameters
            order_request = self._create_order_request(
                signal=signal,
                position_size=position_size,
                execution_strategy=execution_strategy,
            )
            
            if not order_request:
                return None
            
            # Validate order
            if not self._validate_order_request(order_request):
                return None
            
            # Execute based on strategy
            if execution_strategy == ExecutionStrategy.MARKET:
                return await self._execute_market_order(order_request)
            elif execution_strategy == ExecutionStrategy.LIMIT:
                return await self._execute_limit_order(order_request)
            elif execution_strategy == ExecutionStrategy.TWAP:
                return await self._execute_twap_order(order_request)
            elif execution_strategy == ExecutionStrategy.ICEBERG:
                return await self._execute_iceberg_order(order_request)
            else:  # SMART
                return await self._execute_smart_order(order_request)
        
        except Exception as e:
            self.logger.log_error(
                error_type="signal_execution_failed",
                component="order_manager",
                error_message=str(e),
                details={
                    "symbol": signal.symbol,
                    "signal_type": signal.signal_type.value,
                    "position_size": position_size,
                },
            )
            return None
    
    async def close_position(
        self,
        symbol: str,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.MARKET,
        partial_close_ratio: float = 1.0,
    ) -> Optional[TradeExecution]:
        """Close position with specified strategy."""
        try:
            # Get current position
            position = self.current_positions.get(symbol)
            if not position:
                self.logger.log_error(
                    error_type="position_not_found",
                    component="order_manager",
                    error_message=f"No position found for {symbol}",
                )
                return None
            
            # Calculate close quantity
            close_quantity = abs(position.size) * partial_close_ratio
            
            # Determine side (opposite of position)
            close_side = Side.SELL if position.side.value == "LONG" else Side.BUY
            
            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                side=close_side,
                quantity=close_quantity,
                order_type=OrderType.MARKET if execution_strategy == ExecutionStrategy.MARKET else OrderType.LIMIT,
                reduce_only=True,
                execution_strategy=execution_strategy,
            )
            
            # Execute close order
            return await self._execute_order(order_request)
        
        except Exception as e:
            self.logger.log_error(
                error_type="position_close_failed",
                component="order_manager",
                error_message=str(e),
                details={"symbol": symbol, "partial_ratio": partial_close_ratio},
            )
            return None
    
    async def set_stop_loss(
        self,
        symbol: str,
        stop_price: float,
        quantity: Optional[float] = None,
    ) -> Optional[str]:
        """Set stop loss order for position."""
        try:
            position = self.current_positions.get(symbol)
            if not position:
                return None
            
            # Use full position size if not specified
            if quantity is None:
                quantity = abs(position.size)
            
            # Determine side (opposite of position)
            stop_side = Side.SELL if position.side.value == "LONG" else Side.BUY
            
            order_request = OrderRequest(
                symbol=symbol,
                side=stop_side,
                quantity=quantity,
                order_type=OrderType.STOP_MARKET,
                stop_price=stop_price,
                reduce_only=True,
            )
            
            # Submit stop loss order
            order_id = await self._submit_order(order_request)
            
            if order_id:
                self.logger.log_order(
                    order_id=order_id,
                    symbol=symbol,
                    side=stop_side.value,
                    quantity=quantity,
                    price=stop_price,
                    order_type="STOP_LOSS",
                    status="SUBMITTED",
                )
            
            return order_id
        
        except Exception as e:
            self.logger.log_error(
                error_type="stop_loss_failed",
                component="order_manager",
                error_message=str(e),
                details={"symbol": symbol, "stop_price": stop_price},
            )
            return None
    
    async def set_take_profit(
        self,
        symbol: str,
        take_profit_price: float,
        quantity: Optional[float] = None,
    ) -> Optional[str]:
        """Set take profit order for position."""
        try:
            position = self.current_positions.get(symbol)
            if not position:
                return None
            
            # Use full position size if not specified
            if quantity is None:
                quantity = abs(position.size)
            
            # Determine side (opposite of position)
            tp_side = Side.SELL if position.side.value == "LONG" else Side.BUY
            
            order_request = OrderRequest(
                symbol=symbol,
                side=tp_side,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price=take_profit_price,
                reduce_only=True,
                post_only=True,
            )
            
            # Submit take profit order
            order_id = await self._submit_order(order_request)
            
            if order_id:
                self.logger.log_order(
                    order_id=order_id,
                    symbol=symbol,
                    side=tp_side.value,
                    quantity=quantity,
                    price=take_profit_price,
                    order_type="TAKE_PROFIT",
                    status="SUBMITTED",
                )
            
            return order_id
        
        except Exception as e:
            self.logger.log_error(
                error_type="take_profit_failed",
                component="order_manager",
                error_message=str(e),
                details={"symbol": symbol, "tp_price": take_profit_price},
            )
            return None
    
    def _create_order_request(
        self,
        signal: Signal,
        position_size: float,
        execution_strategy: ExecutionStrategy,
    ) -> Optional[OrderRequest]:
        """Create order request from signal."""
        try:
            # Determine side
            if signal.signal_type == SignalType.BUY:
                side = Side.BUY
            elif signal.signal_type == SignalType.SELL:
                side = Side.SELL
            else:
                return None  # Close signals handled separately
            
            # Calculate quantity from position size (in USD)
            if signal.price and signal.price > 0:
                quantity = position_size / signal.price
            else:
                return None
            
            # Determine order type based on strategy
            if execution_strategy == ExecutionStrategy.MARKET:
                order_type = OrderType.MARKET
                price = None
            else:
                order_type = OrderType.LIMIT
                # Set aggressive limit price for better fill probability
                if side == Side.BUY:
                    price = signal.price * 1.0005  # 0.05% above market
                else:
                    price = signal.price * 0.9995  # 0.05% below market
            
            return OrderRequest(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                execution_strategy=execution_strategy,
                max_slippage=self.default_slippage_limit,
                timeout_seconds=self.default_timeout,
            )
        
        except Exception as e:
            self.logger.log_error(
                error_type="order_request_creation_failed",
                component="order_manager",
                error_message=str(e),
            )
            return None
    
    def _validate_order_request(self, order_request: OrderRequest) -> bool:
        """Validate order request parameters."""
        try:
            # Check minimum order size
            if order_request.price:
                order_value = order_request.quantity * order_request.price
                if order_value < self.min_order_size:
                    self.logger.log_error(
                        error_type="order_too_small",
                        component="order_manager",
                        error_message=f"Order value ${order_value:.2f} below minimum ${self.min_order_size}",
                    )
                    return False
            
            # Check quantity is positive
            if order_request.quantity <= 0:
                return False
            
            # Check price is positive for limit orders
            if order_request.order_type == OrderType.LIMIT and (not order_request.price or order_request.price <= 0):
                return False
            
            return True
        
        except Exception:
            return False
    
    async def _execute_smart_order(self, order_request: OrderRequest) -> Optional[TradeExecution]:
        """Execute order using smart routing logic."""
        try:
            # Get current market conditions
            ticker = await self.exchange_client.get_ticker(order_request.symbol)
            orderbook = await self.exchange_client.get_orderbook(order_request.symbol)
            
            if not ticker or not orderbook:
                return None
            
            # Analyze market conditions
            spread = (ticker.ask - ticker.bid) / ticker.price
            order_value = order_request.quantity * ticker.price
            
            # Estimate available liquidity
            if order_request.side == Side.BUY:
                available_liquidity = sum([level[1] for level in orderbook.asks[:5]])
            else:
                available_liquidity = sum([level[1] for level in orderbook.bids[:5]])
            
            liquidity_ratio = order_request.quantity / available_liquidity if available_liquidity > 0 else 1.0
            
            # Choose execution strategy based on conditions
            if spread > 0.002:  # Wide spread > 0.2%
                return await self._execute_limit_order(order_request)
            elif liquidity_ratio > 0.1:  # Large order relative to liquidity
                return await self._execute_iceberg_order(order_request)
            elif order_value > self.liquidity_threshold:  # Large order value
                return await self._execute_twap_order(order_request)
            else:
                # Small order in good conditions - use aggressive limit
                return await self._execute_limit_order(order_request)
        
        except Exception as e:
            self.logger.log_error(
                error_type="smart_execution_failed",
                component="order_manager",
                error_message=str(e),
            )
            # Fallback to market order
            return await self._execute_market_order(order_request)
    
    async def _execute_market_order(self, order_request: OrderRequest) -> Optional[TradeExecution]:
        """Execute market order."""
        start_time = datetime.now()
        
        try:
            # Get current price for slippage calculation
            ticker = await self.exchange_client.get_ticker(order_request.symbol)
            if not ticker:
                return None
            
            expected_price = ticker.ask if order_request.side == Side.BUY else ticker.bid
            
            # Submit market order
            order_id = await self._submit_order(order_request)
            if not order_id:
                return None
            
            # Wait for execution
            execution_result = await self._wait_for_execution(order_id, order_request.timeout_seconds)
            
            execution_time = datetime.now() - start_time
            
            if execution_result:
                # Calculate slippage
                slippage = abs(execution_result["avg_price"] - expected_price) / expected_price
                
                execution = TradeExecution(
                    order_id=order_id,
                    symbol=order_request.symbol,
                    side=order_request.side,
                    quantity=order_request.quantity,
                    executed_quantity=execution_result["executed_quantity"],
                    avg_price=execution_result["avg_price"],
                    slippage=slippage,
                    execution_time=execution_time,
                    commission=execution_result.get("commission", 0.0),
                    status=OrderStatus.FILLED if execution_result["executed_quantity"] >= order_request.quantity * 0.95 else OrderStatus.PARTIALLY_FILLED,
                )
                
                self._record_execution(execution)
                return execution
            
            return None
        
        except Exception as e:
            self.logger.log_error(
                error_type="market_order_failed",
                component="order_manager",
                error_message=str(e),
            )
            return None
    
    async def _execute_limit_order(self, order_request: OrderRequest) -> Optional[TradeExecution]:
        """Execute limit order with timeout and fallback."""
        start_time = datetime.now()
        
        try:
            # Submit limit order
            order_id = await self._submit_order(order_request)
            if not order_id:
                return None
            
            # Wait for execution with timeout
            execution_result = await self._wait_for_execution(order_id, order_request.timeout_seconds)
            
            execution_time = datetime.now() - start_time
            
            if execution_result and execution_result["executed_quantity"] >= order_request.quantity * 0.95:
                # Order filled successfully
                execution = TradeExecution(
                    order_id=order_id,
                    symbol=order_request.symbol,
                    side=order_request.side,
                    quantity=order_request.quantity,
                    executed_quantity=execution_result["executed_quantity"],
                    avg_price=execution_result["avg_price"],
                    slippage=0.0,  # No slippage on limit orders
                    execution_time=execution_time,
                    commission=execution_result.get("commission", 0.0),
                    status=OrderStatus.FILLED,
                )
                
                self._record_execution(execution)
                return execution
            
            else:
                # Order not filled or partially filled - cancel and retry as market
                await self._cancel_order(order_id)
                
                # Convert to market order for immediate execution
                order_request.order_type = OrderType.MARKET
                order_request.price = None
                
                self.logger.log_system_event(
                    event_type="limit_order_timeout",
                    component="order_manager",
                    status="retrying_as_market",
                    details={"original_order_id": order_id, "symbol": order_request.symbol},
                )
                
                return await self._execute_market_order(order_request)
        
        except Exception as e:
            self.logger.log_error(
                error_type="limit_order_failed",
                component="order_manager",
                error_message=str(e),
            )
            return None
    
    async def _execute_twap_order(self, order_request: OrderRequest) -> Optional[TradeExecution]:
        """Execute TWAP (Time-Weighted Average Price) order."""
        try:
            chunk_size = order_request.quantity / self.twap_intervals
            total_executed = 0.0
            total_cost = 0.0
            total_commission = 0.0
            start_time = datetime.now()
            
            for i in range(self.twap_intervals):
                # Create chunk order
                chunk_request = OrderRequest(
                    symbol=order_request.symbol,
                    side=order_request.side,
                    quantity=chunk_size,
                    order_type=OrderType.MARKET,
                    execution_strategy=ExecutionStrategy.MARKET,
                )
                
                # Execute chunk
                chunk_execution = await self._execute_market_order(chunk_request)
                
                if chunk_execution and chunk_execution.status == OrderStatus.FILLED:
                    total_executed += chunk_execution.executed_quantity
                    total_cost += chunk_execution.executed_quantity * chunk_execution.avg_price
                    total_commission += chunk_execution.commission
                
                # Wait between chunks (except last one)
                if i < self.twap_intervals - 1:
                    await asyncio.sleep(order_request.timeout_seconds / self.twap_intervals)
            
            if total_executed > 0:
                avg_price = total_cost / total_executed
                execution_time = datetime.now() - start_time
                
                execution = TradeExecution(
                    order_id=f"TWAP_{uuid.uuid4()}",
                    symbol=order_request.symbol,
                    side=order_request.side,
                    quantity=order_request.quantity,
                    executed_quantity=total_executed,
                    avg_price=avg_price,
                    slippage=0.0,  # TWAP reduces slippage
                    execution_time=execution_time,
                    commission=total_commission,
                    status=OrderStatus.FILLED if total_executed >= order_request.quantity * 0.95 else OrderStatus.PARTIALLY_FILLED,
                )
                
                self._record_execution(execution)
                return execution
            
            return None
        
        except Exception as e:
            self.logger.log_error(
                error_type="twap_execution_failed",
                component="order_manager",
                error_message=str(e),
            )
            return None
    
    async def _execute_iceberg_order(self, order_request: OrderRequest) -> Optional[TradeExecution]:
        """Execute iceberg order (hidden large order)."""
        try:
            chunk_size = order_request.quantity * self.iceberg_chunk_size
            remaining_quantity = order_request.quantity
            total_executed = 0.0
            total_cost = 0.0
            total_commission = 0.0
            start_time = datetime.now()
            
            while remaining_quantity > 0:
                # Current chunk size (smaller of remaining or chunk_size)
                current_chunk = min(remaining_quantity, chunk_size)
                
                # Create chunk order as limit order
                chunk_request = OrderRequest(
                    symbol=order_request.symbol,
                    side=order_request.side,
                    quantity=current_chunk,
                    order_type=OrderType.LIMIT,
                    price=order_request.price,
                    execution_strategy=ExecutionStrategy.LIMIT,
                    timeout_seconds=10,  # Shorter timeout for chunks
                )
                
                # Execute chunk
                chunk_execution = await self._execute_limit_order(chunk_request)
                
                if chunk_execution and chunk_execution.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                    total_executed += chunk_execution.executed_quantity
                    total_cost += chunk_execution.executed_quantity * chunk_execution.avg_price
                    total_commission += chunk_execution.commission
                    remaining_quantity -= chunk_execution.executed_quantity
                else:
                    # Chunk failed, break
                    break
                
                # Small delay between chunks
                await asyncio.sleep(1)
            
            if total_executed > 0:
                avg_price = total_cost / total_executed
                execution_time = datetime.now() - start_time
                
                execution = TradeExecution(
                    order_id=f"ICEBERG_{uuid.uuid4()}",
                    symbol=order_request.symbol,
                    side=order_request.side,
                    quantity=order_request.quantity,
                    executed_quantity=total_executed,
                    avg_price=avg_price,
                    slippage=0.0,  # Iceberg reduces market impact
                    execution_time=execution_time,
                    commission=total_commission,
                    status=OrderStatus.FILLED if total_executed >= order_request.quantity * 0.95 else OrderStatus.PARTIALLY_FILLED,
                )
                
                self._record_execution(execution)
                return execution
            
            return None
        
        except Exception as e:
            self.logger.log_error(
                error_type="iceberg_execution_failed",
                component="order_manager",
                error_message=str(e),
            )
            return None
    
    async def _submit_order(self, order_request: OrderRequest) -> Optional[str]:
        """Submit order to exchange."""
        try:
            # Use exchange client to submit order
            # This is a simplified version - actual implementation would use exchange client
            
            # For now, return a mock order ID
            order_id = f"ORDER_{uuid.uuid4()}"
            
            # Create order object for tracking
            order = Order(
                order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                price=order_request.price,
                stop_price=order_request.stop_price,
                status=OrderStatus.NEW,
                filled_quantity=0.0,
                average_price=None,
                created_at=datetime.now(),
                updated_at=None,
            )
            
            self.active_orders[order_id] = order
            
            self.logger.log_order(
                order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side.value,
                quantity=order_request.quantity,
                price=order_request.price,
                order_type=order_request.order_type.value,
                status="SUBMITTED",
            )
            
            return order_id
        
        except Exception as e:
            self.logger.log_error(
                error_type="order_submission_failed",
                component="order_manager",
                error_message=str(e),
            )
            return None
    
    async def _wait_for_execution(self, order_id: str, timeout_seconds: int) -> Optional[Dict[str, Any]]:
        """Wait for order execution with timeout."""
        try:
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < timeout_seconds:
                # Check order status
                order = self.active_orders.get(order_id)
                if not order:
                    break
                
                # Simulate order execution (in real implementation, would check with exchange)
                if order.status == OrderStatus.NEW:
                    # Simulate execution after random delay
                    await asyncio.sleep(np.random.uniform(0.1, 2.0))
                    
                    # Simulate execution
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.average_price = order.price if order.price else 100.0  # Mock price
                    order.updated_at = datetime.now()
                    
                    return {
                        "executed_quantity": order.filled_quantity,
                        "avg_price": order.average_price,
                        "commission": order.quantity * order.average_price * 0.001,  # Mock 0.1% commission
                    }
                
                elif order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                    return {
                        "executed_quantity": order.filled_quantity,
                        "avg_price": order.average_price,
                        "commission": order.filled_quantity * order.average_price * 0.001,
                    }
                
                await asyncio.sleep(0.1)
            
            return None
        
        except Exception as e:
            self.logger.log_error(
                error_type="execution_wait_failed",
                component="order_manager",
                error_message=str(e),
            )
            return None
    
    async def _cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order.status = OrderStatus.CANCELED
                order.updated_at = datetime.now()
                
                self.logger.log_order(
                    order_id=order_id,
                    symbol=order.symbol,
                    side=order.side.value,
                    quantity=order.quantity,
                    price=order.price,
                    order_type=order.order_type.value,
                    status="CANCELED",
                )
                
                return True
            
            return False
        
        except Exception as e:
            self.logger.log_error(
                error_type="order_cancel_failed",
                component="order_manager",
                error_message=str(e),
            )
            return False
    
    def _record_execution(self, execution: TradeExecution) -> None:
        """Record execution for statistics."""
        self.execution_history.append(execution)
        
        # Update statistics
        self.total_executions += 1
        self.total_slippage += execution.slippage
        
        execution_time_seconds = execution.execution_time.total_seconds()
        self.avg_execution_time = (
            (self.avg_execution_time * (self.total_executions - 1) + execution_time_seconds)
            / self.total_executions
        )
        
        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
        
        # Call callbacks
        for callback in self.execution_callbacks:
            try:
                callback(execution)
            except Exception as e:
                self.logger.log_error(
                    error_type="execution_callback_failed",
                    component="order_manager",
                    error_message=str(e),
                )
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution performance statistics."""
        if not self.execution_history:
            return {}
        
        recent_executions = self.execution_history[-100:]  # Last 100 executions
        
        return {
            "total_executions": self.total_executions,
            "avg_slippage": self.total_slippage / self.total_executions if self.total_executions > 0 else 0,
            "avg_execution_time": self.avg_execution_time,
            "recent_fill_rate": len([e for e in recent_executions if e.status == OrderStatus.FILLED]) / len(recent_executions),
            "recent_avg_slippage": np.mean([e.slippage for e in recent_executions]),
            "recent_avg_execution_time": np.mean([e.execution_time.total_seconds() for e in recent_executions]),
        }
    
    def add_order_callback(self, callback: Callable) -> None:
        """Add order status callback."""
        self.order_callbacks.append(callback)
    
    def add_execution_callback(self, callback: Callable) -> None:
        """Add execution callback."""
        self.execution_callbacks.append(callback)
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all active orders."""
        canceled_count = 0
        
        orders_to_cancel = [
            order_id for order_id, order in self.active_orders.items()
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED] and
            (symbol is None or order.symbol == symbol)
        ]
        
        for order_id in orders_to_cancel:
            if await self._cancel_order(order_id):
                canceled_count += 1
        
        return canceled_count
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get active orders."""
        return [
            order for order in self.active_orders.values()
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED] and
            (symbol is None or order.symbol == symbol)
        ]