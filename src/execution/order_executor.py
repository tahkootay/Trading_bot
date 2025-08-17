"""
Order Execution Engine for SOL/USDT Trading Algorithm

Implements sophisticated order execution with smart routing, iceberg orders,
slippage monitoring, and retry logic as specified in the algorithm.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    LIMIT_AGGRESSIVE = "limit_aggressive"
    ICEBERG = "iceberg"

class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMEOUT = "timeout"

class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill

@dataclass
class OrderRequest:
    """Order execution request"""
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    order_type: OrderType
    price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    parent_id: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class OrderResult:
    """Order execution result"""
    order_id: str
    status: OrderStatus
    filled_size: float
    avg_price: float
    total_cost: float
    slippage_pct: float
    execution_time_ms: float
    fees: float
    remaining_size: float
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class IcebergOrder:
    """Iceberg order tracking"""
    parent_id: str
    total_size: float
    filled_size: float
    show_size: float
    child_orders: List[str]
    status: OrderStatus
    symbol: str
    side: str
    price: float

class OrderExecutor:
    """
    Advanced order execution engine with smart routing and iceberg orders
    Implements algorithm specification for order execution
    """
    
    def __init__(self, exchange_api, config: Dict = None):
        self.api = exchange_api
        self.config = config or {}
        self.pending_orders = {}
        self.iceberg_orders = {}
        self.execution_history = []
        self.logger = logging.getLogger(__name__)
        
        # Algorithm constants from specification
        self.EXECUTION_PARAMS = {
            'entry_offset_pct': self.config.get('entry_offset_pct', 0.0005),  # 0.05%
            'fill_timeout_sec': self.config.get('fill_timeout_sec', 30),
            'partial_fill_min': self.config.get('partial_fill_min', 0.5),
            'iceberg_show_pct': self.config.get('iceberg_show_pct', 0.2),  # 20%
            'retry_attempts': self.config.get('retry_attempts', 3),
            'retry_delay_sec': self.config.get('retry_delay_sec', 5),
            'max_slippage_pct': self.config.get('max_slippage_pct', 0.001)  # 0.1%
        }
    
    async def execute_entry(self, setup: Dict) -> Optional[OrderResult]:
        """
        Execute entry order with smart routing as per algorithm specification
        
        Args:
            setup: Trading setup containing entry details
            
        Returns:
            OrderResult or None if execution failed
        """
        try:
            # Determine order type based on setup confidence
            confidence = setup.get('confidence', 0.6)
            if confidence > 0.8:
                order_type = OrderType.LIMIT
            else:
                order_type = OrderType.LIMIT_AGGRESSIVE
            
            # Create order request
            order_request = OrderRequest(
                symbol=setup.get('symbol', 'SOLUSDT'),
                side='buy' if setup['direction'] == 'long' else 'sell',
                size=setup['position_size'],
                order_type=order_type,
                price=setup['entry'],
                time_in_force=TimeInForce.IOC if order_type == OrderType.LIMIT_AGGRESSIVE else TimeInForce.GTC,
                metadata={
                    'setup_type': setup.get('setup_type', 'unknown'),
                    'confidence': confidence,
                    'entry_time': datetime.now().isoformat()
                }
            )
            
            # Check if should use iceberg
            if self._should_use_iceberg(order_request):
                return await self._execute_iceberg_order(order_request)
            else:
                return await self._execute_single_order(order_request)
                
        except Exception as e:
            self.logger.error(f"Error executing entry order: {e}")
            return None
    
    async def execute_exit(self, position: Dict, reason: str = "manual") -> Optional[OrderResult]:
        """
        Execute exit order (market order for quick execution)
        
        Args:
            position: Position to close
            reason: Reason for exit
            
        Returns:
            OrderResult or None if execution failed
        """
        try:
            # Use market order for exits to ensure quick execution
            order_request = OrderRequest(
                symbol=position.get('symbol', 'SOLUSDT'),
                side='sell' if position['direction'] == 'long' else 'buy',
                size=position['size'],
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.IOC,
                metadata={
                    'exit_reason': reason,
                    'exit_time': datetime.now().isoformat(),
                    'position_id': position.get('id', 'unknown')
                }
            )
            
            return await self._execute_single_order(order_request)
            
        except Exception as e:
            self.logger.error(f"Error executing exit order: {e}")
            return None
    
    async def execute_partial_close(self, position: Dict, size: float, reason: str = "take_profit") -> Optional[OrderResult]:
        """
        Execute partial position close
        
        Args:
            position: Position to partially close
            size: Size to close
            reason: Reason for partial close
            
        Returns:
            OrderResult or None if execution failed
        """
        try:
            order_request = OrderRequest(
                symbol=position.get('symbol', 'SOLUSDT'),
                side='sell' if position['direction'] == 'long' else 'buy',
                size=size,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.IOC,
                metadata={
                    'partial_close_reason': reason,
                    'close_time': datetime.now().isoformat(),
                    'position_id': position.get('id', 'unknown')
                }
            )
            
            return await self._execute_single_order(order_request)
            
        except Exception as e:
            self.logger.error(f"Error executing partial close: {e}")
            return None
    
    def _should_use_iceberg(self, order_request: OrderRequest) -> bool:
        """
        Determine if order should be split into iceberg as per algorithm
        """
        try:
            # Get current market data
            orderbook = self.api.get_orderbook(order_request.symbol)
            if not orderbook:
                return False
            
            avg_order_size = self._calculate_avg_order_size(orderbook)
            
            # Use iceberg if order is larger than 5x average (algorithm specification)
            return order_request.size > avg_order_size * 5
            
        except Exception as e:
            self.logger.error(f"Error checking iceberg requirement: {e}")
            return False
    
    def _calculate_avg_order_size(self, orderbook: Dict) -> float:
        """Calculate average order size from orderbook"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            all_sizes = []
            for bid in bids[:10]:  # Top 10 levels
                all_sizes.append(float(bid[1]))
            for ask in asks[:10]:
                all_sizes.append(float(ask[1]))
            
            if all_sizes:
                return sum(all_sizes) / len(all_sizes)
            else:
                return 1000.0  # Default fallback
                
        except Exception:
            return 1000.0
    
    async def _execute_iceberg_order(self, order_request: OrderRequest) -> Optional[OrderResult]:
        """
        Execute order as iceberg (show only part) as per algorithm specification
        """
        try:
            total_size = order_request.size
            show_size = total_size * self.EXECUTION_PARAMS['iceberg_show_pct']
            
            # Create parent iceberg tracking
            parent_id = str(uuid.uuid4())
            iceberg = IcebergOrder(
                parent_id=parent_id,
                total_size=total_size,
                filled_size=0.0,
                show_size=show_size,
                child_orders=[],
                status=OrderStatus.PENDING,
                symbol=order_request.symbol,
                side=order_request.side,
                price=order_request.price or 0
            )
            
            self.iceberg_orders[parent_id] = iceberg
            
            # Place first child order
            child_request = OrderRequest(
                symbol=order_request.symbol,
                side=order_request.side,
                size=show_size,
                order_type=order_request.order_type,
                price=order_request.price,
                time_in_force=order_request.time_in_force,
                parent_id=parent_id,
                metadata=order_request.metadata
            )
            
            child_result = await self._execute_single_order(child_request)
            if child_result:
                iceberg.child_orders.append(child_result.order_id)
                iceberg.status = OrderStatus.SUBMITTED
                
                # Return parent order result
                return OrderResult(
                    order_id=parent_id,
                    status=OrderStatus.SUBMITTED,
                    filled_size=child_result.filled_size,
                    avg_price=child_result.avg_price,
                    total_cost=child_result.total_cost,
                    slippage_pct=child_result.slippage_pct,
                    execution_time_ms=child_result.execution_time_ms,
                    fees=child_result.fees,
                    remaining_size=total_size - child_result.filled_size,
                    metadata={'iceberg': True, 'child_orders': iceberg.child_orders}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing iceberg order: {e}")
            return None
    
    async def _execute_single_order(self, order_request: OrderRequest) -> Optional[OrderResult]:
        """
        Execute single order with retry logic as per algorithm specification
        """
        attempts = 0
        start_time = time.time()
        
        while attempts < self.EXECUTION_PARAMS['retry_attempts']:
            try:
                # Place order
                order_result = await self._place_order(order_request)
                if not order_result:
                    attempts += 1
                    await asyncio.sleep(self.EXECUTION_PARAMS['retry_delay_sec'])
                    continue
                
                # Wait for fill or timeout
                fill_result = await self._wait_for_fill(
                    order_result['order_id'],
                    self.EXECUTION_PARAMS['fill_timeout_sec']
                )
                
                if fill_result['status'] == 'filled':
                    # Check slippage
                    slippage_ok = self._check_slippage_acceptable(
                        order_request.price or fill_result['avg_price'],
                        fill_result['avg_price'],
                        order_request.side
                    )
                    
                    if not slippage_ok:
                        self.logger.warning(f"High slippage detected: {fill_result.get('slippage_pct', 0):.3%}")
                    
                    execution_time = (time.time() - start_time) * 1000  # ms
                    
                    result = OrderResult(
                        order_id=fill_result['order_id'],
                        status=OrderStatus.FILLED,
                        filled_size=fill_result['filled_size'],
                        avg_price=fill_result['avg_price'],
                        total_cost=fill_result['filled_size'] * fill_result['avg_price'],
                        slippage_pct=fill_result.get('slippage_pct', 0),
                        execution_time_ms=execution_time,
                        fees=fill_result.get('fees', 0),
                        remaining_size=0,
                        metadata=order_request.metadata
                    )
                    
                    self._record_execution(result)
                    return result
                
                elif fill_result['status'] == 'partially_filled':
                    filled_ratio = fill_result['filled_size'] / order_request.size
                    if filled_ratio >= self.EXECUTION_PARAMS['partial_fill_min']:
                        # Accept partial fill
                        execution_time = (time.time() - start_time) * 1000
                        
                        result = OrderResult(
                            order_id=fill_result['order_id'],
                            status=OrderStatus.PARTIALLY_FILLED,
                            filled_size=fill_result['filled_size'],
                            avg_price=fill_result['avg_price'],
                            total_cost=fill_result['filled_size'] * fill_result['avg_price'],
                            slippage_pct=fill_result.get('slippage_pct', 0),
                            execution_time_ms=execution_time,
                            fees=fill_result.get('fees', 0),
                            remaining_size=order_request.size - fill_result['filled_size'],
                            metadata=order_request.metadata
                        )
                        
                        self._record_execution(result)
                        return result
                    else:
                        # Cancel and retry with more aggressive price
                        await self._cancel_order(order_result['order_id'])
                        order_request.price = self._adjust_price_for_retry(
                            order_request.price or fill_result['avg_price'],
                            order_request.side
                        )
                
                attempts += 1
                await asyncio.sleep(self.EXECUTION_PARAMS['retry_delay_sec'])
                
            except Exception as e:
                self.logger.error(f"Error in order execution attempt {attempts + 1}: {e}")
                attempts += 1
                await asyncio.sleep(self.EXECUTION_PARAMS['retry_delay_sec'])
        
        # All attempts failed
        execution_time = (time.time() - start_time) * 1000
        return OrderResult(
            order_id="failed",
            status=OrderStatus.FAILED,
            filled_size=0,
            avg_price=0,
            total_cost=0,
            slippage_pct=0,
            execution_time_ms=execution_time,
            fees=0,
            remaining_size=order_request.size,
            error_message="Max retry attempts exceeded",
            metadata=order_request.metadata
        )
    
    async def _place_order(self, order_request: OrderRequest) -> Optional[Dict]:
        """Place order via exchange API"""
        try:
            # Convert to exchange API format
            order_params = {
                'symbol': order_request.symbol,
                'side': order_request.side,
                'type': order_request.order_type.value,
                'quantity': order_request.size,
                'timeInForce': order_request.time_in_force.value
            }
            
            if order_request.price:
                order_params['price'] = order_request.price
            
            # Place order through exchange API
            result = await self.api.place_order(**order_params)
            return result
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    async def _wait_for_fill(self, order_id: str, timeout_sec: int) -> Dict:
        """Wait for order fill or timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_sec:
            try:
                order_status = await self.api.get_order_status(order_id)
                
                if order_status['status'] in ['FILLED', 'filled']:
                    return {
                        'order_id': order_id,
                        'status': 'filled',
                        'filled_size': float(order_status['executedQty']),
                        'avg_price': float(order_status.get('avgPrice', order_status.get('price', 0))),
                        'fees': float(order_status.get('commission', 0)),
                        'slippage_pct': 0  # Calculate if needed
                    }
                
                elif order_status['status'] in ['PARTIALLY_FILLED', 'partially_filled']:
                    return {
                        'order_id': order_id,
                        'status': 'partially_filled',
                        'filled_size': float(order_status['executedQty']),
                        'avg_price': float(order_status.get('avgPrice', order_status.get('price', 0))),
                        'fees': float(order_status.get('commission', 0)),
                        'slippage_pct': 0
                    }
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                self.logger.error(f"Error checking order status: {e}")
                await asyncio.sleep(1)
        
        # Timeout
        return {
            'order_id': order_id,
            'status': 'timeout',
            'filled_size': 0,
            'avg_price': 0,
            'fees': 0,
            'slippage_pct': 0
        }
    
    async def _cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        try:
            await self.api.cancel_order(order_id)
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def _adjust_price_for_retry(self, price: float, side: str) -> float:
        """Adjust price for retry (more aggressive)"""
        offset = self.EXECUTION_PARAMS['entry_offset_pct']
        
        if side == 'buy':
            return price * (1 + offset)  # Higher price for buy
        else:
            return price * (1 - offset)  # Lower price for sell
    
    def _check_slippage_acceptable(self, expected_price: float, actual_price: float, side: str) -> bool:
        """Check if slippage is within acceptable range"""
        try:
            if expected_price == 0:
                return True
            
            if side == 'buy':
                slippage = (actual_price - expected_price) / expected_price
            else:
                slippage = (expected_price - actual_price) / expected_price
            
            return abs(slippage) <= self.EXECUTION_PARAMS['max_slippage_pct']
            
        except Exception:
            return True
    
    def _record_execution(self, result: OrderResult):
        """Record execution for analysis"""
        execution_record = {
            'timestamp': datetime.now().isoformat(),
            'order_id': result.order_id,
            'status': result.status.value,
            'filled_size': result.filled_size,
            'avg_price': result.avg_price,
            'slippage_pct': result.slippage_pct,
            'execution_time_ms': result.execution_time_ms,
            'fees': result.fees,
            'metadata': result.metadata
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    async def update_iceberg_orders(self):
        """Update iceberg orders - refill when executed as per algorithm"""
        for parent_id, iceberg in list(self.iceberg_orders.items()):
            if iceberg.status != OrderStatus.SUBMITTED:
                continue
            
            try:
                # Check child orders
                for child_id in iceberg.child_orders:
                    status = await self.api.get_order_status(child_id)
                    
                    if status['status'] in ['FILLED', 'filled']:
                        filled_size = float(status['executedQty'])
                        iceberg.filled_size += filled_size
                        
                        # Place next slice if not complete
                        remaining = iceberg.total_size - iceberg.filled_size
                        if remaining > 0:
                            next_size = min(remaining, iceberg.show_size)
                            
                            # Create new child order
                            child_request = OrderRequest(
                                symbol=iceberg.symbol,
                                side=iceberg.side,
                                size=next_size,
                                order_type=OrderType.LIMIT,
                                price=iceberg.price,
                                parent_id=parent_id
                            )
                            
                            child_result = await self._execute_single_order(child_request)
                            if child_result:
                                iceberg.child_orders.append(child_result.order_id)
                        else:
                            iceberg.status = OrderStatus.FILLED
                            
            except Exception as e:
                self.logger.error(f"Error updating iceberg order {parent_id}: {e}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution performance statistics"""
        if not self.execution_history:
            return {}
        
        filled_executions = [
            ex for ex in self.execution_history 
            if ex['status'] in ['FILLED', 'PARTIALLY_FILLED']
        ]
        
        if not filled_executions:
            return {}
        
        avg_execution_time = sum(ex['execution_time_ms'] for ex in filled_executions) / len(filled_executions)
        avg_slippage = sum(abs(ex['slippage_pct']) for ex in filled_executions) / len(filled_executions)
        fill_rate = len(filled_executions) / len(self.execution_history)
        
        return {
            'total_executions': len(self.execution_history),
            'filled_executions': len(filled_executions),
            'fill_rate': fill_rate,
            'avg_execution_time_ms': avg_execution_time,
            'avg_slippage_pct': avg_slippage,
            'active_icebergs': len([ib for ib in self.iceberg_orders.values() if ib.status == OrderStatus.SUBMITTED])
        }