"""
Enhanced Trading Bot Implementation
Full algorithm specification compliance for SOL/USDT
"""

import asyncio
import time
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import logging

from .data_collector.market_data_collector import MarketDataCollector
from .feature_engine.market_regime import MarketRegimeClassifier
from .feature_engine.technical_indicators import TechnicalIndicatorCalculator
from .feature_engine.order_flow import OrderFlowAnalyzer
from .signal_generator.enhanced_setup_detector import EnhancedSetupDetector
from .models.ml_predictor import MLPredictor
from .risk_manager.adaptive_position_sizer import AdaptivePositionSizer, TradeStats
from .risk_manager.risk_manager import RiskManager
from .execution.order_executor import OrderExecutor
from .execution.trade_manager import TradeManager
from .notifications.telegram_notifier import TelegramNotifier
from .utils.logger import TradingLogger
from .utils.config import TradingConfig
from .utils.algorithm_constants import (
    TIMEFRAMES, SESSIONS, MARKET_REGIMES, ENTRY_FILTERS, 
    RISK_PARAMS, SL_TP_PARAMS, TIME_STOPS, LIQUIDITY_PARAMS,
    get_regime_entry_filters, validate_trading_session
)

class EnhancedTradingBot:
    """
    Enhanced Trading Bot implementing full SOL/USDT algorithm specification
    
    Main trading loop exactly as specified in algorithm document:
    1. Fetch market data for all timeframes
    2. Classify market regime
    3. Calculate technical indicators
    4. Analyze order flow
    5. Get ML predictions
    6. Manage existing positions
    7. Check for new setups
    8. Execute trades with full risk management
    """
    
    def __init__(self, config_path: str):
        self.config = TradingConfig(config_path)
        self.logger = TradingLogger("enhanced_trading_bot")
        
        # Component initialization
        self._initialize_components()
        
        # State tracking as per algorithm
        self.active_trades = {}
        self.market_data = {}
        self.last_update = {}
        self.session_stats = {}
        self.running = False
        
    def _initialize_components(self):
        """Initialize all components exactly as per algorithm specification"""
        try:
            # Data collection for all timeframes
            self.market_data_collector = MarketDataCollector(self.config.exchange)
            
            # Analysis components
            self.regime_classifier = MarketRegimeClassifier(self.config.analysis)
            self.technical_indicators = TechnicalIndicatorCalculator()
            self.order_flow_analyzer = OrderFlowAnalyzer(self.config.order_flow)
            
            # Enhanced components per algorithm
            self.setup_detector = EnhancedSetupDetector()
            self.ml_predictor = MLPredictor(
                models_path=self.config.models.get('path', './models'),
                config=self.config.ml
            )
            
            # Risk management with adaptive Kelly
            self.trade_stats = TradeStats(
                total_trades=0,
                win_rate=0.55,
                avg_win=1.5,
                avg_loss=1.0,
                consecutive_wins=0,
                consecutive_losses=0,
                profit_factor=1.5,
                sharpe_ratio=1.2
            )
            
            self.position_sizer = AdaptivePositionSizer(
                account_equity=self.config.account.get('initial_balance', 10000),
                trade_stats=self.trade_stats
            )
            
            self.risk_manager = RiskManager(
                initial_balance=self.config.account.get('initial_balance', 10000),
                config=self.config.risk
            )
            
            # Execution
            self.order_executor = OrderExecutor(
                exchange_api=None,  # Will be set from config
                config=self.config.execution
            )
            
            # Notifications
            if self.config.notifications.get('telegram', {}).get('enabled', False):
                self.notifier = TelegramNotifier(self.config.notifications.telegram)
            else:
                self.notifier = None
                
            self.logger.log_system_event(
                event_type="components_initialized",
                component="enhanced_trading_bot",
                status="ready",
                details={'components': [
                    'data_collector', 'regime_classifier', 'technical_indicators',
                    'order_flow_analyzer', 'setup_detector', 'ml_predictor',
                    'position_sizer', 'risk_manager', 'order_executor'
                ]}
            )
            
        except Exception as e:
            self.logger.log_error(
                error_type="component_initialization_failed",
                component="enhanced_trading_bot",
                error_message=str(e)
            )
            raise
    
    async def run(self):
        """
        Main trading loop exactly as specified in algorithm
        Implements complete flow from data collection to trade execution
        """
        self.running = True
        self.logger.log_system_event(
            event_type="trading_bot_started",
            component="enhanced_trading_bot",
            status="running",
            details={}
        )
        
        while self.running:
            try:
                loop_start_time = time.time()
                
                # 1. Update market data for all timeframes
                market_data = await self.fetch_market_data()
                
                # 2. Classify market regime
                regime = self.regime_classifier.classify(market_data)
                
                # 3. Calculate all technical indicators
                indicators = self.technical_indicators.calculate_all(market_data, regime)
                
                # 4. Analyze order flow
                order_flow = self.order_flow_analyzer.analyze(
                    market_data.get('trades', []),
                    market_data.get('orderbook', {})
                )
                
                # 5. Get ML predictions with regime adjustment
                ml_prediction = self.ml_predictor.predict(market_data, regime)
                
                # 6. Update existing positions
                await self.manage_positions(market_data)
                
                # 7. Update iceberg orders
                await self.order_executor.update_iceberg_orders()
                
                # 8. Risk management checks
                emergency_action = self.risk_manager.check_emergency_conditions(market_data)
                if emergency_action:
                    await self.handle_emergency_action(emergency_action)
                
                # 9. Check for new setups if conditions allow
                if self.should_look_for_setups():
                    await self.check_for_new_setups(indicators, order_flow, regime, ml_prediction)
                
                # 10. Log performance metrics
                loop_duration = time.time() - loop_start_time
                self._log_loop_metrics(loop_duration, regime, ml_prediction)
                
                # Sleep until next iteration (1 second as per algorithm)
                await asyncio.sleep(1)
                
            except Exception as e:
                await self.handle_error(e)
                await asyncio.sleep(5)  # Wait before retrying on error
    
    async def fetch_market_data(self) -> Dict[str, Any]:
        """
        Fetch all required market data as specified in algorithm
        
        Returns:
            Complete market data dictionary with OHLCV, trades, orderbook, BTC data
        """
        try:
            data = {
                'timestamp': time.time(),
                'ohlcv': {},
                'trades': [],
                'orderbook': {},
                'btc_data': {}
            }
            
            # Fetch OHLCV for all timeframes as per algorithm
            for tf_name, tf_value in TIMEFRAMES.items():
                try:
                    ohlcv = await self.market_data_collector.get_candles(
                        'SOL/USDT', tf_value, limit=200
                    )
                    data['ohlcv'][tf_name] = ohlcv
                except Exception as e:
                    self.logger.log_error(
                        error_type="timeframe_data_fetch_failed",
                        component="enhanced_trading_bot",
                        error_message=str(e),
                        details={'timeframe': tf_name}
                    )
            
            # Fetch recent trades for order flow analysis
            data['trades'] = await self.market_data_collector.get_recent_trades(
                'SOL/USDT', limit=1000
            )
            
            # Fetch orderbook for imbalance analysis
            data['orderbook'] = await self.market_data_collector.get_orderbook(
                'SOL/USDT', limit=20
            )
            
            # Fetch BTC data for correlation analysis
            data['btc_data'] = await self.market_data_collector.get_candles(
                'BTC/USDT', '5m', limit=100
            )
            
            # Calculate derived metrics
            if 'primary' in data['ohlcv'] and not data['ohlcv']['primary'].empty:
                primary_data = data['ohlcv']['primary']
                data['close'] = primary_data['close'].iloc[-1]
                data['volume'] = primary_data['volume'].iloc[-1]
                
                # Basic ATR calculation
                if len(primary_data) >= 14:
                    high_low = primary_data['high'] - primary_data['low']
                    high_close_prev = abs(primary_data['high'] - primary_data['close'].shift(1))
                    low_close_prev = abs(primary_data['low'] - primary_data['close'].shift(1))
                    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
                    data['atr'] = true_range.rolling(14).mean().iloc[-1]
                else:
                    data['atr'] = 0
            
            return data
            
        except Exception as e:
            self.logger.log_error(
                error_type="market_data_fetch_failed",
                component="enhanced_trading_bot",
                error_message=str(e)
            )
            return {}
    
    def should_look_for_setups(self) -> bool:
        """
        Check if we should look for new setups as per algorithm specification
        
        Returns:
            True if conditions allow new setup detection
        """
        try:
            # Check trading session
            if not validate_trading_session():
                return False
            
            # Check if we have capacity for new positions
            if len(self.active_trades) >= RISK_PARAMS['max_corr_positions']:
                return False
            
            # Check if risk manager allows new trades
            if self.risk_manager.is_trading_paused:
                return False
            
            # Check if emergency mode is active
            if getattr(self.risk_manager, 'emergency_mode', False):
                return False
            
            return True
            
        except Exception as e:
            self.logger.log_error(
                error_type="setup_check_failed",
                component="enhanced_trading_bot",
                error_message=str(e)
            )
            return False
    
    async def check_for_new_setups(
        self, 
        indicators: Dict, 
        order_flow: Dict, 
        regime: str, 
        ml_prediction
    ):
        """
        Check for new trading setups as per algorithm specification
        
        Args:
            indicators: Technical indicators
            order_flow: Order flow analysis
            regime: Current market regime
            ml_prediction: ML prediction results
        """
        try:
            # Check entry filters first
            if not self.check_entry_filters(indicators, order_flow, regime, ml_prediction):
                return
            
            # Detect setups using enhanced detector
            setup = self.setup_detector.detect_all_setups(
                indicators, order_flow, regime, self.market_data
            )
            
            if not setup:
                return
            
            # Add ML optimal stop loss to setup
            setup_dict = {
                'symbol': 'SOL/USDT',
                'setup_type': setup.setup_type,
                'direction': setup.direction,
                'entry': setup.entry,
                'stop_loss': setup.stop_loss,
                'targets': setup.targets,
                'confidence': setup.confidence,
                'regime': setup.regime,
                'metadata': setup.metadata
            }
            
            # Apply ML optimal stop loss adjustment
            atr = indicators.get('atr', 0)
            if atr > 0:
                ml_sl_multiplier = ml_prediction.optimal_sl_multiplier
                ml_stop = setup.entry - (ml_sl_multiplier * atr if setup.direction == 'long' else -ml_sl_multiplier * atr)
                
                # Use the more conservative stop
                if setup.direction == 'long':
                    setup_dict['stop_loss'] = max(setup.stop_loss, ml_stop)
                else:
                    setup_dict['stop_loss'] = min(setup.stop_loss, ml_stop)
            
            # Calculate position size using adaptive Kelly
            current_positions = list(self.active_trades.values())
            market_conditions = {
                'regime': regime,
                'atr': indicators.get('atr', 0),
                'atr_20d_avg': indicators.get('atr_20d_avg', indicators.get('atr', 0)),
                'volume_5m_usd': indicators.get('volume_5m_usd', 100000),
                'orderbook_depth_usd': indicators.get('orderbook_depth_usd', 50000)
            }
            
            size_result = self.position_sizer.calculate_position_size(
                setup_dict, current_positions, market_conditions
            )
            
            setup_dict['position_size'] = size_result.size
            setup_dict['risk_amount'] = size_result.risk_amount
            setup_dict['risk_pct'] = size_result.risk_pct
            
            # Final risk checks
            risk_check = self.risk_manager.pre_trade_checks(setup_dict, indicators)
            
            if not risk_check['passed']:
                self.logger.log_trade(
                    action="setup_rejected",
                    symbol=setup_dict['symbol'],
                    side=setup_dict['direction'],
                    size=0,
                    price=setup_dict['entry'],
                    metadata={
                        'rejection_reasons': risk_check['reasons'],
                        'setup_type': setup_dict['setup_type'],
                        'confidence': setup_dict['confidence']
                    }
                )
                return
            
            # Execute setup
            await self.execute_setup(setup_dict)
            
        except Exception as e:
            self.logger.log_error(
                error_type="setup_detection_failed",
                component="enhanced_trading_bot",
                error_message=str(e)
            )
    
    def check_entry_filters(
        self, 
        indicators: Dict, 
        order_flow: Dict, 
        regime: str, 
        ml_prediction
    ) -> bool:
        """
        Check if all entry filters pass as per algorithm specification
        
        Returns:
            True if all filters pass
        """
        try:
            # Get regime-specific filters
            filters = get_regime_entry_filters(regime)
            
            # Check each filter as specified in algorithm
            checks = {
                'session': validate_trading_session(),
                'trend': indicators.get('adx', 0) >= filters['adx_min'],
                'volume': indicators.get('zvol', 0) >= filters['zvol_min'],
                'ema_alignment': indicators.get('ema_alignment', 'neutral') in ['bullish', 'bearish', 'strong'],
                'momentum': filters['rsi_range'][0] <= indicators.get('rsi', 50) <= filters['rsi_range'][1],
                'correlation': abs(indicators.get('btc_correlation', 0.7)) < filters['btc_corr_max'],
                'ml': ml_prediction.passes_ml if hasattr(ml_prediction, 'passes_ml') else False
            }
            
            # Log filter results
            failed_filters = [name for name, passed in checks.items() if not passed]
            if failed_filters:
                self.logger.log_system_event(
                    event_type="entry_filters_failed",
                    component="enhanced_trading_bot",
                    status="rejected",
                    details={
                        'failed_filters': failed_filters,
                        'regime': regime,
                        'all_checks': checks
                    }
                )
            
            return all(checks.values())
            
        except Exception as e:
            self.logger.log_error(
                error_type="entry_filter_check_failed",
                component="enhanced_trading_bot",
                error_message=str(e)
            )
            return False
    
    async def execute_setup(self, setup: Dict):
        """
        Execute a trading setup as per algorithm specification
        
        Args:
            setup: Complete setup dictionary with all parameters
        """
        try:
            # Generate unique trade ID
            trade_id = f"SOL_{setup['setup_type']}_{int(time.time())}"
            
            # Execute entry order using enhanced order executor
            entry_result = await self.order_executor.execute_entry(setup)
            
            if entry_result and entry_result.get('filled_size', 0) > 0:
                # Create trade manager for position management
                position_data = {
                    'id': trade_id,
                    'symbol': setup['symbol'],
                    'direction': setup['direction'],
                    'entry': entry_result['avg_price'],
                    'size': entry_result['filled_size'],
                    'stop_loss': setup['stop_loss'],
                    'targets': setup['targets'],
                    'setup_type': setup['setup_type'],
                    'entry_time': time.time(),
                    'metadata': setup.get('metadata', {})
                }
                
                trade_manager = TradeManager(position_data, self.config.trade_management)
                
                trade = {
                    'id': trade_id,
                    'setup': setup,
                    'position': position_data,
                    'entry_result': entry_result,
                    'manager': trade_manager
                }
                
                self.active_trades[trade_id] = trade
                
                # Place protective stop loss order
                await self.place_protective_orders(trade)
                
                # Log successful trade entry
                self.logger.log_trade(
                    action="entry_executed",
                    symbol=setup['symbol'],
                    side=setup['direction'],
                    size=entry_result['filled_size'],
                    price=entry_result['avg_price'],
                    metadata={
                        'trade_id': trade_id,
                        'setup_type': setup['setup_type'],
                        'confidence': setup['confidence'],
                        'risk_pct': setup['risk_pct'],
                        'stop_loss': setup['stop_loss'],
                        'target_count': len(setup['targets'])
                    }
                )
                
                # Send notification
                if self.notifier:
                    await self.notifier.send_trade_notification(
                        f"✅ ENTRY: {setup['direction'].upper()} {setup['symbol']}\n"
                        f"Setup: {setup['setup_type']}\n"
                        f"Entry: ${entry_result['avg_price']:.3f}\n"
                        f"Size: {entry_result['filled_size']:.4f}\n"
                        f"Stop: ${setup['stop_loss']:.3f}\n"
                        f"Confidence: {setup['confidence']:.1%}"
                    )
            
        except Exception as e:
            self.logger.log_error(
                error_type="setup_execution_failed",
                component="enhanced_trading_bot",
                error_message=str(e),
                details={'setup': setup}
            )
    
    async def manage_positions(self, market_data: Dict):
        """
        Manage all active positions as per algorithm specification
        
        Args:
            market_data: Current market data
        """
        try:
            current_price = market_data.get('close', 0)
            if current_price == 0:
                return
            
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    # Update trade manager
                    action = trade['manager'].update(current_price, market_data)
                    
                    if action.action == 'close':
                        await self.close_trade(trade_id, action.reason)
                    elif action.action == 'partial_close':
                        await self.execute_partial_close(trade_id, action.data)
                    elif action.action == 'adjust_stop':
                        await self.adjust_stop_loss(trade_id, action.data)
                        
                except Exception as e:
                    self.logger.log_error(
                        error_type="position_management_failed",
                        component="enhanced_trading_bot",
                        error_message=str(e),
                        details={'trade_id': trade_id}
                    )
                    
        except Exception as e:
            self.logger.log_error(
                error_type="positions_management_failed",
                component="enhanced_trading_bot",
                error_message=str(e)
            )
    
    async def close_trade(self, trade_id: str, reason: str):
        """
        Close a trade completely
        
        Args:
            trade_id: Trade identifier
            reason: Reason for closing
        """
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                return
            
            # Execute close order
            close_result = await self.order_executor.execute_close(trade['position'])
            
            if close_result:
                # Calculate PnL
                entry_price = trade['position']['entry']
                exit_price = close_result['avg_price']
                size = close_result['filled_size']
                direction = trade['position']['direction']
                
                if direction == 'long':
                    pnl = (exit_price - entry_price) * size
                else:
                    pnl = (entry_price - exit_price) * size
                
                # Update risk manager and trade stats
                self.risk_manager.record_trade_result(
                    symbol=trade['position']['symbol'],
                    side=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=size,
                    pnl=pnl,
                    signal_confidence=trade['setup']['confidence'],
                    trade_duration=timedelta(seconds=time.time() - trade['position']['entry_time'])
                )
                
                # Update position sizer stats
                self.position_sizer.update_trade_stats(
                    pnl=pnl,
                    win=pnl > 0,
                    trade_duration_minutes=(time.time() - trade['position']['entry_time']) / 60
                )
                
                # Remove from active trades
                del self.active_trades[trade_id]
                
                # Log trade close
                self.logger.log_trade(
                    action="trade_closed",
                    symbol=trade['position']['symbol'],
                    side=direction,
                    size=size,
                    price=exit_price,
                    metadata={
                        'trade_id': trade_id,
                        'entry_price': entry_price,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * size)) * 100,
                        'reason': reason,
                        'duration_minutes': (time.time() - trade['position']['entry_time']) / 60
                    }
                )
                
                # Send notification
                if self.notifier:
                    await self.notifier.send_trade_notification(
                        f"🔄 CLOSE: {direction.upper()} {trade['position']['symbol']}\n"
                        f"Entry: ${entry_price:.3f} → Exit: ${exit_price:.3f}\n"
                        f"P&L: ${pnl:.2f} ({(pnl/(entry_price*size))*100:+.1f}%)\n"
                        f"Reason: {reason}"
                    )
                    
        except Exception as e:
            self.logger.log_error(
                error_type="trade_close_failed",
                component="enhanced_trading_bot",
                error_message=str(e),
                details={'trade_id': trade_id, 'reason': reason}
            )
    
    async def handle_emergency_action(self, action):
        """Handle emergency risk actions"""
        try:
            if action.name == 'EMERGENCY_STOP':
                # Close all positions immediately
                for trade_id in list(self.active_trades.keys()):
                    await self.close_trade(trade_id, 'emergency_stop')
                
                # Halt trading
                self.risk_manager.emergency_mode = True
                
                if self.notifier:
                    await self.notifier.send_alert("🚨 EMERGENCY STOP ACTIVATED")
                    
        except Exception as e:
            self.logger.log_error(
                error_type="emergency_action_failed",
                component="enhanced_trading_bot",
                error_message=str(e)
            )
    
    async def handle_error(self, error: Exception):
        """Handle errors in main loop"""
        try:
            self.logger.log_error(
                error_type="main_loop_error",
                component="enhanced_trading_bot",
                error_message=str(error)
            )
            
            # Check if critical error
            if self.is_critical_error(error):
                await self.handle_emergency_action(type('EmergencyAction', (), {'name': 'EMERGENCY_STOP'})())
            
        except Exception as e:
            self.logger.log_error(
                error_type="error_handling_failed",
                component="enhanced_trading_bot",
                error_message=str(e)
            )
    
    def is_critical_error(self, error: Exception) -> bool:
        """Determine if error is critical"""
        critical_types = (ConnectionError, TimeoutError, MemoryError)
        return isinstance(error, critical_types)
    
    def _log_loop_metrics(self, loop_duration: float, regime: str, ml_prediction):
        """Log performance metrics for monitoring"""
        try:
            self.logger.log_system_event(
                event_type="trading_loop_completed",
                component="enhanced_trading_bot",
                status="success",
                details={
                    'loop_duration_ms': loop_duration * 1000,
                    'active_trades': len(self.active_trades),
                    'regime': regime,
                    'ml_confidence': getattr(ml_prediction, 'confidence', 0),
                    'ml_passes': getattr(ml_prediction, 'passes_ml', False),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
        except Exception:
            pass  # Don't let logging errors affect main loop
    
    def stop(self):
        """Stop the trading bot gracefully"""
        self.running = False
        self.logger.log_system_event(
            event_type="trading_bot_stopped",
            component="enhanced_trading_bot",
            status="stopped",
            details={}
        )