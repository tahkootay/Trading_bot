"""
Main Trading Bot Implementation for SOL/USDT Algorithm

Coordinates all components and implements the main trading loop
as specified in the algorithm document.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import logging

from .data_collector.market_data_collector import MarketDataCollector
from .feature_engine.market_regime import MarketRegimeClassifier
from .feature_engine.technical_indicators import TechnicalIndicatorCalculator
from .feature_engine.order_flow import OrderFlowAnalyzer
from .signal_generator.setup_detector import SetupDetector
from .models.ml_predictor import MLPredictor
from .risk_manager.position_sizer import PositionSizer, TradeStats
from .risk_manager.risk_manager import RiskManager
from .execution.order_executor import OrderExecutor
from .execution.trade_manager import TradeManager
from .notifications.telegram_notifier import TelegramNotifier
from .utils.logger import TradingLogger
from .utils.config import TradingConfig

class TradingBot:
    """
    Main trading bot implementing the SOL/USDT algorithm specification
    
    Coordinates all components and runs the main trading loop with:
    - Real-time data collection
    - Market regime classification
    - Technical analysis
    - ML predictions
    - Setup detection
    - Risk management
    - Order execution
    - Trade management
    """
    
    def __init__(self, config_path: str):
        self.config = TradingConfig(config_path)
        self.logger = TradingLogger("trading_bot")
        
        # Import algorithm constants from specification
        from .utils.algorithm_constants import (
            TIMEFRAMES, SESSIONS, MARKET_REGIMES, ENTRY_FILTERS, 
            RISK_PARAMS, SL_TP_PARAMS, TIME_STOPS, LIQUIDITY_PARAMS,
            get_regime_entry_filters, validate_trading_session
        )
        
        self.TIMEFRAMES = TIMEFRAMES
        self.SESSIONS = SESSIONS
        self.MARKET_REGIMES = MARKET_REGIMES
        self.ENTRY_FILTERS = ENTRY_FILTERS
        self.RISK_PARAMS = RISK_PARAMS
        
        # Component initialization
        self._initialize_components()
        
        # State tracking
        self.active_trades = {}
        self.market_data = {}
        self.last_update = {}
        self.session_stats = {}
        self.running = False
        
    def _initialize_components(self):
        """Initialize all trading bot components"""
        try:
            # Data collection
            self.market_data_collector = MarketDataCollector(self.config.exchange)
            
            # Analysis components
            self.regime_classifier = MarketRegimeClassifier(self.config.analysis)
            self.technical_indicators = TechnicalIndicatorCalculator()
            self.order_flow_analyzer = OrderFlowAnalyzer(self.config.order_flow)
            
            # ML and signals
            self.ml_predictor = MLPredictor(
                models_path=self.config.models.get('path', './models'),
                config=self.config.ml
            )
            self.setup_detector = SetupDetector(
                indicators={},
                order_flow={},
                regime='normal_range',
                config=self.config.setups
            )
            
            # Risk management
            self.trade_stats = TradeStats(
                total_trades=0,
                win_rate=0.55,
                avg_win=1.5,
                avg_loss=1.0,
                sharpe_ratio=1.2,
                max_drawdown=0.05,
                profit_factor=1.3
            )
            
            self.position_sizer = PositionSizer(
                account_equity=self.config.account.get('initial_balance', 10000),
                trade_stats=self.trade_stats,
                config=self.config.risk
            )
            
            self.risk_manager = RiskManager(self.config.risk)
            
            # Execution
            self.order_executor = OrderExecutor(
                exchange_api=self.market_data_collector.exchange,
                config=self.config.execution
            )
            
            # Notifications
            self.telegram_notifier = TelegramNotifier(self.config.notifications)
            
            self.logger.log_system_event(
                event_type="components_initialized",
                component="trading_bot",
                status="success"
            )
            
        except Exception as e:
            self.logger.log_error(
                error_type="component_initialization_failed",
                component="trading_bot",
                error_message=str(e)
            )
            raise
    
    async def run(self):
        """Main trading loop implementation"""
        self.logger.log_system_event(
            event_type="bot_started",
            component="trading_bot",
            status="starting"
        )
        
        self.running = True
        
        try:
            while self.running:
                loop_start = time.time()
                
                # 1. Update market data
                await self._update_market_data()
                
                # 2. Classify market regime
                regime = self._classify_market_regime()
                
                # 3. Calculate technical indicators
                indicators = self._calculate_indicators(regime)
                
                # 4. Analyze order flow
                order_flow = self._analyze_order_flow()
                
                # 5. Get ML predictions
                ml_prediction = self._get_ml_predictions(regime)
                
                # 6. Update existing positions
                await self._manage_positions()
                
                # 7. Check for new setups
                if self._should_look_for_setups():
                    await self._check_for_new_setups(indicators, order_flow, regime, ml_prediction)
                
                # 8. Update iceberg orders
                await self.order_executor.update_iceberg_orders()
                
                # 9. Risk management checks
                await self._perform_risk_checks()
                
                # 10. Update statistics and notifications
                self._update_session_stats()
                
                # Calculate loop time and sleep
                loop_time = time.time() - loop_start
                sleep_time = max(0, 1.0 - loop_time)  # Target 1-second loops
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            self.logger.log_error(
                error_type="main_loop_error",
                component="trading_bot",
                error_message=str(e)
            )
            await self._emergency_shutdown()
        
        finally:
            await self._shutdown()
    
    async def _update_market_data(self):
        """Fetch all required market data"""
        try:
            data = {
                'timestamp': time.time(),
                'ohlcv': {},
                'trades': [],
                'orderbook': {},
                'btc_data': {}
            }
            
            # Fetch OHLCV for all timeframes
            for tf_name, tf_value in self.TIMEFRAMES.items():
                try:
                    candles = await self.market_data_collector.get_historical_klines(
                        symbol='SOLUSDT',
                        interval=tf_value,
                        limit=200
                    )
                    data['ohlcv'][tf_name] = candles
                except Exception as e:
                    self.logger.log_error(
                        error_type="data_fetch_failed",
                        component="trading_bot",
                        error_message=f"Failed to fetch {tf_name} data: {e}"
                    )
            
            # Fetch recent trades
            try:
                trades = await self.market_data_collector.get_recent_trades(
                    symbol='SOLUSDT',
                    limit=1000
                )
                data['trades'] = trades
            except Exception as e:
                self.logger.log_error(
                    error_type="trades_fetch_failed",
                    component="trading_bot",
                    error_message=str(e)
                )
            
            # Fetch orderbook
            try:
                orderbook = await self.market_data_collector.get_orderbook(
                    symbol='SOLUSDT',
                    limit=20
                )
                data['orderbook'] = orderbook
            except Exception as e:
                self.logger.log_error(
                    error_type="orderbook_fetch_failed",
                    component="trading_bot",
                    error_message=str(e)
                )
            
            # Fetch BTC data for correlation
            try:
                btc_candles = await self.market_data_collector.get_historical_klines(
                    symbol='BTCUSDT',
                    interval='5m',
                    limit=100
                )
                data['btc_data'] = btc_candles
            except Exception as e:
                self.logger.log_error(
                    error_type="btc_data_fetch_failed",
                    component="trading_bot",
                    error_message=str(e)
                )
            
            # Calculate derived metrics
            if 'primary' in data['ohlcv'] and not data['ohlcv']['primary'].empty:
                primary_data = data['ohlcv']['primary']
                data['close'] = primary_data['close'].iloc[-1]
                data['volume'] = primary_data['volume'].iloc[-1]
                
                # Calculate ATR
                indicators = self.technical_indicators.calculate_all(
                    primary_data, 'SOLUSDT', 'M5'
                )
                if indicators:
                    data['atr'] = indicators.atr
            
            self.market_data = data
            self.last_update['market_data'] = time.time()
            
        except Exception as e:
            self.logger.log_error(
                error_type="market_data_update_failed",
                component="trading_bot",
                error_message=str(e)
            )
    
    def _classify_market_regime(self) -> str:
        """Classify current market regime"""
        try:
            if not self.market_data.get('ohlcv', {}).get('primary', {}).empty:
                primary_data = self.market_data['ohlcv']['primary']
                
                # Prepare market data for regime classification
                regime_data = {
                    'close': primary_data['close'].tolist(),
                    'volume': primary_data['volume'].tolist(),
                    'atr': self.market_data.get('atr', 0),
                    'atr_20d_avg': primary_data['close'].rolling(20).std().iloc[-1] if len(primary_data) >= 20 else self.market_data.get('atr', 0)
                }
                
                # Add EMAs
                for period in [8, 13, 21, 34, 55]:
                    ema_key = f'ema_{period}'
                    regime_data[ema_key] = primary_data['close'].ewm(span=period).mean().iloc[-1]
                
                # Add volume average
                regime_data['volume_avg_50'] = primary_data['volume'].rolling(50).mean().iloc[-1] if len(primary_data) >= 50 else regime_data['volume']
                
                regime_signals = self.regime_classifier.classify(regime_data)
                return regime_signals.regime.value
            
            return 'normal_range'
            
        except Exception as e:
            self.logger.log_error(
                error_type="regime_classification_failed",
                component="trading_bot",
                error_message=str(e)
            )
            return 'normal_range'
    
    def _calculate_indicators(self, regime: str) -> Dict:
        """Calculate all technical indicators"""
        try:
            if not self.market_data.get('ohlcv', {}).get('primary', {}).empty:
                primary_data = self.market_data['ohlcv']['primary']
                
                indicators = self.technical_indicators.calculate_all(
                    primary_data, 'SOLUSDT', 'M5'
                )
                
                if indicators:
                    # Convert to dict format for compatibility
                    indicators_dict = {
                        'close': primary_data['close'].tolist(),
                        'high': primary_data['high'].tolist(),
                        'low': primary_data['low'].tolist(),
                        'open': primary_data['open'].tolist(),
                        'volume': primary_data['volume'].tolist(),
                        'atr': indicators.atr,
                        'rsi': indicators.rsi,
                        'adx': indicators.adx,
                        'ema_8': indicators.ema_8,
                        'ema_13': indicators.ema_13,
                        'ema_21': indicators.ema_21,
                        'ema_34': indicators.ema_34,
                        'ema_55': indicators.ema_55,
                        'vwap': indicators.vwap,
                        'vwap_upper_2sigma': indicators.vwap_std2_upper,
                        'vwap_lower_2sigma': indicators.vwap_std2_lower,
                        'regime': regime
                    }
                    
                    # Add z-score volume
                    zvol_data = self.technical_indicators._calculate_z_score_volume(primary_data)
                    indicators_dict['zvol'] = zvol_data.get('zvol', 0)
                    
                    # Add swing points
                    swing_data = self.technical_indicators._identify_swing_points(primary_data)
                    indicators_dict['swing_points'] = swing_data.get('swing_points', [])
                    indicators_dict['nearest_resistance'] = swing_data.get('nearest_swing_high', indicators_dict['close'][-1] * 1.02)
                    indicators_dict['nearest_support'] = swing_data.get('nearest_swing_low', indicators_dict['close'][-1] * 0.98)
                    
                    return indicators_dict
            
            return {}
            
        except Exception as e:
            self.logger.log_error(
                error_type="indicators_calculation_failed",
                component="trading_bot",
                error_message=str(e)
            )
            return {}
    
    def _analyze_order_flow(self) -> Dict:
        """Analyze order flow"""
        try:
            trades_data = self.market_data.get('trades', [])
            orderbook_data = self.market_data.get('orderbook', {})
            
            if trades_data and orderbook_data:
                signals = self.order_flow_analyzer.analyze(trades_data, orderbook_data)
                
                return {
                    'delta': signals.delta,
                    'cvd': signals.cvd,
                    'cvd_trend': signals.cvd_trend,
                    'imbalance': signals.imbalance,
                    'large_orders': signals.large_orders,
                    'absorption': signals.absorption,
                    'intensity': signals.intensity,
                    'aggressive_buyers': signals.aggressive_buyers,
                    'aggressive_sellers': signals.aggressive_sellers,
                    'order_imbalance': signals.imbalance
                }
            
            return {}
            
        except Exception as e:
            self.logger.log_error(
                error_type="order_flow_analysis_failed",
                component="trading_bot",
                error_message=str(e)
            )
            return {}
    
    def _get_ml_predictions(self, regime: str) -> Dict:
        """Get ML predictions"""
        try:
            # Prepare market data for ML
            ml_market_data = {}
            
            if self.market_data.get('ohlcv', {}).get('primary', {}).empty is False:
                primary_data = self.market_data['ohlcv']['primary']
                ml_market_data = {
                    'close': primary_data['close'].tolist(),
                    'volume': primary_data['volume'].iloc[-1],
                    'volume_ma': primary_data['volume'].rolling(20).mean().iloc[-1],
                    'atr': self.market_data.get('atr', 0),
                    'rsi': 50,  # Will be calculated properly in feature calculator
                    'adx': 25,  # Will be calculated properly in feature calculator
                    'ema_21': primary_data['close'].ewm(span=21).mean().iloc[-1],
                    'nearest_resistance': self.market_data.get('close', 0) * 1.02,
                    'nearest_support': self.market_data.get('close', 0) * 0.98,
                    'cvd': 0,
                    'order_imbalance': 1.0,
                    'trade_intensity': 0,
                    'btc_correlation': 0.7,
                    'btc_trend': 0
                }
            
            prediction = self.ml_predictor.predict(ml_market_data, regime)
            
            return {
                'p_up': prediction.p_up,
                'p_down': prediction.p_down,
                'direction': prediction.direction,
                'confidence': prediction.confidence,
                'margin': prediction.margin,
                'agreement': prediction.agreement,
                'passes_ml': prediction.passes_ml,
                'optimal_sl_multiplier': prediction.optimal_sl_multiplier,
                'expected_time': prediction.expected_time
            }
            
        except Exception as e:
            self.logger.log_error(
                error_type="ml_prediction_failed",
                component="trading_bot",
                error_message=str(e)
            )
            return {'passes_ml': False}
    
    async def _manage_positions(self):
        """Manage all active positions"""
        try:
            current_price = self.market_data.get('close', 0)
            
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    # Update trade manager
                    action = trade['manager'].update(current_price, self.market_data)
                    
                    if action.action == 'close':
                        await self._close_trade(trade_id, action.reason)
                    elif action.action == 'partial_close':
                        await self._partial_close_trade(trade_id, action.data)
                    elif action.action == 'adjust_stop':
                        self._adjust_stop_loss(trade_id, action.data)
                        
                except Exception as e:
                    self.logger.log_error(
                        error_type="position_management_failed",
                        component="trading_bot",
                        error_message=f"Error managing trade {trade_id}: {e}"
                    )
            
        except Exception as e:
            self.logger.log_error(
                error_type="positions_management_failed",
                component="trading_bot",
                error_message=str(e)
            )
    
    def _should_look_for_setups(self) -> bool:
        """Check if we should look for new setups"""
        try:
            # Check trading session
            current_hour = datetime.now(timezone.utc).hour
            if current_hour not in self.SESSIONS['active_hours']:
                return False
            
            # Check if we have capacity for new positions
            max_positions = self.config.risk.get('max_positions', 3)
            if len(self.active_trades) >= max_positions:
                return False
            
            # Check if risk manager allows new trades
            if self.risk_manager.is_trading_paused:
                return False
            
            return True
            
        except Exception as e:
            self.logger.log_error(
                error_type="setup_check_failed",
                component="trading_bot",
                error_message=str(e)
            )
            return False
    
    async def _check_for_new_setups(self, indicators: Dict, order_flow: Dict, regime: str, ml_prediction: Dict):
        """Check for new trading setups"""
        try:
            # Check entry filters first
            if not self._check_entry_filters(indicators, order_flow, regime, ml_prediction):
                return
            
            # Update setup detector with current data
            self.setup_detector.indicators = indicators
            self.setup_detector.order_flow = order_flow
            self.setup_detector.regime = regime
            
            # Detect setup
            setup = self.setup_detector.detect_all_setups()
            
            if not setup:
                return
            
            # Add ML optimal stop loss
            if 'atr' in indicators:
                ml_sl_mult = ml_prediction.get('optimal_sl_multiplier', 1.0)
                setup_dict = {
                    'setup_type': setup.setup_type,
                    'direction': setup.direction,
                    'entry': setup.entry,
                    'stop_loss': setup.stop_loss,
                    'targets': setup.targets,
                    'confidence': setup.confidence,
                    'symbol': 'SOLUSDT'
                }
                
                # Adjust stop loss with ML prediction
                atr = indicators['atr']
                if setup.direction == 'long':
                    setup_dict['stop_loss'] = setup.entry - (ml_sl_mult * atr)
                else:
                    setup_dict['stop_loss'] = setup.entry + (ml_sl_mult * atr)
                
                # Calculate position size
                size_result = self.position_sizer.calculate_position_size(
                    setup_dict,
                    list(self.active_trades.values()),
                    {'regime': regime, **indicators}
                )
                
                setup_dict['position_size'] = size_result.size
                setup_dict['risk_amount'] = size_result.risk_amount
                
                # Final risk checks
                # Mock account and positions for risk check
                from ..utils.types import Account, Position
                mock_account = Account(
                    total_balance=self.position_sizer.equity,
                    available_balance=self.position_sizer.equity * 0.9,
                    symbol="SOLUSDT"
                )
                mock_positions = []
                
                # Execute setup if all checks pass
                if size_result.size > 0:
                    await self._execute_setup(setup_dict)
                    
        except Exception as e:
            self.logger.log_error(
                error_type="setup_detection_failed",
                component="trading_bot",
                error_message=str(e)
            )
    
    def _check_entry_filters(self, indicators: Dict, order_flow: Dict, regime: str, ml_prediction: Dict) -> bool:
        """Check if all entry filters pass using regime-specific thresholds"""
        try:
            # Get regime-specific filters from algorithm constants
            from .utils.algorithm_constants import get_regime_entry_filters
            filters = get_regime_entry_filters(regime)
            
            # Check each filter as per algorithm specification
            checks = {
                'session': datetime.now(timezone.utc).hour in self.SESSIONS['active_hours'],
                'trend': indicators.get('adx', 0) >= filters['adx_min'],
                'volume': indicators.get('zvol', 0) >= filters['zvol_min'],
                'momentum': filters['rsi_range'][0] <= indicators.get('rsi', 50) <= filters['rsi_range'][1],
                'correlation': abs(indicators.get('btc_correlation', 0)) < filters['btc_corr_max'],
                'ml_margin': ml_prediction.get('margin', 0) >= filters['ml_margin_min'],
                'ml_confidence': ml_prediction.get('confidence', 0) >= filters['ml_conf_min'],
                'ml_agreement': ml_prediction.get('agreement', 0) >= filters['ml_agreement_min']
            }
            
            # Log filter results for debugging
            failed_filters = [name for name, passed in checks.items() if not passed]
            if failed_filters:
                self.logger.log_info(f"Entry filters failed for {regime}: {failed_filters}")
                self.logger.log_info(f"Filter thresholds: {filters}")
            
            return all(checks.values())
            
        except Exception as e:
            self.logger.log_error(
                error_type="entry_filters_failed",
                component="trading_bot",
                error_message=str(e)
            )
            return False
    
    async def _execute_setup(self, setup: Dict):
        """Execute a trading setup"""
        try:
            # Generate trade ID
            trade_id = f"trade_{int(time.time())}_{len(self.active_trades)}"
            
            # Execute entry order
            entry_result = await self.order_executor.execute_entry(setup)
            
            if entry_result and entry_result.status.value in ['FILLED', 'PARTIALLY_FILLED']:
                # Create trade manager
                trade = {
                    'id': trade_id,
                    'setup': setup,
                    'entry_result': entry_result,
                    'manager': TradeManager(setup, self.config.trade_management),
                    'created_at': datetime.now()
                }
                
                self.active_trades[trade_id] = trade
                
                # Send notification
                await self.telegram_notifier.send_trade_notification(
                    action="ENTRY",
                    symbol=setup['symbol'],
                    side=setup['direction'],
                    size=entry_result.filled_size,
                    price=entry_result.avg_price,
                    setup_type=setup['setup_type'],
                    confidence=setup['confidence']
                )
                
                self.logger.log_trade(
                    action="entry",
                    symbol=setup['symbol'],
                    side=setup['direction'],
                    size=entry_result.filled_size,
                    price=entry_result.avg_price,
                    metadata={
                        'trade_id': trade_id,
                        'setup_type': setup['setup_type'],
                        'confidence': setup['confidence']
                    }
                )
                
        except Exception as e:
            self.logger.log_error(
                error_type="setup_execution_failed",
                component="trading_bot",
                error_message=str(e)
            )
    
    async def _close_trade(self, trade_id: str, reason: str):
        """Close a trade"""
        try:
            trade = self.active_trades[trade_id]
            
            # Execute close order
            close_result = await self.order_executor.execute_exit(
                trade['setup'],
                reason
            )
            
            if close_result:
                # Calculate PnL
                entry_price = trade['entry_result'].avg_price
                exit_price = close_result.avg_price
                size = close_result.filled_size
                direction = trade['setup']['direction']
                
                if direction == 'long':
                    pnl = (exit_price - entry_price) * size
                else:
                    pnl = (entry_price - exit_price) * size
                
                # Update risk manager
                self.risk_manager.record_trade_result(
                    symbol=trade['setup']['symbol'],
                    side=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=size,
                    pnl=pnl,
                    signal_confidence=trade['setup']['confidence'],
                    trade_duration=datetime.now() - trade['created_at']
                )
                
                # Send notification
                await self.telegram_notifier.send_trade_notification(
                    action="EXIT",
                    symbol=trade['setup']['symbol'],
                    side=direction,
                    size=size,
                    price=exit_price,
                    pnl=pnl,
                    reason=reason
                )
                
                # Remove from active trades
                del self.active_trades[trade_id]
                
                self.logger.log_trade(
                    action="exit",
                    symbol=trade['setup']['symbol'],
                    side=direction,
                    size=size,
                    price=exit_price,
                    metadata={
                        'trade_id': trade_id,
                        'reason': reason,
                        'pnl': pnl
                    }
                )
                
        except Exception as e:
            self.logger.log_error(
                error_type="trade_close_failed",
                component="trading_bot",
                error_message=f"Error closing trade {trade_id}: {e}"
            )
    
    async def _partial_close_trade(self, trade_id: str, close_data: Dict):
        """Execute partial close of trade"""
        try:
            trade = self.active_trades[trade_id]
            
            close_result = await self.order_executor.execute_partial_close(
                trade['setup'],
                close_data['size'],
                close_data.get('reason', 'take_profit')
            )
            
            if close_result:
                # Update trade size
                trade['setup']['size'] = close_data['remaining_size']
                
                # Send notification
                await self.telegram_notifier.send_trade_notification(
                    action="PARTIAL_EXIT",
                    symbol=trade['setup']['symbol'],
                    side=trade['setup']['direction'],
                    size=close_result.filled_size,
                    price=close_result.avg_price,
                    reason=close_data.get('reason', 'take_profit')
                )
                
        except Exception as e:
            self.logger.log_error(
                error_type="partial_close_failed",
                component="trading_bot",
                error_message=f"Error partial closing trade {trade_id}: {e}"
            )
    
    def _adjust_stop_loss(self, trade_id: str, adjust_data: Dict):
        """Adjust stop loss for trade"""
        try:
            if trade_id in self.active_trades:
                trade = self.active_trades[trade_id]
                old_stop = trade['setup']['stop_loss']
                new_stop = adjust_data.get('new_stop', old_stop)
                
                trade['setup']['stop_loss'] = new_stop
                
                self.logger.log_info(
                    f"Stop loss adjusted for {trade_id}: {old_stop:.4f} -> {new_stop:.4f}"
                )
                
        except Exception as e:
            self.logger.log_error(
                error_type="stop_adjustment_failed",
                component="trading_bot",
                error_message=f"Error adjusting stop for trade {trade_id}: {e}"
            )
    
    async def _perform_risk_checks(self):
        """Perform risk management checks"""
        try:
            # Check for emergency conditions
            emergency_action = self.risk_manager.check_emergency_conditions(self.market_data)
            
            if emergency_action:
                if emergency_action.value == 'EMERGENCY_STOP':
                    await self._emergency_shutdown()
                elif emergency_action.value == 'CLOSE_POSITIONS':
                    await self._close_all_positions("emergency")
                    
        except Exception as e:
            self.logger.log_error(
                error_type="risk_checks_failed",
                component="trading_bot",
                error_message=str(e)
            )
    
    def _update_session_stats(self):
        """Update session statistics"""
        try:
            today = datetime.now().date()
            
            if today not in self.session_stats:
                self.session_stats[today] = {
                    'trades_count': 0,
                    'winning_trades': 0,
                    'total_pnl': 0.0,
                    'start_time': datetime.now()
                }
            
            # Update with current active trades count
            self.session_stats[today]['active_trades'] = len(self.active_trades)
            
        except Exception as e:
            self.logger.log_error(
                error_type="stats_update_failed",
                component="trading_bot",
                error_message=str(e)
            )
    
    async def _close_all_positions(self, reason: str = "shutdown"):
        """Close all active positions"""
        try:
            for trade_id in list(self.active_trades.keys()):
                await self._close_trade(trade_id, reason)
                
        except Exception as e:
            self.logger.log_error(
                error_type="close_all_failed",
                component="trading_bot",
                error_message=str(e)
            )
    
    async def _emergency_shutdown(self):
        """Emergency shutdown procedure"""
        try:
            self.logger.log_system_event(
                event_type="emergency_shutdown",
                component="trading_bot",
                status="initiated"
            )
            
            # Close all positions
            await self._close_all_positions("emergency")
            
            # Send emergency notification
            await self.telegram_notifier.send_alert(
                "🚨 EMERGENCY SHUTDOWN",
                "Trading bot emergency shutdown initiated. All positions closed."
            )
            
            # Stop the bot
            self.running = False
            
        except Exception as e:
            self.logger.log_error(
                error_type="emergency_shutdown_failed",
                component="trading_bot",
                error_message=str(e)
            )
    
    async def _shutdown(self):
        """Graceful shutdown procedure"""
        try:
            self.logger.log_system_event(
                event_type="bot_shutdown",
                component="trading_bot",
                status="initiated"
            )
            
            # Close positions if configured to do so
            if self.config.shutdown.get('close_positions_on_shutdown', True):
                await self._close_all_positions("shutdown")
            
            # Send shutdown notification
            await self.telegram_notifier.send_alert(
                "🛑 Bot Shutdown",
                "Trading bot gracefully shutdown."
            )
            
            self.logger.log_system_event(
                event_type="bot_shutdown",
                component="trading_bot",
                status="completed"
            )
            
        except Exception as e:
            self.logger.log_error(
                error_type="shutdown_failed",
                component="trading_bot",
                error_message=str(e)
            )
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            'running': self.running,
            'active_trades': len(self.active_trades),
            'last_update': self.last_update.get('market_data', 0),
            'current_regime': getattr(self, '_current_regime', 'unknown'),
            'session_stats': self.session_stats.get(datetime.now().date(), {}),
            'risk_metrics': self.risk_manager.get_risk_metrics() if hasattr(self, 'risk_manager') else {}
        }