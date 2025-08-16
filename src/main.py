"""Main trading bot application."""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import click

# Import bot components
from .data_collector.market_data_collector import MarketDataCollector
from .models.ml_models import MLModelPredictor
from .signal_generator.signal_generator import TradingSignalGenerator
from .risk_manager.risk_manager import RiskManager
from .execution.order_manager import OrderManager
from .data_collector.bybit_client import BybitHTTPClient
from .utils.config import load_config, Settings
from .utils.logger import setup_logging, TradingLogger


class TradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config: Settings = load_config(config_path)
        
        # Setup logging
        setup_logging(
            log_level=self.config.monitoring.log_level,
            log_format=self.config.monitoring.log_format,
        )
        
        self.logger = TradingLogger("trading_bot")
        
        # Initialize components
        self.market_data_collector: Optional[MarketDataCollector] = None
        self.ml_predictor: Optional[MLModelPredictor] = None
        self.signal_generator: Optional[TradingSignalGenerator] = None
        self.risk_manager: Optional[RiskManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.exchange_client: Optional[BybitHTTPClient] = None
        
        # State
        self.is_running = False
        self.shutdown_requested = False
        
        # Main loop task
        self.main_task: Optional[asyncio.Task] = None
        
        self.logger.log_system_event(
            event_type="bot_initialized",
            component="trading_bot",
            status="ready",
            details={
                "environment": self.config.environment,
                "paper_trading": self.config.paper_trading,
                "symbol": self.config.trading.symbol,
            },
        )
    
    async def initialize(self) -> bool:
        """Initialize all bot components."""
        try:
            self.logger.log_system_event(
                event_type="bot_initialization_start",
                component="trading_bot",
                status="starting",
            )
            
            # Initialize exchange client
            self.exchange_client = BybitHTTPClient(
                api_key=self.config.exchange.api_key,
                api_secret=self.config.exchange.api_secret,
                testnet=self.config.exchange.testnet,
                rate_limit=self.config.exchange.rate_limit,
            )
            
            # Initialize market data collector
            self.market_data_collector = MarketDataCollector(
                api_key=self.config.exchange.api_key,
                api_secret=self.config.exchange.api_secret,
                testnet=self.config.exchange.testnet,
                symbols=[self.config.trading.symbol],
                timeframes=self.config.trading.timeframes,
            )
            
            # Initialize ML models
            self.ml_predictor = MLModelPredictor(
                models_path=self.config.models_path,
                ensemble_weights=self.config.ml.ensemble_weights,
            )
            
            # Load existing models or create placeholders
            models_loaded = self.ml_predictor.load_models(self.config.ml.model_version)
            if not models_loaded:
                self.logger.log_system_event(
                    event_type="models_not_found",
                    component="trading_bot",
                    status="warning",
                    details={"model_version": self.config.ml.model_version},
                )
            
            # Initialize risk manager
            risk_config = {
                "max_daily_loss": self.config.risk.max_daily_loss,
                "max_drawdown": self.config.risk.max_drawdown,
                "max_position_size": self.config.trading.max_position_size,
                "max_consecutive_losses": self.config.risk.max_consecutive_losses,
                "max_correlation": self.config.risk.max_correlation,
                "max_positions": self.config.risk.max_positions,
                "pause_after_losses_hours": self.config.risk.pause_after_losses_hours,
                "position_sizing_method": self.config.risk.position_sizing_method,
                "kelly_fraction": self.config.risk.kelly_fraction,
            }
            self.risk_manager = RiskManager(risk_config)
            
            # Initialize order manager
            execution_config = {
                "max_slippage": getattr(self.config, 'execution', {}).get('max_slippage', 0.002),
                "order_timeout": getattr(self.config, 'execution', {}).get('order_timeout', 30),
                "min_order_size": self.config.trading.min_order_size,
                "liquidity_threshold": getattr(self.config, 'execution', {}).get('liquidity_threshold', 1000.0),
                "iceberg_chunk_size": getattr(self.config, 'execution', {}).get('iceberg_chunk_size', 0.1),
                "twap_intervals": getattr(self.config, 'execution', {}).get('twap_intervals', 5),
            }
            self.order_manager = OrderManager(self.exchange_client, execution_config)
            
            # Initialize signal generator
            signal_config = {
                "min_signal_confidence": self.config.trading.min_signal_confidence,
                "min_volume_ratio": self.config.trading.min_volume_ratio,
                "min_adx": self.config.trading.min_adx,
            }
            self.signal_generator = TradingSignalGenerator(
                market_data_collector=self.market_data_collector,
                ml_predictor=self.ml_predictor,
                config=signal_config,
            )
            
            # Start market data collection
            await self.market_data_collector.start()
            
            # Wait for initial data
            await asyncio.sleep(5)
            
            # Verify data is available
            if not self.market_data_collector.is_data_ready(self.config.trading.symbol):
                raise RuntimeError("Market data not ready after initialization")
            
            self.logger.log_system_event(
                event_type="bot_initialization_complete",
                component="trading_bot",
                status="success",
            )
            
            return True
        
        except Exception as e:
            self.logger.log_error(
                error_type="bot_initialization_failed",
                component="trading_bot",
                error_message=str(e),
            )
            return False
    
    async def start(self) -> None:
        """Start the trading bot."""
        if self.is_running:
            return
        
        # Initialize if not done yet
        if not self.market_data_collector:
            if not await self.initialize():
                return
        
        self.is_running = True
        
        self.logger.log_system_event(
            event_type="bot_start",
            component="trading_bot",
            status="starting",
        )
        
        # Start main trading loop
        self.main_task = asyncio.create_task(self._main_loop())
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        try:
            await self.main_task
        except asyncio.CancelledError:
            self.logger.log_system_event(
                event_type="bot_cancelled",
                component="trading_bot",
                status="cancelled",
            )
        except Exception as e:
            self.logger.log_error(
                error_type="bot_main_loop_failed",
                component="trading_bot",
                error_message=str(e),
            )
        finally:
            await self.shutdown()
    
    async def _main_loop(self) -> None:
        """Main trading loop."""
        self.logger.log_system_event(
            event_type="main_loop_start",
            component="trading_bot",
            status="running",
        )
        
        last_signal_check = datetime.min
        signal_check_interval = 30  # Check for signals every 30 seconds
        
        while self.is_running and not self.shutdown_requested:
            try:
                current_time = datetime.now()
                
                # Check for new signals
                if (current_time - last_signal_check).total_seconds() >= signal_check_interval:
                    await self._check_for_signals()
                    last_signal_check = current_time
                
                # Monitor existing positions
                await self._monitor_positions()
                
                # Update risk metrics
                await self._update_risk_metrics()
                
                # Log heartbeat
                if current_time.minute % 5 == 0 and current_time.second < 30:
                    await self._log_heartbeat()
                
                # Sleep before next iteration
                await asyncio.sleep(1)
            
            except Exception as e:
                self.logger.log_error(
                    error_type="main_loop_iteration_failed",
                    component="trading_bot",
                    error_message=str(e),
                )
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _check_for_signals(self) -> None:
        """Check for new trading signals."""
        try:
            symbol = self.config.trading.symbol
            
            # Generate signal
            signal = await self.signal_generator.generate_signal(symbol)
            
            if not signal:
                return
            
            # Get account info
            account = await self.exchange_client.get_account_info()
            if not account:
                return
            
            # Get current positions
            positions = await self.exchange_client.get_positions(symbol)
            
            # Validate signal with risk manager
            is_valid, risk_action, reason = await self.risk_manager.validate_signal(
                signal=signal,
                account=account,
                current_positions=positions,
            )
            
            if not is_valid:
                self.logger.log_system_event(
                    event_type="signal_rejected",
                    component="trading_bot",
                    status="rejected",
                    details={"reason": reason, "risk_action": risk_action.value if risk_action else None},
                )
                return
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(signal, account)
            
            if position_size <= 0:
                return
            
            # Execute signal
            if not self.config.paper_trading:
                execution = await self.order_manager.execute_signal(
                    signal=signal,
                    position_size=position_size,
                )
                
                if execution:
                    self.logger.log_system_event(
                        event_type="signal_executed",
                        component="trading_bot",
                        status="success",
                        details={
                            "symbol": signal.symbol,
                            "signal_type": signal.signal_type.value,
                            "position_size": position_size,
                            "execution_id": execution.order_id,
                        },
                    )
                    
                    # Set stop loss and take profit
                    if signal.stop_loss:
                        await self.order_manager.set_stop_loss(symbol, signal.stop_loss)
                    
                    if signal.take_profit:
                        await self.order_manager.set_take_profit(symbol, signal.take_profit)
            else:
                # Paper trading - just log the signal
                self.logger.log_system_event(
                    event_type="paper_trade_signal",
                    component="trading_bot",
                    status="simulated",
                    details={
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type.value,
                        "confidence": signal.confidence,
                        "position_size": position_size,
                        "entry_price": signal.price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                    },
                )
        
        except Exception as e:
            self.logger.log_error(
                error_type="signal_check_failed",
                component="trading_bot",
                error_message=str(e),
            )
    
    async def _monitor_positions(self) -> None:
        """Monitor existing positions."""
        try:
            if self.config.paper_trading:
                return  # Skip position monitoring in paper trading
            
            # Get current positions
            positions = await self.exchange_client.get_positions()
            
            for position in positions:
                # Update risk manager with position
                self.risk_manager._update_positions([position])
                
                # Check for position management rules
                # (trailing stops, time stops, etc.)
                await self._apply_position_management(position)
        
        except Exception as e:
            self.logger.log_error(
                error_type="position_monitoring_failed",
                component="trading_bot",
                error_message=str(e),
            )
    
    async def _apply_position_management(self, position) -> None:
        """Apply position management rules."""
        try:
            # Time-based stops
            time_in_position = datetime.now() - position.created_at
            max_time = self.config.trading.time_stop_hours * 3600  # Convert to seconds
            
            if time_in_position.total_seconds() > max_time:
                await self.order_manager.close_position(
                    symbol=position.symbol,
                    execution_strategy="MARKET",
                )
                
                self.logger.log_system_event(
                    event_type="time_stop_triggered",
                    component="trading_bot",
                    status="executed",
                    details={"symbol": position.symbol, "time_in_position": str(time_in_position)},
                )
        
        except Exception as e:
            self.logger.log_error(
                error_type="position_management_failed",
                component="trading_bot",
                error_message=str(e),
            )
    
    async def _update_risk_metrics(self) -> None:
        """Update and monitor risk metrics."""
        try:
            # Get current account state
            if not self.config.paper_trading:
                account = await self.exchange_client.get_account_info()
                positions = await self.exchange_client.get_positions()
                
                if account and positions is not None:
                    self.risk_manager._update_account_state(account)
                    self.risk_manager._update_positions(positions)
            
            # Get risk metrics
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            # Check for risk events
            recent_events = self.risk_manager.get_recent_risk_events(hours=1)
            
            if recent_events:
                for event in recent_events[-3:]:  # Log last 3 events
                    self.logger.log_risk_event(
                        event_type=event.risk_type,
                        symbol=event.symbol,
                        risk_level=event.risk_level.value,
                        details=event.details,
                        action_taken=event.action_taken.value,
                    )
        
        except Exception as e:
            self.logger.log_error(
                error_type="risk_metrics_update_failed",
                component="trading_bot",
                error_message=str(e),
            )
    
    async def _log_heartbeat(self) -> None:
        """Log system heartbeat."""
        try:
            # Get system status
            market_data_status = self.market_data_collector.get_health_status()
            signal_performance = self.signal_generator.get_signal_performance(days=1)
            execution_stats = self.order_manager.get_execution_statistics()
            model_status = self.ml_predictor.get_model_status()
            
            self.logger.log_system_event(
                event_type="heartbeat",
                component="trading_bot",
                status="running",
                details={
                    "uptime": str(datetime.now()),
                    "market_data": market_data_status.get("is_running", False),
                    "recent_signals": signal_performance.get("total_signals", 0),
                    "recent_executions": execution_stats.get("total_executions", 0),
                    "models_loaded": len(model_status.get("models_loaded", [])),
                },
            )
        
        except Exception as e:
            self.logger.log_error(
                error_type="heartbeat_failed",
                component="trading_bot",
                error_message=str(e),
            )
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.log_system_event(
                event_type="shutdown_signal_received",
                component="trading_bot",
                status="shutting_down",
                details={"signal": signum},
            )
            self.shutdown_requested = True
            
            if self.main_task:
                self.main_task.cancel()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the bot."""
        if not self.is_running:
            return
        
        self.logger.log_system_event(
            event_type="bot_shutdown_start",
            component="trading_bot",
            status="shutting_down",
        )
        
        self.is_running = False
        
        try:
            # Stop market data collection
            if self.market_data_collector:
                await self.market_data_collector.stop()
            
            # Cancel any remaining orders (if in live trading)
            if self.order_manager and not self.config.paper_trading:
                canceled_orders = await self.order_manager.cancel_all_orders()
                if canceled_orders > 0:
                    self.logger.log_system_event(
                        event_type="orders_canceled_on_shutdown",
                        component="trading_bot",
                        status="cleanup",
                        details={"canceled_count": canceled_orders},
                    )
            
            # Close exchange client
            if self.exchange_client:
                await self.exchange_client.__aexit__(None, None, None)
            
            self.logger.log_system_event(
                event_type="bot_shutdown_complete",
                component="trading_bot",
                status="stopped",
            )
        
        except Exception as e:
            self.logger.log_error(
                error_type="bot_shutdown_failed",
                component="trading_bot",
                error_message=str(e),
            )


@click.command()
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--paper-trading", is_flag=True, help="Enable paper trading mode")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def main(config: Optional[str], paper_trading: bool, debug: bool):
    """Start the trading bot."""
    print("🤖 Starting SOL/USDT Trading Bot...")
    print(f"📅 Time: {datetime.now()}")
    
    try:
        # Create bot instance
        bot = TradingBot(config_path=config)
        
        # Override config with CLI flags
        if paper_trading:
            bot.config.paper_trading = True
        if debug:
            bot.config.debug = True
        
        print(f"🔧 Environment: {bot.config.environment}")
        print(f"📊 Paper Trading: {bot.config.paper_trading}")
        print(f"🎯 Symbol: {bot.config.trading.symbol}")
        print(f"💰 Max Position Size: {bot.config.trading.max_position_size * 100}%")
        print()
        
        # Run bot
        asyncio.run(bot.start())
    
    except KeyboardInterrupt:
        print("\n⏹️  Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()