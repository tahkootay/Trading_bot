"""Structured logging configuration."""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.typing import FilteringBoundLogger


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[Path] = None,
) -> None:
    """Setup structured logging with structlog."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Disable noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    # Structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a structured logger."""
    return structlog.get_logger(name)


class TradingLogger:
    """Specialized logger for trading operations."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        price: float,
        reasoning: str,
        features: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log trading signal."""
        self.logger.info(
            "Trading signal generated",
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            reasoning=reasoning,
            features=features or {},
        )
    
    def log_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float],
        order_type: str,
        status: str,
    ) -> None:
        """Log order action."""
        self.logger.info(
            "Order action",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            status=status,
        )
    
    def log_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        current_price: float,
        pnl: float,
        action: str,
    ) -> None:
        """Log position update."""
        self.logger.info(
            "Position update",
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            current_price=current_price,
            pnl=pnl,
            action=action,
        )
    
    def log_risk_event(
        self,
        event_type: str,
        symbol: str,
        risk_level: str,
        details: Dict[str, Any],
        action_taken: str,
    ) -> None:
        """Log risk management event."""
        self.logger.warning(
            "Risk event",
            event_type=event_type,
            symbol=symbol,
            risk_level=risk_level,
            details=details,
            action_taken=action_taken,
        )
    
    def log_performance(
        self,
        period: str,
        total_trades: int,
        win_rate: float,
        profit_factor: float,
        pnl: float,
        max_drawdown: float,
    ) -> None:
        """Log performance metrics."""
        self.logger.info(
            "Performance update",
            period=period,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            pnl=pnl,
            max_drawdown=max_drawdown,
        )
    
    def log_market_data(
        self,
        symbol: str,
        timestamp: str,
        price: float,
        volume: float,
        data_source: str,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Log market data received."""
        self.logger.debug(
            "Market data received",
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            volume=volume,
            data_source=data_source,
            latency_ms=latency_ms,
        )
    
    def log_model_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction: float,
        confidence: float,
        features_count: int,
        inference_time_ms: float,
    ) -> None:
        """Log ML model prediction."""
        self.logger.debug(
            "Model prediction",
            model_name=model_name,
            symbol=symbol,
            prediction=prediction,
            confidence=confidence,
            features_count=features_count,
            inference_time_ms=inference_time_ms,
        )
    
    def log_system_event(
        self,
        event_type: str,
        component: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log system event."""
        self.logger.info(
            "System event",
            event_type=event_type,
            component=component,
            status=status,
            details=details or {},
        )
    
    def log_error(
        self,
        error_type: str,
        component: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
    ) -> None:
        """Log error with context."""
        self.logger.error(
            "Error occurred",
            error_type=error_type,
            component=component,
            error_message=error_message,
            details=details or {},
            exc_info=exc_info,
        )


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Simple logger setup for compatibility."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create handler if not exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger