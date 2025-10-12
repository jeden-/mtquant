"""
Centralized Logging Configuration for MTQuant

Provides centralized logging setup using loguru with support for
console logging, file logging, correlation IDs, and sensitive data masking.
"""

import os
import re
import sys
from typing import Optional, Dict, Any
from loguru import logger
from datetime import datetime


# Sensitive data patterns to mask in logs
SENSITIVE_PATTERNS = [
    (r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)', r'password="***"'),
    (r'api_key["\']?\s*[:=]\s*["\']?([^"\'\s]+)', r'api_key="***"'),
    (r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)', r'token="***"'),
    (r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)', r'secret="***"'),
    (r'key["\']?\s*[:=]\s*["\']?([^"\'\s]+)', r'key="***"'),
]


def mask_sensitive_data(message: str) -> str:
    """Mask sensitive data in log messages.
    
    Args:
        message: Original log message
        
    Returns:
        Message with sensitive data masked
    """
    masked_message = message
    
    for pattern, replacement in SENSITIVE_PATTERNS:
        masked_message = re.sub(pattern, replacement, masked_message, flags=re.IGNORECASE)
    
    return masked_message


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    environment: Optional[str] = None
) -> None:
    """Setup centralized logger configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        rotation: Log rotation size/time
        retention: Log retention period
        environment: Environment name (development, production, etc.)
    """
    # Remove default handler
    logger.remove()
    
    # Get environment from parameter or environment variable
    env = environment or os.getenv("ENVIRONMENT", "development")
    
    # Console logging with colors
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # File logging format (JSON for production, readable for development)
    if env == "production":
        file_format = (
            '{{"timestamp": "{time:YYYY-MM-DD HH:mm:ss}", '
            '"level": "{level}", '
            '"logger": "{name}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}", '
            '"extra": {extra}}}'
        )
    else:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message} | "
            "{extra}"
        )
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=console_format,
        level=level,
        colorize=True,
        filter=lambda record: mask_sensitive_data(record["message"])
    )
    
    # Add file handler if specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            serialize=env == "production",
            filter=lambda record: mask_sensitive_data(record["message"])
        )
    
    # Log startup message
    logger.info(f"Logger initialized - Level: {level}, Environment: {env}")


def get_logger(name: str) -> Any:
    """Get logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


def log_with_context(
    level: str,
    message: str,
    correlation_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    symbol: Optional[str] = None,
    order_id: Optional[str] = None,
    position_id: Optional[str] = None,
    broker_id: Optional[str] = None,
    **kwargs
) -> None:
    """Log message with trading context.
    
    Args:
        level: Log level
        message: Log message
        correlation_id: Request correlation ID
        agent_id: Agent ID
        symbol: Trading symbol
        order_id: Order ID
        position_id: Position ID
        broker_id: Broker ID
        **kwargs: Additional context data
    """
    context = {}
    
    if correlation_id:
        context["correlation_id"] = correlation_id
    if agent_id:
        context["agent_id"] = agent_id
    if symbol:
        context["symbol"] = symbol
    if order_id:
        context["order_id"] = order_id
    if position_id:
        context["position_id"] = position_id
    if broker_id:
        context["broker_id"] = broker_id
    
    context.update(kwargs)
    
    # Get the appropriate log method
    log_method = getattr(logger, level.lower())
    log_method(message, extra=context)


def log_trade_event(
    event_type: str,
    message: str,
    agent_id: str,
    symbol: str,
    order_id: Optional[str] = None,
    position_id: Optional[str] = None,
    **kwargs
) -> None:
    """Log trading event with standard context.
    
    Args:
        event_type: Type of trading event
        message: Event message
        agent_id: Agent ID
        symbol: Trading symbol
        order_id: Order ID if applicable
        position_id: Position ID if applicable
        **kwargs: Additional event data
    """
    log_with_context(
        level="INFO",
        message=f"[TRADE] {event_type}: {message}",
        agent_id=agent_id,
        symbol=symbol,
        order_id=order_id,
        position_id=position_id,
        event_type=event_type,
        **kwargs
    )


def log_risk_event(
    event_type: str,
    message: str,
    severity: str = "medium",
    **kwargs
) -> None:
    """Log risk management event.
    
    Args:
        event_type: Type of risk event
        message: Event message
        severity: Event severity (low, medium, high, critical)
        **kwargs: Additional event data
    """
    level = "WARNING" if severity in ["high", "critical"] else "INFO"
    
    log_with_context(
        level=level,
        message=f"[RISK] {event_type}: {message}",
        event_type=event_type,
        severity=severity,
        **kwargs
    )


def log_broker_event(
    event_type: str,
    message: str,
    broker_id: str,
    severity: str = "medium",
    **kwargs
) -> None:
    """Log broker-related event.
    
    Args:
        event_type: Type of broker event
        message: Event message
        broker_id: Broker ID
        severity: Event severity
        **kwargs: Additional event data
    """
    level = "ERROR" if severity == "critical" else "WARNING" if severity == "high" else "INFO"
    
    log_with_context(
        level=level,
        message=f"[BROKER] {event_type}: {message}",
        broker_id=broker_id,
        event_type=event_type,
        severity=severity,
        **kwargs
    )


def log_performance_metric(
    metric_name: str,
    value: float,
    unit: str = "",
    agent_id: Optional[str] = None,
    symbol: Optional[str] = None,
    **kwargs
) -> None:
    """Log performance metric.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        unit: Unit of measurement
        agent_id: Agent ID if applicable
        symbol: Symbol if applicable
        **kwargs: Additional metric data
    """
    message = f"[PERF] {metric_name}: {value:.4f} {unit}".strip()
    
    log_with_context(
        level="INFO",
        message=message,
        agent_id=agent_id,
        symbol=symbol,
        metric_name=metric_name,
        metric_value=value,
        metric_unit=unit,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Setup logger for testing
    setup_logger(
        level="DEBUG",
        log_file="logs/test.log",
        environment="development"
    )
    
    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test context logging
    log_with_context(
        level="INFO",
        message="Testing context logging",
        correlation_id="test-123",
        agent_id="agent-1",
        symbol="XAUUSD"
    )
    
    # Test trade event logging
    log_trade_event(
        event_type="ORDER_PLACED",
        message="Order placed successfully",
        agent_id="agent-1",
        symbol="XAUUSD",
        order_id="order-123"
    )
    
    # Test risk event logging
    log_risk_event(
        event_type="POSITION_SIZE_LIMIT",
        message="Position size exceeds limit",
        severity="high"
    )
    
    # Test sensitive data masking
    logger.info("Testing password masking: password=secret123")
    logger.info("Testing API key masking: api_key=sk-1234567890")
    
    print("Logger testing completed. Check logs/test.log for file output.")
