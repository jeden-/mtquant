"""
Utility Functions and Classes

Shared utilities used throughout the MTQuant system:
- logger.py: Centralized logging configuration
- exceptions.py: Custom exception classes
- helpers: Common utility functions (to be implemented)
- validators: Data validation utilities (to be implemented)
- decorators: Common decorators (timing, retry, etc.) (to be implemented)

All utilities follow production standards with proper error handling,
type hints, and comprehensive documentation.
"""

__version__ = "0.1.0"

from .exceptions import (
    MTQuantError,
    BrokerError,
    BrokerConnectionError,
    BrokerAPIError,
    BrokerTimeoutError,
    OrderExecutionError,
    RiskViolationError,
    PositionSizeError,
    CircuitBreakerError,
    TradingError,
    InvalidOrderError,
    InsufficientMarginError,
    DataError,
    SymbolNotFoundError,
    MarketDataError,
    AgentError,
    AgentTrainingError,
    AgentDeploymentError,
    ConfigurationError,
    ValidationError,
    DatabaseError,
    ConnectionError,
    QueryError,
    create_error_context,
    is_retryable_error,
    get_error_severity,
)

from .logger import (
    setup_logger,
    get_logger,
    log_with_context,
    log_trade_event,
    log_risk_event,
    log_broker_event,
    log_performance_metric,
    mask_sensitive_data,
)

__all__ = [
    # Exceptions
    "MTQuantError",
    "BrokerError",
    "BrokerConnectionError",
    "BrokerAPIError",
    "BrokerTimeoutError",
    "OrderExecutionError",
    "RiskViolationError",
    "PositionSizeError",
    "CircuitBreakerError",
    "TradingError",
    "InvalidOrderError",
    "InsufficientMarginError",
    "DataError",
    "SymbolNotFoundError",
    "MarketDataError",
    "AgentError",
    "AgentTrainingError",
    "AgentDeploymentError",
    "ConfigurationError",
    "ValidationError",
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "create_error_context",
    "is_retryable_error",
    "get_error_severity",
    # Logger
    "setup_logger",
    "get_logger",
    "log_with_context",
    "log_trade_event",
    "log_risk_event",
    "log_broker_event",
    "log_performance_metric",
    "mask_sensitive_data",
]
