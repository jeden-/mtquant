"""
MTQuant Custom Exception Hierarchy

Defines custom exceptions for the MTQuant trading system following
the patterns from .cursorrules for proper error handling and debugging.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class MTQuantError(Exception):
    """Base exception for all MTQuant-related errors.
    
    This is the root exception class that all other MTQuant exceptions
    inherit from. It provides common functionality for error handling
    and debugging across the entire system.
    
    Args:
        message: Human-readable error message
        details: Optional dictionary with additional error details
        timestamp: When the error occurred (defaults to now)
    """
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = timestamp or datetime.utcnow()
    
    def __str__(self) -> str:
        """Return formatted error message with timestamp and details."""
        base_msg = f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {self.__class__.__name__}: {self.message}"
        
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_msg += f" | Details: {details_str}"
        
        return base_msg


# Broker-related exceptions
class BrokerError(MTQuantError):
    """General broker communication error.
    
    Raised when there are issues communicating with brokers
    that don't fit into more specific categories.
    """
    pass


class BrokerConnectionError(BrokerError):
    """Broker connection issues.
    
    Raised when unable to establish or maintain connection
    with broker (network issues, authentication failures, etc.).
    """
    pass


class BrokerAPIError(BrokerError):
    """Broker API response errors.
    
    Raised when broker API returns error responses,
    invalid data, or unexpected response formats.
    """
    pass


class BrokerTimeoutError(BrokerError):
    """Broker operation timeout.
    
    Raised when broker operations exceed configured timeout
    limits (order placement, data fetching, etc.).
    """
    pass


class OrderExecutionError(BrokerError):
    """Order placement failures.
    
    Raised when orders fail to execute due to broker-specific
    issues (rejected orders, insufficient margin, etc.).
    """
    pass


# Risk-related exceptions
class RiskViolationError(MTQuantError):
    """Risk limit breaches.
    
    Raised when trading operations would violate risk management
    rules (position size limits, daily loss limits, etc.).
    """
    pass


class PositionSizeError(RiskViolationError):
    """Invalid position size.
    
    Raised when calculated position size exceeds limits
    or is invalid for the instrument.
    """
    pass


class CircuitBreakerError(RiskViolationError):
    """Circuit breaker triggered.
    
    Raised when circuit breaker levels are activated
    and trading must be halted or restricted.
    """
    pass


# Trading exceptions
class TradingError(MTQuantError):
    """General trading error.
    
    Raised for trading-related issues that don't fit
    into more specific categories.
    """
    pass


class InvalidOrderError(TradingError):
    """Malformed order.
    
    Raised when orders contain invalid data or parameters
    that prevent execution.
    """
    pass


class InsufficientMarginError(TradingError):
    """Not enough margin.
    
    Raised when account doesn't have sufficient margin
    to execute the requested trade.
    """
    pass


# Data exceptions
class DataError(MTQuantError):
    """Data-related issues.
    
    Raised for general data processing, storage,
    or retrieval problems.
    """
    pass


class SymbolNotFoundError(DataError):
    """Symbol mapping failed.
    
    Raised when symbol mapping between standard symbols
    and broker-specific symbols fails.
    """
    pass


class MarketDataError(DataError):
    """Market data fetch failed.
    
    Raised when market data cannot be retrieved
    from data sources or brokers.
    """
    pass


# Agent-related exceptions
class AgentError(MTQuantError):
    """General agent error.
    
    Raised for RL agent-related issues that don't fit
    into more specific categories.
    """
    pass


class AgentTrainingError(AgentError):
    """Agent training failures.
    
    Raised when RL agent training fails due to
    data issues, model problems, or training errors.
    """
    pass


class AgentDeploymentError(AgentError):
    """Agent deployment failures.
    
    Raised when agent deployment fails due to
    configuration issues or system problems.
    """
    pass


# Configuration exceptions
class ConfigurationError(MTQuantError):
    """Configuration-related errors.
    
    Raised when configuration files are invalid,
    missing, or contain incorrect values.
    """
    pass


class ValidationError(MTQuantError):
    """Data validation errors.
    
    Raised when data validation fails for models,
    orders, positions, or other data structures.
    """
    pass


# Database exceptions
class DatabaseError(MTQuantError):
    """Database operation errors.
    
    Raised for database connection, query,
    or transaction failures.
    """
    pass


class ConnectionError(DatabaseError):
    """Database connection errors.
    
    Raised when unable to establish or maintain
    database connections.
    """
    pass


class QueryError(DatabaseError):
    """Database query errors.
    
    Raised when database queries fail due to
    syntax errors, constraint violations, etc.
    """
    pass


# Utility functions for error handling
def create_error_context(
    operation: str,
    symbol: Optional[str] = None,
    agent_id: Optional[str] = None,
    broker_id: Optional[str] = None,
    order_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create standardized error context dictionary.
    
    Args:
        operation: Name of the operation that failed
        symbol: Trading symbol involved
        agent_id: Agent that initiated the operation
        broker_id: Broker used for the operation
        order_id: Order ID if applicable
        **kwargs: Additional context data
        
    Returns:
        Dictionary with standardized error context
    """
    context = {
        "operation": operation,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if symbol:
        context["symbol"] = symbol
    if agent_id:
        context["agent_id"] = agent_id
    if broker_id:
        context["broker_id"] = broker_id
    if order_id:
        context["order_id"] = order_id
    
    context.update(kwargs)
    return context


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable.
    
    Args:
        error: Exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    retryable_errors = (
        BrokerConnectionError,
        BrokerTimeoutError,
        ConnectionError,
        MarketDataError,
    )
    
    return isinstance(error, retryable_errors)


def get_error_severity(error: Exception) -> str:
    """Get error severity level.
    
    Args:
        error: Exception to analyze
        
    Returns:
        Severity level: 'low', 'medium', 'high', 'critical'
    """
    if isinstance(error, CircuitBreakerError):
        return 'critical'
    elif isinstance(error, (RiskViolationError, OrderExecutionError)):
        return 'high'
    elif isinstance(error, (BrokerError, TradingError)):
        return 'medium'
    else:
        return 'low'
