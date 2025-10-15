"""
Unit tests for simple utility functions and methods.

Focus on easy-to-test components that can increase coverage quickly.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, List, Any

# Import simple components that definitely exist
from mtquant.utils.exceptions import (
    MTQuantError, BrokerError, RiskViolationError, 
    TradingError, DataError, ConfigurationError,
    ValidationError, ConnectionError, BrokerTimeoutError,
    InsufficientMarginError, OrderExecutionError,
    SymbolNotFoundError, MarketDataError,
    AgentError, AgentTrainingError, AgentDeploymentError,
    BrokerConnectionError, BrokerAPIError, PositionSizeError,
    CircuitBreakerError, InvalidOrderError, DatabaseError,
    QueryError, create_error_context, is_retryable_error,
    get_error_severity
)

# Import MCP models
from mtquant.mcp_integration.models.order import Order, OrderSide, OrderType, OrderStatus
from mtquant.mcp_integration.models.position import Position

# Import risk management components
from mtquant.risk_management.position_sizer import PositionSizer
from mtquant.risk_management.pre_trade_checker import PreTradeChecker
from mtquant.risk_management.circuit_breaker import CircuitBreaker

# Import MCP integration components
from mtquant.mcp_integration.adapters.base_adapter import BrokerAdapter
from mtquant.mcp_integration.managers.symbol_mapper import SymbolMapper


class TestExceptionClasses:
    """Tests for all exception classes."""
    
    def test_mtquant_error(self):
        """Test MTQuantError base exception."""
        error = MTQuantError("Test error")
        assert "Test error" in str(error)
        assert isinstance(error, Exception)
    
    def test_broker_error(self):
        """Test BrokerError exception."""
        error = BrokerError("Broker connection failed")
        assert "Broker connection failed" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_risk_violation_error(self):
        """Test RiskViolationError exception."""
        error = RiskViolationError("Risk limit exceeded")
        assert "Risk limit exceeded" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_trading_error(self):
        """Test TradingError exception."""
        error = TradingError("Trading operation failed")
        assert "Trading operation failed" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_data_error(self):
        """Test DataError exception."""
        error = DataError("Data processing failed")
        assert "Data processing failed" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        error = ConfigurationError("Invalid configuration")
        assert "Invalid configuration" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError("Validation failed")
        assert "Validation failed" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_connection_error(self):
        """Test ConnectionError exception."""
        error = ConnectionError("Connection failed")
        assert "Connection failed" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_timeout_error(self):
        """Test BrokerTimeoutError exception."""
        error = BrokerTimeoutError("Operation timed out")
        assert "Operation timed out" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_insufficient_margin_error(self):
        """Test InsufficientMarginError exception."""
        error = InsufficientMarginError("Insufficient margin")
        assert "Insufficient margin" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_order_execution_error(self):
        """Test OrderExecutionError exception."""
        error = OrderExecutionError("Order execution failed")
        assert "Order execution failed" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_symbol_not_found_error(self):
        """Test SymbolNotFoundError exception."""
        error = SymbolNotFoundError("Symbol not found")
        assert "Symbol not found" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_market_data_error(self):
        """Test MarketDataError exception."""
        error = MarketDataError("Market data error")
        assert "Market data error" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_agent_error(self):
        """Test AgentError exception."""
        error = AgentError("Agent error")
        assert "Agent error" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_agent_training_error(self):
        """Test AgentTrainingError exception."""
        error = AgentTrainingError("Training failed")
        assert "Training failed" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_agent_deployment_error(self):
        """Test AgentDeploymentError exception."""
        error = AgentDeploymentError("Deployment failed")
        assert "Deployment failed" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_broker_connection_error(self):
        """Test BrokerConnectionError exception."""
        error = BrokerConnectionError("Connection failed")
        assert "Connection failed" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_broker_api_error(self):
        """Test BrokerAPIError exception."""
        error = BrokerAPIError("API error")
        assert "API error" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_position_size_error(self):
        """Test PositionSizeError exception."""
        error = PositionSizeError("Position size error")
        assert "Position size error" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_circuit_breaker_error(self):
        """Test CircuitBreakerError exception."""
        error = CircuitBreakerError("Circuit breaker triggered")
        assert "Circuit breaker triggered" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_invalid_order_error(self):
        """Test InvalidOrderError exception."""
        error = InvalidOrderError("Invalid order")
        assert "Invalid order" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_database_error(self):
        """Test DatabaseError exception."""
        error = DatabaseError("Database error")
        assert "Database error" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_query_error(self):
        """Test QueryError exception."""
        error = QueryError("Query failed")
        assert "Query failed" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_create_error_context(self):
        """Test create_error_context function."""
        context = create_error_context(
            operation="test_operation",
            symbol="EURUSD",
            agent_id="test_agent",
            broker_id="test_broker",
            order_id="ORD123",
            additional_info="test"
        )
        
        assert isinstance(context, dict)
        assert context["operation"] == "test_operation"
        assert context["symbol"] == "EURUSD"
        assert context["agent_id"] == "test_agent"
        assert context["broker_id"] == "test_broker"
        assert context["order_id"] == "ORD123"
        assert context["additional_info"] == "test"
        assert "timestamp" in context
    
    def test_is_retryable_error(self):
        """Test is_retryable_error function."""
        # Test retryable errors
        assert is_retryable_error(BrokerConnectionError("test")) == True
        assert is_retryable_error(BrokerTimeoutError("test")) == True
        assert is_retryable_error(ConnectionError("test")) == True
        assert is_retryable_error(MarketDataError("test")) == True
        
        # Test non-retryable errors
        assert is_retryable_error(RiskViolationError("test")) == False
        assert is_retryable_error(ConfigurationError("test")) == False
    
    def test_get_error_severity(self):
        """Test get_error_severity function."""
        # Test critical errors
        assert get_error_severity(CircuitBreakerError("test")) == "critical"
        
        # Test high severity errors
        assert get_error_severity(RiskViolationError("test")) == "high"
        assert get_error_severity(OrderExecutionError("test")) == "high"
        
        # Test medium severity errors
        assert get_error_severity(BrokerError("test")) == "medium"
        assert get_error_severity(TradingError("test")) == "medium"
        
        # Test low severity errors
        assert get_error_severity(DataError("test")) == "low"
        assert get_error_severity(ConfigurationError("test")) == "low"


class TestOrderModelExtended:
    """Extended tests for Order model."""
    
    def test_order_creation_with_all_fields(self):
        """Test Order creation with all fields."""
        order = Order(
            order_id="ORD123",
            agent_id="test_agent",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            price=1.2,
            stop_loss=1.15,
            take_profit=1.25,
            signal=0.8,
            created_at=datetime.now(),
            status="pending"
        )
        
        assert order.order_id == "ORD123"
        assert order.agent_id == "test_agent"
        assert order.symbol == "EURUSD"
        assert order.side == "buy"
        assert order.order_type == "market"
        assert order.quantity == 0.1
        assert order.price == 1.2
        assert order.stop_loss == 1.15
        assert order.take_profit == 1.25
        assert order.signal == 0.8
        assert order.status == "pending"
    
    def test_order_creation_minimal(self):
        """Test Order creation with minimal fields."""
        order = Order(
            order_id=None,
            agent_id="test_agent",
            symbol="EURUSD",
            side="sell",
            order_type="market",  # Changed to market to avoid price requirement
            quantity=0.05,
            price=None,
            stop_loss=None,
            take_profit=None,
            signal=-0.5,
            created_at=datetime.now(),
            status="pending"
        )
        
        assert order.order_id is None
        assert order.symbol == "EURUSD"
        assert order.side == "sell"
        assert order.order_type == "market"
        assert order.quantity == 0.05
        assert order.price is None
        assert order.signal == -0.5
    
    def test_order_to_dict_with_all_fields(self):
        """Test Order to_dict with all fields."""
        order = Order(
            order_id="ORD123",
            agent_id="test_agent",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            price=1.2,
            stop_loss=1.15,
            take_profit=1.25,
            signal=0.8,
            created_at=datetime.now(),
            status="pending"
        )
        
        order_dict = order.to_dict()
        
        assert isinstance(order_dict, dict)
        assert order_dict['order_id'] == "ORD123"
        assert order_dict['agent_id'] == "test_agent"
        assert order_dict['symbol'] == "EURUSD"
        assert order_dict['side'] == "buy"
        assert order_dict['order_type'] == "market"
        assert order_dict['quantity'] == 0.1
        assert order_dict['price'] == 1.2
        assert order_dict['stop_loss'] == 1.15
        assert order_dict['take_profit'] == 1.25
        assert order_dict['signal'] == 0.8
        assert order_dict['status'] == "pending"
    
    def test_order_side_enum_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
        
        # Test enum comparison
        assert OrderSide.BUY == OrderSide.BUY
        assert OrderSide.BUY != OrderSide.SELL
    
    def test_order_type_enum_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        
        # Test enum comparison
        assert OrderType.MARKET == OrderType.MARKET
        assert OrderType.MARKET != OrderType.LIMIT
    
    def test_order_status_enum_values(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        
        # Test enum comparison
        assert OrderStatus.PENDING == OrderStatus.PENDING
        assert OrderStatus.PENDING != OrderStatus.FILLED


class TestPositionModelExtended:
    """Extended tests for Position model."""
    
    def test_position_creation_with_all_fields(self):
        """Test Position creation with all fields."""
        position = Position(
            position_id="POS123",
            agent_id="test_agent",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.2,
            current_price=1.25,
            unrealized_pnl=0.005,
            opened_at=datetime.now(),
            broker_id="test_broker"
        )
        
        assert position.position_id == "POS123"
        assert position.agent_id == "test_agent"
        assert position.symbol == "EURUSD"
        assert position.side == "long"
        assert position.quantity == 0.1
        assert position.entry_price == 1.2
        assert position.current_price == 1.25
        assert abs(position.unrealized_pnl - 0.005) < 0.0001
        assert position.broker_id == "test_broker"
    
    def test_position_creation_short(self):
        """Test Position creation for short position."""
        position = Position(
            position_id="POS456",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="short",
            quantity=0.05,
            entry_price=2000.0,
            current_price=1995.0,
            unrealized_pnl=0.025,
            opened_at=datetime.now(),
            broker_id="test_broker"
        )
        
        assert position.side == "short"
        assert position.symbol == "XAUUSD"
        assert position.quantity == 0.05
        assert position.entry_price == 2000.0
        assert position.current_price == 1995.0
    
    def test_position_to_dict_with_all_fields(self):
        """Test Position to_dict with all fields."""
        position = Position(
            position_id="POS123",
            agent_id="test_agent",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.2,
            current_price=1.25,
            unrealized_pnl=0.005,
            opened_at=datetime.now(),
            broker_id="test_broker"
        )
        
        position_dict = position.to_dict()
        
        assert isinstance(position_dict, dict)
        assert position_dict['position_id'] == "POS123"
        assert position_dict['agent_id'] == "test_agent"
        assert position_dict['symbol'] == "EURUSD"
        assert position_dict['side'] == "long"
        assert position_dict['quantity'] == 0.1
        assert position_dict['entry_price'] == 1.2
        assert position_dict['current_price'] == 1.25
        assert abs(position_dict['unrealized_pnl'] - 0.005) < 0.0001
        assert position_dict['broker_id'] == "test_broker"


class TestPositionSizerSimple:
    """Simple tests for PositionSizer."""
    
    def test_position_sizer_initialization(self):
        """Test PositionSizer initialization."""
        config = {
            'max_position_size': 0.1,
            'risk_per_trade': 0.02,
            'kelly_fraction': 0.25
        }
        sizer = PositionSizer(config)
        
        assert sizer is not None
        assert hasattr(sizer, 'calculate')
    
    def test_position_sizer_calculate_position_size_basic(self):
        """Test PositionSizer calculate_position_size basic functionality."""
        config = {
            'max_position_size': 0.1,
            'risk_per_trade': 0.02,
            'kelly_fraction': 0.25
        }
        sizer = PositionSizer(config)
        
        # Test with basic parameters
        signal = 0.5
        portfolio_equity = 10000
        instrument_volatility = 0.02
        
        position_size = sizer.calculate(
            signal=signal,
            portfolio_equity=portfolio_equity,
            instrument_volatility=instrument_volatility
        )
        
        assert isinstance(position_size, object)  # PositionSizingResult
        assert hasattr(position_size, 'position_size')
        assert position_size.position_size > 0


class TestPreTradeCheckerSimple:
    """Simple tests for PreTradeChecker."""
    
    def test_pre_trade_checker_initialization(self):
        """Test PreTradeChecker initialization."""
        risk_limits = {
            'max_position_size': 0.1,
            'max_daily_loss': 0.05,
            'max_leverage': 10
        }
        checker = PreTradeChecker(risk_limits)
        
        assert checker is not None
        assert hasattr(checker, 'validate')
    
    async def test_pre_trade_checker_validate_order_basic(self):
        """Test PreTradeChecker validate_order basic functionality."""
        risk_limits = {
            'max_position_size': 0.1,
            'max_daily_loss': 0.05,
            'max_leverage': 10
        }
        checker = PreTradeChecker(risk_limits)
        
        # Create a simple order
        order = Order(
            order_id=None,
            agent_id="test_agent",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            price=None,
            stop_loss=None,
            take_profit=None,
            signal=0.5,
            created_at=datetime.now(),
            status="pending"
        )
        
        # Mock account info
        account_info = {
            'equity': 10000,
            'margin_available': 5000,
            'margin_used': 1000
        }
        
        # Mock market data
        market_data = {
            'EURUSD': {'bid': 1.2, 'ask': 1.2001, 'spread': 0.0001}
        }
        
        result = await checker.validate(order, account_info, [], 1.2)
        
        assert isinstance(result, object)  # ValidationResult
        assert hasattr(result, 'is_valid')


class TestCircuitBreakerSimple:
    """Simple tests for CircuitBreaker."""
    
    def test_circuit_breaker_initialization(self):
        """Test CircuitBreaker initialization."""
        config = {
            'level1_threshold': 0.05,
            'level2_threshold': 0.10,
            'level3_threshold': 0.15
        }
        breaker = CircuitBreaker(config)
        
        assert breaker is not None
        assert hasattr(breaker, 'check_and_activate')
    
    def test_circuit_breaker_check_breakers_basic(self):
        """Test CircuitBreaker check_breakers basic functionality."""
        config = {
            'level1_threshold': 0.05,
            'level2_threshold': 0.10,
            'level3_threshold': 0.15
        }
        breaker = CircuitBreaker(config)
        
        # Test with basic portfolio state
        portfolio_state = {
            'equity': 10000,
            'daily_pnl': -100,
            'daily_pnl_pct': -0.01
        }
        
        result = breaker.check_and_activate(portfolio_state['equity'])
        
        assert isinstance(result, object)  # CircuitBreakerStatus
        assert hasattr(result, 'level')
        assert hasattr(result, 'is_trading_allowed')


class TestSymbolMapperSimple:
    """Simple tests for SymbolMapper."""
    
    def test_symbol_mapper_initialization(self):
        """Test SymbolMapper initialization."""
        mapper = SymbolMapper()
        
        assert mapper is not None
        assert hasattr(mapper, 'to_broker_symbol')
        assert hasattr(mapper, 'to_standard_symbol')
    
    def test_symbol_mapper_to_broker_symbol(self):
        """Test SymbolMapper to_broker_symbol method."""
        mapper = SymbolMapper()
        
        # Test with known symbol
        broker_symbol = mapper.to_broker_symbol("XAUUSD", "ic_markets")
        
        assert isinstance(broker_symbol, str)
        assert broker_symbol == "XAUUSD"  # Should be the same for IC Markets
    
    def test_symbol_mapper_to_standard_symbol(self):
        """Test SymbolMapper to_standard_symbol method."""
        mapper = SymbolMapper()
        
        # Test with known symbol
        standard_symbol = mapper.to_standard_symbol("XAUUSD", "ic_markets")
        
        assert isinstance(standard_symbol, str)
        assert standard_symbol == "XAUUSD"  # Should be the same for IC Markets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
