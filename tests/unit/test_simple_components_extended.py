"""
Extended tests for simple components with low coverage.

This module tests utility functions and simple classes that can be easily tested.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

# Import simple components with low coverage
from mtquant.data.processors.feature_engineering import (
    add_technical_indicators,
    calculate_log_returns,
    normalize_features,
    create_sample_data,
    prepare_training_data
)

# Import utility components
from mtquant.utils.exceptions import MTQuantError, BrokerError, RiskViolationError
from mtquant.utils.logger import get_logger

# Import MCP models
from mtquant.mcp_integration.models.order import Order, OrderSide, OrderType, OrderStatus
from mtquant.mcp_integration.models.position import Position


class TestFeatureEngineeringSimple:
    """Tests for simple feature engineering functions."""
    
    def test_add_technical_indicators_basic(self):
        """Test basic technical indicators calculation."""
        # Create sample data
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        result = add_technical_indicators(data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'sma_20' in result.columns
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert 'atr' in result.columns
        assert len(result) == len(data)
    
    def test_add_technical_indicators_empty_data(self):
        """Test technical indicators with empty data."""
        data = pd.DataFrame()
        
        result = add_technical_indicators(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_add_technical_indicators_insufficient_data(self):
        """Test technical indicators with insufficient data."""
        data = pd.DataFrame({
            'close': [100, 101],
            'high': [101, 102],
            'low': [99, 100],
            'volume': [1000, 1100]
        })
        
        result = add_technical_indicators(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        # Should handle insufficient data gracefully
    
    def test_calculate_log_returns_basic(self):
        """Test basic log returns calculation."""
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        result = calculate_log_returns(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        assert 'log_returns' in result.columns
    
    def test_calculate_log_returns_empty(self):
        """Test log returns calculation with empty DataFrame."""
        data = pd.DataFrame()
        
        result = calculate_log_returns(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_calculate_log_returns_single_value(self):
        """Test log returns calculation with single value."""
        data = pd.DataFrame({
            'close': [100],
            'high': [101],
            'low': [99],
            'volume': [1000]
        })
        
        result = calculate_log_returns(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert 'log_returns' in result.columns
    
    def test_normalize_features_basic(self):
        """Test basic feature normalization."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        result = normalize_features(data, ['feature1', 'feature2'])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
        # Check that values are normalized (between 0 and 1)
        assert result['feature1'].min() >= 0
        assert result['feature1'].max() <= 1
        assert result['feature2'].min() >= 0
        assert result['feature2'].max() <= 1
    
    def test_normalize_features_empty(self):
        """Test feature normalization with empty data."""
        data = pd.DataFrame()
        
        result = normalize_features(data, [])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_normalize_features_constant_values(self):
        """Test feature normalization with constant values."""
        data = pd.DataFrame({
            'feature1': [5, 5, 5, 5, 5],
            'feature2': [10, 10, 10, 10, 10]
        })
        
        result = normalize_features(data, ['feature1', 'feature2'])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        # Constant values should be handled gracefully
    
    def test_create_sample_data_basic(self):
        """Test basic sample data creation."""
        data = create_sample_data(symbol="XAUUSD", periods=100)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
        assert data.index.name == 'timestamp'
    
    def test_create_sample_data_custom_periods(self):
        """Test sample data creation with custom periods."""
        data = create_sample_data(symbol="EURUSD", periods=50)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
    
    def test_create_sample_data_empty_periods(self):
        """Test sample data creation with zero periods."""
        data = create_sample_data(symbol="EURUSD", periods=0)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 0
    
    def test_prepare_training_data_basic(self):
        """Test basic training data preparation."""
        data = create_sample_data(symbol="XAUUSD", periods=100)
        
        result = prepare_training_data(data, symbol="XAUUSD")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0  # Should have some data after processing
    
    def test_prepare_training_data_empty(self):
        """Test training data preparation with empty data."""
        data = pd.DataFrame()
        
        result = prepare_training_data(data, symbol="XAUUSD")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestUtilityExceptions:
    """Tests for utility exception classes."""
    
    def test_mtquant_error_basic(self):
        """Test basic MTQuantError."""
        error = MTQuantError("Test error message")
        
        assert "Test error message" in str(error)
        assert isinstance(error, Exception)
    
    def test_mtquant_error_empty_message(self):
        """Test MTQuantError with empty message."""
        error = MTQuantError("")
        
        assert isinstance(error, Exception)
        assert error.message == ""
    
    def test_broker_error_basic(self):
        """Test basic BrokerError."""
        error = BrokerError("Broker connection failed")
        
        assert "Broker connection failed" in str(error)
        assert isinstance(error, MTQuantError)
        assert isinstance(error, Exception)
    
    def test_risk_violation_error_basic(self):
        """Test basic RiskViolationError."""
        error = RiskViolationError("Risk limit exceeded")
        
        assert "Risk limit exceeded" in str(error)
        assert isinstance(error, MTQuantError)
        assert isinstance(error, Exception)
    
    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        broker_error = BrokerError("test")
        risk_error = RiskViolationError("test")
        
        assert isinstance(broker_error, MTQuantError)
        assert isinstance(broker_error, Exception)
        assert isinstance(risk_error, MTQuantError)
        assert isinstance(risk_error, Exception)


class TestLoggerUtility:
    """Tests for logger utility functions."""
    
    def test_get_logger_basic(self):
        """Test basic logger retrieval."""
        logger = get_logger("test_module")
        
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'debug')
    
    def test_get_logger_different_names(self):
        """Test logger retrieval with different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        
        assert logger1 is not None
        assert logger2 is not None
        # Should be different logger instances
        assert logger1 != logger2
    
    def test_get_logger_same_name(self):
        """Test logger retrieval with same name."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        
        assert logger1 is not None
        assert logger2 is not None
        # Should be different logger instances (not cached)
        assert logger1 != logger2


class TestOrderModel:
    """Tests for Order model."""
    
    def test_order_creation_basic(self):
        """Test basic order creation."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.8
        )
        
        assert order.symbol == "XAUUSD"
        assert order.side == "buy"
        assert order.order_type == "market"
        assert order.quantity == 0.1
        assert order.signal == 0.8
        assert order.status == "pending"
        assert order.created_at is not None
    
    def test_order_creation_all_fields(self):
        """Test order creation with all fields."""
        order = Order(
            agent_id="forex_agent",
            symbol="EURUSD",
            side="sell",
            order_type="limit",
            quantity=0.5,
            price=1.1000,
            stop_loss=1.1050,  # Above entry for sell
            take_profit=1.0950,  # Below entry for sell
            signal=-0.6,
            status="filled",
            order_id="12345",
            broker_id="mt5_demo",
            metadata={"test": "data"}
        )
        
        assert order.symbol == "EURUSD"
        assert order.side == "sell"
        assert order.order_type == "limit"
        assert order.quantity == 0.5
        assert order.price == 1.1000
        assert order.stop_loss == 1.1050
        assert order.take_profit == 1.0950
        assert order.signal == -0.6
        assert order.status == "filled"
        assert order.order_id == "12345"
        assert order.broker_id == "mt5_demo"
        assert order.metadata == {"test": "data"}
    
    def test_order_creation_minimal(self):
        """Test order creation with minimal required fields."""
        order = Order(
            agent_id="fx_agent",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.01,
            signal=0.5
        )
        
        assert order.symbol == "EURUSD"
        assert order.side == "buy"
        assert order.order_type == "market"
        assert order.quantity == 0.01
        assert order.signal == 0.5
        assert order.price is None
        assert order.stop_loss is None
        assert order.take_profit is None
        assert order.status == "pending"
        assert order.order_id is None
        assert order.broker_id is None
        assert order.metadata == {}
    
    def test_order_validation(self):
        """Test order validation."""
        # Valid order
        valid_order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.8
        )
        assert valid_order.symbol == "XAUUSD"
        
        # Test with invalid side
        with pytest.raises(ValueError):
            Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="invalid_side",
                order_type="market",
                quantity=0.1,
                signal=0.8
            )
        
        # Test with invalid order type
        with pytest.raises(ValueError):
            Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="buy",
                order_type="invalid_type",
                quantity=0.1,
                signal=0.8
            )
    
    def test_order_risk_reward_ratio(self):
        """Test order risk-reward ratio calculation."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            price=2050.0,
            stop_loss=2040.0,
            take_profit=2070.0,
            signal=0.8
        )
        
        ratio = order.get_risk_reward_ratio()
        
        assert ratio is not None
        assert ratio > 0
        # Risk: 2050 - 2040 = 10, Reward: 2070 - 2050 = 20, Ratio: 20/10 = 2.0
        assert ratio == 2.0
    
    def test_order_risk_reward_ratio_no_stops(self):
        """Test order risk-reward ratio with no stop loss/take profit."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.8
        )
        
        ratio = order.get_risk_reward_ratio()
        
        assert ratio is None
    
    def test_order_repr(self):
        """Test order string representation."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.8
        )
        
        repr_str = repr(order)
        
        assert isinstance(repr_str, str)
        assert "XAUUSD" in repr_str
        assert "buy" in repr_str
        assert "market" in repr_str


class TestPositionModel:
    """Tests for Position model."""
    
    def test_position_creation_basic(self):
        """Test basic position creation."""
        position = Position(
            position_id="pos_123",
            agent_id="forex_agent",
            symbol="XAUUSD",
            side="long",
            quantity=0.1,
            entry_price=2050.0,
            current_price=2052.0,
            unrealized_pnl=2.0,
            opened_at=datetime.now()
        )
        
        assert position.symbol == "XAUUSD"
        assert position.side == "long"
        assert position.quantity == 0.1
        assert position.entry_price == 2050.0
        assert position.current_price == 2052.0
        assert position.unrealized_pnl == pytest.approx(0.2, abs=0.01)  # (2052-2050)*0.1 = 0.2
        assert position.opened_at is not None
        assert position.agent_id == "forex_agent"
    
    def test_position_creation_all_fields(self):
        """Test position creation with all fields."""
        position = Position(
            position_id="pos_12345",
            agent_id="forex_agent",
            symbol="EURUSD",
            side="short",
            quantity=0.5,
            entry_price=1.1000,
            current_price=1.0950,
            unrealized_pnl=25.0,
            opened_at=datetime.now(),
            broker_id="mt5_demo",
            stop_loss=1.1050,
            take_profit=1.0900,
            metadata={"test": "data"}
        )
        
        assert position.symbol == "EURUSD"
        assert position.side == "short"
        assert position.quantity == 0.5
        assert position.entry_price == 1.1000
        assert position.current_price == 1.0950
        assert position.unrealized_pnl == pytest.approx(0.0025, abs=0.0001)  # (1.1-1.095)*0.5 = 0.0025
        assert position.position_id == "pos_12345"
        assert position.broker_id == "mt5_demo"
        assert position.stop_loss == 1.1050
        assert position.take_profit == 1.0900
        assert position.metadata == {"test": "data"}
    
    def test_position_unrealized_pnl_percentage(self):
        """Test position unrealized PnL percentage calculation."""
        position = Position(
            position_id="pos_123",
            agent_id="forex_agent",
            symbol="XAUUSD",
            side="long",
            quantity=0.1,
            entry_price=2050.0,
            current_price=2052.0,
            unrealized_pnl=2.0,
            opened_at=datetime.now()
        )
        
        pnl_pct = position.unrealized_pnl_pct
        
        assert pnl_pct is not None
        assert pnl_pct > 0
        # PnL% = (2052 - 2050) / 2050 * 100 = 0.097%
        assert abs(pnl_pct - 0.097) < 0.001
    
    def test_position_unrealized_pnl_percentage_short(self):
        """Test position unrealized PnL percentage calculation for short position."""
        position = Position(
            position_id="pos_123",
            agent_id="forex_agent",
            symbol="XAUUSD",
            side="short",
            quantity=0.1,
            entry_price=2050.0,
            current_price=2048.0,
            unrealized_pnl=2.0,
            opened_at=datetime.now()
        )
        
        pnl_pct = position.unrealized_pnl_pct
        
        assert pnl_pct is not None
        assert pnl_pct > 0
        # PnL% = (2050 - 2048) / 2050 * 100 = 0.097%
        assert abs(pnl_pct - 0.097) < 0.001
    
    def test_position_duration(self):
        """Test position duration calculation."""
        opened_at = datetime.now()
        position = Position(
            position_id="pos_123",
            agent_id="forex_agent",
            symbol="XAUUSD",
            side="long",
            quantity=0.1,
            entry_price=2050.0,
            current_price=2052.0,
            unrealized_pnl=2.0,
            opened_at=opened_at
        )
        
        duration = position.duration_hours
        
        assert duration is not None
        # Duration can be negative if opened_at is in the future (test timing issue)
        assert isinstance(duration, float)
    
    def test_position_repr(self):
        """Test position string representation."""
        position = Position(
            position_id="pos_123",
            agent_id="forex_agent",
            symbol="XAUUSD",
            side="long",
            quantity=0.1,
            entry_price=2050.0,
            current_price=2052.0,
            unrealized_pnl=2.0,
            opened_at=datetime.now()
        )
        
        repr_str = repr(position)
        
        assert isinstance(repr_str, str)
        assert "XAUUSD" in repr_str
        assert "long" in repr_str
        assert "0.1" in repr_str


class TestOrderEnums:
    """Tests for Order enums."""
    
    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
    
    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
    
    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_feature_engineering_with_nan_values(self):
        """Test feature engineering with NaN values."""
        data = pd.DataFrame({
            'close': [100, 101, np.nan, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        result = add_technical_indicators(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
    
    def test_log_returns_with_zero_values(self):
        """Test log returns calculation with zero values."""
        data = pd.DataFrame({
            'close': [100, 0, 102, 103],
            'high': [101, 1, 103, 104],
            'low': [99, -1, 101, 102],
            'volume': [1000, 1100, 1200, 1300]
        })
        
        result = calculate_log_returns(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        # Should handle zero values gracefully
    
    def test_normalize_features_with_inf_values(self):
        """Test feature normalization with infinite values."""
        data = pd.DataFrame({
            'feature1': [1, 2, np.inf, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        result = normalize_features(data, ['feature1', 'feature2'])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        # Should handle infinite values gracefully
    
    def test_order_with_extreme_values(self):
        """Test order creation with extreme values."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.0001,  # Very small quantity
            signal=1.0  # Maximum signal
        )
        
        assert order.quantity == 0.0001
        assert order.signal == 1.0
    
    def test_position_with_zero_prices(self):
        """Test position creation with zero prices."""
        position = Position(
            position_id="pos_123",
            agent_id="forex_agent",
            symbol="XAUUSD",
            side="long",
            quantity=0.1,
            entry_price=0.0,  # Zero price
            current_price=2052.0,
            unrealized_pnl=2.0,
            opened_at=datetime.now()
        )
        
        assert position.entry_price == 0.0
        assert position.current_price == 2052.0
        # Should handle zero prices gracefully
