"""
Unit tests for simple components with low coverage.

Focus on utility functions and simple classes that can be easily tested.
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
        """Test add_technical_indicators with basic data."""
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5, 117.5, 118.5, 119.5, 120.5],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
        })
        
        result = add_technical_indicators(data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert 'sma_20' in result.columns
        assert 'bb_upper' in result.columns
        assert len(result) == len(data)
    
    def test_add_technical_indicators_empty(self):
        """Test add_technical_indicators with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = add_technical_indicators(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_add_technical_indicators_single_row(self):
        """Test add_technical_indicators with single row."""
        single_row = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [1000]
        })
        result = add_technical_indicators(single_row)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_calculate_log_returns_basic(self):
        """Test calculate_log_returns with basic data."""
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        
        result = calculate_log_returns(data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'log_returns' in result.columns
        assert len(result) == len(data)
        assert result['log_returns'].iloc[0] == 0.0  # First value should be 0.0
    
    def test_calculate_log_returns_empty(self):
        """Test calculate_log_returns with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = calculate_log_returns(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_calculate_log_returns_single_row(self):
        """Test calculate_log_returns with single row."""
        single_row = pd.DataFrame({
            'close': [100]
        })
        result = calculate_log_returns(single_row)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_normalize_features_basic(self):
        """Test normalize_features with basic data."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'other_col': [100, 200, 300, 400, 500]
        })
        
        result = normalize_features(data, ['feature1', 'feature2'])
        
        assert isinstance(result, pd.DataFrame)
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
        assert 'other_col' in result.columns
        assert len(result) == len(data)
        
        # Check that features are normalized (min-max normalization to 0-1)
        assert result['feature1'].min() >= 0.0
        assert result['feature1'].max() <= 1.0
        assert result['feature2'].min() >= 0.0
        assert result['feature2'].max() <= 1.0
    
    def test_normalize_features_empty(self):
        """Test normalize_features with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = normalize_features(empty_df, [])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_normalize_features_no_features(self):
        """Test normalize_features with no feature columns."""
        data = pd.DataFrame({
            'other_col': [100, 200, 300]
        })
        result = normalize_features(data, ['non_existent'])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
    
    def test_create_sample_data_basic(self):
        """Test create_sample_data with basic parameters."""
        data = create_sample_data("XAUUSD", 100)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
    
    def test_create_sample_data_different_symbols(self):
        """Test create_sample_data with different symbols."""
        symbols = ['XAUUSD', 'EURUSD', 'SPX500']
        
        for symbol in symbols:
            data = create_sample_data(symbol, 50)
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 50
            assert 'close' in data.columns
    
    def test_create_sample_data_with_seed(self):
        """Test create_sample_data with seed for reproducibility."""
        data1 = create_sample_data("XAUUSD", 100, seed=42)
        data2 = create_sample_data("XAUUSD", 100, seed=42)
        
        assert data1.equals(data2)
    
    def test_prepare_training_data_basic(self):
        """Test prepare_training_data with basic data."""
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5, 117.5, 118.5, 119.5, 120.5],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
        })
        
        result = prepare_training_data(data, "XAUUSD")
        
        assert isinstance(result, pd.DataFrame)
        assert 'log_returns' in result.columns
        assert len(result) > 0  # Should have some rows after processing
    
    def test_prepare_training_data_empty(self):
        """Test prepare_training_data with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = prepare_training_data(empty_df, "XAUUSD")
        assert isinstance(result, pd.DataFrame)


class TestUtilityComponents:
    """Tests for utility components."""
    
    def test_mtquant_error_creation(self):
        """Test MTQuantError creation."""
        error = MTQuantError("Test error message")
        assert "Test error message" in str(error)
        assert isinstance(error, Exception)
    
    def test_broker_error_creation(self):
        """Test BrokerError creation."""
        error = BrokerError("Broker connection failed")
        assert "Broker connection failed" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_risk_violation_error_creation(self):
        """Test RiskViolationError creation."""
        error = RiskViolationError("Risk limit exceeded")
        assert "Risk limit exceeded" in str(error)
        assert isinstance(error, MTQuantError)
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_module")
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')


class TestOrderModel:
    """Tests for Order model."""
    
    def test_order_creation(self):
        """Test Order creation."""
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
            signal=0.8,
            created_at=datetime.now(),
            status="pending"
        )
        
        assert order.symbol == "EURUSD"
        assert order.side == "buy"
        assert order.quantity == 0.1
        assert order.signal == 0.8
        assert order.status == "pending"
    
    def test_order_to_dict(self):
        """Test Order to_dict method."""
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
            signal=0.8,
            created_at=datetime.now(),
            status="pending"
        )
        
        order_dict = order.to_dict()
        
        assert isinstance(order_dict, dict)
        assert order_dict['symbol'] == "EURUSD"
        assert order_dict['side'] == "buy"
        assert order_dict['quantity'] == 0.1
        assert order_dict['signal'] == 0.8
    
    def test_order_side_enum(self):
        """Test OrderSide enum."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
    
    def test_order_type_enum(self):
        """Test OrderType enum."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
    
    def test_order_status_enum(self):
        """Test OrderStatus enum."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"


class TestPositionModel:
    """Tests for Position model."""
    
    def test_position_creation(self):
        """Test Position creation."""
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
        assert position.symbol == "EURUSD"
        assert position.side == "long"
        assert position.quantity == 0.1
        assert position.entry_price == 1.2
        assert position.current_price == 1.25
        assert abs(position.unrealized_pnl - 0.005) < 0.0001
    
    def test_position_to_dict(self):
        """Test Position to_dict method."""
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
        assert position_dict['symbol'] == "EURUSD"
        assert position_dict['side'] == "long"
        assert position_dict['quantity'] == 0.1
        assert position_dict['entry_price'] == 1.2
        assert position_dict['current_price'] == 1.25
        assert abs(position_dict['unrealized_pnl'] - 0.005) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
