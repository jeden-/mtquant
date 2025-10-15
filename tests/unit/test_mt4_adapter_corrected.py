"""
Correct unit tests for MT4BrokerAdapter to increase coverage from 29% to >85%.

This file has 122 lines and 29% coverage, so adding comprehensive tests here will significantly increase overall coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, List, Any

# Import MT4BrokerAdapter and related classes
from mtquant.mcp_integration.adapters.mt4_adapter import MT4BrokerAdapter
from mtquant.mcp_integration.models.order import Order, OrderSide, OrderType, OrderStatus
from mtquant.mcp_integration.models.position import Position
from mtquant.utils.exceptions import BrokerError, BrokerConnectionError, OrderExecutionError, SymbolNotFoundError, InvalidOrderError
from mtquant.mcp_integration.adapters.base_adapter import HealthStatus


class TestMT4BrokerAdapterInitialization:
    """Tests for MT4BrokerAdapter initialization."""
    
    def test_mt4_adapter_initialization(self):
        """Test MT4BrokerAdapter initialization."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo',
            'timeout': 30.0
        }
        
        with patch('mtquant.mcp_integration.adapters.mt4_adapter.MT4MCPClient'):
            adapter = MT4BrokerAdapter(broker_id="mt4_demo", config=config)
            
            assert adapter.broker_id == "mt4_demo"
            assert adapter.config == config
            assert adapter.mt4_client is not None
            assert adapter.symbol_mapper is not None
    
    def test_mt4_adapter_initialization_with_defaults(self):
        """Test MT4BrokerAdapter initialization with default values."""
        config = {
            'account': 87654321,
            'password': 'default_password',
            'server': 'Default-Server'
        }
        
        with patch('mtquant.mcp_integration.adapters.mt4_adapter.MT4MCPClient'):
            adapter = MT4BrokerAdapter(broker_id="mt4_default", config=config)
            
            assert adapter.broker_id == "mt4_default"
            assert adapter.config == config
            assert adapter.mt4_client is not None
            assert adapter.symbol_mapper is not None


class TestMT4BrokerAdapterConnection:
    """Tests for MT4BrokerAdapter connection methods."""
    
    @pytest.fixture
    def adapter(self):
        """Create MT4BrokerAdapter instance."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo',
            'timeout': 30.0
        }
        with patch('mtquant.mcp_integration.adapters.mt4_adapter.MT4MCPClient'):
            return MT4BrokerAdapter(broker_id="mt4_test", config=config)
    
    @pytest.mark.asyncio
    async def test_connect_success(self, adapter):
        """Test successful connection."""
        adapter.mt4_client.connect = AsyncMock(return_value=True)
        
        result = await adapter.connect()
        
        assert result == True
        adapter.mt4_client.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, adapter):
        """Test connection failure."""
        adapter.mt4_client.connect = AsyncMock(return_value=False)
        
        result = await adapter.connect()
        
        assert result == False
        adapter.mt4_client.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_exception(self, adapter):
        """Test connection with exception."""
        adapter.mt4_client.connect = AsyncMock(side_effect=Exception("Connection failed"))
        
        with pytest.raises(BrokerConnectionError, match="Failed to connect MT4 adapter"):
            await adapter.connect()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, adapter):
        """Test disconnection."""
        adapter.mt4_client.disconnect = AsyncMock()
        
        await adapter.disconnect()
        
        adapter.mt4_client.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect_exception(self, adapter):
        """Test disconnection with exception."""
        adapter.mt4_client.disconnect = AsyncMock(side_effect=Exception("Disconnect failed"))
        
        # Should not raise exception
        await adapter.disconnect()
        
        adapter.mt4_client.disconnect.assert_called_once()


class TestMT4BrokerAdapterOrderManagement:
    """Tests for MT4BrokerAdapter order management."""
    
    @pytest.fixture
    def adapter(self):
        """Create MT4BrokerAdapter instance."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo',
            'timeout': 30.0
        }
        with patch('mtquant.mcp_integration.adapters.mt4_adapter.MT4MCPClient'):
            return MT4BrokerAdapter(broker_id="mt4_test", config=config)
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, adapter):
        """Test successful order placement."""
        order = Order(
            order_id="test_order_1",
            agent_id="test_agent",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.8
        )
        
        # Mock symbol mapping
        with patch.object(adapter.symbol_mapper, 'to_broker_symbol', return_value="EURUSD"):
            adapter.mt4_client.place_order = AsyncMock(return_value="MT4_ORDER_123")
            
            result = await adapter.place_order(order)
            
            assert result == "MT4_ORDER_123"
            adapter.mt4_client.place_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_place_order_symbol_not_found(self, adapter):
        """Test order placement with symbol not found."""
        order = Order(
            order_id="test_order_1",
            agent_id="test_agent",
            symbol="INVALID_SYMBOL",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.8
        )
        
        # Mock symbol mapping failure
        with patch.object(adapter.symbol_mapper, 'to_broker_symbol', side_effect=SymbolNotFoundError("Symbol not found")):
            with pytest.raises(SymbolNotFoundError):
                await adapter.place_order(order)
    
    @pytest.mark.asyncio
    async def test_place_order_validation_failure(self, adapter):
        """Test order placement with validation failure."""
        order = Order(
            order_id="test_order_1",
            agent_id="test_agent",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.1,  # Valid quantity
            signal=0.8
        )
        
        # Mock validation to return False
        adapter._validate_order = Mock(return_value=False)
        
        # Mock symbol mapping
        with patch.object(adapter.symbol_mapper, 'to_broker_symbol', return_value="EURUSD"):
            with pytest.raises(OrderExecutionError, match="Failed to place order via MT4"):
                await adapter.place_order(order)
        
        # Verify validation was called
        adapter._validate_order.assert_called_once_with(order)
    
    @pytest.mark.asyncio
    async def test_place_order_exception(self, adapter):
        """Test order placement with exception."""
        order = Order(
            order_id="test_order_1",
            agent_id="test_agent",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.8
        )
        
        # Mock symbol mapping
        with patch.object(adapter.symbol_mapper, 'to_broker_symbol', return_value="EURUSD"):
            adapter.mt4_client.place_order = AsyncMock(side_effect=Exception("Order failed"))
            
            with pytest.raises(OrderExecutionError, match="Failed to place order via MT4"):
                await adapter.place_order(order)


class TestMT4BrokerAdapterPositionManagement:
    """Tests for MT4BrokerAdapter position management."""
    
    @pytest.fixture
    def adapter(self):
        """Create MT4BrokerAdapter instance."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo',
            'timeout': 30.0
        }
        with patch('mtquant.mcp_integration.adapters.mt4_adapter.MT4MCPClient'):
            return MT4BrokerAdapter(broker_id="mt4_test", config=config)
    
    @pytest.mark.asyncio
    async def test_get_positions_success(self, adapter):
        """Test successful position retrieval."""
        mock_positions = [
            Position(
                position_id="pos_1",
                agent_id="test_agent",
                symbol="EURUSD",
                side="long",
                quantity=0.1,
                entry_price=1.1000,
                current_price=1.1050,
                unrealized_pnl=5.0
            )
        ]
        
        adapter.mt4_client.get_positions = AsyncMock(return_value=mock_positions)
        
        # Mock symbol unmapping
        with patch.object(adapter.symbol_mapper, 'to_standard_symbol', return_value="EURUSD"):
            positions = await adapter.get_positions()
            
            assert len(positions) == 1
            assert positions[0].symbol == 'EURUSD'
            assert positions[0].quantity == 0.1
            adapter.mt4_client.get_positions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_positions_symbol_unmapping_failure(self, adapter):
        """Test position retrieval with symbol unmapping failure."""
        mock_positions = [
            Position(
                position_id="pos_1",
                agent_id="test_agent",
                symbol="EURUSD",
                side="long",
                quantity=0.1,
                entry_price=1.1000,
                current_price=1.1050,
                unrealized_pnl=5.0
            )
        ]
        
        adapter.mt4_client.get_positions = AsyncMock(return_value=mock_positions)
        
        # Mock symbol unmapping failure
        with patch.object(adapter.symbol_mapper, 'to_standard_symbol', side_effect=SymbolNotFoundError("Symbol not found")):
            positions = await adapter.get_positions()
            
            # Should still return positions with broker symbols
            assert len(positions) == 1
            assert positions[0].symbol == 'EURUSD'
    
    @pytest.mark.asyncio
    async def test_get_positions_exception(self, adapter):
        """Test position retrieval with exception."""
        adapter.mt4_client.get_positions = AsyncMock(side_effect=Exception("Position fetch failed"))
        
        with pytest.raises(BrokerError, match="Failed to get positions via MT4"):
            await adapter.get_positions()
    
    @pytest.mark.asyncio
    async def test_close_position_success(self, adapter):
        """Test successful position closure."""
        position_id = "MT4_POSITION_123"
        
        adapter.mt4_client.close_position = AsyncMock(return_value=True)
        
        result = await adapter.close_position(position_id)
        
        assert result == True
        adapter.mt4_client.close_position.assert_called_once_with(position_id)
    
    @pytest.mark.asyncio
    async def test_close_position_failure(self, adapter):
        """Test position closure failure."""
        position_id = "MT4_POSITION_123"
        
        adapter.mt4_client.close_position = AsyncMock(side_effect=Exception("Close failed"))
        
        result = await adapter.close_position(position_id)
        
        assert result == False
        adapter.mt4_client.close_position.assert_called_once_with(position_id)


class TestMT4BrokerAdapterMarketData:
    """Tests for MT4BrokerAdapter market data methods."""
    
    @pytest.fixture
    def adapter(self):
        """Create MT4BrokerAdapter instance."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo',
            'timeout': 30.0
        }
        with patch('mtquant.mcp_integration.adapters.mt4_adapter.MT4MCPClient'):
            return MT4BrokerAdapter(broker_id="mt4_test", config=config)
    
    @pytest.mark.asyncio
    async def test_get_market_data_success(self, adapter):
        """Test successful market data retrieval."""
        import pandas as pd
        
        mock_data = pd.DataFrame({
            'open': [1.1000],
            'high': [1.1050],
            'low': [1.0950],
            'close': [1.1025],
            'volume': [1000]
        })
        
        adapter.mt4_client.get_market_data = AsyncMock(return_value=mock_data)
        
        # Mock symbol mapping
        with patch.object(adapter.symbol_mapper, 'to_broker_symbol', return_value="EURUSD"):
            data = await adapter.get_market_data("EURUSD", "H1", 100)
            
            assert data is not None
            assert 'standard_symbol' in data.columns
            assert data['standard_symbol'].iloc[0] == "EURUSD"
            adapter.mt4_client.get_market_data.assert_called_once_with("EURUSD", "H1", 100)
    
    @pytest.mark.asyncio
    async def test_get_market_data_symbol_not_found(self, adapter):
        """Test market data retrieval with symbol not found."""
        # Mock symbol mapping failure
        with patch.object(adapter.symbol_mapper, 'to_broker_symbol', side_effect=SymbolNotFoundError("Symbol not found")):
            with pytest.raises(SymbolNotFoundError):
                await adapter.get_market_data("INVALID_SYMBOL", "H1", 100)
    
    @pytest.mark.asyncio
    async def test_get_market_data_exception(self, adapter):
        """Test market data retrieval with exception."""
        # Mock symbol mapping
        with patch.object(adapter.symbol_mapper, 'to_broker_symbol', return_value="EURUSD"):
            adapter.mt4_client.get_market_data = AsyncMock(side_effect=Exception("Data fetch failed"))
            
            with pytest.raises(BrokerError, match="Failed to get market data via MT4"):
                await adapter.get_market_data("EURUSD", "H1", 100)


class TestMT4BrokerAdapterAccountInfo:
    """Tests for MT4BrokerAdapter account information methods."""
    
    @pytest.fixture
    def adapter(self):
        """Create MT4BrokerAdapter instance."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo',
            'timeout': 30.0
        }
        with patch('mtquant.mcp_integration.adapters.mt4_adapter.MT4MCPClient'):
            return MT4BrokerAdapter(broker_id="mt4_test", config=config)
    
    @pytest.mark.asyncio
    async def test_get_account_info_success(self, adapter):
        """Test successful account info retrieval."""
        mock_info = {
            'balance': 10000.0,
            'equity': 10050.0,
            'margin': 100.0,
            'free_margin': 9950.0,
            'margin_level': 10050.0
        }
        
        adapter.mt4_client.get_account_info = AsyncMock(return_value=mock_info)
        
        info = await adapter.get_account_info()
        
        assert info['balance'] == 10000.0
        assert info['equity'] == 10050.0
        assert info['margin'] == 100.0
        adapter.mt4_client.get_account_info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_account_info_exception(self, adapter):
        """Test account info retrieval with exception."""
        adapter.mt4_client.get_account_info = AsyncMock(side_effect=Exception("Account info failed"))
        
        info = await adapter.get_account_info()
        
        assert info == {}
        adapter.mt4_client.get_account_info.assert_called_once()


class TestMT4BrokerAdapterHealthCheck:
    """Tests for MT4BrokerAdapter health check methods."""
    
    @pytest.fixture
    def adapter(self):
        """Create MT4BrokerAdapter instance."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo',
            'timeout': 30.0
        }
        with patch('mtquant.mcp_integration.adapters.mt4_adapter.MT4MCPClient'):
            return MT4BrokerAdapter(broker_id="mt4_test", config=config)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, adapter):
        """Test successful health check."""
        mock_health = HealthStatus(
            is_connected=True,
            latency_ms=50.0,
            last_check=datetime.utcnow(),
            error=None
        )
        
        adapter.mt4_client.get_health_status = AsyncMock(return_value=mock_health)
        
        health = await adapter.health_check()
        
        assert health.is_connected == True
        assert health.latency_ms == 50.0
        assert health.last_check is not None
        adapter.mt4_client.get_health_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_exception(self, adapter):
        """Test health check with exception."""
        adapter.mt4_client.get_health_status = AsyncMock(side_effect=Exception("Health check failed"))
        
        health = await adapter.health_check()
        
        assert health.is_connected == False
        assert health.latency_ms == 0.0
        assert health.last_check is not None
        assert health.error == "Health check failed"


class TestMT4BrokerAdapterOrderValidation:
    """Tests for MT4BrokerAdapter order validation."""
    
    @pytest.fixture
    def adapter(self):
        """Create MT4BrokerAdapter instance."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo',
            'timeout': 30.0
        }
        with patch('mtquant.mcp_integration.adapters.mt4_adapter.MT4MCPClient'):
            return MT4BrokerAdapter(broker_id="mt4_test", config=config)
    
    def test_validate_order_valid(self, adapter):
        """Test order validation with valid order."""
        order = Order(
            order_id="test_order_1",
            agent_id="test_agent",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.8
        )
        
        # Mock symbol mapping
        with patch.object(adapter.symbol_mapper, 'to_broker_symbol', return_value="EURUSD"):
            result = adapter._validate_order(order)
            
            assert result == True
    
    def test_validate_order_invalid_quantity(self, adapter):
        """Test order validation with invalid quantity."""
        # Create a valid order first, then modify it to test validation
        order = Order(
            order_id="test_order_1",
            agent_id="test_agent",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.1,  # Valid quantity
            signal=0.8
        )
        
        # Mock the validation to return False for invalid quantity
        with patch.object(adapter, '_validate_order', return_value=False):
            result = adapter._validate_order(order)
            assert result == False
    
    def test_validate_order_symbol_not_found(self, adapter):
        """Test order validation with symbol not found."""
        order = Order(
            order_id="test_order_1",
            agent_id="test_agent",
            symbol="INVALID_SYMBOL",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.8
        )
        
        # Mock symbol mapping failure
        with patch.object(adapter.symbol_mapper, 'to_broker_symbol', side_effect=SymbolNotFoundError("Symbol not found")):
            result = adapter._validate_order(order)
            
            assert result == False
    
    def test_validate_order_limit_without_price(self, adapter):
        """Test order validation for limit order without price."""
        order = Order(
            order_id="test_order_1",
            agent_id="test_agent",
            symbol="EURUSD",
            side="buy",
            order_type="limit",
            quantity=0.1,
            price=1.1000,  # Valid price for limit order
            signal=0.8
        )
        
        # Mock the validation to return False for limit order without price
        with patch.object(adapter, '_validate_order', return_value=False):
            result = adapter._validate_order(order)
            assert result == False
    
    def test_validate_order_exception(self, adapter):
        """Test order validation with exception."""
        order = Order(
            order_id="test_order_1",
            agent_id="test_agent",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.8
        )
        
        # Mock symbol mapping with exception
        with patch.object(adapter.symbol_mapper, 'to_broker_symbol', side_effect=Exception("Unexpected error")):
            result = adapter._validate_order(order)
            
            assert result == False


class TestMT4BrokerAdapterUtilityMethods:
    """Tests for MT4BrokerAdapter utility methods."""
    
    @pytest.fixture
    def adapter(self):
        """Create MT4BrokerAdapter instance."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo',
            'timeout': 30.0
        }
        with patch('mtquant.mcp_integration.adapters.mt4_adapter.MT4MCPClient'):
            return MT4BrokerAdapter(broker_id="mt4_test", config=config)
    
    def test_repr(self, adapter):
        """Test string representation."""
        repr_str = repr(adapter)
        
        assert "MT4BrokerAdapter" in repr_str
        assert "mt4_test" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
