"""
Comprehensive tests for MT5MCPClient to achieve >85% coverage.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path

from mtquant.mcp_integration.clients.mt5_mcp_client import MT5MCPClient
from mtquant.utils.exceptions import BrokerConnectionError, BrokerError


class TestMT5MCPClientComprehensive:
    """Comprehensive tests for MT5MCPClient."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for MT5MCPClient."""
        return {
            'mcp_server_path': 'mcp_servers/mt5/server',
            'account': 62675178,
            'password': '9Rb!Z8*K',
            'server': 'OANDATMS-MT5'
        }
    
    @pytest.fixture
    def mt5_client(self, sample_config):
        """Create MT5MCPClient instance."""
        return MT5MCPClient(broker_id="test_broker", config=sample_config)
    
    def test_mt5_client_initialization(self, mt5_client, sample_config):
        """Test MT5MCPClient initialization."""
        assert mt5_client.broker_id == "test_broker"
        assert mt5_client.config == sample_config
        assert mt5_client.mcp_server_path == sample_config['mcp_server_path']
        assert mt5_client.account == sample_config['account']
        assert mt5_client.password == sample_config['password']
        assert mt5_client.server == sample_config['server']
        assert mt5_client.is_connected is False
        assert mt5_client.logger is not None
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mt5_client):
        """Test successful connection."""
        # Mock MCP client
        mock_mcp_client = AsyncMock()
        mock_mcp_client.initialize.return_value = {"status": "success"}
        mock_mcp_client.login.return_value = {"status": "success", "account_info": {"balance": 10000}}
        
        with patch('mtquant.mcp_integration.clients.mt5_mcp_client.MCPClient', return_value=mock_mcp_client):
            result = await mt5_client.connect()
            
            assert result is True
            assert mt5_client.is_connected is True
            assert mt5_client.mcp_client == mock_mcp_client
            mock_mcp_client.initialize.assert_called_once()
            mock_mcp_client.login.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_initialization_failure(self, mt5_client):
        """Test connection failure during initialization."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.initialize.return_value = {"status": "error", "message": "Initialization failed"}
        
        with patch('mtquant.mcp_integration.clients.mt5_mcp_client.MCPClient', return_value=mock_mcp_client):
            with pytest.raises(BrokerConnectionError):
                await mt5_client.connect()
            
            assert mt5_client.is_connected is False
    
    @pytest.mark.asyncio
    async def test_connect_login_failure(self, mt5_client):
        """Test connection failure during login."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.initialize.return_value = {"status": "success"}
        mock_mcp_client.login.return_value = {"status": "error", "message": "Login failed"}
        
        with patch('mtquant.mcp_integration.clients.mt5_mcp_client.MCPClient', return_value=mock_mcp_client):
            with pytest.raises(BrokerConnectionError):
                await mt5_client.connect()
            
            assert mt5_client.is_connected is False
    
    @pytest.mark.asyncio
    async def test_connect_already_connected(self, mt5_client):
        """Test connecting when already connected."""
        mt5_client.is_connected = True
        mt5_client.mcp_client = AsyncMock()
        
        result = await mt5_client.connect()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_disconnect_success(self, mt5_client):
        """Test successful disconnection."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.logout.return_value = {"status": "success"}
        mock_mcp_client.close.return_value = None
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        result = await mt5_client.disconnect()
        
        assert result is True
        assert mt5_client.is_connected is False
        mock_mcp_client.logout.assert_called_once()
        mock_mcp_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, mt5_client):
        """Test disconnecting when not connected."""
        result = await mt5_client.disconnect()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_disconnect_error(self, mt5_client):
        """Test disconnection error."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.logout.side_effect = Exception("Logout failed")
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        with pytest.raises(BrokerError):
            await mt5_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_account_info_success(self, mt5_client):
        """Test successful account info retrieval."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.get_account_info.return_value = {
            "status": "success",
            "account_info": {
                "balance": 10000.0,
                "equity": 10500.0,
                "margin": 500.0,
                "free_margin": 10000.0,
                "margin_level": 2100.0
            }
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        account_info = await mt5_client.get_account_info()
        
        assert account_info["balance"] == 10000.0
        assert account_info["equity"] == 10500.0
        assert account_info["margin"] == 500.0
        mock_mcp_client.get_account_info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_account_info_not_connected(self, mt5_client):
        """Test getting account info when not connected."""
        with pytest.raises(BrokerConnectionError):
            await mt5_client.get_account_info()
    
    @pytest.mark.asyncio
    async def test_get_account_info_error(self, mt5_client):
        """Test account info retrieval error."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.get_account_info.return_value = {
            "status": "error",
            "message": "Failed to get account info"
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        with pytest.raises(BrokerError):
            await mt5_client.get_account_info()
    
    @pytest.mark.asyncio
    async def test_get_positions_success(self, mt5_client):
        """Test successful positions retrieval."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.get_positions.return_value = {
            "status": "success",
            "positions": [
                {
                    "ticket": 12345,
                    "symbol": "EURUSD",
                    "type": 0,  # Buy
                    "volume": 0.1,
                    "price_open": 1.1000,
                    "price_current": 1.1050,
                    "profit": 50.0
                }
            ]
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        positions = await mt5_client.get_positions()
        
        assert len(positions) == 1
        assert positions[0]["symbol"] == "EURUSD"
        assert positions[0]["volume"] == 0.1
        mock_mcp_client.get_positions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_positions_not_connected(self, mt5_client):
        """Test getting positions when not connected."""
        with pytest.raises(BrokerConnectionError):
            await mt5_client.get_positions()
    
    @pytest.mark.asyncio
    async def test_get_positions_error(self, mt5_client):
        """Test positions retrieval error."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.get_positions.return_value = {
            "status": "error",
            "message": "Failed to get positions"
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        with pytest.raises(BrokerError):
            await mt5_client.get_positions()
    
    @pytest.mark.asyncio
    async def test_get_market_data_success(self, mt5_client):
        """Test successful market data retrieval."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.get_market_data.return_value = {
            "status": "success",
            "data": [
                {"time": "2024-01-01T00:00:00", "open": 1.1000, "high": 1.1050, "low": 1.0950, "close": 1.1025, "volume": 1000},
                {"time": "2024-01-01T01:00:00", "open": 1.1025, "high": 1.1075, "low": 1.1000, "close": 1.1050, "volume": 1200}
            ]
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        market_data = await mt5_client.get_market_data("EURUSD", "1H", 2)
        
        assert len(market_data) == 2
        assert market_data[0]["open"] == 1.1000
        assert market_data[1]["close"] == 1.1050
        mock_mcp_client.get_market_data.assert_called_once_with("EURUSD", "1H", 2)
    
    @pytest.mark.asyncio
    async def test_get_market_data_not_connected(self, mt5_client):
        """Test getting market data when not connected."""
        with pytest.raises(BrokerConnectionError):
            await mt5_client.get_market_data("EURUSD", "1H", 100)
    
    @pytest.mark.asyncio
    async def test_get_market_data_error(self, mt5_client):
        """Test market data retrieval error."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.get_market_data.return_value = {
            "status": "error",
            "message": "Failed to get market data"
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        with pytest.raises(BrokerError):
            await mt5_client.get_market_data("EURUSD", "1H", 100)
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, mt5_client):
        """Test successful order placement."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.place_order.return_value = {
            "status": "success",
            "order_ticket": 67890,
            "message": "Order placed successfully"
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        order_result = await mt5_client.place_order(
            symbol="EURUSD",
            order_type="buy",
            volume=0.1,
            price=1.1000
        )
        
        assert order_result["order_ticket"] == 67890
        assert order_result["status"] == "success"
        mock_mcp_client.place_order.assert_called_once_with(
            symbol="EURUSD",
            order_type="buy",
            volume=0.1,
            price=1.1000
        )
    
    @pytest.mark.asyncio
    async def test_place_order_not_connected(self, mt5_client):
        """Test placing order when not connected."""
        with pytest.raises(BrokerConnectionError):
            await mt5_client.place_order("EURUSD", "buy", 0.1, 1.1000)
    
    @pytest.mark.asyncio
    async def test_place_order_error(self, mt5_client):
        """Test order placement error."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.place_order.return_value = {
            "status": "error",
            "message": "Insufficient margin"
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        with pytest.raises(BrokerError):
            await mt5_client.place_order("EURUSD", "buy", 0.1, 1.1000)
    
    @pytest.mark.asyncio
    async def test_close_position_success(self, mt5_client):
        """Test successful position closure."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.close_position.return_value = {
            "status": "success",
            "message": "Position closed successfully"
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        result = await mt5_client.close_position(12345)
        
        assert result["status"] == "success"
        mock_mcp_client.close_position.assert_called_once_with(12345)
    
    @pytest.mark.asyncio
    async def test_close_position_not_connected(self, mt5_client):
        """Test closing position when not connected."""
        with pytest.raises(BrokerConnectionError):
            await mt5_client.close_position(12345)
    
    @pytest.mark.asyncio
    async def test_close_position_error(self, mt5_client):
        """Test position closure error."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.close_position.return_value = {
            "status": "error",
            "message": "Position not found"
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        with pytest.raises(BrokerError):
            await mt5_client.close_position(12345)
    
    @pytest.mark.asyncio
    async def test_get_symbols_success(self, mt5_client):
        """Test successful symbols retrieval."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.get_symbols.return_value = {
            "status": "success",
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        symbols = await mt5_client.get_symbols()
        
        assert len(symbols) == 4
        assert "EURUSD" in symbols
        assert "XAUUSD" in symbols
        mock_mcp_client.get_symbols.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_symbols_not_connected(self, mt5_client):
        """Test getting symbols when not connected."""
        with pytest.raises(BrokerConnectionError):
            await mt5_client.get_symbols()
    
    @pytest.mark.asyncio
    async def test_get_symbols_error(self, mt5_client):
        """Test symbols retrieval error."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.get_symbols.return_value = {
            "status": "error",
            "message": "Failed to get symbols"
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        with pytest.raises(BrokerError):
            await mt5_client.get_symbols()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, mt5_client):
        """Test successful health check."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.health_check.return_value = {
            "status": "success",
            "connection_status": "connected",
            "server_time": "2024-01-01T12:00:00"
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        health = await mt5_client.health_check()
        
        assert health["status"] == "success"
        assert health["connection_status"] == "connected"
        mock_mcp_client.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, mt5_client):
        """Test health check when not connected."""
        health = await mt5_client.health_check()
        
        assert health["status"] == "error"
        assert health["message"] == "Not connected to broker"
    
    @pytest.mark.asyncio
    async def test_health_check_error(self, mt5_client):
        """Test health check error."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.health_check.return_value = {
            "status": "error",
            "message": "Connection lost"
        }
        
        mt5_client.mcp_client = mock_mcp_client
        mt5_client.is_connected = True
        
        health = await mt5_client.health_check()
        
        assert health["status"] == "error"
        assert health["message"] == "Connection lost"
    
    def test_validate_config_valid(self, sample_config):
        """Test valid configuration validation."""
        client = MT5MCPClient(broker_id="test", config=sample_config)
        
        # Should not raise any exception
        assert client.config == sample_config
    
    def test_validate_config_missing_mcp_server_path(self):
        """Test configuration validation with missing mcp_server_path."""
        config = {
            'account': 62675178,
            'password': '9Rb!Z8*K',
            'server': 'OANDATMS-MT5'
        }
        
        with pytest.raises(ValueError):
            MT5MCPClient(broker_id="test", config=config)
    
    def test_validate_config_missing_account(self):
        """Test configuration validation with missing account."""
        config = {
            'mcp_server_path': 'mcp_servers/mt5/server',
            'password': '9Rb!Z8*K',
            'server': 'OANDATMS-MT5'
        }
        
        with pytest.raises(ValueError):
            MT5MCPClient(broker_id="test", config=config)
    
    def test_validate_config_missing_password(self):
        """Test configuration validation with missing password."""
        config = {
            'mcp_server_path': 'mcp_servers/mt5/server',
            'account': 62675178,
            'server': 'OANDATMS-MT5'
        }
        
        with pytest.raises(ValueError):
            MT5MCPClient(broker_id="test", config=config)
    
    def test_validate_config_missing_server(self):
        """Test configuration validation with missing server."""
        config = {
            'mcp_server_path': 'mcp_servers/mt5/server',
            'account': 62675178,
            'password': '9Rb!Z8*K'
        }
        
        with pytest.raises(ValueError):
            MT5MCPClient(broker_id="test", config=config)
    
    @pytest.mark.asyncio
    async def test_reconnect_after_disconnection(self, mt5_client):
        """Test reconnection after disconnection."""
        # First connection
        mock_mcp_client1 = AsyncMock()
        mock_mcp_client1.initialize.return_value = {"status": "success"}
        mock_mcp_client1.login.return_value = {"status": "success", "account_info": {"balance": 10000}}
        mock_mcp_client1.logout.return_value = {"status": "success"}
        mock_mcp_client1.close.return_value = None
        
        with patch('mtquant.mcp_integration.clients.mt5_mcp_client.MCPClient', return_value=mock_mcp_client1):
            await mt5_client.connect()
            assert mt5_client.is_connected is True
            
            await mt5_client.disconnect()
            assert mt5_client.is_connected is False
        
        # Second connection
        mock_mcp_client2 = AsyncMock()
        mock_mcp_client2.initialize.return_value = {"status": "success"}
        mock_mcp_client2.login.return_value = {"status": "success", "account_info": {"balance": 10000}}
        
        with patch('mtquant.mcp_integration.clients.mt5_mcp_client.MCPClient', return_value=mock_mcp_client2):
            await mt5_client.connect()
            assert mt5_client.is_connected is True
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mt5_client):
        """Test concurrent operations."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.initialize.return_value = {"status": "success"}
        mock_mcp_client.login.return_value = {"status": "success", "account_info": {"balance": 10000}}
        mock_mcp_client.get_account_info.return_value = {
            "status": "success",
            "account_info": {"balance": 10000.0, "equity": 10500.0}
        }
        mock_mcp_client.get_positions.return_value = {
            "status": "success",
            "positions": []
        }
        
        with patch('mtquant.mcp_integration.clients.mt5_mcp_client.MCPClient', return_value=mock_mcp_client):
            await mt5_client.connect()
            
            # Run concurrent operations
            tasks = [
                mt5_client.get_account_info(),
                mt5_client.get_positions(),
                mt5_client.health_check()
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert results[0]["balance"] == 10000.0
            assert len(results[1]) == 0
            assert results[2]["status"] == "success"

