"""
Corrected tests for MT4MCPClient matching actual implementation.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from mtquant.mcp_integration.clients.mt4_mcp_client import MT4MCPClient, HealthStatus, TIMEFRAME_MAP
from mtquant.mcp_integration.models.order import Order
from mtquant.mcp_integration.models.position import Position
from mtquant.utils.exceptions import (
    BrokerConnectionError,
    OrderExecutionError,
    MarketDataError
)


class TestMT4MCPClientInitialization:
    """Tests for MT4MCPClient initialization."""
    
    def test_mt4_mcp_client_initialization(self):
        """Test MT4MCPClient initialization with config."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo',
            'mcp_endpoint': 'http://localhost:3000'
        }
        
        client = MT4MCPClient(broker_id="test_broker", config=config)
        
        assert client.broker_id == "test_broker"
        assert client.config == config
        assert client.base_url == "http://localhost:3000"
        assert client._connected == False
        assert client._last_health_check is None
    
    def test_mt4_mcp_client_initialization_defaults(self):
        """Test MT4MCPClient initialization with default mcp_endpoint."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        
        client = MT4MCPClient(broker_id="test_broker", config=config)
        
        # Check default base_url
        assert client.base_url == "http://localhost:3000"
        assert client._connected == False


class TestMT4MCPClientConnection:
    """Tests for MT4MCPClient connection management."""
    
    @pytest.fixture
    def client(self):
        """Create MT4MCPClient instance for testing."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        return MT4MCPClient(broker_id="test_broker", config=config)
    
    @pytest.mark.asyncio
    async def test_connect_success(self, client):
        """Test successful connection."""
        mock_response = Mock()
        mock_response.json = Mock(return_value={'status': 'success'})
        mock_response.raise_for_status = Mock()
        
        client.client.post = AsyncMock(return_value=mock_response)
        
        connected = await client.connect()
        
        assert connected == True
        assert client._connected == True
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, client):
        """Test connection failure."""
        import httpx
        client.client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
        
        with pytest.raises(BrokerConnectionError):
            await client.connect()
    
    @pytest.mark.asyncio
    async def test_disconnect_success(self, client):
        """Test successful disconnection."""
        client._connected = True
        client.client.post = AsyncMock()
        client.client.aclose = AsyncMock()
        
        await client.disconnect()
        
        assert client._connected == False


class TestMT4MCPClientHealthCheck:
    """Tests for MT4MCPClient health checking."""
    
    @pytest.fixture
    def client(self):
        """Create MT4MCPClient instance for testing."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        return MT4MCPClient(broker_id="test_broker", config=config)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test successful health check."""
        client._connected = True
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        client.client.get = AsyncMock(return_value=mock_response)
        
        result = await client.health_check()
        
        assert result == True
        assert client._last_health_check is not None
    
    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, client):
        """Test health check when not connected."""
        client._connected = False
        
        result = await client.health_check()
        
        assert result == False
    
    @pytest.mark.asyncio
    async def test_get_health_status(self, client):
        """Test getting health status."""
        client._connected = True
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        client.client.get = AsyncMock(return_value=mock_response)
        
        status = await client.get_health_status()
        
        assert isinstance(status, HealthStatus)
        assert status.is_connected == True
        assert status.latency_ms >= 0


class TestMT4MCPClientMarketData:
    """Tests for MT4MCPClient market data operations."""
    
    @pytest.fixture
    def client(self):
        """Create MT4MCPClient instance for testing."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        client = MT4MCPClient(broker_id="test_broker", config=config)
        client._connected = True
        return client
    
    @pytest.mark.asyncio
    async def test_get_symbols_success(self, client):
        """Test getting symbols list."""
        mock_response = Mock()
        mock_response.json = Mock(return_value={'symbols': ['EURUSD', 'GBPUSD', 'USDJPY']})
        mock_response.raise_for_status = Mock()
        client.client.get = AsyncMock(return_value=mock_response)
        
        symbols = await client.get_symbols()
        
        assert len(symbols) == 3
        assert 'EURUSD' in symbols
    
    @pytest.mark.asyncio
    async def test_get_market_data_success(self, client):
        """Test fetching market data."""
        mock_bars = [
            {'timestamp': 1609459200, 'open': 1.22, 'high': 1.23, 'low': 1.21, 'close': 1.225, 'volume': 1000},
            {'timestamp': 1609462800, 'open': 1.225, 'high': 1.23, 'low': 1.22, 'close': 1.228, 'volume': 1100}
        ]
        mock_response = Mock()
        mock_response.json = Mock(return_value={'bars': mock_bars})
        mock_response.raise_for_status = Mock()
        client.client.get = AsyncMock(return_value=mock_response)
        
        df = await client.get_market_data('EURUSD', 'H1', bars=100)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'timestamp' in df.columns
        assert 'close' in df.columns
    
    @pytest.mark.asyncio
    async def test_get_market_data_not_connected(self, client):
        """Test fetching market data when not connected."""
        client._connected = False
        
        with pytest.raises(BrokerConnectionError):
            await client.get_market_data('EURUSD', 'H1')
    
    @pytest.mark.asyncio
    async def test_get_market_data_invalid_timeframe(self, client):
        """Test fetching market data with invalid timeframe."""
        with pytest.raises(ValueError):
            await client.get_market_data('EURUSD', 'INVALID')


class TestMT4MCPClientOrderOperations:
    """Tests for MT4MCPClient order operations."""
    
    @pytest.fixture
    def client(self):
        """Create MT4MCPClient instance for testing."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        client = MT4MCPClient(broker_id="test_broker", config=config)
        client._connected = True
        return client
    
    @pytest.fixture
    def sample_order(self):
        """Create sample order."""
        return Order(
            agent_id="test_agent",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.8
        )
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, client, sample_order):
        """Test placing order successfully."""
        mock_response = Mock()
        mock_response.json = Mock(return_value={'status': 'success', 'order_id': '12345'})
        mock_response.raise_for_status = Mock()
        client.client.post = AsyncMock(return_value=mock_response)
        
        order_id = await client.place_order(sample_order)
        
        assert order_id == '12345'
    
    @pytest.mark.asyncio
    async def test_place_order_not_connected(self, client, sample_order):
        """Test placing order when not connected."""
        client._connected = False
        
        with pytest.raises(BrokerConnectionError):
            await client.place_order(sample_order)


class TestMT4MCPClientPositionOperations:
    """Tests for MT4MCPClient position operations."""
    
    @pytest.fixture
    def client(self):
        """Create MT4MCPClient instance for testing."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        client = MT4MCPClient(broker_id="test_broker", config=config)
        client._connected = True
        return client
    
    @pytest.mark.asyncio
    async def test_get_positions_success(self, client):
        """Test getting positions."""
        mock_positions = [
            {
                'ticket': 123456,
                'symbol': 'EURUSD',
                'type': 0,  # Buy
                'volume': 0.1,
                'price_open': 1.2200,
                'price_current': 1.2250,
                'profit': 50.0,
                'time': 1609459200
            }
        ]
        mock_response = Mock()
        mock_response.json = Mock(return_value={'positions': mock_positions})
        mock_response.raise_for_status = Mock()
        client.client.get = AsyncMock(return_value=mock_response)
        
        positions = await client.get_positions()
        
        assert len(positions) == 1
        assert isinstance(positions[0], Position)
        assert positions[0].symbol == 'EURUSD'
    
    @pytest.mark.asyncio
    async def test_get_positions_not_connected(self, client):
        """Test getting positions when not connected."""
        client._connected = False
        
        with pytest.raises(BrokerConnectionError):
            await client.get_positions()
    
    @pytest.mark.asyncio
    async def test_close_position_success(self, client):
        """Test closing position."""
        mock_response = Mock()
        mock_response.json = Mock(return_value={'status': 'success'})
        mock_response.raise_for_status = Mock()
        client.client.post = AsyncMock(return_value=mock_response)
        
        result = await client.close_position('123456')
        
        assert result == True


class TestMT4MCPClientAccountOperations:
    """Tests for MT4MCPClient account operations."""
    
    @pytest.fixture
    def client(self):
        """Create MT4MCPClient instance for testing."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        client = MT4MCPClient(broker_id="test_broker", config=config)
        client._connected = True
        return client
    
    @pytest.mark.asyncio
    async def test_get_account_info_success(self, client):
        """Test getting account info."""
        mock_account = {
            'balance': 10000.0,
            'equity': 10050.0,
            'margin': 200.0,
            'margin_free': 9800.0,
            'profit': 50.0,
            'leverage': 100
        }
        mock_response = Mock()
        mock_response.json = Mock(return_value=mock_account)
        mock_response.raise_for_status = Mock()
        client.client.get = AsyncMock(return_value=mock_response)
        
        account_info = await client.get_account_info()
        
        assert 'balance' in account_info
        assert account_info['balance'] == 10000.0
        assert account_info['equity'] == 10050.0
    
    @pytest.mark.asyncio
    async def test_get_account_info_not_connected(self, client):
        """Test getting account info when not connected."""
        client._connected = False
        
        with pytest.raises(BrokerConnectionError):
            await client.get_account_info()


class TestMT4MCPClientUtilityMethods:
    """Tests for MT4MCPClient utility methods."""
    
    def test_repr(self):
        """Test string representation."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        client = MT4MCPClient(broker_id="test_broker", config=config)
        
        repr_str = repr(client)
        
        assert 'test_broker' in repr_str
        assert 'disconnected' in repr_str
    
    def test_timeframe_map(self):
        """Test timeframe mapping constant."""
        assert 'H1' in TIMEFRAME_MAP
        assert TIMEFRAME_MAP['H1'] == 60
        assert TIMEFRAME_MAP['D1'] == 1440

