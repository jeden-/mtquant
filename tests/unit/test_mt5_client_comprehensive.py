"""
Comprehensive tests for MT5Client (native MetaTrader5 package client).

Goal: Increase coverage from 28% to 80%+.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from datetime import datetime, timedelta
from typing import Dict, List, Any

from mtquant.mcp_integration.clients.mt5_client import MT5Client, HealthStatus, TIMEFRAME_MAP
from mtquant.mcp_integration.models.order import Order
from mtquant.mcp_integration.models.position import Position
from mtquant.utils.exceptions import (
    BrokerConnectionError, BrokerAPIError, BrokerTimeoutError,
    OrderExecutionError, InsufficientMarginError, MarketDataError
)


class TestMT5ClientBasics:
    """Test MT5Client initialization and basic operations."""
    
    def test_initialization(self):
        """Test client initialization."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo',
            'max_retries': 5,
            'retry_delay': 2.0,
            'timeout': 10.0
        }
        
        client = MT5Client(broker_id="test_broker", config=config)
        
        assert client.broker_id == "test_broker"
        assert client.account == 12345678
        assert client.password == "test_password"
        assert client.server == "ICMarkets-Demo"
        assert client.max_retries == 5
        assert client.retry_delay == 2.0
        assert client.timeout == 10.0
        assert client._connected == False
    
    def test_initialization_defaults(self):
        """Test default values."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        
        client = MT5Client(broker_id="test_broker", config=config)
        
        assert client.max_retries == 3
        assert client.retry_delay == 1.0
        assert client.timeout == 5.0


class TestMT5ClientConnection:
    """Test MT5Client connection logic."""
    
    @pytest.fixture
    def client(self):
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        return MT5Client(broker_id="test_broker", config=config)
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_connect_success(self, mock_mt5, client):
        """Test successful connection."""
        # Mock initialize
        mock_mt5.initialize = Mock(return_value=True)
        
        # Mock account_info (not logged in yet)
        mock_mt5.account_info = Mock(return_value=None)
        
        # Mock login
        mock_mt5.login = Mock(return_value=True)
        
        # Mock account info after login
        mock_account = Mock()
        mock_account.balance = 10000.0
        mock_account.equity = 10000.0
        mock_account.login = 12345678
        mock_account.server = 'ICMarkets-Demo'
        
        # Change account_info to return mock_account after login
        mock_mt5.account_info = Mock(side_effect=[None, mock_account])
        
        result = await client.connect()
        
        assert result == True
        assert client._connected == True
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_connect_already_logged_in(self, mock_mt5, client):
        """Test connect when already logged in."""
        mock_mt5.initialize = Mock(return_value=True)
        
        # Mock already logged in
        mock_account = Mock()
        mock_account.login = 12345678
        mock_account.server = 'ICMarkets-Demo'
        mock_account.balance = 10000.0
        mock_account.equity = 10000.0
        mock_mt5.account_info = Mock(return_value=mock_account)
        
        result = await client.connect()
        
        assert result == True
        assert client._connected == True
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_connect_initialization_failed(self, mock_mt5, client):
        """Test connection failure during initialization."""
        mock_mt5.initialize = Mock(return_value=False)
        mock_mt5.last_error = Mock(return_value=(1, "Initialization failed"))
        
        with pytest.raises(BrokerConnectionError):
            await client.connect()
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_connect_login_failed(self, mock_mt5, client):
        """Test connection failure during login."""
        mock_mt5.initialize = Mock(return_value=True)
        mock_mt5.account_info = Mock(return_value=None)
        mock_mt5.login = Mock(return_value=False)
        mock_mt5.last_error = Mock(return_value=(10004, "Login failed"))
        
        with pytest.raises(BrokerConnectionError):
            await client.connect()
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_disconnect(self, mock_mt5, client):
        """Test disconnection."""
        client._connected = True
        mock_mt5.shutdown = Mock()
        
        await client.disconnect()
        
        assert client._connected == False


class TestMT5ClientHealthCheck:
    """Test health check functionality."""
    
    @pytest.fixture
    def client(self):
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        return MT5Client(broker_id="test_broker", config=config)
    
    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, client):
        """Test health check when not connected."""
        client._connected = False
        result = await client.health_check()
        assert result == False
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_health_check_connected(self, mock_mt5, client):
        """Test health check when connected."""
        client._connected = True
        
        mock_account = Mock()
        mock_account.balance = 10000.0
        mock_mt5.account_info = Mock(return_value=mock_account)
        
        result = await client.health_check()
        
        assert result == True
        assert client._last_health_check is not None
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_health_check_failed(self, mock_mt5, client):
        """Test health check when account_info returns None."""
        client._connected = True
        mock_mt5.account_info = Mock(return_value=None)
        
        result = await client.health_check()
        
        assert result == False
        assert client._connected == False
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_get_health_status_connected(self, mock_mt5, client):
        """Test get_health_status when connected."""
        client._connected = True
        
        mock_account = Mock()
        mock_account.balance = 10000.0
        mock_account.equity = 10050.0
        mock_account.margin = 200.0
        mock_account.margin_free = 9800.0
        mock_mt5.account_info = Mock(return_value=mock_account)
        
        status = await client.get_health_status()
        
        # HealthStatus is a dataclass, check its attributes
        assert isinstance(status, HealthStatus)
        assert status.is_connected == True


class TestMT5ClientMarketData:
    """Test market data retrieval."""
    
    @pytest.fixture
    def client(self):
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        c = MT5Client(broker_id="test_broker", config=config)
        c._connected = True
        return c
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_get_symbols(self, mock_mt5, client):
        """Test getting available symbols."""
        mock_symbol1 = Mock()
        mock_symbol1.name = 'EURUSD'
        mock_symbol2 = Mock()
        mock_symbol2.name = 'GBPUSD'
        
        mock_mt5.symbols_get = Mock(return_value=(mock_symbol1, mock_symbol2))
        
        symbols = await client.get_symbols()
        
        assert len(symbols) == 2
        assert 'EURUSD' in symbols
        assert 'GBPUSD' in symbols
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_get_symbols_not_connected(self, mock_mt5):
        """Test getting symbols when not connected."""
        config = {'account': 12345678, 'password': 'test', 'server': 'demo'}
        client = MT5Client(broker_id="test_broker", config=config)
        client._connected = False
        
        # Mock that returns error when not connected
        mock_mt5.symbols_get = Mock(return_value=None)
        mock_mt5.last_error = Mock(return_value=(-10004, 'No IPC connection'))
        
        with pytest.raises((BrokerConnectionError, BrokerAPIError)):
            await client.get_symbols()
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_get_market_data(self, mock_mt5, client):
        """Test fetching OHLCV data."""
        # Mock copy_rates_from_pos
        mock_rates = np.array([
            (1609459200, 1.22, 1.23, 1.21, 1.225, 1000, 0, 0),
            (1609462800, 1.225, 1.23, 1.22, 1.228, 1100, 0, 0)
        ], dtype=[
            ('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'),
            ('close', 'f8'), ('tick_volume', 'i8'), ('spread', 'i4'), ('real_volume', 'i8')
        ])
        
        mock_mt5.copy_rates_from_pos = Mock(return_value=mock_rates)
        
        df = await client.get_market_data('EURUSD', 'H1', bars=100)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    @pytest.mark.asyncio
    async def test_get_market_data_invalid_timeframe(self, client):
        """Test market data with invalid timeframe."""
        with pytest.raises(MarketDataError):
            await client.get_market_data('EURUSD', 'INVALID', bars=100)
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_get_market_data_not_connected(self, mock_mt5):
        """Test market data when not connected."""
        config = {'account': 12345678, 'password': 'test', 'server': 'demo'}
        client = MT5Client(broker_id="test_broker", config=config)
        client._connected = False
        
        # Mock error
        mock_mt5.copy_rates_from_pos = Mock(return_value=None)
        mock_mt5.last_error = Mock(return_value=(-10004, 'No IPC connection'))
        
        with pytest.raises((BrokerConnectionError, MarketDataError)):
            await client.get_market_data('EURUSD', 'H1', bars=100)


class TestMT5ClientOrders:
    """Test order operations."""
    
    @pytest.fixture
    def client(self):
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        c = MT5Client(broker_id="test_broker", config=config)
        c._connected = True
        return c
    
    @pytest.fixture
    def sample_order(self):
        return Order(
            symbol='EURUSD',
            side='buy',
            order_type='market',
            quantity=0.1,
            agent_id='test_agent',
            signal=0.8
        )
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_place_order_market(self, mock_mt5, client, sample_order):
        """Test placing market order."""
        # Mock MT5 constants
        mock_mt5.TRADE_RETCODE_DONE = 10009
        
        # Mock order_send
        mock_result = Mock()
        mock_result.retcode = 10009  # TRADE_RETCODE_DONE
        mock_result.order = 123456
        mock_result.volume = 0.1
        mock_result.price = 1.2200
        mock_result.comment = "Success"
        
        mock_mt5.order_send = Mock(return_value=mock_result)
        mock_mt5.symbol_info = Mock(return_value=Mock(ask=1.2200, bid=1.2198))
        mock_mt5.last_error = Mock(return_value=(0, "OK"))
        
        order_id = await client.place_order(sample_order)
        
        assert order_id == '123456'
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_place_order_not_connected(self, mock_mt5, sample_order):
        """Test placing order when not connected."""
        config = {'account': 12345678, 'password': 'test', 'server': 'demo'}
        client = MT5Client(broker_id="test_broker", config=config)
        client._connected = False
        
        mock_mt5.order_send = Mock(return_value=None)
        mock_mt5.symbol_info = Mock(return_value=Mock(ask=1.2200, bid=1.2198))
        mock_mt5.last_error = Mock(return_value=(-10004, 'No IPC connection'))
        
        with pytest.raises((BrokerConnectionError, OrderExecutionError)):
            await client.place_order(sample_order)
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_place_order_failed(self, mock_mt5, client, sample_order):
        """Test order placement failure."""
        mock_result = Mock()
        mock_result.retcode = 10013  # TRADE_RETCODE_INVALID_PRICE
        mock_result.comment = "Invalid price"
        
        mock_mt5.order_send = Mock(return_value=mock_result)
        mock_mt5.symbol_info = Mock(return_value=Mock(ask=1.2200, bid=1.2198))
        
        with pytest.raises(OrderExecutionError):
            await client.place_order(sample_order)


class TestMT5ClientPositions:
    """Test position operations."""
    
    @pytest.fixture
    def client(self):
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        c = MT5Client(broker_id="test_broker", config=config)
        c._connected = True
        return c
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_get_positions(self, mock_mt5, client):
        """Test getting open positions."""
        # Mock positions
        mock_pos = Mock()
        mock_pos.ticket = 123456
        mock_pos.symbol = 'EURUSD'
        mock_pos.type = 0  # Buy
        mock_pos.volume = 0.1
        mock_pos.price_open = 1.2200
        mock_pos.price_current = 1.2250
        mock_pos.profit = 50.0
        mock_pos.time = 1609459200
        mock_pos.sl = 1.2100
        mock_pos.tp = 1.2300
        
        mock_mt5.positions_get = Mock(return_value=(mock_pos,))
        
        positions = await client.get_positions()
        
        assert len(positions) == 1
        assert isinstance(positions[0], Position)
        assert positions[0].symbol == 'EURUSD'
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_get_positions_not_connected(self, mock_mt5):
        """Test getting positions when not connected."""
        config = {'account': 12345678, 'password': 'test', 'server': 'demo'}
        client = MT5Client(broker_id="test_broker", config=config)
        client._connected = False
        
        mock_mt5.positions_get = Mock(return_value=None)
        mock_mt5.last_error = Mock(return_value=(-10004, 'No IPC connection'))
        
        # This may return empty list or raise error depending on implementation
        result = await client.get_positions()
        assert result == [] or isinstance(result, list)
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_close_position(self, mock_mt5, client):
        """Test closing position."""
        # Mock MT5 constants
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.POSITION_TYPE_BUY = 0
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_FILLING_IOC = 1
        
        # Mock position
        mock_pos = Mock()
        mock_pos.ticket = 123456
        mock_pos.symbol = 'EURUSD'
        mock_pos.type = 0  # Buy
        mock_pos.volume = 0.1
        mock_pos.sl = 1.2100
        mock_pos.tp = 1.2300
        
        mock_mt5.positions_get = Mock(return_value=(mock_pos,))
        
        # Mock close order
        mock_result = Mock()
        mock_result.retcode = 10009
        mock_result.order = 789
        mock_result.comment = "Position closed"
        
        mock_mt5.order_send = Mock(return_value=mock_result)
        mock_mt5.symbol_info = Mock(return_value=Mock(bid=1.2250))
        mock_mt5.last_error = Mock(return_value=(0, "OK"))
        
        result = await client.close_position('123456')
        
        assert result == True
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_close_position_not_found(self, mock_mt5, client):
        """Test closing non-existent position."""
        mock_mt5.positions_get = Mock(return_value=())
        
        with pytest.raises(OrderExecutionError):
            await client.close_position('999999')


class TestMT5ClientAccountInfo:
    """Test account information methods."""
    
    @pytest.fixture
    def client(self):
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        c = MT5Client(broker_id="test_broker", config=config)
        c._connected = True
        return c
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_get_account_info(self, mock_mt5, client):
        """Test getting account information."""
        mock_account = Mock()
        mock_account.balance = 10000.0
        mock_account.equity = 10050.0
        mock_account.margin = 200.0
        mock_account.margin_free = 9800.0
        mock_account.profit = 50.0
        mock_account.leverage = 100
        
        mock_mt5.account_info = Mock(return_value=mock_account)
        
        info = await client.get_account_info()
        
        assert info['balance'] == 10000.0
        assert info['equity'] == 10050.0
        assert info['margin'] == 200.0
    
    @pytest.mark.asyncio
    @patch('mtquant.mcp_integration.clients.mt5_client.MT5')
    async def test_get_account_info_not_connected(self, mock_mt5):
        """Test getting account info when not connected."""
        config = {'account': 12345678, 'password': 'test', 'server': 'demo'}
        client = MT5Client(broker_id="test_broker", config=config)
        client._connected = False
        
        mock_mt5.account_info = Mock(return_value=None)
        mock_mt5.last_error = Mock(return_value=(-10004, 'No IPC connection'))
        
        with pytest.raises((BrokerConnectionError, BrokerAPIError)):
            await client.get_account_info()


class TestMT5ClientUtilities:
    """Test utility methods."""
    
    def test_repr(self):
        """Test __repr__ method."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        client = MT5Client(broker_id="test_broker", config=config)
        
        repr_str = repr(client)
        
        assert 'MT5Client' in repr_str
        assert 'test_broker' in repr_str
        assert ('disconnected' in repr_str or 'connected=False' in repr_str)
    
    def test_repr_connected(self):
        """Test __repr__ when connected."""
        config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'ICMarkets-Demo'
        }
        client = MT5Client(broker_id="test_broker", config=config)
        client._connected = True
        
        repr_str = repr(client)
        
        assert 'connected' in repr_str

