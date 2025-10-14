"""
Unit tests for MT5BrokerAdapter.

These tests don't require MT5 terminal to be running.
They test the adapter logic, symbol mapping, validation, etc.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from mtquant.mcp_integration.adapters import MT5BrokerAdapter
from mtquant.mcp_integration.models.order import Order
from mtquant.mcp_integration.models.position import Position
from mtquant.utils.exceptions import SymbolNotFoundError, InvalidOrderError


@pytest.fixture
def mock_broker_config():
    """Mock broker configuration."""
    return {
        'broker_id': 'ic_markets',
        'platform': 'mt5',
        'account': 12345678,
        'password': 'test_password',
        'server': 'ICMarkets-Demo',
        'description': 'IC Markets MT5 Demo Account',
        'is_demo': True,
        'enabled': True
    }


@pytest.fixture
def mt5_adapter(mock_broker_config):
    """Create MT5BrokerAdapter with mocked MT5MCPClient."""
    with patch('mtquant.mcp_integration.adapters.mt5_adapter.MT5MCPClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        adapter = MT5BrokerAdapter(
            broker_id=mock_broker_config['broker_id'],
            config=mock_broker_config
        )
        
        # Mock the MT5MCPClient instance
        adapter.mt5_client = mock_client
        
        return adapter


@pytest.fixture
def sample_order():
    """Create sample order for testing."""
    return Order(
        agent_id="test_agent",
        symbol="XAUUSD",
        side="buy",
        order_type="market",
        quantity=0.1,
        signal=0.8,
        created_at=datetime.utcnow(),
        status="pending"
    )


@pytest.fixture
def sample_position():
    """Create sample position for testing."""
    return Position(
        position_id="12345",
        agent_id="test_agent",
        symbol="XAUUSD",
        side="long",
        quantity=0.1,
        entry_price=2050.0,
        current_price=2055.0,
        unrealized_pnl=5.0,
        opened_at=datetime.utcnow(),
        broker_id="ic_markets"
    )


class TestMT5BrokerAdapter:
    """Test cases for MT5BrokerAdapter."""
    
    def test_adapter_initialization(self, mt5_adapter, mock_broker_config):
        """Test adapter initialization."""
        assert mt5_adapter.broker_id == mock_broker_config['broker_id']
        assert mt5_adapter.config == mock_broker_config
        assert mt5_adapter.mt5_client is not None
    
    def test_get_broker_id(self, mt5_adapter):
        """Test get_broker_id method."""
        assert mt5_adapter.get_broker_id() == "ic_markets"
    
    def test_get_config(self, mt5_adapter):
        """Test get_config method."""
        config = mt5_adapter.get_config()
        assert isinstance(config, dict)
        assert config['broker_id'] == "ic_markets"
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mt5_adapter):
        """Test successful connection."""
        mt5_adapter.mt5_client.connect.return_value = True
        
        result = await mt5_adapter.connect()
        
        assert result is True
        mt5_adapter.mt5_client.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, mt5_adapter):
        """Test connection failure."""
        mt5_adapter.mt5_client.connect.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception):
            await mt5_adapter.connect()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mt5_adapter):
        """Test disconnect."""
        await mt5_adapter.disconnect()
        mt5_adapter.mt5_client.disconnect.assert_called_once()
    
    def test_validate_order_valid(self, mt5_adapter, sample_order):
        """Test order validation with valid order."""
        # Mock symbol mapping
        with patch.object(mt5_adapter.symbol_mapper, 'to_broker_symbol', return_value='XAUUSD'):
            result = mt5_adapter._validate_order(sample_order)
            assert result is True
    
    def test_validate_order_invalid_symbol(self, mt5_adapter, sample_order):
        """Test order validation with invalid symbol."""
        # Mock symbol mapping failure
        with patch.object(mt5_adapter.symbol_mapper, 'to_broker_symbol', side_effect=SymbolNotFoundError("Symbol not found")):
            result = mt5_adapter._validate_order(sample_order)
            assert result is False
    
    def test_validate_order_invalid_quantity(self, mt5_adapter):
        """Test order validation with invalid quantity."""
        # Create valid order first, then modify quantity
        valid_order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,  # Valid quantity
            signal=0.8,
            created_at=datetime.utcnow(),
            status="pending"
        )
        
        # Manually set invalid quantity to bypass validation
        valid_order.quantity = -0.1
        
        with patch.object(mt5_adapter.symbol_mapper, 'to_broker_symbol', return_value='XAUUSD'):
            result = mt5_adapter._validate_order(valid_order)
            assert result is False
    
    def test_convert_order_to_mt5(self, mt5_adapter, sample_order):
        """Test order conversion to MT5 format."""
        mt5_order = mt5_adapter._convert_order_to_mt5(sample_order, "XAUUSD")
        
        assert mt5_order.symbol == "XAUUSD"
        assert mt5_order.agent_id == sample_order.agent_id
        assert mt5_order.side == sample_order.side
        assert mt5_order.quantity == sample_order.quantity
    
    def test_convert_mt5_to_position(self, mt5_adapter, sample_position):
        """Test position conversion from MT5 format."""
        # Mock symbol mapping
        with patch.object(mt5_adapter.symbol_mapper, 'to_standard_symbol', return_value='XAUUSD'):
            position = mt5_adapter._convert_mt5_to_position(sample_position)
            
            assert position.symbol == "XAUUSD"
            assert position.position_id == sample_position.position_id
            assert position.quantity == sample_position.quantity
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, mt5_adapter, sample_order):
        """Test successful order placement."""
        # Mock symbol mapping and validation
        with patch.object(mt5_adapter.symbol_mapper, 'to_broker_symbol', return_value='XAUUSD'):
            mt5_adapter.mt5_client.place_order.return_value = "ORDER123"
            
            order_id = await mt5_adapter.place_order(sample_order)
            
            assert order_id == "ORDER123"
            mt5_adapter.mt5_client.place_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_place_order_symbol_not_found(self, mt5_adapter, sample_order):
        """Test order placement with symbol not found."""
        with patch.object(mt5_adapter.symbol_mapper, 'to_broker_symbol', side_effect=SymbolNotFoundError("Symbol not found")):
            with pytest.raises(SymbolNotFoundError):
                await mt5_adapter.place_order(sample_order)
    
    @pytest.mark.asyncio
    async def test_get_positions(self, mt5_adapter, sample_position):
        """Test getting positions."""
        mt5_adapter.mt5_client.get_positions.return_value = [sample_position]
        
        # Mock symbol mapping
        with patch.object(mt5_adapter.symbol_mapper, 'to_standard_symbol', return_value='XAUUSD'):
            positions = await mt5_adapter.get_positions()
            
            assert len(positions) == 1
            assert positions[0].symbol == "XAUUSD"
    
    @pytest.mark.asyncio
    async def test_get_market_data(self, mt5_adapter):
        """Test getting market data."""
        import pandas as pd
        
        # Mock market data
        mock_data = pd.DataFrame({
            'close': [2050.0, 2051.0, 2052.0],
            'open': [2049.0, 2050.0, 2051.0],
            'high': [2051.0, 2052.0, 2053.0],
            'low': [2048.0, 2049.0, 2050.0],
            'volume': [1000, 1100, 1200]
        })
        
        mt5_adapter.mt5_client.get_market_data.return_value = mock_data
        
        # Mock symbol mapping
        with patch.object(mt5_adapter.symbol_mapper, 'to_broker_symbol', return_value='XAUUSD'):
            data = await mt5_adapter.get_market_data('XAUUSD', 'H1', bars=50)
            
            assert 'symbol' in data.columns
            assert data['symbol'].iloc[0] == 'XAUUSD'
            assert len(data) == 3
    
    @pytest.mark.asyncio
    async def test_health_check(self, mt5_adapter):
        """Test health check."""
        from mtquant.mcp_integration.adapters.base_adapter import HealthStatus
        
        mock_health = HealthStatus(
            is_connected=True,
            latency_ms=10.0,
            last_check=datetime.utcnow(),
            error=None
        )
        
        mt5_adapter.mt5_client.get_health_status.return_value = mock_health
        
        health = await mt5_adapter.health_check()
        
        assert health.is_connected is True
        assert health.latency_ms == 10.0
        assert health.error is None
    
    @pytest.mark.asyncio
    async def test_get_account_info(self, mt5_adapter):
        """Test getting account info."""
        mock_account_info = {
            'balance': 10000.0,
            'equity': 10000.0,
            'margin': 0.0,
            'free_margin': 10000.0
        }
        
        mt5_adapter.mt5_client.get_account_info.return_value = mock_account_info
        
        account_info = await mt5_adapter.get_account_info()
        
        assert account_info['balance'] == 10000.0
        assert account_info['equity'] == 10000.0
    
    @pytest.mark.asyncio
    async def test_close_position(self, mt5_adapter):
        """Test closing position."""
        mt5_adapter.mt5_client.close_position.return_value = True
        
        result = await mt5_adapter.close_position("12345")
        
        assert result is True
        mt5_adapter.mt5_client.close_position.assert_called_once_with("12345")
    
    def test_repr(self, mt5_adapter):
        """Test string representation."""
        repr_str = repr(mt5_adapter)
        assert "MT5BrokerAdapter" in repr_str
        assert "ic_markets" in repr_str


# Test markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.mt5
]
