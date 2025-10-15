"""
Extended unit tests for broker_manager.py to increase coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from mtquant.mcp_integration.managers.broker_manager import BrokerManager
from mtquant.mcp_integration.models.order import Order, OrderSide, OrderType
from mtquant.mcp_integration.models.position import Position
from mtquant.utils.exceptions import BrokerError, BrokerConnectionError


class TestBrokerManager:
    """Test BrokerManager class."""
    
    def test_broker_manager_initialization(self):
        """Test BrokerManager initialization."""
        manager = BrokerManager()
        
        assert manager.connection_pool is not None
        assert manager.symbol_mapper is not None
        assert manager.logger is not None
        assert manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        manager = BrokerManager()
        
        # Mock broker configs
        broker_configs = [
            {
                'broker_id': 'ic_markets',
                'platform': 'mt5',
                'is_primary': True,
                'account': 123456,
                'password': 'password',
                'server': 'ICMarkets-Demo'
            },
            {
                'broker_id': 'exness',
                'platform': 'mt4',
                'is_primary': False,
                'account': 789012,
                'password': 'password',
                'server': 'Exness-Demo'
            }
        ]
        
        # Mock connection pool methods
        with patch.object(manager.connection_pool, 'add_adapter', new_callable=AsyncMock) as mock_add, \
             patch.object(manager.connection_pool, 'connect_all', new_callable=AsyncMock) as mock_connect, \
             patch.object(manager.connection_pool, 'start_health_monitoring', new_callable=AsyncMock) as mock_start, \
             patch.object(manager.connection_pool, 'health_check_all', new_callable=AsyncMock) as mock_health:
            
            # Mock successful connections
            mock_connect.return_value = {'ic_markets': True, 'exness': True}
            mock_health.return_value = {'ic_markets': Mock(is_connected=True), 'exness': Mock(is_connected=True)}
            
            await manager.initialize(broker_configs)
            
            assert manager._initialized is True
            mock_add.assert_called()
            mock_connect.assert_called_once()
            mock_start.assert_called_once()
            mock_health.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_no_adapters_created(self):
        """Test initialization when no adapters are created."""
        manager = BrokerManager()
        
        # Mock broker configs with unsupported platform
        broker_configs = [
            {
                'broker_id': 'unsupported',
                'platform': 'unsupported_platform',
                'is_primary': True
            }
        ]
        
        with pytest.raises(BrokerError, match="No adapters were created successfully"):
            await manager.initialize(broker_configs)
        
        assert manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_no_connections(self):
        """Test initialization when no connections succeed."""
        manager = BrokerManager()
        
        # Mock broker configs
        broker_configs = [
            {
                'broker_id': 'ic_markets',
                'platform': 'mt5',
                'is_primary': True,
                'account': 123456,
                'password': 'password',
                'server': 'ICMarkets-Demo'
            }
        ]
        
        # Mock connection pool methods
        with patch.object(manager.connection_pool, 'add_adapter', new_callable=AsyncMock) as mock_add, \
             patch.object(manager.connection_pool, 'connect_all', new_callable=AsyncMock) as mock_connect:
            
            # Mock failed connections
            mock_connect.return_value = {'ic_markets': False}
            
            with pytest.raises(BrokerError, match="Initialization failed"):
                await manager.initialize(broker_configs)
            
            assert manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_exception(self):
        """Test initialization with exception."""
        manager = BrokerManager()
        
        # Mock broker configs
        broker_configs = [
            {
                'broker_id': 'ic_markets',
                'platform': 'mt5',
                'is_primary': True,
                'account': 123456,
                'password': 'password',
                'server': 'ICMarkets-Demo'
            }
        ]
        
        # Mock connection pool methods to raise exception
        with patch.object(manager.connection_pool, 'add_adapter', new_callable=AsyncMock) as mock_add:
            mock_add.side_effect = Exception("Connection failed")
            
            with pytest.raises(BrokerError, match="Initialization failed"):
                await manager.initialize(broker_configs)
            
            assert manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_shutdown_success(self):
        """Test successful shutdown."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock connection pool methods
        with patch.object(manager.connection_pool, 'stop_health_monitoring', new_callable=AsyncMock) as mock_stop, \
             patch.object(manager.connection_pool, 'disconnect_all', new_callable=AsyncMock) as mock_disconnect:
            
            await manager.shutdown()
            
            assert manager._initialized is False
            mock_stop.assert_called_once()
            mock_disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_exception(self):
        """Test shutdown with exception."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock connection pool methods to raise exception
        with patch.object(manager.connection_pool, 'stop_health_monitoring', new_callable=AsyncMock) as mock_stop:
            mock_stop.side_effect = Exception("Stop failed")
            
            # Should not raise exception, just log error
            await manager.shutdown()
            
            # _initialized should remain True if exception occurs during shutdown
            assert manager._initialized is True
    
    @pytest.mark.asyncio
    async def test_place_order_success(self):
        """Test successful order placement."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Create test order
        order = Order(
            order_id="test_order",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            created_at=datetime.now()
        )
        
        # Mock connection pool and adapter
        mock_adapter = AsyncMock()
        mock_adapter.place_order.return_value = "order_123"
        
        with patch.object(manager, '_select_broker_for_order', return_value='ic_markets') as mock_select, \
             patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock, return_value=mock_adapter) as mock_get:
            
            order_id = await manager.place_order(order)
            
            assert order_id == "order_123"
            mock_select.assert_called_once_with(order, None)
            mock_get.assert_called_once_with('ic_markets')
            mock_adapter.place_order.assert_called_once_with(order)
    
    @pytest.mark.asyncio
    async def test_place_order_with_preferred_broker(self):
        """Test order placement with preferred broker."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Create test order
        order = Order(
            order_id="test_order",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            created_at=datetime.now()
        )
        
        # Mock connection pool and adapter
        mock_adapter = AsyncMock()
        mock_adapter.place_order.return_value = "order_123"
        
        with patch.object(manager, '_select_broker_for_order', return_value='exness') as mock_select, \
             patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock, return_value=mock_adapter) as mock_get:
            
            order_id = await manager.place_order(order, preferred_broker='exness')
            
            assert order_id == "order_123"
            mock_select.assert_called_once_with(order, 'exness')
            mock_get.assert_called_once_with('exness')
    
    @pytest.mark.asyncio
    async def test_place_order_not_initialized(self):
        """Test order placement when not initialized."""
        manager = BrokerManager()
        
        # Create test order
        order = Order(
            order_id="test_order",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            created_at=datetime.now()
        )
        
        with pytest.raises(BrokerError, match="BrokerManager not initialized"):
            await manager.place_order(order)
    
    @pytest.mark.asyncio
    async def test_place_order_exception(self):
        """Test order placement with exception."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Create test order
        order = Order(
            order_id="test_order",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            created_at=datetime.now()
        )
        
        # Mock connection pool to raise exception
        with patch.object(manager, '_select_broker_for_order', return_value='ic_markets') as mock_select, \
             patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Adapter not found")
            
            with pytest.raises(BrokerError, match="Order placement failed"):
                await manager.place_order(order)
    
    @pytest.mark.asyncio
    async def test_get_positions_specific_broker(self):
        """Test getting positions from specific broker."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock positions
        mock_positions = [
            Position(
                position_id="pos_1",
                agent_id="test_agent",
                symbol="XAUUSD",
                side="long",
                quantity=0.1,
                entry_price=2000.0,
                current_price=2010.0,
                opened_at=datetime.now()
            )
        ]
        
        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.get_positions.return_value = mock_positions
        
        with patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock, return_value=mock_adapter) as mock_get:
            positions = await manager.get_positions(broker_id='ic_markets')
            
            assert positions == mock_positions
            mock_get.assert_called_once_with('ic_markets')
            mock_adapter.get_positions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_positions_all_brokers(self):
        """Test getting positions from all brokers."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock positions
        mock_positions_1 = [
            Position(
                position_id="pos_1",
                agent_id="test_agent",
                symbol="XAUUSD",
                side="long",
                quantity=0.1,
                entry_price=2000.0,
                current_price=2010.0,
                opened_at=datetime.now()
            )
        ]
        
        mock_positions_2 = [
            Position(
                position_id="pos_2",
                agent_id="test_agent",
                symbol="EURUSD",
                side="short",
                quantity=0.2,
                entry_price=1.1,
                current_price=1.09,
                opened_at=datetime.now()
            )
        ]
        
        # Mock connection pool
        with patch.object(manager.connection_pool, 'adapters', {'ic_markets': Mock(), 'exness': Mock()}), \
             patch.object(manager.connection_pool, 'get_connection_stats', return_value=Mock()), \
             patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock) as mock_get:
            
            # Mock adapter responses
            mock_adapter_1 = AsyncMock()
            mock_adapter_1.get_positions.return_value = mock_positions_1
            mock_adapter_2 = AsyncMock()
            mock_adapter_2.get_positions.return_value = mock_positions_2
            
            mock_get.side_effect = [mock_adapter_1, mock_adapter_2]
            
            positions = await manager.get_positions()
            
            assert len(positions) == 2
            assert positions[0].position_id == "pos_1"
            assert positions[1].position_id == "pos_2"
    
    @pytest.mark.asyncio
    async def test_get_positions_not_initialized(self):
        """Test getting positions when not initialized."""
        manager = BrokerManager()
        
        with pytest.raises(BrokerError, match="BrokerManager not initialized"):
            await manager.get_positions()
    
    @pytest.mark.asyncio
    async def test_get_positions_exception(self):
        """Test getting positions with exception."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock connection pool to raise exception
        with patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Adapter not found")
            
            with pytest.raises(BrokerError, match="Failed to get positions"):
                await manager.get_positions(broker_id='ic_markets')
    
    @pytest.mark.asyncio
    async def test_close_position_success(self):
        """Test successful position closure."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.close_position.return_value = True
        
        with patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock, return_value=mock_adapter) as mock_get:
            result = await manager.close_position("pos_1", "ic_markets")
            
            assert result is True
            mock_get.assert_called_once_with("ic_markets")
            mock_adapter.close_position.assert_called_once_with("pos_1")
    
    @pytest.mark.asyncio
    async def test_close_position_not_initialized(self):
        """Test closing position when not initialized."""
        manager = BrokerManager()
        
        with pytest.raises(BrokerError, match="BrokerManager not initialized"):
            await manager.close_position("pos_1", "ic_markets")
    
    @pytest.mark.asyncio
    async def test_close_position_exception(self):
        """Test closing position with exception."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock connection pool to raise exception
        with patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Adapter not found")
            
            with pytest.raises(BrokerError, match="Failed to close position"):
                await manager.close_position("pos_1", "ic_markets")
    
    @pytest.mark.asyncio
    async def test_get_market_data_specific_broker(self):
        """Test getting market data from specific broker."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock market data
        import pandas as pd
        mock_data = pd.DataFrame({
            'open': [2000.0, 2010.0],
            'high': [2005.0, 2015.0],
            'low': [1995.0, 2005.0],
            'close': [2010.0, 2020.0],
            'volume': [100, 110]
        })
        
        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.get_market_data.return_value = mock_data
        
        with patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock, return_value=mock_adapter) as mock_get:
            data = await manager.get_market_data("XAUUSD", "H1", 100, broker_id="ic_markets")
            
            assert data.equals(mock_data)
            mock_get.assert_called_once_with("ic_markets")
            mock_adapter.get_market_data.assert_called_once_with("XAUUSD", "H1", 100)
    
    @pytest.mark.asyncio
    async def test_get_market_data_healthy_adapter(self):
        """Test getting market data using healthy adapter."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock market data
        import pandas as pd
        mock_data = pd.DataFrame({
            'open': [2000.0, 2010.0],
            'high': [2005.0, 2015.0],
            'low': [1995.0, 2005.0],
            'close': [2010.0, 2020.0],
            'volume': [100, 110]
        })
        
        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.get_market_data.return_value = mock_data
        mock_adapter.get_broker_id.return_value = "ic_markets"
        
        with patch.object(manager.connection_pool, 'get_healthy_adapter', new_callable=AsyncMock, return_value=mock_adapter) as mock_get:
            data = await manager.get_market_data("XAUUSD", "H1", 100)
            
            assert data.equals(mock_data)
            mock_get.assert_called_once()
            mock_adapter.get_market_data.assert_called_once_with("XAUUSD", "H1", 100)
    
    @pytest.mark.asyncio
    async def test_get_market_data_not_initialized(self):
        """Test getting market data when not initialized."""
        manager = BrokerManager()
        
        with pytest.raises(BrokerError, match="BrokerManager not initialized"):
            await manager.get_market_data("XAUUSD", "H1", 100)
    
    @pytest.mark.asyncio
    async def test_get_market_data_exception(self):
        """Test getting market data with exception."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock connection pool to raise exception
        with patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Adapter not found")
            
            with pytest.raises(BrokerError, match="Failed to get market data"):
                await manager.get_market_data("XAUUSD", "H1", 100, broker_id="ic_markets")
    
    @pytest.mark.asyncio
    async def test_get_account_info_specific_broker(self):
        """Test getting account info from specific broker."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock account info
        mock_account_info = {
            'balance': 10000.0,
            'equity': 10050.0,
            'margin': 500.0,
            'free_margin': 9500.0
        }
        
        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.get_account_info.return_value = mock_account_info
        
        with patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock, return_value=mock_adapter) as mock_get:
            account_info = await manager.get_account_info(broker_id="ic_markets")
            
            assert account_info == {"ic_markets": mock_account_info}
            mock_get.assert_called_once_with("ic_markets")
            mock_adapter.get_account_info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_account_info_all_brokers(self):
        """Test getting account info from all brokers."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock account info
        mock_account_info_1 = {
            'balance': 10000.0,
            'equity': 10050.0,
            'margin': 500.0,
            'free_margin': 9500.0
        }
        
        mock_account_info_2 = {
            'balance': 5000.0,
            'equity': 5025.0,
            'margin': 250.0,
            'free_margin': 4750.0
        }
        
        # Mock connection pool
        with patch.object(manager.connection_pool, 'adapters', {'ic_markets': Mock(), 'exness': Mock()}), \
             patch.object(manager.connection_pool, 'get_connection_stats', return_value=Mock()), \
             patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock) as mock_get:
            
            # Mock adapter responses
            mock_adapter_1 = AsyncMock()
            mock_adapter_1.get_account_info.return_value = mock_account_info_1
            mock_adapter_2 = AsyncMock()
            mock_adapter_2.get_account_info.return_value = mock_account_info_2
            
            mock_get.side_effect = [mock_adapter_1, mock_adapter_2]
            
            account_info = await manager.get_account_info()
            
            assert account_info == {
                "ic_markets": mock_account_info_1,
                "exness": mock_account_info_2
            }
    
    @pytest.mark.asyncio
    async def test_get_account_info_not_initialized(self):
        """Test getting account info when not initialized."""
        manager = BrokerManager()
        
        with pytest.raises(BrokerError, match="BrokerManager not initialized"):
            await manager.get_account_info()
    
    @pytest.mark.asyncio
    async def test_get_account_info_exception(self):
        """Test getting account info with exception."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock connection pool to raise exception
        with patch.object(manager.connection_pool, 'get_adapter', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Adapter not found")
            
            with pytest.raises(BrokerError, match="Failed to get account info"):
                await manager.get_account_info(broker_id="ic_markets")
    
    @pytest.mark.asyncio
    async def test_get_broker_status_success(self):
        """Test getting broker status successfully."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock health results
        mock_health_results = {
            'ic_markets': Mock(is_connected=True),
            'exness': Mock(is_connected=False)
        }
        
        # Mock connection stats
        mock_connection_stats = Mock()
        mock_connection_stats.primary_broker = 'ic_markets'
        mock_connection_stats.total_adapters = 2
        mock_connection_stats.healthy_adapters = 1
        mock_connection_stats.backup_brokers = ['exness']
        mock_connection_stats.total_uptime_hours = 100.0
        mock_connection_stats.total_failures = 5
        mock_connection_stats.last_health_check = datetime.now()
        
        with patch.object(manager.connection_pool, 'health_check_all', new_callable=AsyncMock, return_value=mock_health_results) as mock_health, \
             patch.object(manager.connection_pool, 'get_connection_stats', return_value=mock_connection_stats) as mock_stats:
            
            status = await manager.get_broker_status()
            
            assert status['healthy_brokers'] == ['ic_markets']
            assert status['unhealthy_brokers'] == ['exness']
            assert status['primary_broker'] == 'ic_markets'
            assert status['connection_stats']['total_adapters'] == 2
            assert status['connection_stats']['healthy_adapters'] == 1
            assert status['connection_stats']['backup_brokers'] == ['exness']
            assert status['connection_stats']['total_uptime_hours'] == 100.0
            assert status['connection_stats']['total_failures'] == 5
            assert status['last_health_check'] == mock_connection_stats.last_health_check
    
    @pytest.mark.asyncio
    async def test_get_broker_status_not_initialized(self):
        """Test getting broker status when not initialized."""
        manager = BrokerManager()
        
        with pytest.raises(BrokerError, match="BrokerManager not initialized"):
            await manager.get_broker_status()
    
    @pytest.mark.asyncio
    async def test_get_broker_status_exception(self):
        """Test getting broker status with exception."""
        manager = BrokerManager()
        manager._initialized = True
        
        # Mock connection pool to raise exception
        with patch.object(manager.connection_pool, 'health_check_all', new_callable=AsyncMock) as mock_health:
            mock_health.side_effect = Exception("Health check failed")
            
            with pytest.raises(BrokerError, match="Failed to get broker status"):
                await manager.get_broker_status()
    
    def test_is_initialized(self):
        """Test is_initialized method."""
        manager = BrokerManager()
        
        # Initially not initialized
        assert manager.is_initialized() is False
        
        # After initialization
        manager._initialized = True
        assert manager.is_initialized() is True
    
    def test_select_broker_for_order_preferred(self):
        """Test broker selection with preferred broker."""
        manager = BrokerManager()
        
        # Create test order
        order = Order(
            order_id="test_order",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            created_at=datetime.now()
        )
        
        # Mock connection pool
        with patch.object(manager.connection_pool, 'adapters', {'ic_markets': Mock(), 'exness': Mock()}):
            broker_id = manager._select_broker_for_order(order, 'ic_markets')
            
            assert broker_id == 'ic_markets'
    
    def test_select_broker_for_order_primary(self):
        """Test broker selection with primary broker."""
        manager = BrokerManager()
        
        # Create test order
        order = Order(
            order_id="test_order",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            created_at=datetime.now()
        )
        
        # Mock connection stats
        mock_stats = Mock()
        mock_stats.primary_broker = 'ic_markets'
        mock_stats.backup_brokers = ['exness']
        mock_stats.total_adapters = 2
        
        with patch.object(manager.connection_pool, 'adapters', {}), \
             patch.object(manager.connection_pool, 'get_connection_stats', return_value=mock_stats):
            broker_id = manager._select_broker_for_order(order, None)
            
            assert broker_id == 'ic_markets'
    
    def test_select_broker_for_order_backup(self):
        """Test broker selection with backup broker."""
        manager = BrokerManager()
        
        # Create test order
        order = Order(
            order_id="test_order",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            created_at=datetime.now()
        )
        
        # Mock connection stats
        mock_stats = Mock()
        mock_stats.primary_broker = None
        mock_stats.backup_brokers = ['exness']
        mock_stats.total_adapters = 1
        
        with patch.object(manager.connection_pool, 'adapters', {}), \
             patch.object(manager.connection_pool, 'get_connection_stats', return_value=mock_stats):
            broker_id = manager._select_broker_for_order(order, None)
            
            assert broker_id == 'exness'
    
    def test_select_broker_for_order_fallback(self):
        """Test broker selection with fallback."""
        manager = BrokerManager()
        
        # Create test order
        order = Order(
            order_id="test_order",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            created_at=datetime.now()
        )
        
        # Mock connection stats
        mock_stats = Mock()
        mock_stats.primary_broker = None
        mock_stats.backup_brokers = []
        mock_stats.total_adapters = 1
        
        with patch.object(manager.connection_pool, 'adapters', {'ic_markets': Mock()}), \
             patch.object(manager.connection_pool, 'get_connection_stats', return_value=mock_stats):
            broker_id = manager._select_broker_for_order(order, None)
            
            assert broker_id == 'ic_markets'
    
    def test_select_broker_for_order_no_brokers(self):
        """Test broker selection with no brokers available."""
        manager = BrokerManager()
        
        # Create test order
        order = Order(
            order_id="test_order",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            created_at=datetime.now()
        )
        
        # Mock connection stats
        mock_stats = Mock()
        mock_stats.primary_broker = None
        mock_stats.backup_brokers = []
        mock_stats.total_adapters = 0
        
        with patch.object(manager.connection_pool, 'adapters', {}), \
             patch.object(manager.connection_pool, 'get_connection_stats', return_value=mock_stats):
            with pytest.raises(BrokerConnectionError, match="No brokers available"):
                manager._select_broker_for_order(order, None)
    
    def test_select_broker_for_order_exception(self):
        """Test broker selection with exception."""
        manager = BrokerManager()
        
        # Create test order
        order = Order(
            order_id="test_order",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            created_at=datetime.now()
        )
        
        # Mock connection pool to raise exception
        with patch.object(manager.connection_pool, 'adapters', {}), \
             patch.object(manager.connection_pool, 'get_connection_stats', side_effect=Exception("Stats failed")):
            with pytest.raises(BrokerConnectionError, match="Broker selection failed"):
                manager._select_broker_for_order(order, None)
    
    def test_repr(self):
        """Test string representation."""
        manager = BrokerManager()
        
        # Mock connection stats
        mock_stats = Mock()
        mock_stats.total_adapters = 2
        mock_stats.healthy_adapters = 1
        
        with patch.object(manager.connection_pool, 'get_connection_stats', return_value=mock_stats):
            # Not initialized
            repr_str = repr(manager)
            assert "status=not initialized" in repr_str
            assert "adapters=2" in repr_str
            assert "healthy=1" in repr_str
            
            # Initialized
            manager._initialized = True
            repr_str = repr(manager)
            assert "status=initialized" in repr_str
            assert "adapters=2" in repr_str
            assert "healthy=1" in repr_str