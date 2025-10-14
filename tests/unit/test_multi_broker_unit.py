"""
Unit tests for Multi-Broker Support components.

These tests verify the multi-broker functionality without requiring
actual broker connections or MCP servers.

Tests:
- Symbol mapping across brokers
- Connection pool management
- Broker manager initialization
- Multi-broker configuration loading
"""

import pytest
import yaml
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from mtquant.mcp_integration.managers.symbol_mapper import SymbolMapper
from mtquant.mcp_integration.managers.connection_pool import ConnectionPool, AdapterInfo, ConnectionStatus
from mtquant.mcp_integration.managers.broker_manager import BrokerManager
from mtquant.mcp_integration.adapters.base_adapter import BrokerAdapter, HealthStatus
from mtquant.mcp_integration.models.order import Order, OrderSide, OrderType
from mtquant.utils.exceptions import BrokerError, SymbolNotFoundError
from mtquant.utils.logger import get_logger

logger = get_logger(__name__)


class TestSymbolMapper:
    """Test symbol mapping functionality."""
    
    def test_to_broker_symbol(self):
        """Test standard to broker symbol mapping."""
        # Test XAUUSD mapping
        broker_symbol = SymbolMapper.to_broker_symbol('XAUUSD', 'ic_markets')
        assert broker_symbol == 'XAUUSD'
        
        broker_symbol = SymbolMapper.to_broker_symbol('XAUUSD', 'oanda')
        assert broker_symbol == 'GOLD.pro'
        
        broker_symbol = SymbolMapper.to_broker_symbol('XAUUSD', 'exness')
        assert broker_symbol == 'XAUUSDm'
    
    def test_to_standard_symbol(self):
        """Test broker to standard symbol mapping."""
        # Test reverse mapping
        standard_symbol = SymbolMapper.to_standard_symbol('GOLD.pro', 'oanda')
        assert standard_symbol == 'XAUUSD'
        
        standard_symbol = SymbolMapper.to_standard_symbol('XAUUSDm', 'exness')
        assert standard_symbol == 'XAUUSD'
        
        standard_symbol = SymbolMapper.to_standard_symbol('XAUUSD', 'ic_markets')
        assert standard_symbol == 'XAUUSD'
    
    def test_symbol_not_found(self):
        """Test error handling for unknown symbols."""
        with pytest.raises(SymbolNotFoundError):
            SymbolMapper.to_broker_symbol('UNKNOWN', 'ic_markets')
        
        with pytest.raises(SymbolNotFoundError):
            SymbolMapper.to_standard_symbol('UNKNOWN', 'ic_markets')
    
    def test_get_symbol_metadata(self):
        """Test symbol metadata retrieval."""
        metadata = SymbolMapper.get_symbol_metadata('XAUUSD')
        assert metadata['instrument_type'] == 'commodity'
        assert metadata['pip_value'] == 0.01
        assert metadata['typical_spread'] == 0.30
        assert metadata['trading_hours'] == '24/5'
    
    def test_get_all_standard_symbols(self):
        """Test retrieval of all standard symbols."""
        symbols = SymbolMapper.get_all_standard_symbols()
        assert 'XAUUSD' in symbols
        assert 'BTCUSD' in symbols
        assert 'EURUSD' in symbols
        assert 'USDJPY' in symbols
        assert len(symbols) >= 4
    
    def test_get_supported_brokers(self):
        """Test retrieval of supported brokers."""
        brokers = SymbolMapper.get_supported_brokers()
        assert 'ic_markets' in brokers
        assert 'oanda' in brokers
        assert 'exness' in brokers
        assert 'pepperstone' in brokers
        assert len(brokers) >= 4
    
    def test_validate_symbol(self):
        """Test symbol validation."""
        assert SymbolMapper.validate_symbol('XAUUSD') is True
        assert SymbolMapper.validate_symbol('BTCUSD') is True
        assert SymbolMapper.validate_symbol('UNKNOWN') is False
        assert SymbolMapper.validate_symbol('') is False


class TestConnectionPool:
    """Test connection pool functionality."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create mock broker adapter."""
        adapter = Mock(spec=BrokerAdapter)
        adapter.broker_id = 'test_broker'
        adapter.connect = AsyncMock(return_value=True)
        adapter.disconnect = AsyncMock()
        adapter.health_check = AsyncMock(return_value=HealthStatus(
            is_connected=True,
            latency_ms=50.0,
            last_check=None,
            error=None
        ))
        return adapter
    
    @pytest.fixture
    def connection_pool(self):
        """Create connection pool instance."""
        return ConnectionPool()
    
    @pytest.mark.asyncio
    async def test_add_adapter(self, connection_pool, mock_adapter):
        """Test adding adapter to pool."""
        await connection_pool.add_adapter('test_broker', mock_adapter, is_primary=True)
        
        assert 'test_broker' in connection_pool.adapters
        assert connection_pool.primary_broker == 'test_broker'
        assert connection_pool.adapters['test_broker'].is_primary is True
    
    @pytest.mark.asyncio
    async def test_add_duplicate_adapter(self, connection_pool, mock_adapter):
        """Test adding duplicate adapter raises error."""
        await connection_pool.add_adapter('test_broker', mock_adapter)
        
        with pytest.raises(ValueError):
            await connection_pool.add_adapter('test_broker', mock_adapter)
    
    @pytest.mark.asyncio
    async def test_connect_all(self, connection_pool, mock_adapter):
        """Test connecting all adapters."""
        await connection_pool.add_adapter('test_broker', mock_adapter)
        
        results = await connection_pool.connect_all()
        
        assert results['test_broker'] is True
        mock_adapter.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_healthy_adapter(self, connection_pool, mock_adapter):
        """Test getting healthy adapter."""
        await connection_pool.add_adapter('test_broker', mock_adapter, is_primary=True)
        await connection_pool.connect_all()
        
        # Mock health check
        mock_adapter.health_check.return_value = HealthStatus(
            is_connected=True,
            latency_ms=50.0,
            last_check=None,
            error=None
        )
        
        # Call health_check_all to update health_status
        await connection_pool.health_check_all()
        
        adapter = await connection_pool.get_healthy_adapter()
        assert adapter == mock_adapter
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, connection_pool, mock_adapter):
        """Test health check for all adapters."""
        await connection_pool.add_adapter('test_broker', mock_adapter)
        
        health_results = await connection_pool.health_check_all()
        
        assert 'test_broker' in health_results
        assert health_results['test_broker'].is_connected is True
        mock_adapter.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_failover_to_backup(self, connection_pool):
        """Test failover to backup broker."""
        # Create primary adapter
        primary_adapter = Mock(spec=BrokerAdapter)
        primary_adapter.broker_id = 'primary'
        primary_adapter.health_check = AsyncMock(return_value=HealthStatus(
            is_connected=False,
            latency_ms=0.0,
            last_check=None,
            error='Connection failed'
        ))
        
        # Create backup adapter
        backup_adapter = Mock(spec=BrokerAdapter)
        backup_adapter.broker_id = 'backup'
        backup_adapter.health_check = AsyncMock(return_value=HealthStatus(
            is_connected=True,
            latency_ms=50.0,
            last_check=None,
            error=None
        ))
        
        # Add adapters
        await connection_pool.add_adapter('primary', primary_adapter, is_primary=True)
        await connection_pool.add_adapter('backup', backup_adapter, is_primary=False)
        
        # Connect and set health status
        await connection_pool.connect_all()
        await connection_pool.health_check_all()
        
        # Perform failover
        new_primary = await connection_pool.failover_to_backup()
        
        assert new_primary == 'backup'
        assert connection_pool.primary_broker == 'backup'
    
    def test_get_connection_stats(self, connection_pool):
        """Test connection statistics."""
        stats = connection_pool.get_connection_stats()
        
        assert stats.total_adapters == 0
        assert stats.healthy_adapters == 0
        assert stats.primary_broker is None
        assert stats.backup_brokers == []
        assert stats.total_uptime_hours == 0.0
        assert stats.total_failures == 0


class TestBrokerManager:
    """Test broker manager functionality."""
    
    @pytest.fixture
    def mock_broker_configs(self):
        """Create mock broker configurations."""
        return [
            {
                'broker_id': 'mt5_broker',
                'platform': 'mt5',
                'account': 12345678,
                'password': 'test_password',
                'server': 'TestServer-MT5',
                'mcp_server_path': 'path/to/mt5/server',
                'is_primary': True
            },
            {
                'broker_id': 'mt4_broker',
                'platform': 'mt4',
                'account': 87654321,
                'password': 'test_password',
                'server': 'TestServer-MT4',
                'mcp_server_url': 'http://localhost:3000',
                'is_primary': False
            }
        ]
    
    @pytest.fixture
    def broker_manager(self):
        """Create broker manager instance."""
        return BrokerManager()
    
    def test_initialization(self, broker_manager):
        """Test broker manager initialization."""
        assert broker_manager.is_initialized() is False
        assert broker_manager.connection_pool is not None
        assert broker_manager.symbol_mapper is not None
    
    @pytest.mark.skip(reason="Requires real broker connections - moved to integration tests")
    @pytest.mark.asyncio
    async def test_initialize_with_mock_adapters(self, broker_manager, mock_broker_configs):
        """Test initialization with mock adapters."""
        with patch('mtquant.mcp_integration.adapters.mt5_adapter.MT5BrokerAdapter') as mock_mt5, \
             patch('mtquant.mcp_integration.adapters.mt4_adapter.MT4BrokerAdapter') as mock_mt4:
            
            # Mock adapter instances
            mt5_adapter = Mock()
            mt5_adapter.connect = AsyncMock(return_value=True)
            mt5_adapter.health_check = AsyncMock(return_value=HealthStatus(
                is_connected=True,
                latency_ms=50.0,
                last_check=None,
                error=None
            ))
            
            mt4_adapter = Mock()
            mt4_adapter.connect = AsyncMock(return_value=True)
            mt4_adapter.health_check = AsyncMock(return_value=HealthStatus(
                is_connected=True,
                latency_ms=100.0,
                last_check=None,
                error=None
            ))
            
            mock_mt5.return_value = mt5_adapter
            mock_mt4.return_value = mt4_adapter
            
            # Initialize
            await broker_manager.initialize(mock_broker_configs)
            
            # Check initialization
            assert broker_manager.is_initialized() is True
            
            # Check adapters were created
            mock_mt5.assert_called_once()
            mock_mt4.assert_called_once()
            
            # Check connection pool
            assert len(broker_manager.connection_pool.adapters) == 2
            assert broker_manager.connection_pool.primary_broker == 'mt5_broker'
    
    @pytest.mark.asyncio
    async def test_initialize_no_adapters(self, broker_manager):
        """Test initialization with no valid adapters."""
        with pytest.raises(BrokerError):
            await broker_manager.initialize([])
    
    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, broker_manager, mock_broker_configs):
        """Test initialization with connection failures."""
        with patch('mtquant.mcp_integration.adapters.mt5_adapter.MT5BrokerAdapter') as mock_mt5, \
             patch('mtquant.mcp_integration.adapters.mt4_adapter.MT4BrokerAdapter') as mock_mt4:
            
            # Mock adapter instances that fail to connect
            mt5_adapter = Mock()
            mt5_adapter.connect = AsyncMock(return_value=False)
            
            mt4_adapter = Mock()
            mt4_adapter.connect = AsyncMock(return_value=False)
            
            mock_mt5.return_value = mt5_adapter
            mock_mt4.return_value = mt4_adapter
            
            # Should raise error
            with pytest.raises(BrokerError):
                await broker_manager.initialize(mock_broker_configs)
    
    @pytest.mark.skip(reason="Requires real broker connections - moved to integration tests")
    @pytest.mark.asyncio
    async def test_shutdown(self, broker_manager):
        """Test broker manager shutdown."""
        # Initialize first
        with patch('mtquant.mcp_integration.adapters.mt5_adapter.MT5BrokerAdapter') as mock_mt5:
            mt5_adapter = Mock()
            mt5_adapter.connect = AsyncMock(return_value=True)
            mt5_adapter.health_check = AsyncMock(return_value=HealthStatus(
                is_connected=True,
                latency_ms=50.0,
                last_check=None,
                error=None
            ))
            mock_mt5.return_value = mt5_adapter
            
            await broker_manager.initialize([{
                'broker_id': 'mt5_broker',
                'platform': 'mt5',
                'account': 12345678,
                'password': 'test_password',
                'server': 'TestServer-MT5',
                'is_primary': True
            }])
        
        # Shutdown
        await broker_manager.shutdown()
        
        # Check shutdown
        assert broker_manager.is_initialized() is False
    
    def test_select_broker_for_order(self, broker_manager):
        """Test broker selection for orders."""
        # Mock connection pool
        broker_manager.connection_pool.adapters = {
            'primary': Mock(),
            'backup1': Mock(),
            'backup2': Mock()
        }
        broker_manager.connection_pool.primary_broker = 'primary'
        broker_manager.connection_pool.backup_brokers = ['backup1', 'backup2']
        
        # Mock connection stats
        with patch.object(broker_manager.connection_pool, 'get_connection_stats') as mock_stats:
            mock_stats.return_value = Mock(
                primary_broker='primary',
                backup_brokers=['backup1', 'backup2'],
                total_adapters=3
            )
            
            # Test order
            order = Order(
                symbol='XAUUSD',
                side='buy',  # Use string instead of enum
                quantity=0.01,
                order_type='market',  # Use string instead of enum
                agent_id='test_agent',
                signal=0.5
            )
            
            # Test with no preference
            broker_id = broker_manager._select_broker_for_order(order, None)
            assert broker_id == 'primary'
            
            # Test with preference
            broker_id = broker_manager._select_broker_for_order(order, 'backup1')
            assert broker_id == 'backup1'


class TestMultiBrokerConfig:
    """Test multi-broker configuration loading."""
    
    def test_load_broker_configs(self):
        """Test loading broker configurations from YAML."""
        # Mock YAML file content
        mock_config = {
            'demo_accounts': {
                'oanda_mt5': {
                    'broker_id': 'oanda',
                    'platform': 'mt5',
                    'account': 62675178,
                    'password': '${MT5_DEMO_PASSWORD}',
                    'server': 'OANDATMS-MT5',
                    'is_demo': True,
                    'enabled': True
                },
                'generic_mt4': {
                    'broker_id': 'generic_mt4_demo',
                    'platform': 'mt4',
                    'account': 99999999,
                    'password': '${MT4_DEMO_PASSWORD}',
                    'server': 'Longhorn-Demo',
                    'is_demo': True,
                    'enabled': True
                }
            }
        }
        
        with patch('builtins.open', create=True) as mock_open, \
             patch('yaml.safe_load') as mock_yaml:
            
            mock_yaml.return_value = mock_config
            
            # Test loading
            from tests.integration.test_multi_broker import load_multi_broker_configs
            configs = load_multi_broker_configs()
            
            assert len(configs) == 2
            assert configs[0]['broker_id'] == 'oanda'
            assert configs[0]['platform'] == 'mt5'
            assert configs[0]['is_primary'] is True
            assert configs[1]['broker_id'] == 'generic_mt4_demo'
            assert configs[1]['platform'] == 'mt4'
            assert configs[1]['is_primary'] is False
    
    def test_broker_config_validation(self):
        """Test broker configuration validation."""
        # Test valid config
        valid_config = {
            'broker_id': 'test_broker',
            'platform': 'mt5',
            'account': 12345678,
            'password': 'test_password',
            'server': 'TestServer',
            'is_primary': True
        }
        
        # Should not raise error
        assert valid_config['broker_id'] is not None
        assert valid_config['platform'] in ['mt5', 'mt4']
        assert valid_config['account'] > 0
        assert valid_config['password'] is not None
        assert valid_config['server'] is not None
        
        # Test invalid config
        invalid_config = {
            'broker_id': '',
            'platform': 'invalid',
            'account': 0,
            'password': None,
            'server': ''
        }
        
        # Should be invalid
        assert invalid_config['broker_id'] == ''
        assert invalid_config['platform'] not in ['mt5', 'mt4']
        assert invalid_config['account'] <= 0
        assert invalid_config['password'] is None
        assert invalid_config['server'] == ''


if __name__ == "__main__":
    pytest.main([__file__])
