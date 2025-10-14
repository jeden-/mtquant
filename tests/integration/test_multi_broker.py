"""
Integration tests for Multi-Broker Support.

These tests verify that the system can handle multiple brokers (MT4/MT5)
with intelligent routing, failover, and aggregated operations.

Test Setup:
- Load broker configs from YAML
- Create and initialize BrokerManager with multiple brokers
- Test various multi-broker operations end-to-end

Note: These tests require MT4/MT5 terminals to be running and logged in.
If brokers are not available, tests will be skipped.
"""

import pytest
import pytest_asyncio
import asyncio
import yaml
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

from mtquant.mcp_integration.managers import BrokerManager
from mtquant.mcp_integration.models.order import Order, OrderSide, OrderType, OrderStatus
from mtquant.mcp_integration.models.position import Position
from mtquant.utils.exceptions import BrokerError, BrokerConnectionError
from mtquant.utils.logger import setup_logger, get_logger

# Setup logging
setup_logger(level="INFO")
logger = get_logger(__name__)

# Markers for test categorization
pytestmark = [
    pytest.mark.integration,
    pytest.mark.multi_broker,
    pytest.mark.manager
]


def load_multi_broker_configs() -> List[Dict[str, Any]]:
    """Load multi-broker configurations from YAML file."""
    try:
        with open('config/brokers.yaml', 'r') as f:
            brokers_config = yaml.safe_load(f)
        
        # Get demo accounts for both MT4 and MT5
        demo_accounts = brokers_config.get('demo_accounts', {})
        configs = []
        
        # Add MT5 demo account
        if 'oanda_mt5' in demo_accounts:
            mt5_config = demo_accounts['oanda_mt5'].copy()
            mt5_config['is_primary'] = True
            configs.append(mt5_config)
        
        # Add MT4 demo account
        if 'generic_mt4' in demo_accounts:
            mt4_config = demo_accounts['generic_mt4'].copy()
            mt4_config['is_primary'] = False
            configs.append(mt4_config)
        
        logger.info(f"Loaded {len(configs)} multi-broker configs")
        return configs
        
    except Exception as e:
        logger.error(f"Failed to load broker configs: {e}")
        return []


@pytest_asyncio.fixture
async def multi_broker_manager():
    """Create and initialize BrokerManager with multiple brokers."""
    configs = load_multi_broker_configs()
    
    if len(configs) < 2:
        pytest.skip("Need at least 2 broker configs for multi-broker tests")
    
    manager = BrokerManager()
    
    try:
        await manager.initialize(configs)
        yield manager
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_multiple_broker_initialization(multi_broker_manager):
    """Test that multiple brokers can be initialized successfully."""
    manager = multi_broker_manager
    
    # Check initialization
    assert manager.is_initialized()
    
    # Get broker status
    status = await manager.get_broker_status()
    
    # Should have multiple brokers
    assert status['connection_stats']['total_adapters'] >= 2
    
    # Should have at least one healthy broker
    assert len(status['healthy_brokers']) >= 1
    
    # Should have a primary broker
    assert status['primary_broker'] is not None
    
    logger.info(f"Multi-broker initialization successful: {status}")


@pytest.mark.asyncio
async def test_intelligent_routing(multi_broker_manager):
    """Test intelligent broker routing for orders."""
    manager = multi_broker_manager
    
    # Create test order
    order = Order(
        symbol='XAUUSD',
        side=OrderSide.BUY,
        quantity=0.01,
        order_type=OrderType.MARKET,
        agent_id='test_agent',
        signal=0.5  # Required parameter
    )
    
    # Test 1: Place order with no preference (should use primary)
    try:
        order_id = await manager.place_order(order)
        assert order_id is not None
        logger.info(f"Order placed via primary broker: {order_id}")
    except BrokerError as e:
        logger.warning(f"Order placement failed (expected in test): {e}")
        pytest.skip("Order placement not available in test environment")
    
    # Test 2: Place order with specific broker preference
    status = await manager.get_broker_status()
    if len(status['healthy_brokers']) >= 2:
        preferred_broker = status['healthy_brokers'][1]  # Use second healthy broker
        
        try:
            order_id = await manager.place_order(order, preferred_broker=preferred_broker)
            assert order_id is not None
            logger.info(f"Order placed via preferred broker {preferred_broker}: {order_id}")
        except BrokerError as e:
            logger.warning(f"Order placement with preference failed: {e}")


@pytest.mark.asyncio
async def test_failover(multi_broker_manager):
    """Test failover functionality."""
    manager = multi_broker_manager
    
    # Get initial status
    initial_status = await manager.get_broker_status()
    initial_primary = initial_status['primary_broker']
    
    # Test failover (simulate primary broker failure)
    try:
        # Get connection pool
        pool = manager.connection_pool
        
        # Simulate failover
        new_primary = await pool.failover_to_backup()
        
        # Check that failover occurred
        assert new_primary != initial_primary
        assert new_primary in initial_status['healthy_brokers']
        
        logger.info(f"Failover successful: {initial_primary} -> {new_primary}")
        
    except BrokerConnectionError as e:
        logger.warning(f"Failover test skipped: {e}")
        pytest.skip("Failover not available (need multiple healthy brokers)")


@pytest.mark.asyncio
async def test_aggregated_positions(multi_broker_manager):
    """Test position aggregation across multiple brokers."""
    manager = multi_broker_manager
    
    # Get positions from all brokers
    all_positions = await manager.get_positions()
    
    # Should return a list (even if empty)
    assert isinstance(all_positions, list)
    
    # Log position details
    logger.info(f"Retrieved {len(all_positions)} positions from all brokers")
    
    for position in all_positions:
        logger.info(f"Position: {position.symbol} {position.side} {position.quantity} "
                   f"via {position.broker_id}")


@pytest.mark.asyncio
async def test_aggregated_account_info(multi_broker_manager):
    """Test account information aggregation."""
    manager = multi_broker_manager
    
    # Get account info from all brokers
    all_account_info = await manager.get_account_info()
    
    # Should return a dictionary
    assert isinstance(all_account_info, dict)
    
    # Should have entries for each broker
    status = await manager.get_broker_status()
    assert len(all_account_info) >= len(status['healthy_brokers'])
    
    # Log account details
    for broker_id, account_info in all_account_info.items():
        logger.info(f"Account {broker_id}: Balance={account_info.get('balance', 'N/A')}, "
                   f"Equity={account_info.get('equity', 'N/A')}")


@pytest.mark.asyncio
async def test_market_data_routing(multi_broker_manager):
    """Test market data fetching with broker routing."""
    manager = multi_broker_manager
    
    # Test market data from different brokers
    status = await manager.get_broker_status()
    
    for broker_id in status['healthy_brokers'][:2]:  # Test first 2 healthy brokers
        try:
            data = await manager.get_market_data(
                symbol='XAUUSD',
                timeframe='H1',
                bars=10,
                broker_id=broker_id
            )
            
            # Should return DataFrame
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert 'close' in data.columns
            
            logger.info(f"Market data from {broker_id}: {len(data)} bars")
            
        except BrokerError as e:
            logger.warning(f"Market data from {broker_id} failed: {e}")


@pytest.mark.asyncio
async def test_connection_pool_health_monitoring(multi_broker_manager):
    """Test connection pool health monitoring."""
    manager = multi_broker_manager
    
    # Get connection stats
    stats = manager.connection_pool.get_connection_stats()
    
    # Should have statistics
    assert stats.total_adapters >= 2
    assert stats.last_health_check is not None
    
    # Test health check
    health_results = await manager.connection_pool.health_check_all()
    
    # Should have health status for each broker
    assert len(health_results) == stats.total_adapters
    
    # Log health status
    for broker_id, health in health_results.items():
        logger.info(f"Health {broker_id}: {health.is_connected}, "
                   f"latency={health.latency_ms:.1f}ms")


@pytest.mark.asyncio
async def test_symbol_mapping_multi_broker(multi_broker_manager):
    """Test symbol mapping across multiple brokers."""
    from mtquant.mcp_integration.managers.symbol_mapper import SymbolMapper
    
    # Test symbol mapping for different brokers
    test_symbols = ['XAUUSD', 'EURUSD', 'BTCUSD']
    status = await multi_broker_manager.get_broker_status()
    
    for symbol in test_symbols:
        for broker_id in status['healthy_brokers']:
            try:
                # Test standard to broker symbol mapping
                broker_symbol = SymbolMapper.to_broker_symbol(symbol, broker_id)
                assert broker_symbol is not None
                
                # Test reverse mapping
                standard_symbol = SymbolMapper.to_standard_symbol(broker_symbol, broker_id)
                assert standard_symbol == symbol
                
                logger.info(f"Symbol mapping {symbol} <-> {broker_symbol} for {broker_id}")
                
            except Exception as e:
                logger.warning(f"Symbol mapping failed for {symbol} at {broker_id}: {e}")


@pytest.mark.asyncio
async def test_broker_status_monitoring(multi_broker_manager):
    """Test comprehensive broker status monitoring."""
    manager = multi_broker_manager
    
    # Get detailed status
    status = await manager.get_broker_status()
    
    # Should have all required fields
    required_fields = [
        'healthy_brokers', 'unhealthy_brokers', 'primary_broker',
        'connection_stats', 'last_health_check'
    ]
    
    for field in required_fields:
        assert field in status
    
    # Connection stats should have required fields
    stats_fields = [
        'total_adapters', 'healthy_adapters', 'backup_brokers',
        'total_uptime_hours', 'total_failures'
    ]
    
    for field in stats_fields:
        assert field in status['connection_stats']
    
    # Log comprehensive status
    logger.info(f"Broker Status:")
    logger.info(f"  Healthy: {status['healthy_brokers']}")
    logger.info(f"  Unhealthy: {status['unhealthy_brokers']}")
    logger.info(f"  Primary: {status['primary_broker']}")
    logger.info(f"  Stats: {status['connection_stats']}")


@pytest.mark.asyncio
async def test_multi_broker_error_handling(multi_broker_manager):
    """Test error handling in multi-broker scenarios."""
    manager = multi_broker_manager
    
    # Test with invalid broker ID
    try:
        await manager.get_positions(broker_id="invalid_broker")
        assert False, "Should have raised BrokerError"
    except BrokerError:
        logger.info("Correctly handled invalid broker ID")
    
    # Test with invalid symbol
    try:
        await manager.get_market_data(
            symbol='INVALID_SYMBOL',
            timeframe='H1',
            bars=10
        )
        assert False, "Should have raised BrokerError"
    except BrokerError:
        logger.info("Correctly handled invalid symbol")
    
    # Test with uninitialized manager
    uninit_manager = BrokerManager()
    try:
        await uninit_manager.get_positions()
        assert False, "Should have raised BrokerError"
    except BrokerError:
        logger.info("Correctly handled uninitialized manager")


@pytest.mark.asyncio
async def test_multi_broker_performance(multi_broker_manager):
    """Test performance of multi-broker operations."""
    manager = multi_broker_manager
    
    # Test concurrent operations
    import time
    start_time = time.time()
    
    # Run multiple operations concurrently
    tasks = [
        manager.get_positions(),
        manager.get_account_info(),
        manager.get_broker_status(),
        manager.get_market_data('XAUUSD', 'H1', bars=5),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Should complete within reasonable time
    assert duration < 10.0  # 10 seconds max
    
    # Log performance
    logger.info(f"Concurrent operations completed in {duration:.2f}s")
    
    # Check results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Task {i} failed: {result}")
        else:
            logger.info(f"Task {i} completed successfully")


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_multiple_broker_initialization())
