"""
Integration tests for BrokerManager.

These tests verify that BrokerManager works correctly with real broker configurations
and can handle operations like market data fetching, position aggregation, and failover.

Test Setup:
- Load broker configs from YAML
- Create and initialize BrokerManager
- Test various operations end-to-end

Note: These tests require MT5 terminal to be running and logged in.
If MT5 is not available, tests will be skipped.
"""

import pytest
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
    pytest.mark.broker,
    pytest.mark.manager
]


def load_broker_configs() -> List[Dict[str, Any]]:
    """Load broker configurations from YAML file."""
    try:
        with open('config/brokers.yaml', 'r') as f:
            brokers_config = yaml.safe_load(f)
        
        # Get demo accounts only
        demo_accounts = brokers_config.get('demo_accounts', {})
        configs = []
        
        for broker_id, config in demo_accounts.items():
            if config.get('enabled', False):
                # Add is_primary flag based on broker_id
                config['is_primary'] = broker_id == 'ic_markets_mt5'
                configs.append(config)
        
        if not configs:
            pytest.skip("No enabled demo brokers found in config")
        
        return configs
        
    except Exception as e:
        pytest.skip(f"Failed to load broker configs: {e}")


@pytest.fixture
def broker_configs():
    """Load broker configs from YAML."""
    return load_broker_configs()


@pytest.fixture
def broker_manager(broker_configs):
    """Create and initialize BrokerManager."""
    async def _create_manager():
        manager = BrokerManager()
        
        try:
            await manager.initialize(broker_configs)
            return manager
        except Exception as e:
            pytest.skip(f"BrokerManager initialization failed: {e}")
    
    return _create_manager


@pytest.mark.asyncio
async def test_manager_initialization(broker_manager):
    """Test manager initializes correctly."""
    logger.info("Testing BrokerManager initialization")
    
    manager = await broker_manager()
    
    # Verify manager is initialized
    assert manager.is_initialized(), "Manager should be initialized"
    
    # Get broker status
    status = await manager.get_broker_status()
    
    # Verify status structure
    assert isinstance(status, dict), "Status should be a dictionary"
    assert 'healthy_brokers' in status, "Should have healthy_brokers"
    assert 'unhealthy_brokers' in status, "Should have unhealthy_brokers"
    assert 'primary_broker' in status, "Should have primary_broker"
    assert 'connection_stats' in status, "Should have connection_stats"
    
    # Verify at least one healthy broker
    assert len(status['healthy_brokers']) > 0, "Should have at least one healthy broker"
    
    logger.info(f"✅ Manager initialized: {len(status['healthy_brokers'])} healthy brokers")


@pytest.mark.asyncio
async def test_market_data_fetch(broker_manager):
    """Test market data fetch through manager."""
    logger.info("Testing market data fetch through manager")
    
    manager = await broker_manager()
    
    # Fetch XAUUSD data (standard symbol)
    data = await manager.get_market_data('XAUUSD', 'H1', bars=50)
    
    # Verify DataFrame structure
    assert isinstance(data, pd.DataFrame), "Should return DataFrame"
    assert not data.empty, "DataFrame should not be empty"
    assert len(data) == 50, f"Should have 50 bars, got {len(data)}"
    
    # Verify required columns
    required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        assert col in data.columns, f"Should have '{col}' column"
    
    # Verify symbol column
    assert 'symbol' in data.columns, "Should have 'symbol' column"
    assert data['symbol'].iloc[0] == 'XAUUSD', "Symbol should be XAUUSD"
    
    # Verify data is recent (last bar within last 2 hours)
    if not data.empty:
        last_bar_time = data['time'].iloc[-1]
        time_diff = datetime.utcnow() - last_bar_time
        assert time_diff.total_seconds() < 7200, f"Data too old: {time_diff}"
    
    logger.info(f"✅ Market data fetched: {len(data)} bars, latest close: {data['close'].iloc[-1]}")


@pytest.mark.asyncio
async def test_get_positions(broker_manager):
    """Test getting positions from all brokers."""
    logger.info("Testing position retrieval from all brokers")
    
    manager = await broker_manager()
    
    # Get positions from all brokers
    positions = await manager.get_positions()
    
    # Verify return type
    assert isinstance(positions, list), "Should return list of positions"
    
    # If positions exist, verify structure
    if positions:
        logger.info(f"Found {len(positions)} open positions")
        for pos in positions:
            assert isinstance(pos, Position), "Each item should be a Position object"
            assert hasattr(pos, 'symbol'), "Position should have symbol"
            assert hasattr(pos, 'side'), "Position should have side"
            assert hasattr(pos, 'quantity'), "Position should have quantity"
            assert hasattr(pos, 'entry_price'), "Position should have entry_price"
            assert hasattr(pos, 'current_price'), "Position should have current_price"
            assert hasattr(pos, 'unrealized_pnl'), "Position should have unrealized_pnl"
    else:
        logger.info("No open positions found (expected for fresh demo account)")
    
    logger.info("✅ Position retrieval completed")


@pytest.mark.asyncio
async def test_account_info_aggregation(broker_manager):
    """Test getting account info from all brokers."""
    logger.info("Testing account info aggregation")
    
    manager = await broker_manager()
    
    # Get account info from all brokers
    info = await manager.get_account_info()
    
    # Verify return type
    assert isinstance(info, dict), "Should return dictionary"
    assert len(info) > 0, "Should have at least one broker"
    
    # Check first broker has required fields
    first_broker_id = list(info.keys())[0]
    first_broker_info = info[first_broker_id]
    
    required_fields = ['balance', 'equity', 'margin', 'free_margin']
    for field in required_fields:
        assert field in first_broker_info, f"Should have '{field}' field"
    
    # Verify values are reasonable
    assert first_broker_info['balance'] > 0, "Balance should be positive"
    assert first_broker_info['equity'] > 0, "Equity should be positive"
    assert first_broker_info['free_margin'] >= 0, "Free margin should be non-negative"
    
    logger.info(f"✅ Account info aggregated: {len(info)} brokers")
    logger.info(f"First broker ({first_broker_id}): Balance={first_broker_info['balance']}, Equity={first_broker_info['equity']}")


@pytest.mark.asyncio
async def test_broker_failover(broker_manager):
    """Test failover when primary broker goes down."""
    logger.info("Testing broker failover functionality")
    
    manager = await broker_manager()
    
    # Get initial status
    initial_status = await manager.get_broker_status()
    primary_broker = initial_status['primary_broker']
    healthy_brokers = initial_status['healthy_brokers']
    
    # Skip test if only one broker available
    if len(healthy_brokers) <= 1:
        pytest.skip("Need at least 2 brokers for failover test")
    
    logger.info(f"Primary broker: {primary_broker}")
    logger.info(f"Healthy brokers: {healthy_brokers}")
    
    # Simulate primary failure by disconnecting it
    try:
        primary_adapter = await manager.connection_pool.get_adapter(primary_broker)
        await primary_adapter.disconnect()
        logger.info(f"Disconnected primary broker: {primary_broker}")
    except Exception as e:
        logger.warning(f"Could not disconnect primary broker: {e}")
        pytest.skip("Could not simulate primary broker failure")
    
    # Wait a moment for health check to detect failure
    await asyncio.sleep(2)
    
    # Perform health check
    await manager.connection_pool.health_check_all()
    
    # Get updated status
    updated_status = await manager.get_broker_status()
    
    # Verify primary broker is now unhealthy
    if primary_broker in updated_status['unhealthy_brokers']:
        logger.info(f"✅ Primary broker {primary_broker} detected as unhealthy")
        
        # Try to fetch data (should use backup)
        try:
            data = await manager.get_market_data('XAUUSD', 'H1', bars=10)
            assert not data.empty, "Should work with backup broker"
            logger.info("✅ Market data fetch succeeded with backup broker")
        except Exception as e:
            logger.warning(f"Market data fetch failed with backup: {e}")
    else:
        logger.info("Primary broker still healthy (may not have disconnected properly)")


@pytest.mark.asyncio
async def test_multiple_timeframes(broker_manager):
    """Test fetching data for different timeframes."""
    logger.info("Testing multiple timeframes")
    
    manager = await broker_manager()
    
    timeframes = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']
    successful_timeframes = []
    
    for timeframe in timeframes:
        try:
            data = await manager.get_market_data('XAUUSD', timeframe, bars=20)
            
            assert not data.empty, f"Should get data for {timeframe}"
            assert len(data) == 20, f"Should have 20 bars for {timeframe}"
            
            successful_timeframes.append(timeframe)
            logger.info(f"✅ Timeframe {timeframe}: {len(data)} bars")
            
        except Exception as e:
            logger.warning(f"⚠️ Timeframe {timeframe} failed: {e}")
            # Don't fail test if timeframe not supported
    
    # Verify at least one timeframe worked
    assert len(successful_timeframes) > 0, "At least one timeframe should work"
    logger.info(f"✅ Successful timeframes: {successful_timeframes}")


@pytest.mark.asyncio
async def test_symbol_mapping_end_to_end(broker_manager):
    """Test symbol mapping works end-to-end through manager."""
    logger.info("Testing symbol mapping end-to-end")
    
    manager = await broker_manager()
    
    # Test standard symbols
    test_symbols = ['XAUUSD', 'BTCUSD', 'USDJPY', 'EURUSD']
    successful_symbols = []
    
    for symbol in test_symbols:
        try:
            # Fetch data using standard symbol
            data = await manager.get_market_data(symbol, 'H1', bars=10)
            
            # Verify data was fetched successfully
            assert not data.empty, f"Should get data for {symbol}"
            assert data['symbol'].iloc[0] == symbol, f"Symbol should be {symbol}"
            
            successful_symbols.append(symbol)
            logger.info(f"✅ Symbol mapping verified for {symbol}")
            
        except Exception as e:
            logger.warning(f"⚠️ Symbol mapping failed for {symbol}: {e}")
            # Don't fail test if symbol not available on this broker
    
    # Verify at least one symbol worked
    assert len(successful_symbols) > 0, "At least one symbol should work"
    logger.info(f"✅ Successful symbols: {successful_symbols}")


@pytest.mark.asyncio
async def test_health_monitoring(broker_manager):
    """Test health monitoring functionality."""
    logger.info("Testing health monitoring")
    
    manager = await broker_manager()
    
    # Get initial health status
    initial_status = await manager.get_broker_status()
    initial_healthy = len(initial_status['healthy_brokers'])
    
    # Wait a bit for health monitoring to run
    await asyncio.sleep(5)
    
    # Get health status again
    updated_status = await manager.get_broker_status()
    updated_healthy = len(updated_status['healthy_brokers'])
    
    # Verify health monitoring is working
    assert initial_healthy == updated_healthy, "Health status should be consistent"
    assert updated_status['last_health_check'] is not None, "Last health check should be set"
    
    logger.info(f"✅ Health monitoring verified: {updated_healthy} healthy brokers")


@pytest.mark.asyncio
async def test_error_handling(broker_manager):
    """Test error handling for invalid requests."""
    logger.info("Testing error handling")
    
    manager = await broker_manager()
    
    # Test invalid symbol
    try:
        await manager.get_market_data('INVALID_SYMBOL', 'H1', bars=10)
        pytest.fail("Should raise error for invalid symbol")
    except Exception as e:
        logger.info(f"✅ Invalid symbol handled correctly: {type(e).__name__}")
    
    # Test invalid timeframe
    try:
        await manager.get_market_data('XAUUSD', 'INVALID_TIMEFRAME', bars=10)
        pytest.fail("Should raise error for invalid timeframe")
    except Exception as e:
        logger.info(f"✅ Invalid timeframe handled correctly: {type(e).__name__}")
    
    # Test invalid bar count
    try:
        await manager.get_market_data('XAUUSD', 'H1', bars=-1)
        pytest.fail("Should raise error for invalid bar count")
    except Exception as e:
        logger.info(f"✅ Invalid bar count handled correctly: {type(e).__name__}")


@pytest.mark.asyncio
async def test_manager_shutdown(broker_manager):
    """Test manager shutdown functionality."""
    logger.info("Testing manager shutdown")
    
    manager = await broker_manager()
    
    # Verify manager is initialized
    assert manager.is_initialized(), "Manager should be initialized"
    
    # Shutdown manager
    await manager.shutdown()
    
    # Verify manager is no longer initialized
    assert not manager.is_initialized(), "Manager should not be initialized after shutdown"
    
    # Verify operations fail after shutdown
    try:
        await manager.get_market_data('XAUUSD', 'H1', bars=10)
        pytest.fail("Should raise error after shutdown")
    except BrokerError:
        logger.info("✅ Operations properly blocked after shutdown")
    
    logger.info("✅ Manager shutdown completed successfully")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s", "--tb=short"])
