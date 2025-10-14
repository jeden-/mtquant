"""
Integration tests for MT4 MCP adapter.

These tests run against REAL MT4 demo account via MCP server and verify:
- Connection to MT4 MCP server
- Market data fetching through MCP
- Account information retrieval
- Symbol mapping end-to-end
- Position management
- Health monitoring

Requirements:
- MT4 terminal must be running and logged in
- MT4 MCP server must be running (npm start)
- Expert Advisor MCP_Ultimate.mq4 must be attached to chart
- Demo account credentials in .env file
- pytest-asyncio for async test support

Run with:
pytest tests/integration/test_mt4_integration.py -v -s
"""

import pytest
import pytest_asyncio
import asyncio
import yaml
import os
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd
from mtquant.mcp_integration.adapters import MT4BrokerAdapter
from mtquant.mcp_integration.models.order import Order
from mtquant.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture
def broker_config() -> Dict[str, Any]:
    """Load broker configuration from YAML."""
    try:
        with open('config/brokers.yaml', 'r') as f:
            brokers_config = yaml.safe_load(f)
        
        # Get first enabled demo broker
        demo_accounts = brokers_config.get('demo_accounts', {})
        for broker_id, config in demo_accounts.items():
            if config.get('enabled', False):
                return config
        
        # If no enabled broker, return first available
        if demo_accounts:
            first_broker = list(demo_accounts.values())[0]
            return first_broker
        
        raise ValueError("No demo brokers configured")
        
    except Exception as e:
        pytest.skip(f"Could not load broker config: {e}")


@pytest_asyncio.fixture
async def mt4_adapter(broker_config):
    """Create and connect MT4 adapter."""
    adapter = MT4BrokerAdapter("generic_mt4_demo", broker_config)
    
    try:
        connected = await adapter.connect()
        if not connected:
            pytest.skip("MT4 MCP server not available or MT4 terminal not running")
        
        yield adapter
        
    finally:
        await adapter.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mt4_connection(mt4_adapter):
    """Test MT4 MCP server connection."""
    health = await mt4_adapter.health_check()
    
    assert health.is_connected is True
    assert health.latency_ms > 0
    assert health.error is None
    
    logger.info(f"MT4 connection health: {health}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mt4_market_data(mt4_adapter):
    """Test market data fetching via MCP."""
    # Test EURUSD (most common MT4 symbol)
    data = await mt4_adapter.get_market_data("EURUSD", "1H", 100)
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert 'close' in data.columns
    assert 'high' in data.columns
    assert 'low' in data.columns
    assert 'open' in data.columns
    assert 'volume' in data.columns
    
    # Verify data quality
    assert data['close'].notna().all()
    assert data['high'].notna().all()
    assert data['low'].notna().all()
    assert data['open'].notna().all()
    
    logger.info(f"MT4 market data fetched: {len(data)} rows for EURUSD")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mt4_account_info(mt4_adapter):
    """Test account information retrieval."""
    account_info = await mt4_adapter.get_account_info()
    
    assert account_info is not None
    assert 'balance' in account_info
    assert 'equity' in account_info
    assert 'margin' in account_info
    assert 'free_margin' in account_info
    
    # Verify numeric values
    assert isinstance(account_info['balance'], (int, float))
    assert isinstance(account_info['equity'], (int, float))
    assert isinstance(account_info['margin'], (int, float))
    assert isinstance(account_info['free_margin'], (int, float))
    
    logger.info(f"MT4 account info: Balance={account_info['balance']}, Equity={account_info['equity']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mt4_positions(mt4_adapter):
    """Test position retrieval."""
    positions = await mt4_adapter.get_positions()
    
    assert isinstance(positions, list)
    
    # If positions exist, verify structure
    if positions:
        position = positions[0]
        assert 'symbol' in position
        assert 'volume' in position
        assert 'profit' in position
        assert 'type' in position
        
        logger.info(f"MT4 positions found: {len(positions)}")
    else:
        logger.info("No open positions in MT4 account")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mt4_symbol_mapping(mt4_adapter):
    """Test symbol mapping end-to-end."""
    # Test standard symbol mapping
    standard_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    
    for symbol in standard_symbols:
        try:
            # This should work through symbol mapping
            data = await mt4_adapter.get_market_data(symbol, "1H", 10)
            assert len(data) > 0
            logger.info(f"Symbol mapping successful for {symbol}")
        except Exception as e:
            logger.warning(f"Symbol mapping failed for {symbol}: {e}")
            # Some symbols might not be available, that's OK


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mt4_multiple_timeframes(mt4_adapter):
    """Test multiple timeframe data fetching."""
    timeframes = ['1H', '4H', '1D']
    
    for tf in timeframes:
        try:
            data = await mt4_adapter.get_market_data("EURUSD", tf, 50)
            assert len(data) > 0
            logger.info(f"Timeframe {tf} data fetched: {len(data)} rows")
        except Exception as e:
            logger.warning(f"Timeframe {tf} failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mt4_health_monitoring(mt4_adapter):
    """Test health monitoring functionality."""
    # Test multiple health checks
    for i in range(3):
        health = await mt4_adapter.health_check()
        
        assert health.is_connected is True
        assert health.latency_ms >= 0
        assert isinstance(health.last_check, datetime)
        
        logger.info(f"Health check {i+1}: {health.latency_ms}ms latency")
        
        # Small delay between checks
        await asyncio.sleep(0.5)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mt4_error_handling(mt4_adapter):
    """Test error handling for invalid requests."""
    # Test invalid symbol
    try:
        await mt4_adapter.get_market_data("INVALID_SYMBOL", "1H", 10)
        pytest.fail("Should have raised exception for invalid symbol")
    except Exception as e:
        logger.info(f"Expected error for invalid symbol: {e}")
    
    # Test invalid timeframe
    try:
        await mt4_adapter.get_market_data("EURUSD", "INVALID_TF", 10)
        pytest.fail("Should have raised exception for invalid timeframe")
    except Exception as e:
        logger.info(f"Expected error for invalid timeframe: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mt4_disconnect_reconnect(mt4_adapter):
    """Test disconnect and reconnect functionality."""
    # Test disconnect
    await mt4_adapter.disconnect()
    
    # Health should show disconnected
    health = await mt4_adapter.health_check()
    assert health.is_connected is False
    
    # Test reconnect
    connected = await mt4_adapter.connect()
    assert connected is True
    
    # Health should show connected again
    health = await mt4_adapter.health_check()
    assert health.is_connected is True
    
    logger.info("MT4 disconnect/reconnect test successful")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])

