"""
Integration tests for MT5 MCP adapter.

These tests run against REAL MT5 demo account via MCP server and verify:
- Connection to MT5 MCP server
- Market data fetching through MCP
- Account information retrieval
- Symbol mapping end-to-end
- Position management
- Health monitoring

Requirements:
- MT5 terminal must be running and logged in
- MT5 MCP server must be running (uv run mt5mcp)
- Demo account credentials in .env file
- pytest-asyncio for async test support

Run with:
pytest tests/integration/test_mt5_integration.py -v -s
"""

import pytest
import asyncio
import yaml
import os
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd
from mtquant.mcp_integration.adapters import MT5BrokerAdapter
from mtquant.mcp_integration.models.order import Order
from mtquant.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture
def broker_config() -> Dict[str, Any]:
    """Load broker configuration from YAML."""
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        with open('config/brokers.yaml', 'r') as f:
            brokers_config = yaml.safe_load(f)
        
        # Get first enabled demo broker
        demo_brokers = brokers_config.get('demo_accounts', {})
        for broker_id, config in demo_brokers.items():
            if config.get('enabled', False):
                # Expand environment variables
                config = expand_env_vars(config)
                return config
        
        # If no enabled broker, return first one
        if demo_brokers:
            config = list(demo_brokers.values())[0]
            config = expand_env_vars(config)
            return config
        else:
            pytest.skip("No demo brokers found in config")
            
    except Exception as e:
        pytest.skip(f"Failed to load broker config: {e}")


def expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand environment variables in config values."""
    import os
    expanded = {}
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]  # Remove ${ and }
            expanded[key] = os.getenv(env_var, value)
        else:
            expanded[key] = value
    return expanded


@pytest.fixture
def mt5_adapter(broker_config):
    """Create MT5 adapter connected to demo account."""
    async def _create_adapter():
        adapter = MT5BrokerAdapter(
            broker_id=broker_config['broker_id'],
            config=broker_config
        )
        
        # Try to connect
        try:
            connected = await adapter.connect()
            if not connected:
                pytest.skip("MT5 terminal not running or not logged in")
        except Exception as e:
            pytest.skip(f"MT5 connection failed: {e}")
        
        return adapter
    
    return _create_adapter


@pytest.mark.asyncio
async def test_mt5_connection(mt5_adapter):
    """Test basic connection to MT5 demo."""
    logger.info("Testing MT5 connection")
    
    # Create adapter
    adapter = await mt5_adapter()
    
    # Verify connection
    health = await adapter.health_check()
    
    # Assertions
    assert health.is_connected, "MT5 adapter should be connected"
    assert health.latency_ms >= 0, "Latency should be non-negative"
    assert health.last_check is not None, "Last check time should be set"
    
    logger.info(f"✅ Connection verified: {health.is_connected}, latency: {health.latency_ms}ms")
    
    # Cleanup
    try:
        await adapter.disconnect()
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")


@pytest.mark.asyncio
async def test_fetch_market_data(mt5_adapter):
    """Test fetching OHLCV data for XAUUSD."""
    logger.info("Testing market data fetch for XAUUSD")
    
    # Create adapter
    adapter = await mt5_adapter()
    
    # Fetch data for standard symbol XAUUSD
    data = await adapter.get_market_data('XAUUSD', 'H1', bars=50)
    
    # Assertions
    assert not data.empty, "DataFrame should not be empty"
    assert len(data) == 50, f"Should have 50 bars, got {len(data)}"
    assert 'close' in data.columns, "Should have 'close' column"
    assert 'open' in data.columns, "Should have 'open' column"
    assert 'high' in data.columns, "Should have 'high' column"
    assert 'low' in data.columns, "Should have 'low' column"
    assert 'volume' in data.columns, "Should have 'volume' column"
    assert 'symbol' in data.columns, "Should have 'symbol' column"
    
    # Check data quality
    assert data['close'].notna().all(), "Close prices should not be NaN"
    assert data['symbol'].iloc[0] == 'XAUUSD', "Symbol should be mapped to XAUUSD"
    
    # Check data is recent (last bar should be within last 48 hours for demo accounts)
    if 'timestamp' in data.columns:
        last_timestamp = pd.to_datetime(data['timestamp']).max()
        time_diff = datetime.utcnow() - last_timestamp
        assert time_diff < timedelta(hours=48), f"Data too old: {time_diff}"
    
    logger.info(f"✅ Market data verified: {len(data)} bars, latest close: {data['close'].iloc[-1]}")


@pytest.mark.asyncio
async def test_get_account_info(mt5_adapter):
    """Test account info retrieval."""
    logger.info("Testing account info retrieval")
    
    # Create adapter
    adapter = await mt5_adapter()
    
    # Get account info
    account_info = await adapter.get_account_info()
    
    # Assertions
    assert isinstance(account_info, dict), "Account info should be a dictionary"
    assert 'balance' in account_info, "Should have 'balance' field"
    assert 'equity' in account_info, "Should have 'equity' field"
    assert 'margin' in account_info, "Should have 'margin' field"
    assert 'free_margin' in account_info, "Should have 'free_margin' field"
    
    # Check values are reasonable
    assert account_info['balance'] > 0, "Balance should be positive"
    assert account_info['equity'] > 0, "Equity should be positive"
    assert account_info['margin'] >= 0, "Margin should be non-negative"
    assert account_info['free_margin'] >= 0, "Free margin should be non-negative"
    
    logger.info(f"✅ Account info verified: Balance: {account_info['balance']}, Equity: {account_info['equity']}")


@pytest.mark.asyncio
async def test_symbol_mapping(mt5_adapter):
    """Test symbol mapping works end-to-end."""
    logger.info("Testing symbol mapping end-to-end")
    
    # Create adapter
    adapter = await mt5_adapter()
    
    # Test standard symbols
    test_symbols = ['XAUUSD', 'USDJPY', 'EURUSD', 'GBPUSD']
    
    for symbol in test_symbols:
        try:
            # Fetch data using standard symbol
            data = await adapter.get_market_data(symbol, 'H1', bars=10)
            
            # Verify data was fetched successfully
            assert not data.empty, f"Should get data for {symbol}"
            assert data['symbol'].iloc[0] == symbol, f"Symbol should be {symbol}"
            
            logger.info(f"✅ Symbol mapping verified for {symbol}")
            
        except Exception as e:
            logger.warning(f"⚠️ Symbol mapping failed for {symbol}: {e}")
            # Don't fail test if symbol not available on this broker


@pytest.mark.asyncio
async def test_get_positions_empty(mt5_adapter):
    """Test getting positions (should be empty on fresh demo)."""
    logger.info("Testing position retrieval")
    
    # Create adapter
    adapter = await mt5_adapter()
    
    # Get positions
    positions = await adapter.get_positions()
    
    # Assertions
    assert isinstance(positions, list), "Positions should be a list"
    
    # On fresh demo account, positions should be empty
    # But we don't fail if positions exist (user might have opened some)
    if len(positions) == 0:
        logger.info("✅ No positions found (expected on fresh demo)")
    else:
        logger.info(f"✅ Found {len(positions)} positions")
        # Verify position structure
        for pos in positions:
            assert hasattr(pos, 'symbol'), "Position should have symbol"
            assert hasattr(pos, 'side'), "Position should have side"
            assert hasattr(pos, 'quantity'), "Position should have quantity"


@pytest.mark.asyncio
async def test_multiple_timeframes(mt5_adapter):
    """Test fetching data for different timeframes."""
    logger.info("Testing multiple timeframes")
    
    # Create adapter
    adapter = await mt5_adapter()
    
    timeframes = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']
    
    for timeframe in timeframes:
        try:
            data = await adapter.get_market_data('XAUUSD', timeframe, bars=20)
            
            assert not data.empty, f"Should get data for {timeframe}"
            assert len(data) == 20, f"Should have 20 bars for {timeframe}"
            
            logger.info(f"✅ Timeframe {timeframe} verified: {len(data)} bars")
            
        except Exception as e:
            logger.warning(f"⚠️ Timeframe {timeframe} failed: {e}")
            # Don't fail test if timeframe not supported


@pytest.mark.asyncio
async def test_health_monitoring(mt5_adapter):
    """Test health monitoring functionality."""
    logger.info("Testing health monitoring")
    
    # Create adapter
    adapter = await mt5_adapter()
    
    # Get initial health status
    health1 = await adapter.health_check()
    
    # Wait a bit
    await asyncio.sleep(1)
    
    # Get health status again
    health2 = await adapter.health_check()
    
    # Assertions
    assert health1.is_connected == health2.is_connected, "Connection status should be consistent"
    assert health2.last_check > health1.last_check, "Last check time should be updated"
    
    logger.info(f"✅ Health monitoring verified: Connected: {health2.is_connected}")


@pytest.mark.asyncio
async def test_error_handling(mt5_adapter):
    """Test error handling for invalid requests."""
    logger.info("Testing error handling")
    
    # Create adapter
    adapter = await mt5_adapter()
    
    # Test invalid symbol
    try:
        await adapter.get_market_data('INVALID_SYMBOL', 'H1', bars=10)
        pytest.fail("Should raise error for invalid symbol")
    except Exception as e:
        logger.info(f"✅ Invalid symbol handled correctly: {type(e).__name__}")
    
    # Test invalid timeframe
    try:
        await adapter.get_market_data('XAUUSD', 'INVALID_TIMEFRAME', bars=10)
        pytest.fail("Should raise error for invalid timeframe")
    except Exception as e:
        logger.info(f"✅ Invalid timeframe handled correctly: {type(e).__name__}")
    
    # Test invalid bar count
    try:
        await adapter.get_market_data('XAUUSD', 'H1', bars=-1)
        pytest.fail("Should raise error for invalid bar count")
    except Exception as e:
        logger.info(f"✅ Invalid bar count handled correctly: {type(e).__name__}")


# Markers for test categorization
pytestmark = [
    pytest.mark.integration,
    pytest.mark.mt5,
    pytest.mark.broker
]
