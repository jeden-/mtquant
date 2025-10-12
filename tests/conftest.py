"""
Pytest configuration and shared fixtures for MTQuant tests.

This file contains common fixtures and configuration that can be used
across all test modules in the MTQuant project.

Fixtures:
- broker_config: Load broker configuration from YAML
- mt5_adapter: Create connected MT5 adapter
- mock_broker: Mock broker for unit tests
- sample_order: Sample order for testing
- sample_position: Sample position for testing

Configuration:
- Async test support
- Logging setup
- Test markers
"""

import pytest
import asyncio
import yaml
import os
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from mtquant.mcp_integration.adapters import MT5BrokerAdapter
from mtquant.mcp_integration.models.order import Order
from mtquant.mcp_integration.models.position import Position
from mtquant.utils.logger import setup_logger, get_logger

# Setup logging for tests
setup_logger(level="INFO")
logger = get_logger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def broker_config() -> Dict[str, Any]:
    """Load broker configuration from YAML."""
    try:
        with open('config/brokers.yaml', 'r') as f:
            brokers_config = yaml.safe_load(f)
        
        # Get first enabled demo broker
        demo_brokers = brokers_config.get('demo_accounts', {})
        for broker_id, config in demo_brokers.items():
            if config.get('enabled', False):
                return config
        
        # If no enabled broker, return first one
        if demo_brokers:
            return list(demo_brokers.values())[0]
        else:
            pytest.skip("No demo brokers found in config")
            
    except Exception as e:
        pytest.skip(f"Failed to load broker config: {e}")


@pytest.fixture
async def mt5_adapter(broker_config) -> AsyncGenerator[MT5BrokerAdapter, None]:
    """Create MT5 adapter connected to demo account."""
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
    
    yield adapter
    
    # Cleanup
    try:
        await adapter.disconnect()
    except Exception as e:
        logger.warning(f"Cleanup disconnect failed: {e}")


@pytest.fixture
def mock_broker():
    """Create mock broker for unit tests."""
    broker = AsyncMock()
    
    # Mock methods
    broker.connect.return_value = True
    broker.disconnect.return_value = None
    broker.health_check.return_value = MagicMock(
        is_connected=True,
        latency_ms=10.0,
        last_check=MagicMock(),
        error=None
    )
    broker.get_account_info.return_value = {
        'balance': 10000.0,
        'equity': 10000.0,
        'margin': 0.0,
        'free_margin': 10000.0
    }
    broker.get_positions.return_value = []
    broker.get_market_data.return_value = MagicMock()  # Mock DataFrame
    
    return broker


@pytest.fixture
def sample_order() -> Order:
    """Create sample order for testing."""
    return Order(
        agent_id="test_agent",
        symbol="XAUUSD",
        side="buy",
        order_type="market",
        quantity=0.1,
        signal=0.8,
        created_at=MagicMock(),
        status="pending"
    )


@pytest.fixture
def sample_position() -> Position:
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
        opened_at=MagicMock(),
        broker_id="ic_markets"
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create sample OHLCV data
    dates = pd.date_range(
        start=datetime.utcnow() - timedelta(hours=100),
        end=datetime.utcnow(),
        freq='H'
    )
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': 2050.0 + pd.Series(range(len(dates))) * 0.1,
        'high': 2055.0 + pd.Series(range(len(dates))) * 0.1,
        'low': 2045.0 + pd.Series(range(len(dates))) * 0.1,
        'close': 2052.0 + pd.Series(range(len(dates))) * 0.1,
        'volume': [1000] * len(dates),
        'symbol': ['XAUUSD'] * len(dates)
    })
    
    return data


# Test markers
pytestmark = [
    pytest.mark.asyncio
]


# Skip tests if MT5 not available
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring external services"
    )
    config.addinivalue_line(
        "markers", "mt5: Tests requiring MT5 terminal"
    )
    config.addinivalue_line(
        "markers", "broker: Tests requiring broker connection"
    )
    config.addinivalue_line(
        "markers", "unit: Unit tests (no external dependencies)"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add integration marker to tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add mt5 marker to MT5-related tests
        if "mt5" in item.name.lower() or "mt5" in str(item.fspath):
            item.add_marker(pytest.mark.mt5)
