"""
Broker Adapters

Broker-specific implementations using the Adapter Pattern.
Each adapter provides a unified interface to different brokers while handling
broker-specific protocols, symbol formats, and API differences.

Supported Brokers:
- MetaTrader 4/5 (via MCP servers)
- OANDA (REST API)
- Interactive Brokers (TWS API)
- Alpaca (REST API)

All adapters implement the same interface for seamless broker switching.

CRITICAL: All adapters use MCP clients, NOT direct broker APIs.
"""

__version__ = "0.1.0"

from .base_adapter import BrokerAdapter, HealthStatus
from .mt5_adapter import MT5BrokerAdapter
from .mt4_adapter import MT4BrokerAdapter

__all__ = [
    "BrokerAdapter",
    "HealthStatus", 
    "MT5BrokerAdapter",
    "MT4BrokerAdapter",
]
