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
"""

__version__ = "0.1.0"

# Adapter classes will be imported here when implemented
__all__ = []
