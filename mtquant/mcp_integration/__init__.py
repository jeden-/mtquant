"""
MCP Integration Module

Model Context Protocol integration for broker connectivity.
Provides unified interface to MetaTrader 4/5 brokers through MCP servers.

Components:
- clients/: MCP client implementations
- adapters/: Broker-specific adapters (MT4, MT5, OANDA, etc.)
- managers/: Connection pooling, symbol mapping, failover
- models/: Order, position, and market data models

Architecture:
- Adapter Pattern for broker abstraction
- Connection pooling for performance
- Automatic failover to backup brokers
- Symbol mapping for cross-broker compatibility
"""

__version__ = "0.1.0"

# Manager classes will be imported here when implemented
# from .managers.broker_manager import BrokerManager
# from .managers.symbol_mapper import SymbolMapper

__all__ = []
