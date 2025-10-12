"""
MCP Integration Managers

High-level management components for broker integration:
- BrokerManager: Central coordination of all broker connections
- ConnectionPool: Efficient connection pooling and reuse
- SymbolMapper: Cross-broker symbol mapping and normalization

These managers provide the business logic layer above the low-level
clients and adapters, handling complex scenarios like failover,
load balancing, and symbol translation.
"""

__version__ = "0.1.0"

from .symbol_mapper import SymbolMapper
from .connection_pool import ConnectionPool, ConnectionStatus, AdapterInfo, ConnectionStats
from .broker_manager import BrokerManager

__all__ = [
    "SymbolMapper",
    "ConnectionPool",
    "ConnectionStatus",
    "AdapterInfo", 
    "ConnectionStats",
    "BrokerManager",
]
