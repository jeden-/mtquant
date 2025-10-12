"""
MCP Client Implementations

Base client classes and MCP protocol implementations for broker communication.
Handles low-level MCP protocol details, message serialization, and connection management.

Key Features:
- Async/await support for non-blocking operations
- Automatic reconnection and error handling
- Message queuing and retry logic
- Connection health monitoring
- Protocol version negotiation
"""

__version__ = "0.1.0"

from .mt5_client import MT5Client, HealthStatus

__all__ = [
    "MT5Client",
    "HealthStatus",
]
