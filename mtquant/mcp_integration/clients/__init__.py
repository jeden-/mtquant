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

CRITICAL: All clients use MCP protocol, NOT direct broker APIs.
"""

__version__ = "0.1.0"

from .mt5_mcp_client import MT5MCPClient, HealthStatus as MT5HealthStatus
from .mt4_mcp_client import MT4MCPClient, HealthStatus as MT4HealthStatus

__all__ = [
    "MT5MCPClient",
    "MT4MCPClient", 
    "MT5HealthStatus",
    "MT4HealthStatus",
]
