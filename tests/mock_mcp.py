"""
Mock MCP module for testing when MCP is not available.
This allows tests to run without requiring MCP installation.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio


class StdioServerParameters:
    """Mock StdioServerParameters."""
    def __init__(self, command: str, args: List[str] = None):
        self.command = command
        self.args = args or []


class ClientSession:
    """Mock ClientSession."""
    def __init__(self):
        self.connected = False
    
    async def connect(self, *args, **kwargs):
        self.connected = True
    
    async def disconnect(self):
        self.connected = False
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        return {"content": [{"text": "mock response"}]}
    
    async def list_tools(self):
        return {"tools": []}


class HealthStatus(Enum):
    """Mock HealthStatus enum."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# Mock stdio_client function
async def stdio_client(*args, **kwargs):
    """Mock stdio_client function."""
    return ClientSession()


# Mock the mcp module and submodules
import sys
from unittest.mock import MagicMock

# Create mock mcp module
mock_mcp = MagicMock()
mock_mcp.ClientSession = ClientSession
mock_mcp.StdioServerParameters = StdioServerParameters
mock_mcp.HealthStatus = HealthStatus

# Create mock mcp.client module
mock_mcp_client = MagicMock()
mock_mcp_client.stdio = MagicMock()
mock_mcp_client.stdio.stdio_client = stdio_client

# Create mock mcp.client.stdio module
mock_mcp_client_stdio = MagicMock()
mock_mcp_client_stdio.stdio_client = stdio_client

# Add to sys.modules so imports work
sys.modules['mcp'] = mock_mcp
sys.modules['mcp.client'] = mock_mcp_client
sys.modules['mcp.client.stdio'] = mock_mcp_client_stdio
