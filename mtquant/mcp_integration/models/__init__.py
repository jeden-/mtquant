"""
MCP Integration Models

Data models for orders, positions, and market data used throughout
the MCP integration layer. All models use Pydantic for validation
and serialization.

Models:
- Order: Trading orders with validation and serialization
- Position: Current positions with P&L tracking
- MarketData: OHLCV data with metadata
- Trade: Executed trades with audit information

All models include proper type hints and validation rules for
production safety.
"""

__version__ = "0.1.0"

# Model classes will be imported here when implemented
__all__ = []
