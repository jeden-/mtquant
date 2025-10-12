"""
MCP Integration Models

Data models for orders, positions, and market data used throughout
the MCP integration layer. All models use dataclasses with validation
and serialization.

Models:
- Order: Trading orders with validation and serialization
- Position: Current positions with P&L tracking
- MarketData: OHLCV data with metadata (to be implemented)
- Trade: Executed trades with audit information (to be implemented)

All models include proper type hints and validation rules for
production safety.
"""

__version__ = "0.1.0"

from .order import Order, OrderSide, OrderType, OrderStatus
from .position import Position

__all__ = [
    "Order",
    "OrderSide", 
    "OrderType",
    "OrderStatus",
    "Position",
]
