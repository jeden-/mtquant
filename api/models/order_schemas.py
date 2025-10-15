"""
Order API Schemas - Pydantic Models.

Defines request/response schemas for order-related API endpoints.

Author: MTQuant Development Team
Date: October 15, 2025
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator


class OrderCreateRequest(BaseModel):
    """Request schema for creating an order."""
    agent_id: str = Field(..., description="Agent identifier")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="buy or sell")
    order_type: str = Field(..., description="market, limit, or stop")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: Optional[float] = Field(None, gt=0, description="Limit price (for limit orders)")
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[float] = Field(None, gt=0, description="Take profit price")
    
    @validator('side')
    def validate_side(cls, v):
        if v not in ['buy', 'sell']:
            raise ValueError("side must be 'buy' or 'sell'")
        return v
    
    @validator('order_type')
    def validate_order_type(cls, v):
        if v not in ['market', 'limit', 'stop']:
            raise ValueError("order_type must be 'market', 'limit', or 'stop'")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "forex_eur_001",
                "symbol": "EURUSD",
                "side": "buy",
                "order_type": "market",
                "quantity": 0.1,
                "price": None,
                "stop_loss": 1.0830,
                "take_profit": 1.0900
            }
        }


class OrderResponse(BaseModel):
    """Response schema for order information."""
    order_id: int
    agent_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str
    broker_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    commission: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    filled_at: Optional[datetime] = None
    created_by: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "order_id": 1,
                "agent_id": "forex_eur_001",
                "symbol": "EURUSD",
                "side": "buy",
                "order_type": "market",
                "quantity": 0.1,
                "price": None,
                "stop_loss": 1.0830,
                "take_profit": 1.0900,
                "status": "filled",
                "broker_order_id": "BR123456",
                "filled_quantity": 0.1,
                "avg_fill_price": 1.0850,
                "commission": 2.50,
                "created_at": "2025-10-15T10:00:00",
                "updated_at": "2025-10-15T10:00:05",
                "filled_at": "2025-10-15T10:00:05",
                "created_by": "system"
            }
        }


class OrderListResponse(BaseModel):
    """Response schema for list of orders."""
    orders: List[OrderResponse]
    total: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "orders": [
                    {
                        "order_id": 1,
                        "agent_id": "forex_eur_001",
                        "symbol": "EURUSD",
                        "side": "buy",
                        "order_type": "market",
                        "quantity": 0.1,
                        "price": None,
                        "stop_loss": 1.0830,
                        "take_profit": 1.0900,
                        "status": "filled",
                        "broker_order_id": "BR123456",
                        "filled_quantity": 0.1,
                        "avg_fill_price": 1.0850,
                        "commission": 2.50,
                        "created_at": "2025-10-15T10:00:00",
                        "updated_at": "2025-10-15T10:00:05",
                        "filled_at": "2025-10-15T10:00:05",
                        "created_by": "system"
                    }
                ],
                "total": 1
            }
        }


class OrderCancelResponse(BaseModel):
    """Response schema for order cancellation."""
    success: bool
    message: str
    order_id: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Order cancelled successfully",
                "order_id": 1
            }
        }


class TradeResponse(BaseModel):
    """Response schema for trade information."""
    trade_id: int
    order_id: int
    agent_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: Optional[float] = None
    pnl: Optional[float] = None
    broker_trade_id: Optional[str] = None
    executed_at: datetime
    metadata: Optional[dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "trade_id": 1,
                "order_id": 1,
                "agent_id": "forex_eur_001",
                "symbol": "EURUSD",
                "side": "buy",
                "quantity": 0.1,
                "price": 1.0850,
                "commission": 2.50,
                "pnl": 25.50,
                "broker_trade_id": "TR123456",
                "executed_at": "2025-10-15T10:00:05",
                "metadata": {"slippage": 0.0001}
            }
        }


class TradeListResponse(BaseModel):
    """Response schema for list of trades."""
    trades: List[TradeResponse]
    total: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "trades": [
                    {
                        "trade_id": 1,
                        "order_id": 1,
                        "agent_id": "forex_eur_001",
                        "symbol": "EURUSD",
                        "side": "buy",
                        "quantity": 0.1,
                        "price": 1.0850,
                        "commission": 2.50,
                        "pnl": 25.50,
                        "broker_trade_id": "TR123456",
                        "executed_at": "2025-10-15T10:00:05",
                        "metadata": {"slippage": 0.0001}
                    }
                ],
                "total": 1
            }
        }


