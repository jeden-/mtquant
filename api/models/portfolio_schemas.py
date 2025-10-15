"""
Portfolio API Schemas - Pydantic Models.

Defines request/response schemas for portfolio-related API endpoints.

Author: MTQuant Development Team
Date: October 15, 2025
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class PositionSchema(BaseModel):
    """Position schema."""
    position_id: int
    agent_id: str
    symbol: str
    side: str = Field(..., description="long or short")
    quantity: float
    entry_price: float
    current_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    realized_pnl: float = 0.0
    opened_at: datetime
    updated_at: datetime
    status: str = "open"
    
    class Config:
        json_schema_extra = {
            "example": {
                "position_id": 1,
                "agent_id": "forex_eur_001",
                "symbol": "EURUSD",
                "side": "long",
                "quantity": 0.1,
                "entry_price": 1.0850,
                "current_price": 1.0875,
                "stop_loss": 1.0830,
                "take_profit": 1.0900,
                "unrealized_pnl": 25.0,
                "unrealized_pnl_pct": 0.0023,
                "realized_pnl": 0.0,
                "opened_at": "2025-10-15T10:00:00",
                "updated_at": "2025-10-15T12:00:00",
                "status": "open"
            }
        }


class PortfolioSummarySchema(BaseModel):
    """Portfolio summary schema."""
    total_equity: float
    cash_balance: float
    total_positions_value: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    daily_pnl: float
    daily_pnl_pct: float
    num_open_positions: int
    num_agents_active: int
    margin_used: float
    margin_available: float
    margin_level: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_equity": 105250.50,
                "cash_balance": 95000.00,
                "total_positions_value": 10250.50,
                "total_unrealized_pnl": 1250.50,
                "total_realized_pnl": 4000.00,
                "total_pnl": 5250.50,
                "daily_pnl": 350.25,
                "daily_pnl_pct": 0.0033,
                "num_open_positions": 5,
                "num_agents_active": 3,
                "margin_used": 2500.00,
                "margin_available": 97500.00,
                "margin_level": 4210.02
            }
        }


class RiskMetricsSchema(BaseModel):
    """Risk metrics schema."""
    var_1d: float = Field(..., description="1-day Value at Risk")
    var_5d: float = Field(..., description="5-day Value at Risk")
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    correlation_risk: float = Field(..., description="Max pairwise correlation")
    sector_concentration: Dict[str, float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "var_1d": 1250.50,
                "var_5d": 2800.75,
                "max_drawdown": 0.08,
                "current_drawdown": 0.02,
                "sharpe_ratio": 2.35,
                "sortino_ratio": 3.12,
                "correlation_risk": 0.65,
                "sector_concentration": {
                    "forex": 0.60,
                    "commodities": 0.25,
                    "equity": 0.15
                }
            }
        }


class PortfolioResponse(BaseModel):
    """Portfolio response schema."""
    summary: PortfolioSummarySchema
    positions: List[PositionSchema]
    risk_metrics: RiskMetricsSchema
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "summary": {
                    "total_equity": 105250.50,
                    "cash_balance": 95000.00,
                    "total_positions_value": 10250.50,
                    "total_unrealized_pnl": 1250.50,
                    "total_realized_pnl": 4000.00,
                    "total_pnl": 5250.50,
                    "daily_pnl": 350.25,
                    "daily_pnl_pct": 0.0033,
                    "num_open_positions": 5,
                    "num_agents_active": 3,
                    "margin_used": 2500.00,
                    "margin_available": 97500.00,
                    "margin_level": 4210.02
                },
                "positions": [],
                "risk_metrics": {
                    "var_1d": 1250.50,
                    "var_5d": 2800.75,
                    "max_drawdown": 0.08,
                    "current_drawdown": 0.02,
                    "sharpe_ratio": 2.35,
                    "sortino_ratio": 3.12,
                    "correlation_risk": 0.65,
                    "sector_concentration": {
                        "forex": 0.60,
                        "commodities": 0.25,
                        "equity": 0.15
                    }
                },
                "timestamp": "2025-10-15T12:00:00"
            }
        }


class PerformanceMetricsSchema(BaseModel):
    """Performance metrics schema."""
    period_start: datetime
    period_end: datetime
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "period_start": "2025-10-01T00:00:00",
                "period_end": "2025-10-15T23:59:59",
                "total_return": 5250.50,
                "total_return_pct": 0.0525,
                "sharpe_ratio": 2.35,
                "sortino_ratio": 3.12,
                "max_drawdown": 0.08,
                "win_rate": 0.633,
                "profit_factor": 2.15,
                "total_trades": 150,
                "winning_trades": 95,
                "losing_trades": 55,
                "avg_win": 85.50,
                "avg_loss": -45.25,
                "largest_win": 350.75,
                "largest_loss": -180.50
            }
        }


class EquityCurvePoint(BaseModel):
    """Equity curve data point."""
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-10-15T12:00:00",
                "equity": 105250.50,
                "cash": 95000.00,
                "positions_value": 10250.50,
                "unrealized_pnl": 1250.50,
                "realized_pnl": 4000.00
            }
        }


class EquityCurveResponse(BaseModel):
    """Equity curve response schema."""
    data_points: List[EquityCurvePoint]
    period_start: datetime
    period_end: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "data_points": [
                    {
                        "timestamp": "2025-10-15T12:00:00",
                        "equity": 105250.50,
                        "cash": 95000.00,
                        "positions_value": 10250.50,
                        "unrealized_pnl": 1250.50,
                        "realized_pnl": 4000.00
                    }
                ],
                "period_start": "2025-10-01T00:00:00",
                "period_end": "2025-10-15T23:59:59"
            }
        }


class ClosePositionRequest(BaseModel):
    """Request schema for closing a position."""
    reason: Optional[str] = Field(None, description="Reason for closing")
    
    class Config:
        json_schema_extra = {
            "example": {
                "reason": "Take profit target reached"
            }
        }


class ClosePositionResponse(BaseModel):
    """Response schema for closing a position."""
    success: bool
    message: str
    position_id: int
    realized_pnl: float
    closed_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Position closed successfully",
                "position_id": 1,
                "realized_pnl": 125.50,
                "closed_at": "2025-10-15T12:00:00"
            }
        }

