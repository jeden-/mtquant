"""
Agent API Schemas - Pydantic Models.

Defines request/response schemas for agent-related API endpoints.

Author: MTQuant Development Team
Date: October 15, 2025
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class AgentMetricsSchema(BaseModel):
    """Agent performance metrics schema."""
    total_trades: int = Field(0, description="Total number of trades")
    winning_trades: int = Field(0, description="Number of winning trades")
    losing_trades: int = Field(0, description="Number of losing trades")
    total_pnl: float = Field(0.0, description="Total profit/loss")
    sharpe_ratio: float = Field(0.0, description="Sharpe ratio")
    max_drawdown: float = Field(0.0, description="Maximum drawdown")
    win_rate: float = Field(0.0, ge=0.0, le=1.0, description="Win rate (0-1)")
    last_updated: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_trades": 150,
                "winning_trades": 95,
                "losing_trades": 55,
                "total_pnl": 5250.50,
                "sharpe_ratio": 2.35,
                "max_drawdown": 0.08,
                "win_rate": 0.633,
                "last_updated": "2025-10-15T12:00:00"
            }
        }


class AgentConfigSchema(BaseModel):
    """Agent configuration schema."""
    learning_rate: Optional[float] = Field(None, gt=0, description="Learning rate")
    batch_size: Optional[int] = Field(None, gt=0, description="Batch size")
    gamma: Optional[float] = Field(None, ge=0, le=1, description="Discount factor")
    buffer_size: Optional[int] = Field(None, gt=0, description="Replay buffer size")
    update_frequency: Optional[int] = Field(None, gt=0, description="Update frequency")
    risk_tolerance: Optional[float] = Field(None, ge=0, le=1, description="Risk tolerance")
    max_position_size: Optional[float] = Field(None, gt=0, description="Max position size")
    stop_loss_pct: Optional[float] = Field(None, gt=0, description="Stop loss percentage")
    take_profit_pct: Optional[float] = Field(None, gt=0, description="Take profit percentage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "learning_rate": 0.0003,
                "batch_size": 64,
                "gamma": 0.99,
                "buffer_size": 100000,
                "update_frequency": 4,
                "risk_tolerance": 0.5,
                "max_position_size": 0.1,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04
            }
        }


class AgentCreateRequest(BaseModel):
    """Request schema for creating a new agent."""
    agent_id: str = Field(..., min_length=1, max_length=50, description="Unique agent identifier")
    agent_type: str = Field(..., description="Agent type (specialist, meta_controller, ppo)")
    symbol: Optional[str] = Field(None, max_length=20, description="Trading symbol")
    config: Optional[AgentConfigSchema] = Field(None, description="Agent configuration")
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        allowed_types = ['specialist', 'meta_controller', 'ppo', 'forex_specialist', 
                        'commodities_specialist', 'equity_specialist']
        if v not in allowed_types:
            raise ValueError(f"agent_type must be one of {allowed_types}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "forex_eur_001",
                "agent_type": "forex_specialist",
                "symbol": "EURUSD",
                "config": {
                    "learning_rate": 0.0003,
                    "batch_size": 64,
                    "risk_tolerance": 0.5
                }
            }
        }


class AgentUpdateRequest(BaseModel):
    """Request schema for updating an agent."""
    config: Optional[AgentConfigSchema] = Field(None, description="Updated configuration")
    
    class Config:
        json_schema_extra = {
            "example": {
                "config": {
                    "learning_rate": 0.0001,
                    "risk_tolerance": 0.3
                }
            }
        }


class AgentStateTransitionRequest(BaseModel):
    """Request schema for agent state transition."""
    new_state: str = Field(..., description="Target state")
    error_message: Optional[str] = Field(None, description="Error message if transitioning to ERROR")
    
    @validator('new_state')
    def validate_state(cls, v):
        allowed_states = ['initialized', 'training', 'paper_trading', 'live', 'paused', 'error', 'stopped']
        if v not in allowed_states:
            raise ValueError(f"new_state must be one of {allowed_states}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "new_state": "training",
                "error_message": None
            }
        }


class AgentResponse(BaseModel):
    """Response schema for agent information."""
    agent_id: str
    agent_type: str
    symbol: Optional[str] = None
    state: str
    created_at: datetime
    last_state_change: datetime
    metrics: AgentMetricsSchema
    config: Dict[str, Any]
    error_message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "forex_eur_001",
                "agent_type": "forex_specialist",
                "symbol": "EURUSD",
                "state": "live",
                "created_at": "2025-10-01T10:00:00",
                "last_state_change": "2025-10-15T12:00:00",
                "metrics": {
                    "total_trades": 150,
                    "winning_trades": 95,
                    "losing_trades": 55,
                    "total_pnl": 5250.50,
                    "sharpe_ratio": 2.35,
                    "max_drawdown": 0.08,
                    "win_rate": 0.633,
                    "last_updated": "2025-10-15T12:00:00"
                },
                "config": {
                    "learning_rate": 0.0003,
                    "batch_size": 64
                },
                "error_message": None
            }
        }


class AgentListResponse(BaseModel):
    """Response schema for list of agents."""
    agents: List[AgentResponse]
    total: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "agents": [
                    {
                        "agent_id": "forex_eur_001",
                        "agent_type": "forex_specialist",
                        "symbol": "EURUSD",
                        "state": "live",
                        "created_at": "2025-10-01T10:00:00",
                        "last_state_change": "2025-10-15T12:00:00",
                        "metrics": {
                            "total_trades": 150,
                            "winning_trades": 95,
                            "losing_trades": 55,
                            "total_pnl": 5250.50,
                            "sharpe_ratio": 2.35,
                            "max_drawdown": 0.08,
                            "win_rate": 0.633,
                            "last_updated": "2025-10-15T12:00:00"
                        },
                        "config": {},
                        "error_message": None
                    }
                ],
                "total": 1
            }
        }


class AgentHealthResponse(BaseModel):
    """Response schema for agent health status."""
    agent_id: str
    state: str
    is_healthy: bool
    last_heartbeat: Optional[datetime] = None
    uptime_seconds: float
    error_message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "forex_eur_001",
                "state": "live",
                "is_healthy": True,
                "last_heartbeat": "2025-10-15T12:00:00",
                "uptime_seconds": 1209600.0,
                "error_message": None
            }
        }


class AgentActionResponse(BaseModel):
    """Response schema for agent actions (start, pause, stop)."""
    success: bool
    message: str
    agent_id: str
    new_state: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Agent paused successfully",
                "agent_id": "forex_eur_001",
                "new_state": "paused"
            }
        }


class AgentPerformanceResponse(BaseModel):
    """Response schema for agent performance metrics."""
    agent_id: str
    symbol: Optional[str] = None
    period_start: datetime
    period_end: datetime
    metrics: AgentMetricsSchema
    daily_pnl: List[Dict[str, Any]]
    trade_history: List[Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "forex_eur_001",
                "symbol": "EURUSD",
                "period_start": "2025-10-01T00:00:00",
                "period_end": "2025-10-15T23:59:59",
                "metrics": {
                    "total_trades": 150,
                    "winning_trades": 95,
                    "losing_trades": 55,
                    "total_pnl": 5250.50,
                    "sharpe_ratio": 2.35,
                    "max_drawdown": 0.08,
                    "win_rate": 0.633,
                    "last_updated": "2025-10-15T12:00:00"
                },
                "daily_pnl": [
                    {"date": "2025-10-15", "pnl": 350.25}
                ],
                "trade_history": [
                    {
                        "trade_id": 1,
                        "timestamp": "2025-10-15T10:30:00",
                        "side": "buy",
                        "quantity": 0.1,
                        "price": 1.0850,
                        "pnl": 25.50
                    }
                ]
            }
        }

