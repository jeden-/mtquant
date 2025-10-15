"""
API Models - Pydantic Schemas.

Defines request/response schemas for all API endpoints.
"""

from api.models.agent_schemas import (
    AgentMetricsSchema,
    AgentConfigSchema,
    AgentCreateRequest,
    AgentUpdateRequest,
    AgentStateTransitionRequest,
    AgentResponse,
    AgentListResponse,
    AgentHealthResponse,
    AgentActionResponse,
    AgentPerformanceResponse,
)

from api.models.portfolio_schemas import (
    PositionSchema,
    PortfolioSummarySchema,
    RiskMetricsSchema,
    PortfolioResponse,
    PerformanceMetricsSchema,
    EquityCurvePoint,
    EquityCurveResponse,
    ClosePositionRequest,
    ClosePositionResponse,
)

from api.models.order_schemas import (
    OrderCreateRequest,
    OrderResponse,
    OrderListResponse,
    OrderCancelResponse,
    TradeResponse,
    TradeListResponse,
)

__version__ = "0.1.0"

__all__ = [
    # Agent schemas
    "AgentMetricsSchema",
    "AgentConfigSchema",
    "AgentCreateRequest",
    "AgentUpdateRequest",
    "AgentStateTransitionRequest",
    "AgentResponse",
    "AgentListResponse",
    "AgentHealthResponse",
    "AgentActionResponse",
    "AgentPerformanceResponse",
    # Portfolio schemas
    "PositionSchema",
    "PortfolioSummarySchema",
    "RiskMetricsSchema",
    "PortfolioResponse",
    "PerformanceMetricsSchema",
    "EquityCurvePoint",
    "EquityCurveResponse",
    "ClosePositionRequest",
    "ClosePositionResponse",
    # Order schemas
    "OrderCreateRequest",
    "OrderResponse",
    "OrderListResponse",
    "OrderCancelResponse",
    "TradeResponse",
    "TradeListResponse",
]
