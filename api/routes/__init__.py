"""
API Routes - FastAPI Endpoints.

Provides REST API endpoints for the MTQuant trading system.
"""

from api.routes.agents import router as agents_router, initialize_agent_routes
from api.routes.portfolio import router as portfolio_router, initialize_portfolio_routes
from api.routes.orders import router as orders_router, initialize_order_routes
from api.routes.websocket import router as websocket_router, initialize_websocket_routes
from api.routes.metrics import router as metrics_router, initialize_metrics_routes

__version__ = "0.1.0"

__all__ = [
    "agents_router",
    "portfolio_router",
    "orders_router",
    "websocket_router",
    "metrics_router",
    "initialize_agent_routes",
    "initialize_portfolio_routes",
    "initialize_order_routes",
    "initialize_websocket_routes",
    "initialize_metrics_routes",
]
