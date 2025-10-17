"""
Metrics API Routes - System and Performance Metrics.

Provides REST API endpoints for system metrics and performance analytics.

Author: MTQuant Development Team
Date: October 15, 2025
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel

from mtquant.data.storage.redis_client import RedisClient
from mtquant.data.storage.postgresql_client import PostgreSQLClient
from mtquant.utils.logger import get_logger


logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/metrics", tags=["metrics"])

# Global instances
redis_client: Optional[RedisClient] = None
db_client: Optional[PostgreSQLClient] = None


# Response schemas

class SystemMetricsResponse(BaseModel):
    """System metrics response."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    active_agents: int
    open_positions: int
    orders_today: int
    websocket_connections: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-10-15T12:00:00",
                "cpu_usage_percent": 45.5,
                "memory_usage_percent": 62.3,
                "disk_usage_percent": 38.7,
                "active_agents": 8,
                "open_positions": 12,
                "orders_today": 45,
                "websocket_connections": 3
            }
        }


class AgentMetricsResponse(BaseModel):
    """Agent performance metrics response."""
    agent_id: str
    symbol: Optional[str]
    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration_minutes: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "forex_eur_001",
                "symbol": "EURUSD",
                "period_start": "2025-10-01T00:00:00",
                "period_end": "2025-10-15T23:59:59",
                "total_trades": 150,
                "winning_trades": 95,
                "losing_trades": 55,
                "win_rate": 0.633,
                "total_pnl": 5250.50,
                "sharpe_ratio": 2.35,
                "max_drawdown": 0.08,
                "avg_trade_duration_minutes": 240.5
            }
        }


class PortfolioMetricsResponse(BaseModel):
    """Portfolio performance metrics response."""
    period_start: datetime
    period_end: datetime
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_1d: float
    daily_avg_pnl: float
    best_day: float
    worst_day: float
    total_trades: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "period_start": "2025-10-01T00:00:00",
                "period_end": "2025-10-15T23:59:59",
                "total_return": 5250.50,
                "total_return_pct": 5.25,
                "sharpe_ratio": 2.35,
                "sortino_ratio": 3.12,
                "max_drawdown": 0.08,
                "var_1d": 1250.50,
                "daily_avg_pnl": 350.03,
                "best_day": 850.75,
                "worst_day": -420.30,
                "total_trades": 450
            }
        }


def get_redis_client() -> RedisClient:
    """Dependency to get Redis client."""
    if redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis client not initialized"
        )
    return redis_client


def get_db_client() -> PostgreSQLClient:
    """Dependency to get database client."""
    if db_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database client not initialized"
        )
    return db_client


@router.get("/system", response_model=SystemMetricsResponse)
async def get_system_metrics(
    redis: RedisClient = Depends(get_redis_client),
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    Get current system metrics.
    
    Returns:
        System metrics including CPU, memory, active agents, etc.
    """
    try:
        # Get metrics from Redis (real-time)
        active_agents = await redis.get_metric('active_agents') or 0
        websocket_connections = await redis.get_metric('websocket_connections') or 0
        
        # Get metrics from database
        open_positions = await db.get_open_positions()
        
        # TODO: Get actual system metrics (CPU, memory, disk)
        # For now, return placeholder data
        
        return SystemMetricsResponse(
            timestamp=datetime.now(),
            cpu_usage_percent=45.5,
            memory_usage_percent=62.3,
            disk_usage_percent=38.7,
            active_agents=active_agents,
            open_positions=len(open_positions),
            orders_today=45,  # Placeholder
            websocket_connections=websocket_connections
        )
    
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system metrics: {str(e)}"
        )


@router.get("/agents/{agent_id}", response_model=AgentMetricsResponse)
async def get_agent_metrics(
    agent_id: str,
    period_days: int = Query(30, ge=1, le=365, description="Period in days"),
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    Get performance metrics for a specific agent.
    
    Args:
        agent_id: Agent identifier
        period_days: Number of days to analyze
        
    Returns:
        Agent performance metrics
    """
    try:
        period_start = datetime.now() - timedelta(days=period_days)
        period_end = datetime.now()
        
        # TODO: Calculate actual metrics from database
        # For now, return placeholder data
        
        return AgentMetricsResponse(
            agent_id=agent_id,
            symbol="EURUSD",
            period_start=period_start,
            period_end=period_end,
            total_trades=150,
            winning_trades=95,
            losing_trades=55,
            win_rate=0.633,
            total_pnl=5250.50,
            sharpe_ratio=2.35,
            max_drawdown=0.08,
            avg_trade_duration_minutes=240.5
        )
    
    except Exception as e:
        logger.error(f"Failed to get agent metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent metrics: {str(e)}"
        )


@router.get("/agents", response_model=List[AgentMetricsResponse])
async def get_all_agents_metrics(
    period_days: int = Query(30, ge=1, le=365, description="Period in days"),
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    Get performance metrics for all agents.
    
    Args:
        period_days: Number of days to analyze
        
    Returns:
        List of agent performance metrics
    """
    try:
        period_start = datetime.now() - timedelta(days=period_days)
        period_end = datetime.now()
        
        # TODO: Get all agents from database and calculate metrics
        # For now, return placeholder data for a few agents
        
        agents = [
            AgentMetricsResponse(
                agent_id=f"agent_{i}",
                symbol="EURUSD",
                period_start=period_start,
                period_end=period_end,
                total_trades=150,
                winning_trades=95,
                losing_trades=55,
                win_rate=0.633,
                total_pnl=5250.50,
                sharpe_ratio=2.35,
                max_drawdown=0.08,
                avg_trade_duration_minutes=240.5
            )
            for i in range(3)
        ]
        
        return agents
    
    except Exception as e:
        logger.error(f"Failed to get all agents metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get all agents metrics: {str(e)}"
        )


@router.get("/portfolio", response_model=PortfolioMetricsResponse)
async def get_portfolio_metrics(
    period_days: int = Query(30, ge=1, le=365, description="Period in days"),
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    Get portfolio performance metrics.
    
    Args:
        period_days: Number of days to analyze
        
    Returns:
        Portfolio performance metrics
    """
    try:
        period_start = datetime.now() - timedelta(days=period_days)
        period_end = datetime.now()
        
        # TODO: Calculate actual portfolio metrics from database
        # For now, return placeholder data
        
        return PortfolioMetricsResponse(
            period_start=period_start,
            period_end=period_end,
            total_return=5250.50,
            total_return_pct=5.25,
            sharpe_ratio=2.35,
            sortino_ratio=3.12,
            max_drawdown=0.08,
            var_1d=1250.50,
            daily_avg_pnl=350.03,
            best_day=850.75,
            worst_day=-420.30,
            total_trades=450
        )
    
    except Exception as e:
        logger.error(f"Failed to get portfolio metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get portfolio metrics: {str(e)}"
        )


@router.get("/realtime/{metric_name}")
async def get_realtime_metric(
    metric_name: str,
    agent_id: Optional[str] = Query(None, description="Agent ID (optional)"),
    redis: RedisClient = Depends(get_redis_client)
):
    """
    Get real-time metric value.
    
    Args:
        metric_name: Metric name (e.g., 'active_trades', 'total_pnl')
        agent_id: Optional agent ID filter
        
    Returns:
        Current metric value
    """
    try:
        value = await redis.get_metric(metric_name, agent_id=agent_id)
        
        return {
            'metric_name': metric_name,
            'agent_id': agent_id,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get realtime metric: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get realtime metric: {str(e)}"
        )


@router.post("/realtime/{metric_name}")
async def update_realtime_metric(
    metric_name: str,
    value: float,
    agent_id: Optional[str] = Query(None, description="Agent ID (optional)"),
    redis: RedisClient = Depends(get_redis_client)
):
    """
    Update real-time metric value.
    
    Args:
        metric_name: Metric name
        value: New metric value
        agent_id: Optional agent ID
        
    Returns:
        Success message
    """
    try:
        await redis.set_metric(metric_name, value, agent_id=agent_id)
        
        return {
            'success': True,
            'message': f"Metric '{metric_name}' updated",
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to update realtime metric: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update realtime metric: {str(e)}"
        )


@router.get("/health")
async def get_health_status(
    redis: RedisClient = Depends(get_redis_client),
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    Get health status of all system components.
    
    Returns:
        Health status for database, Redis, agents, etc.
    """
    try:
        # Check Redis health
        redis_health = await redis.health_check()
        
        # Check database health
        db_health = await db.health_check()
        
        # Overall health
        all_healthy = (
            redis_health.get('connected', False) and
            db_health.get('connected', False)
        )
        
        return {
            'status': 'healthy' if all_healthy else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'redis': redis_health,
                'database': db_health
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        return {
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }


# Initialize function (should be called on app startup)
def initialize_metrics_routes(
    redis: RedisClient,
    db: PostgreSQLClient
):
    """
    Initialize metrics routes with client instances.
    
    Args:
        redis: Redis client instance
        db: Database client instance
    """
    global redis_client, db_client
    redis_client = redis
    db_client = db
    logger.info("Metrics routes initialized")



