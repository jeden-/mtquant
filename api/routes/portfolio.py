"""
Portfolio API Routes - FastAPI Endpoints.

Provides REST API endpoints for portfolio management and monitoring.

Author: MTQuant Development Team
Date: October 15, 2025
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import Optional
from datetime import datetime, timedelta

from api.models.portfolio_schemas import (
    PortfolioResponse,
    PortfolioSummarySchema,
    PositionSchema,
    RiskMetricsSchema,
    PerformanceMetricsSchema,
    EquityCurveResponse,
    EquityCurvePoint,
    ClosePositionRequest,
    ClosePositionResponse,
)
from mtquant.data.storage.postgresql_client import PostgreSQLClient
from mtquant.risk_management.portfolio_risk_manager import PortfolioRiskManager
from mtquant.utils.logger import get_logger


logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

# Global instances (should be injected via dependency injection in production)
db_client: Optional[PostgreSQLClient] = None
risk_manager: Optional[PortfolioRiskManager] = None


def get_db_client() -> PostgreSQLClient:
    """Dependency to get database client."""
    if db_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database client not initialized"
        )
    return db_client


def get_risk_manager() -> PortfolioRiskManager:
    """Dependency to get risk manager."""
    if risk_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk manager not initialized"
        )
    return risk_manager


@router.get("/", response_model=PortfolioResponse)
async def get_portfolio(
    db: PostgreSQLClient = Depends(get_db_client),
    rm: PortfolioRiskManager = Depends(get_risk_manager)
):
    """
    Get complete portfolio information.
    
    Returns:
        Portfolio summary, positions, and risk metrics
    """
    try:
        # Get open positions
        positions_data = await db.get_open_positions()
        
        # Calculate portfolio summary
        total_equity = 100000.0  # Placeholder - should come from broker
        cash_balance = 95000.0
        total_positions_value = sum(p.get('unrealized_pnl', 0) for p in positions_data)
        total_unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions_data)
        total_realized_pnl = 4000.0  # Placeholder - should query from trades
        
        summary = PortfolioSummarySchema(
            total_equity=total_equity,
            cash_balance=cash_balance,
            total_positions_value=total_positions_value,
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=total_realized_pnl,
            total_pnl=total_unrealized_pnl + total_realized_pnl,
            daily_pnl=350.25,  # Placeholder
            daily_pnl_pct=0.0033,
            num_open_positions=len(positions_data),
            num_agents_active=len(set(p['agent_id'] for p in positions_data)),
            margin_used=2500.0,  # Placeholder
            margin_available=97500.0,
            margin_level=4210.02
        )
        
        # Convert positions
        positions = [
            PositionSchema(
                position_id=p['position_id'],
                agent_id=p['agent_id'],
                symbol=p['symbol'],
                side=p['side'],
                quantity=float(p['quantity']),
                entry_price=float(p['entry_price']),
                current_price=float(p['current_price']) if p['current_price'] else None,
                stop_loss=float(p['stop_loss']) if p['stop_loss'] else None,
                take_profit=float(p['take_profit']) if p['take_profit'] else None,
                unrealized_pnl=float(p['unrealized_pnl']) if p['unrealized_pnl'] else None,
                unrealized_pnl_pct=None,  # Calculate if needed
                realized_pnl=float(p['realized_pnl']),
                opened_at=p['opened_at'],
                updated_at=p['updated_at'],
                status=p['status']
            )
            for p in positions_data
        ]
        
        # Get risk metrics
        risk_metrics = RiskMetricsSchema(
            var_1d=1250.50,  # Placeholder - should calculate
            var_5d=2800.75,
            max_drawdown=0.08,
            current_drawdown=0.02,
            sharpe_ratio=2.35,
            sortino_ratio=3.12,
            correlation_risk=0.65,
            sector_concentration={
                "forex": 0.60,
                "commodities": 0.25,
                "equity": 0.15
            }
        )
        
        return PortfolioResponse(
            summary=summary,
            positions=positions,
            risk_metrics=risk_metrics,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Failed to get portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get portfolio: {str(e)}"
        )


@router.get("/summary", response_model=PortfolioSummarySchema)
async def get_portfolio_summary(
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    Get portfolio summary only (lightweight).
    
    Returns:
        Portfolio summary metrics
    """
    try:
        positions_data = await db.get_open_positions()
        
        total_equity = 100000.0
        cash_balance = 95000.0
        total_positions_value = sum(p.get('unrealized_pnl', 0) for p in positions_data)
        total_unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions_data)
        total_realized_pnl = 4000.0
        
        return PortfolioSummarySchema(
            total_equity=total_equity,
            cash_balance=cash_balance,
            total_positions_value=total_positions_value,
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=total_realized_pnl,
            total_pnl=total_unrealized_pnl + total_realized_pnl,
            daily_pnl=350.25,
            daily_pnl_pct=0.0033,
            num_open_positions=len(positions_data),
            num_agents_active=len(set(p['agent_id'] for p in positions_data)),
            margin_used=2500.0,
            margin_available=97500.0,
            margin_level=4210.02
        )
    
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get portfolio summary: {str(e)}"
        )


@router.get("/positions", response_model=list[PositionSchema])
async def get_positions(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    Get all open positions.
    
    Args:
        agent_id: Optional filter by agent ID
        
    Returns:
        List of open positions
    """
    try:
        positions_data = await db.get_open_positions(agent_id=agent_id)
        
        return [
            PositionSchema(
                position_id=p['position_id'],
                agent_id=p['agent_id'],
                symbol=p['symbol'],
                side=p['side'],
                quantity=float(p['quantity']),
                entry_price=float(p['entry_price']),
                current_price=float(p['current_price']) if p['current_price'] else None,
                stop_loss=float(p['stop_loss']) if p['stop_loss'] else None,
                take_profit=float(p['take_profit']) if p['take_profit'] else None,
                unrealized_pnl=float(p['unrealized_pnl']) if p['unrealized_pnl'] else None,
                unrealized_pnl_pct=None,
                realized_pnl=float(p['realized_pnl']),
                opened_at=p['opened_at'],
                updated_at=p['updated_at'],
                status=p['status']
            )
            for p in positions_data
        ]
    
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get positions: {str(e)}"
        )


@router.post("/positions/{position_id}/close", response_model=ClosePositionResponse)
async def close_position(
    position_id: int,
    request: ClosePositionRequest,
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    Close a position.
    
    Args:
        position_id: Position ID to close
        request: Close request with optional reason
        
    Returns:
        Close position response
    """
    try:
        # TODO: Implement actual position closing logic with broker
        # For now, just return success
        
        logger.info(f"Closing position {position_id}: {request.reason}")
        
        return ClosePositionResponse(
            success=True,
            message="Position closed successfully",
            position_id=position_id,
            realized_pnl=125.50,  # Placeholder
            closed_at=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Failed to close position: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to close position: {str(e)}"
        )


@router.get("/risk", response_model=RiskMetricsSchema)
async def get_risk_metrics(
    rm: PortfolioRiskManager = Depends(get_risk_manager)
):
    """
    Get portfolio risk metrics.
    
    Returns:
        Risk metrics including VaR, drawdown, correlations
    """
    try:
        # TODO: Calculate actual risk metrics from PortfolioRiskManager
        
        return RiskMetricsSchema(
            var_1d=1250.50,
            var_5d=2800.75,
            max_drawdown=0.08,
            current_drawdown=0.02,
            sharpe_ratio=2.35,
            sortino_ratio=3.12,
            correlation_risk=0.65,
            sector_concentration={
                "forex": 0.60,
                "commodities": 0.25,
                "equity": 0.15
            }
        )
    
    except Exception as e:
        logger.error(f"Failed to get risk metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get risk metrics: {str(e)}"
        )


@router.get("/performance", response_model=PerformanceMetricsSchema)
async def get_performance_metrics(
    period_days: int = Query(30, ge=1, le=365, description="Period in days"),
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    Get portfolio performance metrics for a period.
    
    Args:
        period_days: Number of days to analyze
        
    Returns:
        Performance metrics
    """
    try:
        period_start = datetime.now() - timedelta(days=period_days)
        period_end = datetime.now()
        
        # TODO: Calculate actual performance metrics from database
        
        return PerformanceMetricsSchema(
            period_start=period_start,
            period_end=period_end,
            total_return=5250.50,
            total_return_pct=0.0525,
            sharpe_ratio=2.35,
            sortino_ratio=3.12,
            max_drawdown=0.08,
            win_rate=0.633,
            profit_factor=2.15,
            total_trades=150,
            winning_trades=95,
            losing_trades=55,
            avg_win=85.50,
            avg_loss=-45.25,
            largest_win=350.75,
            largest_loss=-180.50
        )
    
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.get("/equity-curve", response_model=EquityCurveResponse)
async def get_equity_curve(
    period_days: int = Query(30, ge=1, le=365, description="Period in days"),
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    Get equity curve data for charting.
    
    Args:
        period_days: Number of days of data
        
    Returns:
        Equity curve data points
    """
    try:
        period_start = datetime.now() - timedelta(days=period_days)
        period_end = datetime.now()
        
        # TODO: Fetch actual equity curve data from database
        # For now, return placeholder data
        
        data_points = [
            EquityCurvePoint(
                timestamp=datetime.now(),
                equity=105250.50,
                cash=95000.00,
                positions_value=10250.50,
                unrealized_pnl=1250.50,
                realized_pnl=4000.00
            )
        ]
        
        return EquityCurveResponse(
            data_points=data_points,
            period_start=period_start,
            period_end=period_end
        )
    
    except Exception as e:
        logger.error(f"Failed to get equity curve: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get equity curve: {str(e)}"
        )


# Initialize function (should be called on app startup)
def initialize_portfolio_routes(
    db: PostgreSQLClient,
    rm: PortfolioRiskManager
):
    """
    Initialize portfolio routes with client instances.
    
    Args:
        db: Database client instance
        rm: Risk manager instance
    """
    global db_client, risk_manager
    db_client = db
    risk_manager = rm
    logger.info("Portfolio routes initialized")

