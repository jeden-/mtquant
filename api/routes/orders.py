"""
Orders API Routes - FastAPI Endpoints.

Provides REST API endpoints for order management.

Author: MTQuant Development Team
Date: October 15, 2025
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import Optional

from api.models.order_schemas import (
    OrderCreateRequest,
    OrderResponse,
    OrderListResponse,
    OrderCancelResponse,
    TradeResponse,
    TradeListResponse,
)
from mtquant.data.storage.postgresql_client import PostgreSQLClient
from mtquant.mcp_integration.managers.broker_manager import BrokerManager
from mtquant.risk_management.pre_trade_checker import PreTradeChecker
from mtquant.utils.logger import get_logger
from mtquant.utils.exceptions import RiskViolationError, OrderExecutionError


logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/orders", tags=["orders"])

# Global instances
db_client: Optional[PostgreSQLClient] = None
broker_manager: Optional[BrokerManager] = None
pre_trade_checker: Optional[PreTradeChecker] = None


def get_db_client() -> PostgreSQLClient:
    """Dependency to get database client."""
    if db_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database client not initialized"
        )
    return db_client


def get_broker_manager() -> BrokerManager:
    """Dependency to get broker manager."""
    if broker_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Broker manager not initialized"
        )
    return broker_manager


def get_pre_trade_checker() -> PreTradeChecker:
    """Dependency to get pre-trade checker."""
    if pre_trade_checker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pre-trade checker not initialized"
        )
    return pre_trade_checker


@router.post("/", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(
    request: OrderCreateRequest,
    db: PostgreSQLClient = Depends(get_db_client),
    bm: BrokerManager = Depends(get_broker_manager),
    ptc: PreTradeChecker = Depends(get_pre_trade_checker)
):
    """
    Create and execute a new order.
    
    Args:
        request: Order creation request
        db: Database client (injected)
        bm: Broker manager (injected)
        ptc: Pre-trade checker (injected)
        
    Returns:
        Created order information
        
    Raises:
        HTTPException: If order validation fails or execution fails
    """
    try:
        # Insert order into database
        order_id = await db.insert_order(
            agent_id=request.agent_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            status="pending",
            created_by="api"
        )
        
        logger.info(f"Created order {order_id} for {request.symbol}")
        
        # TODO: Implement actual order execution with broker
        # For now, just update status to filled
        await db.update_order_status(
            order_id=order_id,
            status="filled",
            filled_quantity=request.quantity,
            avg_fill_price=request.price or 0.0,
            commission=2.50
        )
        
        # Get updated order
        orders = await db.get_orders(limit=1)
        if orders:
            order = orders[0]
            return OrderResponse(
                order_id=order['order_id'],
                agent_id=order['agent_id'],
                symbol=order['symbol'],
                side=order['side'],
                order_type=order['order_type'],
                quantity=float(order['quantity']),
                price=float(order['price']) if order['price'] else None,
                stop_loss=float(order['stop_loss']) if order['stop_loss'] else None,
                take_profit=float(order['take_profit']) if order['take_profit'] else None,
                status=order['status'],
                broker_order_id=order['broker_order_id'],
                filled_quantity=float(order['filled_quantity']),
                avg_fill_price=float(order['avg_fill_price']) if order['avg_fill_price'] else None,
                commission=float(order['commission']) if order['commission'] else None,
                created_at=order['created_at'],
                updated_at=order['updated_at'],
                filled_at=order['filled_at'],
                created_by=order['created_by']
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Order created but not found"
        )
    
    except RiskViolationError as e:
        logger.warning(f"Order rejected by risk checks: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Order rejected: {str(e)}"
        )
    except OrderExecutionError as e:
        logger.error(f"Order execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Order execution failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to create order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create order: {str(e)}"
        )


@router.get("/", response_model=OrderListResponse)
async def list_orders(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of orders"),
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    List orders with optional filters.
    
    Args:
        agent_id: Filter by agent ID (optional)
        symbol: Filter by symbol (optional)
        status: Filter by status (optional)
        limit: Maximum number of orders
        db: Database client (injected)
        
    Returns:
        List of orders
    """
    try:
        orders_data = await db.get_orders(
            agent_id=agent_id,
            symbol=symbol,
            status=status,
            limit=limit
        )
        
        orders = [
            OrderResponse(
                order_id=order['order_id'],
                agent_id=order['agent_id'],
                symbol=order['symbol'],
                side=order['side'],
                order_type=order['order_type'],
                quantity=float(order['quantity']),
                price=float(order['price']) if order['price'] else None,
                stop_loss=float(order['stop_loss']) if order['stop_loss'] else None,
                take_profit=float(order['take_profit']) if order['take_profit'] else None,
                status=order['status'],
                broker_order_id=order['broker_order_id'],
                filled_quantity=float(order['filled_quantity']),
                avg_fill_price=float(order['avg_fill_price']) if order['avg_fill_price'] else None,
                commission=float(order['commission']) if order['commission'] else None,
                created_at=order['created_at'],
                updated_at=order['updated_at'],
                filled_at=order['filled_at'],
                created_by=order['created_by']
            )
            for order in orders_data
        ]
        
        return OrderListResponse(
            orders=orders,
            total=len(orders)
        )
    
    except Exception as e:
        logger.error(f"Failed to list orders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list orders: {str(e)}"
        )


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: int,
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    Get order details.
    
    Args:
        order_id: Order ID
        db: Database client (injected)
        
    Returns:
        Order information
        
    Raises:
        HTTPException: If order not found
    """
    try:
        orders = await db.get_orders(limit=1000)  # TODO: Add get_order_by_id method
        
        for order in orders:
            if order['order_id'] == order_id:
                return OrderResponse(
                    order_id=order['order_id'],
                    agent_id=order['agent_id'],
                    symbol=order['symbol'],
                    side=order['side'],
                    order_type=order['order_type'],
                    quantity=float(order['quantity']),
                    price=float(order['price']) if order['price'] else None,
                    stop_loss=float(order['stop_loss']) if order['stop_loss'] else None,
                    take_profit=float(order['take_profit']) if order['take_profit'] else None,
                    status=order['status'],
                    broker_order_id=order['broker_order_id'],
                    filled_quantity=float(order['filled_quantity']),
                    avg_fill_price=float(order['avg_fill_price']) if order['avg_fill_price'] else None,
                    commission=float(order['commission']) if order['commission'] else None,
                    created_at=order['created_at'],
                    updated_at=order['updated_at'],
                    filled_at=order['filled_at'],
                    created_by=order['created_by']
                )
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get order: {str(e)}"
        )


@router.delete("/{order_id}", response_model=OrderCancelResponse)
async def cancel_order(
    order_id: int,
    db: PostgreSQLClient = Depends(get_db_client),
    bm: BrokerManager = Depends(get_broker_manager)
):
    """
    Cancel an order.
    
    Args:
        order_id: Order ID to cancel
        db: Database client (injected)
        bm: Broker manager (injected)
        
    Returns:
        Cancellation response
        
    Raises:
        HTTPException: If order not found or cancellation fails
    """
    try:
        # TODO: Implement actual order cancellation with broker
        
        # Update order status
        await db.update_order_status(
            order_id=order_id,
            status="cancelled"
        )
        
        logger.info(f"Cancelled order {order_id}")
        
        return OrderCancelResponse(
            success=True,
            message="Order cancelled successfully",
            order_id=order_id
        )
    
    except Exception as e:
        logger.error(f"Failed to cancel order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel order: {str(e)}"
        )


@router.get("/trades/", response_model=TradeListResponse)
async def list_trades(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of trades"),
    db: PostgreSQLClient = Depends(get_db_client)
):
    """
    List trades (executed orders).
    
    Args:
        agent_id: Filter by agent ID (optional)
        symbol: Filter by symbol (optional)
        limit: Maximum number of trades
        db: Database client (injected)
        
    Returns:
        List of trades
    """
    try:
        # TODO: Implement get_trades method in PostgreSQLClient
        # For now, return empty list
        
        return TradeListResponse(
            trades=[],
            total=0
        )
    
    except Exception as e:
        logger.error(f"Failed to list trades: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list trades: {str(e)}"
        )


# Initialize function (should be called on app startup)
def initialize_order_routes(
    db: PostgreSQLClient,
    bm: BrokerManager,
    ptc: PreTradeChecker
):
    """
    Initialize order routes with client instances.
    
    Args:
        db: Database client instance
        bm: Broker manager instance
        ptc: Pre-trade checker instance
    """
    global db_client, broker_manager, pre_trade_checker
    db_client = db
    broker_manager = bm
    pre_trade_checker = ptc
    logger.info("Order routes initialized")



