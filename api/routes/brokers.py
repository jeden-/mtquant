"""
Broker Management API Routes.

Handles broker connections, disconnections, and status monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import asyncpg

from mtquant.utils.logger import get_logger
from mtquant.mcp_integration.managers.broker_manager import BrokerManager
from mtquant.mcp_integration.clients.mt5_mcp_client import MT5MCPClient
from mtquant.mcp_integration.clients.mt4_mcp_client import MT4MCPClient

# PostgreSQL connection for persistence
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'mtquantum',
    'user': 'postgres',
    'password': 'MARiusz@!2025'
}


logger = get_logger(__name__)

# Global broker manager instance
broker_manager: Optional[BrokerManager] = None


class BrokerConnectRequest(BaseModel):
    """Broker connection request."""
    broker_type: str  # 'mt5' or 'mt4'
    account: int
    password: str
    server: str
    broker_id: Optional[str] = None


class BrokerResponse(BaseModel):
    """Broker connection response."""
    success: bool
    broker_id: str
    message: str
    account_info: Optional[Dict] = None


def initialize_broker_routes(manager: BrokerManager):
    """Initialize broker routes with broker manager instance."""
    global broker_manager
    broker_manager = manager
    logger.info("Broker routes initialized")


# Create router
router = APIRouter(prefix="/api/brokers", tags=["brokers"])


@router.post("/connect", response_model=BrokerResponse)
async def connect_broker(request: BrokerConnectRequest):
    """
    Connect to a broker (MT5 or MT4).
    
    This endpoint:
    1. Validates broker credentials
    2. Establishes connection via MCP
    3. Retrieves account information
    4. Registers broker in the system
    """
    if broker_manager is None:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")
    
    try:
        logger.info(f"Attempting to connect to {request.broker_type.upper()} broker...")
        
        # Generate broker ID if not provided
        broker_id = request.broker_id or f"{request.broker_type}_{request.account}"
        
        # Prepare MCP config
        import os
        mcp_server_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "mcp_servers", "mt5", "server"
        )
        
        config = {
            'mcp_server_path': mcp_server_path,
            'account': request.account,
            'password': request.password,
            'server': request.server
        }
        
        # Create appropriate MCP client
        if request.broker_type.lower() == 'mt5':
            mcp_client = MT5MCPClient(broker_id=broker_id, config=config)
        elif request.broker_type.lower() == 'mt4':
            mcp_client = MT4MCPClient(broker_id=broker_id, config=config)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported broker type: {request.broker_type}"
            )
        
        # Connect to MCP server (this also logs in)
        logger.info(f"Connecting to MCP server at: {mcp_server_path}")
        await mcp_client.connect()
        
        # Get account info to verify connection
        logger.info(f"Getting account info for {broker_id}...")
        account_info = await mcp_client.get_account_info()
        logger.info(f"Account info retrieved: {account_info}")
        
        # Register broker with manager
        logger.info(f"Registering broker {broker_id} with manager...")
        await broker_manager.register_broker(broker_id, mcp_client)
        logger.info(f"Broker {broker_id} registered!")
        
        # Verify registration
        registered = broker_manager.get_broker(broker_id)
        logger.info(f"Verification: broker_manager.get_broker({broker_id}) = {registered}")
        all_brokers = broker_manager.list_brokers()
        logger.info(f"All registered brokers: {all_brokers}")
        
        # Save to database for persistence
        try:
            conn = await asyncpg.connect(**DB_CONFIG)
            await conn.execute("""
                INSERT INTO broker_connections 
                (broker_id, broker_type, account, password_encrypted, server, last_connected_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (broker_id) 
                DO UPDATE SET last_connected_at = NOW(), is_active = TRUE
            """, broker_id, request.broker_type.lower(), request.account, request.password, request.server)
            await conn.close()
            logger.info(f"💾 Broker {broker_id} saved to database")
        except Exception as e:
            logger.warning(f"Failed to save broker to database: {e}")
        
        logger.info(f"✅ Successfully connected to {request.broker_type.upper()} broker {broker_id}")
        
        return BrokerResponse(
            success=True,
            broker_id=broker_id,
            message=f"Connected to {request.broker_type.upper()} broker successfully",
            account_info=account_info
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to connect broker: {e}")
        raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")


@router.post("/{broker_id}/disconnect")
async def disconnect_broker(broker_id: str):
    """Disconnect from a broker."""
    if broker_manager is None:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")
    
    try:
        # Get broker client
        client = broker_manager.get_broker(broker_id)
        if client is None:
            raise HTTPException(status_code=404, detail=f"Broker {broker_id} not found")
        
        # Disconnect
        await client.disconnect()
        
        # Unregister from manager
        broker_manager.unregister_broker(broker_id)
        
        logger.info(f"Disconnected broker {broker_id}")
        
        return {"success": True, "message": f"Disconnected from {broker_id}"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disconnect broker: {e}")
        raise HTTPException(status_code=500, detail=f"Disconnection failed: {str(e)}")


@router.get("/{broker_id}/status")
async def get_broker_status(broker_id: str):
    """Get broker connection status."""
    if broker_manager is None:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")
    
    try:
        client = broker_manager.get_broker(broker_id)
        if client is None:
            return {"connected": False, "broker_id": broker_id}
        
        # Get account info to verify connection
        account_info = await client.get_account_info()
        
        return {
            "connected": True,
            "broker_id": broker_id,
            "account_info": account_info
        }
    
    except Exception as e:
        logger.error(f"Failed to get broker status: {e}")
        return {"connected": False, "broker_id": broker_id, "error": str(e)}


@router.get("/{broker_id}/symbols")
async def get_broker_symbols(broker_id: str):
    """Get available symbols from broker."""
    try:
        client = broker_manager.get_broker(broker_id)
        if not client:
            raise HTTPException(status_code=404, detail=f"Broker {broker_id} not found")
        
        # Get symbols from MT5
        symbols = await client.get_symbols()
        
        return {
            "broker_id": broker_id,
            "symbols": symbols,
            "total": len(symbols) if symbols else 0
        }
    
    except Exception as e:
        logger.error(f"Failed to get symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{broker_id}/market_data/{symbol}")
async def get_market_data(
    broker_id: str, 
    symbol: str,
    timeframe: str = "1H",
    bars: int = 100
):
    """Get market data (OHLCV) for a symbol."""
    try:
        client = broker_manager.get_broker(broker_id)
        if not client:
            raise HTTPException(status_code=404, detail=f"Broker {broker_id} not found")
        
        # Get market data from MT5
        df = await client.get_market_data(symbol, timeframe, bars)
        
        # Convert DataFrame to list of dicts for JSON serialization
        if df is not None and not df.empty:
            # Convert timestamp to ISO format string
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            data_list = df.to_dict(orient='records')
        else:
            data_list = []
        
        return {
            "broker_id": broker_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": len(data_list),
            "data": data_list
        }
    
    except Exception as e:
        logger.error(f"Failed to get market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug")
async def debug_broker_manager():
    """Debug broker manager state."""
    return {
        "broker_manager_exists": broker_manager is not None,
        "broker_manager_type": str(type(broker_manager)) if broker_manager else None,
        "brokers_in_memory": broker_manager.list_brokers() if broker_manager else [],
        "broker_manager_id": id(broker_manager) if broker_manager else None,
        "broker_manager_has_brokers": hasattr(broker_manager, '_brokers') if broker_manager else False,
        "broker_manager_dir": dir(broker_manager) if broker_manager else [],
        "broker_manager_brokers_keys": list(broker_manager._brokers.keys()) if broker_manager and hasattr(broker_manager, '_brokers') else []
    }


@router.get("/list")
async def list_brokers():
    """List all brokers (from database)."""
    logger.info(f"🔍 DEBUG: broker_manager = {broker_manager}")
    logger.info(f"🔍 DEBUG: broker_manager type = {type(broker_manager)}")
    
    try:
        # Get brokers from database
        conn = await asyncpg.connect(**DB_CONFIG)
        rows = await conn.fetch("""
            SELECT broker_id, broker_type, account, server, last_connected_at, is_active
            FROM broker_connections
            WHERE is_active = TRUE
            ORDER BY last_connected_at DESC
        """)
        await conn.close()
        
        broker_list = []
        for row in rows:
            # Check if broker is in memory
            client = broker_manager.get_broker(row['broker_id']) if broker_manager else None
            logger.info(f"🔍 DEBUG: broker {row['broker_id']} -> client = {client}")
            
            broker_list.append({
                "broker_id": row['broker_id'],
                "connected": client is not None,
                "account_info": {
                    "login": row['account'],
                    "server": row['server']
                } if client is None else None
            })
        
        logger.info(f"Listed {len(broker_list)} brokers from database")
        return {"brokers": broker_list}
    
    except Exception as e:
        logger.error(f"Failed to list brokers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list brokers: {str(e)}")


@router.get("/{broker_id}/account")
async def get_account_info(broker_id: str):
    """Get broker account information."""
    if broker_manager is None:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")
    
    try:
        client = broker_manager.get_broker(broker_id)
        if client is None:
            raise HTTPException(status_code=404, detail=f"Broker {broker_id} not found")
        
        account_info = await client.get_account_info()
        
        return account_info
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get account info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get account info: {str(e)}")


def initialize_broker_routes(bm: BrokerManager) -> None:
    """Initialize broker routes with dependencies."""
    global broker_manager
    broker_manager = bm
    logger.info("✅ Broker routes initialized")


# Export for main.py
__all__ = ["router", "initialize_broker_routes"]

