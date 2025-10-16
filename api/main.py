"""
Main FastAPI Application - MTQuant Trading System.

Production-grade REST API and WebSocket endpoints for managing
multi-agent RL trading system.

Author: MTQuant Development Team
Date: October 15, 2025
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from typing import Optional

from api.routes import (
    agents_router,
    portfolio_router,
    orders_router,
    websocket_router,
    metrics_router,
    brokers_router,
    initialize_agent_routes,
    initialize_portfolio_routes,
    initialize_order_routes,
    initialize_websocket_routes,
    initialize_metrics_routes,
    initialize_broker_routes
)
from mtquant.data.storage.redis_client import RedisClient, RedisConfig
from mtquant.data.storage.postgresql_client import PostgreSQLClient, PostgreSQLConfig
from mtquant.data.storage.questdb_client import QuestDBClient, QuestDBConfig
from mtquant.agents.agent_manager import AgentLifecycleManager
from mtquant.risk_management.portfolio_risk_manager import PortfolioRiskManager
from mtquant.mcp_integration.managers.broker_manager import BrokerManager
from mtquant.mcp_integration.clients.mt5_mcp_client import MT5MCPClient
from mtquant.mcp_integration.clients.mt4_mcp_client import MT4MCPClient
from mtquant.utils.logger import get_logger
import os


logger = get_logger(__name__)


async def reconnect_brokers_from_db(broker_mgr: BrokerManager, db: PostgreSQLClient):
    """
    Auto-reconnect to all active brokers from database on backend startup.
    
    This function:
    1. Fetches all active brokers from PostgreSQL
    2. Reconnects to each broker's MT4/MT5 terminal
    3. Registers them in BrokerManager
    
    Args:
        broker_mgr: BrokerManager instance
        db: PostgreSQL client instance
    """
    try:
        logger.info("🔄 Auto-reconnecting to brokers from database...")
        
        # Fetch active brokers from database using connection pool
        query = """
            SELECT broker_id, broker_type, account, password_encrypted, server
            FROM broker_connections
            WHERE is_active = TRUE
            ORDER BY last_connected_at DESC
        """
        async with db._pool.acquire() as conn:
            rows = await conn.fetch(query)
        
        if not rows:
            logger.info("No active brokers found in database")
            return
        
        logger.info(f"Found {len(rows)} active broker(s) in database")
        
        # Reconnect to each broker
        for row in rows:
            broker_id = row['broker_id']
            broker_type = row['broker_type']
            
            try:
                logger.info(f"Reconnecting to {broker_type} broker: {broker_id}")
                
                # Prepare config
                config = {
                    'mcp_server_path': os.path.join(os.getcwd(), 'mcp_servers', broker_type, 'server'),
                    'account': row['account'],
                    'password': row['password_encrypted'],  # Use password_encrypted column
                    'server': row['server']
                }
                
                # Create appropriate client
                if broker_type.lower() == 'mt5':
                    client = MT5MCPClient(broker_id=broker_id, config=config)
                elif broker_type.lower() == 'mt4':
                    client = MT4MCPClient(broker_id=broker_id, config=config)
                else:
                    logger.warning(f"Unknown broker type: {broker_type}")
                    continue
                
                # Connect
                connected = await client.connect()
                
                if connected:
                    # Register with BrokerManager
                    await broker_mgr.register_broker(broker_id, client)
                    logger.info(f"✅ Reconnected and registered: {broker_id}")
                else:
                    logger.warning(f"⚠️ Failed to reconnect: {broker_id}")
            
            except Exception as e:
                logger.error(f"❌ Failed to reconnect {broker_id}: {e}")
                continue
        
        logger.info(f"✅ Auto-reconnect complete. {len(broker_mgr.list_brokers())} broker(s) active")
    
    except Exception as e:
        logger.error(f"❌ Auto-reconnect failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Don't raise - allow app to start even if reconnection fails

# Global instances
redis_client: Optional[RedisClient] = None
db_client: Optional[PostgreSQLClient] = None
questdb_client: Optional[QuestDBClient] = None
agent_manager: Optional[AgentLifecycleManager] = None
risk_manager: Optional[PortfolioRiskManager] = None
broker_manager: Optional[BrokerManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown tasks.
    """
    # === STARTUP ===
    logger.info("Starting MTQuant API...")
    
    try:
        # Initialize Redis client (optional)
        global redis_client
        try:
            redis_config = RedisConfig(
                host="localhost",
                port=6379,
                db=0
            )
            redis_client = RedisClient(redis_config)
            await redis_client.connect()
            logger.info("✅ Redis client connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Continuing without Redis.")
            redis_client = None
        
        # Initialize PostgreSQL client (optional)
        global db_client
        try:
            db_config = PostgreSQLConfig(
                host="localhost",
                port=5432,
                database="mtquantum",
                user="postgres",
                password="MARiusz@!2025"
            )
            db_client = PostgreSQLClient(db_config)
            await db_client.connect()
            logger.info("✅ PostgreSQL client connected")
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL connection failed: {e}. Continuing without PostgreSQL.")
            db_client = None
        
        # Initialize QuestDB client (optional)
        global questdb_client
        try:
            questdb_config = QuestDBConfig(
                host="localhost",
                port=8812
            )
            questdb_client = QuestDBClient(questdb_config)
            await questdb_client.connect()
            logger.info("✅ QuestDB client connected")
        except Exception as e:
            logger.warning(f"⚠️ QuestDB connection failed: {e}. Continuing without QuestDB.")
            questdb_client = None
        
        # Initialize Agent Manager components (optional)
        global agent_manager
        agent_scheduler = None
        agent_registry = None
        try:
            from mtquant.agents.agent_manager import AgentScheduler, AgentRegistry
            agent_manager = AgentLifecycleManager()
            agent_scheduler = AgentScheduler()
            agent_registry = AgentRegistry(lifecycle_manager=agent_manager)
            logger.info("✅ Agent Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Agent Manager initialization failed: {e}. Continuing without Agent Manager.")
            agent_manager = None
        
        # Initialize Risk Manager (optional)
        global risk_manager
        try:
            risk_manager = PortfolioRiskManager(
                config_path="config/risk_limits.yaml"
            )
            logger.info("✅ Risk Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Risk Manager initialization failed: {e}. Continuing without Risk Manager.")
            risk_manager = None
        
        # Initialize Broker Manager (optional)
        global broker_manager
        try:
            broker_manager = BrokerManager()
            logger.info("✅ Broker Manager initialized")
            
            # Auto-reconnect to brokers from database
            if db_client:
                await reconnect_brokers_from_db(broker_manager, db_client)
        except Exception as e:
            logger.warning(f"⚠️ Broker Manager initialization failed: {e}. Continuing without Broker Manager.")
            broker_manager = None
        
        # Initialize route dependencies (with None checks)
        if agent_manager and agent_scheduler and agent_registry:
            initialize_agent_routes(agent_manager, agent_scheduler, agent_registry)
        if db_client and risk_manager:
            initialize_portfolio_routes(db_client, risk_manager)
        # Orders route needs BrokerManager and PreTradeChecker - skip for now
        # if db_client:
        #     initialize_order_routes(db_client, broker_manager, pre_trade_checker)
        if redis_client:
            initialize_websocket_routes(redis_client)
        if redis_client or db_client:
            initialize_metrics_routes(redis_client, db_client)
        if broker_manager:
            initialize_broker_routes(broker_manager)
        
        logger.info("✅ MTQuant API started successfully (some services may be unavailable)!")
    
    except Exception as e:
        logger.error(f"❌ Critical error during startup: {e}", exc_info=True)
        logger.warning("⚠️ Starting API in degraded mode...")
        # Don't raise - allow API to start in degraded mode
    
    yield
    
    # === SHUTDOWN ===
    logger.info("Shutting down MTQuant API...")
    
    try:
        # Disconnect Redis
        if redis_client:
            await redis_client.disconnect()
            logger.info("Redis client disconnected")
        
        # Disconnect PostgreSQL
        if db_client:
            await db_client.disconnect()
            logger.info("PostgreSQL client disconnected")
        
        # Disconnect QuestDB
        if questdb_client:
            await questdb_client.disconnect()
            logger.info("QuestDB client disconnected")
        
        logger.info("✅ MTQuant API shutdown complete")
    
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# Create FastAPI application
app = FastAPI(
    title="MTQuant Trading System API",
    description="Production-grade REST API for multi-agent RL trading",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} "
        f"[{response.status_code}] {duration:.3f}s"
    )
    
    return response


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Path {request.url.path} not found",
            "timestamp": time.time()
        }
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": time.time()
        }
    )


# Include routers
app.include_router(agents_router)
app.include_router(portfolio_router)
app.include_router(orders_router)
app.include_router(websocket_router)
app.include_router(metrics_router)
app.include_router(brokers_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": "MTQuant Trading System API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/api/docs",
        "health": "/api/metrics/health"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


# Get global instances (for use in other modules)
def get_redis_client() -> Optional[RedisClient]:
    """Get Redis client instance."""
    return redis_client


def get_db_client() -> Optional[PostgreSQLClient]:
    """Get PostgreSQL client instance."""
    return db_client


def get_questdb_client() -> Optional[QuestDBClient]:
    """Get QuestDB client instance."""
    return questdb_client


def get_agent_manager() -> Optional[AgentLifecycleManager]:
    """Get Agent Manager instance."""
    return agent_manager


def get_risk_manager() -> Optional[PortfolioRiskManager]:
    """Get Risk Manager instance."""
    return risk_manager


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

