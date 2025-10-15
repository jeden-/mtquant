"""
WebSocket API Routes - Real-time Updates.

Provides WebSocket endpoints for real-time portfolio, order, and agent updates.

Author: MTQuant Development Team
Date: October 15, 2025
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set, Optional
import asyncio
import json
from datetime import datetime

from mtquant.utils.logger import get_logger
from mtquant.data.storage.redis_client import RedisClient


logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["websocket"])

# Connection manager
redis_client: Optional[RedisClient] = None


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasting.
    
    Supports multiple channels:
    - portfolio: Portfolio updates
    - orders: Order updates
    - agents: Agent status updates
    - market: Market data updates
    """
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, Set[WebSocket]] = {
            'portfolio': set(),
            'orders': set(),
            'agents': set(),
            'market': set(),
            'all': set()
        }
        self.logger = get_logger(__name__)
    
    async def connect(self, websocket: WebSocket, channel: str = 'all'):
        """
        Accept new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            channel: Channel to subscribe to
        """
        await websocket.accept()
        self.active_connections[channel].add(websocket)
        self.active_connections['all'].add(websocket)
        
        self.logger.info(
            f"New WebSocket connection to channel '{channel}' "
            f"(total: {len(self.active_connections[channel])})"
        )
    
    def disconnect(self, websocket: WebSocket, channel: str = 'all'):
        """
        Remove WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            channel: Channel to unsubscribe from
        """
        if websocket in self.active_connections[channel]:
            self.active_connections[channel].remove(websocket)
        
        if websocket in self.active_connections['all']:
            self.active_connections['all'].remove(websocket)
        
        self.logger.info(
            f"WebSocket disconnected from channel '{channel}' "
            f"(remaining: {len(self.active_connections[channel])})"
        )
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send message to specific connection.
        
        Args:
            message: Message dictionary
            websocket: Target WebSocket
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            self.logger.error(f"Failed to send personal message: {e}")
    
    async def broadcast(self, message: dict, channel: str = 'all'):
        """
        Broadcast message to all connections in channel.
        
        Args:
            message: Message dictionary
            channel: Target channel
        """
        if channel not in self.active_connections:
            self.logger.warning(f"Unknown channel: {channel}")
            return
        
        disconnected = []
        
        for connection in self.active_connections[channel].copy():
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.warning(f"Failed to send to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection, channel)
    
    def get_stats(self) -> dict:
        """
        Get connection statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'channels': {
                channel: len(connections)
                for channel, connections in self.active_connections.items()
            },
            'total_connections': len(self.active_connections['all'])
        }


# Global connection manager
manager = ConnectionManager()


def get_redis_client() -> RedisClient:
    """Dependency to get Redis client."""
    if redis_client is None:
        raise RuntimeError("Redis client not initialized")
    return redis_client


@router.websocket("/ws/portfolio")
async def websocket_portfolio(websocket: WebSocket):
    """
    WebSocket endpoint for portfolio updates.
    
    Sends real-time updates for:
    - Portfolio summary (equity, P&L, etc.)
    - Position updates
    - Risk metrics
    """
    await manager.connect(websocket, 'portfolio')
    
    try:
        # Send initial connection message
        await manager.send_personal_message({
            'type': 'connection',
            'status': 'connected',
            'channel': 'portfolio',
            'timestamp': datetime.now().isoformat()
        }, websocket)
        
        # Keep connection alive and send updates
        while True:
            # Wait for client messages (ping/pong)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                # Handle ping
                if data == 'ping':
                    await manager.send_personal_message({
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }, websocket)
            
            except asyncio.TimeoutError:
                # Send heartbeat
                await manager.send_personal_message({
                    'type': 'heartbeat',
                    'timestamp': datetime.now().isoformat()
                }, websocket)
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, 'portfolio')
        logger.info("Portfolio WebSocket disconnected")
    
    except Exception as e:
        logger.error(f"Portfolio WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket, 'portfolio')


@router.websocket("/ws/orders")
async def websocket_orders(websocket: WebSocket):
    """
    WebSocket endpoint for order updates.
    
    Sends real-time updates for:
    - Order status changes
    - Trade executions
    - Order rejections
    """
    await manager.connect(websocket, 'orders')
    
    try:
        # Send initial connection message
        await manager.send_personal_message({
            'type': 'connection',
            'status': 'connected',
            'channel': 'orders',
            'timestamp': datetime.now().isoformat()
        }, websocket)
        
        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                if data == 'ping':
                    await manager.send_personal_message({
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }, websocket)
            
            except asyncio.TimeoutError:
                await manager.send_personal_message({
                    'type': 'heartbeat',
                    'timestamp': datetime.now().isoformat()
                }, websocket)
            
            await asyncio.sleep(0.1)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, 'orders')
        logger.info("Orders WebSocket disconnected")
    
    except Exception as e:
        logger.error(f"Orders WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket, 'orders')


@router.websocket("/ws/agents")
async def websocket_agents(websocket: WebSocket):
    """
    WebSocket endpoint for agent status updates.
    
    Sends real-time updates for:
    - Agent state changes
    - Agent health status
    - Performance metrics
    """
    await manager.connect(websocket, 'agents')
    
    try:
        # Send initial connection message
        await manager.send_personal_message({
            'type': 'connection',
            'status': 'connected',
            'channel': 'agents',
            'timestamp': datetime.now().isoformat()
        }, websocket)
        
        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                if data == 'ping':
                    await manager.send_personal_message({
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }, websocket)
            
            except asyncio.TimeoutError:
                await manager.send_personal_message({
                    'type': 'heartbeat',
                    'timestamp': datetime.now().isoformat()
                }, websocket)
            
            await asyncio.sleep(0.1)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, 'agents')
        logger.info("Agents WebSocket disconnected")
    
    except Exception as e:
        logger.error(f"Agents WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket, 'agents')


@router.websocket("/ws/market")
async def websocket_market(websocket: WebSocket):
    """
    WebSocket endpoint for market data updates.
    
    Sends real-time updates for:
    - Latest prices
    - Market indicators
    - Trading signals
    """
    await manager.connect(websocket, 'market')
    
    try:
        # Send initial connection message
        await manager.send_personal_message({
            'type': 'connection',
            'status': 'connected',
            'channel': 'market',
            'timestamp': datetime.now().isoformat()
        }, websocket)
        
        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                if data == 'ping':
                    await manager.send_personal_message({
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }, websocket)
            
            except asyncio.TimeoutError:
                await manager.send_personal_message({
                    'type': 'heartbeat',
                    'timestamp': datetime.now().isoformat()
                }, websocket)
            
            await asyncio.sleep(0.1)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, 'market')
        logger.info("Market WebSocket disconnected")
    
    except Exception as e:
        logger.error(f"Market WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket, 'market')


# Helper functions for broadcasting updates

async def broadcast_portfolio_update(data: dict):
    """
    Broadcast portfolio update to all connected clients.
    
    Args:
        data: Portfolio update data
    """
    message = {
        'type': 'portfolio_update',
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    await manager.broadcast(message, 'portfolio')


async def broadcast_order_update(data: dict):
    """
    Broadcast order update to all connected clients.
    
    Args:
        data: Order update data
    """
    message = {
        'type': 'order_update',
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    await manager.broadcast(message, 'orders')


async def broadcast_agent_update(data: dict):
    """
    Broadcast agent update to all connected clients.
    
    Args:
        data: Agent update data
    """
    message = {
        'type': 'agent_update',
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    await manager.broadcast(message, 'agents')


async def broadcast_market_update(data: dict):
    """
    Broadcast market update to all connected clients.
    
    Args:
        data: Market update data
    """
    message = {
        'type': 'market_update',
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    await manager.broadcast(message, 'market')


# Redis Pub/Sub listener (background task)

async def redis_pubsub_listener():
    """
    Listen to Redis Pub/Sub and broadcast to WebSocket clients.
    
    This should be run as a background task.
    """
    if redis_client is None:
        logger.warning("Redis client not initialized, skipping Pub/Sub")
        return
    
    try:
        # Subscribe to channels
        pubsub = await redis_client.subscribe('mtquant:updates')
        
        logger.info("Redis Pub/Sub listener started")
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    channel = data.get('channel', 'all')
                    
                    # Route to appropriate WebSocket channel
                    if channel == 'portfolio':
                        await broadcast_portfolio_update(data)
                    elif channel == 'orders':
                        await broadcast_order_update(data)
                    elif channel == 'agents':
                        await broadcast_agent_update(data)
                    elif channel == 'market':
                        await broadcast_market_update(data)
                
                except Exception as e:
                    logger.error(f"Failed to process Pub/Sub message: {e}")
    
    except Exception as e:
        logger.error(f"Redis Pub/Sub listener error: {e}", exc_info=True)


# Initialize function (should be called on app startup)
def initialize_websocket_routes(redis: Optional[RedisClient] = None):
    """
    Initialize WebSocket routes with Redis client.
    
    Args:
        redis: Redis client instance (optional)
    """
    global redis_client
    redis_client = redis
    logger.info("WebSocket routes initialized")


# Get connection manager stats
def get_connection_stats() -> dict:
    """Get WebSocket connection statistics."""
    return manager.get_stats()

