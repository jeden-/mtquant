"""
Redis Client - Hot Data & Caching Layer.

Handles hot data caching, replay buffers, and real-time state management.

Redis is used for:
- Latest price caching (TTL 60s)
- Experience replay buffers (Sorted Sets)
- Agent state caching
- Real-time metrics
- Session management

Author: MTQuant Development Team
Date: October 15, 2025
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import redis.asyncio as redis
from dataclasses import dataclass

from mtquant.utils.logger import get_logger
from mtquant.utils.exceptions import DatabaseError, ConnectionError, QueryError


logger = get_logger(__name__)


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    decode_responses: bool = True
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0


class RedisClient:
    """
    Redis client for hot data and caching operations.
    
    Provides async interface for:
    - Price caching (TTL-based)
    - Experience replay buffers (Sorted Sets)
    - Agent state caching
    - Real-time metrics
    - Pub/Sub messaging
    
    Example:
        >>> config = RedisConfig(host='localhost')
        >>> client = RedisClient(config)
        >>> await client.connect()
        >>> 
        >>> # Cache latest price
        >>> await client.cache_price('XAUUSD', 2050.5, ttl=60)
        >>> 
        >>> # Get cached price
        >>> price = await client.get_price('XAUUSD')
        >>> 
        >>> # Add to replay buffer
        >>> await client.add_experience(
        ...     agent_id='agent_1',
        ...     experience={'state': [...], 'action': 0, 'reward': 0.5}
        ... )
    """
    
    def __init__(self, config: RedisConfig):
        """
        Initialize Redis client.
        
        Args:
            config: Redis configuration
        """
        self.config = config
        self._client: Optional[redis.Redis] = None
        self.logger = get_logger(__name__)
        self._connected = False
    
    async def connect(self) -> bool:
        """
        Establish connection to Redis.
        
        Returns:
            True if connection successful
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                decode_responses=self.config.decode_responses,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout
            )
            
            # Test connection
            await self._client.ping()
            
            self._connected = True
            self.logger.info(
                f"Connected to Redis at {self.config.host}:{self.config.port}"
            )
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._connected = False
            self.logger.info("Disconnected from Redis")
    
    # ========== Price Caching ==========
    
    async def cache_price(
        self,
        symbol: str,
        price: float,
        ttl: int = 60
    ) -> None:
        """
        Cache latest price with TTL.
        
        Args:
            symbol: Trading symbol
            price: Current price
            ttl: Time to live in seconds (default: 60)
        """
        key = f"price:{symbol}"
        
        try:
            await self._client.setex(key, ttl, str(price))
            self.logger.debug(f"Cached price for {symbol}: {price}")
        
        except Exception as e:
            self.logger.error(f"Failed to cache price: {e}")
            raise QueryError(f"Price cache failed: {e}")
    
    async def get_price(self, symbol: str) -> Optional[float]:
        """
        Get cached price.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Price or None if not cached
        """
        key = f"price:{symbol}"
        
        try:
            price_str = await self._client.get(key)
            return float(price_str) if price_str else None
        
        except Exception as e:
            self.logger.error(f"Failed to get price: {e}")
            return None
    
    async def cache_prices_bulk(
        self,
        prices: Dict[str, float],
        ttl: int = 60
    ) -> None:
        """
        Cache multiple prices at once.
        
        Args:
            prices: Dictionary of symbol -> price
            ttl: Time to live in seconds
        """
        pipe = self._client.pipeline()
        
        for symbol, price in prices.items():
            key = f"price:{symbol}"
            pipe.setex(key, ttl, str(price))
        
        try:
            await pipe.execute()
            self.logger.debug(f"Cached {len(prices)} prices")
        
        except Exception as e:
            self.logger.error(f"Failed to cache prices bulk: {e}")
            raise QueryError(f"Bulk price cache failed: {e}")
    
    # ========== Experience Replay Buffer ==========
    
    async def add_experience(
        self,
        agent_id: str,
        experience: Dict[str, Any],
        max_size: int = 100000
    ) -> None:
        """
        Add experience to replay buffer (Sorted Set).
        
        Uses timestamp as score for FIFO behavior.
        Automatically trims buffer to max_size.
        
        Args:
            agent_id: Agent identifier
            experience: Experience dictionary
            max_size: Maximum buffer size (default: 100K)
        """
        key = f"replay:{agent_id}"
        score = datetime.now().timestamp()
        value = json.dumps(experience)
        
        try:
            # Add experience
            await self._client.zadd(key, {value: score})
            
            # Trim to max size (keep newest)
            await self._client.zremrangebyrank(key, 0, -(max_size + 1))
            
            self.logger.debug(f"Added experience to {agent_id} replay buffer")
        
        except Exception as e:
            self.logger.error(f"Failed to add experience: {e}")
            raise QueryError(f"Experience add failed: {e}")
    
    async def sample_experiences(
        self,
        agent_id: str,
        batch_size: int = 32,
        prioritized: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Sample experiences from replay buffer.
        
        Args:
            agent_id: Agent identifier
            batch_size: Number of experiences to sample
            prioritized: If True, sample recent experiences (default: False for random)
            
        Returns:
            List of experience dictionaries
        """
        key = f"replay:{agent_id}"
        
        try:
            if prioritized:
                # Get most recent experiences
                experiences_json = await self._client.zrevrange(key, 0, batch_size - 1)
            else:
                # Random sampling
                # Get total count
                total = await self._client.zcard(key)
                
                if total == 0:
                    return []
                
                # Random indices
                import random
                indices = random.sample(range(total), min(batch_size, total))
                
                # Fetch by rank
                experiences_json = []
                for idx in indices:
                    exp = await self._client.zrange(key, idx, idx)
                    if exp:
                        experiences_json.extend(exp)
            
            # Parse JSON
            experiences = [json.loads(exp) for exp in experiences_json]
            
            self.logger.debug(f"Sampled {len(experiences)} experiences from {agent_id}")
            return experiences
        
        except Exception as e:
            self.logger.error(f"Failed to sample experiences: {e}")
            return []
    
    async def get_replay_buffer_size(self, agent_id: str) -> int:
        """
        Get replay buffer size.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Buffer size
        """
        key = f"replay:{agent_id}"
        
        try:
            return await self._client.zcard(key)
        
        except Exception as e:
            self.logger.error(f"Failed to get buffer size: {e}")
            return 0
    
    async def clear_replay_buffer(self, agent_id: str) -> None:
        """
        Clear replay buffer.
        
        Args:
            agent_id: Agent identifier
        """
        key = f"replay:{agent_id}"
        
        try:
            await self._client.delete(key)
            self.logger.info(f"Cleared replay buffer for {agent_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to clear buffer: {e}")
            raise QueryError(f"Buffer clear failed: {e}")
    
    # ========== Agent State Caching ==========
    
    async def cache_agent_state(
        self,
        agent_id: str,
        state: Dict[str, Any],
        ttl: int = 300
    ) -> None:
        """
        Cache agent state.
        
        Args:
            agent_id: Agent identifier
            state: State dictionary
            ttl: Time to live in seconds (default: 5 minutes)
        """
        key = f"agent:state:{agent_id}"
        value = json.dumps(state)
        
        try:
            await self._client.setex(key, ttl, value)
            self.logger.debug(f"Cached state for agent {agent_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to cache agent state: {e}")
            raise QueryError(f"Agent state cache failed: {e}")
    
    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached agent state.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            State dictionary or None
        """
        key = f"agent:state:{agent_id}"
        
        try:
            state_json = await self._client.get(key)
            return json.loads(state_json) if state_json else None
        
        except Exception as e:
            self.logger.error(f"Failed to get agent state: {e}")
            return None
    
    # ========== Real-time Metrics ==========
    
    async def increment_metric(
        self,
        metric_name: str,
        agent_id: Optional[str] = None,
        increment: int = 1
    ) -> int:
        """
        Increment a metric counter.
        
        Args:
            metric_name: Metric name
            agent_id: Agent identifier (optional)
            increment: Increment value (default: 1)
            
        Returns:
            New metric value
        """
        if agent_id:
            key = f"metric:{agent_id}:{metric_name}"
        else:
            key = f"metric:global:{metric_name}"
        
        try:
            new_value = await self._client.incrby(key, increment)
            return new_value
        
        except Exception as e:
            self.logger.error(f"Failed to increment metric: {e}")
            return 0
    
    async def get_metric(
        self,
        metric_name: str,
        agent_id: Optional[str] = None
    ) -> int:
        """
        Get metric value.
        
        Args:
            metric_name: Metric name
            agent_id: Agent identifier (optional)
            
        Returns:
            Metric value
        """
        if agent_id:
            key = f"metric:{agent_id}:{metric_name}"
        else:
            key = f"metric:global:{metric_name}"
        
        try:
            value = await self._client.get(key)
            return int(value) if value else 0
        
        except Exception as e:
            self.logger.error(f"Failed to get metric: {e}")
            return 0
    
    async def set_metric(
        self,
        metric_name: str,
        value: float,
        agent_id: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set metric value.
        
        Args:
            metric_name: Metric name
            value: Metric value
            agent_id: Agent identifier (optional)
            ttl: Time to live in seconds (optional)
        """
        if agent_id:
            key = f"metric:{agent_id}:{metric_name}"
        else:
            key = f"metric:global:{metric_name}"
        
        try:
            if ttl:
                await self._client.setex(key, ttl, str(value))
            else:
                await self._client.set(key, str(value))
        
        except Exception as e:
            self.logger.error(f"Failed to set metric: {e}")
            raise QueryError(f"Metric set failed: {e}")
    
    # ========== Pub/Sub Messaging ==========
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> int:
        """
        Publish message to channel.
        
        Args:
            channel: Channel name
            message: Message dictionary
            
        Returns:
            Number of subscribers that received the message
        """
        message_json = json.dumps(message)
        
        try:
            subscribers = await self._client.publish(channel, message_json)
            self.logger.debug(f"Published to {channel}: {subscribers} subscribers")
            return subscribers
        
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
            return 0
    
    async def subscribe(self, channel: str) -> redis.client.PubSub:
        """
        Subscribe to channel.
        
        Args:
            channel: Channel name
            
        Returns:
            PubSub object
        """
        try:
            pubsub = self._client.pubsub()
            await pubsub.subscribe(channel)
            self.logger.info(f"Subscribed to channel: {channel}")
            return pubsub
        
        except Exception as e:
            self.logger.error(f"Failed to subscribe: {e}")
            raise QueryError(f"Subscribe failed: {e}")
    
    # ========== Session Management ==========
    
    async def create_session(
        self,
        session_id: str,
        user_id: str,
        data: Dict[str, Any],
        ttl: int = 3600
    ) -> None:
        """
        Create user session.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            data: Session data
            ttl: Time to live in seconds (default: 1 hour)
        """
        key = f"session:{session_id}"
        value = json.dumps({
            'user_id': user_id,
            'data': data,
            'created_at': datetime.now().isoformat()
        })
        
        try:
            await self._client.setex(key, ttl, value)
            self.logger.info(f"Created session {session_id} for user {user_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            raise QueryError(f"Session creation failed: {e}")
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None
        """
        key = f"session:{session_id}"
        
        try:
            session_json = await self._client.get(key)
            return json.loads(session_json) if session_json else None
        
        except Exception as e:
            self.logger.error(f"Failed to get session: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> None:
        """
        Delete session.
        
        Args:
            session_id: Session identifier
        """
        key = f"session:{session_id}"
        
        try:
            await self._client.delete(key)
            self.logger.info(f"Deleted session {session_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to delete session: {e}")
            raise QueryError(f"Session deletion failed: {e}")
    
    # ========== Utility Methods ==========
    
    async def set_key(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set key-value pair.
        
        Args:
            key: Key name
            value: Value
            ttl: Time to live in seconds (optional)
        """
        try:
            if ttl:
                await self._client.setex(key, ttl, value)
            else:
                await self._client.set(key, value)
        
        except Exception as e:
            self.logger.error(f"Failed to set key: {e}")
            raise QueryError(f"Key set failed: {e}")
    
    async def get_key(self, key: str) -> Optional[str]:
        """
        Get value by key.
        
        Args:
            key: Key name
            
        Returns:
            Value or None
        """
        try:
            return await self._client.get(key)
        
        except Exception as e:
            self.logger.error(f"Failed to get key: {e}")
            return None
    
    async def delete_key(self, key: str) -> bool:
        """
        Delete key.
        
        Args:
            key: Key name
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            result = await self._client.delete(key)
            return result > 0
        
        except Exception as e:
            self.logger.error(f"Failed to delete key: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists.
        
        Args:
            key: Key name
            
        Returns:
            True if exists, False otherwise
        """
        try:
            return await self._client.exists(key) > 0
        
        except Exception as e:
            self.logger.error(f"Failed to check key existence: {e}")
            return False
    
    async def get_ttl(self, key: str) -> int:
        """
        Get key TTL.
        
        Args:
            key: Key name
            
        Returns:
            TTL in seconds (-1 if no TTL, -2 if key doesn't exist)
        """
        try:
            return await self._client.ttl(key)
        
        except Exception as e:
            self.logger.error(f"Failed to get TTL: {e}")
            return -2
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Redis health.
        
        Returns:
            Health status dictionary
        """
        try:
            # Ping
            await self._client.ping()
            
            # Get info
            info = await self._client.info()
            
            return {
                'connected': True,
                'host': self.config.host,
                'port': self.config.port,
                'db': self.config.db,
                'version': info.get('redis_version'),
                'uptime_seconds': info.get('uptime_in_seconds'),
                'connected_clients': info.get('connected_clients'),
                'used_memory_mb': info.get('used_memory') / (1024 * 1024) if info.get('used_memory') else 0,
                'total_commands_processed': info.get('total_commands_processed')
            }
        
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self._client is not None
    
    async def get_metric(self, metric_name: str, agent_id: Optional[str] = None) -> Optional[float]:
        """
        Get real-time metric value.
        
        Args:
            metric_name: Metric name
            agent_id: Optional agent ID filter
            
        Returns:
            Metric value or None if not found
        """
        key = f"metric:{metric_name}"
        if agent_id:
            key = f"metric:{agent_id}:{metric_name}"
        
        try:
            value = await self._client.get(key)
            return float(value) if value else None
        
        except Exception as e:
            self.logger.error(f"Failed to get metric '{metric_name}': {e}")
            return None
    
    async def set_metric(
        self,
        metric_name: str,
        value: float,
        agent_id: Optional[str] = None,
        ttl: int = 3600
    ) -> None:
        """
        Set real-time metric value.
        
        Args:
            metric_name: Metric name
            value: Metric value
            agent_id: Optional agent ID
            ttl: Time to live in seconds (default 1 hour)
        """
        key = f"metric:{metric_name}"
        if agent_id:
            key = f"metric:{agent_id}:{metric_name}"
        
        try:
            await self._client.setex(key, ttl, str(value))
        
        except Exception as e:
            self.logger.error(f"Failed to set metric '{metric_name}': {e}")
            raise DatabaseError(f"Failed to set metric: {e}")

