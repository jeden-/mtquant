"""
PostgreSQL Client - Transactional Data Storage.

Handles transactional data including orders, trades, positions,
agent configurations, and audit logs.

Author: MTQuant Development Team
Date: October 15, 2025
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import asyncpg
from dataclasses import dataclass

from mtquant.utils.logger import get_logger
from mtquant.utils.exceptions import DatabaseError, ConnectionError, QueryError


logger = get_logger(__name__)


@dataclass
class PostgreSQLConfig:
    """PostgreSQL connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "mtquant"
    user: str = "mtquant"
    password: str = "mtquant"
    min_pool_size: int = 5
    max_pool_size: int = 20
    command_timeout: float = 60.0


class PostgreSQLClient:
    """
    PostgreSQL client for transactional data operations.
    
    Provides async interface for:
    - Orders and trades storage
    - Positions tracking
    - Agent configurations (JSONB)
    - Audit logs
    - Performance metrics
    
    Example:
        >>> config = PostgreSQLConfig(host='localhost')
        >>> client = PostgreSQLClient(config)
        >>> await client.connect()
        >>> 
        >>> # Insert order
        >>> order_id = await client.insert_order(
        ...     agent_id='agent_1',
        ...     symbol='XAUUSD',
        ...     side='buy',
        ...     order_type='market',
        ...     quantity=0.1,
        ...     price=2050.0
        ... )
    """
    
    def __init__(self, config: PostgreSQLConfig):
        """
        Initialize PostgreSQL client.
        
        Args:
            config: PostgreSQL configuration
        """
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self.logger = get_logger(__name__)
        self._connected = False
    
    async def connect(self, retries: int = 5, delay: int = 3) -> bool:
        """
        Establish connection pool to PostgreSQL with retries.
        
        Args:
            retries: Number of connection attempts
            delay: Delay between attempts in seconds

        Returns:
            True if connection successful
            
        Raises:
            ConnectionError: If connection fails after all retries
        """
        for attempt in range(retries):
            try:
                self._pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,
                    min_size=self.config.min_pool_size,
                    max_size=self.config.max_pool_size,
                    command_timeout=self.config.command_timeout
                )
                
                # Test connection
                async with self._pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')
                
                self._connected = True
                self.logger.info(
                    f"Connected to PostgreSQL at {self.config.host}:{self.config.port}"
                )
                
                # Create tables if they don't exist
                await self._create_tables()
                
                return True
            
            except Exception as e:
                self.logger.warning(
                    f"PostgreSQL connection attempt {attempt + 1}/{retries} failed: {e}"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"Failed to connect to PostgreSQL after {retries} attempts.")
                    raise ConnectionError(f"PostgreSQL connection failed: {e}")
        return False
    
    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._connected = False
            self.logger.info("Disconnected from PostgreSQL")
    
    async def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        tables = [
            # Orders table
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id BIGSERIAL PRIMARY KEY,
                agent_id VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
                order_type VARCHAR(20) NOT NULL,
                quantity DECIMAL(18, 8) NOT NULL,
                price DECIMAL(18, 8),
                stop_loss DECIMAL(18, 8),
                take_profit DECIMAL(18, 8),
                status VARCHAR(20) NOT NULL,
                broker_order_id VARCHAR(100),
                filled_quantity DECIMAL(18, 8) DEFAULT 0,
                avg_fill_price DECIMAL(18, 8),
                commission DECIMAL(18, 8),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                filled_at TIMESTAMPTZ,
                created_by VARCHAR(100) NOT NULL DEFAULT 'system'
            );
            
            CREATE INDEX IF NOT EXISTS idx_orders_agent_symbol ON orders(agent_id, symbol);
            CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            """,
            
            # Trades table
            """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id BIGSERIAL PRIMARY KEY,
                order_id BIGINT REFERENCES orders(order_id),
                agent_id VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
                quantity DECIMAL(18, 8) NOT NULL,
                price DECIMAL(18, 8) NOT NULL,
                commission DECIMAL(18, 8),
                pnl DECIMAL(18, 8),
                broker_trade_id VARCHAR(100),
                executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata JSONB
            );
            
            CREATE INDEX IF NOT EXISTS idx_trades_agent_symbol ON trades(agent_id, symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_executed_at ON trades(executed_at DESC);
            """,
            
            # Positions table
            """
            CREATE TABLE IF NOT EXISTS positions (
                position_id BIGSERIAL PRIMARY KEY,
                agent_id VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(4) NOT NULL CHECK (side IN ('long', 'short')),
                quantity DECIMAL(18, 8) NOT NULL,
                entry_price DECIMAL(18, 8) NOT NULL,
                current_price DECIMAL(18, 8),
                stop_loss DECIMAL(18, 8),
                take_profit DECIMAL(18, 8),
                unrealized_pnl DECIMAL(18, 8),
                realized_pnl DECIMAL(18, 8) DEFAULT 0,
                opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                closed_at TIMESTAMPTZ,
                status VARCHAR(20) NOT NULL DEFAULT 'open',
                UNIQUE(agent_id, symbol, status)
            );
            
            CREATE INDEX IF NOT EXISTS idx_positions_agent_symbol ON positions(agent_id, symbol);
            CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
            """,
            
            # Agent configurations
            """
            CREATE TABLE IF NOT EXISTS agent_config (
                agent_id VARCHAR(50) PRIMARY KEY,
                agent_type VARCHAR(50) NOT NULL,
                symbol VARCHAR(20),
                config JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_agent_config_jsonb ON agent_config USING GIN(config);
            """,
            
            # Audit logs
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id BIGSERIAL PRIMARY KEY,
                event_type VARCHAR(50) NOT NULL,
                user_id VARCHAR(100) NOT NULL,
                agent_id VARCHAR(50),
                symbol VARCHAR(20),
                action TEXT NOT NULL,
                details JSONB,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
            CREATE INDEX IF NOT EXISTS idx_audit_log_agent_id ON audit_log(agent_id);
            """,
            
            # Performance metrics
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id BIGSERIAL PRIMARY KEY,
                agent_id VARCHAR(50) NOT NULL,
                symbol VARCHAR(20),
                metric_date DATE NOT NULL,
                total_trades INT DEFAULT 0,
                winning_trades INT DEFAULT 0,
                losing_trades INT DEFAULT 0,
                total_pnl DECIMAL(18, 8) DEFAULT 0,
                sharpe_ratio DECIMAL(10, 4),
                max_drawdown DECIMAL(10, 4),
                win_rate DECIMAL(10, 4),
                avg_win DECIMAL(18, 8),
                avg_loss DECIMAL(18, 8),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(agent_id, symbol, metric_date)
            );
            
            CREATE INDEX IF NOT EXISTS idx_performance_agent_date ON performance_metrics(agent_id, metric_date DESC);
            """
        ]
        
        async with self._pool.acquire() as conn:
            for table_sql in tables:
                try:
                    await conn.execute(table_sql)
                except Exception as e:
                    self.logger.debug(f"Table creation note: {e}")
    
    async def insert_order(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        status: str = "pending",
        created_by: str = "system"
    ) -> int:
        """
        Insert new order.
        
        Args:
            agent_id: Agent identifier
            symbol: Trading symbol
            side: 'buy' or 'sell'
            order_type: Order type ('market', 'limit', 'stop')
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            status: Order status
            created_by: User who created the order
            
        Returns:
            Order ID
        """
        query = """
            INSERT INTO orders (
                agent_id, symbol, side, order_type, quantity, price,
                stop_loss, take_profit, status, created_by
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING order_id
        """
        
        async with self._pool.acquire() as conn:
            try:
                order_id = await conn.fetchval(
                    query,
                    agent_id, symbol, side, order_type, quantity, price,
                    stop_loss, take_profit, status, created_by
                )
                
                self.logger.info(f"Inserted order {order_id} for agent {agent_id}")
                return order_id
            
            except Exception as e:
                self.logger.error(f"Failed to insert order: {e}")
                raise QueryError(f"Order insert failed: {e}")
    
    async def update_order_status(
        self,
        order_id: int,
        status: str,
        broker_order_id: Optional[str] = None,
        filled_quantity: Optional[float] = None,
        avg_fill_price: Optional[float] = None,
        commission: Optional[float] = None
    ) -> None:
        """
        Update order status.
        
        Args:
            order_id: Order ID
            status: New status
            broker_order_id: Broker's order ID
            filled_quantity: Filled quantity
            avg_fill_price: Average fill price
            commission: Commission paid
        """
        query = """
            UPDATE orders
            SET status = $2,
                broker_order_id = COALESCE($3, broker_order_id),
                filled_quantity = COALESCE($4, filled_quantity),
                avg_fill_price = COALESCE($5, avg_fill_price),
                commission = COALESCE($6, commission),
                updated_at = NOW(),
                filled_at = CASE WHEN $2 = 'filled' THEN NOW() ELSE filled_at END
            WHERE order_id = $1
        """
        
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    query,
                    order_id, status, broker_order_id, filled_quantity,
                    avg_fill_price, commission
                )
                
                self.logger.info(f"Updated order {order_id} status to {status}")
            
            except Exception as e:
                self.logger.error(f"Failed to update order: {e}")
                raise QueryError(f"Order update failed: {e}")
    
    async def get_orders(
        self,
        agent_id: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get orders with optional filters.
        
        Args:
            agent_id: Filter by agent ID
            symbol: Filter by symbol
            status: Filter by status
            limit: Maximum number of orders
            
        Returns:
            List of order dictionaries
        """
        conditions = []
        params = []
        param_count = 1
        
        if agent_id:
            conditions.append(f"agent_id = ${param_count}")
            params.append(agent_id)
            param_count += 1
        
        if symbol:
            conditions.append(f"symbol = ${param_count}")
            params.append(symbol)
            param_count += 1
        
        if status:
            conditions.append(f"status = ${param_count}")
            params.append(status)
            param_count += 1
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT * FROM orders
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_count}
        """
        params.append(limit)
        
        async with self._pool.acquire() as conn:
            try:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
            
            except Exception as e:
                self.logger.error(f"Failed to get orders: {e}")
                raise QueryError(f"Orders fetch failed: {e}")
    
    async def insert_trade(
        self,
        order_id: int,
        agent_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: Optional[float] = None,
        pnl: Optional[float] = None,
        broker_trade_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Insert trade record.
        
        Args:
            order_id: Related order ID
            agent_id: Agent identifier
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            price: Execution price
            commission: Commission paid
            pnl: Profit/loss
            broker_trade_id: Broker's trade ID
            metadata: Additional metadata
            
        Returns:
            Trade ID
        """
        query = """
            INSERT INTO trades (
                order_id, agent_id, symbol, side, quantity, price,
                commission, pnl, broker_trade_id, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING trade_id
        """
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        async with self._pool.acquire() as conn:
            try:
                trade_id = await conn.fetchval(
                    query,
                    order_id, agent_id, symbol, side, quantity, price,
                    commission, pnl, broker_trade_id, metadata_json
                )
                
                self.logger.info(f"Inserted trade {trade_id} for order {order_id}")
                return trade_id
            
            except Exception as e:
                self.logger.error(f"Failed to insert trade: {e}")
                raise QueryError(f"Trade insert failed: {e}")
    
    async def upsert_position(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        current_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        unrealized_pnl: Optional[float] = None
    ) -> int:
        """
        Insert or update position.
        
        Args:
            agent_id: Agent identifier
            symbol: Trading symbol
            side: 'long' or 'short'
            quantity: Position quantity
            entry_price: Entry price
            current_price: Current price
            stop_loss: Stop loss price
            take_profit: Take profit price
            unrealized_pnl: Unrealized P&L
            
        Returns:
            Position ID
        """
        query = """
            INSERT INTO positions (
                agent_id, symbol, side, quantity, entry_price, current_price,
                stop_loss, take_profit, unrealized_pnl, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'open')
            ON CONFLICT (agent_id, symbol, status)
            DO UPDATE SET
                quantity = EXCLUDED.quantity,
                current_price = EXCLUDED.current_price,
                stop_loss = EXCLUDED.stop_loss,
                take_profit = EXCLUDED.take_profit,
                unrealized_pnl = EXCLUDED.unrealized_pnl,
                updated_at = NOW()
            RETURNING position_id
        """
        
        async with self._pool.acquire() as conn:
            try:
                position_id = await conn.fetchval(
                    query,
                    agent_id, symbol, side, quantity, entry_price, current_price,
                    stop_loss, take_profit, unrealized_pnl
                )
                
                self.logger.info(f"Upserted position {position_id} for {agent_id}/{symbol}")
                return position_id
            
            except Exception as e:
                self.logger.error(f"Failed to upsert position: {e}")
                raise QueryError(f"Position upsert failed: {e}")
    
    async def get_open_positions(
        self,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get open positions.
        
        Args:
            agent_id: Filter by agent ID (optional)
            
        Returns:
            List of position dictionaries
        """
        if agent_id:
            query = """
                SELECT * FROM positions
                WHERE agent_id = $1 AND status = 'open'
                ORDER BY opened_at DESC
            """
            params = [agent_id]
        else:
            query = """
                SELECT * FROM positions
                WHERE status = 'open'
                ORDER BY opened_at DESC
            """
            params = []
        
        async with self._pool.acquire() as conn:
            try:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
            
            except Exception as e:
                self.logger.error(f"Failed to get positions: {e}")
                raise QueryError(f"Positions fetch failed: {e}")
    
    async def save_agent_config(
        self,
        agent_id: str,
        agent_type: str,
        config: Dict[str, Any],
        symbol: Optional[str] = None
    ) -> None:
        """
        Save agent configuration.
        
        Args:
            agent_id: Agent identifier
            agent_type: Agent type
            config: Configuration dictionary
            symbol: Trading symbol (optional)
        """
        query = """
            INSERT INTO agent_config (agent_id, agent_type, symbol, config)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (agent_id)
            DO UPDATE SET
                agent_type = EXCLUDED.agent_type,
                symbol = EXCLUDED.symbol,
                config = EXCLUDED.config,
                updated_at = NOW()
        """
        
        config_json = json.dumps(config)
        
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(query, agent_id, agent_type, symbol, config_json)
                self.logger.info(f"Saved config for agent {agent_id}")
            
            except Exception as e:
                self.logger.error(f"Failed to save agent config: {e}")
                raise QueryError(f"Agent config save failed: {e}")
    
    async def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent configuration.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Configuration dictionary or None
        """
        query = "SELECT * FROM agent_config WHERE agent_id = $1"
        
        async with self._pool.acquire() as conn:
            try:
                row = await conn.fetchrow(query, agent_id)
                return dict(row) if row else None
            
            except Exception as e:
                self.logger.error(f"Failed to get agent config: {e}")
                raise QueryError(f"Agent config fetch failed: {e}")
    
    async def log_audit_event(
        self,
        event_type: str,
        user_id: str,
        action: str,
        agent_id: Optional[str] = None,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Log audit event.
        
        Args:
            event_type: Event type
            user_id: User identifier
            action: Action description
            agent_id: Agent identifier (optional)
            symbol: Trading symbol (optional)
            details: Additional details (optional)
            
        Returns:
            Log ID
        """
        query = """
            INSERT INTO audit_log (
                event_type, user_id, agent_id, symbol, action, details
            ) VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING log_id
        """
        
        details_json = json.dumps(details) if details else None
        
        async with self._pool.acquire() as conn:
            try:
                log_id = await conn.fetchval(
                    query,
                    event_type, user_id, agent_id, symbol, action, details_json
                )
                
                return log_id
            
            except Exception as e:
                self.logger.error(f"Failed to log audit event: {e}")
                raise QueryError(f"Audit log failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check database health.
        
        Returns:
            Health status dictionary
        """
        try:
            async with self._pool.acquire() as conn:
                # Check connection
                await conn.fetchval('SELECT 1')
                
                # Get table sizes
                tables_query = """
                    SELECT 
                        schemaname,
                        tablename,
                        pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY size_bytes DESC
                """
                
                tables = await conn.fetch(tables_query)
                
                return {
                    'connected': True,
                    'host': self.config.host,
                    'port': self.config.port,
                    'database': self.config.database,
                    'tables': [
                        {
                            'name': row['tablename'],
                            'size_mb': row['size_bytes'] / (1024 * 1024)
                        }
                        for row in tables
                    ]
                }
        
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self._pool is not None
    
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            List of open positions
        """
        query = """
            SELECT 
                position_id,
                agent_id,
                symbol,
                side,
                quantity,
                entry_price,
                current_price,
                unrealized_pnl,
                opened_at,
                updated_at,
                status
            FROM positions
            WHERE status = 'open'
            ORDER BY opened_at DESC
        """
        
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query)
                return [dict(row) for row in rows]
        
        except Exception as e:
            self.logger.error(f"Failed to get open positions: {e}")
            raise QueryError(f"Failed to get open positions: {e}")

