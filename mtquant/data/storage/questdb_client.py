"""
QuestDB Client - Time-Series Data Storage.

High-performance time-series database client for OHLCV data, indicators,
and trading signals storage and retrieval.

QuestDB is optimized for time-series data with:
- Designated timestamp columns
- Partitioning by time
- SYMBOL type for efficient string storage
- ASOF JOIN for time-series alignment

Author: MTQuant Development Team
Date: October 15, 2025
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
import asyncpg
from dataclasses import dataclass

from mtquant.utils.logger import get_logger
from mtquant.utils.exceptions import DatabaseError, ConnectionError, QueryError


logger = get_logger(__name__)


@dataclass
class QuestDBConfig:
    """QuestDB connection configuration."""
    host: str = "localhost"
    port: int = 8812  # PostgreSQL wire protocol port
    database: str = "qdb"
    user: str = "admin"
    password: str = "quest"
    min_pool_size: int = 2
    max_pool_size: int = 10
    command_timeout: float = 60.0


class QuestDBClient:
    """
    QuestDB client for time-series data operations.
    
    Provides async interface for:
    - OHLCV data storage and retrieval
    - Technical indicators storage
    - Time-series queries with ASOF JOIN
    - Bulk inserts for performance
    
    Example:
        >>> config = QuestDBConfig(host='localhost', port=8812)
        >>> client = QuestDBClient(config)
        >>> await client.connect()
        >>> 
        >>> # Insert OHLCV data
        >>> await client.insert_ohlcv(
        ...     symbol='XAUUSD',
        ...     timestamp=datetime.now(),
        ...     open=2050.5,
        ...     high=2055.0,
        ...     low=2048.0,
        ...     close=2053.0,
        ...     volume=1000.0
        ... )
        >>> 
        >>> # Fetch OHLCV data
        >>> df = await client.fetch_ohlcv(
        ...     symbol='XAUUSD',
        ...     start=datetime.now() - timedelta(days=1),
        ...     end=datetime.now(),
        ...     timeframe='1m'
        ... )
    """
    
    def __init__(self, config: QuestDBConfig):
        """
        Initialize QuestDB client.
        
        Args:
            config: QuestDB configuration
        """
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self.logger = get_logger(__name__)
        self._connected = False
    
    async def connect(self) -> bool:
        """
        Establish connection pool to QuestDB.
        
        Returns:
            True if connection successful
            
        Raises:
            ConnectionError: If connection fails
        """
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
                f"Connected to QuestDB at {self.config.host}:{self.config.port}"
            )
            
            # Create tables if they don't exist
            await self._create_tables()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to connect to QuestDB: {e}")
            raise ConnectionError(f"QuestDB connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._connected = False
            self.logger.info("Disconnected from QuestDB")
    
    async def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        tables = [
            # OHLCV data (1-minute bars)
            """
            CREATE TABLE IF NOT EXISTS ohlcv_1m (
                timestamp TIMESTAMP,
                symbol SYMBOL CAPACITY 1000 CACHE,
                exchange SYMBOL CAPACITY 50 CACHE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                vwap DOUBLE
            ) TIMESTAMP(timestamp) PARTITION BY DAY;
            """,
            
            # Technical indicators
            """
            CREATE TABLE IF NOT EXISTS indicators_1m (
                timestamp TIMESTAMP,
                symbol SYMBOL CAPACITY 1000 CACHE,
                rsi DOUBLE,
                macd DOUBLE,
                macd_signal DOUBLE,
                macd_hist DOUBLE,
                bb_upper DOUBLE,
                bb_middle DOUBLE,
                bb_lower DOUBLE,
                atr DOUBLE,
                adx DOUBLE,
                ema_20 DOUBLE,
                ema_50 DOUBLE,
                ema_200 DOUBLE
            ) TIMESTAMP(timestamp) PARTITION BY DAY;
            """,
            
            # Trading signals
            """
            CREATE TABLE IF NOT EXISTS signals (
                timestamp TIMESTAMP,
                agent_id SYMBOL CAPACITY 100 CACHE,
                symbol SYMBOL CAPACITY 1000 CACHE,
                signal_type SYMBOL CAPACITY 10 CACHE,
                signal_value DOUBLE,
                confidence DOUBLE,
                metadata STRING
            ) TIMESTAMP(timestamp) PARTITION BY DAY;
            """
        ]
        
        async with self._pool.acquire() as conn:
            for table_sql in tables:
                try:
                    await conn.execute(table_sql)
                except Exception as e:
                    # Table might already exist
                    self.logger.debug(f"Table creation note: {e}")
    
    async def insert_ohlcv(
        self,
        symbol: str,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        exchange: str = "default",
        vwap: Optional[float] = None,
        timeframe: str = "1m"
    ) -> None:
        """
        Insert single OHLCV bar.
        
        Args:
            symbol: Trading symbol
            timestamp: Bar timestamp
            open: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            exchange: Exchange name
            vwap: Volume-weighted average price (optional)
            timeframe: Timeframe (1m, 5m, 15m, etc.)
        """
        table_name = f"ohlcv_{timeframe}"
        
        query = f"""
            INSERT INTO {table_name} (
                timestamp, symbol, exchange, open, high, low, close, volume, vwap
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    query,
                    timestamp, symbol, exchange, open, high, low, close, volume,
                    vwap or close
                )
            except Exception as e:
                self.logger.error(f"Failed to insert OHLCV: {e}")
                raise QueryError(f"OHLCV insert failed: {e}")
    
    async def insert_ohlcv_bulk(
        self,
        data: List[Dict[str, Any]],
        timeframe: str = "1m"
    ) -> int:
        """
        Bulk insert OHLCV data.
        
        Args:
            data: List of OHLCV dictionaries
            timeframe: Timeframe (1m, 5m, 15m, etc.)
            
        Returns:
            Number of rows inserted
        """
        if not data:
            return 0
        
        table_name = f"ohlcv_{timeframe}"
        
        # Prepare values
        values = []
        for row in data:
            values.append((
                row['timestamp'],
                row['symbol'],
                row.get('exchange', 'default'),
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume'],
                row.get('vwap', row['close'])
            ))
        
        query = f"""
            INSERT INTO {table_name} (
                timestamp, symbol, exchange, open, high, low, close, volume, vwap
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        
        async with self._pool.acquire() as conn:
            try:
                await conn.executemany(query, values)
                self.logger.info(f"Inserted {len(values)} OHLCV rows into {table_name}")
                return len(values)
            except Exception as e:
                self.logger.error(f"Bulk insert failed: {e}")
                raise QueryError(f"Bulk OHLCV insert failed: {e}")
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol and time range.
        
        Args:
            symbol: Trading symbol
            start: Start timestamp
            end: End timestamp
            timeframe: Timeframe (1m, 5m, 15m, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        table_name = f"ohlcv_{timeframe}"
        
        query = f"""
            SELECT 
                timestamp, symbol, exchange, open, high, low, close, volume, vwap
            FROM {table_name}
            WHERE symbol = $1 
                AND timestamp >= $2 
                AND timestamp < $3
            ORDER BY timestamp ASC
        """
        
        async with self._pool.acquire() as conn:
            try:
                rows = await conn.fetch(query, symbol, start, end)
                
                if not rows:
                    return pd.DataFrame()
                
                df = pd.DataFrame(
                    [dict(row) for row in rows],
                    columns=['timestamp', 'symbol', 'exchange', 'open', 'high', 
                            'low', 'close', 'volume', 'vwap']
                )
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                return df
            
            except Exception as e:
                self.logger.error(f"Failed to fetch OHLCV: {e}")
                raise QueryError(f"OHLCV fetch failed: {e}")
    
    async def fetch_latest_ohlcv(
        self,
        symbol: str,
        limit: int = 100,
        timeframe: str = "1m"
    ) -> pd.DataFrame:
        """
        Fetch latest N OHLCV bars for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of bars to fetch
            timeframe: Timeframe
            
        Returns:
            DataFrame with latest OHLCV data
        """
        table_name = f"ohlcv_{timeframe}"
        
        query = f"""
            SELECT 
                timestamp, symbol, exchange, open, high, low, close, volume, vwap
            FROM {table_name}
            WHERE symbol = $1
            ORDER BY timestamp DESC
            LIMIT $2
        """
        
        async with self._pool.acquire() as conn:
            try:
                rows = await conn.fetch(query, symbol, limit)
                
                if not rows:
                    return pd.DataFrame()
                
                df = pd.DataFrame(
                    [dict(row) for row in rows],
                    columns=['timestamp', 'symbol', 'exchange', 'open', 'high',
                            'low', 'close', 'volume', 'vwap']
                )
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)  # Sort ascending
                
                return df
            
            except Exception as e:
                self.logger.error(f"Failed to fetch latest OHLCV: {e}")
                raise QueryError(f"Latest OHLCV fetch failed: {e}")
    
    async def insert_indicators(
        self,
        symbol: str,
        timestamp: datetime,
        indicators: Dict[str, float],
        timeframe: str = "1m"
    ) -> None:
        """
        Insert technical indicators.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            indicators: Dictionary of indicator values
            timeframe: Timeframe
        """
        table_name = f"indicators_{timeframe}"
        
        # Build dynamic query based on provided indicators
        columns = ['timestamp', 'symbol'] + list(indicators.keys())
        placeholders = ', '.join(f'${i+1}' for i in range(len(columns)))
        
        query = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({placeholders})
        """
        
        values = [timestamp, symbol] + list(indicators.values())
        
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(query, *values)
            except Exception as e:
                self.logger.error(f"Failed to insert indicators: {e}")
                raise QueryError(f"Indicators insert failed: {e}")
    
    async def fetch_ohlcv_with_indicators(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with indicators using ASOF JOIN.
        
        Args:
            symbol: Trading symbol
            start: Start timestamp
            end: End timestamp
            timeframe: Timeframe
            
        Returns:
            DataFrame with OHLCV and indicators
        """
        ohlcv_table = f"ohlcv_{timeframe}"
        indicators_table = f"indicators_{timeframe}"
        
        query = f"""
            SELECT 
                o.timestamp,
                o.symbol,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume,
                o.vwap,
                i.rsi,
                i.macd,
                i.macd_signal,
                i.macd_hist,
                i.bb_upper,
                i.bb_middle,
                i.bb_lower,
                i.atr,
                i.adx,
                i.ema_20,
                i.ema_50,
                i.ema_200
            FROM {ohlcv_table} o
            ASOF JOIN {indicators_table} i ON (o.symbol = i.symbol AND o.timestamp = i.timestamp)
            WHERE o.symbol = $1 
                AND o.timestamp >= $2 
                AND o.timestamp < $3
            ORDER BY o.timestamp ASC
        """
        
        async with self._pool.acquire() as conn:
            try:
                rows = await conn.fetch(query, symbol, start, end)
                
                if not rows:
                    return pd.DataFrame()
                
                df = pd.DataFrame([dict(row) for row in rows])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                return df
            
            except Exception as e:
                self.logger.error(f"Failed to fetch OHLCV with indicators: {e}")
                raise QueryError(f"OHLCV+indicators fetch failed: {e}")
    
    async def insert_signal(
        self,
        agent_id: str,
        symbol: str,
        timestamp: datetime,
        signal_type: str,
        signal_value: float,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Insert trading signal.
        
        Args:
            agent_id: Agent identifier
            symbol: Trading symbol
            timestamp: Signal timestamp
            signal_type: Signal type ('buy', 'sell', 'hold')
            signal_value: Signal strength (-1 to 1)
            confidence: Confidence level (0 to 1)
            metadata: Additional metadata (JSON)
        """
        import json
        
        query = """
            INSERT INTO signals (
                timestamp, agent_id, symbol, signal_type, signal_value, confidence, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        
        metadata_str = json.dumps(metadata) if metadata else None
        
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    query,
                    timestamp, agent_id, symbol, signal_type, signal_value,
                    confidence, metadata_str
                )
            except Exception as e:
                self.logger.error(f"Failed to insert signal: {e}")
                raise QueryError(f"Signal insert failed: {e}")
    
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
                    SELECT table_name, 
                           pg_total_relation_size(table_name::regclass) as size_bytes
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name LIKE 'ohlcv_%' OR table_name LIKE 'indicators_%'
                """
                
                tables = await conn.fetch(tables_query)
                
                return {
                    'connected': True,
                    'host': self.config.host,
                    'port': self.config.port,
                    'database': self.config.database,
                    'tables': [
                        {
                            'name': row['table_name'],
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

