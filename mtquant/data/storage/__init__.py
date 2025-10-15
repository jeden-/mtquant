"""
Data Storage Layer.

Provides database clients for different storage backends:
- QuestDB: Time-series data (OHLCV, indicators)
- PostgreSQL: Transactional data (orders, trades, positions)
- Redis: Hot data caching and replay buffers
"""

from mtquant.data.storage.questdb_client import QuestDBClient, QuestDBConfig
from mtquant.data.storage.postgresql_client import PostgreSQLClient, PostgreSQLConfig
from mtquant.data.storage.redis_client import RedisClient, RedisConfig

__version__ = "0.1.0"

__all__ = [
    "QuestDBClient",
    "QuestDBConfig",
    "PostgreSQLClient",
    "PostgreSQLConfig",
    "RedisClient",
    "RedisConfig",
]
