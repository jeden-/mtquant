"""
Data Storage

Database clients and storage utilities for time-series and transactional data.
Provides optimized interfaces for different data types and access patterns.

Storage Types:
- QuestDB: Time-series data (OHLCV, indicators, ticks)
- PostgreSQL: Transactional data (orders, trades, positions)
- Redis: Hot data caching (latest prices, positions, session data)

Features:
- Connection pooling and optimization
- Async/await support for high performance
- Data compression and archiving
- Backup and recovery utilities
- Query optimization and indexing
"""

__version__ = "0.1.0"

# Storage classes will be imported here when implemented
__all__ = []
