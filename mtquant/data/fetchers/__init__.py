"""
Data Fetchers

Data source integrations for retrieving market data from various sources:
- Broker APIs (MT4/MT5, OANDA, Interactive Brokers)
- Public APIs (Alpha Vantage, Yahoo Finance, Crypto exchanges)
- File sources (CSV, Parquet, HDF5)
- Database sources (QuestDB, PostgreSQL)

All fetchers provide unified interfaces and handle:
- Rate limiting and API quotas
- Error handling and retries
- Data validation and normalization
- Caching and optimization
"""

__version__ = "0.1.0"

# Fetcher classes will be imported here when implemented
__all__ = []
