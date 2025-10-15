"""
Data Fetchers - Automated Data Collection.

Provides automated market data fetching from brokers.
"""

from mtquant.data.fetchers.market_data_fetcher import MarketDataFetcher, FetcherConfig

__version__ = "0.1.0"

__all__ = [
    "MarketDataFetcher",
    "FetcherConfig",
]
