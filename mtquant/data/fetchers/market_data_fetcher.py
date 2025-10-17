"""
Market Data Fetcher - Automated Data Collection.

Fetches market data from brokers and stores it in QuestDB.
Supports multiple brokers and symbols with configurable intervals.

Author: MTQuant Development Team
Date: October 15, 2025
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
from dataclasses import dataclass

from mtquant.utils.logger import get_logger
from mtquant.utils.exceptions import MarketDataError, BrokerError
from mtquant.mcp_integration.managers.broker_manager import BrokerManager
from mtquant.data.storage.questdb_client import QuestDBClient


logger = get_logger(__name__)


@dataclass
class FetcherConfig:
    """Market data fetcher configuration."""
    symbols: List[str]
    timeframes: List[str] = None  # ['1m', '5m', '15m', '1h']
    fetch_interval: int = 60  # seconds
    lookback_days: int = 1
    enabled: bool = True
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1m']


class MarketDataFetcher:
    """
    Automated market data fetcher.
    
    Fetches OHLCV data from brokers at regular intervals
    and stores it in QuestDB for later use by agents.
    
    Features:
    - Multi-symbol support
    - Multi-timeframe support
    - Automatic retry on failures
    - Gap detection and backfilling
    - Health monitoring
    
    Example:
        >>> config = FetcherConfig(
        ...     symbols=['XAUUSD', 'EURUSD'],
        ...     timeframes=['1m', '5m'],
        ...     fetch_interval=60
        ... )
        >>> fetcher = MarketDataFetcher(config, broker_manager, questdb_client)
        >>> await fetcher.start()
    """
    
    def __init__(
        self,
        config: FetcherConfig,
        broker_manager: BrokerManager,
        questdb_client: QuestDBClient
    ):
        """
        Initialize market data fetcher.
        
        Args:
            config: Fetcher configuration
            broker_manager: Broker manager instance
            questdb_client: QuestDB client instance
        """
        self.config = config
        self.broker_manager = broker_manager
        self.questdb_client = questdb_client
        self.logger = get_logger(__name__)
        
        self._running = False
        self._fetcher_task: Optional[asyncio.Task] = None
        self._fetch_count = 0
        self._error_count = 0
        self._last_fetch_time: Dict[str, datetime] = {}
    
    async def start(self) -> None:
        """Start the fetcher."""
        if self._running:
            self.logger.warning("Fetcher already running")
            return
        
        if not self.config.enabled:
            self.logger.info("Fetcher is disabled")
            return
        
        self._running = True
        self._fetcher_task = asyncio.create_task(self._run_fetcher())
        self.logger.info(
            f"Market data fetcher started for {len(self.config.symbols)} symbols"
        )
    
    async def stop(self) -> None:
        """Stop the fetcher."""
        if not self._running:
            return
        
        self._running = False
        
        if self._fetcher_task:
            self._fetcher_task.cancel()
            try:
                await self._fetcher_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Market data fetcher stopped")
    
    async def _run_fetcher(self) -> None:
        """Main fetcher loop."""
        # Initial backfill
        await self._backfill_data()
        
        # Continuous fetching
        while self._running:
            try:
                await self._fetch_all_data()
                
                # Sleep until next fetch
                await asyncio.sleep(self.config.fetch_interval)
            
            except Exception as e:
                self.logger.error(f"Fetcher error: {e}", exc_info=True)
                self._error_count += 1
                await asyncio.sleep(5)  # Wait before retry
    
    async def _backfill_data(self) -> None:
        """Backfill historical data."""
        self.logger.info(
            f"Backfilling {self.config.lookback_days} days of historical data"
        )
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.config.lookback_days)
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                try:
                    await self._fetch_and_store(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    self.logger.info(
                        f"Backfilled {symbol} {timeframe} data"
                    )
                
                except Exception as e:
                    self.logger.error(
                        f"Failed to backfill {symbol} {timeframe}: {e}"
                    )
        
        self.logger.info("Backfill completed")
    
    async def _fetch_all_data(self) -> None:
        """Fetch data for all symbols and timeframes."""
        fetch_tasks = []
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                task = self._fetch_and_store_latest(symbol, timeframe)
                fetch_tasks.append(task)
        
        # Execute all fetches in parallel
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Count successes and failures
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = len(results) - successes
        
        self._fetch_count += successes
        self._error_count += failures
        
        if failures > 0:
            self.logger.warning(
                f"Fetch completed: {successes} successes, {failures} failures"
            )
        else:
            self.logger.debug(f"Fetch completed: {successes} symbols fetched")
    
    async def _fetch_and_store_latest(
        self,
        symbol: str,
        timeframe: str
    ) -> None:
        """
        Fetch and store latest data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, etc.)
        """
        # Calculate time range
        now = datetime.now()
        
        # Get last fetch time for this symbol
        last_fetch = self._last_fetch_time.get(f"{symbol}:{timeframe}")
        
        if last_fetch:
            start_time = last_fetch
        else:
            # First fetch - get last hour
            start_time = now - timedelta(hours=1)
        
        end_time = now
        
        await self._fetch_and_store(symbol, timeframe, start_time, end_time)
        
        # Update last fetch time
        self._last_fetch_time[f"{symbol}:{timeframe}"] = now
    
    async def _fetch_and_store(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> None:
        """
        Fetch data from broker and store in QuestDB.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_time: Start time
            end_time: End time
        """
        try:
            # Fetch from broker
            df = await self._fetch_from_broker(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            if df.empty:
                self.logger.debug(f"No new data for {symbol} {timeframe}")
                return
            
            # Store in QuestDB
            await self._store_in_questdb(df, symbol, timeframe)
            
            self.logger.debug(
                f"Stored {len(df)} bars for {symbol} {timeframe}"
            )
        
        except Exception as e:
            self.logger.error(
                f"Failed to fetch/store {symbol} {timeframe}: {e}"
            )
            raise
    
    async def _fetch_from_broker(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetch data from broker.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_time: Start time
            end_time: End time
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Get broker for this symbol
            broker_id = self.broker_manager.get_broker_for_symbol(symbol)
            adapter = self.broker_manager.get_adapter(broker_id)
            
            # Fetch market data
            df = await adapter.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start=start_time,
                end=end_time
            )
            
            return df
        
        except Exception as e:
            self.logger.error(f"Broker fetch failed for {symbol}: {e}")
            raise MarketDataError(f"Failed to fetch {symbol} from broker: {e}")
    
    async def _store_in_questdb(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> None:
        """
        Store data in QuestDB.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
        """
        # Prepare data for bulk insert
        data = []
        
        for idx, row in df.iterrows():
            data.append({
                'timestamp': idx if isinstance(idx, datetime) else datetime.fromisoformat(str(idx)),
                'symbol': symbol,
                'exchange': 'default',
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0)),
                'vwap': float(row.get('vwap', row['close']))
            })
        
        try:
            await self.questdb_client.insert_ohlcv_bulk(data, timeframe=timeframe)
        
        except Exception as e:
            self.logger.error(f"QuestDB insert failed: {e}")
            raise MarketDataError(f"Failed to store data in QuestDB: {e}")
    
    async def fetch_on_demand(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetch data on demand (not part of regular schedule).
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_time: Start time
            end_time: End time
            
        Returns:
            DataFrame with OHLCV data
        """
        self.logger.info(
            f"On-demand fetch: {symbol} {timeframe} "
            f"from {start_time} to {end_time}"
        )
        
        # Fetch from broker
        df = await self._fetch_from_broker(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Store in QuestDB
        if not df.empty:
            await self._store_in_questdb(df, symbol, timeframe)
        
        return df
    
    async def detect_gaps(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[tuple]:
        """
        Detect gaps in stored data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_time: Start time
            end_time: End time
            
        Returns:
            List of (gap_start, gap_end) tuples
        """
        # Fetch from QuestDB
        df = await self.questdb_client.fetch_ohlcv(
            symbol=symbol,
            start=start_time,
            end=end_time,
            timeframe=timeframe
        )
        
        if df.empty:
            return [(start_time, end_time)]
        
        # Calculate expected interval
        interval_map = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }
        
        expected_interval = interval_map.get(timeframe, timedelta(minutes=1))
        
        # Detect gaps
        gaps = []
        timestamps = df.index.tolist()
        
        for i in range(len(timestamps) - 1):
            current = timestamps[i]
            next_ts = timestamps[i + 1]
            
            expected_next = current + expected_interval
            
            # Allow 10% tolerance
            tolerance = expected_interval * 0.1
            
            if next_ts - expected_next > tolerance:
                # Gap detected
                gaps.append((current, next_ts))
        
        if gaps:
            self.logger.info(
                f"Detected {len(gaps)} gaps in {symbol} {timeframe} data"
            )
        
        return gaps
    
    async def backfill_gaps(
        self,
        symbol: str,
        timeframe: str,
        gaps: List[tuple]
    ) -> None:
        """
        Backfill detected gaps.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            gaps: List of (gap_start, gap_end) tuples
        """
        self.logger.info(f"Backfilling {len(gaps)} gaps for {symbol} {timeframe}")
        
        for gap_start, gap_end in gaps:
            try:
                await self._fetch_and_store(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=gap_start,
                    end_time=gap_end
                )
                
                self.logger.info(
                    f"Backfilled gap: {gap_start} to {gap_end}"
                )
            
            except Exception as e:
                self.logger.error(
                    f"Failed to backfill gap {gap_start} to {gap_end}: {e}"
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get fetcher statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'running': self._running,
            'enabled': self.config.enabled,
            'symbols': self.config.symbols,
            'timeframes': self.config.timeframes,
            'fetch_interval': self.config.fetch_interval,
            'total_fetches': self._fetch_count,
            'total_errors': self._error_count,
            'error_rate': self._error_count / max(self._fetch_count, 1),
            'last_fetch_times': {
                k: v.isoformat() for k, v in self._last_fetch_time.items()
            }
        }
    
    @property
    def is_running(self) -> bool:
        """Check if fetcher is running."""
        return self._running



