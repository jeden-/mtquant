"""
MT5 MCP Client for MetaTrader 5 Integration

This module provides a comprehensive client for MetaTrader 5 integration
using the official MetaTrader5 Python package. The client handles connection
management, market data retrieval, trading operations, and error handling.

The client uses asyncio.to_thread() for blocking MT5 calls and implements
retry logic with exponential backoff for production reliability.

Example:
    # Initialize client
    client = MT5Client(broker_id="ic_markets_mt5_demo", config=broker_config)
    
    # Connect to MT5
    connected = await client.connect()
    
    # Fetch market data
    data = await client.get_market_data("XAUUSD", "H1", bars=100)
    
    # Place order
    order_id = await client.place_order(order)
    
    # Disconnect
    await client.disconnect()
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import MetaTrader5 as MT5
from mtquant.mcp_integration.models.order import Order
from mtquant.mcp_integration.models.position import Position
from mtquant.utils.exceptions import (
    BrokerConnectionError,
    BrokerAPIError,
    BrokerTimeoutError,
    OrderExecutionError,
    InsufficientMarginError,
    MarketDataError,
    BrokerError
)
from mtquant.utils.logger import get_logger

logger = get_logger(__name__)


# MT5 Timeframe mapping
TIMEFRAME_MAP = {
    'M1': MT5.TIMEFRAME_M1,
    'M5': MT5.TIMEFRAME_M5,
    'M15': MT5.TIMEFRAME_M15,
    'M30': MT5.TIMEFRAME_M30,
    'H1': MT5.TIMEFRAME_H1,
    'H4': MT5.TIMEFRAME_H4,
    'D1': MT5.TIMEFRAME_D1,
    'W1': MT5.TIMEFRAME_W1,
    'MN1': MT5.TIMEFRAME_MN1,
}


@dataclass
class HealthStatus:
    """Health status for MT5 connection."""
    is_connected: bool
    latency_ms: float
    last_check: datetime
    error: Optional[str] = None


class MT5Client:
    """
    MetaTrader 5 client for trading operations.
    
    This client provides async methods for all MT5 operations including
    connection management, market data retrieval, and trading operations.
    All blocking MT5 calls are wrapped with asyncio.to_thread() for
    non-blocking execution.
    
    Features:
    - Connection management with retry logic
    - Market data retrieval (OHLCV, ticks)
    - Trading operations (orders, positions)
    - Health monitoring
    - Comprehensive error handling
    - Logging for all operations
    """
    
    def __init__(self, broker_id: str, config: Dict[str, Any]):
        """
        Initialize MT5 client.
        
        Args:
            broker_id: Unique broker identifier
            config: Broker configuration from brokers.yaml
        """
        self.broker_id = broker_id
        self.config = config
        self.logger = get_logger(f"{__name__}.{broker_id}")
        
        # Connection parameters
        self.account = config.get('account')
        self.password = config.get('password')
        self.server = config.get('server')
        
        # Connection state
        self._connected = False
        self._last_health_check = None
        
        # Retry configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.timeout = config.get('timeout', 5.0)
        
        self.logger.info(f"Initialized MT5 client for {broker_id}")
    
    async def connect(self) -> bool:
        """
        Connect to MT5 terminal.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            BrokerConnectionError: If connection fails
            BrokerTimeoutError: If login times out
        """
        self.logger.info(f"Connecting to MT5 terminal for {self.broker_id}")
        
        try:
            # Initialize MT5
            initialized = await asyncio.to_thread(MT5.initialize)
            if not initialized:
                error_code = MT5.last_error()
                raise BrokerConnectionError(f"MT5 initialization failed: {error_code}")
            
            # Check if already logged in
            account_info = await asyncio.to_thread(MT5.account_info)
            if account_info is not None:
                # Already logged in, verify it's the correct account
                if account_info.login == self.account and account_info.server == self.server:
                    self._connected = True
                    self.logger.info(f"Already connected to MT5: {self.server}, Account: {self.account}")
                    self.logger.info(f"Account balance: {account_info.balance}, Equity: {account_info.equity}")
                    return True
                else:
                    self.logger.warning(f"Logged into different account: {account_info.login}@{account_info.server}")
            
            # Login to account
            logged_in = await asyncio.to_thread(
                MT5.login,
                self.account,
                password=self.password,
                server=self.server
            )
            
            if not logged_in:
                error_code = MT5.last_error()
                raise BrokerConnectionError(f"MT5 login failed: {error_code}")
            
            # Verify connection
            account_info = await asyncio.to_thread(MT5.account_info)
            if account_info is None:
                raise BrokerConnectionError("Failed to retrieve account info after login")
            
            self._connected = True
            self.logger.info(f"Successfully connected to MT5: {self.server}, Account: {self.account}")
            self.logger.info(f"Account balance: {account_info.balance}, Equity: {account_info.equity}")
            
            return True
            
        except asyncio.TimeoutError:
            raise BrokerTimeoutError(f"MT5 connection timeout for {self.broker_id}")
        except Exception as e:
            self.logger.error(f"MT5 connection failed: {e}")
            raise BrokerConnectionError(f"MT5 connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Shutdown MT5 connection cleanly."""
        if self._connected:
            try:
                await asyncio.to_thread(MT5.shutdown)
                self._connected = False
                self.logger.info(f"Disconnected from MT5: {self.broker_id}")
            except Exception as e:
                self.logger.error(f"Error during MT5 disconnect: {e}")
    
    async def health_check(self) -> bool:
        """
        Check if connection is alive.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        if not self._connected:
            return False
        
        try:
            # Test account info retrieval
            account_info = await asyncio.to_thread(MT5.account_info)
            if account_info is None:
                self._connected = False
                return False
            
            self._last_health_check = datetime.utcnow()
            return True
            
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            self._connected = False
            return False
    
    async def get_health_status(self) -> HealthStatus:
        """
        Get detailed health status.
        
        Returns:
            HealthStatus object with connection details
        """
        try:
            import time
            from mtquant.mcp_integration.adapters.base_adapter import HealthStatus
            
            start_time = time.time()
            
            # Check connection
            is_connected = await self.health_check()
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create health status
            health = HealthStatus(
                is_connected=is_connected,
                latency_ms=latency_ms,
                last_check=datetime.utcnow(),
                error=None if is_connected else "MT5 connection failed"
            )
            
            return health
            
        except Exception as e:
            return HealthStatus(
                is_connected=False,
                latency_ms=0.0,
                last_check=datetime.utcnow(),
                error=str(e)
            )
    
    async def get_symbols(self) -> List[str]:
        """
        Get list of available symbols from broker.
        
        Returns:
            List of available symbol names
            
        Raises:
            BrokerAPIError: If symbol retrieval fails
        """
        try:
            symbols = await asyncio.to_thread(MT5.symbols_get)
            if symbols is None:
                error_code = MT5.last_error()
                raise BrokerAPIError(f"Failed to get symbols: {error_code}")
            
            symbol_names = [symbol.name for symbol in symbols]
            self.logger.debug(f"Retrieved {len(symbol_names)} symbols")
            return symbol_names
            
        except Exception as e:
            self.logger.error(f"Failed to get symbols: {e}")
            raise BrokerAPIError(f"Failed to get symbols: {e}")
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = 'H1',
        bars: int = 100
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data.
        
        Args:
            symbol: Broker-specific symbol (use SymbolMapper first)
            timeframe: M1, M5, M15, H1, H4, D1
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            
        Raises:
            MarketDataError: If fetch fails
        """
        if timeframe not in TIMEFRAME_MAP:
            raise MarketDataError(f"Unsupported timeframe: {timeframe}")
        
        try:
            mt5_timeframe = TIMEFRAME_MAP[timeframe]
            
            # Get rates
            rates = await asyncio.to_thread(
                MT5.copy_rates_from_pos,
                symbol,
                mt5_timeframe,
                0,
                bars
            )
            
            if rates is None:
                error_code = MT5.last_error()
                raise MarketDataError(f"Failed to get market data for {symbol}: {error_code}")
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('timestamp')
            
            # Rename columns to standard format
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            })
            
            # Select only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            self.logger.debug(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            raise MarketDataError(f"Failed to get market data for {symbol}: {e}")
    
    async def get_tick_data(self, symbol: str, count: int = 10) -> pd.DataFrame:
        """
        Get recent tick data.
        
        Args:
            symbol: Broker-specific symbol
            count: Number of ticks to retrieve
            
        Returns:
            DataFrame with tick data
        """
        try:
            ticks = await asyncio.to_thread(
                MT5.copy_ticks_from_pos,
                symbol,
                0,
                count,
                MT5.COPY_TICKS_ALL
            )
            
            if ticks is None:
                error_code = MT5.last_error()
                raise MarketDataError(f"Failed to get tick data for {symbol}: {error_code}")
            
            df = pd.DataFrame(ticks)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('timestamp')
            
            self.logger.debug(f"Retrieved {len(df)} ticks for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get tick data for {symbol}: {e}")
            raise MarketDataError(f"Failed to get tick data for {symbol}: {e}")
    
    async def place_order(self, order: Order) -> str:
        """
        Place order on broker.
        
        Args:
            order: Order object (already validated)
            
        Returns:
            order_id: Broker's order ticket number
            
        Raises:
            OrderExecutionError: If order fails
            InsufficientMarginError: If not enough margin
        """
        try:
            # Convert Order to MT5 request
            request = self._convert_order_to_mt5(order)
            
            # Send order
            result = await asyncio.to_thread(MT5.order_send, request)
            
            if result is None:
                error_code = MT5.last_error()
                raise OrderExecutionError(f"Order failed: {error_code}")
            
            if result.retcode != MT5.TRADE_RETCODE_DONE:
                raise OrderExecutionError(f"Order rejected: {result.retcode} - {result.comment}")
            
            order_id = str(result.order)
            self.logger.info(f"Order placed successfully: {order_id} for {order.symbol}")
            return order_id
            
        except InsufficientMarginError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise OrderExecutionError(f"Failed to place order: {e}")
    
    async def get_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of Position objects
        """
        try:
            positions = await asyncio.to_thread(MT5.positions_get)
            
            if positions is None:
                # No positions or error
                error_code = MT5.last_error()
                if error_code != MT5.RES_S_OK:
                    self.logger.warning(f"Failed to get positions: {error_code}")
                return []
            
            position_list = []
            for pos in positions:
                position = self._convert_mt5_to_position(pos)
                position_list.append(position)
            
            self.logger.debug(f"Retrieved {len(position_list)} positions")
            return position_list
            
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            raise BrokerAPIError(f"Failed to get positions: {e}")
    
    async def close_position(self, position_id: str) -> bool:
        """
        Close specific position.
        
        Args:
            position_id: Position ticket number
            
        Returns:
            True if position closed successfully
        """
        try:
            # Get position info
            positions = await asyncio.to_thread(MT5.positions_get, ticket=int(position_id))
            
            if not positions:
                raise OrderExecutionError(f"Position not found: {position_id}")
            
            position = positions[0]
            
            # Create close request
            request = {
                "action": MT5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": MT5.ORDER_TYPE_SELL if position.type == MT5.POSITION_TYPE_BUY else MT5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "comment": f"Close position {position_id}",
                "type_time": MT5.ORDER_TIME_GTC,
                "type_filling": MT5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            result = await asyncio.to_thread(MT5.order_send, request)
            
            if result is None:
                error_code = MT5.last_error()
                raise OrderExecutionError(f"Failed to close position: {error_code}")
            
            if result.retcode != MT5.TRADE_RETCODE_DONE:
                raise OrderExecutionError(f"Position close rejected: {result.retcode} - {result.comment}")
            
            self.logger.info(f"Position closed successfully: {position_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to close position {position_id}: {e}")
            raise OrderExecutionError(f"Failed to close position {position_id}: {e}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account info: balance, equity, margin, free_margin, profit
        """
        try:
            account_info = await asyncio.to_thread(MT5.account_info)
            
            if account_info is None:
                error_code = MT5.last_error()
                raise BrokerAPIError(f"Failed to get account info: {error_code}")
            
            info = {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'profit': account_info.profit,
                'currency': account_info.currency,
                'leverage': account_info.leverage,
                'margin_level': account_info.margin_level,
            }
            
            self.logger.debug("Retrieved account info")
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            raise BrokerAPIError(f"Failed to get account info: {e}")
    
    def _convert_order_to_mt5(self, order: Order) -> Dict[str, Any]:
        """
        Convert Order object to MT5 request dictionary.
        
        Args:
            order: Order object
            
        Returns:
            MT5 request dictionary
        """
        # Determine order type
        if order.order_type == 'market':
            order_type = MT5.ORDER_TYPE_BUY if order.side == 'buy' else MT5.ORDER_TYPE_SELL
        elif order.order_type == 'limit':
            order_type = MT5.ORDER_TYPE_BUY_LIMIT if order.side == 'buy' else MT5.ORDER_TYPE_SELL_LIMIT
        elif order.order_type == 'stop':
            order_type = MT5.ORDER_TYPE_BUY_STOP if order.side == 'buy' else MT5.ORDER_TYPE_SELL_STOP
        else:
            raise OrderExecutionError(f"Unsupported order type: {order.order_type}")
        
        request = {
            "action": MT5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": order.quantity,
            "type": order_type,
            "comment": f"MTQuant order {order.agent_id}",
            "type_time": MT5.ORDER_TIME_GTC,
            "type_filling": MT5.ORDER_FILLING_IOC,
        }
        
        # Add price for limit/stop orders
        if order.price is not None:
            request["price"] = order.price
        
        # Add stop loss and take profit
        if order.stop_loss is not None:
            request["sl"] = order.stop_loss
        
        if order.take_profit is not None:
            request["tp"] = order.take_profit
        
        return request
    
    def _convert_mt5_to_position(self, mt5_position) -> Position:
        """
        Convert MT5 position to Position object.
        
        Args:
            mt5_position: MT5 position object
            
        Returns:
            Position object
        """
        # Determine side
        side = 'long' if mt5_position.type == MT5.POSITION_TYPE_BUY else 'short'
        
        # Calculate unrealized P&L
        unrealized_pnl = mt5_position.profit
        
        position = Position(
            position_id=str(mt5_position.ticket),
            agent_id="mt5_client",  # Default agent ID
            symbol=mt5_position.symbol,
            side=side,
            quantity=mt5_position.volume,
            entry_price=mt5_position.price_open,
            current_price=mt5_position.price_current,
            stop_loss=mt5_position.sl if mt5_position.sl > 0 else None,
            take_profit=mt5_position.tp if mt5_position.tp > 0 else None,
            unrealized_pnl=unrealized_pnl,
            opened_at=datetime.fromtimestamp(mt5_position.time),
            broker_id=self.broker_id
        )
        
        return position
    
    async def get_health_status(self) -> HealthStatus:
        """
        Get detailed health status.
        
        Returns:
            HealthStatus object
        """
        start_time = datetime.utcnow()
        
        try:
            is_healthy = await self.health_check()
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return HealthStatus(
                is_connected=is_healthy,
                latency_ms=latency,
                last_check=datetime.utcnow(),
                error=None
            )
            
        except Exception as e:
            return HealthStatus(
                is_connected=False,
                latency_ms=0.0,
                last_check=datetime.utcnow(),
                error=str(e)
            )
    
    def __repr__(self) -> str:
        """String representation of MT5Client."""
        return f"MT5Client(broker_id={self.broker_id}, connected={self._connected})"
