"""
MT4 MCP Client for MetaTrader 4 Integration via HTTP MCP Server

This module provides a comprehensive client for MetaTrader 4 integration
using the Model Context Protocol (MCP) via HTTP communication with
a Node.js MCP server.

Architecture:
MT4MCPClient → HTTP Requests → MT4 MCP Server (Node.js) → File I/O → MT4 Expert Advisor → MT4 Terminal

Example:
    # Initialize client
    client = MT4MCPClient(broker_id="ic_markets_mt4_demo", config=broker_config)
    
    # Connect to MCP server
    connected = await client.connect()
    
    # Fetch market data
    data = await client.get_market_data("EURUSD", "H1", bars=100)
    
    # Place order
    order_id = await client.place_order(order)
    
    # Disconnect
    await client.disconnect()
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import httpx

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


# MT4 Timeframe mapping (for HTTP API)
TIMEFRAME_MAP = {
    'M1': 1,
    'M5': 5,
    'M15': 15,
    'M30': 30,
    'H1': 60,
    'H4': 240,
    'D1': 1440,
    'W1': 10080,
    'MN1': 43200,
}


@dataclass
class HealthStatus:
    """Health status for MT4 MCP connection."""
    is_connected: bool
    latency_ms: float
    last_check: datetime
    error: Optional[str] = None


class MT4MCPClient:
    """
    MetaTrader 4 MCP client for trading operations.
    
    This client provides async methods for all MT4 operations including
    connection management, market data retrieval, and trading operations.
    All operations go through HTTP API provided by the Node.js MCP server.
    
    Features:
    - HTTP-based MCP communication
    - Connection management with retry logic
    - Market data retrieval (OHLCV, ticks)
    - Trading operations (orders, positions)
    - Health monitoring
    - Comprehensive error handling
    - Logging for all operations
    
    Args:
        broker_id: Unique broker identifier
        config: Broker configuration dictionary containing:
            - mcp_endpoint: HTTP endpoint for MCP server (default: http://localhost:3000)
            - account: MT4 account number
            - password: MT4 account password
            - server: MT4 server name
    """
    
    def __init__(self, broker_id: str, config: Dict[str, Any]):
        self.broker_id = broker_id
        self.config = config
        self.logger = get_logger(__name__)
        self.base_url = config.get('mcp_endpoint', 'http://localhost:3000')
        self.client = httpx.AsyncClient(timeout=10.0)
        self._connected = False
        self._last_health_check: Optional[datetime] = None
        
        self.logger.info(f"MT4MCPClient initialized for broker: {broker_id}")
    
    async def connect(self) -> bool:
        """
        Connect to MT4 via HTTP MCP server.
        
        Steps:
        1. POST /initialize with credentials
        2. Verify response
        3. Wait for MT4 EA to confirm connection
        
        Returns:
            True if connected successfully
            
        Raises:
            BrokerConnectionError: If connection fails
            BrokerTimeoutError: If connection times out
        """
        try:
            self.logger.info(f"Connecting to MT4 MCP server for {self.broker_id}")
            
            response = await self.client.post(
                f"{self.base_url}/initialize",
                json={
                    "account": self.config['account'],
                    "password": self.config['password'],
                    "server": self.config['server']
                }
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('status') == 'success':
                self._connected = True
                self.logger.info(f"Connected to MT4: {self.broker_id}")
                return True
            else:
                raise BrokerConnectionError(f"MT4 init failed: {result.get('message')}")
                
        except httpx.HTTPError as e:
            raise BrokerConnectionError(f"HTTP error: {e}")
        except Exception as e:
            self.logger.exception("MT4 MCP connection error")
            raise BrokerConnectionError(f"Failed to connect: {e}")
    
    async def disconnect(self) -> None:
        """Shutdown MT4 connection via HTTP MCP server."""
        try:
            await self.client.post(f"{self.base_url}/shutdown")
            self.logger.info(f"Disconnected from MT4: {self.broker_id}")
        except Exception as e:
            self.logger.warning(f"Disconnect error (non-fatal): {e}")
        finally:
            self._connected = False
            await self.client.aclose()
    
    async def health_check(self) -> bool:
        """
        Check if HTTP MCP connection is alive.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        if not self._connected:
            return False
        
        try:
            # Try to get account info as health check
            response = await self.client.get(f"{self.base_url}/account")
            response.raise_for_status()
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
                error=None if is_connected else "MT4 MCP connection failed"
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
        Get list of available symbols from broker via HTTP MCP server.
        
        Returns:
            List of available symbol names
            
        Raises:
            BrokerConnectionError: If not connected
            MarketDataError: If fetch fails
        """
        if not self._connected:
            raise BrokerConnectionError("Not connected")
        
        try:
            response = await self.client.get(f"{self.base_url}/symbols")
            response.raise_for_status()
            
            result = response.json()
            symbols = result.get('symbols', [])
            
            self.logger.debug(f"Retrieved {len(symbols)} symbols")
            return symbols
            
        except Exception as e:
            raise MarketDataError(f"Failed to get symbols: {e}")
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = 'H1',
        bars: int = 100
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data via HTTP GET /market_data/<symbol>.
        
        Args:
            symbol: Broker-specific symbol (pre-mapped)
            timeframe: M1, M5, M15, H1, H4, D1
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            
        Raises:
            BrokerConnectionError: If not connected
            MarketDataError: If fetch fails
            ValueError: If invalid timeframe
        """
        if not self._connected:
            raise BrokerConnectionError("Not connected")
        
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        try:
            response = await self.client.get(
                f"{self.base_url}/market_data/{symbol}",
                params={'timeframe': timeframe, 'bars': bars}
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data['bars'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            self.logger.debug(f"Fetched {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            raise MarketDataError(f"Failed to fetch market data: {e}")
    
    async def get_tick_data(self, symbol: str, count: int = 10) -> pd.DataFrame:
        """
        Get recent tick data via HTTP GET /tick_data/<symbol>.
        
        Args:
            symbol: Symbol name
            count: Number of ticks to fetch
            
        Returns:
            DataFrame with tick data
            
        Raises:
            BrokerConnectionError: If not connected
            MarketDataError: If fetch fails
        """
        if not self._connected:
            raise BrokerConnectionError("Not connected")
        
        try:
            response = await self.client.get(
                f"{self.base_url}/tick_data/{symbol}",
                params={'count': count}
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data['ticks'])
            
            return df
            
        except Exception as e:
            raise MarketDataError(f"Failed to get tick data: {e}")
    
    async def place_order(self, order: Order) -> str:
        """
        Place order via HTTP POST /order.
        
        Args:
            order: Order object (already validated)
            
        Returns:
            order_id: Broker's order ticket number (as string)
            
        Raises:
            BrokerConnectionError: If not connected
            OrderExecutionError: If order fails
        """
        if not self._connected:
            raise BrokerConnectionError("Not connected")
        
        try:
            # Convert Order to MT4 request format
            mt4_request = {
                "symbol": order.symbol,
                "volume": order.quantity,
                "type": "buy" if order.side == "buy" else "sell",
                "price": order.price if order.price else 0.0,  # 0 for market orders
                "sl": order.stop_loss if order.stop_loss else 0.0,
                "tp": order.take_profit if order.take_profit else 0.0,
                "deviation": 20,  # Max deviation in points
                "magic": 234000,  # Magic number for identification
                "comment": f"MTQuant-{order.agent_id}"
            }
            
            response = await self.client.post(
                f"{self.base_url}/order",
                json=mt4_request
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('status') == 'success':
                order_id = str(result.get('order_id'))
                self.logger.info(f"Order placed: {order_id} - {order.symbol} {order.side} {order.quantity}")
                return order_id
            else:
                error_msg = result.get('message', 'Unknown error')
                raise OrderExecutionError(f"Order rejected: {error_msg}")
                
        except Exception as e:
            self.logger.exception("Order placement error")
            raise OrderExecutionError(f"Failed to place order: {e}")
    
    async def get_positions(self) -> List[Position]:
        """
        Get all open positions via HTTP GET /positions.
        
        Returns:
            List of Position objects
            
        Raises:
            BrokerConnectionError: If not connected
        """
        if not self._connected:
            raise BrokerConnectionError("Not connected")
        
        try:
            response = await self.client.get(f"{self.base_url}/positions")
            response.raise_for_status()
            
            result = response.json()
            positions_data = result.get('positions', [])
            
            # Convert to Position objects
            positions = []
            for pos in positions_data:
                position = Position(
                    position_id=str(pos['ticket']),
                    agent_id="unknown",  # Will be mapped from comment field
                    symbol=pos['symbol'],
                    side='long' if pos['type'] == 0 else 'short',
                    quantity=pos['volume'],
                    entry_price=pos['price_open'],
                    current_price=pos['price_current'],
                    stop_loss=pos.get('sl'),
                    take_profit=pos.get('tp'),
                    unrealized_pnl=pos['profit'],
                    opened_at=pd.to_datetime(pos['time'], unit='s'),
                    broker_id=self.broker_id
                )
                positions.append(position)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []
    
    async def close_position(self, position_id: str) -> bool:
        """
        Close specific position via HTTP POST /close_position.
        
        Args:
            position_id: Position ticket ID
            
        Returns:
            True if position closed successfully
            
        Raises:
            BrokerConnectionError: If not connected
            OrderExecutionError: If close fails
        """
        if not self._connected:
            raise BrokerConnectionError("Not connected")
        
        try:
            response = await self.client.post(
                f"{self.base_url}/close_position",
                json={"position_id": position_id}
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('status') == 'success':
                self.logger.info(f"Position closed: {position_id}")
                return True
            else:
                error_msg = result.get('message', 'Unknown error')
                raise OrderExecutionError(f"Close position failed: {error_msg}")
                
        except Exception as e:
            self.logger.exception("Close position error")
            raise OrderExecutionError(f"Failed to close position: {e}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information via HTTP GET /account.
        
        Returns:
            Dictionary with account information:
            - balance: Account balance
            - equity: Account equity
            - margin: Used margin
            - free_margin: Free margin
            - profit: Current profit/loss
            - leverage: Account leverage
            
        Raises:
            BrokerConnectionError: If not connected
        """
        if not self._connected:
            raise BrokerConnectionError("Not connected")
        
        try:
            response = await self.client.get(f"{self.base_url}/account")
            response.raise_for_status()
            
            account_data = response.json()
            
            return {
                'balance': account_data['balance'],
                'equity': account_data['equity'],
                'margin': account_data['margin'],
                'free_margin': account_data['margin_free'],
                'profit': account_data['profit'],
                'leverage': account_data['leverage']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return {}
    
    def __repr__(self) -> str:
        """String representation of the client."""
        status = "connected" if self._connected else "disconnected"
        return f"MT4MCPClient(broker_id='{self.broker_id}', status='{status}')"
