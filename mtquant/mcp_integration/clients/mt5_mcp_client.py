"""
MT5 MCP Client for MetaTrader 5 Integration via MCP Protocol

This module provides a comprehensive client for MetaTrader 5 integration
using the Model Context Protocol (MCP). The client communicates with
mcp-metatrader5-server via stdio communication.

CRITICAL: This client does NOT import MetaTrader5 directly - all operations
go through MCP tools provided by the MCP server.

Architecture:
MT5MCPClient → MCP Protocol (stdio) → MCP Server Process → MetaTrader5 Package → MT5 Terminal

Example:
    # Initialize client
    client = MT5MCPClient(broker_id="ic_markets_mt5_demo", config=broker_config)
    
    # Connect to MCP server
    connected = await client.connect()
    
    # Fetch market data
    data = await client.get_market_data("XAUUSD", "H1", bars=100)
    
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

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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


# MT5 Timeframe mapping (for MCP server)
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
    """Health status for MT5 MCP connection."""
    is_connected: bool
    latency_ms: float
    last_check: datetime
    error: Optional[str] = None


class MT5MCPClient:
    """
    MetaTrader 5 MCP client for trading operations.
    
    This client provides async methods for all MT5 operations including
    connection management, market data retrieval, and trading operations.
    All operations go through MCP tools provided by the MCP server.
    
    Features:
    - MCP protocol communication (stdio)
    - Connection management with retry logic
    - Market data retrieval (OHLCV, ticks)
    - Trading operations (orders, positions)
    - Health monitoring
    - Comprehensive error handling
    - Logging for all operations
    
    Args:
        broker_id: Unique broker identifier
        config: Broker configuration dictionary containing:
            - mcp_server_path: Path to MCP server directory
            - account: MT5 account number
            - password: MT5 account password
            - server: MT5 server name
    """
    
    def __init__(self, broker_id: str, config: Dict[str, Any]):
        self.broker_id = broker_id
        self.config = config
        self.logger = get_logger(__name__)
        self.session: Optional[ClientSession] = None
        self._connected = False
        self._last_health_check: Optional[datetime] = None
        
        # MCP server parameters
        # Use global Python 3.11 directly
        # Note: env variables will be passed during login call instead
        self.server_params = StdioServerParameters(
            command="py",
            args=[
                "-3.11",
                "-c",
                f"import sys; sys.path.insert(0, r'{config['mcp_server_path']}/src'); from mcp_mt5 import main; main()"
            ]
        )
        
        self.logger.info(f"MT5MCPClient initialized for broker: {broker_id}")
    
    async def connect(self) -> bool:
        """
        Connect to MT5 terminal via MCP server.
        
        Steps:
        1. Start MCP server process via stdio
        2. Call 'initialize' tool
        3. Call 'login' tool with credentials
        4. Verify connection successful
        
        Returns:
            True if connected successfully
            
        Raises:
            BrokerConnectionError: If connection fails
            BrokerTimeoutError: If login times out
        """
        try:
            self.logger.info(f"Connecting to MT5 MCP server for {self.broker_id}")
            
            # Start MCP server process and keep it alive
            self._stdio_client = stdio_client(self.server_params)
            self._read, self._write = await self._stdio_client.__aenter__()
            
            self.session = ClientSession(self._read, self._write)
            await self.session.__aenter__()
            
            # Initialize MCP session
            await self.session.initialize()
            
            # Initialize MT5
            init_result = await asyncio.wait_for(
                self.session.call_tool("initialize", arguments={}),
                timeout=10.0
            )
            self.logger.info(f"MT5 initialize: {init_result.content[0].text}")
            
            # Login
            login_result = await asyncio.wait_for(
                self.session.call_tool(
                    "login",
                    arguments={
                        "login": self.config['account'],
                        "password": self.config['password'],
                        "server": self.config['server']
                    }
                ),
                timeout=10.0
            )
            
            response_text = login_result.content[0].text.lower()
            if "success" in response_text or "logged in" in response_text or response_text == "true":
                self._connected = True
                self.logger.info(f"Connected to MT5: {self.broker_id}")
                return True
            else:
                raise BrokerConnectionError(f"Login failed: {response_text}")
                
        except asyncio.TimeoutError as e:
            raise BrokerTimeoutError(f"Connection timeout: {e}")
        except Exception as e:
            self.logger.exception("MT5 MCP connection error")
            raise BrokerConnectionError(f"Failed to connect: {e}")
    
    async def disconnect(self) -> None:
        """Shutdown MT5 connection via MCP."""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                self.logger.warning(f"Error closing MCP session: {e}")
            finally:
                self.session = None
        
        if hasattr(self, '_stdio_client'):
            try:
                await self._stdio_client.__aexit__(None, None, None)
            except Exception as e:
                self.logger.warning(f"Error closing stdio client: {e}")
            finally:
                self._stdio_client = None
        
        self._connected = False
        self.logger.info(f"Disconnected from MT5: {self.broker_id}")
    
    async def health_check(self) -> bool:
        """
        Check if MCP connection is alive.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        if not self._connected or not self.session:
            return False
        
        try:
            # Try to get symbols as health check
            result = await asyncio.wait_for(
                self.session.call_tool("get_symbols", arguments={}),
                timeout=5.0
            )
            self._last_health_check = datetime.utcnow()
            
            # Debug: log the response
            response_text = result.content[0].text.lower()
            self.logger.info(f"Health check response: {response_text}")
            
            # Check if response contains symbol names (like "eurusd", "btcusd", etc.)
            return len(response_text) > 0 and ("eurusd" in response_text or "btcusd" in response_text or "gold" in response_text)
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
                error=None if is_connected else "MT5 MCP connection failed"
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
        Get list of available symbols from broker via MCP.
        
        Returns:
            List of available symbol names
            
        Raises:
            BrokerConnectionError: If not connected
            MarketDataError: If fetch fails
        """
        if not self.session:
            raise BrokerConnectionError("Not connected")
        
        try:
            result = await asyncio.wait_for(
                self.session.call_tool("get_symbols", arguments={}),
                timeout=5.0
            )
            
            # Parse response - format depends on MCP server implementation
            symbols_text = result.content[0].text
            if "," in symbols_text:
                symbols = [s.strip() for s in symbols_text.split(",")]
            else:
                # Try to parse as JSON array
                try:
                    symbols = json.loads(symbols_text)
                except json.JSONDecodeError:
                    symbols = []
            
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
        Fetch OHLCV data via MCP 'copy_rates_from_pos' tool.
        
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
        if not self.session:
            raise BrokerConnectionError("Not connected")
        
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(
                    "copy_rates_from_pos",
                    arguments={
                        "symbol": symbol,
                        "timeframe": TIMEFRAME_MAP[timeframe],
                        "start_pos": 0,
                        "count": bars
                    }
                ),
                timeout=10.0
            )
            
            # Parse MCP response into DataFrame
            # Response format: JSON array of OHLCV bars
            data = json.loads(result.content[0].text)
            df = pd.DataFrame(data)
            
            # Rename columns to standard format
            df.rename(columns={
                'time': 'timestamp',
                'tick_volume': 'volume'
            }, inplace=True)
            
            # Convert timestamp to datetime (server returns ISO format)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.logger.debug(f"Fetched {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            raise MarketDataError(f"Failed to fetch market data: {e}")
    
    async def get_tick_data(self, symbol: str, count: int = 10) -> pd.DataFrame:
        """
        Get recent tick data via MCP 'get_symbol_info_tick' tool.
        
        Args:
            symbol: Symbol name
            count: Number of ticks (not used in current implementation)
            
        Returns:
            DataFrame with tick data
            
        Raises:
            BrokerConnectionError: If not connected
            MarketDataError: If fetch fails
        """
        if not self.session:
            raise BrokerConnectionError("Not connected")
        
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(
                    "get_symbol_info_tick",
                    arguments={"symbol": symbol}
                ),
                timeout=5.0
            )
            
            # Parse tick data
            tick_data = json.loads(result.content[0].text)
            
            return pd.DataFrame([tick_data])  # Single tick
            
        except Exception as e:
            raise MarketDataError(f"Failed to get tick data: {e}")
    
    async def place_order(self, order: Order) -> str:
        """
        Place order via MCP 'order_send' tool.
        
        Args:
            order: Order object (already validated)
            
        Returns:
            order_id: Broker's order ticket number (as string)
            
        Raises:
            BrokerConnectionError: If not connected
            OrderExecutionError: If order fails
        """
        if not self.session:
            raise BrokerConnectionError("Not connected")
        
        try:
            # Convert Order to MT5 request format
            mt5_request = {
                "action": "TRADE_ACTION_DEAL",  # Market execution
                "symbol": order.symbol,
                "volume": order.quantity,
                "type": "ORDER_TYPE_BUY" if order.side == "buy" else "ORDER_TYPE_SELL",
                "price": order.price if order.price else 0.0,  # 0 for market orders
                "sl": order.stop_loss if order.stop_loss else 0.0,
                "tp": order.take_profit if order.take_profit else 0.0,
                "deviation": 20,  # Max deviation in points
                "magic": 234000,  # Magic number for identification
                "comment": f"MTQuant-{order.agent_id}",
                "type_time": "ORDER_TIME_GTC",
                "type_filling": "ORDER_FILLING_IOC"
            }
            
            result = await asyncio.wait_for(
                self.session.call_tool(
                    "order_send",
                    arguments={"request": mt5_request}
                ),
                timeout=10.0
            )
            
            # Parse response
            response = json.loads(result.content[0].text)
            
            if response.get("retcode") == 10009:  # TRADE_RETCODE_DONE
                order_id = str(response.get("order"))
                self.logger.info(f"Order placed: {order_id} - {order.symbol} {order.side} {order.quantity}")
                return order_id
            else:
                error_msg = response.get("comment", "Unknown error")
                raise OrderExecutionError(f"Order rejected: {error_msg}")
                
        except Exception as e:
            self.logger.exception("Order placement error")
            raise OrderExecutionError(f"Failed to place order: {e}")
    
    async def get_positions(self) -> List[Position]:
        """
        Get all open positions via MCP 'positions_get' tool.
        
        Returns:
            List of Position objects
            
        Raises:
            BrokerConnectionError: If not connected
        """
        if not self.session:
            raise BrokerConnectionError("Not connected")
        
        try:
            result = await asyncio.wait_for(
                self.session.call_tool("positions_get", arguments={}),
                timeout=5.0
            )
            
            positions_data = json.loads(result.content[0].text)
            
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
        Close specific position via MCP order_send (reverse order).
        
        Args:
            position_id: Position ticket ID
            
        Returns:
            True if position closed successfully
            
        Raises:
            BrokerConnectionError: If not connected
            OrderExecutionError: If close fails
        """
        if not self.session:
            raise BrokerConnectionError("Not connected")
        
        try:
            # Get position details first
            positions = await self.get_positions()
            position = next((p for p in positions if p.position_id == position_id), None)
            
            if not position:
                raise OrderExecutionError(f"Position {position_id} not found")
            
            # Create reverse order
            reverse_request = {
                "action": "TRADE_ACTION_DEAL",
                "symbol": position.symbol,
                "volume": position.quantity,
                "type": "ORDER_TYPE_SELL" if position.side == "long" else "ORDER_TYPE_BUY",
                "price": 0.0,  # Market order
                "deviation": 20,
                "magic": 234000,
                "comment": f"MTQuant-Close-{position_id}",
                "type_time": "ORDER_TIME_GTC",
                "type_filling": "ORDER_FILLING_IOC"
            }
            
            result = await asyncio.wait_for(
                self.session.call_tool(
                    "order_send",
                    arguments={"request": reverse_request}
                ),
                timeout=10.0
            )
            
            response = json.loads(result.content[0].text)
            
            if response.get("retcode") == 10009:  # TRADE_RETCODE_DONE
                self.logger.info(f"Position closed: {position_id}")
                return True
            else:
                error_msg = response.get("comment", "Unknown error")
                raise OrderExecutionError(f"Close position failed: {error_msg}")
                
        except Exception as e:
            self.logger.exception("Close position error")
            raise OrderExecutionError(f"Failed to close position: {e}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information via MCP 'account_info' tool.
        
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
        if not self.session:
            raise BrokerConnectionError("Not connected")
        
        try:
            result = await asyncio.wait_for(
                self.session.call_tool("get_account_info", arguments={}),
                timeout=5.0
            )
            
            account_data = json.loads(result.content[0].text)
            
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
        return f"MT5MCPClient(broker_id='{self.broker_id}', status='{status}')"
