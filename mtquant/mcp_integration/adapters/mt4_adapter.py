"""
MT4 Broker Adapter for MetaTrader 4 Integration

This adapter wraps MT4MCPClient and adds:
1. Symbol mapping (standard <-> broker symbols)
2. Order conversion (our Order model <-> MT4 orders)
3. Position conversion (MT4 positions <-> our Position model)
4. Additional validation layer

The adapter implements the BrokerAdapter interface and provides
a high-level API for MT4 operations while handling broker-specific
details internally.

Example:
    # Initialize adapter
    adapter = MT4BrokerAdapter(broker_id="ic_markets_mt4_demo", config=broker_config)
    
    # Connect
    await adapter.connect()
    
    # Place order with standard symbol
    order = Order(symbol='EURUSD', side='buy', quantity=0.1, ...)
    order_id = await adapter.place_order(order)
    
    # Get positions
    positions = await adapter.get_positions()
    
    # Get market data
    data = await adapter.get_market_data('EURUSD', 'H1', bars=100)
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
from mtquant.mcp_integration.adapters.base_adapter import BrokerAdapter, HealthStatus
from mtquant.mcp_integration.clients.mt4_mcp_client import MT4MCPClient
from mtquant.mcp_integration.managers.symbol_mapper import SymbolMapper
from mtquant.mcp_integration.models.order import Order
from mtquant.mcp_integration.models.position import Position
from mtquant.utils.exceptions import (
    BrokerError,
    BrokerConnectionError,
    OrderExecutionError,
    SymbolNotFoundError,
    InvalidOrderError
)
from mtquant.utils.logger import get_logger

logger = get_logger(__name__)


class MT4BrokerAdapter(BrokerAdapter):
    """
    MT4 Broker Adapter implementing BrokerAdapter interface.
    
    This adapter wraps MT4MCPClient and adds symbol mapping, order conversion,
    and additional validation. It provides a high-level API for MT4 operations
    while handling broker-specific details internally.
    
    Features:
    - Symbol mapping between standard and broker-specific symbols
    - Order conversion (Order model <-> MT4 orders)
    - Position conversion (MT4 positions <-> Position model)
    - Additional validation layer
    - Health monitoring
    - Comprehensive error handling
    """
    
    def __init__(self, broker_id: str, config: Dict[str, Any]):
        """
        Initialize MT4 broker adapter.
        
        Args:
            broker_id: Unique broker identifier
            config: Broker configuration from brokers.yaml
        """
        super().__init__(broker_id, config)
        
        # Initialize MT4 MCP client
        self.mt4_client = MT4MCPClient(broker_id, config)
        
        # Symbol mapper for standard <-> broker symbol conversion
        self.symbol_mapper = SymbolMapper
        
        self.logger.info(f"Initialized MT4 broker adapter for {broker_id}")
    
    async def connect(self) -> bool:
        """
        Connect to broker.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            BrokerConnectionError: If connection fails
        """
        self.logger.info(f"Connecting MT4 adapter for {self.broker_id}")
        
        try:
            connected = await self.mt4_client.connect()
            if connected:
                self.logger.info(f"MT4 adapter connected successfully: {self.broker_id}")
            else:
                self.logger.error(f"MT4 adapter connection failed: {self.broker_id}")
            return connected
            
        except Exception as e:
            self.logger.exception(f"MT4 adapter connection error: {self.broker_id}")
            raise BrokerConnectionError(f"Failed to connect MT4 adapter: {e}")
    
    async def disconnect(self) -> None:
        """
        Disconnect from broker.
        
        Performs clean shutdown of MT4 connection.
        """
        self.logger.info(f"Disconnecting MT4 adapter for {self.broker_id}")
        
        try:
            await self.mt4_client.disconnect()
            self.logger.info(f"MT4 adapter disconnected: {self.broker_id}")
        except Exception as e:
            self.logger.warning(f"MT4 adapter disconnect error (non-fatal): {e}")
    
    async def place_order(self, order: Order) -> str:
        """
        Place order with symbol mapping.
        
        Steps:
        1. Map standard symbol to broker symbol
        2. Validate order (price, quantity, etc.)
        3. Convert Order to MT4 order format
        4. Call mt4_client.place_order()
        5. Log trade for audit
        6. Return order_id
        
        Args:
            order: Order object with standard symbol
            
        Returns:
            order_id: Broker's order ticket number
            
        Raises:
            SymbolNotFoundError: If symbol mapping fails
            InvalidOrderError: If order validation fails
            OrderExecutionError: If order placement fails
        """
        self.logger.info(f"Placing order via MT4 adapter: {order.symbol} {order.side} {order.quantity}")
        
        try:
            # Step 1: Map standard symbol to broker symbol
            broker_symbol = self.symbol_mapper.to_broker_symbol(
                order.symbol, self.broker_id
            )
            self.logger.debug(f"Symbol mapping: {order.symbol} -> {broker_symbol}")
            
            # Step 2: Validate order
            if not self._validate_order(order):
                raise InvalidOrderError(f"Order validation failed: {order}")
            
            # Step 3: Create order copy with broker symbol
            broker_order = Order(
                order_id=order.order_id,
                agent_id=order.agent_id,
                symbol=broker_symbol,  # Use broker-specific symbol
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                signal=order.signal,
                created_at=order.created_at,
                status=order.status,
                broker_id=order.broker_id
            )
            
            # Step 4: Place order via MCP client
            order_id = await self.mt4_client.place_order(broker_order)
            
            # Step 5: Log for audit
            self.logger.info(
                f"Order placed via MT4: {order_id} | {order.symbol}->{broker_symbol} | "
                f"{order.side} {order.quantity} @ {order.price or 'market'}"
            )
            
            return order_id
            
        except SymbolNotFoundError:
            raise
        except Exception as e:
            self.logger.exception(f"MT4 order placement error: {self.broker_id}")
            raise OrderExecutionError(f"Failed to place order via MT4: {e}")
    
    async def get_positions(self) -> List[Position]:
        """
        Get positions with symbol unmapping.
        
        Steps:
        1. Fetch MT4 positions
        2. Convert to our Position model
        3. Map broker symbols back to standard
        4. Return List[Position]
        
        Returns:
            List of Position objects with standard symbols
            
        Raises:
            BrokerConnectionError: If not connected
        """
        self.logger.debug(f"Getting positions via MT4 adapter: {self.broker_id}")
        
        try:
            # Step 1: Fetch MT4 positions
            broker_positions = await self.mt4_client.get_positions()
            
            # Step 2 & 3: Convert and unmap symbols
            positions = []
            for pos in broker_positions:
                try:
                    # Map broker symbol back to standard
                    standard_symbol = self.symbol_mapper.to_standard_symbol(
                        pos.symbol, self.broker_id
                    )
                    
                    # Create position with standard symbol
                    position = Position(
                        position_id=pos.position_id,
                        agent_id=pos.agent_id,
                        symbol=standard_symbol,  # Use standard symbol
                        side=pos.side,
                        quantity=pos.quantity,
                        entry_price=pos.entry_price,
                        current_price=pos.current_price,
                        stop_loss=pos.stop_loss,
                        take_profit=pos.take_profit,
                        unrealized_pnl=pos.unrealized_pnl,
                        opened_at=pos.opened_at,
                        broker_id=pos.broker_id
                    )
                    positions.append(position)
                    
                except Exception as e:
                    self.logger.warning(f"Symbol unmapping failed: {pos.symbol} - {e}")
                    # Keep position with broker symbol if unmapping fails
                    positions.append(pos)
            
            self.logger.debug(f"Retrieved {len(positions)} positions via MT4 adapter")
            return positions
            
        except Exception as e:
            self.logger.exception(f"MT4 get positions error: {self.broker_id}")
            raise BrokerError(f"Failed to get positions via MT4: {e}")
    
    async def get_market_data(
        self,
        symbol: str,  # STANDARD symbol
        timeframe: str = 'H1',
        bars: int = 100
    ) -> pd.DataFrame:
        """
        Fetch market data with symbol mapping.
        
        Args:
            symbol: STANDARD symbol (e.g., XAUUSD)
            timeframe: Timeframe string (H1, H4, D1, etc.)
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLCV data and standard symbol column
            
        Raises:
            SymbolNotFoundError: If symbol mapping fails
            MarketDataError: If data fetch fails
        """
        self.logger.debug(f"Getting market data via MT4 adapter: {symbol} {timeframe}")
        
        try:
            # Step 1: Map standard symbol to broker symbol
            broker_symbol = self.symbol_mapper.to_broker_symbol(symbol, self.broker_id)
            self.logger.debug(f"Symbol mapping: {symbol} -> {broker_symbol}")
            
            # Step 2: Fetch data from MCP client
            df = await self.mt4_client.get_market_data(broker_symbol, timeframe, bars)
            
            # Step 3: Add standard symbol column
            df['standard_symbol'] = symbol
            
            self.logger.debug(f"Retrieved {len(df)} bars for {symbol} via MT4 adapter")
            return df
            
        except SymbolNotFoundError:
            raise
        except Exception as e:
            self.logger.exception(f"MT4 market data error: {self.broker_id}")
            raise BrokerError(f"Failed to get market data via MT4: {e}")
    
    async def health_check(self) -> HealthStatus:
        """
        Check adapter health.
        
        Returns:
            HealthStatus object with connection details
        """
        try:
            # Get health status from MT4 client
            mt4_health = await self.mt4_client.get_health_status()
            
            # Create adapter health status
            health = HealthStatus(
                is_connected=mt4_health.is_connected,
                latency_ms=mt4_health.latency_ms,
                last_check=mt4_health.last_check,
                error=mt4_health.error
            )
            
            return health
            
        except Exception as e:
            return HealthStatus(
                is_connected=False,
                latency_ms=0.0,
                last_check=datetime.utcnow(),
                error=str(e)
            )
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account information
        """
        try:
            return await self.mt4_client.get_account_info()
        except Exception as e:
            self.logger.error(f"Failed to get account info via MT4: {e}")
            return {}
    
    async def close_position(self, position_id: str) -> bool:
        """
        Close specific position.
        
        Args:
            position_id: Position identifier
            
        Returns:
            True if position closed successfully
        """
        try:
            return await self.mt4_client.close_position(position_id)
        except Exception as e:
            self.logger.error(f"Failed to close position {position_id} via MT4: {e}")
            return False
    
    def _validate_order(self, order: Order) -> bool:
        """
        Validate order before sending to broker.
        
        Checks:
        - Symbol exists
        - Quantity > 0
        - Price reasonable (if limit/stop)
        
        Args:
            order: Order to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check quantity
            if order.quantity <= 0:
                self.logger.warning(f"Invalid quantity: {order.quantity}")
                return False
            
            # Check symbol exists
            try:
                self.symbol_mapper.to_broker_symbol(order.symbol, self.broker_id)
            except SymbolNotFoundError:
                self.logger.warning(f"Symbol not found: {order.symbol}")
                return False
            
            # Check price for limit/stop orders
            if order.order_type in ['limit', 'stop'] and order.price is None:
                self.logger.warning(f"Price required for {order.order_type} order")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Order validation error: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return f"MT4BrokerAdapter(broker_id='{self.broker_id}')"
