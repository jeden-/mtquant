"""
MT5 Broker Adapter for MetaTrader 5 Integration

This adapter wraps MT5Client and adds:
1. Symbol mapping (standard <-> broker symbols)
2. Order conversion (our Order model <-> MT5 orders)
3. Position conversion (MT5 positions <-> our Position model)
4. Additional validation layer

The adapter implements the BrokerAdapter interface and provides
a high-level API for MT5 operations while handling broker-specific
details internally.

Example:
    # Initialize adapter
    adapter = MT5BrokerAdapter(broker_id="ic_markets_mt5_demo", config=broker_config)
    
    # Connect
    await adapter.connect()
    
    # Place order with standard symbol
    order = Order(symbol='XAUUSD', side='buy', quantity=0.1, ...)
    order_id = await adapter.place_order(order)
    
    # Get positions
    positions = await adapter.get_positions()
    
    # Get market data
    data = await adapter.get_market_data('XAUUSD', 'H1', bars=100)
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
from mtquant.mcp_integration.adapters.base_adapter import BrokerAdapter, HealthStatus
from mtquant.mcp_integration.clients.mt5_mcp_client import MT5MCPClient
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


class MT5BrokerAdapter(BrokerAdapter):
    """
    MT5 Broker Adapter implementing BrokerAdapter interface.
    
    This adapter wraps MT5Client and adds symbol mapping, order conversion,
    and additional validation. It provides a high-level API for MT5 operations
    while handling broker-specific details internally.
    
    Features:
    - Symbol mapping between standard and broker-specific symbols
    - Order conversion (Order model <-> MT5 orders)
    - Position conversion (MT5 positions <-> Position model)
    - Additional validation layer
    - Health monitoring
    - Comprehensive error handling
    """
    
    def __init__(self, broker_id: str, config: Dict[str, Any]):
        """
        Initialize MT5 broker adapter.
        
        Args:
            broker_id: Unique broker identifier
            config: Broker configuration from brokers.yaml
        """
        super().__init__(broker_id, config)
        
        # Initialize MT5 MCP client
        self.mt5_client = MT5MCPClient(broker_id, config)
        
        # Symbol mapper for standard <-> broker symbol conversion
        self.symbol_mapper = SymbolMapper
        
        self.logger.info(f"Initialized MT5 broker adapter for {broker_id}")
    
    async def connect(self) -> bool:
        """
        Connect to broker.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            BrokerConnectionError: If connection fails
        """
        self.logger.info(f"Connecting MT5 adapter for {self.broker_id}")
        
        try:
            connected = await self.mt5_client.connect()
            if connected:
                self.logger.info(f"Successfully connected MT5 adapter: {self.broker_id}")
            return connected
            
        except Exception as e:
            self.logger.error(f"MT5 adapter connection failed: {e}")
            raise BrokerConnectionError(f"MT5 adapter connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect cleanly."""
        try:
            await self.mt5_client.disconnect()
            self.logger.info(f"Disconnected MT5 adapter: {self.broker_id}")
        except Exception as e:
            self.logger.error(f"Error during MT5 adapter disconnect: {e}")
    
    async def place_order(self, order: Order) -> str:
        """
        Place order with symbol mapping.
        
        Steps:
        1. Map standard symbol to broker symbol
        2. Validate order (price, quantity, etc.)
        3. Convert Order to MT5 order format
        4. Call mt5_client.place_order()
        5. Log trade for audit
        6. Return order_id
        
        Args:
            order: Order object (with STANDARD symbol)
            
        Returns:
            order_id: Broker's order ticket number
            
        Raises:
            OrderExecutionError: If order fails
            SymbolNotFoundError: If symbol mapping fails
            InvalidOrderError: If order validation fails
        """
        self.logger.info(f"Placing order: {order.symbol} {order.side} {order.quantity}")
        
        try:
            # Step 1: Map standard symbol to broker symbol
            broker_symbol = self.symbol_mapper.to_broker_symbol(order.symbol, self.broker_id)
            self.logger.debug(f"Mapped {order.symbol} -> {broker_symbol}")
            
            # Step 2: Validate order
            if not self._validate_order(order):
                raise InvalidOrderError(f"Order validation failed: {order}")
            
            # Step 3: Convert Order to MT5 order format
            mt5_order = self._convert_order_to_mt5(order, broker_symbol)
            
            # Step 4: Call mt5_client.place_order()
            order_id = await self.mt5_client.place_order(mt5_order)
            
            # Step 5: Log trade for audit
            self.logger.info(f"Order placed successfully: {order_id} for {order.symbol}")
            
            return order_id
            
        except SymbolNotFoundError:
            raise
        except InvalidOrderError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise OrderExecutionError(f"Failed to place order: {e}")
    
    async def get_positions(self) -> List[Position]:
        """
        Get positions with symbol unmapping.
        
        Steps:
        1. Fetch MT5 positions
        2. Convert to our Position model
        3. Map broker symbols back to standard
        4. Return List[Position]
        
        Returns:
            List of Position objects with standard symbols
        """
        try:
            # Step 1: Fetch MT5 positions
            mt5_positions = await self.mt5_client.get_positions()
            
            # Step 2 & 3: Convert to Position model and map symbols
            positions = []
            for mt5_pos in mt5_positions:
                try:
                    # Map broker symbol back to standard
                    standard_symbol = self.symbol_mapper.to_standard_symbol(
                        mt5_pos.symbol, self.broker_id
                    )
                    
                    # Update position with standard symbol
                    mt5_pos.symbol = standard_symbol
                    positions.append(mt5_pos)
                    
                except SymbolNotFoundError:
                    self.logger.warning(f"Unknown broker symbol: {mt5_pos.symbol}")
                    # Keep position with broker symbol if mapping fails
                    positions.append(mt5_pos)
            
            self.logger.debug(f"Retrieved {len(positions)} positions")
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            raise BrokerError(f"Failed to get positions: {e}")
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = 'H1',
        bars: int = 100
    ) -> pd.DataFrame:
        """
        Fetch market data with symbol mapping.
        
        Args:
            symbol: STANDARD symbol (e.g., XAUUSD)
            timeframe: Timeframe string
            bars: Number of bars to fetch
            
        Steps:
        1. Map standard symbol to broker symbol
        2. Fetch data from mt5_client
        3. Add standard symbol column to DataFrame
        4. Return DataFrame
        
        Returns:
            DataFrame with OHLCV data and standard symbol
        """
        try:
            # Step 1: Map standard symbol to broker symbol
            broker_symbol = self.symbol_mapper.to_broker_symbol(symbol, self.broker_id)
            self.logger.debug(f"Fetching data for {symbol} -> {broker_symbol}")
            
            # Step 2: Fetch data from mt5_client
            data = await self.mt5_client.get_market_data(broker_symbol, timeframe, bars)
            
            # Step 3: Add standard symbol column to DataFrame
            data['symbol'] = symbol
            
            # Step 4: Return DataFrame
            self.logger.debug(f"Retrieved {len(data)} bars for {symbol}")
            return data
            
        except SymbolNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            raise BrokerError(f"Failed to get market data for {symbol}: {e}")
    
    async def health_check(self) -> HealthStatus:
        """
        Check adapter health.
        
        Returns:
            HealthStatus object with connection details
        """
        try:
            # Get health status from MT5 client
            mt5_health = await self.mt5_client.get_health_status()
            
            # Create adapter health status
            health = HealthStatus(
                is_connected=mt5_health.is_connected,
                latency_ms=mt5_health.latency_ms,
                last_check=mt5_health.last_check,
                error=mt5_health.error
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
            Dictionary with account details
        """
        try:
            account_info = await self.mt5_client.get_account_info()
            self.logger.debug("Retrieved account info")
            return account_info
            
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            raise BrokerError(f"Failed to get account info: {e}")
    
    async def close_position(self, position_id: str) -> bool:
        """
        Close specific position.
        
        Args:
            position_id: Position identifier
            
        Returns:
            True if position closed successfully
        """
        try:
            result = await self.mt5_client.close_position(position_id)
            self.logger.info(f"Position closed: {position_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to close position {position_id}: {e}")
            raise BrokerError(f"Failed to close position {position_id}: {e}")
    
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
            # Check if symbol exists in mapping
            self.symbol_mapper.to_broker_symbol(order.symbol, self.broker_id)
            
            # Check quantity
            if order.quantity <= 0:
                self.logger.warning(f"Invalid quantity: {order.quantity}")
                return False
            
            # Check price for limit/stop orders
            if order.order_type in ['limit', 'stop'] and order.price is not None:
                if order.price <= 0:
                    self.logger.warning(f"Invalid price: {order.price}")
                    return False
            
            return True
            
        except SymbolNotFoundError:
            self.logger.warning(f"Symbol not found: {order.symbol}")
            return False
        except Exception as e:
            self.logger.warning(f"Order validation error: {e}")
            return False
    
    def _convert_order_to_mt5(self, order: Order, broker_symbol: str) -> Order:
        """
        Convert Order object to MT5 format.
        
        Args:
            order: Order object with standard symbol
            broker_symbol: Broker-specific symbol
            
        Returns:
            Order object with broker symbol
        """
        # Create new order with broker symbol
        mt5_order = Order(
            order_id=order.order_id,
            agent_id=order.agent_id,
            symbol=broker_symbol,  # Use broker symbol
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
        
        return mt5_order
    
    def _convert_mt5_to_position(self, mt5_position: Position) -> Position:
        """
        Convert MT5 position to Position object with standard symbol.
        
        Args:
            mt5_position: Position object with broker symbol
            
        Returns:
            Position object with standard symbol
        """
        try:
            # Map broker symbol back to standard
            standard_symbol = self.symbol_mapper.to_standard_symbol(
                mt5_position.symbol, self.broker_id
            )
            
            # Create new position with standard symbol
            position = Position(
                position_id=mt5_position.position_id,
                agent_id=mt5_position.agent_id,
                symbol=standard_symbol,  # Use standard symbol
                side=mt5_position.side,
                quantity=mt5_position.quantity,
                entry_price=mt5_position.entry_price,
                current_price=mt5_position.current_price,
                stop_loss=mt5_position.stop_loss,
                take_profit=mt5_position.take_profit,
                unrealized_pnl=mt5_position.unrealized_pnl,
                opened_at=mt5_position.opened_at,
                broker_id=mt5_position.broker_id
            )
            
            return position
            
        except SymbolNotFoundError:
            # Return position with broker symbol if mapping fails
            self.logger.warning(f"Symbol mapping failed for {mt5_position.symbol}")
            return mt5_position
    
    async def health_check(self) -> HealthStatus:
        """
        Check broker connection health.
        
        Returns:
            HealthStatus object with connection details
        """
        return await self.mt5_client.get_health_status()
    
    def __repr__(self) -> str:
        """String representation of MT5BrokerAdapter."""
        return f"MT5BrokerAdapter(broker_id={self.broker_id})"
