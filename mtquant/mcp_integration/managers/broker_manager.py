"""
Broker Manager for MTQuant Trading System.

This module provides the main orchestrator for broker operations.
This is the high-level API that other parts of MTQuant will use.

Features:
- Intelligent broker selection for order routing
- Position aggregation across multiple brokers
- Market data fetching with automatic broker selection
- Account information aggregation
- Comprehensive broker status monitoring
- Integration with ConnectionPool for failover

Example:
    # Initialize manager
    manager = BrokerManager()
    configs = load_broker_configs()  # from YAML
    await manager.initialize(configs)

    # Place order (automatic broker selection)
    order = Order(symbol='XAUUSD', side='buy', quantity=0.1, ...)
    order_id = await manager.place_order(order)

    # Get positions from all brokers
    positions = await manager.get_positions()

    # Get market data
    data = await manager.get_market_data('XAUUSD', 'H1', bars=200)
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
from mtquant.mcp_integration.managers.connection_pool import ConnectionPool
from mtquant.mcp_integration.managers.symbol_mapper import SymbolMapper
from mtquant.mcp_integration.adapters.base_adapter import BrokerAdapter
from mtquant.mcp_integration.models.order import Order
from mtquant.mcp_integration.models.position import Position
from mtquant.utils.exceptions import BrokerError, BrokerConnectionError, RiskViolationError
from mtquant.utils.logger import get_logger

logger = get_logger(__name__)


class BrokerManager:
    """
    Main orchestrator for broker operations.
    
    This is the high-level API that other parts of MTQuant will use.
    It provides intelligent broker selection, position aggregation,
    and comprehensive monitoring capabilities.
    
    Features:
    - Intelligent broker selection for order routing
    - Position aggregation across multiple brokers
    - Market data fetching with automatic broker selection
    - Account information aggregation
    - Comprehensive broker status monitoring
    - Integration with ConnectionPool for failover
    """
    
    def __init__(self):
        """Initialize broker manager."""
        self.connection_pool = ConnectionPool()
        self.symbol_mapper = SymbolMapper
        self.logger = get_logger(__name__)
        self._initialized = False
        
        self.logger.info("BrokerManager initialized")
    
    async def initialize(self, broker_configs: List[Dict[str, Any]]) -> None:
        """
        Initialize broker manager with configurations.
        
        Args:
            broker_configs: List of broker config dicts from brokers.yaml
            
        Steps:
        1. Create adapter for each broker config
        2. Add to connection pool
        3. Connect all adapters
        4. Start health monitoring
        5. Set _initialized = True
        
        Raises:
            BrokerError: If initialization fails
        """
        self.logger.info(f"Initializing BrokerManager with {len(broker_configs)} brokers")
        
        try:
            # Step 1: Create adapter for each broker config
            adapters_created = 0
            primary_broker_id = None
            
            for config in broker_configs:
                try:
                    # Create adapter based on platform
                    if config['platform'] == 'mt5':
                        from mtquant.mcp_integration.adapters import MT5BrokerAdapter
                        adapter = MT5BrokerAdapter(
                            broker_id=config['broker_id'],
                            config=config
                        )
                    else:
                        self.logger.warning(f"Unsupported platform: {config['platform']}")
                        continue
                    
                    # Step 2: Add to connection pool
                    is_primary = config.get('is_primary', False)
                    if is_primary:
                        primary_broker_id = config['broker_id']
                    
                    await self.connection_pool.add_adapter(
                        broker_id=config['broker_id'],
                        adapter=adapter,
                        is_primary=is_primary
                    )
                    
                    adapters_created += 1
                    self.logger.info(f"Created adapter for {config['broker_id']}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to create adapter for {config['broker_id']}: {e}")
                    continue
            
            if adapters_created == 0:
                raise BrokerError("No adapters were created successfully")
            
            # Step 3: Connect all adapters
            connection_results = await self.connection_pool.connect_all()
            successful_connections = sum(1 for success in connection_results.values() if success)
            
            if successful_connections == 0:
                raise BrokerConnectionError("No brokers connected successfully")
            
            # Step 4: Start health monitoring
            await self.connection_pool.start_health_monitoring(interval=30)
            
            # Step 5: Perform initial health check
            await self.connection_pool.health_check_all()
            
            # Step 6: Set _initialized = True
            self._initialized = True
            
            self.logger.info(f"BrokerManager initialized successfully: "
                           f"{adapters_created} adapters, {successful_connections} connected")
            
        except Exception as e:
            self.logger.error(f"BrokerManager initialization failed: {e}")
            raise BrokerError(f"Initialization failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown all connections cleanly."""
        self.logger.info("Shutting down BrokerManager")
        
        try:
            # Stop health monitoring
            await self.connection_pool.stop_health_monitoring()
            
            # Disconnect all adapters
            await self.connection_pool.disconnect_all()
            
            self._initialized = False
            self.logger.info("BrokerManager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def place_order(
        self,
        order: Order,
        preferred_broker: Optional[str] = None
    ) -> str:
        """
        Place order with intelligent broker selection.
        
        Args:
            order: Order object (with STANDARD symbol)
            preferred_broker: Optional broker preference
            
        Logic:
        1. If preferred_broker specified and healthy -> use it
        2. Else get healthy adapter from pool
        3. Place order
        4. Return order_id
        
        Returns:
            order_id: Broker's order ticket number
            
        Raises:
            BrokerError: If order fails
            RiskViolationError: If order violates limits (future)
        """
        if not self._initialized:
            raise BrokerError("BrokerManager not initialized")
        
        self.logger.info(f"Placing order: {order.symbol} {order.side} {order.quantity}")
        
        try:
            # Step 1: Select broker
            broker_id = self._select_broker_for_order(order, preferred_broker)
            
            # Step 2: Get adapter
            adapter = await self.connection_pool.get_adapter(broker_id)
            
            # Step 3: Place order
            order_id = await adapter.place_order(order)
            
            self.logger.info(f"Order placed successfully: {order_id} via {broker_id}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise BrokerError(f"Order placement failed: {e}")
    
    async def get_positions(
        self,
        broker_id: Optional[str] = None
    ) -> List[Position]:
        """
        Get positions from broker(s).
        
        Args:
            broker_id: Specific broker, or None for all brokers
            
        Returns:
            List of positions (aggregated if multiple brokers)
        """
        if not self._initialized:
            raise BrokerError("BrokerManager not initialized")
        
        try:
            if broker_id:
                # Get positions from specific broker
                adapter = await self.connection_pool.get_adapter(broker_id)
                positions = await adapter.get_positions()
                self.logger.debug(f"Retrieved {len(positions)} positions from {broker_id}")
            else:
                # Get positions from all brokers
                all_positions = []
                stats = self.connection_pool.get_connection_stats()
                
                for broker_id in self.connection_pool.adapters.keys():
                    try:
                        adapter = await self.connection_pool.get_adapter(broker_id)
                        positions = await adapter.get_positions()
                        all_positions.extend(positions)
                    except Exception as e:
                        self.logger.warning(f"Failed to get positions from {broker_id}: {e}")
                        continue
                
                positions = all_positions
                self.logger.debug(f"Retrieved {len(positions)} positions from all brokers")
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            raise BrokerError(f"Failed to get positions: {e}")
    
    async def close_position(
        self,
        position_id: str,
        broker_id: str
    ) -> bool:
        """
        Close specific position at specific broker.
        
        Args:
            position_id: Position identifier
            broker_id: Broker identifier
            
        Returns:
            True if position closed successfully
        """
        if not self._initialized:
            raise BrokerError("BrokerManager not initialized")
        
        try:
            adapter = await self.connection_pool.get_adapter(broker_id)
            result = await adapter.close_position(position_id)
            
            self.logger.info(f"Position {position_id} closed via {broker_id}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to close position {position_id}: {e}")
            raise BrokerError(f"Failed to close position: {e}")
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = 'H1',
        bars: int = 100,
        broker_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch market data with automatic broker selection.
        
        Args:
            symbol: STANDARD symbol (XAUUSD, BTCUSD, etc.)
            timeframe: Timeframe string
            bars: Number of bars
            broker_id: Optional specific broker
            
        Logic:
        1. Map symbol for target broker
        2. Fetch data from healthy adapter
        3. Return DataFrame with standard symbol
        
        Returns:
            DataFrame with OHLCV data and standard symbol
        """
        if not self._initialized:
            raise BrokerError("BrokerManager not initialized")
        
        try:
            if broker_id:
                # Use specific broker
                adapter = await self.connection_pool.get_adapter(broker_id)
            else:
                # Use healthy adapter
                adapter = await self.connection_pool.get_healthy_adapter()
                broker_id = adapter.get_broker_id()
            
            # Fetch data
            data = await adapter.get_market_data(symbol, timeframe, bars)
            
            self.logger.debug(f"Retrieved {len(data)} bars for {symbol} via {broker_id}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            raise BrokerError(f"Failed to get market data: {e}")
    
    async def get_account_info(
        self,
        broker_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get account info from broker(s).
        
        Args:
            broker_id: Optional specific broker
            
        Returns:
            Dict of broker_id -> account_info
        """
        if not self._initialized:
            raise BrokerError("BrokerManager not initialized")
        
        try:
            if broker_id:
                # Get info from specific broker
                adapter = await self.connection_pool.get_adapter(broker_id)
                account_info = await adapter.get_account_info()
                return {broker_id: account_info}
            else:
                # Get info from all brokers
                all_info = {}
                stats = self.connection_pool.get_connection_stats()
                
                for broker_id in self.connection_pool.adapters.keys():
                    try:
                        adapter = await self.connection_pool.get_adapter(broker_id)
                        account_info = await adapter.get_account_info()
                        all_info[broker_id] = account_info
                    except Exception as e:
                        self.logger.warning(f"Failed to get account info from {broker_id}: {e}")
                        continue
                
                return all_info
                
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            raise BrokerError(f"Failed to get account info: {e}")
    
    async def get_broker_status(self) -> Dict[str, Any]:
        """
        Get comprehensive broker status.
        
        Returns:
            {
                'healthy_brokers': List[str],
                'unhealthy_brokers': List[str],
                'primary_broker': str,
                'connection_stats': Dict,
                'last_health_check': datetime
            }
        """
        if not self._initialized:
            raise BrokerError("BrokerManager not initialized")
        
        try:
            # Get health status from all adapters
            health_results = await self.connection_pool.health_check_all()
            
            # Categorize brokers
            healthy_brokers = []
            unhealthy_brokers = []
            
            for broker_id, health in health_results.items():
                if health.is_connected:
                    healthy_brokers.append(broker_id)
                else:
                    unhealthy_brokers.append(broker_id)
            
            # Get connection stats
            connection_stats = self.connection_pool.get_connection_stats()
            
            status = {
                'healthy_brokers': healthy_brokers,
                'unhealthy_brokers': unhealthy_brokers,
                'primary_broker': connection_stats.primary_broker,
                'connection_stats': {
                    'total_adapters': connection_stats.total_adapters,
                    'healthy_adapters': connection_stats.healthy_adapters,
                    'backup_brokers': connection_stats.backup_brokers,
                    'total_uptime_hours': connection_stats.total_uptime_hours,
                    'total_failures': connection_stats.total_failures
                },
                'last_health_check': connection_stats.last_health_check
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get broker status: {e}")
            raise BrokerError(f"Failed to get broker status: {e}")
    
    def is_initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized
    
    def _select_broker_for_order(
        self,
        order: Order,
        preferred: Optional[str]
    ) -> str:
        """
        Intelligent broker selection logic.
        
        Priority:
        1. Preferred broker (if healthy)
        2. Primary broker (if healthy)
        3. First healthy backup
        
        Consider:
        - Broker health
        - Spread/commission (future)
        - Latency (future)
        
        Args:
            order: Order to place
            preferred: Preferred broker ID
            
        Returns:
            Selected broker ID
        """
        try:
            # Priority 1: Preferred broker (if healthy)
            if preferred:
                try:
                    # Check if preferred broker exists in pool
                    if preferred in self.connection_pool.adapters:
                        return preferred
                except Exception:
                    self.logger.warning(f"Preferred broker {preferred} not available")
            
            # Priority 2: Primary broker (if healthy)
            stats = self.connection_pool.get_connection_stats()
            if stats.primary_broker:
                return stats.primary_broker
            
            # Priority 3: First healthy backup
            if stats.backup_brokers:
                return stats.backup_brokers[0]
            
            # Fallback: any available broker
            if stats.total_adapters > 0:
                return list(self.connection_pool.adapters.keys())[0]
            
            raise BrokerConnectionError("No brokers available")
            
        except Exception as e:
            self.logger.error(f"Broker selection failed: {e}")
            raise BrokerConnectionError(f"Broker selection failed: {e}")
    
    def __repr__(self) -> str:
        """String representation of BrokerManager."""
        status = "initialized" if self._initialized else "not initialized"
        stats = self.connection_pool.get_connection_stats()
        return (f"BrokerManager(status={status}, "
                f"adapters={stats.total_adapters}, "
                f"healthy={stats.healthy_adapters})")
