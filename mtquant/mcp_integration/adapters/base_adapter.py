"""
Base Broker Adapter for MTQuant Trading System

This module provides the abstract base class for all broker adapters.
The BrokerAdapter defines the interface that all broker implementations
must follow, ensuring consistency across different brokers.

The adapter pattern allows MTQuant to support multiple brokers
(MT5, MT4, OANDA, etc.) through a unified interface while handling
broker-specific details internally.

Example:
    class MT5BrokerAdapter(BrokerAdapter):
        async def connect(self) -> bool:
            # MT5-specific connection logic
            pass
        
        async def place_order(self, order: Order) -> str:
            # MT5-specific order placement
            pass
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import pandas as pd

from mtquant.mcp_integration.models.order import Order
from mtquant.mcp_integration.models.position import Position
from mtquant.utils.exceptions import BrokerError
from mtquant.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    """Health status for broker adapter."""
    is_connected: bool
    latency_ms: float
    last_check: datetime
    error: Optional[str] = None


class BrokerAdapter(ABC):
    """
    Abstract base class for all broker adapters.
    
    This class defines the interface that all broker implementations
    must follow. It provides a unified interface for broker operations
    while allowing broker-specific implementations.
    
    All methods are async to support non-blocking operations and
    comprehensive error handling is expected.
    """
    
    def __init__(self, broker_id: str, config: Dict[str, Any]):
        """
        Initialize broker adapter.
        
        Args:
            broker_id: Unique broker identifier
            config: Broker configuration from brokers.yaml
        """
        self.broker_id = broker_id
        self.config = config
        self.logger = get_logger(f"{__name__}.{broker_id}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to broker.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            BrokerError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from broker cleanly.
        
        This method should ensure all resources are properly cleaned up
        and connections are closed gracefully.
        """
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """
        Place order on broker.
        
        Args:
            order: Order object (already validated)
            
        Returns:
            order_id: Broker's order ticket number
            
        Raises:
            BrokerError: If order fails
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of Position objects
        """
        pass
    
    @abstractmethod
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = 'H1',
        bars: int = 100
    ) -> pd.DataFrame:
        """
        Fetch market data.
        
        Args:
            symbol: Standard symbol (e.g., EURUSD, XAUUSD)
            timeframe: Timeframe string
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """
        Check broker health.
        
        Returns:
            HealthStatus object with connection details
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account details
        """
        pass
    
    @abstractmethod
    async def close_position(self, position_id: str) -> bool:
        """
        Close specific position.
        
        Args:
            position_id: Position identifier
            
        Returns:
            True if position closed successfully
        """
        pass
    
    def get_broker_id(self) -> str:
        """Get broker identifier."""
        return self.broker_id
    
    def get_config(self) -> Dict[str, Any]:
        """Get broker configuration."""
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation of broker adapter."""
        return f"{self.__class__.__name__}(broker_id={self.broker_id})"
