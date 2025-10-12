"""
Order Model for MTQuant Trading System

Defines the Order dataclass for representing trading orders
with proper validation and serialization methods.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Literal
from enum import Enum


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order model with validation and serialization.
    
    Represents a trading order with all necessary fields for execution
    and tracking. Includes validation for signal range, quantity, and side.
    
    Attributes:
        order_id: Broker order ID after execution (None for new orders)
        agent_id: Which agent created this order
        symbol: Standard symbol like XAUUSD
        side: Order side (buy/sell)
        order_type: Order type (market/limit/stop)
        quantity: Position size in lots
        price: Price for limit/stop orders (None for market orders)
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        signal: RL model signal (-1 to 1)
        created_at: When the order was created
        status: Current order status
        broker_id: Which broker to use (optional)
        metadata: Additional order metadata
    """
    
    # Required fields
    agent_id: str
    symbol: str
    side: Literal['buy', 'sell']
    order_type: Literal['market', 'limit', 'stop']
    quantity: float
    signal: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Optional fields
    order_id: Optional[str] = None
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: Literal['pending', 'filled', 'cancelled', 'rejected'] = 'pending'
    broker_id: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate order data after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate order fields.
        
        Raises:
            ValueError: If validation fails
        """
        # Validate signal range
        if not -1.0 <= self.signal <= 1.0:
            raise ValueError(f"Signal must be between -1 and 1, got {self.signal}")
        
        # Validate quantity
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")
        
        # Validate side
        if self.side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side '{self.side}', must be 'buy' or 'sell'")
        
        # Validate order type
        if self.order_type not in ['market', 'limit', 'stop']:
            raise ValueError(f"Invalid order type '{self.order_type}', must be 'market', 'limit', or 'stop'")
        
        # Validate price for limit/stop orders
        if self.order_type in ['limit', 'stop'] and self.price is None:
            raise ValueError(f"Price required for {self.order_type} orders")
        
        # Validate stop loss and take profit
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError(f"Stop loss must be positive, got {self.stop_loss}")
        
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError(f"Take profit must be positive, got {self.take_profit}")
        
        # Validate stop loss/take profit logic
        if self.stop_loss is not None and self.take_profit is not None:
            if self.side == 'buy':
                if self.stop_loss >= self.take_profit:
                    raise ValueError("For buy orders, stop loss must be below take profit")
            else:  # sell
                if self.stop_loss <= self.take_profit:
                    raise ValueError("For sell orders, stop loss must be above take profit")
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary.
        
        Returns:
            Dictionary representation of the order
        """
        return {
            'order_id': self.order_id,
            'agent_id': self.agent_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'quantity': self.quantity,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'signal': self.signal,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'broker_id': self.broker_id,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Order':
        """Create order from dictionary.
        
        Args:
            data: Dictionary with order data
            
        Returns:
            Order instance
            
        Raises:
            ValueError: If required fields are missing
        """
        # Parse datetime
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        # Remove None values for optional fields
        filtered_data = {k: v for k, v in data.items() if v is not None}
        
        return cls(**filtered_data)
    
    def validate(self) -> bool:
        """Check if order is valid.
        
        Returns:
            True if order is valid, False otherwise
        """
        try:
            self._validate()
            return True
        except ValueError:
            return False
    
    def is_market_order(self) -> bool:
        """Check if this is a market order.
        
        Returns:
            True if market order, False otherwise
        """
        return self.order_type == 'market'
    
    def is_limit_order(self) -> bool:
        """Check if this is a limit order.
        
        Returns:
            True if limit order, False otherwise
        """
        return self.order_type == 'limit'
    
    def is_stop_order(self) -> bool:
        """Check if this is a stop order.
        
        Returns:
            True if stop order, False otherwise
        """
        return self.order_type == 'stop'
    
    def has_stop_loss(self) -> bool:
        """Check if order has stop loss.
        
        Returns:
            True if stop loss is set, False otherwise
        """
        return self.stop_loss is not None
    
    def has_take_profit(self) -> bool:
        """Check if order has take profit.
        
        Returns:
            True if take profit is set, False otherwise
        """
        return self.take_profit is not None
    
    def get_risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk-reward ratio.
        
        Returns:
            Risk-reward ratio if both stop loss and take profit are set,
            None otherwise
        """
        if not self.has_stop_loss() or not self.has_take_profit():
            return None
        
        if self.side == 'buy':
            risk = abs(self.price - self.stop_loss) if self.price else 0
            reward = abs(self.take_profit - self.price) if self.price else 0
        else:  # sell
            risk = abs(self.stop_loss - self.price) if self.price else 0
            reward = abs(self.price - self.take_profit) if self.price else 0
        
        if risk == 0:
            return None
        
        return reward / risk
    
    def __repr__(self) -> str:
        """Return string representation of the order."""
        return (
            f"Order(order_id={self.order_id}, agent_id={self.agent_id}, "
            f"symbol={self.symbol}, side={self.side}, type={self.order_type}, "
            f"quantity={self.quantity}, signal={self.signal:.3f}, "
            f"status={self.status})"
        )
    
    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"Order {self.order_id or 'NEW'}: {self.side.upper()} {self.quantity} "
            f"{self.symbol} @ {self.price or 'MARKET'} "
            f"(Signal: {self.signal:.3f}, Status: {self.status})"
        )
