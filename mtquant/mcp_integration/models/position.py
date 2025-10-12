"""
Position Model for MTQuant Trading System

Defines the Position dataclass for representing trading positions
with calculated properties and update methods.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Literal


@dataclass
class Position:
    """Trading position model with calculated properties.
    
    Represents an open trading position with all necessary fields
    for tracking, P&L calculation, and risk management.
    
    Attributes:
        position_id: Unique position identifier
        agent_id: Which agent opened this position
        symbol: Trading symbol
        side: Position side (long/short)
        quantity: Position size in lots
        entry_price: Price at which position was opened
        current_price: Current market price
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        unrealized_pnl: Current unrealized P&L
        opened_at: When the position was opened
        broker_id: Which broker holds this position
        metadata: Additional position metadata
    """
    
    # Required fields
    position_id: str
    agent_id: str
    symbol: str
    side: Literal['long', 'short']
    quantity: float
    entry_price: float
    current_price: float
    opened_at: datetime = field(default_factory=datetime.utcnow)
    broker_id: str = "default"
    
    # Optional fields
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Calculate initial unrealized P&L."""
        self.unrealized_pnl = self._calculate_unrealized_pnl()
    
    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L based on current price.
        
        Returns:
            Unrealized P&L in account currency
        """
        if self.side == 'long':
            return (self.current_price - self.entry_price) * self.quantity
        else:  # short
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L as percentage.
        
        Returns:
            Unrealized P&L as percentage of entry value
        """
        if self.entry_price == 0:
            return 0.0
        
        return (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100
    
    @property
    def position_value(self) -> float:
        """Calculate current position value.
        
        Returns:
            Current position value (quantity * current_price)
        """
        return self.quantity * self.current_price
    
    @property
    def entry_value(self) -> float:
        """Calculate entry position value.
        
        Returns:
            Entry position value (quantity * entry_price)
        """
        return self.quantity * self.entry_price
    
    @property
    def duration_hours(self) -> float:
        """Calculate how long position is open.
        
        Returns:
            Duration in hours since position was opened
        """
        delta = datetime.utcnow() - self.opened_at
        return delta.total_seconds() / 3600
    
    @property
    def duration_days(self) -> float:
        """Calculate how long position is open in days.
        
        Returns:
            Duration in days since position was opened
        """
        return self.duration_hours / 24
    
    @property
    def is_winning(self) -> bool:
        """Check if position is currently winning.
        
        Returns:
            True if unrealized P&L > 0, False otherwise
        """
        return self.unrealized_pnl > 0
    
    @property
    def is_losing(self) -> bool:
        """Check if position is currently losing.
        
        Returns:
            True if unrealized P&L < 0, False otherwise
        """
        return self.unrealized_pnl < 0
    
    @property
    def is_at_breakeven(self) -> bool:
        """Check if position is at breakeven.
        
        Returns:
            True if unrealized P&L == 0, False otherwise
        """
        return self.unrealized_pnl == 0
    
    @property
    def has_stop_loss(self) -> bool:
        """Check if position has stop loss.
        
        Returns:
            True if stop loss is set, False otherwise
        """
        return self.stop_loss is not None
    
    @property
    def has_take_profit(self) -> bool:
        """Check if position has take profit.
        
        Returns:
            True if take profit is set, False otherwise
        """
        return self.take_profit is not None
    
    @property
    def stop_loss_distance(self) -> Optional[float]:
        """Calculate distance to stop loss.
        
        Returns:
            Distance to stop loss in price units, None if no stop loss
        """
        if not self.has_stop_loss:
            return None
        
        if self.side == 'long':
            return self.entry_price - self.stop_loss
        else:  # short
            return self.stop_loss - self.entry_price
    
    @property
    def take_profit_distance(self) -> Optional[float]:
        """Calculate distance to take profit.
        
        Returns:
            Distance to take profit in price units, None if no take profit
        """
        if not self.has_take_profit:
            return None
        
        if self.side == 'long':
            return self.take_profit - self.entry_price
        else:  # short
            return self.entry_price - self.take_profit
    
    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk-reward ratio.
        
        Returns:
            Risk-reward ratio if both stop loss and take profit are set,
            None otherwise
        """
        if not self.has_stop_loss or not self.has_take_profit:
            return None
        
        risk_distance = self.stop_loss_distance
        reward_distance = self.take_profit_distance
        
        if risk_distance is None or reward_distance is None or risk_distance == 0:
            return None
        
        return reward_distance / risk_distance
    
    def update_current_price(self, new_price: float) -> None:
        """Update current price and recalculate P&L.
        
        Args:
            new_price: New current market price
            
        Raises:
            ValueError: If new_price is negative
        """
        if new_price < 0:
            raise ValueError(f"Price must be positive, got {new_price}")
        
        self.current_price = new_price
        self.unrealized_pnl = self._calculate_unrealized_pnl()
    
    def is_stop_loss_hit(self) -> bool:
        """Check if stop loss has been hit.
        
        Returns:
            True if current price has hit stop loss, False otherwise
        """
        if not self.has_stop_loss:
            return False
        
        if self.side == 'long':
            return self.current_price <= self.stop_loss
        else:  # short
            return self.current_price >= self.stop_loss
    
    def is_take_profit_hit(self) -> bool:
        """Check if take profit has been hit.
        
        Returns:
            True if current price has hit take profit, False otherwise
        """
        if not self.has_take_profit:
            return False
        
        if self.side == 'long':
            return self.current_price >= self.take_profit
        else:  # short
            return self.current_price <= self.take_profit
    
    def should_close(self) -> bool:
        """Check if position should be closed.
        
        Returns:
            True if stop loss or take profit has been hit, False otherwise
        """
        return self.is_stop_loss_hit() or self.is_take_profit_hit()
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary.
        
        Returns:
            Dictionary representation of the position
        """
        return {
            'position_id': self.position_id,
            'agent_id': self.agent_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'unrealized_pnl': self.unrealized_pnl,
            'opened_at': self.opened_at.isoformat(),
            'broker_id': self.broker_id,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        """Create position from dictionary.
        
        Args:
            data: Dictionary with position data
            
        Returns:
            Position instance
            
        Raises:
            ValueError: If required fields are missing
        """
        # Parse datetime
        if isinstance(data.get('opened_at'), str):
            data['opened_at'] = datetime.fromisoformat(data['opened_at'])
        
        # Remove None values for optional fields
        filtered_data = {k: v for k, v in data.items() if v is not None}
        
        return cls(**filtered_data)
    
    def __repr__(self) -> str:
        """Return string representation of the position."""
        return (
            f"Position(position_id={self.position_id}, agent_id={self.agent_id}, "
            f"symbol={self.symbol}, side={self.side}, quantity={self.quantity}, "
            f"entry_price={self.entry_price}, current_price={self.current_price}, "
            f"unrealized_pnl={self.unrealized_pnl:.2f})"
        )
    
    def __str__(self) -> str:
        """Return human-readable string representation."""
        pnl_sign = "+" if self.unrealized_pnl >= 0 else ""
        return (
            f"Position {self.position_id}: {self.side.upper()} {self.quantity} "
            f"{self.symbol} @ {self.entry_price} -> {self.current_price} "
            f"(P&L: {pnl_sign}{self.unrealized_pnl:.2f}, {self.unrealized_pnl_pct:+.2f}%)"
        )
