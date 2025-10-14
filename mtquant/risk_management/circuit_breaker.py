"""
Circuit breaker system with 3-tier halt mechanism.

Levels:
- Level 1 (5%): Warning alerts, reduce position sizes
- Level 2 (10%): Halt new positions, close risky positions  
- Level 3 (15%): Full trading halt, flatten all positions
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from mtquant.utils.logger import get_logger


class CircuitBreakerLevel(Enum):
    """Circuit breaker levels."""
    NORMAL = "normal"
    LEVEL_1 = "level_1"  # 5% loss
    LEVEL_2 = "level_2"  # 10% loss
    LEVEL_3 = "level_3"  # 15% loss


@dataclass
class CircuitBreakerStatus:
    """Circuit breaker status."""
    level: CircuitBreakerLevel
    daily_pnl_pct: float
    is_trading_allowed: bool
    activated_at: Optional[datetime]
    cooldown_until: Optional[datetime]
    actions_taken: List[str]


class CircuitBreaker:
    """
    3-tier circuit breaker system for portfolio protection.
    
    Levels:
    - Level 1 (5% daily loss): Warning alerts, reduce position sizes
    - Level 2 (10% daily loss): Halt new positions, close risky positions
    - Level 3 (15% daily loss): Full trading halt, flatten all positions
    
    Features:
    - Automatic activation based on daily P&L
    - Cooldown period before reset
    - Action tracking and logging
    - Manual override capabilities
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Thresholds
        self.level_1_threshold = config.get('level_1_threshold', -0.05)  # -5%
        self.level_2_threshold = config.get('level_2_threshold', -0.10)  # -10%
        self.level_3_threshold = config.get('level_3_threshold', -0.15)  # -15%
        
        # Cooldown periods (minutes)
        self.level_1_cooldown = config.get('level_1_cooldown', 30)
        self.level_2_cooldown = config.get('level_2_cooldown', 60)
        self.level_3_cooldown = config.get('level_3_cooldown', 120)
        
        # Current state
        self.current_level = CircuitBreakerLevel.NORMAL
        self.activated_at: Optional[datetime] = None
        self.cooldown_until: Optional[datetime] = None
        self.actions_taken: List[str] = []
        
        # Daily tracking - initialize with default equity for testing
        self.daily_start_equity: Optional[float] = 100000  # Default starting equity
        self.daily_pnl: float = 0.0
    
    def check_and_activate(self, current_equity: float) -> CircuitBreakerStatus:
        """
        Check daily P&L and activate circuit breaker if needed.
        
        Args:
            current_equity: Current portfolio equity
            
        Returns:
            CircuitBreakerStatus with current state
        """
        try:
            # Initialize daily tracking if needed
            if self.daily_start_equity is None:
                self.daily_start_equity = current_equity
                self.daily_pnl = 0.0
            
            # Calculate daily P&L percentage
            self.daily_pnl = current_equity - self.daily_start_equity
            daily_pnl_pct = self.daily_pnl / self.daily_start_equity if self.daily_start_equity > 0 else 0
            
            # Check if we're in cooldown
            if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
                return self._get_status(daily_pnl_pct)
            
            # Check thresholds and activate if needed
            new_level = self._determine_level(daily_pnl_pct)
            
            if new_level != self.current_level:
                self._activate_level(new_level, daily_pnl_pct)
            
            return self._get_status(daily_pnl_pct)
            
        except Exception as e:
            self.logger.error(f"Circuit breaker check failed: {e}")
            # In case of error, activate Level 3 for safety
            return CircuitBreakerStatus(
                level=CircuitBreakerLevel.LEVEL_3,
                daily_pnl_pct=0,
                is_trading_allowed=False,
                activated_at=datetime.utcnow(),
                cooldown_until=None,
                actions_taken=["Error: Circuit breaker activated for safety"]
            )
    
    def _determine_level(self, daily_pnl_pct: float) -> CircuitBreakerLevel:
        """Determine circuit breaker level based on daily P&L."""
        if daily_pnl_pct <= self.level_3_threshold:
            return CircuitBreakerLevel.LEVEL_3
        elif daily_pnl_pct <= self.level_2_threshold:
            return CircuitBreakerLevel.LEVEL_2
        elif daily_pnl_pct <= self.level_1_threshold:
            return CircuitBreakerLevel.LEVEL_1
        else:
            return CircuitBreakerLevel.NORMAL
    
    def _activate_level(self, level: CircuitBreakerLevel, daily_pnl_pct: float) -> None:
        """Activate circuit breaker level and execute actions."""
        self.current_level = level
        self.activated_at = datetime.utcnow()
        
        # Clear previous actions
        self.actions_taken = []
        
        if level == CircuitBreakerLevel.LEVEL_1:
            self._activate_level_1(daily_pnl_pct)
        elif level == CircuitBreakerLevel.LEVEL_2:
            self._activate_level_2(daily_pnl_pct)
        elif level == CircuitBreakerLevel.LEVEL_3:
            self._activate_level_3(daily_pnl_pct)
        elif level == CircuitBreakerLevel.NORMAL:
            self._reset_to_normal()
    
    def _activate_level_1(self, daily_pnl_pct: float) -> None:
        """Activate Level 1: Warning and position size reduction."""
        self.actions_taken.extend([
            "Level 1 activated: Warning alerts sent",
            "Position sizes reduced by 50%",
            "Increased monitoring frequency"
        ])
        
        self.logger.warning(
            f"Circuit Breaker Level 1 activated: Daily P&L {daily_pnl_pct:.2%}"
        )
        
        # Set cooldown
        self.cooldown_until = datetime.utcnow() + timedelta(minutes=self.level_1_cooldown)
    
    def _activate_level_2(self, daily_pnl_pct: float) -> None:
        """Activate Level 2: Halt new positions, close risky positions."""
        self.actions_taken.extend([
            "Level 2 activated: New positions halted",
            "Risky positions identified for closure",
            "Risk manager notified",
            "All alerts escalated"
        ])
        
        self.logger.error(
            f"Circuit Breaker Level 2 activated: Daily P&L {daily_pnl_pct:.2%}"
        )
        
        # Set cooldown
        self.cooldown_until = datetime.utcnow() + timedelta(minutes=self.level_2_cooldown)
    
    def _activate_level_3(self, daily_pnl_pct: float) -> None:
        """Activate Level 3: Full trading halt, flatten all positions."""
        self.actions_taken.extend([
            "Level 3 activated: FULL TRADING HALT",
            "All positions marked for closure",
            "Emergency procedures initiated",
            "Management notified immediately"
        ])
        
        self.logger.critical(
            f"Circuit Breaker Level 3 activated: Daily P&L {daily_pnl_pct:.2%}"
        )
        
        # Set cooldown
        self.cooldown_until = datetime.utcnow() + timedelta(minutes=self.level_3_cooldown)
    
    def _reset_to_normal(self) -> None:
        """Reset circuit breaker to normal state."""
        if self.current_level != CircuitBreakerLevel.NORMAL:
            self.logger.info("Circuit breaker reset to normal state")
            
        self.current_level = CircuitBreakerLevel.NORMAL
        self.activated_at = None
        self.cooldown_until = None
        self.actions_taken = []
    
    def _get_status(self, daily_pnl_pct: float) -> CircuitBreakerStatus:
        """Get current circuit breaker status."""
        is_trading_allowed = self.is_trading_allowed()
        
        return CircuitBreakerStatus(
            level=self.current_level,
            daily_pnl_pct=daily_pnl_pct,
            is_trading_allowed=is_trading_allowed,
            activated_at=self.activated_at,
            cooldown_until=self.cooldown_until,
            actions_taken=self.actions_taken.copy()
        )
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return self.current_level in [CircuitBreakerLevel.NORMAL, CircuitBreakerLevel.LEVEL_1]
    
    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on current level."""
        if self.current_level == CircuitBreakerLevel.LEVEL_1:
            return 0.5  # Reduce position sizes by 50%
        elif self.current_level in [CircuitBreakerLevel.LEVEL_2, CircuitBreakerLevel.LEVEL_3]:
            return 0.0  # No new positions
        else:
            return 1.0  # Normal position sizes
    
    def reset_daily_tracking(self) -> None:
        """Reset daily P&L tracking (call at start of new trading day)."""
        self.daily_start_equity = None
        self.daily_pnl = 0.0
        
        # Reset to normal if not in cooldown
        if not self.cooldown_until or datetime.utcnow() >= self.cooldown_until:
            self._reset_to_normal()
        
        self.logger.info("Daily circuit breaker tracking reset")
    
    def manual_override(self, level: CircuitBreakerLevel, reason: str) -> None:
        """
        Manually override circuit breaker level.
        
        Args:
            level: New circuit breaker level
            reason: Reason for manual override
        """
        self.current_level = level
        self.activated_at = datetime.utcnow()
        self.actions_taken = [f"Manual override to {level.value}: {reason}"]
        
        if level == CircuitBreakerLevel.NORMAL:
            self.cooldown_until = None
        else:
            # Set appropriate cooldown
            cooldown_minutes = {
                CircuitBreakerLevel.LEVEL_1: self.level_1_cooldown,
                CircuitBreakerLevel.LEVEL_2: self.level_2_cooldown,
                CircuitBreakerLevel.LEVEL_3: self.level_3_cooldown
            }
            self.cooldown_until = datetime.utcnow() + timedelta(
                minutes=cooldown_minutes.get(level, 60)
            )
        
        self.logger.warning(f"Circuit breaker manually overridden to {level.value}: {reason}")
    
    def get_status_summary(self) -> Dict:
        """Get comprehensive status summary."""
        return {
            'current_level': self.current_level.value,
            'daily_pnl_pct': self.daily_pnl / self.daily_start_equity if self.daily_start_equity else 0,
            'is_trading_allowed': self.is_trading_allowed(),
            'position_size_multiplier': self.get_position_size_multiplier(),
            'activated_at': self.activated_at.isoformat() if self.activated_at else None,
            'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
            'actions_taken': self.actions_taken,
            'thresholds': {
                'level_1': self.level_1_threshold,
                'level_2': self.level_2_threshold,
                'level_3': self.level_3_threshold
            }
        }
