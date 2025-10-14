"""
Intelligent position sizing strategies.

Supports Kelly Criterion, Volatility-based, and Fixed Fractional methods.
"""

import math
from typing import Dict, Literal, Optional
from dataclasses import dataclass
from enum import Enum
from mtquant.utils.logger import get_logger


class PositionSizingMethod(Enum):
    """Position sizing methods."""
    KELLY = "kelly"
    VOLATILITY = "volatility"
    FIXED = "fixed"


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation."""
    method: str
    position_size: float
    confidence: float
    risk_per_trade: float
    reasoning: str


class PositionSizer:
    """
    Intelligent position sizing with multiple strategies.
    
    Methods:
    1. Kelly Criterion - Optimal position size based on win rate and payoff ratio
    2. Volatility-based - Position size based on instrument volatility (ATR)
    3. Fixed Fractional - Fixed percentage of portfolio per trade
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Position sizing configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def calculate(
        self,
        signal: float,
        portfolio_equity: float,
        instrument_volatility: float,
        method: Literal['kelly', 'volatility', 'fixed'] = 'kelly',
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> PositionSizingResult:
        """
        Calculate position size using specified method.
        
        Args:
            signal: RL model output (-1 to 1, where 0=flat, positive=long, negative=short)
            portfolio_equity: Total portfolio value in USD
            instrument_volatility: ATR or realized volatility
            method: Position sizing method to use
            win_rate: Win rate for Kelly Criterion (optional)
            avg_win: Average win amount for Kelly Criterion (optional)
            avg_loss: Average loss amount for Kelly Criterion (optional)
            
        Returns:
            PositionSizingResult with calculated position size
            
        Raises:
            ValueError: If signal is outside valid range
            RiskViolationError: If calculated size exceeds limits
        """
        if not -1 <= signal <= 1:
            raise ValueError(f"Signal must be -1 to 1, got {signal}")
        
        # Scale signal to 0-1 range for position sizing
        signal_strength = abs(signal)
        
        if method == 'kelly':
            return self._kelly_criterion(
                signal_strength, portfolio_equity, win_rate, avg_win, avg_loss
            )
        elif method == 'volatility':
            return self._volatility_based(
                signal_strength, portfolio_equity, instrument_volatility
            )
        elif method == 'fixed':
            return self._fixed_fractional(
                signal_strength, portfolio_equity
            )
        else:
            raise ValueError(f"Unknown position sizing method: {method}")
    
    def _kelly_criterion(
        self,
        signal_strength: float,
        portfolio_equity: float,
        win_rate: Optional[float],
        avg_win: Optional[float],
        avg_loss: Optional[float]
    ) -> PositionSizingResult:
        """Kelly Criterion position sizing."""
        try:
            # Use default values if not provided
            win_rate = win_rate or self.config.get('kelly', {}).get('default_win_rate', 0.55)
            avg_win = avg_win or self.config.get('kelly', {}).get('default_avg_win', 100)
            avg_loss = avg_loss or self.config.get('kelly', {}).get('default_avg_loss', 80)
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply fractional Kelly (typically 0.25x for safety)
            fractional_kelly = self.config.get('kelly', {}).get('fraction', 0.25)
            kelly_fraction *= fractional_kelly
            
            # Scale by signal strength
            kelly_fraction *= signal_strength
            
            # Ensure positive and within limits
            kelly_fraction = max(0, min(kelly_fraction, 0.1))  # Max 10% per trade
            
            position_size = portfolio_equity * kelly_fraction
            risk_per_trade = kelly_fraction
            
            reasoning = (
                f"Kelly Criterion: win_rate={win_rate:.2f}, "
                f"avg_win={avg_win:.2f}, avg_loss={avg_loss:.2f}, "
                f"kelly_fraction={kelly_fraction:.3f}"
            )
            
            return PositionSizingResult(
                method="kelly",
                position_size=position_size,
                confidence=signal_strength,
                risk_per_trade=risk_per_trade,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Kelly Criterion calculation failed: {e}")
            # Fallback to fixed fractional
            return self._fixed_fractional(signal_strength, portfolio_equity)
    
    def _volatility_based(
        self,
        signal_strength: float,
        portfolio_equity: float,
        instrument_volatility: float
    ) -> PositionSizingResult:
        """Volatility-based position sizing."""
        try:
            # Risk per trade percentage
            risk_per_trade = self.config.get('volatility', {}).get('risk_per_trade', 0.02)  # 2%
            
            # Risk amount in dollars
            risk_amount = portfolio_equity * risk_per_trade
            
            # Position size = Risk Amount / (ATR * Multiplier)
            atr_multiplier = self.config.get('volatility', {}).get('atr_multiplier', 2.0)
            position_size = risk_amount / (instrument_volatility * atr_multiplier)
            
            # Scale by signal strength
            position_size *= signal_strength
            
            # Apply limits
            max_position_pct = self.config.get('volatility', {}).get('max_position_pct', 0.05)
            max_position_size = portfolio_equity * max_position_pct
            position_size = min(position_size, max_position_size)
            
            reasoning = (
                f"Volatility-based: ATR={instrument_volatility:.2f}, "
                f"risk_per_trade={risk_per_trade:.1%}, "
                f"multiplier={atr_multiplier}"
            )
            
            return PositionSizingResult(
                method="volatility",
                position_size=position_size,
                confidence=signal_strength,
                risk_per_trade=risk_per_trade,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Volatility-based calculation failed: {e}")
            # Fallback to fixed fractional
            return self._fixed_fractional(signal_strength, portfolio_equity)
    
    def _fixed_fractional(
        self,
        signal_strength: float,
        portfolio_equity: float
    ) -> PositionSizingResult:
        """Fixed fractional position sizing."""
        try:
            # Fixed percentage of portfolio per trade
            fixed_fraction = self.config.get('fixed', {}).get('fraction', 0.02)  # 2%
            
            # Scale by signal strength
            position_size = portfolio_equity * fixed_fraction * signal_strength
            
            # Apply limits
            max_position_pct = self.config.get('fixed', {}).get('max_position_pct', 0.05)
            max_position_size = portfolio_equity * max_position_pct
            position_size = min(position_size, max_position_size)
            
            risk_per_trade = fixed_fraction * signal_strength
            
            reasoning = (
                f"Fixed fractional: fraction={fixed_fraction:.1%}, "
                f"signal_strength={signal_strength:.2f}"
            )
            
            return PositionSizingResult(
                method="fixed",
                position_size=position_size,
                confidence=signal_strength,
                risk_per_trade=risk_per_trade,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Fixed fractional calculation failed: {e}")
            # Emergency fallback
            return PositionSizingResult(
                method="fixed",
                position_size=portfolio_equity * 0.01,  # 1% emergency size
                confidence=0.1,
                risk_per_trade=0.01,
                reasoning="Emergency fallback position size"
            )
    
    def get_recommended_method(self, portfolio_history: Dict) -> str:
        """
        Recommend position sizing method based on portfolio history.
        
        Args:
            portfolio_history: Historical performance data
            
        Returns:
            Recommended method name
        """
        try:
            # Simple heuristic based on portfolio performance
            sharpe_ratio = portfolio_history.get('sharpe_ratio', 0)
            win_rate = portfolio_history.get('win_rate', 0.5)
            volatility = portfolio_history.get('volatility', 0.2)
            
            # High Volatility = Volatility-based (check first)
            if volatility > 0.3:
                return 'volatility'
            
            # High Sharpe + High Win Rate = Kelly Criterion
            elif sharpe_ratio > 1.5 and win_rate > 0.55:
                return 'kelly'
            
            # Default = Fixed fractional
            else:
                return 'fixed'
                
        except Exception as e:
            self.logger.warning(f"Method recommendation failed: {e}")
            return 'fixed'
