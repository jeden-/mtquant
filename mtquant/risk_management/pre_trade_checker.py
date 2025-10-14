"""
Pre-trade risk validation system.

Executes 6 validation checks in parallel for speed.
Target execution time: <50ms
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from mtquant.utils.logger import get_logger
from mtquant.utils.exceptions import (
    RiskViolationError,
    PositionSizeError,
    InsufficientMarginError
)


@dataclass
class ValidationResult:
    """Result of pre-trade validation."""
    is_valid: bool
    checks_passed: List[str]
    checks_failed: List[str]
    error_message: Optional[str]
    execution_time_ms: float


class PreTradeChecker:
    """
    Pre-trade risk validation system.
    
    Executes 6 validation checks in parallel:
    1. Price bands (±5-10% from last known)
    2. Position size limits (<5% ADV)
    3. Capital verification (sufficient margin)
    4. Portfolio exposure limits
    5. Regulatory compliance
    6. Correlation risk
    
    Target execution time: <50ms
    """
    
    def __init__(self, risk_limits: Dict):
        """
        Args:
            risk_limits: Loaded from config/risk-limits.yaml
        """
        self.limits = risk_limits
        self.logger = get_logger(__name__)
    
    async def validate(
        self, 
        order: Dict, 
        portfolio: Dict,
        current_positions: List[Dict],
        last_price: float
    ) -> ValidationResult:
        """
        Comprehensive pre-trade validation.
        
        Args:
            order: Order to validate
            portfolio: Current portfolio state (equity, positions, etc.)
            current_positions: List of open positions
            last_price: Last known market price for instrument
            
        Returns:
            ValidationResult with validation details
            
        Runs all checks in parallel for speed.
        """
        start_time = asyncio.get_event_loop().time()
        
        # Run all checks in parallel
        checks = await asyncio.gather(
            self.check_price_band(order, last_price),
            self.check_position_size(order, portfolio),
            self.check_capital_availability(order, portfolio),
            self.check_portfolio_exposure(order, current_positions, portfolio),
            self.check_regulatory_limits(order),
            self.check_correlation_risk(order, current_positions),
            return_exceptions=True
        )
        
        # Process results
        checks_passed = []
        checks_failed = []
        
        check_names = [
            'price_band', 'position_size', 'capital', 
            'exposure', 'regulatory', 'correlation'
        ]
        
        for i, result in enumerate(checks):
            check_name = check_names[i]
            
            if isinstance(result, Exception):
                checks_failed.append(f"{check_name}: {str(result)}")
            else:
                checks_passed.append(check_name)
        
        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        is_valid = len(checks_failed) == 0
        
        error_msg = None
        if not is_valid:
            error_msg = f"Validation failed: {', '.join(checks_failed)}"
            self.logger.warning(error_msg)
        
        return ValidationResult(
            is_valid=is_valid,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            error_message=error_msg,
            execution_time_ms=execution_time
        )
    
    async def check_price_band(self, order: Dict, last_price: float) -> bool:
        """Check if order price is within acceptable bands."""
        try:
            order_price = order.get('price', last_price)
            symbol = order.get('symbol', '')
            
            # Get price band limits for symbol
            price_band_pct = self.limits.get('price_bands', {}).get(symbol, 0.10)  # 10% default
            
            # Calculate bands
            upper_band = last_price * (1 + price_band_pct)
            lower_band = last_price * (1 - price_band_pct)
            
            if not (lower_band <= order_price <= upper_band):
                raise RiskViolationError(
                    f"Price {order_price} outside band [{lower_band:.2f}, {upper_band:.2f}] "
                    f"for {symbol}"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Price band check failed: {e}")
            raise
    
    async def check_position_size(self, order: Dict, portfolio: Dict) -> bool:
        """Check if position size is within limits."""
        try:
            quantity = abs(order.get('quantity', 0))
            symbol = order.get('symbol', '')
            portfolio_equity = portfolio.get('equity', 0)
            
            # Get position size limits
            max_position_pct = self.limits.get('max_position_pct', 0.05)  # 5% default
            max_position_value = portfolio_equity * max_position_pct
            
            # Calculate position value (simplified)
            position_value = quantity * order.get('price', 0)
            
            if position_value > max_position_value:
                raise PositionSizeError(
                    f"Position size {position_value:.2f} exceeds limit {max_position_value:.2f} "
                    f"({max_position_pct*100}% of portfolio)"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Position size check failed: {e}")
            raise
    
    async def check_capital_availability(self, order: Dict, portfolio: Dict) -> bool:
        """Check if sufficient capital/margin is available."""
        try:
            required_margin = order.get('required_margin', 0)
            free_margin = portfolio.get('free_margin', 0)
            
            if required_margin > free_margin:
                raise InsufficientMarginError(
                    f"Required margin {required_margin:.2f} exceeds free margin {free_margin:.2f}"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Capital availability check failed: {e}")
            raise
    
    async def check_portfolio_exposure(self, order: Dict, current_positions: List[Dict], portfolio: Dict) -> bool:
        """Check portfolio exposure limits."""
        try:
            symbol = order.get('symbol', '')
            portfolio_equity = portfolio.get('equity', 0)
            
            # Calculate current exposure
            current_exposure = sum(
                abs(pos.get('quantity', 0) * pos.get('current_price', 0))
                for pos in current_positions
            )
            
            # Add new position exposure
            new_exposure = abs(order.get('quantity', 0) * order.get('price', 0))
            total_exposure = current_exposure + new_exposure
            
            # Check exposure limits
            max_exposure_pct = self.limits.get('max_portfolio_exposure', 1.2)  # 120% default
            max_exposure_value = portfolio_equity * max_exposure_pct
            
            if total_exposure > max_exposure_value:
                raise RiskViolationError(
                    f"Total exposure {total_exposure:.2f} exceeds limit {max_exposure_value:.2f} "
                    f"({max_exposure_pct*100}% of portfolio)"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Portfolio exposure check failed: {e}")
            raise
    
    async def check_regulatory_limits(self, order: Dict) -> bool:
        """Check regulatory compliance limits."""
        try:
            symbol = order.get('symbol', '')
            quantity = abs(order.get('quantity', 0))
            
            # Get regulatory limits for symbol
            max_leverage = self.limits.get('regulatory', {}).get('max_leverage', 30)
            symbol_limits = self.limits.get('symbol_limits', {}).get(symbol, {})
            
            # Check symbol-specific limits
            if symbol_limits:
                max_quantity = symbol_limits.get('max_quantity', float('inf'))
                if quantity > max_quantity:
                    raise RiskViolationError(
                        f"Quantity {quantity} exceeds regulatory limit {max_quantity} for {symbol}"
                    )
            
            # Additional regulatory checks can be added here
            # (e.g., trading hours, instrument restrictions)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Regulatory limits check failed: {e}")
            raise
    
    async def check_correlation_risk(self, order: Dict, current_positions: List[Dict]) -> bool:
        """Check correlation risk with existing positions."""
        try:
            symbol = order.get('symbol', '')
            
            # Get correlation matrix (simplified implementation)
            correlation_matrix = self.limits.get('correlation_matrix', {})
            
            # Check if symbol has high correlation with existing positions
            for position in current_positions:
                pos_symbol = position.get('symbol', '')
                if pos_symbol != symbol:
                    correlation = correlation_matrix.get(f"{symbol}_{pos_symbol}", 0)
                    
                    # If correlation > 0.7, check if we're adding to same direction
                    if abs(correlation) > 0.7:
                        order_side = 1 if order.get('quantity', 0) > 0 else -1
                        pos_side = 1 if position.get('quantity', 0) > 0 else -1
                        
                        # If same direction and high correlation, flag as risk
                        if order_side == pos_side and correlation > 0.7:
                            raise RiskViolationError(
                                f"High correlation risk: {symbol} vs {pos_symbol} "
                                f"(correlation: {correlation:.2f})"
                            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Correlation risk check failed: {e}")
            raise
