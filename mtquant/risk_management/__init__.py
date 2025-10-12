"""
Risk Management Module

Comprehensive risk management system with three-tier defense:
1. Pre-trade validation (<50ms response time)
2. Intra-trade monitoring (continuous)
3. Portfolio-level circuit breakers (automatic)

Components:
- pre_trade_checker.py: Order validation before execution
- position_sizer.py: Position sizing algorithms (Kelly, volatility-based)
- circuit_breaker.py: Automatic trading halts on losses

Safety Features:
- Price band validation (±5-10% from last known)
- Position size limits (<5% Average Daily Volume)
- Capital verification (sufficient margin)
- Regulatory compliance (max leverage, restrictions)
- Correlation monitoring (reduce positions if ρ > 0.7)
"""

__version__ = "0.1.0"

# Risk management classes will be imported here when implemented
__all__ = []
