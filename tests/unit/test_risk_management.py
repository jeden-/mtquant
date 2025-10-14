"""
Unit tests for risk management system.

Tests PreTradeChecker, PositionSizer, and CircuitBreaker.
Target: 11+ tests, >80% coverage
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import yaml
import os

from mtquant.risk_management.pre_trade_checker import PreTradeChecker, ValidationResult
from mtquant.risk_management.position_sizer import PositionSizer, PositionSizingResult
from mtquant.risk_management.circuit_breaker import CircuitBreaker, CircuitBreakerLevel


@pytest.fixture
def risk_limits_config():
    """Load risk limits configuration."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'risk-limits.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def pre_trade_checker(risk_limits_config):
    """Create PreTradeChecker instance."""
    return PreTradeChecker(risk_limits_config)


@pytest.fixture
def position_sizer(risk_limits_config):
    """Create PositionSizer instance."""
    return PositionSizer(risk_limits_config['position_sizing'])


@pytest.fixture
def circuit_breaker(risk_limits_config):
    """Create CircuitBreaker instance."""
    return CircuitBreaker(risk_limits_config['circuit_breaker'])


class TestPreTradeChecker:
    """Test PreTradeChecker functionality."""
    
    @pytest.mark.asyncio
    async def test_price_band_check_pass(self, pre_trade_checker):
        """Test price band check passes within limits."""
        order = {'symbol': 'XAUUSD', 'price': 2000.0, 'quantity': 1}
        last_price = 2000.0
        
        result = await pre_trade_checker.check_price_band(order, last_price)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_price_band_check_fail(self, pre_trade_checker):
        """Test price band check fails outside limits."""
        order = {'symbol': 'XAUUSD', 'price': 2200.0, 'quantity': 1}
        last_price = 2000.0  # 10% increase, exceeds 5% limit
        
        with pytest.raises(Exception):
            await pre_trade_checker.check_price_band(order, last_price)
    
    @pytest.mark.asyncio
    async def test_position_size_check_pass(self, pre_trade_checker):
        """Test position size check passes within limits."""
        order = {'symbol': 'XAUUSD', 'price': 2000.0, 'quantity': 1}
        portfolio = {'equity': 100000}  # 1% position size
        
        result = await pre_trade_checker.check_position_size(order, portfolio)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_position_size_check_fail(self, pre_trade_checker):
        """Test position size check fails exceeding limits."""
        order = {'symbol': 'XAUUSD', 'price': 2000.0, 'quantity': 10}
        portfolio = {'equity': 100000}  # 20% position size, exceeds 5% limit
        
        with pytest.raises(Exception):
            await pre_trade_checker.check_position_size(order, portfolio)
    
    @pytest.mark.asyncio
    async def test_capital_availability_check_pass(self, pre_trade_checker):
        """Test capital availability check passes."""
        order = {'required_margin': 1000}
        portfolio = {'free_margin': 5000}
        
        result = await pre_trade_checker.check_capital_availability(order, portfolio)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_capital_availability_check_fail(self, pre_trade_checker):
        """Test capital availability check fails."""
        order = {'required_margin': 10000}
        portfolio = {'free_margin': 5000}
        
        with pytest.raises(Exception):
            await pre_trade_checker.check_capital_availability(order, portfolio)
    
    @pytest.mark.asyncio
    async def test_full_validation_pass(self, pre_trade_checker):
        """Test full validation passes all checks."""
        order = {
            'symbol': 'XAUUSD',
            'price': 2000.0,
            'quantity': 1,
            'required_margin': 1000
        }
        portfolio = {'equity': 100000, 'free_margin': 5000}
        positions = []
        last_price = 2000.0
        
        result = await pre_trade_checker.validate(order, portfolio, positions, last_price)
        
        assert result.is_valid is True
        assert len(result.checks_passed) == 6
        assert len(result.checks_failed) == 0
        assert result.execution_time_ms < 50  # Should be fast
    
    @pytest.mark.asyncio
    async def test_full_validation_fail(self, pre_trade_checker):
        """Test full validation fails on multiple checks."""
        order = {
            'symbol': 'XAUUSD',
            'price': 2200.0,  # Price band violation
            'quantity': 10,   # Position size violation
            'required_margin': 10000  # Margin violation
        }
        portfolio = {'equity': 100000, 'free_margin': 5000}
        positions = []
        last_price = 2000.0
        
        result = await pre_trade_checker.validate(order, portfolio, positions, last_price)
        
        assert result.is_valid is False
        assert len(result.checks_failed) > 0
        assert result.error_message is not None


class TestPositionSizer:
    """Test PositionSizer functionality."""
    
    def test_kelly_criterion_calculation(self, position_sizer):
        """Test Kelly Criterion position sizing."""
        result = position_sizer.calculate(
            signal=0.8,
            portfolio_equity=100000,
            instrument_volatility=20.0,
            method='kelly',
            win_rate=0.6,
            avg_win=120,
            avg_loss=80
        )
        
        assert result.method == 'kelly'
        assert result.position_size > 0
        assert result.confidence == 0.8
        assert 'Kelly Criterion' in result.reasoning
    
    def test_volatility_based_calculation(self, position_sizer):
        """Test volatility-based position sizing."""
        result = position_sizer.calculate(
            signal=0.6,
            portfolio_equity=100000,
            instrument_volatility=25.0,
            method='volatility'
        )
        
        assert result.method == 'volatility'
        assert result.position_size > 0
        assert result.confidence == 0.6
        assert 'Volatility-based' in result.reasoning
    
    def test_fixed_fractional_calculation(self, position_sizer):
        """Test fixed fractional position sizing."""
        result = position_sizer.calculate(
            signal=0.5,
            portfolio_equity=100000,
            instrument_volatility=15.0,
            method='fixed'
        )
        
        assert result.method == 'fixed'
        assert result.position_size > 0
        assert result.confidence == 0.5
        assert 'Fixed fractional' in result.reasoning
    
    def test_signal_scaling(self, position_sizer):
        """Test signal scaling affects position size."""
        # Strong signal
        result_strong = position_sizer.calculate(
            signal=1.0,
            portfolio_equity=100000,
            instrument_volatility=20.0,
            method='fixed'
        )
        
        # Weak signal
        result_weak = position_sizer.calculate(
            signal=0.2,
            portfolio_equity=100000,
            instrument_volatility=20.0,
            method='fixed'
        )
        
        assert result_strong.position_size > result_weak.position_size
        assert result_strong.confidence > result_weak.confidence
    
    def test_position_size_limits(self, position_sizer):
        """Test position size limits are enforced."""
        result = position_sizer.calculate(
            signal=1.0,  # Maximum signal
            portfolio_equity=100000,
            instrument_volatility=10.0,  # Low volatility
            method='volatility'
        )
        
        # Should not exceed 5% of portfolio (5000)
        assert result.position_size <= 5000
    
    def test_method_recommendation(self, position_sizer):
        """Test method recommendation based on portfolio history."""
        # High Sharpe + High Win Rate = Kelly
        portfolio_history = {
            'sharpe_ratio': 2.0,
            'win_rate': 0.65,
            'volatility': 0.15
        }
        
        method = position_sizer.get_recommended_method(portfolio_history)
        assert method == 'kelly'
        
        # High Volatility = Volatility-based
        portfolio_history['volatility'] = 0.35
        method = position_sizer.get_recommended_method(portfolio_history)
        assert method == 'volatility'
        
        # Default = Fixed
        portfolio_history['sharpe_ratio'] = 0.5
        portfolio_history['volatility'] = 0.2
        method = position_sizer.get_recommended_method(portfolio_history)
        assert method == 'fixed'


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    def test_normal_state(self, circuit_breaker):
        """Test circuit breaker in normal state."""
        status = circuit_breaker.check_and_activate(100000)
        
        assert status.level == CircuitBreakerLevel.NORMAL
        assert status.is_trading_allowed is True
        assert status.daily_pnl_pct == 0
    
    def test_level_1_activation(self, circuit_breaker):
        """Test Level 1 activation at -5% loss."""
        # Simulate -5% loss
        current_equity = 95000  # 5% loss from 100000
        status = circuit_breaker.check_and_activate(current_equity)
        
        assert status.level == CircuitBreakerLevel.LEVEL_1
        assert status.is_trading_allowed is True  # Still allowed but reduced
        assert status.daily_pnl_pct <= -0.05
        assert len(status.actions_taken) > 0
    
    def test_level_2_activation(self, circuit_breaker):
        """Test Level 2 activation at -10% loss."""
        # Simulate -10% loss
        current_equity = 90000  # 10% loss from 100000
        status = circuit_breaker.check_and_activate(current_equity)
        
        assert status.level == CircuitBreakerLevel.LEVEL_2
        assert status.is_trading_allowed is False
        assert status.daily_pnl_pct <= -0.10
        assert len(status.actions_taken) > 0
    
    def test_level_3_activation(self, circuit_breaker):
        """Test Level 3 activation at -15% loss."""
        # Simulate -15% loss
        current_equity = 85000  # 15% loss from 100000
        status = circuit_breaker.check_and_activate(current_equity)
        
        assert status.level == CircuitBreakerLevel.LEVEL_3
        assert status.is_trading_allowed is False
        assert status.daily_pnl_pct <= -0.15
        assert len(status.actions_taken) > 0
    
    def test_position_size_multiplier(self, circuit_breaker):
        """Test position size multiplier based on level."""
        # Normal state
        assert circuit_breaker.get_position_size_multiplier() == 1.0
        
        # Level 1 - reduce by 50%
        circuit_breaker.current_level = CircuitBreakerLevel.LEVEL_1
        assert circuit_breaker.get_position_size_multiplier() == 0.5
        
        # Level 2/3 - no new positions
        circuit_breaker.current_level = CircuitBreakerLevel.LEVEL_2
        assert circuit_breaker.get_position_size_multiplier() == 0.0
        
        circuit_breaker.current_level = CircuitBreakerLevel.LEVEL_3
        assert circuit_breaker.get_position_size_multiplier() == 0.0
    
    def test_cooldown_period(self, circuit_breaker):
        """Test cooldown period before reset."""
        # Activate Level 1
        circuit_breaker.check_and_activate(95000)
        
        # Should be in cooldown
        assert circuit_breaker.cooldown_until is not None
        assert circuit_breaker.current_level == CircuitBreakerLevel.LEVEL_1
        
        # Reset daily tracking should not reset if in cooldown
        circuit_breaker.reset_daily_tracking()
        assert circuit_breaker.current_level == CircuitBreakerLevel.LEVEL_1
    
    def test_manual_override(self, circuit_breaker):
        """Test manual override functionality."""
        circuit_breaker.manual_override(
            CircuitBreakerLevel.LEVEL_2,
            "Manual risk management decision"
        )
        
        assert circuit_breaker.current_level == CircuitBreakerLevel.LEVEL_2
        assert circuit_breaker.is_trading_allowed() is False
        assert "Manual override" in circuit_breaker.actions_taken[0]
    
    def test_status_summary(self, circuit_breaker):
        """Test comprehensive status summary."""
        status_summary = circuit_breaker.get_status_summary()
        
        assert 'current_level' in status_summary
        assert 'is_trading_allowed' in status_summary
        assert 'position_size_multiplier' in status_summary
        assert 'thresholds' in status_summary
        assert 'actions_taken' in status_summary


# Performance tests
@pytest.mark.performance
class TestRiskManagementPerformance:
    """Test performance requirements."""
    
    @pytest.mark.asyncio
    async def test_pre_trade_checker_latency(self, pre_trade_checker):
        """Verify PreTradeChecker executes in <50ms."""
        order = {
            'symbol': 'XAUUSD',
            'price': 2000.0,
            'quantity': 1,
            'required_margin': 1000
        }
        portfolio = {'equity': 100000, 'free_margin': 5000}
        positions = []
        last_price = 2000.0
        
        # Run multiple iterations to get average
        latencies = []
        for _ in range(10):
            start_time = asyncio.get_event_loop().time()
            await pre_trade_checker.validate(order, portfolio, positions, last_price)
            latency = (asyncio.get_event_loop().time() - start_time) * 1000
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Max latency: {max_latency:.2f}ms")
        
        assert avg_latency < 50, f"Average latency {avg_latency:.2f}ms exceeds 50ms"
        assert max_latency < 100, f"Max latency {max_latency:.2f}ms exceeds 100ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=mtquant.risk_management"])
