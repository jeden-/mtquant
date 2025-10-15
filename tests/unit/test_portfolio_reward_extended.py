"""
Extended unit tests for portfolio_reward.py to increase coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from mtquant.agents.training.portfolio_reward import (
    RewardComponent,
    RewardConfig,
    PortfolioRewardFunction
)
from mtquant.mcp_integration.models.position import Position


class TestRewardComponent:
    """Test RewardComponent enum."""
    
    def test_reward_component_values(self):
        """Test RewardComponent enum values."""
        assert RewardComponent.PORTFOLIO_RETURN.value == "portfolio_return"
        assert RewardComponent.RISK_ADJUSTED_RETURN.value == "risk_adjusted_return"
        assert RewardComponent.DIVERSIFICATION.value == "diversification"
        assert RewardComponent.ALLOCATION_STABILITY.value == "allocation_stability"
        assert RewardComponent.SPECIALIST_COORDINATION.value == "specialist_coordination"
        assert RewardComponent.RISK_MANAGEMENT.value == "risk_management"
        assert RewardComponent.TRANSACTION_COSTS.value == "transaction_costs"
        assert RewardComponent.DRAWDOWN_PENALTY.value == "drawdown_penalty"
    
    def test_reward_component_membership(self):
        """Test RewardComponent membership."""
        assert RewardComponent.PORTFOLIO_RETURN in RewardComponent
        assert RewardComponent.RISK_ADJUSTED_RETURN in RewardComponent
        assert RewardComponent.DIVERSIFICATION in RewardComponent
        assert RewardComponent.ALLOCATION_STABILITY in RewardComponent
        assert RewardComponent.SPECIALIST_COORDINATION in RewardComponent
        assert RewardComponent.RISK_MANAGEMENT in RewardComponent
        assert RewardComponent.TRANSACTION_COSTS in RewardComponent
        assert RewardComponent.DRAWDOWN_PENALTY in RewardComponent


class TestRewardConfig:
    """Test RewardConfig dataclass."""
    
    def test_reward_config_defaults(self):
        """Test RewardConfig default values."""
        config = RewardConfig()
        
        assert config.portfolio_return_weight == 1.0
        assert config.risk_adjusted_return_weight == 2.0
        assert config.diversification_weight == 0.5
        assert config.allocation_stability_weight == 0.3
        assert config.specialist_coordination_weight == 0.4
        assert config.risk_management_weight == 3.0
        assert config.transaction_cost_weight == 1.0
        assert config.drawdown_penalty_weight == 5.0
        assert config.target_sharpe_ratio == 2.0
        assert config.max_drawdown_threshold == 0.15
        assert config.var_confidence_level == 0.95
        assert config.min_specialist_allocation == 0.1
        assert config.max_specialist_allocation == 0.7
        assert config.target_correlation == 0.3
        assert config.allocation_change_threshold == 0.2
        assert config.max_allocation_volatility == 0.1
        assert config.performance_correlation_threshold == 0.8
        assert config.coordination_bonus_threshold == 0.6
    
    def test_reward_config_custom(self):
        """Test RewardConfig with custom values."""
        config = RewardConfig(
            portfolio_return_weight=2.0,
            risk_adjusted_return_weight=3.0,
            diversification_weight=1.0,
            allocation_stability_weight=0.5,
            specialist_coordination_weight=0.8,
            risk_management_weight=4.0,
            transaction_cost_weight=2.0,
            drawdown_penalty_weight=6.0,
            target_sharpe_ratio=2.5,
            max_drawdown_threshold=0.1,
            var_confidence_level=0.99,
            min_specialist_allocation=0.15,
            max_specialist_allocation=0.6,
            target_correlation=0.4,
            allocation_change_threshold=0.15,
            max_allocation_volatility=0.08,
            performance_correlation_threshold=0.7,
            coordination_bonus_threshold=0.5
        )
        
        assert config.portfolio_return_weight == 2.0
        assert config.risk_adjusted_return_weight == 3.0
        assert config.diversification_weight == 1.0
        assert config.allocation_stability_weight == 0.5
        assert config.specialist_coordination_weight == 0.8
        assert config.risk_management_weight == 4.0
        assert config.transaction_cost_weight == 2.0
        assert config.drawdown_penalty_weight == 6.0
        assert config.target_sharpe_ratio == 2.5
        assert config.max_drawdown_threshold == 0.1
        assert config.var_confidence_level == 0.99
        assert config.min_specialist_allocation == 0.15
        assert config.max_specialist_allocation == 0.6
        assert config.target_correlation == 0.4
        assert config.allocation_change_threshold == 0.15
        assert config.max_allocation_volatility == 0.08
        assert config.performance_correlation_threshold == 0.7
        assert config.coordination_bonus_threshold == 0.5


class TestPortfolioRewardFunction:
    """Test PortfolioRewardFunction class."""
    
    def test_portfolio_reward_function_initialization(self):
        """Test PortfolioRewardFunction initialization."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        assert reward_function.config == config
        assert reward_function.portfolio_risk_manager == portfolio_risk_manager
        assert isinstance(reward_function.portfolio_value_history, list)
        assert isinstance(reward_function.allocation_history, list)
        assert isinstance(reward_function.specialist_performance_history, dict)
        assert isinstance(reward_function.risk_metrics_history, list)
        assert isinstance(reward_function.equal_weight_baseline, list)
        assert isinstance(reward_function.baseline_returns, list)
    
    def test_calculate_reward(self):
        """Test calculating portfolio reward."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Create mock positions
        positions = [
            Position(
                position_id="pos_1",
                agent_id="forex",
                symbol="EURUSD",
                side="long",
                quantity=0.1,
                entry_price=1.1,
                current_price=1.105,
                opened_at=datetime.now()
            ),
            Position(
                position_id="pos_2",
                agent_id="commodities",
                symbol="XAUUSD",
                side="long",
                quantity=0.05,
                entry_price=2000.0,
                current_price=2010.0,
                opened_at=datetime.now()
            )
        ]
        
        allocation = np.array([0.4, 0.3, 0.3])  # forex, commodities, equity
        specialist_performance = {
            'forex': 0.05,
            'commodities': 0.03,
            'equity': 0.02
        }
        
        total_reward, component_dict = reward_function.calculate_reward(
            portfolio_value=100000.0,
            positions=positions,
            allocation=allocation,
            specialist_performance=specialist_performance,
            transaction_costs=100.0
        )
        
        assert isinstance(total_reward, float)
        assert isinstance(component_dict, dict)
        assert 'portfolio_return' in component_dict
        assert 'risk_adjusted_return' in component_dict
        assert 'diversification' in component_dict
        assert 'allocation_stability' in component_dict
        assert 'specialist_coordination' in component_dict
        assert 'risk_management' in component_dict
        assert 'transaction_costs' in component_dict
        assert 'drawdown_penalty' in component_dict
    
    def test_calculate_reward_second_call(self):
        """Test calculating portfolio reward on second call."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # First call
        positions = [
            Position(
                position_id="pos_1",
                agent_id="forex",
                symbol="EURUSD",
                side="long",
                quantity=0.1,
                entry_price=1.1,
                current_price=1.105,
                opened_at=datetime.now()
            )
        ]
        
        allocation = np.array([0.4, 0.3, 0.3])
        specialist_performance = {
            'forex': 0.05,
            'commodities': 0.03,
            'equity': 0.02
        }
        
        reward_function.calculate_reward(
            portfolio_value=100000.0,
            positions=positions,
            allocation=allocation,
            specialist_performance=specialist_performance,
            transaction_costs=100.0
        )
        
        # Second call
        total_reward, component_dict = reward_function.calculate_reward(
            portfolio_value=101000.0,  # Increased value
            positions=positions,
            allocation=allocation,
            specialist_performance=specialist_performance,
            transaction_costs=50.0
        )
        
        assert isinstance(total_reward, float)
        assert isinstance(component_dict, dict)
        assert len(reward_function.portfolio_value_history) == 2
        assert len(reward_function.allocation_history) == 2
    
    def test_update_history(self):
        """Test updating performance history."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        allocation = np.array([0.4, 0.3, 0.3])
        specialist_performance = {
            'forex': 0.05,
            'commodities': 0.03,
            'equity': 0.02
        }
        
        reward_function._update_history(100000.0, allocation, specialist_performance)
        
        assert len(reward_function.portfolio_value_history) == 1
        assert len(reward_function.allocation_history) == 1
        assert reward_function.portfolio_value_history[0] == 100000.0
        assert np.array_equal(reward_function.allocation_history[0], allocation)
        assert 'forex' in reward_function.specialist_performance_history
        assert 'commodities' in reward_function.specialist_performance_history
        assert 'equity' in reward_function.specialist_performance_history
        assert reward_function.specialist_performance_history['forex'][0] == 0.05
        assert reward_function.specialist_performance_history['commodities'][0] == 0.03
        assert reward_function.specialist_performance_history['equity'][0] == 0.02
    
    def test_calculate_portfolio_return_reward_insufficient_data(self):
        """Test calculating portfolio return reward with insufficient data."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # No data
        reward = reward_function._calculate_portfolio_return_reward()
        assert reward == 0.0
        
        # Only one data point
        reward_function.portfolio_value_history.append(100000.0)
        reward = reward_function._calculate_portfolio_return_reward()
        assert reward == 0.0
    
    def test_calculate_portfolio_return_reward_sufficient_data(self):
        """Test calculating portfolio return reward with sufficient data."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Add two data points
        reward_function.portfolio_value_history.extend([100000.0, 101000.0])
        
        reward = reward_function._calculate_portfolio_return_reward()
        
        assert reward == 0.01  # 1% return
    
    def test_calculate_risk_adjusted_return_reward_insufficient_data(self):
        """Test calculating risk-adjusted return reward with insufficient data."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # No data
        reward = reward_function._calculate_risk_adjusted_return_reward()
        assert reward == 0.0
        
        # Less than 20 data points
        reward_function.portfolio_value_history.extend([100000.0] * 10)
        reward = reward_function._calculate_risk_adjusted_return_reward()
        assert reward == 0.0
    
    def test_calculate_risk_adjusted_return_reward_sufficient_data(self):
        """Test calculating risk-adjusted return reward with sufficient data."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Add 20 data points with some variation
        base_value = 100000.0
        for i in range(20):
            value = base_value * (1 + 0.01 * i + 0.001 * np.random.normal())
            reward_function.portfolio_value_history.append(value)
        
        reward = reward_function._calculate_risk_adjusted_return_reward()
        
        assert isinstance(reward, float)
    
    def test_calculate_diversification_reward(self):
        """Test calculating diversification reward."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Test balanced allocation
        allocation = np.array([0.33, 0.33, 0.34])
        reward = reward_function._calculate_diversification_reward(allocation)
        
        assert isinstance(reward, float)
        assert reward > 0  # Should be positive for balanced allocation
    
    def test_calculate_diversification_reward_extreme_allocation(self):
        """Test calculating diversification reward with extreme allocation."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Test extreme allocation (below min threshold)
        allocation = np.array([0.05, 0.05, 0.9])  # Very unbalanced
        reward = reward_function._calculate_diversification_reward(allocation)
        
        assert isinstance(reward, float)
        assert reward < 0  # Should be negative for extreme allocation
    
    def test_calculate_allocation_stability_reward_insufficient_data(self):
        """Test calculating allocation stability reward with insufficient data."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # No data
        reward = reward_function._calculate_allocation_stability_reward()
        assert reward == 0.0
        
        # Only one data point
        reward_function.allocation_history.append(np.array([0.4, 0.3, 0.3]))
        reward = reward_function._calculate_allocation_stability_reward()
        assert reward == 0.0
    
    def test_calculate_allocation_stability_reward_sufficient_data(self):
        """Test calculating allocation stability reward with sufficient data."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Add two similar allocations
        reward_function.allocation_history.extend([
            np.array([0.4, 0.3, 0.3]),
            np.array([0.41, 0.29, 0.3])  # Small change
        ])
        
        reward = reward_function._calculate_allocation_stability_reward()
        
        assert isinstance(reward, float)
        assert reward > 0  # Should be positive for stable allocation
    
    def test_calculate_coordination_reward_insufficient_data(self):
        """Test calculating coordination reward with insufficient data."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # No data
        reward = reward_function._calculate_coordination_reward()
        assert reward == 0.0
        
        # Only one specialist
        reward_function.specialist_performance_history['forex'] = [0.05]
        reward = reward_function._calculate_coordination_reward()
        assert reward == 0.0
    
    def test_calculate_coordination_reward_sufficient_data(self):
        """Test calculating coordination reward with sufficient data."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Add performance data for multiple specialists
        reward_function.specialist_performance_history['forex'] = [0.05, 0.04, 0.06]
        reward_function.specialist_performance_history['commodities'] = [0.03, 0.02, 0.04]
        reward_function.specialist_performance_history['equity'] = [0.02, 0.01, 0.03]
        
        reward = reward_function._calculate_coordination_reward()
        
        assert isinstance(reward, float)
    
    def test_calculate_risk_management_reward_no_positions(self):
        """Test calculating risk management reward with no positions."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        reward = reward_function._calculate_risk_management_reward([])
        
        assert reward == 0.0
    
    def test_calculate_risk_management_reward_with_positions(self):
        """Test calculating risk management reward with positions."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        # Mock risk manager methods
        portfolio_risk_manager.calculate_var.return_value = Mock(var_pct=0.01, var_excess=0.0)
        portfolio_risk_manager.check_correlation_risk.return_value = (True, 0.0)
        portfolio_risk_manager.calculate_sector_allocation.return_value = {'forex': 0.4, 'commodities': 0.3, 'equity': 0.3}
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Add portfolio value history
        reward_function.portfolio_value_history.append(100000.0)
        
        positions = [
            Position(
                position_id="pos_1",
                agent_id="forex",
                symbol="EURUSD",
                side="long",
                quantity=0.1,
                entry_price=1.1,
                current_price=1.105,
                opened_at=datetime.now()
            )
        ]
        
        reward = reward_function._calculate_risk_management_reward(positions)
        
        assert isinstance(reward, float)
        portfolio_risk_manager.calculate_var.assert_called_once()
        portfolio_risk_manager.check_correlation_risk.assert_called_once()
        portfolio_risk_manager.calculate_sector_allocation.assert_called_once()
    
    def test_calculate_risk_management_reward_exception(self):
        """Test calculating risk management reward with exception."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        # Mock risk manager to raise exception
        portfolio_risk_manager.calculate_var.side_effect = Exception("Risk calculation failed")
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Add portfolio value history
        reward_function.portfolio_value_history.append(100000.0)
        
        positions = [
            Position(
                position_id="pos_1",
                agent_id="forex",
                symbol="EURUSD",
                side="long",
                quantity=0.1,
                entry_price=1.1,
                current_price=1.105,
                opened_at=datetime.now()
            )
        ]
        
        reward = reward_function._calculate_risk_management_reward(positions)
        
        assert reward == 0.0
    
    def test_calculate_transaction_cost_penalty(self):
        """Test calculating transaction cost penalty."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # No portfolio value history
        reward = reward_function._calculate_transaction_cost_penalty(100.0)
        assert reward == 0.0
        
        # With portfolio value history
        reward_function.portfolio_value_history.append(100000.0)
        reward = reward_function._calculate_transaction_cost_penalty(100.0)
        
        assert isinstance(reward, float)
        assert reward < 0  # Should be negative (penalty)
    
    def test_calculate_drawdown_penalty_insufficient_data(self):
        """Test calculating drawdown penalty with insufficient data."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # No data
        reward = reward_function._calculate_drawdown_penalty()
        assert reward == 0.0
        
        # Less than 20 data points
        reward_function.portfolio_value_history.extend([100000.0] * 10)
        reward = reward_function._calculate_drawdown_penalty()
        assert reward == 0.0
    
    def test_calculate_drawdown_penalty_sufficient_data(self):
        """Test calculating drawdown penalty with sufficient data."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Add 20 data points with drawdown
        base_value = 100000.0
        for i in range(20):
            if i < 10:
                value = base_value * (1 + 0.01 * i)  # Increasing
            else:
                value = base_value * (1 + 0.1 - 0.02 * (i - 10))  # Decreasing (drawdown)
            reward_function.portfolio_value_history.append(value)
        
        reward = reward_function._calculate_drawdown_penalty()
        
        assert isinstance(reward, float)
    
    def test_get_component_weight(self):
        """Test getting component weight."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Test all components
        assert reward_function._get_component_weight(RewardComponent.PORTFOLIO_RETURN) == 1.0
        assert reward_function._get_component_weight(RewardComponent.RISK_ADJUSTED_RETURN) == 2.0
        assert reward_function._get_component_weight(RewardComponent.DIVERSIFICATION) == 0.5
        assert reward_function._get_component_weight(RewardComponent.ALLOCATION_STABILITY) == 0.3
        assert reward_function._get_component_weight(RewardComponent.SPECIALIST_COORDINATION) == 0.4
        assert reward_function._get_component_weight(RewardComponent.RISK_MANAGEMENT) == 3.0
        assert reward_function._get_component_weight(RewardComponent.TRANSACTION_COSTS) == 1.0
        assert reward_function._get_component_weight(RewardComponent.DRAWDOWN_PENALTY) == 5.0
    
    def test_get_performance_metrics_insufficient_data(self):
        """Test getting performance metrics with insufficient data."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # No data
        metrics = reward_function.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) == 0
        
        # Only one data point
        reward_function.portfolio_value_history.append(100000.0)
        metrics = reward_function.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) == 0
    
    def test_get_performance_metrics_sufficient_data(self):
        """Test getting performance metrics with sufficient data."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Add portfolio value history
        base_value = 100000.0
        for i in range(10):
            value = base_value * (1 + 0.01 * i + 0.001 * np.random.normal())
            reward_function.portfolio_value_history.append(value)
        
        # Add allocation history
        for i in range(5):
            allocation = np.array([0.4 + 0.01 * i, 0.3, 0.3 - 0.01 * i])
            reward_function.allocation_history.append(allocation)
        
        # Add specialist performance history
        reward_function.specialist_performance_history['forex'] = [0.05, 0.04, 0.06, 0.03, 0.07]
        reward_function.specialist_performance_history['commodities'] = [0.03, 0.02, 0.04, 0.01, 0.05]
        
        metrics = reward_function.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'mean_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'current_drawdown' in metrics
        assert 'allocation_volatility' in metrics
        assert 'allocation_entropy' in metrics
        assert 'forex_mean_performance' in metrics
        assert 'forex_sharpe' in metrics
        assert 'commodities_mean_performance' in metrics
        assert 'commodities_sharpe' in metrics
    
    def test_reset(self):
        """Test resetting reward function state."""
        config = RewardConfig()
        portfolio_risk_manager = Mock()
        
        reward_function = PortfolioRewardFunction(config, portfolio_risk_manager)
        
        # Add some data
        reward_function.portfolio_value_history.extend([100000.0, 101000.0])
        reward_function.allocation_history.append(np.array([0.4, 0.3, 0.3]))
        reward_function.specialist_performance_history['forex'] = [0.05, 0.04]
        reward_function.risk_metrics_history.append({'var': 0.01})
        reward_function.equal_weight_baseline.extend([100000.0, 101000.0])
        reward_function.baseline_returns.extend([0.01, 0.02])
        
        reward_function.reset()
        
        # Check that all histories are cleared
        assert len(reward_function.portfolio_value_history) == 0
        assert len(reward_function.allocation_history) == 0
        assert len(reward_function.specialist_performance_history) == 0
        assert len(reward_function.risk_metrics_history) == 0
        assert len(reward_function.equal_weight_baseline) == 0
        assert len(reward_function.baseline_returns) == 0
