"""
Unit tests for MetaControllerEnv
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from mtquant.agents.environments.meta_controller_env import MetaControllerEnv
from mtquant.agents.environments.hierarchical_env import EnvironmentConfig
from mtquant.agents.hierarchical.meta_controller import MetaController
from mtquant.agents.hierarchical.base_specialist import BaseSpecialist
from mtquant.agents.hierarchical.communication import CommunicationHub
from mtquant.risk_management.portfolio_risk_manager import PortfolioRiskManager


# Module-level fixtures
@pytest.fixture
def env_config():
    """Create environment configuration."""
    return EnvironmentConfig(
        instruments=['EURUSD', 'XAUUSD', 'SPX500'],
        timeframe='1H',
        lookback_window=100,
        initial_capital=100000.0,
        transaction_cost=0.003,
        max_position_size=0.1,
        max_portfolio_var=0.02,
        max_correlation_exposure=0.7,
        stop_loss_pct=0.02,
        risk_penalty_weight=2.0,
        transaction_cost_weight=1.0,
        diversification_bonus_weight=0.5,
        episode_length=100,
        warmup_steps=10
    )

@pytest.fixture
def market_data():
    """Create mock market data."""
    dates = pd.date_range('2024-01-01', periods=300, freq='H')
    return {
        'EURUSD': pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(1.0, 1.2, 300),
            'high': np.random.uniform(1.1, 1.3, 300),
            'low': np.random.uniform(0.9, 1.1, 300),
            'close': np.random.uniform(1.0, 1.2, 300),
            'volume': np.random.uniform(1000, 10000, 300)
        }),
        'XAUUSD': pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(2000, 2100, 300),
            'high': np.random.uniform(2050, 2150, 300),
            'low': np.random.uniform(1950, 2050, 300),
            'close': np.random.uniform(2000, 2100, 300),
            'volume': np.random.uniform(100, 1000, 300)
        }),
        'SPX500': pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(4000, 4500, 300),
            'high': np.random.uniform(4100, 4600, 300),
            'low': np.random.uniform(3900, 4400, 300),
            'close': np.random.uniform(4000, 4500, 300),
            'volume': np.random.uniform(1000000, 10000000, 300)
        })
    }

@pytest.fixture
def meta_controller():
    """Create mock MetaController."""
    controller = Mock(spec=MetaController)
    controller.get_portfolio_state.return_value = torch.randn(50)
    controller.detect_market_regime.return_value = 'bull'
    controller.calculate_kelly_allocation.return_value = torch.tensor([0.4, 0.3, 0.3])
    return controller

@pytest.fixture
def specialists():
    """Create mock specialists."""
    forex_specialist = Mock(spec=BaseSpecialist)
    forex_specialist.get_instruments.return_value = ['EURUSD']
    forex_specialist.calculate_confidence.return_value = 0.8
    
    commodities_specialist = Mock(spec=BaseSpecialist)
    commodities_specialist.get_instruments.return_value = ['XAUUSD']
    commodities_specialist.calculate_confidence.return_value = 0.7
    
    equity_specialist = Mock(spec=BaseSpecialist)
    equity_specialist.get_instruments.return_value = ['SPX500']
    equity_specialist.calculate_confidence.return_value = 0.9
    
    return {
        'forex': forex_specialist,
        'commodities': commodities_specialist,
        'equity': equity_specialist
    }

@pytest.fixture
def env_setup(env_config, market_data, meta_controller, specialists):
    """Setup environment for testing."""
    env = MetaControllerEnv(env_config, market_data, meta_controller, specialists)
    env.portfolio_value = env_config.initial_capital
    # Initialize episode_returns to avoid division by zero
    env.episode_returns = [env_config.initial_capital]
    return env


class TestMetaControllerEnvInitialization:
    """Test MetaControllerEnv initialization."""
    
    
    def test_meta_controller_env_initialization(self, env_config, market_data, meta_controller, specialists):
        """Test basic initialization."""
        env = MetaControllerEnv(env_config, market_data, meta_controller, specialists)
        
        assert env.meta_controller == meta_controller
        assert env.specialists == specialists
        assert len(env.specialist_performance) == 3
        assert len(env.allocation_history) == 0
        assert len(env.risk_appetite_history) == 0
        assert env.baseline_performance == 0.0
        assert env.meta_performance == 0.0
    
    def test_meta_controller_env_with_communication_hub(self, env_config, market_data, meta_controller, specialists):
        """Test initialization with communication hub."""
        communication_hub = Mock(spec=CommunicationHub)
        portfolio_risk_manager = Mock(spec=PortfolioRiskManager)
        
        env = MetaControllerEnv(
            env_config, 
            market_data, 
            meta_controller, 
            specialists,
            communication_hub,
            portfolio_risk_manager
        )
        
        assert env.communication_hub == communication_hub
        assert env.portfolio_risk_manager == portfolio_risk_manager


class TestMetaControllerEnvSpaces:
    """Test action and observation spaces."""
    
    def test_action_space_dimensions(self, env_setup):
        """Test action space dimensions."""
        env = env_setup
        
        assert env.action_space.shape == (4,)  # [allocation_forex, allocation_commodities, allocation_equity, risk_appetite]
        assert env.action_space.low.shape == (4,)
        assert env.action_space.high.shape == (4,)
        assert np.all(env.action_space.low == 0.0)
        assert np.all(env.action_space.high == 1.0)
    
    def test_observation_space_dimensions(self, env_setup):
        """Test observation space dimensions."""
        env = env_setup
        
        # Portfolio state (74) + specialist reports (3 * 10) + market regime (5) = 109
        expected_dim = 74 + (3 * 10) + 5
        assert env.observation_space.shape == (expected_dim,)


class TestMetaControllerEnvMethods:
    """Test MetaControllerEnv methods."""
    
    def test_get_observation(self, env_setup):
        """Test getting observation."""
        env = env_setup
        
        observation = env._get_observation()
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (env.observation_space.shape[0],)
    
    def test_get_portfolio_state_vector(self, env_setup):
        """Test portfolio state vector extraction."""
        env = env_setup
        
        state = env._get_portfolio_state_vector()
        assert isinstance(state, np.ndarray)
        assert state.shape == (74,)
    
    def test_get_specialist_reports(self, env_setup):
        """Test specialist reports extraction."""
        env = env_setup
        
        reports = env._get_specialist_reports()
        assert isinstance(reports, np.ndarray)
        assert reports.shape == (30,)  # 3 specialists * 10 metrics
    
    def test_get_market_regime_indicators(self, env_setup):
        """Test market regime indicators."""
        env = env_setup
        
        indicators = env._get_market_regime_indicators()
        assert isinstance(indicators, np.ndarray)
        assert indicators.shape == (5,)
    
    def test_execute_action(self, env_setup):
        """Test action execution."""
        env = env_setup
        
        # Mock action: [allocation_forex, allocation_commodities, allocation_equity, risk_appetite]
        action = np.array([0.3, 0.3, 0.4, 0.7])
        
        # Mock communication hub to avoid errors
        env.communication_hub = Mock()
        env.communication_hub.send_message = Mock()
        
        executed_orders = env._execute_action(action)
        
        # Should return empty list for meta-controller
        assert executed_orders == []
        
        # Check that allocation was stored
        assert len(env.allocation_history) == 1
        assert len(env.risk_appetite_history) == 1
        
        # Check allocation normalization (should sum to 1)
        allocation = env.allocation_history[0]
        assert np.isclose(np.sum(allocation), 1.0)
        assert len(allocation) == 3  # 3 specialists
    
    def test_calculate_reward(self, env_setup):
        """Test reward calculation."""
        env = env_setup
        
        # Set current step > 0 to avoid early return
        env.current_step = 1
        
        # Add some episode returns for reward calculation
        env.episode_returns = [100000, 101000, 102000]
        
        # Mock executed orders (empty for meta-controller)
        executed_orders = []
        
        reward = env._calculate_reward(executed_orders)
        
        assert isinstance(reward, float)
        # Reward should be positive for positive returns
        assert reward > 0


class TestMetaControllerEnvStep:
    """Test step method and episode management."""
    
    def test_step_basic(self, env_setup):
        """Test basic step execution."""
        env = env_setup
        
        # Create valid action
        action = np.array([0.3, 0.3, 0.4, 0.7])
        
        # Mock communication hub to avoid errors
        env.communication_hub = Mock()
        env.communication_hub.send_message = Mock()
        
        observation, reward, done, truncated, info = env.step(action)
        
        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_step_episode_completion(self, env_setup):
        """Test step when episode is complete."""
        env = env_setup
        
        # Set current step to episode length
        env.current_step = env.config.episode_length - 1
        
        # Create valid action
        action = np.array([0.3, 0.3, 0.4, 0.7])
        
        # Mock communication hub to avoid errors
        env.communication_hub = Mock()
        env.communication_hub.send_message = Mock()
        
        observation, reward, done, truncated, info = env.step(action)
        
        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_reset(self, env_setup):
        """Test environment reset."""
        env = env_setup
        
        # Set some state
        env.current_step = 5
        env.portfolio_value = 95000.0
        env.allocation_history = [np.array([0.3, 0.3, 0.4])]
        
        observation, info = env.reset()
        
        assert env.current_step == 0
        assert env.portfolio_value == env.config.initial_capital
        assert isinstance(observation, np.ndarray)
        assert isinstance(info, dict)


class TestMetaControllerEnvEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_action_dimensions(self, env_setup):
        """Test handling of invalid action dimensions."""
        env = env_setup
        
        # Action with wrong dimensions
        invalid_action = np.array([0.1, 0.2])  # Too few dimensions
        
        with pytest.raises((ValueError, IndexError)):
            env.step(invalid_action)
    
    def test_empty_specialists(self, env_config, market_data, meta_controller):
        """Test handling of empty specialists dictionary."""
        empty_specialists = {}
        
        # Should handle empty specialists gracefully
        env = MetaControllerEnv(env_config, market_data, meta_controller, empty_specialists)
        assert env is not None
        assert len(env.specialists) == 0
    
    def test_zero_portfolio_value(self, env_setup):
        """Test handling of zero portfolio value."""
        env = env_setup
        env.portfolio_value = 0.0
        
        # Should handle zero portfolio value gracefully
        state = env._get_portfolio_state_vector()
        assert isinstance(state, np.ndarray)
        assert state.shape == (74,)
    
    def test_negative_rewards(self, env_setup):
        """Test handling of negative rewards."""
        env = env_setup
        
        # Set current step > 0 to avoid early return
        env.current_step = 1
        
        # Add negative episode returns for negative reward calculation
        env.episode_returns = [100000, 99000, 98000]
        
        # Mock executed orders (empty for meta-controller)
        executed_orders = []
        
        reward = env._calculate_reward(executed_orders)
        
        assert isinstance(reward, float)
        # Reward should be negative for negative returns
        assert reward < 0
