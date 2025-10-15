"""
Unit tests for JointTrainingEnv
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from mtquant.agents.environments.joint_training_env import (
    JointTrainingEnv, 
    JointTrainingConfig
)
from mtquant.agents.hierarchical.meta_controller import MetaController
from mtquant.agents.hierarchical.base_specialist import BaseSpecialist
from mtquant.agents.hierarchical.communication import CommunicationHub
from mtquant.risk_management.portfolio_risk_manager import PortfolioRiskManager
from mtquant.mcp_integration.models.order import Order


class TestJointTrainingConfig:
    """Test JointTrainingConfig dataclass."""
    
    def test_joint_training_config_defaults(self):
        """Test default configuration values."""
        config = JointTrainingConfig()
        
        assert config.initial_capital == 100000.0
        assert config.transaction_cost == 0.003
        assert config.max_position_size == 0.1
        assert config.max_portfolio_var == 0.02
        assert config.max_correlation_exposure == 0.7
        assert config.max_sector_allocation == 0.4
        assert config.meta_update_freq == 1
        assert config.specialist_update_freq == 5
        assert config.coordination_reward_weight == 0.5
        assert config.individual_reward_weight == 0.5
        assert config.curriculum_enabled == True
        assert config.episode_length == 1000
        assert config.warmup_steps == 50
    
    def test_joint_training_config_post_init(self):
        """Test post_init sets default values."""
        config = JointTrainingConfig()
        
        assert config.difficulty_progression == ['easy', 'medium', 'hard']
        assert config.scenario_weights == {'easy': 0.4, 'medium': 0.4, 'hard': 0.2}
    
    def test_joint_training_config_custom_values(self):
        """Test custom configuration values."""
        config = JointTrainingConfig(
            initial_capital=50000.0,
            transaction_cost=0.002,
            max_position_size=0.05,
            episode_length=500,
            warmup_steps=25
        )
        
        assert config.initial_capital == 50000.0
        assert config.transaction_cost == 0.002
        assert config.max_position_size == 0.05
        assert config.episode_length == 500
        assert config.warmup_steps == 25


class TestJointTrainingEnvInitialization:
    """Test JointTrainingEnv initialization."""
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        dates = pd.date_range('2024-01-01', periods=300, freq='H')  # Increased to 300
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
            })
        }
    
    @pytest.fixture
    def mock_meta_controller(self):
        """Create mock meta controller."""
        controller = Mock(spec=MetaController)
        controller.forward.return_value = (
            torch.tensor([[0.3, 0.3, 0.4]]),  # allocations
            torch.tensor([[0.7]]),  # risk_appetite
            torch.tensor([[0.5]])  # value
        )
        return controller
    
    @pytest.fixture
    def mock_specialists(self):
        """Create mock specialists."""
        forex_specialist = Mock(spec=BaseSpecialist)
        forex_specialist.get_instruments.return_value = ['EURUSD', 'GBPUSD']
        forex_specialist.forward.return_value = (
            torch.tensor([[0.1, 0.2]]),  # actions
            torch.tensor([[0.6]])  # value
        )
        
        commodities_specialist = Mock(spec=BaseSpecialist)
        commodities_specialist.get_instruments.return_value = ['XAUUSD']
        commodities_specialist.forward.return_value = (
            torch.tensor([[0.3]]),  # actions
            torch.tensor([[0.7]])  # value
        )
        
        return {
            'forex': forex_specialist,
            'commodities': commodities_specialist
        }
    
    @pytest.fixture
    def joint_config(self):
        """Create joint training config."""
        return JointTrainingConfig(
            initial_capital=100000.0,
            episode_length=100,
            warmup_steps=10
        )
    
    def test_joint_training_env_initialization(self, joint_config, mock_market_data, 
                                             mock_meta_controller, mock_specialists):
        """Test JointTrainingEnv initialization."""
        env = JointTrainingEnv(
            config=joint_config,
            market_data=mock_market_data,
            meta_controller=mock_meta_controller,
            specialists=mock_specialists
        )
        
        assert env.joint_config == joint_config
        assert env.meta_controller == mock_meta_controller
        assert env.specialists == mock_specialists
        assert env.current_difficulty == 'easy'
        assert env.difficulty_progress == 0.0
        assert len(env.meta_actions_history) == 0
        assert len(env.specialist_actions_history) == 2
        assert len(env.coordination_rewards) == 0
        assert len(env.individual_rewards) == 2
        assert len(env.joint_performance_history) == 0
        assert len(env.coordination_metrics) == 4
        assert env.executor is not None
    
    def test_joint_training_env_with_communication_hub(self, joint_config, mock_market_data,
                                                      mock_meta_controller, mock_specialists):
        """Test JointTrainingEnv with communication hub."""
        communication_hub = Mock(spec=CommunicationHub)
        portfolio_risk_manager = Mock(spec=PortfolioRiskManager)
        
        env = JointTrainingEnv(
            config=joint_config,
            market_data=mock_market_data,
            meta_controller=mock_meta_controller,
            specialists=mock_specialists,
            communication_hub=communication_hub,
            portfolio_risk_manager=portfolio_risk_manager
        )
        
        assert env.communication_hub == communication_hub
        assert env.portfolio_risk_manager == portfolio_risk_manager


class TestJointTrainingEnvSpaces:
    """Test action and observation spaces."""
    
    @pytest.fixture
    def env_setup(self):
        """Setup environment for testing."""
        config = JointTrainingConfig(episode_length=100)
        market_data = {
            'EURUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=300, freq='H'),  # Increased to 300
                'open': np.random.uniform(1.0, 1.2, 300),
                'high': np.random.uniform(1.1, 1.3, 300),
                'low': np.random.uniform(0.9, 1.1, 300),
                'close': np.random.uniform(1.0, 1.2, 300),
                'volume': np.random.uniform(1000, 10000, 300)
            }),
            'XAUUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=300, freq='H'),  # Increased to 300
                'open': np.random.uniform(2000, 2100, 300),
                'high': np.random.uniform(2050, 2150, 300),
                'low': np.random.uniform(1950, 2050, 300),
                'close': np.random.uniform(2000, 2100, 300),
                'volume': np.random.uniform(100, 1000, 300)
            })
        }
        
        meta_controller = Mock(spec=MetaController)
        specialists = {
            'forex': Mock(spec=BaseSpecialist),
            'commodities': Mock(spec=BaseSpecialist)
        }
        specialists['forex'].get_instruments.return_value = ['EURUSD', 'GBPUSD']
        specialists['commodities'].get_instruments.return_value = ['XAUUSD']
        
        env = JointTrainingEnv(config, market_data, meta_controller, specialists)
        return env
    
    def test_action_space_dimensions(self, env_setup):
        """Test action space dimensions."""
        env = env_setup
        
        # Meta-controller: 4 actions (3 allocations + 1 risk appetite)
        # Forex specialist: 2 instruments
        # Commodities specialist: 1 instrument
        # Total: 4 + 2 + 1 = 7
        expected_action_dim = 4 + 2 + 1
        assert env.action_space.shape[0] == expected_action_dim
    
    def test_observation_space_dimensions(self, env_setup):
        """Test observation space dimensions."""
        env = env_setup
        
        # Should have observation space defined
        assert env.observation_space is not None
        assert hasattr(env.observation_space, 'shape')
        assert len(env.observation_space.shape) > 0


class TestJointTrainingEnvMethods:
    """Test JointTrainingEnv methods."""
    
    @pytest.fixture
    def env_setup(self):
        """Setup environment for testing."""
        config = JointTrainingConfig(episode_length=100)
        market_data = {
            'EURUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=300, freq='H'),  # Increased to 300
                'open': np.random.uniform(1.0, 1.2, 300),
                'high': np.random.uniform(1.1, 1.3, 300),
                'low': np.random.uniform(0.9, 1.1, 300),
                'close': np.random.uniform(1.0, 1.2, 300),
                'volume': np.random.uniform(1000, 10000, 300)
            })
        }
        
        meta_controller = Mock(spec=MetaController)
        specialists = {
            'forex': Mock(spec=BaseSpecialist)
        }
        specialists['forex'].get_instruments.return_value = ['EURUSD']
        
        env = JointTrainingEnv(config, market_data, meta_controller, specialists)
        return env
    
    def test_get_joint_observation(self, env_setup):
        """Test getting joint observation."""
        env = env_setup
        
        # Test the actual _get_observation method
        observation = env._get_observation()
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (env.observation_space.shape[0],)
    
    def test_get_meta_controller_state(self, env_setup):
        """Test getting meta controller state."""
        env = env_setup
        
        # Mock portfolio state
        env.portfolio_value = 100000.0
        env.current_positions = {}
        env.returns_history = [0.01, -0.02, 0.03]
        
        # Test the actual _get_portfolio_state_vector method
        state = env._get_portfolio_state_vector()
        assert isinstance(state, np.ndarray)
        assert state.shape == (74,)
    
    def test_get_specialist_states(self, env_setup):
        """Test getting specialist states."""
        env = env_setup

        # Mock market data access
        env.current_step = 10
        env.market_data = {
            'EURUSD': pd.DataFrame({
                'close': np.random.uniform(1.0, 1.2, 100),
                'volume': np.random.uniform(1000, 10000, 100)
            })
        }

        # Test the actual _get_specialist_states method
        states = env._get_specialist_states()
        assert isinstance(states, np.ndarray)
        assert states.shape == (len(env.specialists) * 20,)
    
    def test_calculate_coordination_reward(self, env_setup):
        """Test coordination reward calculation."""
        env = env_setup
        
        # Mock specialist performance with float values instead of dicts
        specialist_rewards = {
            'forex': 0.1,
            'commodities': 0.05
        }
        
        # Test the actual _calculate_coordination_reward method
        reward = env._calculate_coordination_reward(specialist_rewards)
        assert isinstance(reward, float)
    
    def test_calculate_individual_rewards(self, env_setup):
        """Test individual reward calculation."""
        env = env_setup
        
        # Mock executed orders
        executed_orders = [
            Order(
                order_id="test_order_1",
                agent_id="forex",
                symbol="EURUSD",
                side="buy",
                order_type="market",
                quantity=0.1,
                signal=0.8
            )
        ]
        
        # Test the actual _calculate_reward method
        reward = env._calculate_reward(executed_orders)
        assert isinstance(reward, float)
    
    def test_update_curriculum_difficulty(self, env_setup):
        """Test curriculum difficulty update."""
        env = env_setup
        
        # Mock performance metrics
        env.joint_performance_history = [0.1, 0.2, 0.15, 0.25, 0.3]
        env.coordination_metrics['allocation_efficiency'] = [0.8, 0.85, 0.9]
        
        initial_difficulty = env.current_difficulty
        initial_progress = env.difficulty_progress
        
        # Test the actual _update_curriculum_learning method
        env._update_curriculum_learning(0.2)
        
        # Check that progress was updated
        assert env.difficulty_progress >= initial_progress
    
    def test_get_coordination_metrics(self, env_setup):
        """Test coordination metrics calculation."""
        env = env_setup
        
        # Test the actual get_joint_training_stats method
        stats = env.get_joint_training_stats()
        assert isinstance(stats, dict)
        assert 'joint_performance' in stats
        assert 'coordination_rewards' in stats
        assert 'individual_rewards' in stats


class TestJointTrainingEnvStep:
    """Test step method and episode management."""
    
    @pytest.fixture
    def env_setup(self):
        """Setup environment for testing."""
        config = JointTrainingConfig(episode_length=10, warmup_steps=2)
        market_data = {
            'EURUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=300, freq='H'),  # Increased to 300
                'open': np.random.uniform(1.0, 1.2, 300),
                'high': np.random.uniform(1.1, 1.3, 300),
                'low': np.random.uniform(0.9, 1.1, 300),
                'close': np.random.uniform(1.0, 1.2, 300),
                'volume': np.random.uniform(1000, 10000, 300)
            })
        }
        
        meta_controller = Mock(spec=MetaController)
        meta_controller.forward.return_value = (
            torch.tensor([[0.3, 0.3, 0.4]]),
            torch.tensor([[0.7]]),
            torch.tensor([[0.5]])
        )
        
        specialists = {
            'forex': Mock(spec=BaseSpecialist)
        }
        specialists['forex'].get_instruments.return_value = ['EURUSD']
        specialists['forex'].forward.return_value = (
            torch.tensor([[0.1]]),
            torch.tensor([[0.6]])
        )
        specialists['forex'].calculate_confidence.return_value = 0.8  # Return float instead of Mock
        
        env = JointTrainingEnv(config, market_data, meta_controller, specialists)
        env.portfolio_value = env.config.initial_capital  # Initialize portfolio value
        return env
    
    def test_step_basic(self, env_setup):
        """Test basic step execution."""
        env = env_setup
        
        # Create valid action with correct dimensions
        action = np.random.randn(env.action_space.shape[0])
        
        # Mock necessary methods to avoid errors
        with patch.object(env, '_execute_action', return_value=[]):
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
        action = np.random.randn(env.action_space.shape[0])
        
        # Mock necessary methods
        with patch.object(env, '_execute_action', return_value=[]):
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
        env.meta_actions_history = [np.array([0.1, 0.2, 0.3, 0.4])]
        
        observation, info = env.reset()
        
        assert env.current_step == 0
        assert env.portfolio_value == env.config.initial_capital
        assert isinstance(observation, np.ndarray)
        assert isinstance(info, dict)


class TestJointTrainingEnvEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def env_setup(self):
        """Setup environment for testing."""
        config = JointTrainingConfig(episode_length=10)
        market_data = {
            'EURUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=300, freq='H'),  # Increased to 300
                'open': np.random.uniform(1.0, 1.2, 300),
                'high': np.random.uniform(1.1, 1.3, 300),
                'low': np.random.uniform(0.9, 1.1, 300),
                'close': np.random.uniform(1.0, 1.2, 300),
                'volume': np.random.uniform(1000, 10000, 300)
            })
        }
        
        meta_controller = Mock(spec=MetaController)
        specialists = {
            'forex': Mock(spec=BaseSpecialist)
        }
        specialists['forex'].get_instruments.return_value = ['EURUSD']
        
        env = JointTrainingEnv(config, market_data, meta_controller, specialists)
        return env
    
    def test_invalid_action_dimensions(self, env_setup):
        """Test handling of invalid action dimensions."""
        env = env_setup

        # Action with wrong dimensions
        invalid_action = np.array([0.1, 0.2])  # Too few dimensions

        with pytest.raises((ValueError, IndexError)):
            env.step(invalid_action)
    
    def test_empty_specialists(self, env_setup):
        """Test handling of empty specialists dictionary."""
        config = JointTrainingConfig()
        market_data = {
            'EURUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1100, freq='H'),  # Increased to 1100
                'open': np.random.uniform(1.0, 1.2, 1100),
                'high': np.random.uniform(1.1, 1.3, 1100),
                'low': np.random.uniform(0.9, 1.1, 1100),
                'close': np.random.uniform(1.0, 1.2, 1100),
                'volume': np.random.uniform(1000, 10000, 1100)
            })
        }
        
        meta_controller = Mock(spec=MetaController)
        empty_specialists = {}
        
        # Should handle empty specialists gracefully
        env = JointTrainingEnv(config, market_data, meta_controller, empty_specialists)
        
        assert len(env.specialists) == 0
        assert len(env.specialist_actions_history) == 0
    
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
        
        # Mock negative performance
        specialist_performance = {
            'forex': {'pnl': -100.0, 'sharpe_ratio': -0.5}
        }
        
        reward = env._calculate_coordination_reward(specialist_performance)
        
        # Should handle negative performance gracefully
        assert isinstance(reward, float)
        # Reward might be negative, but should be finite
        assert np.isfinite(reward)
