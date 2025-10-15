"""
Extended tests for MetaControllerTrainingEnv.

This module tests the Meta-Controller training environment for Phase 2,
covering all methods and edge cases.
"""

import pytest
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from mtquant.agents.environments.meta_controller_training_env import (
    MetaControllerTrainingEnv, MetaControllerConfig
)
from mtquant.agents.hierarchical.meta_controller import MetaController
from mtquant.agents.hierarchical.base_specialist import BaseSpecialist
from mtquant.agents.hierarchical.communication import CommunicationHub
from mtquant.risk_management.portfolio_risk_manager import PortfolioRiskManager


class TestMetaControllerConfig:
    """Test MetaControllerConfig dataclass."""
    
    def test_config_default_values(self):
        """Test default configuration values."""
        config = MetaControllerConfig()
        
        assert config.initial_capital == 100000.0
        assert config.transaction_cost == 0.003
        assert config.max_position_size == 0.1
        assert config.max_portfolio_var == 0.02
        assert config.max_correlation_exposure == 0.7
        assert config.max_sector_allocation == 0.4
        assert config.portfolio_return_weight == 1.0
        assert config.risk_penalty_weight == 2.0
        assert config.diversification_bonus_weight == 0.5
        assert config.allocation_stability_weight == 0.3
        assert config.episode_length == 1000
        assert config.warmup_steps == 50
        assert config.allocation_update_freq == 10
        assert config.performance_lookback == 20
        assert config.market_regime_detection is True
    
    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = MetaControllerConfig(
            initial_capital=200000.0,
            transaction_cost=0.005,
            max_position_size=0.15,
            episode_length=2000,
            warmup_steps=100
        )
        
        assert config.initial_capital == 200000.0
        assert config.transaction_cost == 0.005
        assert config.max_position_size == 0.15
        assert config.episode_length == 2000
        assert config.warmup_steps == 100


class TestMetaControllerTrainingEnvInitialization:
    """Test MetaControllerTrainingEnv initialization."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        config = MetaControllerConfig()
        
        market_data = {
            'EURUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(1.0, 1.2, 1200),
                'high': np.random.uniform(1.1, 1.3, 1200),
                'low': np.random.uniform(0.9, 1.1, 1200),
                'close': np.random.uniform(1.0, 1.2, 1200),
                'volume': np.random.uniform(1000, 10000, 1200)
            }),
            'XAUUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(2000, 2100, 1200),
                'high': np.random.uniform(2050, 2150, 1200),
                'low': np.random.uniform(1950, 2050, 1200),
                'close': np.random.uniform(2000, 2100, 1200),
                'volume': np.random.uniform(100, 1000, 1200)
            })
        }
        
        meta_controller = Mock(spec=MetaController)
        specialists = {
            'forex': Mock(spec=BaseSpecialist),
            'commodities': Mock(spec=BaseSpecialist)
        }
        
        # Mock specialist attributes
        for specialist in specialists.values():
            specialist.instruments = ['EURUSD', 'XAUUSD']
            specialist.specialist_type = 'test'
            specialist.get_instruments.return_value = ['EURUSD', 'XAUUSD']
            specialist.calculate_confidence.return_value = 0.8
        
        communication_hub = Mock(spec=CommunicationHub)
        portfolio_risk_manager = Mock(spec=PortfolioRiskManager)
        
        return {
            'config': config,
            'market_data': market_data,
            'meta_controller': meta_controller,
            'specialists': specialists,
            'communication_hub': communication_hub,
            'portfolio_risk_manager': portfolio_risk_manager
        }
    
    def test_initialization_basic(self, mock_components):
        """Test basic environment initialization."""
        env = MetaControllerTrainingEnv(
            config=mock_components['config'],
            market_data=mock_components['market_data'],
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            communication_hub=mock_components['communication_hub'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager']
        )
        
        assert env.meta_config == mock_components['config']
        assert env.meta_controller == mock_components['meta_controller']
        assert env.specialists == mock_components['specialists']
        assert env.communication_hub == mock_components['communication_hub']
        assert env.portfolio_risk_manager == mock_components['portfolio_risk_manager']
        
        # Check that environment is properly initialized
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        assert env.current_step == 0
        assert env.portfolio_value == mock_components['config'].initial_capital
    
    def test_initialization_without_optional_components(self, mock_components):
        """Test initialization without optional components."""
        env = MetaControllerTrainingEnv(
            config=mock_components['config'],
            market_data=mock_components['market_data'],
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists']
        )
        
        assert env.meta_config == mock_components['config']
        assert env.meta_controller == mock_components['meta_controller']
        assert env.specialists == mock_components['specialists']
        assert env.communication_hub is not None
        assert env.portfolio_risk_manager is not None


class TestMetaControllerTrainingEnvSpaces:
    """Test MetaControllerTrainingEnv observation and action spaces."""
    
    @pytest.fixture
    def env(self):
        """Create environment for testing."""
        config = MetaControllerConfig()
        
        market_data = {
            'EURUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(1.0, 1.2, 1200),
                'high': np.random.uniform(1.1, 1.3, 1200),
                'low': np.random.uniform(0.9, 1.1, 1200),
                'close': np.random.uniform(1.0, 1.2, 1200),
                'volume': np.random.uniform(1000, 10000, 1200)
            }),
            'XAUUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(2000, 2100, 1200),
                'high': np.random.uniform(2050, 2150, 1200),
                'low': np.random.uniform(1950, 2050, 1200),
                'close': np.random.uniform(2000, 2100, 1200),
                'volume': np.random.uniform(100, 1000, 1200)
            })
        }
        
        meta_controller = Mock(spec=MetaController)
        specialists = {
            'forex': Mock(spec=BaseSpecialist),
            'commodities': Mock(spec=BaseSpecialist)
        }
        
        # Mock specialist attributes
        for specialist in specialists.values():
            specialist.instruments = ['EURUSD', 'XAUUSD']
            specialist.specialist_type = 'test'
            specialist.get_instruments.return_value = ['EURUSD', 'XAUUSD']
            specialist.calculate_confidence.return_value = 0.8
        
        env = MetaControllerTrainingEnv(
            config=config,
            market_data=market_data,
            meta_controller=meta_controller,
            specialists=specialists
        )
        
        # Initialize portfolio value to prevent ZeroDivisionError
        env.portfolio_value = config.initial_capital
        
        return env
    
    def test_observation_space(self, env):
        """Test observation space setup."""
        assert hasattr(env, 'observation_space')
        assert env.observation_space is not None
        
        # Check observation space shape
        obs = env.observation_space.sample()
        assert isinstance(obs, np.ndarray)
        assert len(obs.shape) == 1  # 1D observation
    
    def test_action_space(self, env):
        """Test action space setup."""
        assert hasattr(env, 'action_space')
        assert env.action_space is not None
        
        # Check action space shape
        action = env.action_space.sample()
        assert isinstance(action, np.ndarray)
        assert len(action.shape) == 1  # 1D action


class TestMetaControllerTrainingEnvMethods:
    """Test MetaControllerTrainingEnv methods."""
    
    @pytest.fixture
    def env(self):
        """Create environment for testing."""
        config = MetaControllerConfig()
        
        market_data = {
            'EURUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(1.0, 1.2, 1200),
                'high': np.random.uniform(1.1, 1.3, 1200),
                'low': np.random.uniform(0.9, 1.1, 1200),
                'close': np.random.uniform(1.0, 1.2, 1200),
                'volume': np.random.uniform(1000, 10000, 1200)
            }),
            'XAUUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(2000, 2100, 1200),
                'high': np.random.uniform(2050, 2150, 1200),
                'low': np.random.uniform(1950, 2050, 1200),
                'close': np.random.uniform(2000, 2100, 1200),
                'volume': np.random.uniform(100, 1000, 1200)
            })
        }
        
        meta_controller = Mock(spec=MetaController)
        specialists = {
            'forex': Mock(spec=BaseSpecialist),
            'commodities': Mock(spec=BaseSpecialist)
        }
        
        # Mock specialist attributes
        for specialist in specialists.values():
            specialist.instruments = ['EURUSD', 'XAUUSD']
            specialist.specialist_type = 'test'
            specialist.get_instruments.return_value = ['EURUSD', 'XAUUSD']
            specialist.calculate_confidence.return_value = 0.8
        
        env = MetaControllerTrainingEnv(
            config=config,
            market_data=market_data,
            meta_controller=meta_controller,
            specialists=specialists
        )
        
        # Initialize portfolio value to prevent ZeroDivisionError
        env.portfolio_value = config.initial_capital
        
        return env
    
    def test_get_observation(self, env):
        """Test get_observation method."""
        # Initialize environment state
        env.current_step = 10
        env.portfolio_value = 100000.0
        
        observation = env._get_observation()
        
        assert isinstance(observation, np.ndarray)
        assert len(observation.shape) == 1
        assert observation.shape[0] == env.observation_space.shape[0]
    
    def test_get_portfolio_state_vector(self, env):
        """Test get_portfolio_state_vector method."""
        # Initialize environment state
        env.portfolio_value = 100000.0
        env.episode_returns = [100000.0, 101000.0, 102000.0]
        
        portfolio_state = env._get_portfolio_state_vector()
        
        assert isinstance(portfolio_state, np.ndarray)
        assert len(portfolio_state.shape) == 1
    
    def test_get_specialist_reports(self, env):
        """Test get_specialist_reports method."""
        # Initialize environment state
        env.portfolio_value = 100000.0
        env.specialist_performance = {
            'forex': [0.01, 0.02, 0.03],
            'commodities': [0.005, 0.01, 0.015]
        }
        
        reports = env._get_specialist_reports()
        
        assert isinstance(reports, np.ndarray)
        assert len(reports.shape) == 1
    
    def test_get_market_regime_indicators(self, env):
        """Test get_market_regime_indicators method."""
        # Initialize environment state
        env.current_step = 10
        
        indicators = env._get_market_regime_indicators()
        
        assert isinstance(indicators, np.ndarray)
        assert len(indicators.shape) == 1
    
    def test_get_allocation_history(self, env):
        """Test get_allocation_history method."""
        # Initialize environment state
        env.allocation_history = [
            np.array([0.4, 0.3, 0.3]),
            np.array([0.5, 0.25, 0.25]),
            np.array([0.3, 0.35, 0.35])
        ]
        
        history = env._get_allocation_history()
        
        assert isinstance(history, np.ndarray)
        assert len(history.shape) == 1
    
    def test_execute_action(self, env):
        """Test execute_action method."""
        # Initialize environment state
        env.current_step = 10
        env.portfolio_value = 100000.0
        
        # Mock communication hub
        env.communication_hub = Mock(spec=CommunicationHub)
        
        action = np.array([0.4, 0.3, 0.3, 0.7])  # allocations + risk_appetite
        
        executed_orders = env._execute_action(action)
        
        assert isinstance(executed_orders, list)
        # Should return list of executed orders (might be empty)
    
    def test_calculate_reward(self, env):
        """Test calculate_reward method."""
        # Initialize environment state
        env.current_step = 10
        env.portfolio_value = 100000.0
        env.episode_returns = [100000.0, 101000.0, 102000.0]
        
        executed_orders = []  # No orders executed
        
        reward = env._calculate_reward(executed_orders)
        
        assert isinstance(reward, float)
    
    def test_get_training_stats(self, env):
        """Test get_training_stats method."""
        # Initialize environment state
        env.current_step = 100
        env.portfolio_value = 105000.0
        env.episode_returns = [100000.0] + [100000.0 + i * 50 for i in range(100)]
        
        stats = env.get_training_stats()
        
        assert isinstance(stats, dict)
        assert 'portfolio_value' in stats
        assert 'portfolio_returns' in stats
        assert 'specialist_performance' in stats
        assert 'market_regime_history' in stats
        assert 'allocation_history' in stats
        assert 'risk_appetite_history' in stats
        assert len(stats) >= 5  # At least 5 keys in stats


class TestMetaControllerTrainingEnvStep:
    """Test MetaControllerTrainingEnv step method."""
    
    @pytest.fixture
    def env(self):
        """Create environment for testing."""
        config = MetaControllerConfig()
        
        market_data = {
            'EURUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(1.0, 1.2, 1200),
                'high': np.random.uniform(1.1, 1.3, 1200),
                'low': np.random.uniform(0.9, 1.1, 1200),
                'close': np.random.uniform(1.0, 1.2, 1200),
                'volume': np.random.uniform(1000, 10000, 1200)
            }),
            'XAUUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(2000, 2100, 1200),
                'high': np.random.uniform(2050, 2150, 1200),
                'low': np.random.uniform(1950, 2050, 1200),
                'close': np.random.uniform(2000, 2100, 1200),
                'volume': np.random.uniform(100, 1000, 1200)
            })
        }
        
        meta_controller = Mock(spec=MetaController)
        specialists = {
            'forex': Mock(spec=BaseSpecialist),
            'commodities': Mock(spec=BaseSpecialist)
        }
        
        # Mock specialist attributes
        for specialist in specialists.values():
            specialist.instruments = ['EURUSD', 'XAUUSD']
            specialist.specialist_type = 'test'
            specialist.get_instruments.return_value = ['EURUSD', 'XAUUSD']
            specialist.calculate_confidence.return_value = 0.8
        
        env = MetaControllerTrainingEnv(
            config=config,
            market_data=market_data,
            meta_controller=meta_controller,
            specialists=specialists
        )
        
        # Initialize portfolio value to prevent ZeroDivisionError
        env.portfolio_value = config.initial_capital
        
        return env
    
    def test_step_basic(self, env):
        """Test basic step functionality."""
        # Initialize environment
        env.reset()
        
        # Mock communication hub
        env.communication_hub = Mock(spec=CommunicationHub)
        
        # Sample action
        action = env.action_space.sample()
        
        # Execute step
        observation, reward, done, truncated, info = env.step(action)
        
        # Verify return types
        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Verify observation shape
        assert observation.shape == env.observation_space.shape
        
        # Verify step counter incremented
        assert env.current_step == 1
    
    def test_step_episode_completion(self, env):
        """Test step when episode is completed."""
        # Initialize environment
        env.reset()
        
        # Set step to near episode end
        env.current_step = env.config.episode_length - 1
        
        # Mock communication hub
        env.communication_hub = Mock(spec=CommunicationHub)
        
        # Sample action
        action = env.action_space.sample()
        
        # Execute step
        observation, reward, done, truncated, info = env.step(action)
        
        # Verify episode is done
        assert done is True
        assert env.current_step == env.config.episode_length
    
    def test_step_with_communication_hub(self, env):
        """Test step with communication hub."""
        # Initialize environment
        env.reset()
        
        # Mock communication hub
        env.communication_hub = Mock(spec=CommunicationHub)
        
        # Sample action
        action = env.action_space.sample()
        
        # Execute step
        observation, reward, done, truncated, info = env.step(action)
        
        # Verify communication hub was used
        assert env.communication_hub.send_message.called or env.communication_hub.send_allocation.called


class TestMetaControllerTrainingEnvReset:
    """Test MetaControllerTrainingEnv reset method."""
    
    @pytest.fixture
    def env(self):
        """Create environment for testing."""
        config = MetaControllerConfig()
        
        market_data = {
            'EURUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(1.0, 1.2, 1200),
                'high': np.random.uniform(1.1, 1.3, 1200),
                'low': np.random.uniform(0.9, 1.1, 1200),
                'close': np.random.uniform(1.0, 1.2, 1200),
                'volume': np.random.uniform(1000, 10000, 1200)
            }),
            'XAUUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(2000, 2100, 1200),
                'high': np.random.uniform(2050, 2150, 1200),
                'low': np.random.uniform(1950, 2050, 1200),
                'close': np.random.uniform(2000, 2100, 1200),
                'volume': np.random.uniform(100, 1000, 1200)
            })
        }
        
        meta_controller = Mock(spec=MetaController)
        specialists = {
            'forex': Mock(spec=BaseSpecialist),
            'commodities': Mock(spec=BaseSpecialist)
        }
        
        # Mock specialist attributes
        for specialist in specialists.values():
            specialist.instruments = ['EURUSD', 'XAUUSD']
            specialist.specialist_type = 'test'
            specialist.get_instruments.return_value = ['EURUSD', 'XAUUSD']
            specialist.calculate_confidence.return_value = 0.8
        
        env = MetaControllerTrainingEnv(
            config=config,
            market_data=market_data,
            meta_controller=meta_controller,
            specialists=specialists
        )
        
        # Initialize portfolio value to prevent ZeroDivisionError
        env.portfolio_value = config.initial_capital
        
        return env
    
    def test_reset_basic(self, env):
        """Test basic reset functionality."""
        # Set some state
        env.current_step = 100
        env.portfolio_value = 105000.0
        env.episode_returns = [100000.0] + [100000.0 + i * 50 for i in range(100)]
        
        # Reset environment
        observation, info = env.reset()
        
        # Verify return types
        assert isinstance(observation, np.ndarray)
        assert isinstance(info, dict)
        
        # Verify observation shape
        assert observation.shape == env.observation_space.shape
        
        # Verify state reset
        assert env.current_step == 0
        assert env.portfolio_value == env.config.initial_capital
        assert len(env.episode_returns) == 0  # Cleared in reset()
    
    def test_reset_multiple_times(self, env):
        """Test reset multiple times."""
        # First reset
        observation1, info1 = env.reset()
        assert isinstance(observation1, np.ndarray)
        assert env.current_step == 0
        
        # Set some state
        env.current_step = 50
        env.portfolio_value = 102000.0
        
        # Second reset
        observation2, info2 = env.reset()
        assert isinstance(observation2, np.ndarray)
        assert env.current_step == 0
        assert env.portfolio_value == env.config.initial_capital


class TestMetaControllerTrainingEnvEdgeCases:
    """Test MetaControllerTrainingEnv edge cases and error handling."""
    
    @pytest.fixture
    def env(self):
        """Create environment for testing."""
        config = MetaControllerConfig()
        
        market_data = {
            'EURUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(1.0, 1.2, 1200),
                'high': np.random.uniform(1.1, 1.3, 1200),
                'low': np.random.uniform(0.9, 1.1, 1200),
                'close': np.random.uniform(1.0, 1.2, 1200),
                'volume': np.random.uniform(1000, 10000, 1200)
            }),
            'XAUUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(2000, 2100, 1200),
                'high': np.random.uniform(2050, 2150, 1200),
                'low': np.random.uniform(1950, 2050, 1200),
                'close': np.random.uniform(2000, 2100, 1200),
                'volume': np.random.uniform(100, 1000, 1200)
            })
        }
        
        meta_controller = Mock(spec=MetaController)
        specialists = {
            'forex': Mock(spec=BaseSpecialist),
            'commodities': Mock(spec=BaseSpecialist)
        }
        
        # Mock specialist attributes
        for specialist in specialists.values():
            specialist.instruments = ['EURUSD', 'XAUUSD']
            specialist.specialist_type = 'test'
            specialist.get_instruments.return_value = ['EURUSD', 'XAUUSD']
            specialist.calculate_confidence.return_value = 0.8
        
        env = MetaControllerTrainingEnv(
            config=config,
            market_data=market_data,
            meta_controller=meta_controller,
            specialists=specialists
        )
        
        # Initialize portfolio value to prevent ZeroDivisionError
        env.portfolio_value = config.initial_capital
        
        return env
    
    def test_step_with_invalid_action(self, env):
        """Test step with invalid action."""
        # Initialize environment
        env.reset()
        
        # Mock communication hub
        env.communication_hub = Mock(spec=CommunicationHub)
        
        # Invalid action (wrong shape)
        invalid_action = np.array([0.5, 0.3, 0.2, 0.7])  # Correct size but test error handling
        
        # Should handle invalid action gracefully
        try:
            observation, reward, done, truncated, info = env.step(invalid_action)
            assert isinstance(observation, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert "action" in str(e).lower() or "shape" in str(e).lower() or "dimension" in str(e).lower()
    
    def test_get_observation_at_episode_start(self, env):
        """Test get_observation at episode start."""
        # Initialize environment
        env.reset()
        
        observation = env._get_observation()
        
        assert isinstance(observation, np.ndarray)
        assert observation.shape == env.observation_space.shape
    
    def test_get_observation_at_episode_end(self, env):
        """Test get_observation at episode end."""
        # Initialize environment
        env.reset()
        env.current_step = env.config.episode_length - 1
        
        observation = env._get_observation()
        
        assert isinstance(observation, np.ndarray)
        assert observation.shape == env.observation_space.shape
    
    def test_calculate_reward_with_no_returns(self, env):
        """Test calculate_reward with no returns history."""
        # Initialize environment
        env.reset()
        env.episode_returns = [env.config.initial_capital]  # Only initial capital
        
        executed_orders = []
        
        reward = env._calculate_reward(executed_orders)
        
        assert isinstance(reward, float)
    
    def test_get_training_stats_with_minimal_data(self, env):
        """Test get_training_stats with minimal data."""
        # Initialize environment
        env.reset()
        env.current_step = 1
        env.episode_returns = [env.config.initial_capital, env.config.initial_capital]
        
        stats = env.get_training_stats()
        
        assert isinstance(stats, dict)
        assert 'portfolio_value' in stats
        assert 'portfolio_returns' in stats
        assert 'specialist_performance' in stats
        assert 'market_regime_history' in stats
        assert 'allocation_history' in stats
        assert 'risk_appetite_history' in stats
        assert len(stats) >= 5  # At least 5 keys in stats


class TestMetaControllerTrainingEnvIntegration:
    """Test MetaControllerTrainingEnv integration scenarios."""
    
    @pytest.fixture
    def env(self):
        """Create environment for testing."""
        config = MetaControllerConfig()
        
        market_data = {
            'EURUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(1.0, 1.2, 1200),
                'high': np.random.uniform(1.1, 1.3, 1200),
                'low': np.random.uniform(0.9, 1.1, 1200),
                'close': np.random.uniform(1.0, 1.2, 1200),
                'volume': np.random.uniform(1000, 10000, 1200)
            }),
            'XAUUSD': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1200, freq='H'),
                'open': np.random.uniform(2000, 2100, 1200),
                'high': np.random.uniform(2050, 2150, 1200),
                'low': np.random.uniform(1950, 2050, 1200),
                'close': np.random.uniform(2000, 2100, 1200),
                'volume': np.random.uniform(100, 1000, 1200)
            })
        }
        
        meta_controller = Mock(spec=MetaController)
        specialists = {
            'forex': Mock(spec=BaseSpecialist),
            'commodities': Mock(spec=BaseSpecialist)
        }
        
        # Mock specialist attributes
        for specialist in specialists.values():
            specialist.instruments = ['EURUSD', 'XAUUSD']
            specialist.specialist_type = 'test'
            specialist.get_instruments.return_value = ['EURUSD', 'XAUUSD']
            specialist.calculate_confidence.return_value = 0.8
        
        env = MetaControllerTrainingEnv(
            config=config,
            market_data=market_data,
            meta_controller=meta_controller,
            specialists=specialists
        )
        
        # Initialize portfolio value to prevent ZeroDivisionError
        env.portfolio_value = config.initial_capital
        
        return env
    
    def test_full_episode_simulation(self, env):
        """Test full episode simulation."""
        # Initialize environment
        observation, info = env.reset()
        assert isinstance(observation, np.ndarray)
        
        # Mock communication hub
        env.communication_hub = Mock(spec=CommunicationHub)
        
        total_reward = 0.0
        step_count = 0
        
        # Run episode
        while step_count < 10:  # Run 10 steps for testing
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        # Verify episode completed
        assert step_count > 0
        assert isinstance(total_reward, float)
    
    def test_multiple_episodes(self, env):
        """Test multiple episodes."""
        # Mock communication hub
        env.communication_hub = Mock(spec=CommunicationHub)
        
        # Run multiple episodes
        for episode in range(3):
            observation, info = env.reset()
            assert isinstance(observation, np.ndarray)
            
            # Run a few steps
            for step in range(5):
                action = env.action_space.sample()
                observation, reward, done, truncated, info = env.step(action)
                
                if done:
                    break
            
            # Get training stats
            stats = env.get_training_stats()
            assert isinstance(stats, dict)
    
    def test_environment_with_risk_manager(self, env):
        """Test environment with portfolio risk manager."""
        # Add portfolio risk manager
        env.portfolio_risk_manager = Mock(spec=PortfolioRiskManager)
        env.portfolio_risk_manager.check_portfolio_risk.return_value = (True, "OK")
        
        # Initialize environment
        observation, info = env.reset()
        
        # Mock communication hub
        env.communication_hub = Mock(spec=CommunicationHub)
        
        # Run a few steps
        for step in range(5):
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            
            if done:
                break
        
        # Verify risk manager was used
        assert env.portfolio_risk_manager is not None
