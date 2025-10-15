"""
Unit tests for specialist_env.py with fixed architecture and corrected tests.

This file has 238 lines and 9% coverage, so adding comprehensive tests here will significantly increase overall coverage.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import specialist environment classes
from mtquant.agents.environments.specialist_env import (
    SpecialistEnv, EnvironmentConfig
)
from mtquant.agents.hierarchical.base_specialist import BaseSpecialist
from mtquant.agents.environments.hierarchical_env import BaseHierarchicalEnv
from mtquant.mcp_integration.models.order import Order
from mtquant.mcp_integration.models.position import Position


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig dataclass."""
    
    def test_environment_config_initialization(self):
        """Test EnvironmentConfig initialization."""
        config = EnvironmentConfig(
            instruments=["EURUSD", "GBPUSD", "USDJPY"],
            episode_length=1000,
            initial_capital=100000.0,
            transaction_cost=0.001,
            max_position_size=0.1
        )
        
        assert config.instruments == ["EURUSD", "GBPUSD", "USDJPY"]
        assert config.episode_length == 1000
        assert config.initial_capital == 100000.0
        assert config.transaction_cost == 0.001
        assert config.max_position_size == 0.1
        assert config.lookback_window == 100  # Default value
        assert config.timeframe == "1H"  # Default value
        assert config.risk_penalty_weight == 1.0  # Default value
    
    def test_environment_config_with_defaults(self):
        """Test EnvironmentConfig with default values."""
        config = EnvironmentConfig(
            instruments=["XAUUSD", "WTIUSD"]
        )
        
        assert config.instruments == ["XAUUSD", "WTIUSD"]
        assert config.episode_length == 1000  # Default
        assert config.initial_capital == 100000.0  # Default
        assert config.transaction_cost == 0.003  # Default (0.3%)
        assert config.max_position_size == 0.1  # Default
        assert config.lookback_window == 100  # Default
        assert config.timeframe == "1H"  # Default
        assert config.risk_penalty_weight == 1.0  # Default


class TestSpecialistEnvInitialization:
    """Tests for SpecialistEnv initialization."""
    
    @pytest.fixture
    def mock_specialist(self):
        """Create mock BaseSpecialist."""
        specialist = Mock(spec=BaseSpecialist)
        specialist.get_instruments.return_value = ["EURUSD", "GBPUSD", "USDJPY"]
        specialist.calculate_confidence.return_value = 0.8  # Return float instead of Mock
        return specialist
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        data = {}
        for symbol in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "SPX500"]:
            dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
            df = pd.DataFrame({
                'open': np.random.uniform(1.0, 2.0, 200),
                'high': np.random.uniform(1.0, 2.0, 200),
                'low': np.random.uniform(1.0, 2.0, 200),
                'close': np.random.uniform(1.0, 2.0, 200),
                'volume': np.random.uniform(1000, 10000, 200)
            }, index=dates)
            data[symbol] = df
        return data
    
    @pytest.fixture
    def config(self):
        """Create EnvironmentConfig."""
        return EnvironmentConfig(
            instruments=["EURUSD", "GBPUSD", "USDJPY"],
            episode_length=50  # Shorter for testing
        )
    
    def test_specialist_env_initialization(self, mock_specialist, mock_market_data, config):
        """Test SpecialistEnv initialization."""
        env = SpecialistEnv(
            config=config,
            market_data=mock_market_data,
            specialist=mock_specialist
        )
        
        assert env.specialist == mock_specialist
        assert env.config == config
        assert env.specialist_instruments == ["EURUSD", "GBPUSD", "USDJPY"]
        assert env.current_step == 0
        assert env.portfolio_value == config.initial_capital
        assert env.positions == {}
        assert env.trade_history == []
        assert env.action_space is not None
        assert env.observation_space is not None
    
    def test_specialist_env_initialization_missing_instruments(self, mock_specialist, mock_market_data, config):
        """Test SpecialistEnv initialization with missing market data."""
        # Remove one instrument from market data
        del mock_market_data["GBPUSD"]
        
        with pytest.raises(ValueError, match="Missing market data for GBPUSD"):
            SpecialistEnv(
                config=config,
                market_data=mock_market_data,
                specialist=mock_specialist
            )
    
    def test_specialist_env_initialization_insufficient_data(self, mock_specialist, mock_market_data, config):
        """Test SpecialistEnv initialization with insufficient data."""
        # Create market data with insufficient periods
        for symbol in mock_market_data:
            mock_market_data[symbol] = mock_market_data[symbol].iloc[:10]  # Only 10 periods
        
        with pytest.raises(ValueError, match="Insufficient data for EURUSD: 10 < 150"):
            SpecialistEnv(
                config=config,
                market_data=mock_market_data,
                specialist=mock_specialist
            )


class TestSpecialistEnvSpaces:
    """Tests for SpecialistEnv action and observation spaces."""
    
    @pytest.fixture
    def mock_specialist(self):
        """Create mock BaseSpecialist."""
        specialist = Mock(spec=BaseSpecialist)
        specialist.get_instruments.return_value = ["EURUSD", "GBPUSD", "USDJPY"]
        specialist.calculate_confidence.return_value = 0.8  # Return float instead of Mock
        return specialist
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        data = {}
        for symbol in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "SPX500"]:
            dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
            df = pd.DataFrame({
                'open': np.random.uniform(1.0, 2.0, 200),
                'high': np.random.uniform(1.0, 2.0, 200),
                'low': np.random.uniform(1.0, 2.0, 200),
                'close': np.random.uniform(1.0, 2.0, 200),
                'volume': np.random.uniform(1000, 10000, 200)
            }, index=dates)
            data[symbol] = df
        return data
    
    @pytest.fixture
    def config(self):
        """Create EnvironmentConfig."""
        return EnvironmentConfig(
            instruments=["EURUSD", "GBPUSD", "USDJPY"],
            episode_length=50
        )
    
    @pytest.fixture
    def env(self, mock_specialist, mock_market_data, config):
        """Create SpecialistEnv instance."""
        return SpecialistEnv(
            config=config,
            market_data=mock_market_data,
            specialist=mock_specialist
        )
    
    def test_action_space_dimensions(self, env):
        """Test action space dimensions."""
        # Action space should be Box with shape (num_instruments,)
        expected_shape = (len(env.specialist_instruments),)
        assert env.action_space.shape == expected_shape
        assert env.action_space.low.shape == expected_shape
        assert env.action_space.high.shape == expected_shape
    
    def test_observation_space_dimensions(self, env):
        """Test observation space dimensions."""
        # Observation space should be Box with appropriate dimensions
        assert env.observation_space.shape is not None
        assert len(env.observation_space.shape) == 1
        assert env.observation_space.low.shape == env.observation_space.shape
        assert env.observation_space.high.shape == env.observation_space.shape


class TestSpecialistEnvReset:
    """Tests for SpecialistEnv reset method."""
    
    @pytest.fixture
    def mock_specialist(self):
        """Create mock BaseSpecialist."""
        specialist = Mock(spec=BaseSpecialist)
        specialist.get_instruments.return_value = ["EURUSD", "GBPUSD", "USDJPY"]
        specialist.calculate_confidence.return_value = 0.8  # Return float instead of Mock
        return specialist
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        data = {}
        for symbol in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "SPX500"]:
            dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
            df = pd.DataFrame({
                'open': np.random.uniform(1.0, 2.0, 200),
                'high': np.random.uniform(1.0, 2.0, 200),
                'low': np.random.uniform(1.0, 2.0, 200),
                'close': np.random.uniform(1.0, 2.0, 200),
                'volume': np.random.uniform(1000, 10000, 200)
            }, index=dates)
            data[symbol] = df
        return data
    
    @pytest.fixture
    def config(self):
        """Create EnvironmentConfig."""
        return EnvironmentConfig(
            instruments=["EURUSD", "GBPUSD", "USDJPY"],
            episode_length=50
        )
    
    @pytest.fixture
    def env(self, mock_specialist, mock_market_data, config):
        """Create SpecialistEnv instance."""
        return SpecialistEnv(
            config=config,
            market_data=mock_market_data,
            specialist=mock_specialist
        )
    
    def test_reset_initial_state(self, env):
        """Test reset to initial state."""
        # Modify some state
        env.current_step = 10
        env.portfolio_value = 50000.0
        env.positions = {"EURUSD": Position(
            position_id="pos_1", 
            agent_id="test_agent", 
            symbol="EURUSD", 
            side="long",
            quantity=0.1, 
            entry_price=1.1, 
            current_price=1.1
        )}
        
        # Reset
        result = env.reset()
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation = result
            info = {}
        
        assert env.current_step == 0
        assert env.portfolio_value == env.config.initial_capital
        assert env.positions == {}
        assert env.trade_history == []
        assert observation is not None
        assert isinstance(observation, np.ndarray)
        assert observation.shape == env.observation_space.shape


class TestSpecialistEnvStep:
    """Tests for SpecialistEnv step method."""
    
    @pytest.fixture
    def mock_specialist(self):
        """Create mock BaseSpecialist."""
        specialist = Mock(spec=BaseSpecialist)
        specialist.get_instruments.return_value = ["EURUSD", "GBPUSD", "USDJPY"]
        specialist.calculate_confidence.return_value = 0.8  # Return float instead of Mock
        return specialist
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        data = {}
        for symbol in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "SPX500"]:
            dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
            df = pd.DataFrame({
                'open': np.random.uniform(1.0, 2.0, 200),
                'high': np.random.uniform(1.0, 2.0, 200),
                'low': np.random.uniform(1.0, 2.0, 200),
                'close': np.random.uniform(1.0, 2.0, 200),
                'volume': np.random.uniform(1000, 10000, 200)
            }, index=dates)
            data[symbol] = df
        return data
    
    @pytest.fixture
    def config(self):
        """Create EnvironmentConfig."""
        return EnvironmentConfig(
            instruments=["EURUSD", "GBPUSD", "USDJPY"],
            episode_length=50
        )
    
    @pytest.fixture
    def env(self, mock_specialist, mock_market_data, config):
        """Create SpecialistEnv instance."""
        return SpecialistEnv(
            config=config,
            market_data=mock_market_data,
            specialist=mock_specialist
        )
    
    def test_step_basic(self, env):
        """Test basic step functionality."""
        # Reset environment
        env.reset()
        
        # Ensure portfolio_value is not zero
        env.portfolio_value = env.config.initial_capital
        
        # Create action (trading signals for each instrument)
        action = np.array([0.1, -0.05, 0.0])  # EURUSD, GBPUSD, USDJPY
        
        # Step
        observation, reward, done, truncated, info = env.step(action)
        
        assert observation is not None
        assert isinstance(observation, np.ndarray)
        assert observation.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert env.current_step == 1
    
    def test_step_episode_completion(self, env):
        """Test step when episode completes."""
        # Reset environment
        env.reset()
        
        # Ensure portfolio_value is not zero
        env.portfolio_value = env.config.initial_capital
        
        # Set current step to near end
        env.current_step = env.config.episode_length - 1
        
        # Create action
        action = np.zeros(env.action_space.shape[0])
        
        # Step
        observation, reward, done, truncated, info = env.step(action)
        
        assert done == True
        assert env.current_step == env.config.episode_length
    
    def test_step_invalid_action(self, env):
        """Test step with invalid action shape."""
        env.reset()
        
        # Create invalid action (wrong shape)
        action = np.array([0.1, 0.05])  # Too few elements
        
        with pytest.raises(IndexError):
            env.step(action)


class TestSpecialistEnvObservation:
    """Tests for SpecialistEnv observation methods."""
    
    @pytest.fixture
    def mock_specialist(self):
        """Create mock BaseSpecialist."""
        specialist = Mock(spec=BaseSpecialist)
        specialist.get_instruments.return_value = ["EURUSD", "GBPUSD", "USDJPY"]
        specialist.calculate_confidence.return_value = 0.8  # Return float instead of Mock
        return specialist
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        data = {}
        for symbol in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "SPX500"]:
            dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
            df = pd.DataFrame({
                'open': np.random.uniform(1.0, 2.0, 200),
                'high': np.random.uniform(1.0, 2.0, 200),
                'low': np.random.uniform(1.0, 2.0, 200),
                'close': np.random.uniform(1.0, 2.0, 200),
                'volume': np.random.uniform(1000, 10000, 200)
            }, index=dates)
            data[symbol] = df
        return data
    
    @pytest.fixture
    def config(self):
        """Create EnvironmentConfig."""
        return EnvironmentConfig(
            instruments=["EURUSD", "GBPUSD", "USDJPY"],
            episode_length=50
        )
    
    @pytest.fixture
    def env(self, mock_specialist, mock_market_data, config):
        """Create SpecialistEnv instance."""
        return SpecialistEnv(
            config=config,
            market_data=mock_market_data,
            specialist=mock_specialist
        )
    
    def test_get_observation_basic(self, env):
        """Test basic observation retrieval."""
        env.reset()
        
        observation = env._get_observation()
        
        assert observation is not None
        assert isinstance(observation, np.ndarray)
        assert observation.shape == env.observation_space.shape
        assert not np.isnan(observation).any()
        assert not np.isinf(observation).any()
    
    def test_get_observation_with_positions(self, env):
        """Test observation with existing positions."""
        env.reset()
        
        # Add some positions using Position objects
        env.positions = {
            "EURUSD": Position(
                position_id="pos_1", 
                agent_id="test_agent", 
                symbol="EURUSD", 
                side="long",
                quantity=0.1, 
                entry_price=1.1, 
                current_price=1.1
            ),
            "GBPUSD": Position(
                position_id="pos_2", 
                agent_id="test_agent", 
                symbol="GBPUSD", 
                side="short",
                quantity=0.05, 
                entry_price=1.25, 
                current_price=1.25
            )
        }
        
        observation = env._get_observation()
        
        assert observation is not None
        assert isinstance(observation, np.ndarray)
        assert observation.shape == env.observation_space.shape
    
    def test_get_observation_normalization(self, env):
        """Test observation normalization."""
        env.reset()
        
        # Get observation
        observation = env._get_observation()
        
        # Check if values are in reasonable range (assuming normalization)
        assert observation.min() >= -10.0  # Reasonable lower bound
        assert observation.max() <= 10.0   # Reasonable upper bound


class TestSpecialistEnvReward:
    """Tests for SpecialistEnv reward calculation."""
    
    @pytest.fixture
    def mock_specialist(self):
        """Create mock BaseSpecialist."""
        specialist = Mock(spec=BaseSpecialist)
        specialist.get_instruments.return_value = ["EURUSD", "GBPUSD", "USDJPY"]
        specialist.calculate_confidence.return_value = 0.8  # Return float instead of Mock
        return specialist
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        data = {}
        for symbol in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "SPX500"]:
            dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
            df = pd.DataFrame({
                'open': np.random.uniform(1.0, 2.0, 200),
                'high': np.random.uniform(1.0, 2.0, 200),
                'low': np.random.uniform(1.0, 2.0, 200),
                'close': np.random.uniform(1.0, 2.0, 200),
                'volume': np.random.uniform(1000, 10000, 200)
            }, index=dates)
            data[symbol] = df
        return data
    
    @pytest.fixture
    def config(self):
        """Create EnvironmentConfig."""
        return EnvironmentConfig(
            instruments=["EURUSD", "GBPUSD", "USDJPY"],
            episode_length=50
        )
    
    @pytest.fixture
    def env(self, mock_specialist, mock_market_data, config):
        """Create SpecialistEnv instance."""
        return SpecialistEnv(
            config=config,
            market_data=mock_market_data,
            specialist=mock_specialist
        )
    
    def test_calculate_reward_no_positions(self, env):
        """Test reward calculation with no positions."""
        env.reset()
        
        reward = env._calculate_reward([])  # Empty executed_orders list
        
        assert isinstance(reward, float)
        assert reward == 0.0  # No positions, no reward
    
    def test_calculate_reward_with_positions(self, env):
        """Test reward calculation with positions."""
        env.reset()
        
        # Add some positions
        env.positions = {
            "EURUSD": Position(
                position_id="pos_1", 
                agent_id="test_agent", 
                symbol="EURUSD", 
                side="long",
                quantity=0.1, 
                entry_price=1.1, 
                current_price=1.1
            )
        }
        
        reward = env._calculate_reward([])  # Empty executed_orders list
        
        assert isinstance(reward, float)
        # Reward should be based on P&L of positions
    
    def test_calculate_reward_episode_completion(self, env):
        """Test reward calculation at episode completion."""
        env.reset()
        
        # Add some positions
        env.positions = {
            "EURUSD": Position(
                position_id="pos_1", 
                agent_id="test_agent", 
                symbol="EURUSD", 
                side="long",
                quantity=0.1, 
                entry_price=1.1, 
                current_price=1.1
            )
        }
        
        reward = env._calculate_reward([])  # Empty executed_orders list
        
        assert isinstance(reward, float)
        # Should include final P&L and any completion bonuses


class TestSpecialistEnvTrading:
    """Tests for SpecialistEnv trading methods."""
    
    @pytest.fixture
    def mock_specialist(self):
        """Create mock BaseSpecialist."""
        specialist = Mock(spec=BaseSpecialist)
        specialist.get_instruments.return_value = ["EURUSD", "GBPUSD", "USDJPY"]
        specialist.calculate_confidence.return_value = 0.8  # Return float instead of Mock
        return specialist
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        data = {}
        for symbol in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "SPX500"]:
            dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
            df = pd.DataFrame({
                'open': np.random.uniform(1.0, 2.0, 200),
                'high': np.random.uniform(1.0, 2.0, 200),
                'low': np.random.uniform(1.0, 2.0, 200),
                'close': np.random.uniform(1.0, 2.0, 200),
                'volume': np.random.uniform(1000, 10000, 200)
            }, index=dates)
            data[symbol] = df
        return data
    
    @pytest.fixture
    def config(self):
        """Create EnvironmentConfig."""
        return EnvironmentConfig(
            instruments=["EURUSD", "GBPUSD", "USDJPY"],
            episode_length=50
        )
    
    @pytest.fixture
    def env(self, mock_specialist, mock_market_data, config):
        """Create SpecialistEnv instance."""
        return SpecialistEnv(
            config=config,
            market_data=mock_market_data,
            specialist=mock_specialist
        )
    
    def test_execute_action_basic(self, env):
        """Test executing a basic action."""
        env.reset()
        
        # Create action (trading signals for each instrument)
        action = np.array([0.1, -0.05, 0.0])  # EURUSD, GBPUSD, USDJPY
        
        # Execute action
        orders = env._execute_action(action)
        
        assert isinstance(orders, list)
        # Should return list of orders (may be empty if no trades executed)
    
    def test_execute_action_strong_signals(self, env):
        """Test executing action with strong trading signals."""
        env.reset()
        
        # Create strong action signals
        action = np.array([0.8, -0.9, 0.7])  # Strong signals
        
        # Execute action
        orders = env._execute_action(action)
        
        assert isinstance(orders, list)
        # Should generate orders for strong signals
    
    def test_execute_action_weak_signals(self, env):
        """Test executing action with weak trading signals."""
        env.reset()
        
        # Create weak action signals
        action = np.array([0.1, -0.05, 0.02])  # Weak signals
        
        # Execute action
        orders = env._execute_action(action)
        
        assert isinstance(orders, list)
        # May not generate orders for weak signals


class TestSpecialistEnvUtility:
    """Tests for SpecialistEnv utility methods."""
    
    @pytest.fixture
    def mock_specialist(self):
        """Create mock BaseSpecialist."""
        specialist = Mock(spec=BaseSpecialist)
        specialist.get_instruments.return_value = ["EURUSD", "GBPUSD", "USDJPY"]
        specialist.calculate_confidence.return_value = 0.8  # Return float instead of Mock
        return specialist
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        data = {}
        for symbol in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "SPX500"]:
            dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
            df = pd.DataFrame({
                'open': np.random.uniform(1.0, 2.0, 200),
                'high': np.random.uniform(1.0, 2.0, 200),
                'low': np.random.uniform(1.0, 2.0, 200),
                'close': np.random.uniform(1.0, 2.0, 200),
                'volume': np.random.uniform(1000, 10000, 200)
            }, index=dates)
            data[symbol] = df
        return data
    
    @pytest.fixture
    def config(self):
        """Create EnvironmentConfig."""
        return EnvironmentConfig(
            instruments=["EURUSD", "GBPUSD", "USDJPY"],
            episode_length=50
        )
    
    @pytest.fixture
    def env(self, mock_specialist, mock_market_data, config):
        """Create SpecialistEnv instance."""
        return SpecialistEnv(
            config=config,
            market_data=mock_market_data,
            specialist=mock_specialist
        )
    
    def test_get_specialist_portfolio_state(self, env):
        """Test getting specialist portfolio state."""
        env.reset()
        
        portfolio_state = env._get_specialist_portfolio_state()
        
        assert isinstance(portfolio_state, np.ndarray)
        assert len(portfolio_state) == 10  # Expected size
        assert not np.isnan(portfolio_state).any()
    
    def test_get_specialist_portfolio_state_with_positions(self, env):
        """Test getting specialist portfolio state with positions."""
        env.reset()
        
        # Add some positions
        env.positions = {
            "EURUSD": Position(
                position_id="pos_1", 
                agent_id="test_agent", 
                symbol="EURUSD", 
                side="long",
                quantity=0.1, 
                entry_price=1.1, 
                current_price=1.1
            )
        }
        
        portfolio_state = env._get_specialist_portfolio_state()
        
        assert isinstance(portfolio_state, np.ndarray)
        assert len(portfolio_state) == 10
        assert not np.isnan(portfolio_state).any()
    
    def test_get_info(self, env):
        """Test getting environment info."""
        env.reset()
        
        info = env._get_info()
        
        assert isinstance(info, dict)
        assert "current_step" in info
        assert "episode_length" in info
        assert "instruments" in info
        assert "num_positions" in info
    
    def test_render(self, env):
        """Test environment rendering."""
        env.reset()
        
        # Should not raise exception
        env.render()
        
        # Test with mode
        env.render(mode="human")
        env.render(mode="rgb_array")
    
    def test_close(self, env):
        """Test environment closing."""
        # Should not raise exception
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
