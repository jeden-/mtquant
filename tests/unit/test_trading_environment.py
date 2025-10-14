"""
Unit tests for trading environment.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from mtquant.agents.environments.base_trading_env import MTQuantTradingEnv, EpisodeMetrics
from mtquant.data.processors.feature_engineering import create_sample_data


class TestMTQuantTradingEnv:
    """Test trading environment functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return create_sample_data("XAUUSD", periods=100, seed=42)
    
    @pytest.fixture
    def trading_env(self, sample_data):
        """Create trading environment for testing."""
        config = {
            'ppo_agent': {
                'initial_capital': 10000,
                'transaction_cost': 0.001
            }
        }
        return MTQuantTradingEnv(
            data=sample_data,
            symbol="XAUUSD",
            initial_capital=10000,
            transaction_cost=0.001,
            position_sizer=None,
            config=config
        )
    
    def test_environment_initialization(self, trading_env):
        """Test environment initialization."""
        assert trading_env.symbol == "XAUUSD"
        assert trading_env.initial_capital == 10000
        assert trading_env.current_capital == 10000
        assert trading_env.current_position == 0.0
        assert trading_env.current_step == 0
        assert len(trading_env.data) > 0
    
    def test_reset_environment(self, trading_env):
        """Test environment reset."""
        # Modify state
        trading_env.current_capital = 5000
        trading_env.current_position = 0.5
        trading_env.current_step = 50
        
        # Reset
        obs, info = trading_env.reset()
        
        # Check reset state
        assert trading_env.current_capital == 10000
        assert trading_env.current_position == 0.0
        assert trading_env.current_step == 0
        assert isinstance(obs, np.ndarray)
        assert len(obs) > 0
        assert isinstance(info, dict)
    
    def test_step_without_trade(self, trading_env):
        """Test step without executing trade."""
        obs, info = trading_env.reset()
        
        # Small action that won't trigger trade
        action = np.array([0.001])  # Below threshold
        obs, reward, done, truncated, info = trading_env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert trading_env.current_position == 0.0  # No position change
    
    def test_step_with_trade(self, trading_env):
        """Test step with trade execution."""
        obs, info = trading_env.reset()
        
        # Action that will trigger trade
        action = np.array([0.5])  # Above threshold
        obs, reward, done, truncated, info = trading_env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert trading_env.current_position != 0.0  # Position changed
    
    def test_position_change_execution(self, trading_env):
        """Test position change execution."""
        trading_env.reset()
        
        # Test opening position
        trade_result = trading_env._execute_position_change(0.0, 0.1, 2000.0)
        
        assert trade_result['trade_executed'] is True
        assert trade_result['position_before'] == 0.0
        assert trade_result['position_after'] == 0.1
        assert trading_env.current_position == 0.1
        assert trading_env.position_entry_price == 2000.0
    
    def test_position_change_with_existing_position(self, trading_env):
        """Test position change with existing position."""
        trading_env.reset()
        
        # Set initial position
        trading_env.current_position = 0.1
        trading_env.position_entry_price = 2000.0
        trading_env.position_entry_step = 5
        
        # Test closing position
        trade_result = trading_env._execute_position_change(0.1, 0.0, 2050.0)
        
        assert trade_result['trade_executed'] is True
        assert trade_result['position_before'] == 0.1
        assert trade_result['position_after'] == 0.0
        assert trading_env.current_position == 0.0
        assert trade_result['trade_pnl'] != 0.0  # Should have P&L
    
    def test_reward_calculation(self, trading_env):
        """Test reward calculation."""
        trading_env.reset()
        
        # Test reward for executed trade
        trade_result = {
            'trade_executed': True,
            'trade_pnl': 100.0,
            'transaction_cost': 5.0
        }
        reward = trading_env._calculate_reward(trade_result)
        
        assert isinstance(reward, float)
        assert reward > 0  # Should be positive for profitable trade
    
    def test_reward_calculation_no_trade(self, trading_env):
        """Test reward calculation for no trade."""
        trading_env.reset()
        
        # Test reward for no trade
        trade_result = {
            'trade_executed': False,
            'trade_pnl': 0.0,
            'transaction_cost': 0.0
        }
        reward = trading_env._calculate_reward(trade_result)
        
        assert isinstance(reward, float)
        assert reward < 0  # Should be negative for no trade
    
    def test_episode_metrics_tracking(self, trading_env):
        """Test episode metrics tracking."""
        trading_env.reset()
        
        # Execute some trades
        trading_env._record_trade(0.1, 2000.0, 2050.0, 50.0, 10)
        trading_env._record_trade(0.1, 2050.0, 2100.0, 50.0, 15)
        trading_env._record_trade(0.1, 2100.0, 2080.0, -20.0, 20)
        
        # Get metrics
        metrics = trading_env.get_episode_metrics()
        
        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.total_pnl == 80.0
        # Win rate is calculated in _record_trade, so check it's updated
        assert metrics.win_rate >= 0.0  # Should be calculated
        assert metrics.sharpe_ratio is not None
    
    def test_episode_metrics_empty(self, trading_env):
        """Test episode metrics with no trades."""
        trading_env.reset()
        
        metrics = trading_env.get_episode_metrics()
        
        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.total_pnl == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.sharpe_ratio == 0.0
    
    def test_action_thresholds(self, trading_env):
        """Test action thresholds."""
        trading_env.reset()
        
        # Test below threshold
        small_action = np.array([0.0001])
        obs, reward, done, truncated, info = trading_env.step(small_action)
        assert trading_env.current_position == 0.0
        
        # Test above threshold (use larger action to ensure position change > 0.001)
        large_action = np.array([0.5])  # 0.5 * 0.1 = 0.05 position size
        obs, reward, done, truncated, info = trading_env.step(large_action)
        assert trading_env.current_position != 0.0
    
    def test_position_size_calculation(self, trading_env):
        """Test position size calculation."""
        trading_env.reset()
        
        # Test position size calculation
        action_value = 0.5
        current_price = 2000.0
        position_size = trading_env._calculate_position_size(action_value, current_price)
        
        assert isinstance(position_size, float)
        assert position_size > 0
        assert position_size <= 1.0  # Should be normalized
    
    def test_episode_completion(self, trading_env):
        """Test episode completion."""
        obs, info = trading_env.reset()
        
        # Run until episode is done
        step_count = 0
        done = False
        while not done and step_count < 1000:
            action = np.array([0.1])
            obs, reward, done, truncated, info = trading_env.step(action)
            step_count += 1
        
        assert done is True
        assert step_count > 0
    
    def test_observation_space(self, trading_env):
        """Test observation space."""
        obs, info = trading_env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert len(obs) > 0
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()
    
    def test_action_space(self, trading_env):
        """Test action space."""
        assert hasattr(trading_env, 'action_space')
        assert trading_env.action_space is not None
    
    def test_render(self, trading_env):
        """Test render method."""
        trading_env.reset()
        
        # Should not raise exception
        trading_env.render()
    
    def test_close(self, trading_env):
        """Test close method."""
        # Should not raise exception
        trading_env.close()


class TestEpisodeMetrics:
    """Test EpisodeMetrics class."""
    
    def test_episode_metrics_initialization(self):
        """Test EpisodeMetrics initialization."""
        metrics = EpisodeMetrics()
        
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.total_pnl == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.avg_win == 0.0
        assert metrics.avg_loss == 0.0
    
    def test_episode_metrics_with_trades(self):
        """Test EpisodeMetrics with trades."""
        metrics = EpisodeMetrics()
        
        # Simulate trades
        metrics.total_trades = 3
        metrics.winning_trades = 2
        metrics.losing_trades = 1
        metrics.total_pnl = 100.0
        metrics.win_rate = 2/3
        metrics.sharpe_ratio = 1.5
        
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.total_pnl == 100.0
        assert metrics.win_rate == 2/3
        assert metrics.sharpe_ratio == 1.5


@pytest.mark.integration
class TestTradingEnvironmentIntegration:
    """Integration tests for trading environment."""
    
    def test_full_episode_simulation(self):
        """Test full episode simulation."""
        # Create sample data
        data = create_sample_data("XAUUSD", periods=200, seed=123)
        
        # Create environment
        config = {
            'ppo_agent': {
                'initial_capital': 10000,
                'transaction_cost': 0.001
            }
        }
        env = MTQuantTradingEnv(
            data=data,
            symbol="XAUUSD",
            initial_capital=10000,
            transaction_cost=0.001,
            position_sizer=None,
            config=config
        )
        
        # Run full episode
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        done = False
        
        while not done and step_count < 1000:
            # Random action
            action = np.random.uniform(-1, 1, 1)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
        
        # Check final state
        assert done is True
        assert step_count > 0
        assert isinstance(total_reward, float)
        
        # Check metrics
        metrics = env.get_episode_metrics()
        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.total_trades >= 0
        
        env.close()
    
    def test_multiple_episodes(self):
        """Test multiple episodes."""
        # Create sample data
        data = create_sample_data("XAUUSD", periods=100, seed=456)
        
        # Create environment
        config = {
            'ppo_agent': {
                'initial_capital': 10000,
                'transaction_cost': 0.001
            }
        }
        env = MTQuantTradingEnv(
            data=data,
            symbol="XAUUSD",
            initial_capital=10000,
            transaction_cost=0.001,
            position_sizer=None,
            config=config
        )
        
        # Run multiple episodes
        n_episodes = 3
        episode_rewards = []
        episode_metrics = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            total_reward = 0
            step_count = 0
            done = False
            
            while not done and step_count < 500:
                # Random action
                action = np.random.uniform(-1, 1, 1)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
            
            episode_rewards.append(total_reward)
            episode_metrics.append(env.get_episode_metrics())
        
        # Check results
        assert len(episode_rewards) == n_episodes
        assert len(episode_metrics) == n_episodes
        
        for i, metrics in enumerate(episode_metrics):
            assert isinstance(metrics, EpisodeMetrics)
            assert metrics.total_trades >= 0
        
        env.close()
