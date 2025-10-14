"""
Unit tests for PPO evaluation functionality.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mtquant.agents.training.train_ppo import evaluate_agent, prepare_data, create_env
from mtquant.agents.environments.base_trading_env import MTQuantTradingEnv


class TestPPOEvaluation:
    """Test PPO evaluation functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock PPO model."""
        model = Mock(spec=PPO)
        model.predict.return_value = (np.array([0.5]), None)
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        from mtquant.data.processors.feature_engineering import create_sample_data
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
    
    @pytest.fixture
    def vec_env(self, trading_env):
        """Create vectorized environment."""
        return DummyVecEnv([lambda: trading_env])
    
    def test_evaluate_agent_basic(self, mock_model, trading_env):
        """Test basic agent evaluation."""
        # Use real trading environment
        mock_model.predict.return_value = (np.array([0.1]), None)
        
        # Run evaluation
        results = evaluate_agent(mock_model, trading_env, n_episodes=2)
        
        # Check results structure
        assert 'n_episodes' in results
        assert 'mean_reward' in results
        assert 'std_reward' in results
        assert 'mean_sharpe_ratio' in results
        assert 'mean_win_rate' in results
        assert 'episode_rewards' in results
        assert 'episode_metrics' in results
        
        assert results['n_episodes'] == 2
        assert len(results['episode_rewards']) == 2
        assert len(results['episode_metrics']) == 2
    
    def test_evaluate_agent_with_real_env(self, mock_model, trading_env):
        """Test evaluation with real environment."""
        # Mock the model to return predictable actions
        mock_model.predict.return_value = (np.array([0.1]), None)
        
        # Run evaluation
        results = evaluate_agent(mock_model, trading_env, n_episodes=1)
        
        # Check results
        assert 'n_episodes' in results
        assert results['n_episodes'] == 1
        assert len(results['episode_rewards']) == 1
        assert len(results['episode_metrics']) == 1
        
        # Check that metrics are properly retrieved
        episode_metrics = results['episode_metrics'][0]
        assert 'total_trades' in episode_metrics
        assert 'total_pnl' in episode_metrics
        assert 'win_rate' in episode_metrics
        assert 'sharpe_ratio' in episode_metrics
    
    def test_evaluate_agent_metrics_retrieval(self, mock_model, trading_env):
        """Test that metrics are properly retrieved from environment."""
        # Use real trading environment
        mock_model.predict.return_value = (np.array([0.1]), None)
        
        # Run evaluation
        results = evaluate_agent(mock_model, trading_env, n_episodes=1)
        
        # Check that metrics are properly retrieved and processed
        episode_metrics = results['episode_metrics'][0]
        assert 'total_trades' in episode_metrics
        assert 'winning_trades' in episode_metrics
        assert 'losing_trades' in episode_metrics
        assert 'total_pnl' in episode_metrics
        assert 'win_rate' in episode_metrics
        assert 'sharpe_ratio' in episode_metrics
    
    def test_evaluate_agent_no_metrics(self, mock_model, trading_env):
        """Test evaluation when environment doesn't provide metrics."""
        # Use real trading environment
        mock_model.predict.return_value = (np.array([0.001]), None)  # Small action to avoid trades
        
        # Run evaluation
        results = evaluate_agent(mock_model, trading_env, n_episodes=1)
        
        # Check that metrics are retrieved (should be 0 for no trades)
        episode_metrics = results['episode_metrics'][0]
        assert 'total_trades' in episode_metrics
        assert 'win_rate' in episode_metrics
        assert 'sharpe_ratio' in episode_metrics
        assert 'total_pnl' in episode_metrics
    
    def test_evaluate_agent_multiple_episodes(self, mock_model, trading_env):
        """Test evaluation with multiple episodes."""
        # Use real trading environment
        mock_model.predict.return_value = (np.array([0.1]), None)
        
        # Run evaluation
        results = evaluate_agent(mock_model, trading_env, n_episodes=2)
        
        # Check results
        assert results['n_episodes'] == 2
        assert len(results['episode_rewards']) == 2
        assert len(results['episode_metrics']) == 2
        
        # Check individual episode metrics
        assert 'total_trades' in results['episode_metrics'][0]
        assert 'total_trades' in results['episode_metrics'][1]
        
        # Check summary statistics
        assert 'mean_sharpe_ratio' in results
        assert 'mean_win_rate' in results
    
    def test_evaluate_agent_error_handling(self, mock_model, vec_env):
        """Test error handling in evaluation."""
        # Mock the environment to raise an exception
        with patch.object(vec_env, 'reset', side_effect=Exception("Test error")):
            with pytest.raises(Exception):
                evaluate_agent(mock_model, vec_env, n_episodes=1)
    
    def test_evaluate_agent_action_processing(self, mock_model, trading_env):
        """Test that actions are properly processed."""
        # Use real trading environment
        mock_model.predict.return_value = (np.array([0.1]), None)
        
        # Run evaluation
        results = evaluate_agent(mock_model, trading_env, n_episodes=1)
        
        # Check that model.predict was called
        mock_model.predict.assert_called()
        
        # Check results
        assert 'n_episodes' in results
        assert len(results['episode_rewards']) == 1
    
    def test_evaluate_agent_reward_processing(self, mock_model, trading_env):
        """Test that rewards are properly processed."""
        # Use real trading environment
        mock_model.predict.return_value = (np.array([0.1]), None)
        
        # Run evaluation
        results = evaluate_agent(mock_model, trading_env, n_episodes=1)
        
        # Check that reward was properly processed
        assert len(results['episode_rewards']) == 1
        assert isinstance(results['episode_rewards'][0], float)
        assert 'mean_reward' in results
        assert isinstance(results['mean_reward'], float)
    
    def test_evaluate_agent_gym_vs_vecenv_api(self, mock_model, trading_env):
        """Test handling of Gym vs VecEnv API differences."""
        # Use real trading environment (Gym API)
        mock_model.predict.return_value = (np.array([0.1]), None)
        
        # Run evaluation
        results = evaluate_agent(mock_model, trading_env, n_episodes=1)
        
        # Check results
        assert results['n_episodes'] == 1
        assert len(results['episode_rewards']) == 1
        assert len(results['episode_metrics']) == 1
        
        # Check that metrics are properly retrieved
        episode_metrics = results['episode_metrics'][0]
        assert 'total_trades' in episode_metrics
        assert 'win_rate' in episode_metrics
        assert 'sharpe_ratio' in episode_metrics


@pytest.mark.integration
class TestPPOEvaluationIntegration:
    """Integration tests for PPO evaluation."""
    
    def test_evaluate_agent_with_real_model(self):
        """Test evaluation with a real PPO model."""
        # Create sample data
        from mtquant.data.processors.feature_engineering import create_sample_data
        data = create_sample_data("XAUUSD", periods=100, seed=789)
        
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
        
        # Create a simple PPO model
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Run evaluation
        results = evaluate_agent(model, env, n_episodes=2)
        
        # Check results
        assert 'n_episodes' in results
        assert results['n_episodes'] == 2
        assert len(results['episode_rewards']) == 2
        assert len(results['episode_metrics']) == 2
        
        # Check that metrics are properly retrieved
        for episode_metrics in results['episode_metrics']:
            assert 'total_trades' in episode_metrics
            assert 'total_pnl' in episode_metrics
            assert 'win_rate' in episode_metrics
            assert 'sharpe_ratio' in episode_metrics
        
        env.close()
    
    def test_evaluate_agent_with_vecenv(self):
        """Test evaluation with DummyVecEnv."""
        # Create sample data
        from mtquant.data.processors.feature_engineering import create_sample_data
        data = create_sample_data("XAUUSD", periods=100, seed=999)
        
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
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: env])
        
        # Create a simple PPO model
        model = PPO("MlpPolicy", vec_env, verbose=0)
        
        # Run evaluation
        results = evaluate_agent(model, vec_env, n_episodes=2)
        
        # Check results
        assert 'n_episodes' in results
        assert results['n_episodes'] == 2
        assert len(results['episode_rewards']) == 2
        assert len(results['episode_metrics']) == 2
        
        # Check that metrics are properly retrieved
        for episode_metrics in results['episode_metrics']:
            assert 'total_trades' in episode_metrics
            assert 'total_pnl' in episode_metrics
            assert 'win_rate' in episode_metrics
            assert 'sharpe_ratio' in episode_metrics
        
        vec_env.close()
