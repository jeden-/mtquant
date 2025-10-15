"""
Extended tests for parallel environment wrapper.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from mtquant.agents.environments.parallel_env import (
    ParallelHierarchicalWrapper,
    CurriculumLearningWrapper
)
from mtquant.agents.environments.hierarchical_env import EnvironmentConfig
from mtquant.agents.hierarchical.meta_controller import MetaController
from mtquant.agents.hierarchical.base_specialist import BaseSpecialist
from mtquant.agents.hierarchical.communication import CommunicationHub
from mtquant.risk_management.portfolio_risk_manager import PortfolioRiskManager


class TestParallelHierarchicalWrapper:
    """Test ParallelHierarchicalWrapper class."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        # Mock environment configs
        env_configs = {
            'forex': EnvironmentConfig(
                instruments=['EURUSD', 'GBPUSD'],
                timeframe='1H',
                lookback_window=100,
                initial_capital=100000,
                transaction_cost=0.001,
                max_position_size=0.1,
                max_portfolio_var=0.05,
                max_correlation_exposure=0.3,
                stop_loss_pct=0.02,
                risk_penalty_weight=1.0,
                transaction_cost_weight=1.0,
                diversification_bonus_weight=0.5,
                episode_length=1000,
                warmup_steps=100
            ),
            'commodities': EnvironmentConfig(
                instruments=['XAUUSD', 'WTIUSD'],
                timeframe='1H',
                lookback_window=100,
                initial_capital=100000,
                transaction_cost=0.001,
                max_position_size=0.1,
                max_portfolio_var=0.05,
                max_correlation_exposure=0.3,
                stop_loss_pct=0.02,
                risk_penalty_weight=1.0,
                transaction_cost_weight=1.0,
                diversification_bonus_weight=0.5,
                episode_length=1000,
                warmup_steps=100
            ),
            'meta_controller': EnvironmentConfig(
                instruments=['EURUSD', 'GBPUSD', 'XAUUSD', 'WTIUSD'],
                timeframe='1H',
                lookback_window=100,
                initial_capital=100000,
                transaction_cost=0.001,
                max_position_size=0.1,
                max_portfolio_var=0.05,
                max_correlation_exposure=0.3,
                stop_loss_pct=0.02,
                risk_penalty_weight=1.0,
                transaction_cost_weight=1.0,
                diversification_bonus_weight=0.5,
                episode_length=1000,
                warmup_steps=100
            )
        }
        
        # Mock market data (need 1100+ points for lookback_window + episode_length)
        import pandas as pd
        market_data = {
            'EURUSD': pd.DataFrame({
                'open': np.random.randn(1200),
                'high': np.random.randn(1200),
                'low': np.random.randn(1200),
                'close': np.random.randn(1200),
                'volume': np.random.randn(1200)
            }),
            'GBPUSD': pd.DataFrame({
                'open': np.random.randn(1200),
                'high': np.random.randn(1200),
                'low': np.random.randn(1200),
                'close': np.random.randn(1200),
                'volume': np.random.randn(1200)
            }),
            'XAUUSD': pd.DataFrame({
                'open': np.random.randn(1200),
                'high': np.random.randn(1200),
                'low': np.random.randn(1200),
                'close': np.random.randn(1200),
                'volume': np.random.randn(1200)
            }),
            'WTIUSD': pd.DataFrame({
                'open': np.random.randn(1200),
                'high': np.random.randn(1200),
                'low': np.random.randn(1200),
                'close': np.random.randn(1200),
                'volume': np.random.randn(1200)
            })
        }
        
        # Mock meta controller
        meta_controller = Mock(spec=MetaController)
        meta_controller.forward.return_value = (
            torch.tensor([[0.4, 0.3, 0.3]]),  # allocations
            torch.tensor([[0.7]]),  # risk_appetite
            torch.tensor([[0.5]])   # value
        )
        
        # Mock specialists
        specialists = {
            'forex': Mock(spec=BaseSpecialist),
            'commodities': Mock(spec=BaseSpecialist)
        }
        for specialist in specialists.values():
            specialist.get_instruments.return_value = ['EURUSD', 'GBPUSD']
            specialist.forward.return_value = (
                torch.tensor([[0.1, 0.2, 0.3, 0.4]]),  # actions
                torch.tensor([[0.8]])  # value
            )
        
        # Mock communication hub
        communication_hub = Mock(spec=CommunicationHub)
        
        # Mock portfolio risk manager
        portfolio_risk_manager = Mock(spec=PortfolioRiskManager)
        portfolio_risk_manager.check_portfolio_risk.return_value = (True, "OK")
        
        return {
            'env_configs': env_configs,
            'market_data': market_data,
            'meta_controller': meta_controller,
            'specialists': specialists,
            'communication_hub': communication_hub,
            'portfolio_risk_manager': portfolio_risk_manager
        }
    
    def test_initialization_default(self, mock_components):
        """Test default initialization."""
        wrapper = ParallelHierarchicalWrapper(
            env_configs=mock_components['env_configs'],
            market_data=mock_components['market_data'],
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            communication_hub=mock_components['communication_hub'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager']
        )
        
        assert wrapper.n_envs == 8
        assert wrapper.async_envs is True
        assert len(wrapper.env_configs) == 3  # forex, commodities, meta_controller
        assert wrapper.meta_controller == mock_components['meta_controller']
        assert len(wrapper.specialists) == 2
    
    def test_initialization_custom(self, mock_components):
        """Test custom initialization."""
        wrapper = ParallelHierarchicalWrapper(
            env_configs=mock_components['env_configs'],
            market_data=mock_components['market_data'],
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            communication_hub=mock_components['communication_hub'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager'],
            n_envs=4,
            async_envs=False
        )
        
        assert wrapper.n_envs == 4
        assert wrapper.async_envs is False
    
    def test_create_specialist_envs(self, mock_components):
        """Test creating specialist environments."""
        wrapper = ParallelHierarchicalWrapper(
            env_configs=mock_components['env_configs'],
            market_data=mock_components['market_data'],
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            communication_hub=mock_components['communication_hub'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager'],
            n_envs=2,
            async_envs=False  # Use SyncVectorEnv to avoid pickling issues
        )
        
        envs = wrapper.create_specialist_envs()
        
        assert len(envs) == 2  # 2 specialists
        assert all(env is not None for env in envs.values())
    
    def test_create_meta_controller_envs(self, mock_components):
        """Test creating meta-controller environments."""
        wrapper = ParallelHierarchicalWrapper(
            env_configs=mock_components['env_configs'],
            market_data=mock_components['market_data'],
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            communication_hub=mock_components['communication_hub'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager'],
            n_envs=2,
            async_envs=False  # Use SyncVectorEnv to avoid pickling issues
        )
        
        env = wrapper.create_meta_controller_envs()
        
        assert env is not None
        assert wrapper.meta_controller_envs is not None
    
    def test_reset_specialist_envs(self, mock_components):
        """Test resetting specialist environments."""
        wrapper = ParallelHierarchicalWrapper(
            env_configs=mock_components['env_configs'],
            market_data=mock_components['market_data'],
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            communication_hub=mock_components['communication_hub'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager'],
            n_envs=2
        )
        
        # Mock specialist environments
        mock_env = Mock()
        mock_env.reset.return_value = (np.random.randn(2, 10), {})
        wrapper.specialist_envs = {'forex': mock_env, 'commodities': mock_env}
        
        results = wrapper.reset_specialist_envs()
        
        assert len(results) == 2  # 2 specialists
        assert 'forex' in results
        assert 'commodities' in results
    
    def test_reset_meta_controller_envs(self, mock_components):
        """Test resetting meta-controller environments."""
        wrapper = ParallelHierarchicalWrapper(
            env_configs=mock_components['env_configs'],
            market_data=mock_components['market_data'],
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            communication_hub=mock_components['communication_hub'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager'],
            n_envs=2
        )
        
        # Mock meta-controller environment
        mock_env = Mock()
        mock_env.reset.return_value = (np.random.randn(2, 10), {})
        wrapper.meta_controller_envs = mock_env
        
        observations, infos = wrapper.reset_meta_controller_envs()
        
        assert observations.shape[0] == 2  # n_envs
        assert isinstance(infos, dict)
        mock_env.reset.assert_called_once()
    
    def test_step_specialist_envs(self, mock_components):
        """Test stepping specialist environments."""
        wrapper = ParallelHierarchicalWrapper(
            env_configs=mock_components['env_configs'],
            market_data=mock_components['market_data'],
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            communication_hub=mock_components['communication_hub'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager'],
            n_envs=2
        )
        
        # Mock specialist environments
        mock_env = Mock()
        mock_env.step.return_value = (
            np.random.randn(2, 10),  # observations
            np.random.randn(2),      # rewards
            np.array([False, False]), # dones
            np.array([False, False]), # truncated
            {}  # infos
        )
        wrapper.specialist_envs = {'forex': mock_env, 'commodities': mock_env}
        
        actions = {'forex': np.random.randn(2, 5), 'commodities': np.random.randn(2, 5)}
        results = wrapper.step_specialist_envs(actions)
        
        assert len(results) == 2  # 2 specialists
        assert 'forex' in results
        assert 'commodities' in results
    
    def test_step_meta_controller_envs(self, mock_components):
        """Test stepping meta-controller environments."""
        wrapper = ParallelHierarchicalWrapper(
            env_configs=mock_components['env_configs'],
            market_data=mock_components['market_data'],
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            communication_hub=mock_components['communication_hub'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager'],
            n_envs=2
        )
        
        # Mock meta-controller environment
        mock_env = Mock()
        mock_env.step.return_value = (
            np.random.randn(2, 10),  # observations
            np.random.randn(2),      # rewards
            np.array([False, False]), # dones
            np.array([False, False]), # truncated
            {}  # infos
        )
        wrapper.meta_controller_envs = mock_env
        
        actions = np.random.randn(2, 5)
        observations, rewards, dones, truncated, infos = wrapper.step_meta_controller_envs(actions)
        
        assert observations.shape[0] == 2  # n_envs
        assert rewards.shape[0] == 2
        assert dones.shape[0] == 2
        assert truncated.shape[0] == 2
        mock_env.step.assert_called_once_with(actions)
    
    def test_close_all_envs(self, mock_components):
        """Test closing all environments."""
        wrapper = ParallelHierarchicalWrapper(
            env_configs=mock_components['env_configs'],
            market_data=mock_components['market_data'],
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            communication_hub=mock_components['communication_hub'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager']
        )
        
        # Mock environments
        mock_specialist_env = Mock()
        mock_meta_env = Mock()
        wrapper.specialist_envs = {'forex': mock_specialist_env, 'commodities': mock_specialist_env}
        wrapper.meta_controller_envs = mock_meta_env
        
        wrapper.close_all_envs()
        
        mock_specialist_env.close.assert_called()
        mock_meta_env.close.assert_called_once()
    
    def test_get_training_statistics(self, mock_components):
        """Test getting training statistics."""
        wrapper = ParallelHierarchicalWrapper(
            env_configs=mock_components['env_configs'],
            market_data=mock_components['market_data'],
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            communication_hub=mock_components['communication_hub'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager']
        )
        
        # Mock environments with statistics
        mock_specialist_env = Mock()
        mock_specialist_env.get_episode_statistics.return_value = {'episode_reward': 100.0}
        mock_meta_env = Mock()
        mock_meta_env.get_episode_statistics.return_value = {'episode_reward': 200.0}
        
        wrapper.specialist_envs = {'forex': mock_specialist_env, 'commodities': mock_specialist_env}
        wrapper.meta_controller_envs = mock_meta_env
        
        # Use the correct method name from the actual implementation
        stats = wrapper.get_training_stats()
        
        assert isinstance(stats, dict)
        assert 'specialist_stats' in stats
        assert 'meta_controller_stats' in stats
        assert len(stats['specialist_stats']) == 2


class TestCurriculumLearningWrapper:
    """Test CurriculumLearningWrapper class."""
    
    @pytest.fixture
    def mock_parallel_wrapper(self):
        """Create mock parallel wrapper for testing."""
        wrapper = Mock()
        wrapper.current_phase = 1
        return wrapper
    
    @pytest.fixture
    def curriculum_config(self):
        """Create curriculum configuration."""
        return {
            'phases': {
                1: {'name': 'Individual', 'difficulty': 'easy'},
                2: {'name': 'Meta', 'difficulty': 'medium'},
                3: {'name': 'Joint', 'difficulty': 'hard'}
            },
            'phase_transitions': [0.4, 0.7],
            'reward_shaping': {
                'transaction_cost_weight': 1.0,
                'risk_penalty_weight': 1.0,
                'diversification_bonus_weight': 0.5
            }
        }
    
    def test_initialization_default(self, mock_parallel_wrapper, curriculum_config):
        """Test default initialization."""
        wrapper = CurriculumLearningWrapper(mock_parallel_wrapper, curriculum_config)
        
        assert wrapper.parallel_wrapper == mock_parallel_wrapper
        assert wrapper.curriculum_config == curriculum_config
        assert wrapper.current_phase == 1
        assert wrapper.phase_progress == 0.0
        assert len(wrapper.phases) == 3
    
    def test_initialization_custom(self, mock_parallel_wrapper, curriculum_config):
        """Test custom initialization."""
        wrapper = CurriculumLearningWrapper(mock_parallel_wrapper, curriculum_config)
        
        assert wrapper.parallel_wrapper == mock_parallel_wrapper
        assert wrapper.curriculum_config == curriculum_config
        assert wrapper.current_phase == 1
    
    def test_update_curriculum(self, mock_parallel_wrapper, curriculum_config):
        """Test curriculum update functionality."""
        wrapper = CurriculumLearningWrapper(mock_parallel_wrapper, curriculum_config)
        
        # Test phase 1 (first 40% of training)
        wrapper.update_curriculum(episode=20, total_episodes=100)
        assert wrapper.current_phase == 1
        assert wrapper.phase_progress == 0.5  # 20/40 = 0.5
        
        # Test phase 2 (next 30% of training)
        wrapper.update_curriculum(episode=50, total_episodes=100)
        assert wrapper.current_phase == 2
        assert wrapper.phase_progress == pytest.approx(0.33, abs=0.1)  # (50-40)/30 ≈ 0.33
        
        # Test phase 3 (final 30% of training)
        wrapper.update_curriculum(episode=85, total_episodes=100)
        assert wrapper.current_phase == 3
        assert wrapper.phase_progress == pytest.approx(0.5, abs=0.1)  # (85-70)/30 = 0.5
    
    def test_get_current_scenario(self, mock_parallel_wrapper, curriculum_config):
        """Test getting current scenario."""
        wrapper = CurriculumLearningWrapper(mock_parallel_wrapper, curriculum_config)
        wrapper.current_phase = 1
        wrapper.phase_progress = 0.5
        
        scenario = wrapper.get_current_scenario()
        
        assert isinstance(scenario, str)
        assert scenario in ['low_volatility', 'trending_market']
    
    def test_get_reward_shaping_config(self, mock_parallel_wrapper, curriculum_config):
        """Test getting reward shaping configuration."""
        wrapper = CurriculumLearningWrapper(mock_parallel_wrapper, curriculum_config)
        wrapper.current_phase = 1
        
        config = wrapper.get_reward_shaping_config()
        
        assert isinstance(config, dict)
        assert 'transaction_cost_weight' in config
        assert 'risk_penalty_weight' in config
        assert 'diversification_bonus_weight' in config
    
    def test_get_curriculum_stats(self, mock_parallel_wrapper, curriculum_config):
        """Test getting curriculum statistics."""
        wrapper = CurriculumLearningWrapper(mock_parallel_wrapper, curriculum_config)
        wrapper.current_phase = 2
        wrapper.phase_progress = 0.6
        
        stats = wrapper.get_curriculum_stats()
        
        assert isinstance(stats, dict)
        assert 'current_phase' in stats
        assert 'phase_progress' in stats
        assert 'current_scenario' in stats
        assert stats['current_phase'] == 2
        assert stats['phase_progress'] == 0.6
    
    def test_phase_transition_logic(self, mock_parallel_wrapper, curriculum_config):
        """Test phase transition logic."""
        wrapper = CurriculumLearningWrapper(mock_parallel_wrapper, curriculum_config)
        
        # Test early training (phase 1)
        wrapper.update_curriculum(episode=10, total_episodes=100)
        assert wrapper.current_phase == 1
        
        # Test mid training (phase 2)
        wrapper.update_curriculum(episode=50, total_episodes=100)
        assert wrapper.current_phase == 2
        
        # Test late training (phase 3)
        wrapper.update_curriculum(episode=90, total_episodes=100)
        assert wrapper.current_phase == 3
    
    def test_phase_progress_calculation(self, mock_parallel_wrapper, curriculum_config):
        """Test phase progress calculation."""
        wrapper = CurriculumLearningWrapper(mock_parallel_wrapper, curriculum_config)
        
        # Test phase 1 progress
        wrapper.update_curriculum(episode=20, total_episodes=100)
        assert wrapper.phase_progress == 0.5  # 20/40 = 0.5
        
        # Test phase 2 progress
        wrapper.update_curriculum(episode=55, total_episodes=100)
        expected_progress = (55 - 40) / 30  # (episode - phase_start) / phase_duration
        assert wrapper.phase_progress == pytest.approx(expected_progress, abs=0.01)
        
        # Test phase 3 progress
        wrapper.update_curriculum(episode=85, total_episodes=100)
        expected_progress = (85 - 70) / 30  # (episode - phase_start) / phase_duration
        assert wrapper.phase_progress == pytest.approx(expected_progress, abs=0.01)
    
    def test_scenario_selection(self, mock_parallel_wrapper, curriculum_config):
        """Test scenario selection based on phase progress."""
        wrapper = CurriculumLearningWrapper(mock_parallel_wrapper, curriculum_config)
        
        # Test phase 1 scenarios
        wrapper.current_phase = 1
        wrapper.phase_progress = 0.0
        scenario1 = wrapper.get_current_scenario()
        
        wrapper.phase_progress = 1.0
        scenario2 = wrapper.get_current_scenario()
        
        assert scenario1 in ['low_volatility', 'trending_market']
        assert scenario2 in ['low_volatility', 'trending_market']
        # Should be different scenarios based on progress
        assert scenario1 != scenario2 or len(['low_volatility', 'trending_market']) == 1
    
    def test_reward_shaping_phase_adjustment(self, mock_parallel_wrapper, curriculum_config):
        """Test reward shaping adjustment based on phase."""
        wrapper = CurriculumLearningWrapper(mock_parallel_wrapper, curriculum_config)
        
        # Test phase 1 (individual performance focus)
        wrapper.current_phase = 1
        config1 = wrapper.get_reward_shaping_config()
        
        # Test phase 3 (risk-adjusted performance focus)
        wrapper.current_phase = 3
        config3 = wrapper.get_reward_shaping_config()
        
        assert isinstance(config1, dict)
        assert isinstance(config3, dict)
        # Configurations should be different for different phases
        assert config1 != config3
