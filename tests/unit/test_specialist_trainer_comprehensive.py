"""
Comprehensive tests for SpecialistTrainer to achieve >85% coverage.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from mtquant.agents.training.specialist_trainer import SpecialistTrainer
from mtquant.agents.hierarchical.forex_specialist import ForexSpecialist
from mtquant.agents.hierarchical.commodities_specialist import CommoditiesSpecialist
from mtquant.agents.hierarchical.equity_specialist import EquitySpecialist
from mtquant.agents.environments.specialist_env import SpecialistEnv
from mtquant.agents.environments.parallel_env import ParallelHierarchicalWrapper
from mtquant.agents.training.curriculum_learning import AdvancedCurriculumLearning
from mtquant.data.processors.feature_engineering import FeatureEngineer
from mtquant.risk_management.portfolio_risk_manager import PortfolioRiskManager
from mtquant.agents.hierarchical.communication import CommunicationHub


class TestSpecialistTrainerComprehensive:
    """Comprehensive tests for SpecialistTrainer."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'specialists': {
                'forex': {
                    'type': 'forex',
                    'instruments': ['EURUSD', 'GBPUSD', 'USDJPY'],
                    'learning_rate': 0.0003,
                    'model_type': 'PPO'
                }
            },
            'training': {
                'phase_1_timesteps': 1000,
                'n_envs': 4,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'eval_interval': 100,
                'eval_episodes': 5
            },
            'portfolio_risk': {
                'max_portfolio_var': 0.02,
                'max_correlation_exposure': 0.7
            }
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing."""
        data = {}
        for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
            dates = pd.date_range('2024-01-01', periods=100, freq='H')
            prices = 1.0 + np.cumsum(np.random.randn(100) * 0.001)
            data[symbol] = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': prices * 1.001,
                'low': prices * 0.999,
                'close': prices,
                'volume': np.random.randint(1000, 10000, 100)
            })
        return data
    
    def test_specialist_trainer_initialization(self, sample_config, temp_dir):
        """Test SpecialistTrainer initialization."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        assert trainer.specialist_type == 'forex'
        assert trainer.config == sample_config
        assert trainer.output_path == temp_dir
        assert trainer.training_stats == {}
        assert trainer.logger is not None
    
    def test_create_specialist_forex(self, sample_config, temp_dir):
        """Test creating forex specialist."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        specialist = trainer._create_specialist()
        assert isinstance(specialist, ForexSpecialist)
        assert specialist.get_instruments() == ['EURUSD', 'GBPUSD', 'USDJPY']
    
    def test_create_specialist_commodities(self, sample_config, temp_dir):
        """Test creating commodities specialist."""
        config = sample_config.copy()
        config['specialists']['commodities'] = {
            'type': 'commodities',
            'instruments': ['XAUUSD', 'WTIUSD'],
            'learning_rate': 0.0003
        }
        
        trainer = SpecialistTrainer(
            specialist_type='commodities',
            config=config,
            output_path=temp_dir
        )
        
        specialist = trainer._create_specialist()
        assert isinstance(specialist, CommoditiesSpecialist)
    
    def test_create_specialist_equity(self, sample_config, temp_dir):
        """Test creating equity specialist."""
        config = sample_config.copy()
        config['specialists']['equity'] = {
            'type': 'equity',
            'instruments': ['SPX500', 'NAS100', 'US30'],
            'learning_rate': 0.0003
        }
        
        trainer = SpecialistTrainer(
            specialist_type='equity',
            config=config,
            output_path=temp_dir
        )
        
        specialist = trainer._create_specialist()
        assert isinstance(specialist, EquitySpecialist)
    
    def test_create_feature_engineer(self, sample_config, temp_dir):
        """Test creating feature engineer."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        feature_engineer = trainer._create_feature_engineer()
        assert isinstance(feature_engineer, FeatureEngineer)
    
    def test_create_environment_config(self, sample_config, temp_dir):
        """Test creating environment configuration."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        env_config = trainer._create_environment_config()
        assert env_config.instruments == ['EURUSD', 'GBPUSD', 'USDJPY']
        assert env_config.initial_capital == 100000.0
        assert env_config.transaction_cost == 0.003
    
    @patch('mtquant.agents.training.specialist_trainer.ParallelHierarchicalWrapper')
    def test_create_parallel_wrapper(self, mock_wrapper, sample_config, temp_dir):
        """Test creating parallel wrapper."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        # Mock the specialist creation
        with patch.object(trainer, '_create_specialist') as mock_specialist:
            mock_specialist.return_value = Mock()
            
            wrapper = trainer._create_parallel_wrapper()
            assert wrapper is not None
            mock_wrapper.assert_called_once()
    
    @patch('mtquant.agents.training.specialist_trainer.AdvancedCurriculumLearning')
    def test_create_curriculum_wrapper(self, mock_curriculum, sample_config, temp_dir):
        """Test creating curriculum wrapper."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        # Mock parallel wrapper
        trainer.parallel_wrapper = Mock()
        
        wrapper = trainer._create_curriculum_wrapper()
        assert wrapper is not None
        mock_curriculum.assert_called_once()
    
    @patch('mtquant.agents.training.specialist_trainer.PPO')
    def test_create_ppo_model(self, mock_ppo, sample_config, temp_dir):
        """Test creating PPO model."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        # Mock environment
        mock_env = Mock()
        
        model = trainer._create_ppo_model(mock_env)
        assert model is not None
        mock_ppo.assert_called_once()
    
    def test_create_callbacks(self, sample_config, temp_dir):
        """Test creating training callbacks."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        # Mock evaluation environment
        mock_eval_env = Mock()
        
        callbacks = trainer._create_callbacks(mock_eval_env)
        assert isinstance(callbacks, list)
        assert len(callbacks) >= 1  # At least evaluation callback
    
    def test_create_eval_environment(self, sample_config, temp_dir):
        """Test creating evaluation environment."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        # Mock specialist and market data
        trainer.specialist = Mock()
        trainer.market_data = {'EURUSD': pd.DataFrame()}
        
        eval_env = trainer._create_eval_environment()
        assert eval_env is not None
    
    @patch('mtquant.agents.training.specialist_trainer.pd.read_csv')
    def test_load_market_data_with_files(self, mock_read_csv, sample_config, temp_dir):
        """Test loading market data from files."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        # Create mock data
        mock_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'open': np.random.rand(100),
            'high': np.random.rand(100),
            'low': np.random.rand(100),
            'close': np.random.rand(100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        mock_read_csv.return_value = mock_data
        
        # Create data directory
        data_dir = Path(temp_dir) / 'data' / 'market_data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock files
        for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
            (data_dir / f'{symbol}_1H.csv').touch()
        
        market_data = trainer._load_market_data()
        assert isinstance(market_data, dict)
        assert len(market_data) == 3
    
    def test_load_market_data_no_files(self, sample_config, temp_dir):
        """Test loading market data when no files exist."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        market_data = trainer._load_market_data()
        assert isinstance(market_data, dict)
        assert len(market_data) == 3  # Should create dummy data
    
    def test_load_config(self, sample_config, temp_dir):
        """Test loading configuration."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        loaded_config = trainer._load_config()
        assert loaded_config == sample_config
        assert 'specialists' in loaded_config
    
    @patch('mtquant.agents.training.specialist_trainer.PPO')
    @patch('mtquant.agents.training.specialist_trainer.ParallelHierarchicalWrapper')
    @patch('mtquant.agents.training.specialist_trainer.AdvancedCurriculumLearning')
    def test_train_method(self, mock_curriculum, mock_parallel, mock_ppo, sample_config, temp_dir):
        """Test the main train method."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        # Mock all dependencies
        mock_model = Mock()
        mock_ppo.return_value = mock_model
        
        mock_parallel_wrapper = Mock()
        mock_parallel.return_value = mock_parallel_wrapper
        
        mock_curriculum_wrapper = Mock()
        mock_curriculum.return_value = mock_curriculum_wrapper
        
        # Mock environment creation
        mock_env = Mock()
        mock_parallel_wrapper.create_specialist_envs.return_value = {'forex': mock_env}
        
        # Mock evaluation environment
        with patch.object(trainer, '_create_eval_environment') as mock_eval_env:
            mock_eval_env.return_value = Mock()
            
            # Mock callbacks
            with patch.object(trainer, '_create_callbacks') as mock_callbacks:
                mock_callbacks.return_value = []
                
                # Run training
                trainer.train(total_timesteps=100)
                
                # Verify model.learn was called
                mock_model.learn.assert_called_once()
                mock_model.save.assert_called_once()
    
    def test_evaluate_method(self, sample_config, temp_dir):
        """Test the evaluate method."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        # Mock model loading
        with patch('mtquant.agents.training.specialist_trainer.PPO.load') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Mock environment
            with patch.object(trainer, '_create_eval_environment') as mock_eval_env:
                mock_env = Mock()
                mock_eval_env.return_value = mock_env
                
                # Mock model.predict
                mock_model.predict.return_value = (np.array([0.5]), None)
                
                # Mock environment step
                mock_env.step.return_value = (np.array([0.1]), 0.1, False, False, {})
                mock_env.reset.return_value = (np.array([0.1]), {})
                
                results = trainer.evaluate('dummy_path', n_episodes=2)
                
                assert isinstance(results, dict)
                assert 'total_reward' in results
                assert 'episode_rewards' in results
                assert 'episode_lengths' in results
    
    def test_save_training_stats(self, sample_config, temp_dir):
        """Test saving training statistics."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        # Add some training stats
        trainer.training_stats = {
            'training_time': 120.5,
            'total_timesteps': 1000,
            'final_reward': 0.15
        }
        
        trainer._save_training_stats()
        
        # Check if file was created
        stats_file = Path(temp_dir) / 'training_stats.json'
        assert stats_file.exists()
    
    def test_get_training_summary(self, sample_config, temp_dir):
        """Test getting training summary."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        # Add training stats
        trainer.training_stats = {
            'training_time': 120.5,
            'total_timesteps': 1000,
            'final_reward': 0.15
        }
        
        summary = trainer.get_training_summary()
        
        assert isinstance(summary, dict)
        assert 'specialist_type' in summary
        assert 'training_time' in summary
        assert 'total_timesteps' in summary
        assert summary['specialist_type'] == 'forex'
    
    def test_error_handling_invalid_specialist_type(self, sample_config, temp_dir):
        """Test error handling for invalid specialist type."""
        trainer = SpecialistTrainer(
            specialist_type='invalid',
            config=sample_config,
            output_path=temp_dir
        )
        
        with pytest.raises(ValueError):
            trainer._create_specialist()
    
    def test_error_handling_missing_config(self, temp_dir):
        """Test error handling for missing configuration."""
        config = {'training': {}}
        
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=config,
            output_path=temp_dir
        )
        
        with pytest.raises(KeyError):
            trainer._load_config()
    
    @patch('mtquant.agents.training.specialist_trainer.PPO')
    def test_training_interruption_handling(self, mock_ppo, sample_config, temp_dir):
        """Test handling of training interruption."""
        trainer = SpecialistTrainer(
            specialist_type='forex',
            config=sample_config,
            output_path=temp_dir
        )
        
        # Mock model that raises KeyboardInterrupt
        mock_model = Mock()
        mock_model.learn.side_effect = KeyboardInterrupt()
        mock_ppo.return_value = mock_model
        
        # Mock all other dependencies
        with patch.object(trainer, '_load_market_data') as mock_load_data:
            mock_load_data.return_value = {}
            
            with patch.object(trainer, '_create_feature_engineer') as mock_feature:
                mock_feature.return_value = Mock()
                
                with patch.object(trainer, '_create_parallel_wrapper') as mock_parallel:
                    mock_parallel_wrapper = Mock()
                    mock_parallel_wrapper.create_specialist_envs.return_value = {'forex': Mock()}
                    mock_parallel.return_value = mock_parallel_wrapper
                    
                    with patch.object(trainer, '_create_curriculum_wrapper') as mock_curriculum:
                        mock_curriculum.return_value = Mock()
                        
                        with patch.object(trainer, '_create_eval_environment') as mock_eval_env:
                            mock_eval_env.return_value = Mock()
                            
                            with patch.object(trainer, '_create_callbacks') as mock_callbacks:
                                mock_callbacks.return_value = []
                                
                                # Run training - should handle interruption gracefully
                                trainer.train(total_timesteps=100)
                                
                                # Verify interrupted model was saved
                                mock_model.save.assert_called_with(f"{temp_dir}/forex_interrupted")
