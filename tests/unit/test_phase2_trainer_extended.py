"""
Extended unit tests for phase2_trainer.py to increase coverage.
"""

import pytest
import os
import tempfile
import yaml
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

from mtquant.agents.training.phase2_trainer import Phase2Trainer


class TestPhase2Trainer:
    """Test Phase2Trainer class."""
    
    def test_phase2_trainer_initialization(self):
        """Test Phase2Trainer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock config file
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD', 'GBPUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD', 'WTIUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500', 'NAS100'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {
                    'phase_2_timesteps': 10000,
                    'learning_rate': 0.0003,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'eval_interval': 10000,
                    'eval_episodes': 10
                },
                'portfolio_risk': {
                    'max_portfolio_var': 0.05,
                    'max_correlation_exposure': 0.3,
                    'max_sector_allocation': 0.4
                }
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(
                config_path=config_path,
                data_path=temp_dir,
                specialist_models_path=temp_dir,
                output_path=temp_dir,
                n_envs=4
            )
            
            assert trainer.config_path == config_path
            assert trainer.data_path == temp_dir
            assert trainer.specialist_models_path == temp_dir
            assert trainer.output_path == temp_dir
            assert trainer.n_envs == 4
            assert trainer.config == config_data
            assert trainer.meta_controller is None
            assert trainer.specialists == {}
            assert trainer.market_data == {}
            assert trainer.feature_engineer is None
            assert trainer.portfolio_reward_function is None
            assert isinstance(trainer.training_stats, dict)
            assert trainer.training_stats['episodes'] == 0
            assert trainer.training_stats['total_timesteps'] == 0
            assert trainer.training_stats['best_reward'] == -np.inf
            assert trainer.training_stats['training_time'] == 0
    
    def test_load_config(self):
        """Test loading configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_2_timesteps': 5000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(config_path=config_path)
            
            assert trainer.config == config_data
    
    def test_load_config_file_not_found(self):
        """Test loading configuration when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Phase2Trainer(config_path="nonexistent_config.yaml")
    
    def test_load_market_data(self):
        """Test loading market data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Create mock data files
            data_dir = os.path.join(temp_dir, 'market_data')
            os.makedirs(data_dir, exist_ok=True)
            
            for instrument in ['EURUSD', 'XAUUSD', 'SPX500']:
                data_file = os.path.join(data_dir, f'{instrument}_1H.csv')
                dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1H')
                df = pd.DataFrame({
                    'open': np.random.uniform(1.0, 2.0, len(dates)),
                    'high': np.random.uniform(1.0, 2.0, len(dates)),
                    'low': np.random.uniform(1.0, 2.0, len(dates)),
                    'close': np.random.uniform(1.0, 2.0, len(dates)),
                    'volume': np.random.uniform(1000, 10000, len(dates))
                }, index=dates)
                df.to_csv(data_file)
            
            trainer = Phase2Trainer(
                config_path=config_path,
                data_path=data_dir,
                output_path=temp_dir
            )
            
            market_data = trainer._load_market_data()
            
            assert isinstance(market_data, dict)
            assert 'EURUSD' in market_data
            assert 'XAUUSD' in market_data
            assert 'SPX500' in market_data
            
            for instrument, df in market_data.items():
                assert isinstance(df, pd.DataFrame)
                assert len(df) > 0
                assert 'open' in df.columns
                assert 'high' in df.columns
                assert 'low' in df.columns
                assert 'close' in df.columns
                assert 'volume' in df.columns
    
    def test_load_market_data_missing_files(self):
        """Test loading market data when files are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(
                config_path=config_path,
                data_path=temp_dir,  # No data files
                output_path=temp_dir
            )
            
            market_data = trainer._load_market_data()
            
            assert isinstance(market_data, dict)
            assert 'EURUSD' in market_data
            assert 'XAUUSD' in market_data
            assert 'SPX500' in market_data
            
            # Should create dummy data
            for instrument, df in market_data.items():
                assert isinstance(df, pd.DataFrame)
                assert len(df) > 0
    
    def test_create_dummy_data(self):
        """Test creating dummy market data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(config_path=config_path)
            
            # Test EURUSD (USD pair)
            df_eurusd = trainer._create_dummy_data('EURUSD')
            assert isinstance(df_eurusd, pd.DataFrame)
            assert len(df_eurusd) > 0
            assert 'open' in df_eurusd.columns
            assert 'high' in df_eurusd.columns
            assert 'low' in df_eurusd.columns
            assert 'close' in df_eurusd.columns
            assert 'volume' in df_eurusd.columns
            
            # Test SPX500 (non-USD pair)
            df_spx500 = trainer._create_dummy_data('SPX500')
            assert isinstance(df_spx500, pd.DataFrame)
            assert len(df_spx500) > 0
    
    def test_create_meta_controller(self):
        """Test creating meta-controller."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(config_path=config_path)
            
            meta_controller = trainer._create_meta_controller()
            
            assert meta_controller is not None
            assert hasattr(meta_controller, 'state_dim')
            assert hasattr(meta_controller, 'hidden_dim')
            assert hasattr(meta_controller, 'hidden_dim_2')
            assert hasattr(meta_controller, 'dropout')
    
    def test_create_specialists(self):
        """Test creating specialists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD', 'GBPUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD', 'WTIUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500', 'NAS100'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(config_path=config_path)
            
            specialists = trainer._create_specialists()
            
            assert isinstance(specialists, dict)
            assert 'forex' in specialists
            assert 'commodities' in specialists
            assert 'equity' in specialists
            
            # Check specialist types
            assert hasattr(specialists['forex'], 'instruments')
            assert hasattr(specialists['commodities'], 'instruments')
            assert hasattr(specialists['equity'], 'instruments')
    
    def test_create_meta_controller_config(self):
        """Test creating meta-controller configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {
                    'max_portfolio_var': 0.05,
                    'max_correlation_exposure': 0.3,
                    'max_sector_allocation': 0.4
                }
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(config_path=config_path)
            
            meta_config = trainer._create_meta_controller_config()
            
            assert meta_config is not None
            assert meta_config.initial_capital == 100000.0
            assert meta_config.transaction_cost == 0.003
            assert meta_config.max_position_size == 0.1
            assert meta_config.max_portfolio_var == 0.05
            assert meta_config.max_correlation_exposure == 0.3
            assert meta_config.max_sector_allocation == 0.4
            assert meta_config.portfolio_return_weight == 1.0
            assert meta_config.risk_penalty_weight == 2.0
            assert meta_config.diversification_bonus_weight == 0.5
            assert meta_config.allocation_stability_weight == 0.3
            assert meta_config.episode_length == 1000
            assert meta_config.warmup_steps == 50
            assert meta_config.allocation_update_freq == 10
            assert meta_config.performance_lookback == 20
            assert meta_config.market_regime_detection is True
    
    def test_create_reward_config(self):
        """Test creating reward configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(config_path=config_path)
            
            reward_config = trainer._create_reward_config()
            
            assert reward_config is not None
            assert reward_config.portfolio_return_weight == 1.0
            assert reward_config.risk_adjusted_return_weight == 2.0
            assert reward_config.diversification_weight == 0.5
            assert reward_config.allocation_stability_weight == 0.3
            assert reward_config.specialist_coordination_weight == 0.4
            assert reward_config.risk_management_weight == 3.0
            assert reward_config.transaction_cost_weight == 1.0
            assert reward_config.drawdown_penalty_weight == 5.0
            assert reward_config.target_sharpe_ratio == 2.0
            assert reward_config.max_drawdown_threshold == 0.15
            assert reward_config.var_confidence_level == 0.95
            assert reward_config.min_specialist_allocation == 0.1
            assert reward_config.max_specialist_allocation == 0.7
            assert reward_config.target_correlation == 0.3
            assert reward_config.allocation_change_threshold == 0.2
            assert reward_config.max_allocation_volatility == 0.1
            assert reward_config.performance_correlation_threshold == 0.8
            assert reward_config.coordination_bonus_threshold == 0.6
    
    def test_create_training_environment(self):
        """Test creating training environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD', 'GBPUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD', 'WTIUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500', 'NAS100'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {
                    'max_portfolio_var': 0.05,
                    'max_correlation_exposure': 0.3,
                    'max_sector_allocation': 0.4
                }
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(config_path=config_path)
            
            # Mock market data
            trainer.market_data = {
                'EURUSD': pd.DataFrame({
                    'open': [1.0, 1.1, 1.2],
                    'high': [1.05, 1.15, 1.25],
                    'low': [0.95, 1.05, 1.15],
                    'close': [1.1, 1.2, 1.3],
                    'volume': [1000, 1100, 1200]
                }),
                'XAUUSD': pd.DataFrame({
                    'open': [2000.0, 2010.0, 2020.0],
                    'high': [2005.0, 2015.0, 2025.0],
                    'low': [1995.0, 2005.0, 2015.0],
                    'close': [2010.0, 2020.0, 2030.0],
                    'volume': [500, 550, 600]
                }),
                'SPX500': pd.DataFrame({
                    'open': [4000.0, 4010.0, 4020.0],
                    'high': [4005.0, 4015.0, 4025.0],
                    'low': [3995.0, 4005.0, 4015.0],
                    'close': [4010.0, 4020.0, 4030.0],
                    'volume': [2000, 2100, 2200]
                })
            }
            
            with patch('mtquant.agents.training.phase2_trainer.MetaControllerTrainingEnv') as mock_env_class, \
                 patch('mtquant.agents.training.phase2_trainer.CommunicationHub') as mock_hub_class, \
                 patch('mtquant.agents.training.phase2_trainer.PortfolioRiskManager') as mock_risk_class:
                
                mock_env = Mock()
                mock_hub = Mock()
                mock_risk = Mock()
                mock_env_class.return_value = mock_env
                mock_hub_class.return_value = mock_hub
                mock_risk_class.return_value = mock_risk
                
                env = trainer._create_training_environment()
                
                assert env == mock_env
                mock_env_class.assert_called_once()
                
                # Check that meta_controller and specialists were created
                assert trainer.meta_controller is not None
                assert isinstance(trainer.specialists, dict)
                assert 'forex' in trainer.specialists
                assert 'commodities' in trainer.specialists
                assert 'equity' in trainer.specialists
    
    def test_create_ppo_model(self):
        """Test creating PPO model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {
                    'phase_2_timesteps': 1000,
                    'learning_rate': 0.0003,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2
                },
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(config_path=config_path)
            
            # Mock environment
            mock_env = Mock()
            
            with patch('mtquant.agents.training.phase2_trainer.PPO') as mock_ppo_class:
                mock_model = Mock()
                mock_ppo_class.return_value = mock_model
                
                model = trainer._create_ppo_model(mock_env)
                
                assert model == mock_model
                mock_ppo_class.assert_called_once()
    
    def test_create_callbacks(self):
        """Test creating callbacks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {
                    'phase_2_timesteps': 1000,
                    'eval_interval': 10000,
                    'eval_episodes': 10
                },
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(config_path=config_path)
            
            # Mock evaluation environment
            mock_eval_env = Mock()
            
            with patch('mtquant.agents.training.phase2_trainer.EvalCallback') as mock_eval_callback_class, \
                 patch('mtquant.agents.training.phase2_trainer.StopTrainingOnRewardThreshold') as mock_stop_callback_class:
                
                mock_eval_callback = Mock()
                mock_stop_callback = Mock()
                mock_eval_callback_class.return_value = mock_eval_callback
                mock_stop_callback_class.return_value = mock_stop_callback
                
                callbacks = trainer._create_callbacks(mock_eval_env)
                
                assert isinstance(callbacks, list)
                assert len(callbacks) == 2
                assert mock_eval_callback in callbacks
                assert mock_stop_callback in callbacks
    
    def test_create_feature_engineer(self):
        """Test creating feature engineer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD', 'GBPUSD']},
                    'commodities': {'instruments': ['XAUUSD', 'WTIUSD']},
                    'equity': {'instruments': ['SPX500', 'NAS100']}
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(config_path=config_path)
            
            feature_engineer = trainer._create_feature_engineer()
            
            assert feature_engineer is not None
            assert hasattr(feature_engineer, 'config')
            assert feature_engineer.config.instruments == ['EURUSD', 'GBPUSD', 'XAUUSD', 'WTIUSD', 'SPX500', 'NAS100']
    
    def test_create_training_report(self):
        """Test creating training report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(
                config_path=config_path,
                output_path=temp_dir
            )
            
            # Mock evaluation
            with patch.object(trainer, 'evaluate') as mock_evaluate:
                mock_evaluate.return_value = {
                    'mean_reward': 0.5,
                    'std_reward': 0.1,
                    'mean_length': 100.0,
                    'std_length': 10.0
                }
                
                # Create mock model file
                model_path = os.path.join(temp_dir, 'meta_controller_final.zip')
                with open(model_path, 'w') as f:
                    f.write('mock model')
                
                report = trainer.create_training_report()
                
                assert isinstance(report, dict)
                assert report['phase'] == 2
                assert report['phase_name'] == 'Meta-Controller Training'
                assert 'training_stats' in report
                assert 'config' in report
                assert 'timestamp' in report
                assert 'evaluation_results' in report
    
    def test_create_training_report_no_model(self):
        """Test creating training report when no model exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(
                config_path=config_path,
                output_path=temp_dir
            )
            
            report = trainer.create_training_report()
            
            assert isinstance(report, dict)
            assert report['phase'] == 2
            assert report['phase_name'] == 'Meta-Controller Training'
            assert 'training_stats' in report
            assert 'config' in report
            assert 'timestamp' in report
            # evaluation_results is not added when no model exists
    
    def test_create_training_report_evaluation_error(self):
        """Test creating training report when evaluation fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(
                config_path=config_path,
                output_path=temp_dir
            )
            
            # Create mock model file
            model_path = os.path.join(temp_dir, 'meta_controller_final.zip')
            with open(model_path, 'w') as f:
                f.write('mock model')
            
            # Mock evaluation to raise exception
            with patch.object(trainer, 'evaluate') as mock_evaluate:
                mock_evaluate.side_effect = Exception("Evaluation failed")
                
                report = trainer.create_training_report()
                
                assert isinstance(report, dict)
                assert report['phase'] == 2
                assert report['phase_name'] == 'Meta-Controller Training'
                assert 'training_stats' in report
                assert 'config' in report
                assert 'timestamp' in report
                assert report['evaluation_results'] is None
    
    def test_save_training_report(self):
        """Test saving training report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(
                config_path=config_path,
                output_path=temp_dir
            )
            
            report = {
                'phase': 2,
                'phase_name': 'Meta-Controller Training',
                'training_stats': trainer.training_stats,
                'config': config_data,
                'timestamp': datetime.now().isoformat()
            }
            
            trainer.save_training_report(report)
            
            # Check if report file was created
            report_files = [f for f in os.listdir(temp_dir) if f.startswith('training_report_phase2_')]
            assert len(report_files) == 1
            
            # Check if report can be loaded
            report_path = os.path.join(temp_dir, report_files[0])
            with open(report_path, 'r') as f:
                loaded_report = yaml.safe_load(f)
            
            assert loaded_report['phase'] == 2
            assert loaded_report['phase_name'] == 'Meta-Controller Training'
    
    def test_save_training_report_custom_filename(self):
        """Test saving training report with custom filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'meta_controller': {
                    'state_dim': 50,
                    'hidden_dim': 128,
                    'hidden_dim_2': 64,
                    'dropout': 0.1
                },
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_2_timesteps': 1000},
                'portfolio_risk': {'max_portfolio_var': 0.05}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase2Trainer(
                config_path=config_path,
                output_path=temp_dir
            )
            
            report = {
                'phase': 2,
                'phase_name': 'Meta-Controller Training',
                'training_stats': trainer.training_stats,
                'config': config_data,
                'timestamp': datetime.now().isoformat()
            }
            
            custom_filename = 'custom_report.yaml'
            trainer.save_training_report(report, custom_filename)
            
            # Check if report file was created with custom name
            report_path = os.path.join(temp_dir, custom_filename)
            assert os.path.exists(report_path)
            
            # Check if report can be loaded
            with open(report_path, 'r') as f:
                loaded_report = yaml.safe_load(f)
            
            assert loaded_report['phase'] == 2
            assert loaded_report['phase_name'] == 'Meta-Controller Training'
