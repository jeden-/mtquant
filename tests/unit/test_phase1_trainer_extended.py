"""
Extended unit tests for phase1_trainer.py to increase coverage.
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

from mtquant.agents.training.phase1_trainer import Phase1Trainer


class TestPhase1Trainer:
    """Test Phase1Trainer class."""
    
    def test_phase1_trainer_initialization(self):
        """Test Phase1Trainer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock config file
            config_data = {
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
                    'phase_1_timesteps': 10000
                }
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(
                config_path=config_path,
                data_path=temp_dir,
                output_path=temp_dir,
                n_envs=4
            )
            
            assert trainer.config_path == config_path
            assert trainer.data_path == temp_dir
            assert trainer.output_path == temp_dir
            assert trainer.n_envs == 4
            assert trainer.config == config_data
            assert isinstance(trainer.training_stats, dict)
            assert 'forex' in trainer.training_stats
            assert 'commodities' in trainer.training_stats
            assert 'equity' in trainer.training_stats
    
    def test_load_config(self):
        """Test loading configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_1_timesteps': 5000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(config_path=config_path)
            
            assert trainer.config == config_data
    
    def test_load_config_file_not_found(self):
        """Test loading configuration when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Phase1Trainer(config_path="nonexistent_config.yaml")
    
    def test_load_market_data(self):
        """Test loading market data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock config
            config_data = {
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_1_timesteps': 1000}
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
            
            trainer = Phase1Trainer(
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
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(
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
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(config_path=config_path)
            
            # Test EURUSD (USD pair)
            df_eurusd = trainer._create_dummy_data('EURUSD')
            assert isinstance(df_eurusd, pd.DataFrame)
            assert len(df_eurusd) > 0
            assert 'open' in df_eurusd.columns
            assert 'high' in df_eurusd.columns
            assert 'low' in df_eurusd.columns
            assert 'close' in df_eurusd.columns
            assert 'volume' in df_eurusd.columns
            
            # Test XAUUSD (USD pair)
            df_xauusd = trainer._create_dummy_data('XAUUSD')
            assert isinstance(df_xauusd, pd.DataFrame)
            assert len(df_xauusd) > 0
            
            # Test SPX500 (non-USD pair)
            df_spx500 = trainer._create_dummy_data('SPX500')
            assert isinstance(df_spx500, pd.DataFrame)
            assert len(df_spx500) > 0
    
    def test_create_feature_engineer(self):
        """Test creating feature engineer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {'instruments': ['EURUSD']},
                    'commodities': {'instruments': ['XAUUSD']},
                    'equity': {'instruments': ['SPX500']}
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(config_path=config_path)
            
            feature_engineer = trainer._create_feature_engineer()
            
            assert feature_engineer is not None
            assert hasattr(feature_engineer, 'config')
            assert feature_engineer.config.instruments == ['EURUSD', 'XAUUSD', 'SPX500']
    
    def test_create_specialists(self):
        """Test creating specialists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(config_path=config_path)
            
            specialists = trainer._create_specialists()
            
            assert isinstance(specialists, dict)
            assert 'forex' in specialists
            assert 'commodities' in specialists
            assert 'equity' in specialists
            
            # Check specialist types
            assert hasattr(specialists['forex'], 'instruments')
            assert hasattr(specialists['commodities'], 'instruments')
            assert hasattr(specialists['equity'], 'instruments')
    
    def test_train_specialist(self):
        """Test training individual specialist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(
                config_path=config_path,
                data_path=temp_dir,
                output_path=temp_dir
            )
            
            # Mock SpecialistTrainer
            with patch('mtquant.agents.training.phase1_trainer.SpecialistTrainer') as mock_trainer_class:
                mock_trainer = Mock()
                mock_trainer_class.return_value = mock_trainer
                
                trainer.train_specialist('forex', 1000)
                
                # Check that SpecialistTrainer was created
                mock_trainer_class.assert_called_once()
                mock_trainer.train.assert_called_once_with(1000)
                
                # Check training stats were updated
                assert trainer.training_stats['forex']['total_timesteps'] == 1000
                assert 'training_time' in trainer.training_stats['forex']
    
    def test_train_specialist_exception(self):
        """Test training specialist with exception."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(
                config_path=config_path,
                data_path=temp_dir,
                output_path=temp_dir
            )
            
            # Mock SpecialistTrainer to raise exception
            with patch('mtquant.agents.training.phase1_trainer.SpecialistTrainer') as mock_trainer_class:
                mock_trainer = Mock()
                mock_trainer.train.side_effect = Exception("Training failed")
                mock_trainer_class.return_value = mock_trainer
                
                with pytest.raises(Exception, match="Training failed"):
                    trainer.train_specialist('forex', 1000)
    
    def test_train_all_specialists_sequential(self):
        """Test training all specialists sequentially."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(
                config_path=config_path,
                data_path=temp_dir,
                output_path=temp_dir
            )
            
            # Mock SpecialistTrainer
            with patch('mtquant.agents.training.phase1_trainer.SpecialistTrainer') as mock_trainer_class:
                mock_trainer = Mock()
                mock_trainer_class.return_value = mock_trainer
                
                trainer.train_all_specialists(1000, parallel=False)
                
                # Should be called 3 times (once for each specialist)
                assert mock_trainer_class.call_count == 3
                assert mock_trainer.train.call_count == 3
    
    def test_train_all_specialists_parallel(self):
        """Test training all specialists in parallel."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(
                config_path=config_path,
                data_path=temp_dir,
                output_path=temp_dir
            )
            
            # Mock multiprocessing
            with patch('mtquant.agents.training.phase1_trainer.mp.Process') as mock_process_class:
                mock_process = Mock()
                mock_process_class.return_value = mock_process
                
                trainer.train_all_specialists(1000, parallel=True)
                
                # Should create 3 processes
                assert mock_process_class.call_count == 3
                assert mock_process.start.call_count == 3
                assert mock_process.join.call_count == 3
    
    def test_evaluate_specialists(self):
        """Test evaluating specialists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(
                config_path=config_path,
                data_path=temp_dir,
                output_path=temp_dir
            )
            
            # Create mock model files
            for specialist_type in ['forex', 'commodities', 'equity']:
                model_path = os.path.join(temp_dir, f'{specialist_type}_final.zip')
                with open(model_path, 'w') as f:
                    f.write('mock model')
            
            # Mock SpecialistTrainer
            with patch('mtquant.agents.training.phase1_trainer.SpecialistTrainer') as mock_trainer_class:
                mock_trainer = Mock()
                mock_trainer.evaluate.return_value = {
                    'mean_reward': 0.5,
                    'std_reward': 0.1,
                    'mean_length': 100.0,
                    'std_length': 10.0
                }
                mock_trainer_class.return_value = mock_trainer
                
                evaluation_results = trainer.evaluate_specialists(10)
                
                assert isinstance(evaluation_results, dict)
                assert 'forex' in evaluation_results
                assert 'commodities' in evaluation_results
                assert 'equity' in evaluation_results
                
                for specialist_type, results in evaluation_results.items():
                    assert results is not None
                    assert 'mean_reward' in results
                    assert 'std_reward' in results
                    assert 'mean_length' in results
                    assert 'std_length' in results
    
    def test_evaluate_specialists_no_models(self):
        """Test evaluating specialists when no models exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(
                config_path=config_path,
                data_path=temp_dir,
                output_path=temp_dir
            )
            
            evaluation_results = trainer.evaluate_specialists(10)
            
            assert isinstance(evaluation_results, dict)
            assert 'forex' in evaluation_results
            assert 'commodities' in evaluation_results
            assert 'equity' in evaluation_results
            
            # All should be None since no models exist
            for specialist_type, results in evaluation_results.items():
                assert results is None
    
    def test_create_training_report(self):
        """Test creating training report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(
                config_path=config_path,
                data_path=temp_dir,
                output_path=temp_dir
            )
            
            # Mock evaluation
            with patch.object(trainer, 'evaluate_specialists') as mock_evaluate:
                mock_evaluate.return_value = {
                    'forex': {'mean_reward': 0.5},
                    'commodities': {'mean_reward': 0.6},
                    'equity': {'mean_reward': 0.7}
                }
                
                report = trainer.create_training_report()
                
                assert isinstance(report, dict)
                assert report['phase'] == 1
                assert report['phase_name'] == 'Individual Specialist Training'
                assert 'training_stats' in report
                assert 'config' in report
                assert 'timestamp' in report
                assert 'evaluation_results' in report
    
    def test_create_training_report_evaluation_error(self):
        """Test creating training report when evaluation fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(
                config_path=config_path,
                data_path=temp_dir,
                output_path=temp_dir
            )
            
            # Mock evaluation to raise exception
            with patch.object(trainer, 'evaluate_specialists') as mock_evaluate:
                mock_evaluate.side_effect = Exception("Evaluation failed")
                
                report = trainer.create_training_report()
                
                assert isinstance(report, dict)
                assert report['phase'] == 1
                assert report['phase_name'] == 'Individual Specialist Training'
                assert 'training_stats' in report
                assert 'config' in report
                assert 'timestamp' in report
                assert report['evaluation_results'] is None
    
    def test_save_training_report(self):
        """Test saving training report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(
                config_path=config_path,
                data_path=temp_dir,
                output_path=temp_dir
            )
            
            report = {
                'phase': 1,
                'phase_name': 'Individual Specialist Training',
                'training_stats': trainer.training_stats,
                'config': config_data,
                'timestamp': datetime.now().isoformat()
            }
            
            trainer.save_training_report(report)
            
            # Check if report file was created
            report_files = [f for f in os.listdir(temp_dir) if f.startswith('training_report_phase1_')]
            assert len(report_files) == 1
            
            # Check if report can be loaded
            report_path = os.path.join(temp_dir, report_files[0])
            with open(report_path, 'r') as f:
                loaded_report = yaml.safe_load(f)
            
            assert loaded_report['phase'] == 1
            assert loaded_report['phase_name'] == 'Individual Specialist Training'
    
    def test_save_training_report_custom_filename(self):
        """Test saving training report with custom filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                'specialists': {
                    'forex': {
                        'instruments': ['EURUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'commodities': {
                        'instruments': ['XAUUSD'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    },
                    'equity': {
                        'instruments': ['SPX500'],
                        'market_features_dim': 8,
                        'observation_dim': 50,
                        'hidden_dim': 128
                    }
                },
                'training': {'phase_1_timesteps': 1000}
            }
            
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            trainer = Phase1Trainer(
                config_path=config_path,
                data_path=temp_dir,
                output_path=temp_dir
            )
            
            report = {
                'phase': 1,
                'phase_name': 'Individual Specialist Training',
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
            
            assert loaded_report['phase'] == 1
            assert loaded_report['phase_name'] == 'Individual Specialist Training'
