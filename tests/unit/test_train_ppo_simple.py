"""
Simple unit tests for train_ppo.py without Stable Baselines3 dependencies.
Tests core business logic without complex ML framework interactions.
"""

import pytest
import os
import tempfile
import yaml
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from mtquant.agents.training.train_ppo import (
    load_config,
    prepare_data,
    create_env
)
from mtquant.agents.environments.base_trading_env import MTQuantTradingEnv
from mtquant.data.processors.feature_engineering import create_sample_data
from mtquant.risk_management.position_sizer import PositionSizer


class TestLoadConfig:
    """Test load_config function."""
    
    def test_load_config_success(self):
        """Test successful config loading."""
        # Create temporary config file
        config_data = {
            'ppo_agent': {
                'initial_capital': 10000,
                'learning_rate': 0.0003
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            result = load_config(config_path)
            assert isinstance(result, dict)
            assert 'ppo_agent' in result
            assert result['ppo_agent']['initial_capital'] == 10000
        finally:
            os.unlink(config_path)
    
    def test_load_config_file_not_found(self):
        """Test config loading with file not found."""
        result = load_config('/nonexistent/path.yaml')
        
        # Should return default config
        assert isinstance(result, dict)
        assert 'ppo_agent' in result
        assert 'position_sizing' in result
    
    def test_load_config_invalid_yaml(self):
        """Test config loading with invalid YAML."""
        # Create temporary invalid YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            result = load_config(config_path)
            # Should return default config on error
            assert isinstance(result, dict)
            assert 'ppo_agent' in result
        finally:
            os.unlink(config_path)
    
    def test_load_config_default_values(self):
        """Test default configuration values."""
        result = load_config('/nonexistent/path.yaml')
        
        # Check default PPO agent config
        ppo_config = result['ppo_agent']
        assert ppo_config['initial_capital'] == 10000
        assert ppo_config['transaction_cost'] == 0.003
        assert ppo_config['learning_rate'] == 0.0003
        assert ppo_config['n_steps'] == 2048
        assert ppo_config['batch_size'] == 64
        assert ppo_config['gamma'] == 0.99
        assert ppo_config['gae_lambda'] == 0.95
        assert ppo_config['ent_coef'] == 0.01
        assert ppo_config['vf_coef'] == 0.5
        assert ppo_config['max_grad_norm'] == 0.5
        assert ppo_config['clip_range'] == 0.2
        assert ppo_config['n_epochs'] == 10
        
        # Check default position sizing config
        position_config = result['position_sizing']
        assert 'volatility' in position_config
        assert position_config['volatility']['risk_per_trade'] == 0.02
        assert position_config['volatility']['atr_multiplier'] == 2.0
        assert position_config['volatility']['max_position_pct'] == 0.05


class TestPrepareData:
    """Test prepare_data function."""
    
    def test_prepare_data_with_file(self):
        """Test data preparation with existing file."""
        # Create temporary data file
        data = pd.DataFrame({
            'open': np.random.randn(1000),
            'high': np.random.randn(1000),
            'low': np.random.randn(1000),
            'close': np.random.randn(1000),
            'volume': np.random.randn(1000)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name)
            data_path = f.name
        
        try:
            result = prepare_data("XAUUSD", data_path)
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
        finally:
            os.unlink(data_path)
    
    def test_prepare_data_generate_sample(self):
        """Test data preparation with sample generation."""
        with patch('mtquant.agents.training.train_ppo.create_sample_data') as mock_create:
            mock_data = pd.DataFrame({
                'open': np.random.randn(1000),
                'high': np.random.randn(1000),
                'low': np.random.randn(1000),
                'close': np.random.randn(1000),
                'volume': np.random.randn(1000)
            })
            mock_create.return_value = mock_data
            
            with patch('mtquant.agents.training.train_ppo.prepare_training_data') as mock_prepare:
                mock_prepare.return_value = mock_data
                
                result = prepare_data("XAUUSD")
                assert isinstance(result, pd.DataFrame)
                assert len(result) > 0
                mock_create.assert_called_once_with("XAUUSD", periods=2000, seed=None)
    
    def test_prepare_data_with_seed(self):
        """Test data preparation with specific seed."""
        with patch('mtquant.agents.training.train_ppo.create_sample_data') as mock_create:
            mock_data = pd.DataFrame({
                'open': np.random.randn(1000),
                'high': np.random.randn(1000),
                'low': np.random.randn(1000),
                'close': np.random.randn(1000),
                'volume': np.random.randn(1000)
            })
            mock_create.return_value = mock_data
            
            with patch('mtquant.agents.training.train_ppo.prepare_training_data') as mock_prepare:
                mock_prepare.return_value = mock_data
                
                result = prepare_data("XAUUSD", seed=42)
                assert isinstance(result, pd.DataFrame)
                mock_create.assert_called_once_with("XAUUSD", periods=2000, seed=42)
    
    def test_prepare_data_insufficient_data_fallback(self):
        """Test data preparation with insufficient data fallback."""
        with patch('mtquant.agents.training.train_ppo.create_sample_data') as mock_create:
            mock_data = pd.DataFrame({
                'open': np.random.randn(50),  # Less than 100 rows
                'high': np.random.randn(50),
                'low': np.random.randn(50),
                'close': np.random.randn(50),
                'volume': np.random.randn(50)
            })
            mock_create.return_value = mock_data
            
            with patch('mtquant.agents.training.train_ppo.prepare_training_data') as mock_prepare:
                mock_prepare.return_value = mock_data
                
                # Should fallback to sample data instead of raising error
                result = prepare_data("XAUUSD")
                assert isinstance(result, pd.DataFrame)
    
    def test_prepare_data_with_missing_values(self):
        """Test data preparation with missing values."""
        with patch('mtquant.agents.training.train_ppo.create_sample_data') as mock_create:
            mock_data = pd.DataFrame({
                'open': [1, 2, np.nan, 4, 5] * 200,
                'high': [1, 2, 3, 4, 5] * 200,
                'low': [1, 2, 3, 4, 5] * 200,
                'close': [1, 2, 3, 4, 5] * 200,
                'volume': [1, 2, 3, 4, 5] * 200
            })
            mock_create.return_value = mock_data
            
            with patch('mtquant.agents.training.train_ppo.prepare_training_data') as mock_prepare:
                mock_prepare.return_value = mock_data
                
                result = prepare_data("XAUUSD")
                assert isinstance(result, pd.DataFrame)
                assert len(result) > 0
    
    def test_prepare_data_exception_fallback(self):
        """Test data preparation with exception fallback."""
        with patch('mtquant.agents.training.train_ppo.create_sample_data') as mock_create:
            mock_create.side_effect = Exception("Test exception")
            
            with patch('mtquant.agents.training.train_ppo.prepare_training_data') as mock_prepare:
                mock_prepare.side_effect = Exception("Test exception")
                
                # Should fallback to sample data
                with patch('mtquant.agents.training.train_ppo.create_sample_data') as mock_fallback:
                    fallback_data = pd.DataFrame({
                        'open': np.random.randn(1000),
                        'high': np.random.randn(1000),
                        'low': np.random.randn(1000),
                        'close': np.random.randn(1000),
                        'volume': np.random.randn(1000)
                    })
                    mock_fallback.return_value = fallback_data
                    
                    result = prepare_data("XAUUSD")
                    assert isinstance(result, pd.DataFrame)


class TestCreateEnv:
    """Test create_env function."""
    
    def test_create_env_success(self):
        """Test successful environment creation."""
        config = {
            'ppo_agent': {
                'initial_capital': 10000,
                'transaction_cost': 0.003
            },
            'position_sizing': {
                'volatility': {
                    'risk_per_trade': 0.02
                }
            }
        }
        
        data = pd.DataFrame({
            'open': np.random.randn(1000),
            'high': np.random.randn(1000),
            'low': np.random.randn(1000),
            'close': np.random.randn(1000),
            'volume': np.random.randn(1000)
        })
        
        with patch('mtquant.agents.training.train_ppo.MTQuantTradingEnv') as mock_env_class:
            mock_env = Mock()
            mock_env.reset.return_value = np.array([1, 2, 3, 4, 5])
            mock_env_class.return_value = mock_env
            
            result = create_env(data, config, "XAUUSD")
            assert result == mock_env
            mock_env_class.assert_called_once()
            mock_env.reset.assert_called_once()
    
    def test_create_env_insufficient_data(self):
        """Test environment creation with insufficient data."""
        config = {
            'ppo_agent': {
                'initial_capital': 10000,
                'transaction_cost': 0.003
            }
        }
        
        data = pd.DataFrame({
            'open': np.random.randn(50),  # Less than 100 rows
            'high': np.random.randn(50),
            'low': np.random.randn(50),
            'close': np.random.randn(50),
            'volume': np.random.randn(50)
        })
        
        with pytest.raises(ValueError, match="Insufficient data for environment"):
            create_env(data, config, "XAUUSD")
    
    def test_create_env_reset_failed(self):
        """Test environment creation with reset failure."""
        config = {
            'ppo_agent': {
                'initial_capital': 10000,
                'transaction_cost': 0.003
            }
        }
        
        data = pd.DataFrame({
            'open': np.random.randn(1000),
            'high': np.random.randn(1000),
            'low': np.random.randn(1000),
            'close': np.random.randn(1000),
            'volume': np.random.randn(1000)
        })
        
        with patch('mtquant.agents.training.train_ppo.MTQuantTradingEnv') as mock_env_class:
            mock_env = Mock()
            mock_env.reset.return_value = None  # Reset fails
            mock_env_class.return_value = mock_env
            
            with pytest.raises(ValueError, match="Environment reset failed"):
                create_env(data, config, "XAUUSD")
    
    def test_create_env_reset_empty_observation(self):
        """Test environment creation with empty observation."""
        config = {
            'ppo_agent': {
                'initial_capital': 10000,
                'transaction_cost': 0.003
            }
        }
        
        data = pd.DataFrame({
            'open': np.random.randn(1000),
            'high': np.random.randn(1000),
            'low': np.random.randn(1000),
            'close': np.random.randn(1000),
            'volume': np.random.randn(1000)
        })
        
        with patch('mtquant.agents.training.train_ppo.MTQuantTradingEnv') as mock_env_class:
            mock_env = Mock()
            mock_env.reset.return_value = []  # Empty observation
            mock_env_class.return_value = mock_env
            
            with pytest.raises(ValueError, match="Environment reset failed"):
                create_env(data, config, "XAUUSD")
    
    def test_create_env_exception(self):
        """Test environment creation with exception."""
        config = {
            'ppo_agent': {
                'initial_capital': 10000,
                'transaction_cost': 0.003
            }
        }
        
        data = pd.DataFrame({
            'open': np.random.randn(1000),
            'high': np.random.randn(1000),
            'low': np.random.randn(1000),
            'close': np.random.randn(1000),
            'volume': np.random.randn(1000)
        })
        
        with patch('mtquant.agents.training.train_ppo.MTQuantTradingEnv') as mock_env_class:
            mock_env_class.side_effect = Exception("Environment creation failed")
            
            with pytest.raises(Exception, match="Environment creation failed"):
                create_env(data, config, "XAUUSD")
    
    def test_create_env_position_sizer_creation(self):
        """Test that PositionSizer is created correctly."""
        config = {
            'ppo_agent': {
                'initial_capital': 10000,
                'transaction_cost': 0.003
            },
            'position_sizing': {
                'volatility': {
                    'risk_per_trade': 0.02,
                    'atr_multiplier': 2.0,
                    'max_position_pct': 0.05
                }
            }
        }
        
        data = pd.DataFrame({
            'open': np.random.randn(1000),
            'high': np.random.randn(1000),
            'low': np.random.randn(1000),
            'close': np.random.randn(1000),
            'volume': np.random.randn(1000)
        })
        
        with patch('mtquant.agents.training.train_ppo.MTQuantTradingEnv') as mock_env_class:
            mock_env = Mock()
            mock_env.reset.return_value = np.array([1, 2, 3, 4, 5])
            mock_env_class.return_value = mock_env
            
            with patch('mtquant.agents.training.train_ppo.PositionSizer') as mock_sizer_class:
                mock_sizer = Mock()
                mock_sizer_class.return_value = mock_sizer
                
                result = create_env(data, config, "XAUUSD")
                
                # Verify PositionSizer was created with correct config
                mock_sizer_class.assert_called_once_with(config['position_sizing'])
                assert result == mock_env


class TestDataValidation:
    """Test data validation logic."""
    
    def test_data_length_validation(self):
        """Test that data length validation works correctly."""
        config = {
            'ppo_agent': {
                'initial_capital': 10000,
                'transaction_cost': 0.003
            }
        }
        
        # Test with exactly 100 rows (minimum)
        data_min = pd.DataFrame({
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.random.randn(100)
        })
        
        with patch('mtquant.agents.training.train_ppo.MTQuantTradingEnv') as mock_env_class:
            mock_env = Mock()
            mock_env.reset.return_value = np.array([1, 2, 3, 4, 5])
            mock_env_class.return_value = mock_env
            
            result = create_env(data_min, config, "XAUUSD")
            assert result == mock_env  # Should succeed with exactly 100 rows
    
    def test_missing_values_handling(self):
        """Test handling of missing values in data."""
        with patch('mtquant.agents.training.train_ppo.create_sample_data') as mock_create:
            # Create data with missing values
            mock_data = pd.DataFrame({
                'open': [1, 2, np.nan, 4, 5] * 200,
                'high': [1, 2, 3, np.nan, 5] * 200,
                'low': [1, 2, 3, 4, 5] * 200,
                'close': [1, 2, 3, 4, 5] * 200,
                'volume': [1, 2, 3, 4, 5] * 200
            })
            mock_create.return_value = mock_data
            
            with patch('mtquant.agents.training.train_ppo.prepare_training_data') as mock_prepare:
                mock_prepare.return_value = mock_data
                
                result = prepare_data("XAUUSD")
                assert isinstance(result, pd.DataFrame)
                # Should handle missing values gracefully
                assert len(result) > 0


class TestConfigHandling:
    """Test configuration handling edge cases."""
    
    def test_config_with_missing_keys(self):
        """Test config handling when some keys are missing."""
        config = {
            'ppo_agent': {
                'initial_capital': 10000,
                'transaction_cost': 0.003
                # Missing other keys
            }
        }
        
        data = pd.DataFrame({
            'open': np.random.randn(1000),
            'high': np.random.randn(1000),
            'low': np.random.randn(1000),
            'close': np.random.randn(1000),
            'volume': np.random.randn(1000)
        })
        
        with patch('mtquant.agents.training.train_ppo.MTQuantTradingEnv') as mock_env_class:
            mock_env = Mock()
            mock_env.reset.return_value = np.array([1, 2, 3, 4, 5])
            mock_env_class.return_value = mock_env
            
            # Should handle missing config keys gracefully
            result = create_env(data, config, "XAUUSD")
            assert result == mock_env
    
    def test_empty_config(self):
        """Test handling of empty configuration."""
        config = {}
        
        data = pd.DataFrame({
            'open': np.random.randn(1000),
            'high': np.random.randn(1000),
            'low': np.random.randn(1000),
            'close': np.random.randn(1000),
            'volume': np.random.randn(1000)
        })
        
        # Empty config should raise KeyError when accessing required keys
        with pytest.raises(KeyError, match="'ppo_agent'"):
            create_env(data, config, "XAUUSD")
