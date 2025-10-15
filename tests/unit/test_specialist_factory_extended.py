"""
Extended tests for SpecialistFactory.

This module tests additional methods in SpecialistFactory that were not
covered in the original tests.
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import Mock, patch

from mtquant.agents.hierarchical.specialist_factory import (
    SpecialistRegistry, get_specialist_registry, register_specialist
)
from mtquant.agents.hierarchical.base_specialist import BaseSpecialist
from mtquant.agents.hierarchical.forex_specialist import ForexSpecialist
from mtquant.agents.hierarchical.commodities_specialist import CommoditiesSpecialist
from mtquant.agents.hierarchical.equity_specialist import EquitySpecialist


class TestSpecialistRegistryExtended:
    """Test additional SpecialistRegistry methods."""
    
    @pytest.fixture
    def registry(self):
        """Create SpecialistRegistry for testing."""
        return SpecialistRegistry()
    
    @pytest.fixture
    def mock_specialist_class(self):
        """Create mock specialist class for testing."""
        class MockSpecialist(BaseSpecialist):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.specialist_type = 'mock'
                self.instruments = ['TEST']
            
            def forward(self, market_state, instrument_states, allocation):
                return {}, {}, 0.5
            
            def get_instruments(self):
                return self.instruments
            
            def get_domain_features(self, market_data):
                return {}
            
            def calculate_confidence(self, market_state):
                return 0.8
        
        return MockSpecialist
    
    def test_register_specialist_invalid_class(self, registry):
        """Test registering specialist with invalid class."""
        with pytest.raises(ValueError, match="Specialist class must inherit from BaseSpecialist"):
            registry.register_specialist('invalid', str)  # str is not BaseSpecialist
    
    def test_register_specialist_duplicate(self, registry):
        """Test registering duplicate specialist type."""
        with pytest.raises(ValueError, match="Specialist type 'forex' is already registered"):
            registry.register_specialist('forex', ForexSpecialist)
    
    def test_register_specialist_with_config(self, registry, mock_specialist_class):
        """Test registering specialist with default config."""
        config = {
            'instruments': ['TEST'],
            'market_features_dim': 10,
            'observation_dim': 20,
            'hidden_dim': 64,
            'dropout': 0.1
        }
        
        registry.register_specialist('mock', mock_specialist_class, config)
        
        assert 'mock' in registry._specialists
        assert 'mock' in registry._configs
        assert registry._configs['mock'] == config
    
    def test_create_specialist_exception_handling(self, registry):
        """Test create_specialist with exception handling."""
        # Create invalid config that will cause exception
        invalid_config = {
            'instruments': ['EURUSD'],
            'market_features_dim': -1,  # Invalid dimension
            'observation_dim': 20,
            'hidden_dim': 64,
            'dropout': 0.1
        }
        
        with pytest.raises(ValueError, match="Invalid market_features_dim for specialist 'forex'"):
            registry.create_specialist('forex', invalid_config)
    
    def test_validate_config_missing_required_param(self, registry):
        """Test validate_specialist_config with missing required parameter."""
        config = {
            'instruments': ['EURUSD'],
            'market_features_dim': 10,
            # Missing observation_dim
            'hidden_dim': 64,
            'dropout': 0.1
        }
        
        with pytest.raises(ValueError, match="Missing required parameter 'observation_dim' for specialist 'forex'"):
            registry.validate_specialist_config('forex', config)
    
    def test_validate_config_invalid_instruments(self, registry):
        """Test validate_specialist_config with invalid instruments list."""
        config = {
            'instruments': [],  # Empty list
            'market_features_dim': 10,
            'observation_dim': 20,
            'hidden_dim': 64,
            'dropout': 0.1
        }
        
        with pytest.raises(ValueError, match="Invalid instruments list for specialist 'forex'"):
            registry.validate_specialist_config('forex', config)
    
    def test_validate_config_invalid_market_features_dim(self, registry):
        """Test validate_specialist_config with invalid market_features_dim."""
        config = {
            'instruments': ['EURUSD'],
            'market_features_dim': -1,  # Invalid dimension
            'observation_dim': 20,
            'hidden_dim': 64,
            'dropout': 0.1
        }
        
        with pytest.raises(ValueError, match="Invalid market_features_dim for specialist 'forex'"):
            registry.validate_specialist_config('forex', config)
    
    def test_validate_config_invalid_observation_dim(self, registry):
        """Test validate_specialist_config with invalid observation_dim."""
        config = {
            'instruments': ['EURUSD'],
            'market_features_dim': 10,
            'observation_dim': 0,  # Invalid dimension
            'hidden_dim': 64,
            'dropout': 0.1
        }
        
        with pytest.raises(ValueError, match="Invalid observation_dim for specialist 'forex'"):
            registry.validate_specialist_config('forex', config)
    
    def test_validate_config_invalid_hidden_dim(self, registry):
        """Test validate_specialist_config with invalid hidden_dim."""
        config = {
            'instruments': ['EURUSD'],
            'market_features_dim': 10,
            'observation_dim': 20,
            'hidden_dim': -1,  # Invalid dimension
            'dropout': 0.1
        }
        
        with pytest.raises(ValueError, match="Invalid hidden_dim for specialist 'forex'"):
            registry.validate_specialist_config('forex', config)
    
    def test_validate_config_invalid_dropout(self, registry):
        """Test validate_specialist_config with invalid dropout."""
        config = {
            'instruments': ['EURUSD'],
            'market_features_dim': 10,
            'observation_dim': 20,
            'hidden_dim': 64,
            'dropout': 1.5  # Invalid dropout (> 1.0)
        }
        
        with pytest.raises(ValueError, match="Invalid dropout rate for specialist 'forex'"):
            registry.validate_specialist_config('forex', config)
    
    def test_get_all_specialists(self, registry):
        """Test get_all_specialists method."""
        specialists = registry.get_all_specialists()
        
        assert isinstance(specialists, dict)
        assert 'forex' in specialists
        assert 'commodities' in specialists
        assert 'equity' in specialists
        assert specialists['forex'] == ForexSpecialist
        assert specialists['commodities'] == CommoditiesSpecialist
        assert specialists['equity'] == EquitySpecialist
    
    def test_get_specialist_info_valid(self, registry):
        """Test get_specialist_info with valid specialist type."""
        info = registry.get_specialist_info('forex')
        
        assert isinstance(info, dict)
        assert 'type' in info
        assert 'class' in info
        assert 'class_name' in info
        assert 'default_config' in info
        assert 'required_params' in info
        
        assert info['type'] == 'forex'
        assert info['class'] == ForexSpecialist
        assert info['class_name'] == 'ForexSpecialist'
    
    def test_get_specialist_info_invalid(self, registry):
        """Test get_specialist_info with invalid specialist type."""
        with pytest.raises(ValueError, match="Unknown specialist type 'invalid'"):
            registry.get_specialist_info('invalid')
    
    def test_list_specialists(self, registry):
        """Test list_specialists method."""
        types = registry.list_specialists()
        
        assert isinstance(types, list)
        assert 'forex' in types
        assert 'commodities' in types
        assert 'equity' in types
        assert len(types) == 3
    
    def test_load_config_from_file_valid(self, registry):
        """Test load_config_from_file with valid YAML file."""
        # Create temporary config file
        config_data = {
            'specialists': {
                'forex': {
                    'instruments': ['EURUSD', 'GBPUSD'],
                    'market_features_dim': 15,
                    'observation_dim': 25,
                    'hidden_dim': 128,
                    'dropout': 0.2
                },
                'commodities': {
                    'instruments': ['XAUUSD', 'WTIUSD'],
                    'market_features_dim': 12,
                    'observation_dim': 22,
                    'hidden_dim': 96,
                    'dropout': 0.15
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Test loading config
            registry.load_config_from_file(config_path)
            
            # Verify configs were loaded
            assert 'forex' in registry._configs
            assert 'commodities' in registry._configs
            assert registry._configs['forex']['instruments'] == ['EURUSD', 'GBPUSD']
            assert registry._configs['commodities']['instruments'] == ['XAUUSD', 'WTIUSD']
        finally:
            os.unlink(config_path)
    
    def test_load_config_from_file_nonexistent(self, registry):
        """Test load_config_from_file with nonexistent file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            registry.load_config_from_file("nonexistent_file.yaml")
    
    def test_load_config_from_file_invalid_yaml(self, registry):
        """Test load_config_from_file with invalid YAML."""
        # Create temporary invalid YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError, match="Failed to parse YAML config file"):
                registry.load_config_from_file(config_path)
        finally:
            os.unlink(config_path)
    
    def test_load_config_from_file_unknown_specialist(self, registry):
        """Test load_config_from_file with unknown specialist type."""
        # Create temporary config file with unknown specialist
        config_data = {
            'specialists': {
                'unknown': {
                    'instruments': ['TEST'],
                    'market_features_dim': 10,
                    'observation_dim': 20,
                    'hidden_dim': 64,
                    'dropout': 0.1
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Test loading config (should not raise exception, just print warning)
            registry.load_config_from_file(config_path)
            
            # Unknown specialist should not be in configs
            assert 'unknown' not in registry._configs
        finally:
            os.unlink(config_path)
    
    def test_save_config_to_file(self, registry):
        """Test save_config_to_file method."""
        # Add some configs to registry
        registry._configs['forex'] = {
            'instruments': ['EURUSD', 'GBPUSD'],
            'market_features_dim': 15,
            'observation_dim': 25,
            'hidden_dim': 128,
            'dropout': 0.2
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            # Test saving config
            registry.save_config_to_file(config_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(config_path)
            
            with open(config_path, 'r') as f:
                loaded_data = yaml.safe_load(f)
            
            assert 'specialists' in loaded_data
            assert 'forex' in loaded_data['specialists']
            assert loaded_data['specialists']['forex']['instruments'] == ['EURUSD', 'GBPUSD']
        finally:
            os.unlink(config_path)
    
    def test_save_config_to_file_exception(self, registry):
        """Test save_config_to_file with exception."""
        # Try to save to invalid path
        with pytest.raises(ValueError, match="Failed to save config file"):
            registry.save_config_to_file("/invalid/path/config.yaml")
    
    def test_create_all_specialists(self, registry):
        """Test create_all_specialists method."""
        configs = {
            'forex': {
                'instruments': ['EURUSD', 'GBPUSD'],
                'market_features_dim': 15,
                'observation_dim': 25,
                'hidden_dim': 128,
                'dropout': 0.2
            },
            'commodities': {
                'instruments': ['XAUUSD', 'WTIUSD'],
                'market_features_dim': 12,
                'observation_dim': 22,
                'hidden_dim': 96,
                'dropout': 0.15
            },
            'equity': {
                'instruments': ['SPX500', 'NAS100'],
                'market_features_dim': 10,
                'observation_dim': 20,
                'hidden_dim': 64,
                'dropout': 0.1
            }
        }
        
        specialists = registry.create_all_specialists(configs)
        
        assert isinstance(specialists, dict)
        assert 'forex' in specialists
        assert 'commodities' in specialists
        assert 'equity' in specialists
        
        # Verify specialists are instances of correct classes
        assert isinstance(specialists['forex'], ForexSpecialist)
        assert isinstance(specialists['commodities'], CommoditiesSpecialist)
        assert isinstance(specialists['equity'], EquitySpecialist)
    
    def test_create_all_specialists_no_configs(self, registry):
        """Test create_all_specialists without configs."""
        # This will fail because no default configs are set
        with pytest.raises(ValueError, match="Missing required parameter"):
            registry.create_all_specialists()
    
    def test_validate_specialist_valid(self, registry):
        """Test validate_specialist with valid specialist."""
        # Create valid specialist
        specialist = ForexSpecialist(
            instruments=['EURUSD', 'GBPUSD'],
            market_features_dim=15,
            observation_dim=25,
            hidden_dim=128,
            dropout=0.2
        )
        
        result = registry.validate_specialist(specialist)
        
        assert result == True
    
    def test_validate_specialist_invalid_class(self, registry):
        """Test validate_specialist with invalid class."""
        with pytest.raises(ValueError, match="Specialist must be instance of BaseSpecialist"):
            registry.validate_specialist("not_a_specialist")
    
    def test_validate_specialist_unknown_type(self, registry):
        """Test validate_specialist with unknown specialist type."""
        # Create mock specialist with unknown type
        specialist = Mock(spec=BaseSpecialist)
        specialist.specialist_type = 'unknown'
        specialist.instruments = ['TEST']
        
        with pytest.raises(ValueError, match="Unknown specialist type 'unknown'"):
            registry.validate_specialist(specialist)
    
    def test_validate_specialist_wrong_class(self, registry):
        """Test validate_specialist with wrong class for type."""
        # Create mock specialist with wrong class
        specialist = Mock(spec=BaseSpecialist)
        specialist.specialist_type = 'forex'
        specialist.instruments = ['EURUSD']
        
        with pytest.raises(ValueError, match="Specialist type 'forex' should be instance of ForexSpecialist"):
            registry.validate_specialist(specialist)
    
    def test_validate_specialist_no_instruments(self, registry):
        """Test validate_specialist with no instruments."""
        # Create mock specialist with no instruments
        specialist = Mock(spec=ForexSpecialist)
        specialist.specialist_type = 'forex'
        specialist.instruments = []  # Empty instruments
        
        with pytest.raises(ValueError, match="Specialist must have non-empty instruments list"):
            registry.validate_specialist(specialist)
    
    def test_validate_specialist_no_specialist_type(self, registry):
        """Test validate_specialist with no specialist_type."""
        # Create mock specialist with no specialist_type
        specialist = Mock(spec=ForexSpecialist)
        specialist.specialist_type = ''  # Empty specialist_type
        specialist.instruments = ['EURUSD']
        
        with pytest.raises(ValueError, match="Unknown specialist type ''"):
            registry.validate_specialist(specialist)


class TestGlobalFunctions:
    """Test global functions in specialist_factory.py."""
    
    @pytest.fixture
    def mock_specialist_class(self):
        """Create mock specialist class for testing."""
        class MockSpecialist(BaseSpecialist):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.specialist_type = 'mock'
                self.instruments = ['TEST']
            
            def forward(self, market_state, instrument_states, allocation):
                return {}, {}, 0.5
            
            def get_instruments(self):
                return self.instruments
            
            def get_domain_features(self, market_data):
                return {}
            
            def calculate_confidence(self, market_state):
                return 0.8
        
        return MockSpecialist
    
    def test_get_specialist_registry_singleton(self):
        """Test get_specialist_registry returns singleton."""
        registry1 = get_specialist_registry()
        registry2 = get_specialist_registry()
        
        assert registry1 is registry2
        assert isinstance(registry1, SpecialistRegistry)
    
    def test_register_specialist_global(self, mock_specialist_class):
        """Test register_specialist function."""
        # Test registering new specialist
        register_specialist('test_global', mock_specialist_class)
        
        # Verify it was registered
        registry = get_specialist_registry()
        assert 'test_global' in registry._specialists
        assert registry._specialists['test_global'] == mock_specialist_class
    
    def test_register_specialist_global_with_config(self, mock_specialist_class):
        """Test register_specialist with config."""
        config = {
            'instruments': ['TEST'],
            'market_features_dim': 10,
            'observation_dim': 20,
            'hidden_dim': 64,
            'dropout': 0.1
        }
        
        # Test registering new specialist with config
        register_specialist('test_global_config', mock_specialist_class, config)
        
        # Verify it was registered with config
        registry = get_specialist_registry()
        assert 'test_global_config' in registry._specialists
        assert 'test_global_config' in registry._configs
        assert registry._configs['test_global_config'] == config
