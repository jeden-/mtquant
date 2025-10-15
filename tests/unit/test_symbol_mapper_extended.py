"""
Extended tests for SymbolMapper.

This module tests additional methods in SymbolMapper that were not
covered in the original tests.
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import patch, Mock

from mtquant.mcp_integration.managers.symbol_mapper import SymbolMapper
from mtquant.utils.exceptions import SymbolNotFoundError


class TestSymbolMapperExtended:
    """Test additional SymbolMapper methods."""
    
    @pytest.fixture
    def sample_symbol_config(self):
        """Create sample symbol configuration."""
        return {
            'XAUUSD': {
                'instrument_type': 'commodity',
                'pip_value': 0.01,
                'typical_spread': 0.30,
                'session': '24/5',
                'broker_mappings': {
                    'ic_markets': 'XAUUSD',
                    'oanda': 'GOLD.pro',
                    'exness': 'XAUUSDm'
                }
            },
            'EURUSD': {
                'instrument_type': 'forex',
                'pip_value': 0.0001,
                'typical_spread': 0.1,
                'session': '24/5',
                'broker_mappings': {
                    'ic_markets': 'EURUSD',
                    'oanda': 'EUR_USD',
                    'exness': 'EURUSDm'
                }
            }
        }
    
    @pytest.fixture
    def temp_config_file(self, sample_symbol_config):
        """Create temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_symbol_config, f)
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        os.unlink(config_path)
    
    def test_load_config_file_not_found(self):
        """Test _load_config with file not found."""
        with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
            with patch.object(SymbolMapper, '_config_path', '/nonexistent/path.yaml'):
                with pytest.raises(FileNotFoundError):
                    SymbolMapper._load_config()
    
    def test_load_config_yaml_error(self):
        """Test _load_config with YAML parsing error."""
        # Create temporary invalid YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                with patch.object(SymbolMapper, '_config_path', config_path):
                    with pytest.raises(yaml.YAMLError):
                        SymbolMapper._load_config()
        finally:
            os.unlink(config_path)
    
    def test_to_broker_symbol_empty_parameters(self):
        """Test to_broker_symbol with empty parameters."""
        with pytest.raises(ValueError, match="standard_symbol and broker_id cannot be empty"):
            SymbolMapper.to_broker_symbol("", "ic_markets")
        
        with pytest.raises(ValueError, match="standard_symbol and broker_id cannot be empty"):
            SymbolMapper.to_broker_symbol("XAUUSD", "")
        
        with pytest.raises(ValueError, match="standard_symbol and broker_id cannot be empty"):
            SymbolMapper.to_broker_symbol("", "")
    
    def test_to_broker_symbol_standard_symbol_not_found(self, temp_config_file):
        """Test to_broker_symbol with standard symbol not found."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            with pytest.raises(SymbolNotFoundError, match="Standard symbol not found"):
                SymbolMapper.to_broker_symbol("UNKNOWN", "ic_markets")
    
    def test_to_broker_symbol_no_broker_mappings(self, temp_config_file):
        """Test to_broker_symbol with no broker mappings."""
        # Create config without broker_mappings
        config_without_mappings = {
            'XAUUSD': {
                'instrument_type': 'commodity',
                'pip_value': 0.01
                # No broker_mappings
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_without_mappings, f)
            config_path = f.name
    
        try:
            with patch.object(SymbolMapper, '_config_path', config_path):
                with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                    with pytest.raises(SymbolNotFoundError, match="No broker mappings for symbol"):
                        SymbolMapper.to_broker_symbol("XAUUSD", "ic_markets")
        finally:
            os.unlink(config_path)
    
    def test_to_broker_symbol_broker_mapping_not_found(self, temp_config_file):
        """Test to_broker_symbol with broker mapping not found."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            with pytest.raises(SymbolNotFoundError, match="Broker mapping not found"):
                SymbolMapper.to_broker_symbol("XAUUSD", "unknown_broker")
    
    def test_to_standard_symbol_empty_parameters(self):
        """Test to_standard_symbol with empty parameters."""
        with pytest.raises(ValueError, match="broker_symbol and broker_id cannot be empty"):
            SymbolMapper.to_standard_symbol("", "ic_markets")
        
        with pytest.raises(ValueError, match="broker_symbol and broker_id cannot be empty"):
            SymbolMapper.to_standard_symbol("XAUUSD", "")
        
        with pytest.raises(ValueError, match="broker_symbol and broker_id cannot be empty"):
            SymbolMapper.to_standard_symbol("", "")
    
    def test_to_standard_symbol_broker_symbol_not_found(self, temp_config_file):
        """Test to_standard_symbol with broker symbol not found."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            with pytest.raises(SymbolNotFoundError, match="Broker symbol not found"):
                SymbolMapper.to_standard_symbol("UNKNOWN", "ic_markets")
    
    def test_get_symbol_metadata_empty_symbol(self):
        """Test get_symbol_metadata with empty symbol."""
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            SymbolMapper.get_symbol_metadata("")
    
    def test_get_symbol_metadata_symbol_not_found(self, temp_config_file):
        """Test get_symbol_metadata with symbol not found."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            with pytest.raises(SymbolNotFoundError, match="Symbol not found"):
                SymbolMapper.get_symbol_metadata("UNKNOWN")
    
    def test_get_symbol_metadata_success(self, temp_config_file):
        """Test successful get_symbol_metadata."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                metadata = SymbolMapper.get_symbol_metadata("XAUUSD")
                
                assert isinstance(metadata, dict)
                assert 'instrument_type' in metadata
                assert 'pip_value' in metadata
                assert 'typical_spread' in metadata
                # Note: sample_symbol_config uses 'session' key
                assert 'session' in metadata
                assert 'broker_mappings' not in metadata  # Should be excluded
    
    def test_validate_symbol_success(self, temp_config_file):
        """Test successful symbol validation."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            assert SymbolMapper.validate_symbol("XAUUSD") == True
            assert SymbolMapper.validate_symbol("EURUSD") == True
            assert SymbolMapper.validate_symbol("UNKNOWN") == False
    
    def test_validate_symbol_config_error(self):
        """Test symbol validation with config error."""
        with patch.object(SymbolMapper, '_config_path', '/nonexistent/path.yaml'):
            with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                assert SymbolMapper.validate_symbol("XAUUSD") == False
    
    def test_get_all_standard_symbols_success(self, temp_config_file):
        """Test successful get_all_standard_symbols."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                symbols = SymbolMapper.get_all_standard_symbols()
                
                assert isinstance(symbols, list)
                assert 'XAUUSD' in symbols
                assert 'EURUSD' in symbols
                # sample_symbol_config has 2 symbols
                assert len(symbols) >= 2
    
    def test_get_all_standard_symbols_config_error(self):
        """Test get_all_standard_symbols with config error."""
        with patch.object(SymbolMapper, '_config_path', '/nonexistent/path.yaml'):
            with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                symbols = SymbolMapper.get_all_standard_symbols()

                assert isinstance(symbols, list)
                assert len(symbols) == 0
    
    def test_get_broker_symbols_empty_broker_id(self):
        """Test get_broker_symbols with empty broker_id."""
        with pytest.raises(ValueError, match="broker_id cannot be empty"):
            SymbolMapper.get_broker_symbols("")
    
    def test_get_broker_symbols_success(self, temp_config_file):
        """Test successful get_broker_symbols."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            broker_symbols = SymbolMapper.get_broker_symbols("ic_markets")
            
            assert isinstance(broker_symbols, dict)
            assert 'XAUUSD' in broker_symbols
            assert 'EURUSD' in broker_symbols
            assert broker_symbols['XAUUSD'] == 'XAUUSD'
            assert broker_symbols['EURUSD'] == 'EURUSD'
    
    def test_get_broker_symbols_partial_mapping(self, temp_config_file):
        """Test get_broker_symbols with partial broker mapping."""
        # Create config with partial broker mapping
        config_partial = {
            'XAUUSD': {
                'instrument_type': 'commodity',
                'broker_mappings': {
                    'ic_markets': 'XAUUSD'
                    # No oanda mapping
                }
            },
            'EURUSD': {
                'instrument_type': 'forex',
                'broker_mappings': {
                    'oanda': 'EUR_USD'
                    # No ic_markets mapping
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_partial, f)
            config_path = f.name
        
        try:
            with patch.object(SymbolMapper, '_config_path', config_path):
                with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                    broker_symbols = SymbolMapper.get_broker_symbols("ic_markets")
                    
                    assert isinstance(broker_symbols, dict)
                    assert 'XAUUSD' in broker_symbols
                    assert 'EURUSD' not in broker_symbols  # No mapping for this broker
                    assert broker_symbols['XAUUSD'] == 'XAUUSD'
        finally:
            os.unlink(config_path)
    
    def test_clear_cache(self):
        """Test reload method (clear_cache doesn't exist)."""
        # Set some cached data
        SymbolMapper._symbol_mappings = {'test': 'data'}

        # Clear cache using reload method
        SymbolMapper.reload()
        
        # Verify cache is cleared
        assert SymbolMapper._symbol_mappings is None
    
    def test_get_supported_brokers_success(self, temp_config_file):
        """Test successful get_supported_brokers."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                brokers = SymbolMapper.get_supported_brokers()

                assert isinstance(brokers, list)
                assert 'ic_markets' in brokers
                assert 'oanda' in brokers
                assert 'exness' in brokers
                assert len(brokers) == 3
    
    def test_get_supported_brokers_config_error(self):
        """Test get_supported_brokers with config error."""
        with patch.object(SymbolMapper, '_config_path', '/nonexistent/path.yaml'):
            with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                brokers = SymbolMapper.get_supported_brokers()

                assert isinstance(brokers, list)
                assert len(brokers) == 0
    
    def test_get_supported_brokers_no_broker_mappings(self):
        """Test get_supported_brokers with no broker mappings."""
        # Create config without broker_mappings
        config_without_mappings = {
            'XAUUSD': {
                'instrument_type': 'commodity',
                'pip_value': 0.01
                # No broker_mappings
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_without_mappings, f)
            config_path = f.name
        
        try:
            with patch.object(SymbolMapper, '_config_path', config_path):
                brokers = SymbolMapper.get_supported_brokers()
                
                assert isinstance(brokers, list)
                assert len(brokers) == 0  # No brokers found
        finally:
            os.unlink(config_path)
    
    def test_to_broker_symbol_success(self, temp_config_file):
        """Test successful to_broker_symbol conversion."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                # Test different brokers
                assert SymbolMapper.to_broker_symbol("XAUUSD", "ic_markets") == "XAUUSD"
                assert SymbolMapper.to_broker_symbol("XAUUSD", "oanda") == "GOLD.pro"
                assert SymbolMapper.to_broker_symbol("XAUUSD", "exness") == "XAUUSDm"
                
                assert SymbolMapper.to_broker_symbol("EURUSD", "ic_markets") == "EURUSD"
                assert SymbolMapper.to_broker_symbol("EURUSD", "oanda") == "EUR_USD"
                assert SymbolMapper.to_broker_symbol("EURUSD", "exness") == "EURUSDm"
    
    def test_to_standard_symbol_success(self, temp_config_file):
        """Test successful to_standard_symbol conversion."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                # Test reverse mapping
                assert SymbolMapper.to_standard_symbol("XAUUSD", "ic_markets") == "XAUUSD"
                assert SymbolMapper.to_standard_symbol("GOLD.pro", "oanda") == "XAUUSD"
                assert SymbolMapper.to_standard_symbol("XAUUSDm", "exness") == "XAUUSD"
                
                assert SymbolMapper.to_standard_symbol("EURUSD", "ic_markets") == "EURUSD"
                assert SymbolMapper.to_standard_symbol("EUR_USD", "oanda") == "EURUSD"
                assert SymbolMapper.to_standard_symbol("EURUSDm", "exness") == "EURUSD"
    
    def test_to_standard_symbol_no_broker_mappings(self, temp_config_file):
        """Test to_standard_symbol with symbol having no broker mappings."""
        # Create config with symbol without broker_mappings
        config_without_mappings = {
            'XAUUSD': {
                'instrument_type': 'commodity',
                'pip_value': 0.01
                # No broker_mappings
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_without_mappings, f)
            config_path = f.name
        
        try:
            with patch.object(SymbolMapper, '_config_path', config_path):
                with pytest.raises(SymbolNotFoundError, match="Broker symbol not found"):
                    SymbolMapper.to_standard_symbol("XAUUSD", "ic_markets")
        finally:
            os.unlink(config_path)
    
    def test_get_broker_symbols_no_broker_mappings(self, temp_config_file):
        """Test get_broker_symbols with symbols having no broker mappings."""
        # Create config with symbols without broker_mappings
        config_without_mappings = {
            'XAUUSD': {
                'instrument_type': 'commodity',
                'pip_value': 0.01
                # No broker_mappings
            },
            'EURUSD': {
                'instrument_type': 'forex',
                'pip_value': 0.0001
                # No broker_mappings
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_without_mappings, f)
            config_path = f.name
        
        try:
            with patch.object(SymbolMapper, '_config_path', config_path):
                broker_symbols = SymbolMapper.get_broker_symbols("ic_markets")
                
                assert isinstance(broker_symbols, dict)
                assert len(broker_symbols) == 0  # No mappings found
        finally:
            os.unlink(config_path)
    
    def test_get_supported_brokers_duplicate_brokers(self, temp_config_file):
        """Test get_supported_brokers with duplicate brokers."""
        # Create config with duplicate brokers
        config_duplicate = {
            'XAUUSD': {
                'instrument_type': 'commodity',
                'broker_mappings': {
                    'ic_markets': 'XAUUSD',
                    'oanda': 'GOLD.pro'
                }
            },
            'EURUSD': {
                'instrument_type': 'forex',
                'broker_mappings': {
                    'ic_markets': 'EURUSD',  # Duplicate broker
                    'exness': 'EURUSDm'
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_duplicate, f)
            config_path = f.name

        try:
            with patch.object(SymbolMapper, '_config_path', config_path):
                with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                    brokers = SymbolMapper.get_supported_brokers()

                    assert isinstance(brokers, list)
                    assert 'ic_markets' in brokers
                    assert 'oanda' in brokers
                    assert 'exness' in brokers
                    assert len(brokers) == 3  # Should not have duplicates
        finally:
            os.unlink(config_path)
    
    def test_load_config_success(self, temp_config_file):
        """Test successful _load_config."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            with patch.object(SymbolMapper, '_symbol_mappings', None):  # Clear cache
                config = SymbolMapper._load_config()

                assert isinstance(config, dict)
                assert 'XAUUSD' in config
                assert 'EURUSD' in config
                assert config['XAUUSD']['instrument_type'] == 'commodity'
                assert config['EURUSD']['instrument_type'] == 'forex'
    
    def test_load_config_caching(self, temp_config_file):
        """Test _load_config caching behavior."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            # First call
            config1 = SymbolMapper._load_config()
            
            # Second call should return cached data
            config2 = SymbolMapper._load_config()
            
            # Should be the same object (cached)
            assert config1 is config2
    
    def test_load_config_cache_cleared(self, temp_config_file):
        """Test _load_config after cache is cleared."""
        with patch.object(SymbolMapper, '_config_path', temp_config_file):
            # First call
            config1 = SymbolMapper._load_config()

            # Clear cache using reload method
            SymbolMapper.reload()
            
            # Second call should reload from file
            config2 = SymbolMapper._load_config()
            
            # Should be different objects (not cached)
            assert config1 is not config2
            # Both should be valid configs
            assert isinstance(config1, dict)
            assert isinstance(config2, dict)
