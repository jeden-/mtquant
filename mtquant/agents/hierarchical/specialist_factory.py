"""
Specialist Factory for Hierarchical Multi-Agent Trading System

This module implements the factory pattern for creating and managing specialists:
- Centralized specialist registration
- Factory method for specialist instantiation
- Configuration validation
- Specialist lifecycle management
"""

from typing import Dict, Type, List, Any, Optional
from abc import ABC, abstractmethod
import yaml
import os

from .base_specialist import BaseSpecialist
from .forex_specialist import ForexSpecialist
from .commodities_specialist import CommoditiesSpecialist
from .equity_specialist import EquitySpecialist


class SpecialistRegistry:
    """
    Registry for managing all available specialists in the hierarchical system.
    
    This class provides:
    - Registration of specialist types
    - Factory method for creating specialists
    - Configuration validation
    - Specialist lifecycle management
    """
    
    def __init__(self):
        """Initialize the specialist registry."""
        self._specialists: Dict[str, Type[BaseSpecialist]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        
        # Pre-register all available specialists
        self._register_default_specialists()
    
    def _register_default_specialists(self) -> None:
        """Register all default specialists."""
        self.register_specialist('forex', ForexSpecialist)
        self.register_specialist('commodities', CommoditiesSpecialist)
        self.register_specialist('equity', EquitySpecialist)
    
    def register_specialist(
        self,
        specialist_type: str,
        specialist_class: Type[BaseSpecialist],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a specialist type with the registry.
        
        Args:
            specialist_type: String identifier for the specialist type
            specialist_class: Specialist class (must inherit from BaseSpecialist)
            config: Optional default configuration for this specialist type
            
        Raises:
            ValueError: If specialist_class doesn't inherit from BaseSpecialist
            ValueError: If specialist_type is already registered
        """
        # Validate specialist class
        if not issubclass(specialist_class, BaseSpecialist):
            raise ValueError(f"Specialist class must inherit from BaseSpecialist, got {specialist_class}")
        
        # Check if already registered
        if specialist_type in self._specialists:
            raise ValueError(f"Specialist type '{specialist_type}' is already registered")
        
        # Register specialist
        self._specialists[specialist_type] = specialist_class
        
        # Store default config if provided
        if config is not None:
            self._configs[specialist_type] = config
    
    def create_specialist(
        self,
        specialist_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseSpecialist:
        """
        Create a specialist instance using the factory method.
        
        Args:
            specialist_type: String identifier for the specialist type
            config: Configuration dictionary (merged with defaults)
            
        Returns:
            specialist: Instance of the requested specialist
            
        Raises:
            ValueError: If specialist_type is not registered
            ValueError: If configuration validation fails
        """
        # Check if specialist type is registered
        if specialist_type not in self._specialists:
            available_types = list(self._specialists.keys())
            raise ValueError(f"Unknown specialist type '{specialist_type}'. Available: {available_types}")
        
        # Get specialist class
        specialist_class = self._specialists[specialist_type]
        
        # Merge with default config
        final_config = self._get_merged_config(specialist_type, config)
        
        # Validate configuration
        self.validate_specialist_config(specialist_type, final_config)
        
        # Create specialist instance
        try:
            specialist = specialist_class(**final_config)
            return specialist
        except Exception as e:
            raise ValueError(f"Failed to create specialist '{specialist_type}': {e}")
    
    def _get_merged_config(
        self,
        specialist_type: str,
        user_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Merge user config with default config.
        
        Args:
            specialist_type: Specialist type
            user_config: User-provided configuration
            
        Returns:
            merged_config: Merged configuration dictionary
        """
        # Start with default config
        default_config = self._configs.get(specialist_type, {})
        merged_config = default_config.copy()
        
        # Override with user config
        if user_config is not None:
            merged_config.update(user_config)
        
        return merged_config
    
    def validate_specialist_config(
        self,
        specialist_type: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        Validate specialist configuration.
        
        Args:
            specialist_type: Specialist type
            config: Configuration to validate
            
        Returns:
            is_valid: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Get specialist class
        specialist_class = self._specialists[specialist_type]
        
        # Check required parameters based on specialist type
        required_params = self._get_required_params(specialist_type)
        
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required parameter '{param}' for specialist '{specialist_type}'")
        
        # Validate instruments list
        if 'instruments' in config:
            instruments = config['instruments']
            if not isinstance(instruments, list) or len(instruments) == 0:
                raise ValueError(f"Invalid instruments list for specialist '{specialist_type}': {instruments}")
        
        # Validate feature dimensions
        if 'market_features_dim' in config:
            dim = config['market_features_dim']
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"Invalid market_features_dim for specialist '{specialist_type}': {dim}")
        
        if 'observation_dim' in config:
            dim = config['observation_dim']
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"Invalid observation_dim for specialist '{specialist_type}': {dim}")
        
        # Validate hidden dimensions
        if 'hidden_dim' in config:
            dim = config['hidden_dim']
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"Invalid hidden_dim for specialist '{specialist_type}': {dim}")
        
        # Validate dropout rate
        if 'dropout' in config:
            dropout = config['dropout']
            if not isinstance(dropout, (int, float)) or not (0.0 <= dropout <= 1.0):
                raise ValueError(f"Invalid dropout rate for specialist '{specialist_type}': {dropout}")
        
        return True
    
    def _get_required_params(self, specialist_type: str) -> List[str]:
        """
        Get required parameters for a specialist type.
        
        Args:
            specialist_type: Specialist type
            
        Returns:
            required_params: List of required parameter names
        """
        # Base required parameters
        base_params = ['instruments']
        
        # Type-specific required parameters
        type_specific_params = {
            'forex': ['market_features_dim', 'observation_dim'],
            'commodities': ['market_features_dim', 'observation_dim'],
            'equity': ['market_features_dim', 'observation_dim']
        }
        
        return base_params + type_specific_params.get(specialist_type, [])
    
    def get_all_specialists(self) -> Dict[str, Type[BaseSpecialist]]:
        """
        Get all registered specialist types.
        
        Returns:
            specialists: Dictionary mapping specialist types to classes
        """
        return self._specialists.copy()
    
    def get_specialist_info(self, specialist_type: str) -> Dict[str, Any]:
        """
        Get information about a registered specialist.
        
        Args:
            specialist_type: Specialist type
            
        Returns:
            info: Dictionary with specialist information
            
        Raises:
            ValueError: If specialist_type is not registered
        """
        if specialist_type not in self._specialists:
            raise ValueError(f"Unknown specialist type '{specialist_type}'")
        
        specialist_class = self._specialists[specialist_type]
        default_config = self._configs.get(specialist_type, {})
        
        return {
            'type': specialist_type,
            'class': specialist_class,
            'class_name': specialist_class.__name__,
            'default_config': default_config,
            'required_params': self._get_required_params(specialist_type),
            'docstring': specialist_class.__doc__
        }
    
    def list_specialists(self) -> List[str]:
        """
        List all registered specialist types.
        
        Returns:
            specialist_types: List of registered specialist type names
        """
        return list(self._specialists.keys())
    
    def load_config_from_file(self, config_path: str) -> None:
        """
        Load specialist configurations from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Load specialist configurations
            if 'specialists' in config_data:
                for specialist_type, config in config_data['specialists'].items():
                    if specialist_type in self._specialists:
                        self._configs[specialist_type] = config
                    else:
                        print(f"Warning: Unknown specialist type '{specialist_type}' in config file")
        
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML config file: {e}")
    
    def save_config_to_file(self, config_path: str) -> None:
        """
        Save current specialist configurations to YAML file.
        
        Args:
            config_path: Path to save YAML configuration file
        """
        config_data = {
            'specialists': self._configs.copy()
        }
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to save config file: {e}")
    
    def create_all_specialists(self, configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, BaseSpecialist]:
        """
        Create all registered specialists.
        
        Args:
            configs: Optional configurations for each specialist type
            
        Returns:
            specialists: Dictionary mapping specialist types to instances
        """
        specialists = {}
        configs = configs or {}
        
        for specialist_type in self._specialists.keys():
            config = configs.get(specialist_type)
            specialists[specialist_type] = self.create_specialist(specialist_type, config)
        
        return specialists
    
    def validate_specialist(self, specialist: BaseSpecialist) -> bool:
        """
        Validate a specialist instance.
        
        Args:
            specialist: Specialist instance to validate
            
        Returns:
            is_valid: True if specialist is valid
            
        Raises:
            ValueError: If specialist is invalid
        """
        # Check if specialist is instance of BaseSpecialist
        if not isinstance(specialist, BaseSpecialist):
            raise ValueError(f"Specialist must be instance of BaseSpecialist, got {type(specialist)}")
        
        # Check if specialist type is registered
        specialist_type = specialist.specialist_type
        if specialist_type not in self._specialists:
            raise ValueError(f"Unknown specialist type '{specialist_type}'")
        
        # Check if specialist class matches registered class
        expected_class = self._specialists[specialist_type]
        if not isinstance(specialist, expected_class):
            raise ValueError(f"Specialist type '{specialist_type}' should be instance of {expected_class.__name__}")
        
        # Validate specialist properties
        if not hasattr(specialist, 'instruments') or not specialist.instruments:
            raise ValueError(f"Specialist must have non-empty instruments list")
        
        if not hasattr(specialist, 'specialist_type') or not specialist.specialist_type:
            raise ValueError(f"Specialist must have specialist_type property")
        
        return True


# Global registry instance
_registry = None

def get_specialist_registry() -> SpecialistRegistry:
    """
    Get the global specialist registry instance.
    
    Returns:
        registry: Global SpecialistRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = SpecialistRegistry()
    return _registry

def create_specialist(
    specialist_type: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseSpecialist:
    """
    Convenience function to create a specialist using the global registry.
    
    Args:
        specialist_type: Specialist type
        config: Optional configuration
        
    Returns:
        specialist: Specialist instance
    """
    registry = get_specialist_registry()
    return registry.create_specialist(specialist_type, config)

def register_specialist(
    specialist_type: str,
    specialist_class: Type[BaseSpecialist],
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convenience function to register a specialist with the global registry.
    
    Args:
        specialist_type: Specialist type
        specialist_class: Specialist class
        config: Optional default configuration
    """
    registry = get_specialist_registry()
    registry.register_specialist(specialist_type, specialist_class, config)


# Unit test stubs (to be implemented in test files)
"""
def test_specialist_registry_initialization():
    '''Test SpecialistRegistry initialization.'''
    registry = SpecialistRegistry()
    
    # Check default specialists are registered
    assert 'forex' in registry.list_specialists()
    assert 'commodities' in registry.list_specialists()
    assert 'equity' in registry.list_specialists()
    
    # Check specialist classes
    assert registry.get_all_specialists()['forex'] == ForexSpecialist
    assert registry.get_all_specialists()['commodities'] == CommoditiesSpecialist
    assert registry.get_all_specialists()['equity'] == EquitySpecialist

def test_specialist_creation():
    '''Test specialist creation through factory.'''
    registry = SpecialistRegistry()
    
    # Create forex specialist
    config = {
        'instruments': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'market_features_dim': 8,
        'observation_dim': 50
    }
    
    forex_specialist = registry.create_specialist('forex', config)
    assert isinstance(forex_specialist, ForexSpecialist)
    assert forex_specialist.instruments == ['EURUSD', 'GBPUSD', 'USDJPY']

def test_config_validation():
    '''Test configuration validation.'''
    registry = SpecialistRegistry()
    
    # Valid config
    valid_config = {
        'instruments': ['XAUUSD', 'WTIUSD'],
        'market_features_dim': 6,
        'observation_dim': 50
    }
    assert registry.validate_specialist_config('commodities', valid_config) == True
    
    # Invalid config (missing required param)
    invalid_config = {'instruments': ['XAUUSD']}
    with pytest.raises(ValueError):
        registry.validate_specialist_config('commodities', invalid_config)

def test_specialist_validation():
    '''Test specialist instance validation.'''
    registry = SpecialistRegistry()
    
    # Valid specialist
    specialist = ForexSpecialist()
    assert registry.validate_specialist(specialist) == True
    
    # Invalid specialist (wrong type)
    with pytest.raises(ValueError):
        registry.validate_specialist("not a specialist")

def test_convenience_functions():
    '''Test convenience functions.'''
    # Test create_specialist
    specialist = create_specialist('equity', {
        'instruments': ['SPX500', 'NAS100', 'US30'],
        'market_features_dim': 7,
        'observation_dim': 50
    })
    assert isinstance(specialist, EquitySpecialist)
    
    # Test get_specialist_registry
    registry = get_specialist_registry()
    assert isinstance(registry, SpecialistRegistry)
"""
