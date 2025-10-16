"""
Symbol Mapper for MTQuant Trading System

This module provides centralized symbol mapping between standard symbols
(e.g., XAUUSD, EURUSD) and broker-specific symbols across different brokers.

The SymbolMapper loads mappings from config/symbols.yaml and provides
methods for converting between standard and broker-specific symbols.

Example:
    # Convert standard symbol to broker-specific
    broker_symbol = SymbolMapper.to_broker_symbol('XAUUSD', 'ic_markets')
    # Returns: 'XAUUSD'
    
    # Convert broker-specific to standard
    standard_symbol = SymbolMapper.to_standard_symbol('GOLD.pro', 'oanda')
    # Returns: 'XAUUSD'
    
    # Get symbol metadata
    metadata = SymbolMapper.get_symbol_metadata('XAUUSD')
    # Returns: {'instrument_type': 'commodity', 'pip_value': 0.01, ...}
"""

import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
from mtquant.utils.exceptions import SymbolNotFoundError
from mtquant.utils.logger import get_logger

logger = get_logger(__name__)


class SymbolMapper:
    """
    Centralized symbol mapping between standard and broker-specific symbols.
    
    This class loads symbol mappings from config/symbols.yaml and provides
    methods for converting between standard symbols (XAUUSD, EURUSD, etc.)
    and broker-specific symbols (GOLD.pro, XAUUSDm, etc.).
    
    The class uses caching to avoid repeated YAML file reads and provides
    comprehensive error handling for missing mappings.
    """
    
    _symbol_mappings: Optional[Dict[str, Any]] = None
    _config_path: Optional[Path] = None
    
    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        """
        Load symbol mappings from config/symbols.yaml.
        
        Returns:
            Dictionary containing symbol mappings
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if cls._symbol_mappings is not None:
            return cls._symbol_mappings
            
        if cls._config_path is None:
            cls._config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'symbols.yaml'
        
        try:
            with open(cls._config_path, 'r', encoding='utf-8') as f:
                cls._symbol_mappings = yaml.safe_load(f)
            logger.info(f"Loaded symbol mappings from {cls._config_path}")
            return cls._symbol_mappings
        except FileNotFoundError:
            logger.error(f"Symbol config file not found: {cls._config_path}")
            raise FileNotFoundError(f"Symbol config file not found: {cls._config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise yaml.YAMLError(f"Failed to parse YAML config: {e}")
    
    @classmethod
    def to_broker_symbol(cls, standard_symbol: str, broker_id: str) -> str:
        """
        Convert standard symbol to broker-specific symbol.
        
        Args:
            standard_symbol: Standard symbol (e.g., 'XAUUSD', 'EURUSD')
            broker_id: Broker identifier (e.g., 'ic_markets', 'oanda')
            
        Returns:
            Broker-specific symbol
            
        Raises:
            SymbolNotFoundError: If symbol or broker mapping doesn't exist
            ValueError: If broker_id is invalid
            
        Example:
            >>> SymbolMapper.to_broker_symbol('XAUUSD', 'ic_markets')
            'XAUUSD'
            >>> SymbolMapper.to_broker_symbol('XAUUSD', 'oanda')
            'GOLD.pro'
        """
        if not standard_symbol or not broker_id:
            raise ValueError("standard_symbol and broker_id cannot be empty")
        
        mappings = cls._load_config()
        
        if standard_symbol not in mappings:
            logger.warning(f"Standard symbol not found: {standard_symbol}")
            raise SymbolNotFoundError(f"Standard symbol not found: {standard_symbol}")
        
        symbol_config = mappings[standard_symbol]
        
        if 'broker_mappings' not in symbol_config:
            logger.warning(f"No broker mappings for symbol: {standard_symbol}")
            raise SymbolNotFoundError(f"No broker mappings for symbol: {standard_symbol}")
        
        broker_mappings = symbol_config['broker_mappings']
        
        if broker_id not in broker_mappings:
            logger.warning(f"Broker mapping not found: {broker_id} for {standard_symbol}")
            raise SymbolNotFoundError(f"Broker mapping not found: {broker_id} for {standard_symbol}")
        
        broker_symbol = broker_mappings[broker_id]
        logger.debug(f"Mapped {standard_symbol} -> {broker_symbol} for {broker_id}")
        
        return broker_symbol
    
    @classmethod
    def to_standard_symbol(cls, broker_symbol: str, broker_id: str) -> str:
        """
        Convert broker-specific symbol to standard symbol.
        
        Args:
            broker_symbol: Broker-specific symbol (e.g., 'GOLD.pro', 'XAUUSDm')
            broker_id: Broker identifier (e.g., 'oanda', 'exness')
            
        Returns:
            Standard symbol
            
        Raises:
            SymbolNotFoundError: If broker symbol or broker mapping doesn't exist
            
        Example:
            >>> SymbolMapper.to_standard_symbol('GOLD.pro', 'oanda')
            'XAUUSD'
            >>> SymbolMapper.to_standard_symbol('XAUUSDm', 'exness')
            'XAUUSD'
        """
        if not broker_symbol or not broker_id:
            raise ValueError("broker_symbol and broker_id cannot be empty")
        
        mappings = cls._load_config()
        
        # Search through all symbols to find the reverse mapping
        for standard_symbol, symbol_config in mappings.items():
            if 'broker_mappings' not in symbol_config:
                continue
                
            broker_mappings = symbol_config['broker_mappings']
            
            if broker_id in broker_mappings and broker_mappings[broker_id] == broker_symbol:
                logger.debug(f"Mapped {broker_symbol} -> {standard_symbol} for {broker_id}")
                return standard_symbol
        
        logger.warning(f"Broker symbol not found: {broker_symbol} for {broker_id}")
        raise SymbolNotFoundError(f"Broker symbol not found: {broker_symbol} for {broker_id}")
    
    @classmethod
    def get_symbol_metadata(cls, symbol: str) -> Dict[str, Any]:
        """
        Get metadata for a standard symbol.
        
        Args:
            symbol: Standard symbol (e.g., 'XAUUSD', 'EURUSD')
            
        Returns:
            Dictionary containing symbol metadata
            
        Raises:
            SymbolNotFoundError: If symbol doesn't exist
            
        Example:
            >>> metadata = SymbolMapper.get_symbol_metadata('XAUUSD')
            >>> print(metadata['instrument_type'])
            'commodity'
        """
        if not symbol:
            raise ValueError("symbol cannot be empty")
        
        mappings = cls._load_config()
        
        if symbol not in mappings:
            logger.warning(f"Symbol not found: {symbol}")
            raise SymbolNotFoundError(f"Symbol not found: {symbol}")
        
        symbol_config = mappings[symbol]
        
        # Return metadata without broker_mappings
        metadata = {k: v for k, v in symbol_config.items() if k != 'broker_mappings'}
        logger.debug(f"Retrieved metadata for {symbol}: {len(metadata)} fields")
        
        return metadata
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> bool:
        """
        Validate if a standard symbol exists in the mappings.
        
        Args:
            symbol: Standard symbol to validate
            
        Returns:
            True if symbol exists, False otherwise
        """
        if not symbol:
            return False
        
        try:
            mappings = cls._load_config()
            return symbol in mappings
        except (FileNotFoundError, yaml.YAMLError):
            logger.error("Failed to load symbol mappings for validation")
            return False
    
    @classmethod
    def get_all_standard_symbols(cls) -> List[str]:
        """
        Get list of all available standard symbols.
        
        Returns:
            List of standard symbol names
            
        Example:
            >>> symbols = SymbolMapper.get_all_standard_symbols()
            >>> print(symbols)
            ['XAUUSD', 'USDJPY', 'EURUSD', 'GBPUSD', 'WTIUSD', 'SPX500', 'NAS100', 'US30']
        """
        try:
            mappings = cls._load_config()
            symbols = list(mappings.keys())
            logger.debug(f"Retrieved {len(symbols)} standard symbols")
            return symbols
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Failed to load symbol mappings: {e}")
            return []
    
    @classmethod
    def get_broker_symbols(cls, broker_id: str) -> Dict[str, str]:
        """
        Get all symbol mappings for a specific broker.
        
        Args:
            broker_id: Broker identifier
            
        Returns:
            Dictionary mapping standard symbols to broker-specific symbols
            
        Example:
            >>> broker_symbols = SymbolMapper.get_broker_symbols('ic_markets')
            >>> print(broker_symbols['XAUUSD'])
            'XAUUSD'
        """
        if not broker_id:
            raise ValueError("broker_id cannot be empty")
        
        mappings = cls._load_config()
        broker_symbols = {}
        
        for standard_symbol, symbol_config in mappings.items():
            if 'broker_mappings' not in symbol_config:
                continue
                
            broker_mappings = symbol_config['broker_mappings']
            
            if broker_id in broker_mappings:
                broker_symbols[standard_symbol] = broker_mappings[broker_id]
        
        logger.debug(f"Retrieved {len(broker_symbols)} symbols for broker {broker_id}")
        return broker_symbols
    
    @classmethod
    def reload(cls) -> None:
        """
        Reload symbol mappings from config file.
        
        This method clears the cached mappings and forces a reload
        on the next access. Useful when the config file has been updated.
        """
        cls._symbol_mappings = None
        logger.info("Symbol mappings cache cleared, will reload on next access")
    
    @classmethod
    def get_supported_brokers(cls) -> List[str]:
        """
        Get list of all supported brokers.
        
        Returns:
            List of broker identifiers
            
        Example:
            >>> brokers = SymbolMapper.get_supported_brokers()
            >>> print(brokers)
            ['ic_markets', 'oanda', 'exness', 'pepperstone']
        """
        try:
            mappings = cls._load_config()
            brokers = set()
            
            for symbol_config in mappings.values():
                if 'broker_mappings' in symbol_config:
                    brokers.update(symbol_config['broker_mappings'].keys())
            
            broker_list = sorted(list(brokers))
            logger.debug(f"Retrieved {len(broker_list)} supported brokers")
            return broker_list
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Failed to load symbol mappings: {e}")
            return []
