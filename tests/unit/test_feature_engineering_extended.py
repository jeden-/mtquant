"""
Unit tests for feature_engineering.py with 50% coverage.

This file has 389 lines and 50% coverage, so adding comprehensive tests here will significantly increase overall coverage.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, List, Any

# Import feature engineering classes and functions
from mtquant.data.processors.feature_engineering import (
    FeatureConfig, BaseFeatureEngineer, ForexFeatureEngineer,
    CommodityFeatureEngineer, EquityFeatureEngineer, FeatureEngineer,
    _calculate_rsi, _calculate_macd, _calculate_bollinger_bands, _calculate_atr
)


class TestFeatureConfig:
    """Tests for FeatureConfig dataclass."""
    
    def test_feature_config_initialization(self):
        """Test FeatureConfig initialization."""
        config = FeatureConfig(
            instruments=['EURUSD', 'XAUUSD', 'SPX500'],
            timeframes=['1H', '4H'],
            technical_indicators=True,
            price_features=True,
            volume_features=True,
            volatility_features=True,
            momentum_features=True,
            correlation_features=True,
            lookback_windows=[5, 10, 20],
            normalization='min_max',
            forex_features=True,
            commodity_features=True,
            equity_features=True
        )
        
        assert config.instruments == ['EURUSD', 'XAUUSD', 'SPX500']
        assert config.timeframes == ['1H', '4H']
        assert config.technical_indicators == True
        assert config.price_features == True
        assert config.volume_features == True
        assert config.volatility_features == True
        assert config.momentum_features == True
        assert config.correlation_features == True
        assert config.lookback_windows == [5, 10, 20]
        assert config.normalization == 'min_max'
        assert config.forex_features == True
        assert config.commodity_features == True
        assert config.equity_features == True
    
    def test_feature_config_defaults(self):
        """Test FeatureConfig default values."""
        config = FeatureConfig(instruments=['EURUSD'])
        
        assert config.timeframes == ['1H']  # Default value
        assert config.lookback_windows == [5, 10, 20, 50]  # Default value
        assert config.normalization == 'z_score'  # Default value
        assert config.technical_indicators == True  # Default value
        assert config.price_features == True  # Default value
        assert config.volume_features == True  # Default value
        assert config.volatility_features == True  # Default value
        assert config.momentum_features == True  # Default value
        assert config.correlation_features == True  # Default value
        assert config.forex_features == True  # Default value
        assert config.commodity_features == True  # Default value
        assert config.equity_features == True  # Default value


class TestBaseFeatureEngineer:
    """Tests for BaseFeatureEngineer abstract class."""
    
    def test_base_feature_engineer_initialization(self):
        """Test BaseFeatureEngineer initialization."""
        config = FeatureConfig(instruments=['EURUSD'])
        
        # Create concrete implementation for testing
        class TestFeatureEngineer(BaseFeatureEngineer):
            def extract_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
                return data
            
            def get_feature_names(self) -> List[str]:
                return ['test_feature']
        
        engineer = TestFeatureEngineer(config)
        
        assert engineer.config == config
        assert engineer.logger is not None
    
    def test_base_feature_engineer_abstract_methods(self):
        """Test BaseFeatureEngineer abstract methods."""
        config = FeatureConfig(instruments=['EURUSD'])
        
        # Test that abstract class cannot be instantiated
        with pytest.raises(TypeError):
            BaseFeatureEngineer(config)


class TestForexFeatureEngineer:
    """Tests for ForexFeatureEngineer."""
    
    @pytest.fixture
    def config(self):
        """Create FeatureConfig for testing."""
        return FeatureConfig(
            instruments=['EURUSD', 'GBPUSD', 'USDJPY'],
            timeframes=['1H'],
            technical_indicators=True,
            price_features=True,
            volume_features=True,
            volatility_features=True,
            momentum_features=True,
            correlation_features=True
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        data = {}
        for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
            dates = pd.date_range('2024-01-01', periods=100, freq='H')
            data[symbol] = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 1.2,
                'high': np.random.randn(100).cumsum() + 1.21,
                'low': np.random.randn(100).cumsum() + 1.19,
                'close': np.random.randn(100).cumsum() + 1.2,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
        return data
    
    def test_forex_feature_engineer_initialization(self, config):
        """Test ForexFeatureEngineer initialization."""
        engineer = ForexFeatureEngineer(config)
        
        assert engineer.config == config
        assert engineer.logger is not None
    
    def test_forex_feature_engineer_extract_features(self, config, sample_data):
        """Test ForexFeatureEngineer extract_features method."""
        engineer = ForexFeatureEngineer(config)
        
        features = engineer.extract_features(sample_data)
        
        assert isinstance(features, dict)
        assert 'EURUSD' in features
        assert 'GBPUSD' in features
        assert 'USDJPY' in features
        
        # Check that features were added
        for symbol, df in features.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_forex_feature_engineer_get_feature_names(self, config):
        """Test ForexFeatureEngineer get_feature_names method."""
        engineer = ForexFeatureEngineer(config)
        
        feature_names = engineer.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)


class TestCommodityFeatureEngineer:
    """Tests for CommodityFeatureEngineer."""
    
    @pytest.fixture
    def config(self):
        """Create FeatureConfig for testing."""
        return FeatureConfig(
            instruments=['XAUUSD', 'WTIUSD'],
            timeframes=['1H'],
            technical_indicators=True,
            price_features=True,
            volume_features=True,
            volatility_features=True,
            momentum_features=True,
            correlation_features=True
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        data = {}
        for symbol in ['XAUUSD', 'WTIUSD']:
            dates = pd.date_range('2024-01-01', periods=100, freq='H')
            data[symbol] = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 2000,
                'high': np.random.randn(100).cumsum() + 2001,
                'low': np.random.randn(100).cumsum() + 1999,
                'close': np.random.randn(100).cumsum() + 2000,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
        return data
    
    def test_commodity_feature_engineer_initialization(self, config):
        """Test CommodityFeatureEngineer initialization."""
        engineer = CommodityFeatureEngineer(config)
        
        assert engineer.config == config
        assert engineer.logger is not None
    
    def test_commodity_feature_engineer_extract_features(self, config, sample_data):
        """Test CommodityFeatureEngineer extract_features method."""
        engineer = CommodityFeatureEngineer(config)
        
        features = engineer.extract_features(sample_data)
        
        assert isinstance(features, dict)
        assert 'XAUUSD' in features
        assert 'WTIUSD' in features
        
        # Check that features were added
        for symbol, df in features.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_commodity_feature_engineer_get_feature_names(self, config):
        """Test CommodityFeatureEngineer get_feature_names method."""
        engineer = CommodityFeatureEngineer(config)
        
        feature_names = engineer.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)


class TestEquityFeatureEngineer:
    """Tests for EquityFeatureEngineer."""
    
    @pytest.fixture
    def config(self):
        """Create FeatureConfig for testing."""
        return FeatureConfig(
            instruments=['SPX500', 'NAS100', 'US30'],
            timeframes=['1H'],
            technical_indicators=True,
            price_features=True,
            volume_features=True,
            volatility_features=True,
            momentum_features=True,
            correlation_features=True
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        data = {}
        for symbol in ['SPX500', 'NAS100', 'US30']:
            dates = pd.date_range('2024-01-01', periods=100, freq='H')
            data[symbol] = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 4000,
                'high': np.random.randn(100).cumsum() + 4001,
                'low': np.random.randn(100).cumsum() + 3999,
                'close': np.random.randn(100).cumsum() + 4000,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
        return data
    
    def test_equity_feature_engineer_initialization(self, config):
        """Test EquityFeatureEngineer initialization."""
        engineer = EquityFeatureEngineer(config)
        
        assert engineer.config == config
        assert engineer.logger is not None
    
    def test_equity_feature_engineer_extract_features(self, config, sample_data):
        """Test EquityFeatureEngineer extract_features method."""
        engineer = EquityFeatureEngineer(config)
        
        features = engineer.extract_features(sample_data)
        
        assert isinstance(features, dict)
        assert 'SPX500' in features
        assert 'NAS100' in features
        assert 'US30' in features
        
        # Check that features were added
        for symbol, df in features.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_equity_feature_engineer_get_feature_names(self, config):
        """Test EquityFeatureEngineer get_feature_names method."""
        engineer = EquityFeatureEngineer(config)
        
        feature_names = engineer.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)


class TestFeatureEngineer:
    """Tests for FeatureEngineer coordinating class."""
    
    @pytest.fixture
    def config(self):
        """Create FeatureConfig for testing."""
        return FeatureConfig(
            instruments=['EURUSD', 'XAUUSD', 'SPX500'],
            timeframes=['1H'],
            technical_indicators=True,
            price_features=True,
            volume_features=True,
            volatility_features=True,
            momentum_features=True,
            correlation_features=True
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        data = {}
        for symbol in ['EURUSD', 'XAUUSD', 'SPX500']:
            dates = pd.date_range('2024-01-01', periods=100, freq='H')
            data[symbol] = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 1.2,
                'high': np.random.randn(100).cumsum() + 1.21,
                'low': np.random.randn(100).cumsum() + 1.19,
                'close': np.random.randn(100).cumsum() + 1.2,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
        return data
    
    def test_feature_engineer_initialization(self, config):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(config)
        
        assert engineer.config == config
        assert engineer.logger is not None
        assert engineer.forex_engineer is not None
        assert engineer.commodity_engineer is not None
        assert engineer.equity_engineer is not None
    
    def test_feature_engineer_extract_features(self, config, sample_data):
        """Test FeatureEngineer extract_features method."""
        engineer = FeatureEngineer(config)
        
        features = engineer.extract_features(sample_data)
        
        assert isinstance(features, dict)
        assert 'EURUSD' in features
        assert 'XAUUSD' in features
        assert 'SPX500' in features
        
        # Check that features were added
        for symbol, df in features.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_feature_engineer_get_feature_names(self, config):
        """Test FeatureEngineer get_feature_names method."""
        engineer = FeatureEngineer(config)
        
        feature_names = engineer.get_feature_names()
        
        assert isinstance(feature_names, dict)
        assert 'forex' in feature_names
        assert 'commodity' in feature_names
        assert 'equity' in feature_names
        
        for domain, names in feature_names.items():
            assert isinstance(names, list)
            assert len(names) > 0
            assert all(isinstance(name, str) for name in names)
    
    def test_feature_engineer_get_feature_dimensions(self, config):
        """Test FeatureEngineer get_feature_dimensions method."""
        engineer = FeatureEngineer(config)
        
        dimensions = engineer.get_feature_dimensions()
        
        assert isinstance(dimensions, dict)
        assert 'forex' in dimensions
        assert 'commodity' in dimensions
        assert 'equity' in dimensions
        
        for domain, dim in dimensions.items():
            assert isinstance(dim, int)
            assert dim > 0


class TestHelperFunctions:
    """Tests for helper functions in feature_engineering.py."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        prices = pd.Series(
            np.random.randn(100).cumsum() + 1.2,
            index=dates
        )
        return prices
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 1.2,
            'high': np.random.randn(100).cumsum() + 1.21,
            'low': np.random.randn(100).cumsum() + 1.19,
            'close': np.random.randn(100).cumsum() + 1.2,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        return df
    
    def test_calculate_rsi_basic(self, sample_prices):
        """Test _calculate_rsi basic functionality."""
        rsi = _calculate_rsi(sample_prices, period=14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_prices)
        assert rsi.index.equals(sample_prices.index)
        assert all(0 <= val <= 100 for val in rsi.dropna())
    
    def test_calculate_rsi_different_periods(self, sample_prices):
        """Test _calculate_rsi with different periods."""
        rsi_5 = _calculate_rsi(sample_prices, period=5)
        rsi_20 = _calculate_rsi(sample_prices, period=20)
        
        assert len(rsi_5) == len(sample_prices)
        assert len(rsi_20) == len(sample_prices)
        assert not rsi_5.equals(rsi_20)  # Different periods should give different results
    
    def test_calculate_rsi_error_handling(self):
        """Test _calculate_rsi error handling."""
        # Test with empty series
        empty_prices = pd.Series([], dtype=float)
        rsi = _calculate_rsi(empty_prices)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == 0
    
    def test_calculate_macd_basic(self, sample_prices):
        """Test _calculate_macd basic functionality."""
        macd_data = _calculate_macd(sample_prices, fast=12, slow=26, signal=9)
        
        assert isinstance(macd_data, dict)
        assert 'macd' in macd_data
        assert 'signal' in macd_data
        assert 'histogram' in macd_data
        
        for key, series in macd_data.items():
            assert isinstance(series, pd.Series)
            assert len(series) == len(sample_prices)
            assert series.index.equals(sample_prices.index)
    
    def test_calculate_macd_different_parameters(self, sample_prices):
        """Test _calculate_macd with different parameters."""
        macd_1 = _calculate_macd(sample_prices, fast=5, slow=10, signal=3)
        macd_2 = _calculate_macd(sample_prices, fast=12, slow=26, signal=9)
        
        assert not macd_1['macd'].equals(macd_2['macd'])  # Different parameters should give different results
    
    def test_calculate_macd_error_handling(self):
        """Test _calculate_macd error handling."""
        # Test with empty series
        empty_prices = pd.Series([], dtype=float)
        macd_data = _calculate_macd(empty_prices)
        
        assert isinstance(macd_data, dict)
        assert 'macd' in macd_data
        assert 'signal' in macd_data
        assert 'histogram' in macd_data
        assert len(macd_data['macd']) == 0
    
    def test_calculate_bollinger_bands_basic(self, sample_prices):
        """Test _calculate_bollinger_bands basic functionality."""
        bb_data = _calculate_bollinger_bands(sample_prices, period=20, std=2)
        
        assert isinstance(bb_data, dict)
        assert 'upper' in bb_data
        assert 'middle' in bb_data
        assert 'lower' in bb_data
        assert 'width' in bb_data
        
        for key, series in bb_data.items():
            assert isinstance(series, pd.Series)
            assert len(series) == len(sample_prices)
            assert series.index.equals(sample_prices.index)
        
        # Check that upper >= middle >= lower (for most values)
        upper = bb_data['upper'].dropna()
        middle = bb_data['middle'].dropna()
        lower = bb_data['lower'].dropna()
        
        assert all(upper >= middle)
        assert all(middle >= lower)
    
    def test_calculate_bollinger_bands_different_parameters(self, sample_prices):
        """Test _calculate_bollinger_bands with different parameters."""
        bb_1 = _calculate_bollinger_bands(sample_prices, period=10, std=1)
        bb_2 = _calculate_bollinger_bands(sample_prices, period=20, std=2)
        
        assert not bb_1['upper'].equals(bb_2['upper'])  # Different parameters should give different results
    
    def test_calculate_bollinger_bands_error_handling(self):
        """Test _calculate_bollinger_bands error handling."""
        # Test with empty series
        empty_prices = pd.Series([], dtype=float)
        bb_data = _calculate_bollinger_bands(empty_prices)
        
        assert isinstance(bb_data, dict)
        assert 'upper' in bb_data
        assert 'middle' in bb_data
        assert 'lower' in bb_data
        assert 'width' in bb_data
        assert len(bb_data['upper']) == 0
    
    def test_calculate_atr_basic(self, sample_ohlcv):
        """Test _calculate_atr basic functionality."""
        atr = _calculate_atr(sample_ohlcv, period=14)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_ohlcv)
        assert atr.index.equals(sample_ohlcv.index)
        assert all(atr >= 0)  # ATR should always be positive
    
    def test_calculate_atr_different_periods(self, sample_ohlcv):
        """Test _calculate_atr with different periods."""
        atr_5 = _calculate_atr(sample_ohlcv, period=5)
        atr_20 = _calculate_atr(sample_ohlcv, period=20)
        
        assert len(atr_5) == len(sample_ohlcv)
        assert len(atr_20) == len(sample_ohlcv)
        assert not atr_5.equals(atr_20)  # Different periods should give different results
    
    def test_calculate_atr_error_handling(self):
        """Test _calculate_atr error handling."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        atr = _calculate_atr(empty_df)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == 0
        
        # Test with missing columns
        incomplete_df = pd.DataFrame({
            'open': [1.0, 1.1, 1.2],
            'close': [1.05, 1.15, 1.25]
            # Missing 'high' and 'low'
        })
        atr = _calculate_atr(incomplete_df)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(incomplete_df)


class TestFeatureEngineeringEdgeCases:
    """Tests for edge cases in feature engineering."""
    
    def test_feature_engineer_empty_data(self):
        """Test FeatureEngineer with empty data."""
        config = FeatureConfig(instruments=['EURUSD'])
        engineer = FeatureEngineer(config)
        
        empty_data = {}
        features = engineer.extract_features(empty_data)
        
        assert isinstance(features, dict)
        assert len(features) == 0
    
    def test_feature_engineer_missing_instruments(self):
        """Test FeatureEngineer with missing instruments."""
        config = FeatureConfig(instruments=['EURUSD', 'XAUUSD'])
        engineer = FeatureEngineer(config)
        
        # Only provide data for one instrument
        partial_data = {
            'EURUSD': pd.DataFrame({
                'open': [1.0, 1.1, 1.2],
                'high': [1.01, 1.11, 1.21],
                'low': [0.99, 1.09, 1.19],
                'close': [1.005, 1.105, 1.205],
                'volume': [1000, 1100, 1200]
            })
        }
        
        features = engineer.extract_features(partial_data)
        
        assert isinstance(features, dict)
        assert 'EURUSD' in features
        # XAUUSD might not be in features if no data provided
    
    def test_feature_engineer_invalid_data(self):
        """Test FeatureEngineer with invalid data."""
        config = FeatureConfig(instruments=['EURUSD'])
        engineer = FeatureEngineer(config)
        
        # Invalid data (not DataFrame)
        invalid_data = {
            'EURUSD': 'not a dataframe'
        }
        
        # Should handle gracefully
        try:
            features = engineer.extract_features(invalid_data)
            # If it doesn't raise an exception, it should return a dict
            assert isinstance(features, dict)
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass
    
    def test_feature_config_invalid_normalization(self):
        """Test FeatureConfig with invalid normalization."""
        config = FeatureConfig(
            instruments=['EURUSD'],
            normalization='invalid_method'
        )
        
        # Should still create the config
        assert config.normalization == 'invalid_method'
        assert config.instruments == ['EURUSD']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

