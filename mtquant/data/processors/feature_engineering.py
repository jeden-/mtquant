"""
Feature engineering for MTQuant trading data.

Functions for adding technical indicators, calculating log returns,
and normalizing features for RL agents. Enhanced for hierarchical
multi-agent system with 8 instruments.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

from mtquant.utils.logger import get_logger


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    logger = get_logger(__name__)
    data = df.copy()
    
    try:
        # RSI (Relative Strength Index)
        data['rsi'] = _calculate_rsi(data['close'], period=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd_data = _calculate_macd(data['close'], fast=12, slow=26, signal=9)
        data['macd'] = macd_data['macd']
        data['macd_signal'] = macd_data['signal']
        data['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = _calculate_bollinger_bands(data['close'], period=20, std=2)
        data['bb_upper'] = bb_data['upper']
        data['bb_middle'] = bb_data['middle']
        data['bb_lower'] = bb_data['lower']
        data['bb_width'] = bb_data['width']
        
        # Simple Moving Averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # ATR (Average True Range)
        data['atr'] = _calculate_atr(data, period=14)
        
        logger.info(f"Added technical indicators: {len(data.columns)} total columns")
        
    except Exception as e:
        logger.error(f"Technical indicators calculation failed: {e}")
        # Return original data if calculation fails
        return df
    
    return data


def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns for price data.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        DataFrame with added log_returns column
    """
    logger = get_logger(__name__)
    data = df.copy()
    
    try:
        # Calculate log returns
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Handle NaN values
        data['log_returns'] = data['log_returns'].fillna(0)
        
        logger.info("Log returns calculated successfully")
        
    except Exception as e:
        logger.error(f"Log returns calculation failed: {e}")
        data['log_returns'] = 0
    
    return data


def normalize_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize features to 0-1 range using min-max scaling.
    
    Args:
        df: DataFrame with features
        columns: List of column names to normalize
        
    Returns:
        DataFrame with normalized features
    """
    logger = get_logger(__name__)
    data = df.copy()
    
    try:
        for column in columns:
            if column in data.columns:
                # Get non-null values for scaling
                values = data[column].dropna()
                
                if len(values) > 0:
                    min_val = values.min()
                    max_val = values.max()
                    
                    # Avoid division by zero
                    if max_val != min_val:
                        data[column] = (data[column] - min_val) / (max_val - min_val)
                    else:
                        data[column] = 0.5  # Set to middle value if no variation
                else:
                    data[column] = 0
        
        logger.info(f"Normalized {len(columns)} features")
        
    except Exception as e:
        logger.error(f"Feature normalization failed: {e}")
    
    return data


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill NaN with neutral value
        
    except Exception:
        return pd.Series([50] * len(prices), index=prices.index)


def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD indicator."""
    try:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd.fillna(0),
            'signal': signal_line.fillna(0),
            'histogram': histogram.fillna(0)
        }
        
    except Exception:
        return {
            'macd': pd.Series([0] * len(prices), index=prices.index),
            'signal': pd.Series([0] * len(prices), index=prices.index),
            'histogram': pd.Series([0] * len(prices), index=prices.index)
        }


def _calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: float = 2) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands."""
    try:
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        width = (upper - lower) / sma
        
        return {
            'upper': upper.fillna(prices),
            'middle': sma.fillna(prices),
            'lower': lower.fillna(prices),
            'width': width.fillna(0)
        }
        
    except Exception:
        return {
            'upper': prices,
            'middle': prices,
            'lower': prices,
            'width': pd.Series([0] * len(prices), index=prices.index)
        }


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=period).mean()
        
        return atr.fillna(true_range.mean())
        
    except Exception:
        return pd.Series([0] * len(df), index=df.index)


def create_sample_data(symbol: str = "XAUUSD", periods: int = 2000, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Create sample OHLCV data for testing with randomization.
    
    Args:
        symbol: Trading symbol
        periods: Number of periods to generate
        seed: Random seed (None for random data)
        
    Returns:
        DataFrame with sample OHLCV data
    """
    logger = get_logger(__name__)
    
    try:
        # Use provided seed or random
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(np.random.randint(0, 10000))
        
        # Start with base price (randomized)
        base_price = (2000.0 if symbol == "XAUUSD" else 100.0) + np.random.normal(0, 50)
        
        # Generate more realistic price movements
        returns = np.random.normal(0.0001, 0.015, periods)  # More realistic volatility
        
        # Add trend component
        trend_strength = np.random.uniform(-0.05, 0.15)
        trend = np.linspace(0, trend_strength, periods)
        
        # Add volatility clustering
        volatility_cluster = np.zeros(periods)
        for i in range(1, periods):
            volatility_cluster[i] = 0.1 * volatility_cluster[i-1] + np.random.normal(0, 0.01)
        
        # Combine components
        cumulative_returns = np.cumsum(returns + trend * 0.001 + volatility_cluster * 0.002)
        prices = base_price * np.exp(cumulative_returns)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            # Generate OHLC from close price
            volatility = abs(np.random.normal(0, 0.005)) * (1 + abs(volatility_cluster[i]))
            
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = low + (high - low) * np.random.random()
            
            # Ensure OHLC consistency
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            # Volume correlated with volatility
            volume_base = np.random.exponential(1000)
            volume_multiplier = 1 + volatility * 10 + abs(trend[i]) * 5
            volume = volume_base * volume_multiplier
            
            data.append({
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Created sample data: {len(df)} periods for {symbol} (seed: {seed})")
        
        return df
        
    except Exception as e:
        logger.error(f"Sample data creation failed: {e}")
        return pd.DataFrame()


def prepare_training_data(df: pd.DataFrame, symbol: str = "XAUUSD") -> pd.DataFrame:
    """
    Prepare data for RL training.
    
    Args:
        df: Raw OHLCV data
        symbol: Trading symbol
        
    Returns:
        Prepared DataFrame with all features
    """
    logger = get_logger(__name__)
    
    try:
        # Add technical indicators
        data = add_technical_indicators(df)
        
        # Calculate log returns
        data = calculate_log_returns(data)
        
        # Normalize features
        feature_columns = [
            'log_returns', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'sma_20', 'sma_50', 'atr'
        ]
        
        # Filter existing columns
        feature_columns = [col for col in feature_columns if col in data.columns]
        
        data = normalize_features(data, feature_columns)
        
        # Remove rows with NaN values
        data = data.dropna()
        
        logger.info(f"Training data prepared: {len(data)} rows, {len(data.columns)} columns")
        
        return data
        
    except Exception as e:
        logger.error(f"Training data preparation failed: {e}")
        return df


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Instrument configuration
    instruments: List[str]
    timeframes: List[str] = None
    
    # Feature types
    technical_indicators: bool = True
    price_features: bool = True
    volume_features: bool = True
    volatility_features: bool = True
    momentum_features: bool = True
    correlation_features: bool = True
    
    # Lookback windows
    lookback_windows: List[int] = None
    
    # Normalization
    normalization: str = 'z_score'  # 'z_score', 'min_max', 'robust'
    
    # Domain-specific features
    forex_features: bool = True
    commodity_features: bool = True
    equity_features: bool = True
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1H']
        if self.lookback_windows is None:
            self.lookback_windows = [5, 10, 20, 50]


class BaseFeatureEngineer(ABC):
    """Base class for feature engineering."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def extract_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Extract features from market data."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        pass


class ForexFeatureEngineer(BaseFeatureEngineer):
    """Feature engineer for Forex instruments."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.forex_instruments = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    def extract_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Extract Forex-specific features."""
        features = {}
        
        for instrument in self.forex_instruments:
            if instrument in data:
                df = data[instrument].copy()
                
                # Basic technical indicators
                df = add_technical_indicators(df)
                df = calculate_log_returns(df)
                
                # Forex-specific features
                df = self._add_forex_features(df)
                
                # Cross-currency features
                df = self._add_cross_currency_features(df, data)
                
                # Normalize features
                feature_columns = self._get_feature_columns(df)
                df = normalize_features(df, feature_columns)
                
                features[instrument] = df.dropna()
        
        return features
    
    def _add_forex_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Forex-specific features."""
        
        # Carry trade signals
        df['carry_signal'] = self._calculate_carry_signal(df)
        
        # Currency strength
        df['currency_strength'] = self._calculate_currency_strength(df)
        
        # Volatility regime
        df['volatility_regime'] = self._calculate_volatility_regime(df)
        
        # Trend strength
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        # Session-based features
        df['session_volatility'] = self._calculate_session_volatility(df)
        
        return df
    
    def _calculate_carry_signal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate carry trade signal."""
        # Simplified carry signal based on interest rate differential
        # In real implementation, would use actual interest rates
        returns = df['close'].pct_change()
        carry_signal = returns.rolling(20).mean() - returns.rolling(50).mean()
        return carry_signal.fillna(0)
    
    def _calculate_currency_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate currency strength indicator."""
        returns = df['close'].pct_change()
        strength = returns.rolling(20).sum() / returns.rolling(20).std()
        return strength.fillna(0)
    
    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility regime."""
        returns = df['close'].pct_change()
        short_vol = returns.rolling(10).std()
        long_vol = returns.rolling(50).std()
        regime = short_vol / long_vol
        return regime.fillna(1)
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength."""
        returns = df['close'].pct_change()
        trend = returns.rolling(20).mean() / returns.rolling(20).std()
        return trend.fillna(0)
    
    def _calculate_session_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate session-based volatility."""
        # Simplified session volatility
        returns = df['close'].pct_change()
        session_vol = returns.rolling(4).std()  # 4-hour sessions
        return session_vol.fillna(0)
    
    def _add_cross_currency_features(self, df: pd.DataFrame, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add cross-currency correlation features."""
        
        # Calculate correlations with other Forex pairs
        correlations = {}
        for other_instrument in self.forex_instruments:
            if other_instrument in all_data and other_instrument != df.name:
                other_returns = all_data[other_instrument]['close'].pct_change()
                corr = df['close'].pct_change().rolling(20).corr(other_returns)
                correlations[f'corr_{other_instrument}'] = corr
        
        # Add correlation features
        for name, corr in correlations.items():
            df[name] = corr.fillna(0)
        
        return df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns."""
        feature_columns = [
            'log_returns', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'sma_20', 'sma_50', 'atr', 'carry_signal', 'currency_strength',
            'volatility_regime', 'trend_strength', 'session_volatility'
        ]
        
        # Add correlation columns
        corr_columns = [col for col in df.columns if col.startswith('corr_')]
        feature_columns.extend(corr_columns)
        
        return [col for col in feature_columns if col in df.columns]
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            'log_returns', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'sma_20', 'sma_50', 'atr', 'carry_signal', 'currency_strength',
            'volatility_regime', 'trend_strength', 'session_volatility'
        ]


class CommodityFeatureEngineer(BaseFeatureEngineer):
    """Feature engineer for Commodity instruments."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.commodity_instruments = ['XAUUSD', 'WTIUSD']
    
    def extract_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Extract Commodity-specific features."""
        features = {}
        
        for instrument in self.commodity_instruments:
            if instrument in data:
                df = data[instrument].copy()
                
                # Basic technical indicators
                df = add_technical_indicators(df)
                df = calculate_log_returns(df)
                
                # Commodity-specific features
                df = self._add_commodity_features(df)
                
                # Cross-commodity features
                df = self._add_cross_commodity_features(df, data)
                
                # Normalize features
                feature_columns = self._get_feature_columns(df)
                df = normalize_features(df, feature_columns)
                
                features[instrument] = df.dropna()
        
        return features
    
    def _add_commodity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Commodity-specific features."""
        
        # Safe haven demand
        df['safe_haven_demand'] = self._calculate_safe_haven_demand(df)
        
        # Inflation hedge
        df['inflation_hedge'] = self._calculate_inflation_hedge(df)
        
        # Supply/demand indicators
        df['supply_demand'] = self._calculate_supply_demand(df)
        
        # Seasonal patterns
        df['seasonal_pattern'] = self._calculate_seasonal_pattern(df)
        
        # Volatility clustering
        df['volatility_clustering'] = self._calculate_volatility_clustering(df)
        
        return df
    
    def _calculate_safe_haven_demand(self, df: pd.DataFrame) -> pd.Series:
        """Calculate safe haven demand indicator."""
        returns = df['close'].pct_change()
        # Safe haven demand increases during market stress
        stress_indicator = -returns.rolling(20).mean() / returns.rolling(20).std()
        return stress_indicator.fillna(0)
    
    def _calculate_inflation_hedge(self, df: pd.DataFrame) -> pd.Series:
        """Calculate inflation hedge indicator."""
        returns = df['close'].pct_change()
        # Inflation hedge based on long-term trend
        inflation_hedge = returns.rolling(50).mean()
        return inflation_hedge.fillna(0)
    
    def _calculate_supply_demand(self, df: pd.DataFrame) -> pd.Series:
        """Calculate supply/demand indicator."""
        # Based on volume and price relationship
        volume = df['volume']
        price_change = df['close'].pct_change()
        supply_demand = (volume * price_change).rolling(20).mean()
        return supply_demand.fillna(0)
    
    def _calculate_seasonal_pattern(self, df: pd.DataFrame) -> pd.Series:
        """Calculate seasonal pattern."""
        # Simplified seasonal pattern based on day of year
        df['day_of_year'] = df.index.dayofyear
        seasonal = np.sin(2 * np.pi * df['day_of_year'] / 365)
        return seasonal
    
    def _calculate_volatility_clustering(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility clustering."""
        returns = df['close'].pct_change()
        vol_clustering = returns.rolling(5).std() / returns.rolling(20).std()
        return vol_clustering.fillna(1)
    
    def _add_cross_commodity_features(self, df: pd.DataFrame, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add cross-commodity correlation features."""
        
        # Calculate correlations with other commodities
        correlations = {}
        for other_instrument in self.commodity_instruments:
            if other_instrument in all_data and other_instrument != df.name:
                other_returns = all_data[other_instrument]['close'].pct_change()
                corr = df['close'].pct_change().rolling(20).corr(other_returns)
                correlations[f'corr_{other_instrument}'] = corr
        
        # Add correlation features
        for name, corr in correlations.items():
            df[name] = corr.fillna(0)
        
        return df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns."""
        feature_columns = [
            'log_returns', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'sma_20', 'sma_50', 'atr', 'safe_haven_demand', 'inflation_hedge',
            'supply_demand', 'seasonal_pattern', 'volatility_clustering'
        ]
        
        # Add correlation columns
        corr_columns = [col for col in df.columns if col.startswith('corr_')]
        feature_columns.extend(corr_columns)
        
        return [col for col in feature_columns if col in df.columns]
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            'log_returns', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'sma_20', 'sma_50', 'atr', 'safe_haven_demand', 'inflation_hedge',
            'supply_demand', 'seasonal_pattern', 'volatility_clustering'
        ]


class EquityFeatureEngineer(BaseFeatureEngineer):
    """Feature engineer for Equity instruments."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.equity_instruments = ['SPX500', 'NAS100', 'US30']
    
    def extract_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Extract Equity-specific features."""
        features = {}
        
        for instrument in self.equity_instruments:
            if instrument in data:
                df = data[instrument].copy()
                
                # Basic technical indicators
                df = add_technical_indicators(df)
                df = calculate_log_returns(df)
                
                # Equity-specific features
                df = self._add_equity_features(df)
                
                # Cross-equity features
                df = self._add_cross_equity_features(df, data)
                
                # Normalize features
                feature_columns = self._get_feature_columns(df)
                df = normalize_features(df, feature_columns)
                
                features[instrument] = df.dropna()
        
        return features
    
    def _add_equity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Equity-specific features."""
        
        # Fear/Greed index
        df['fear_greed_index'] = self._calculate_fear_greed_index(df)
        
        # Sector rotation
        df['sector_rotation'] = self._calculate_sector_rotation(df)
        
        # Market breadth
        df['market_breadth'] = self._calculate_market_breadth(df)
        
        # Earnings momentum
        df['earnings_momentum'] = self._calculate_earnings_momentum(df)
        
        # Risk-on/Risk-off
        df['risk_on_off'] = self._calculate_risk_on_off(df)
        
        return df
    
    def _calculate_fear_greed_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate fear/greed index."""
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std()
        fear_greed = -volatility / returns.rolling(20).mean()
        return fear_greed.fillna(0)
    
    def _calculate_sector_rotation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate sector rotation indicator."""
        returns = df['close'].pct_change()
        # Sector rotation based on momentum
        sector_rotation = returns.rolling(10).mean() - returns.rolling(30).mean()
        return sector_rotation.fillna(0)
    
    def _calculate_market_breadth(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market breadth indicator."""
        returns = df['close'].pct_change()
        # Market breadth based on consistency of returns
        breadth = (returns > 0).rolling(20).mean()
        return breadth.fillna(0.5)
    
    def _calculate_earnings_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate earnings momentum."""
        returns = df['close'].pct_change()
        # Earnings momentum based on price momentum
        earnings_momentum = returns.rolling(20).sum()
        return earnings_momentum.fillna(0)
    
    def _calculate_risk_on_off(self, df: pd.DataFrame) -> pd.Series:
        """Calculate risk-on/risk-off indicator."""
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std()
        risk_on_off = returns.rolling(20).mean() / volatility
        return risk_on_off.fillna(0)
    
    def _add_cross_equity_features(self, df: pd.DataFrame, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add cross-equity correlation features."""
        
        # Calculate correlations with other equity indices
        correlations = {}
        for other_instrument in self.equity_instruments:
            if other_instrument in all_data and other_instrument != df.name:
                other_returns = all_data[other_instrument]['close'].pct_change()
                corr = df['close'].pct_change().rolling(20).corr(other_returns)
                correlations[f'corr_{other_instrument}'] = corr
        
        # Add correlation features
        for name, corr in correlations.items():
            df[name] = corr.fillna(0)
        
        return df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns."""
        feature_columns = [
            'log_returns', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'sma_20', 'sma_50', 'atr', 'fear_greed_index', 'sector_rotation',
            'market_breadth', 'earnings_momentum', 'risk_on_off'
        ]
        
        # Add correlation columns
        corr_columns = [col for col in df.columns if col.startswith('corr_')]
        feature_columns.extend(corr_columns)
        
        return [col for col in feature_columns if col in df.columns]
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            'log_returns', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'sma_20', 'sma_50', 'atr', 'fear_greed_index', 'sector_rotation',
            'market_breadth', 'earnings_momentum', 'risk_on_off'
        ]


class FeatureEngineer:
    """
    Main feature engineer for hierarchical multi-agent system.
    
    Coordinates domain-specific feature engineers and provides
    unified interface for feature extraction.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize domain-specific engineers
        self.forex_engineer = ForexFeatureEngineer(config)
        self.commodity_engineer = CommodityFeatureEngineer(config)
        self.equity_engineer = EquityFeatureEngineer(config)
    
    def extract_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Extract features for all instruments."""
        
        all_features = {}
        
        # Extract Forex features
        forex_features = self.forex_engineer.extract_features(data)
        all_features.update(forex_features)
        
        # Extract Commodity features
        commodity_features = self.commodity_engineer.extract_features(data)
        all_features.update(commodity_features)
        
        # Extract Equity features
        equity_features = self.equity_engineer.extract_features(data)
        all_features.update(equity_features)
        
        self.logger.info(f"Extracted features for {len(all_features)} instruments")
        
        return all_features
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get feature names for each domain."""
        return {
            'forex': self.forex_engineer.get_feature_names(),
            'commodity': self.commodity_engineer.get_feature_names(),
            'equity': self.equity_engineer.get_feature_names()
        }
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get feature dimensions for each domain."""
        feature_names = self.get_feature_names()
        return {
            domain: len(features) for domain, features in feature_names.items()
        }
