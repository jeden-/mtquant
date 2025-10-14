"""
Feature engineering for MTQuant trading data.

Functions for adding technical indicators, calculating log returns,
and normalizing features for RL agents.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings

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
