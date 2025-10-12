"""
Data Processors

Feature engineering and data processing utilities for RL agents.
Transforms raw market data into meaningful features for trading decisions.

Key Features:
- Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- Statistical features (rolling means, volatility, correlations)
- Time-series features (lags, differences, seasonality)
- Risk metrics (VaR, Sharpe ratio, drawdown)
- Data normalization and scaling

All processors maintain stationarity requirements for RL training
and include proper handling of missing data and edge cases.
"""

__version__ = "0.1.0"

# Processor classes will be imported here when implemented
__all__ = []
