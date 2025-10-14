"""
Data processing and feature engineering for MTQuant.
"""

from .processors.feature_engineering import (
    add_technical_indicators,
    calculate_log_returns,
    normalize_features
)

__all__ = [
    'add_technical_indicators',
    'calculate_log_returns',
    'normalize_features'
]