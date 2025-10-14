"""
Data processors for MTQuant

This module provides data processing and feature engineering capabilities:
- Technical indicators calculation
- Feature normalization
- Domain-specific feature engineering
- Hierarchical multi-agent feature extraction
"""

from .feature_engineering import (
    add_technical_indicators,
    calculate_log_returns,
    normalize_features,
    create_sample_data,
    prepare_training_data,
    FeatureConfig,
    BaseFeatureEngineer,
    ForexFeatureEngineer,
    CommodityFeatureEngineer,
    EquityFeatureEngineer,
    FeatureEngineer
)

__all__ = [
    'add_technical_indicators',
    'calculate_log_returns',
    'normalize_features',
    'create_sample_data',
    'prepare_training_data',
    'FeatureConfig',
    'BaseFeatureEngineer',
    'ForexFeatureEngineer',
    'CommodityFeatureEngineer',
    'EquityFeatureEngineer',
    'FeatureEngineer'
]