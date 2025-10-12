"""
Data Management Module

Comprehensive data handling for market data, feature engineering, and storage.
Supports multiple data sources and provides unified interfaces for RL agents.

Components:
- fetchers/: Data source integrations (brokers, APIs, files)
- processors/: Feature engineering, technical indicators, normalization
- storage/: Database clients (QuestDB, PostgreSQL, Redis)

Data Flow:
1. Raw market data → fetchers
2. Feature engineering → processors  
3. Storage and retrieval → storage
4. Agent consumption → unified interface

Supports real-time and historical data with proper caching and
optimization for high-frequency trading requirements.
"""

__version__ = "0.1.0"

# Data management classes will be imported here when implemented
__all__ = []
