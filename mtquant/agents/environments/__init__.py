"""
Trading Environments for MTQuant

This module provides various trading environments for reinforcement learning:
- Base trading environment
- Hierarchical multi-agent environments
- Parallel environment wrappers
- Curriculum learning support
"""

from .base_trading_env import MTQuantTradingEnv
from .hierarchical_env import BaseHierarchicalEnv, EnvironmentConfig
from .meta_controller_env import MetaControllerEnv
from .specialist_env import SpecialistEnv
from .parallel_env import ParallelHierarchicalWrapper, CurriculumLearningWrapper

__all__ = [
    'MTQuantTradingEnv',
    'BaseHierarchicalEnv',
    'EnvironmentConfig',
    'MetaControllerEnv',
    'SpecialistEnv',
    'ParallelHierarchicalWrapper',
    'CurriculumLearningWrapper'
]