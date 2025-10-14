"""
Training modules for MTQuant

This module provides training capabilities for:
- Individual specialist training (Phase 1)
- Meta-controller training (Phase 2)
- Joint fine-tuning (Phase 3)
- PPO training with parallel environments
"""

from .train_ppo import train_ppo_agent, evaluate_agent
from .specialist_trainer import SpecialistTrainer
from .phase1_trainer import Phase1Trainer
from .phase2_trainer import Phase2Trainer
from .portfolio_reward import PortfolioRewardFunction, RewardConfig
from .gradient_coordination import GradientCoordinationSystem, GradientCoordinationConfig
from .curriculum_learning import AdvancedCurriculumLearning, CurriculumConfig
from .model_checkpointing import ModelCheckpointingSystem, CheckpointConfig
from .training_monitoring import TrainingMonitoringDashboard, MonitoringConfig

__all__ = [
    'train_ppo_agent',
    'evaluate_agent',
    'SpecialistTrainer',
    'Phase1Trainer',
    'Phase2Trainer',
    'PortfolioRewardFunction',
    'RewardConfig',
    'GradientCoordinationSystem',
    'GradientCoordinationConfig',
    'AdvancedCurriculumLearning',
    'CurriculumConfig',
    'ModelCheckpointingSystem',
    'CheckpointConfig',
    'TrainingMonitoringDashboard',
    'MonitoringConfig'
]