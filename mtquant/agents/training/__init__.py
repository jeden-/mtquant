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

__all__ = [
    'train_ppo_agent',
    'evaluate_agent',
    'SpecialistTrainer',
    'Phase1Trainer'
]