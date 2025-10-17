"""
Parallel Environment Wrapper for Hierarchical Multi-Agent Training

This module provides wrappers for parallel training of multiple environments,
supporting both individual specialist training and meta-controller coordination.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics, NormalizeReward

from .hierarchical_env import BaseHierarchicalEnv, EnvironmentConfig
from .meta_controller_env import MetaControllerEnv
from .specialist_env import SpecialistEnv
from ..hierarchical.meta_controller import MetaController
from ..hierarchical.base_specialist import BaseSpecialist
from ..hierarchical.communication import CommunicationHub
from ...risk_management.portfolio_risk_manager import PortfolioRiskManager


class ParallelHierarchicalWrapper:
    """
    Wrapper for parallel training of hierarchical multi-agent system.
    
    Supports:
    - Parallel specialist training (Phase 1)
    - Parallel meta-controller training (Phase 2)
    - Joint training coordination (Phase 3)
    """
    
    def __init__(
        self,
        env_configs: Dict[str, EnvironmentConfig],
        market_data: Dict[str, np.ndarray],
        meta_controller: MetaController,
        specialists: Dict[str, BaseSpecialist],
        communication_hub: CommunicationHub,
        portfolio_risk_manager: PortfolioRiskManager,
        n_envs: int = 8,
        async_envs: bool = True
    ):
        self.env_configs = env_configs
        self.market_data = market_data
        self.meta_controller = meta_controller
        self.specialists = specialists
        self.communication_hub = communication_hub
        self.portfolio_risk_manager = portfolio_risk_manager
        self.n_envs = n_envs
        self.async_envs = async_envs
        
        # Environment pools
        self.specialist_envs: Dict[str, gym.Env] = {}
        self.meta_controller_envs: Optional[gym.Env] = None
        
        # Training phase
        self.current_phase = 1  # 1: Individual, 2: Meta, 3: Joint
        
    def create_specialist_envs(self) -> Dict[str, gym.Env]:
        """Create parallel environments for each specialist."""
        
        specialist_envs = {}
        
        for specialist_name, specialist in self.specialists.items():
            # Create environment factory
            def make_env(env_config: EnvironmentConfig, specialist: BaseSpecialist, 
                        market_data: Dict[str, np.ndarray], comm_hub: CommunicationHub,
                        risk_manager: PortfolioRiskManager):
                def _init():
                    env = SpecialistEnv(
                        config=env_config,
                        market_data=market_data,
                        specialist=specialist,
                        communication_hub=comm_hub,
                        portfolio_risk_manager=risk_manager
                    )
                    # Add wrappers
                    env = RecordEpisodeStatistics(env)
                    env = NormalizeReward(env)
                    return env
                return _init
            
            # Create parallel environments
            env_factories = [
                make_env(
                    self.env_configs[specialist_name],
                    specialist,
                    self.market_data,
                    self.communication_hub,
                    self.portfolio_risk_manager
                ) for _ in range(self.n_envs)
            ]
            
            # Choose vectorized environment type
            if self.async_envs:
                specialist_envs[specialist_name] = SyncVectorEnv(env_factories)  # Use SyncVectorEnv to avoid reward array issues
            else:
                specialist_envs[specialist_name] = SyncVectorEnv(env_factories)
        
        self.specialist_envs = specialist_envs
        return specialist_envs
    
    def create_meta_controller_envs(self) -> gym.Env:
        """Create parallel environments for meta-controller training."""
        
        def make_env():
            def _init():
                env = MetaControllerEnv(
                    config=self.env_configs['meta_controller'],
                    market_data=self.market_data,
                    meta_controller=self.meta_controller,
                    specialists=self.specialists,
                    communication_hub=self.communication_hub,
                    portfolio_risk_manager=self.portfolio_risk_manager
                )
                # Add wrappers
                env = RecordEpisodeStatistics(env)
                env = NormalizeReward(env)
                return env
            return _init
        
        # Create parallel environments
        env_factories = [make_env() for _ in range(self.n_envs)]
        
        # Choose vectorized environment type
        if self.async_envs:
            self.meta_controller_envs = AsyncVectorEnv(env_factories)
        else:
            self.meta_controller_envs = SyncVectorEnv(env_factories)
        
        return self.meta_controller_envs
    
    def reset_specialist_envs(self) -> Dict[str, Tuple[np.ndarray, Dict]]:
        """Reset all specialist environments."""
        if not self.specialist_envs:
            self.create_specialist_envs()
        
        reset_results = {}
        for specialist_name, env in self.specialist_envs.items():
            reset_results[specialist_name] = env.reset()
        
        return reset_results
    
    def reset_meta_controller_envs(self) -> Tuple[np.ndarray, Dict]:
        """Reset meta-controller environments."""
        if self.meta_controller_envs is None:
            self.create_meta_controller_envs()
        
        return self.meta_controller_envs.reset()
    
    def step_specialist_envs(self, actions: Dict[str, np.ndarray]) -> Dict[str, Tuple]:
        """Step all specialist environments."""
        results = {}
        
        for specialist_name, env in self.specialist_envs.items():
            if specialist_name in actions:
                results[specialist_name] = env.step(actions[specialist_name])
        
        return results
    
    def step_meta_controller_envs(self, actions: np.ndarray) -> Tuple:
        """Step meta-controller environments."""
        if self.meta_controller_envs is None:
            raise ValueError("Meta-controller environments not created")
        
        return self.meta_controller_envs.step(actions)
    
    def get_specialist_observations(self) -> Dict[str, np.ndarray]:
        """Get current observations from all specialist environments."""
        observations = {}
        
        for specialist_name, env in self.specialist_envs.items():
            observations[specialist_name] = env.observation_space.sample()  # Placeholder
        
        return observations
    
    def get_meta_controller_observations(self) -> np.ndarray:
        """Get current observations from meta-controller environments."""
        if self.meta_controller_envs is None:
            raise ValueError("Meta-controller environments not created")
        
        return self.meta_controller_envs.observation_space.sample()  # Placeholder
    
    def close_all_envs(self) -> None:
        """Close all environments."""
        for env in self.specialist_envs.values():
            env.close()
        
        if self.meta_controller_envs is not None:
            self.meta_controller_envs.close()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics from all environments."""
        stats = {
            'phase': self.current_phase,
            'n_envs': self.n_envs,
            'specialist_stats': {},
            'meta_controller_stats': {}
        }
        
        # Get specialist stats
        for specialist_name, env in self.specialist_envs.items():
            if hasattr(env, 'get_episode_statistics'):
                stats['specialist_stats'][specialist_name] = env.get_episode_statistics()
        
        # Get meta-controller stats
        if self.meta_controller_envs is not None and hasattr(self.meta_controller_envs, 'get_episode_statistics'):
            stats['meta_controller_stats'] = self.meta_controller_envs.get_episode_statistics()
        
        return stats


class CurriculumLearningWrapper:
    """
    Wrapper for curriculum learning in hierarchical multi-agent training.
    
    Gradually increases difficulty:
    - Phase 1: Individual specialist training (easy scenarios)
    - Phase 2: Meta-controller training (medium scenarios)
    - Phase 3: Joint training (hard scenarios)
    """
    
    def __init__(
        self,
        parallel_wrapper: ParallelHierarchicalWrapper,
        curriculum_config: Dict[str, Any]
    ):
        self.parallel_wrapper = parallel_wrapper
        self.curriculum_config = curriculum_config
        
        # Curriculum phases
        self.phases = {
            1: {
                'name': 'Individual Specialist Training',
                'difficulty': 'easy',
                'scenarios': ['low_volatility', 'trending_market'],
                'reward_shaping': 'individual_performance'
            },
            2: {
                'name': 'Meta-Controller Training',
                'difficulty': 'medium',
                'scenarios': ['moderate_volatility', 'ranging_market'],
                'reward_shaping': 'portfolio_performance'
            },
            3: {
                'name': 'Joint Fine-Tuning',
                'difficulty': 'hard',
                'scenarios': ['high_volatility', 'crisis_market'],
                'reward_shaping': 'risk_adjusted_performance'
            }
        }
        
        self.current_phase = 1
        self.phase_progress = 0.0  # 0.0 to 1.0
    
    def update_curriculum(self, episode: int, total_episodes: int) -> None:
        """Update curriculum based on training progress."""
        
        # Calculate overall progress
        overall_progress = episode / total_episodes
        
        # Determine current phase
        if overall_progress < 0.4:  # First 40% of training
            self.current_phase = 1
            self.phase_progress = overall_progress / 0.4
        elif overall_progress < 0.7:  # Next 30% of training
            self.current_phase = 2
            self.phase_progress = (overall_progress - 0.4) / 0.3
        else:  # Final 30% of training
            self.current_phase = 3
            self.phase_progress = (overall_progress - 0.7) / 0.3
        
        # Update parallel wrapper phase
        self.parallel_wrapper.current_phase = self.current_phase
    
    def get_current_scenario(self) -> str:
        """Get current scenario based on curriculum phase."""
        phase_config = self.phases[self.current_phase]
        scenarios = phase_config['scenarios']
        
        # Select scenario based on phase progress
        scenario_index = int(self.phase_progress * len(scenarios))
        scenario_index = min(scenario_index, len(scenarios) - 1)
        
        return scenarios[scenario_index]
    
    def get_reward_shaping_config(self) -> Dict[str, Any]:
        """Get reward shaping configuration for current phase."""
        phase_config = self.phases[self.current_phase]
        
        base_config = {
            'transaction_cost_weight': 1.0,
            'risk_penalty_weight': 1.0,
            'diversification_bonus_weight': 0.5
        }
        
        # Adjust weights based on phase
        if self.current_phase == 1:  # Individual training
            base_config['transaction_cost_weight'] = 0.5  # Lower penalty
            base_config['risk_penalty_weight'] = 0.5  # Lower penalty
            base_config['diversification_bonus_weight'] = 0.3  # Lower bonus
        
        elif self.current_phase == 2:  # Meta-controller training
            base_config['transaction_cost_weight'] = 1.0  # Normal penalty
            base_config['risk_penalty_weight'] = 1.0  # Normal penalty
            base_config['diversification_bonus_weight'] = 0.5  # Normal bonus
        
        else:  # Joint training
            base_config['transaction_cost_weight'] = 1.5  # Higher penalty
            base_config['risk_penalty_weight'] = 1.5  # Higher penalty
            base_config['diversification_bonus_weight'] = 0.7  # Higher bonus
        
        return base_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration for current phase."""
        phase_config = self.phases[self.current_phase]
        
        base_config = {
            'learning_rate': 0.0003,
            'batch_size': 256,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2
        }
        
        # Adjust learning rate based on phase
        if self.current_phase == 1:  # Individual training
            base_config['learning_rate'] = 0.0005  # Higher learning rate
            base_config['n_epochs'] = 5  # Fewer epochs
        
        elif self.current_phase == 2:  # Meta-controller training
            base_config['learning_rate'] = 0.0003  # Normal learning rate
            base_config['n_epochs'] = 10  # Normal epochs
        
        else:  # Joint training
            base_config['learning_rate'] = 0.0001  # Lower learning rate
            base_config['n_epochs'] = 15  # More epochs
        
        return base_config
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get current curriculum statistics."""
        return {
            'current_phase': self.current_phase,
            'phase_name': self.phases[self.current_phase]['name'],
            'phase_progress': self.phase_progress,
            'current_scenario': self.get_current_scenario(),
            'difficulty': self.phases[self.current_phase]['difficulty'],
            'reward_shaping': self.phases[self.current_phase]['reward_shaping']
        }
