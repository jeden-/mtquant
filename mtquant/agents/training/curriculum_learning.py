"""
Advanced Curriculum Learning for Joint Training

This module provides sophisticated curriculum learning for Phase 3 joint training,
progressively increasing difficulty and complexity to improve learning efficiency.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import random

from ..hierarchical.meta_controller import MetaController
from ..hierarchical.base_specialist import BaseSpecialist


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class ScenarioType(Enum):
    """Types of training scenarios."""
    LOW_VOLATILITY = "low_volatility"
    TRENDING_MARKET = "trending_market"
    RANGING_MARKET = "ranging_market"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS_MARKET = "crisis_market"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    SECTOR_ROTATION = "sector_rotation"
    MACRO_EVENTS = "macro_events"


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning system."""
    
    # Difficulty progression
    difficulty_levels: List[DifficultyLevel] = None
    difficulty_weights: Dict[str, float] = None
    
    # Scenario configuration
    scenarios_per_difficulty: Dict[str, List[ScenarioType]] = None
    scenario_weights: Dict[str, float] = None
    
    # Progression criteria
    performance_threshold: float = 0.1
    stability_threshold: float = 0.05
    minimum_episodes: int = 100
    evaluation_window: int = 50
    
    # Adaptive parameters
    adaptive_difficulty: bool = True
    adaptive_scenarios: bool = True
    performance_decay: float = 0.95
    exploration_bonus: float = 0.1
    
    # Monitoring
    progress_tracking: bool = True
    detailed_logging: bool = True
    
    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        
        if self.difficulty_weights is None:
            self.difficulty_weights = {
                'easy': 0.4,
                'medium': 0.4,
                'hard': 0.2
            }
        
        if self.scenarios_per_difficulty is None:
            self.scenarios_per_difficulty = {
                'easy': [ScenarioType.LOW_VOLATILITY, ScenarioType.TRENDING_MARKET],
                'medium': [ScenarioType.RANGING_MARKET, ScenarioType.SECTOR_ROTATION],
                'hard': [ScenarioType.HIGH_VOLATILITY, ScenarioType.CRISIS_MARKET, ScenarioType.CORRELATION_BREAKDOWN]
            }
        
        if self.scenario_weights is None:
            self.scenario_weights = {
                'low_volatility': 0.3,
                'trending_market': 0.3,
                'ranging_market': 0.2,
                'high_volatility': 0.1,
                'crisis_market': 0.05,
                'correlation_breakdown': 0.03,
                'sector_rotation': 0.02
            }


class AdvancedCurriculumLearning:
    """
    Advanced curriculum learning system for joint training.
    
    Features:
    - Multi-level difficulty progression
    - Scenario-based training
    - Adaptive difficulty adjustment
    - Performance-based progression
    - Comprehensive monitoring
    """
    
    def __init__(
        self,
        config: CurriculumConfig,
        meta_controller: MetaController,
        specialists: Dict[str, BaseSpecialist]
    ):
        self.config = config
        self.meta_controller = meta_controller
        self.specialists = specialists
        
        # Curriculum state
        self.current_difficulty = DifficultyLevel.EASY
        self.current_scenario = ScenarioType.LOW_VOLATILITY
        self.difficulty_progress = 0.0
        self.scenario_progress = 0.0
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = {
            'joint': deque(maxlen=1000),
            'meta_controller': deque(maxlen=1000),
            **{name: deque(maxlen=1000) for name in specialists.keys()}
        }
        
        self.scenario_performance: Dict[ScenarioType, deque] = {
            scenario: deque(maxlen=100) for scenario in ScenarioType
        }
        
        self.difficulty_performance: Dict[DifficultyLevel, deque] = {
            difficulty: deque(maxlen=100) for difficulty in DifficultyLevel
        }
        
        # Progression tracking
        self.episode_count = 0
        self.difficulty_episode_count = 0
        self.scenario_episode_count = 0
        
        self.progression_history: List[Dict[str, Any]] = []
        
        # Adaptive parameters
        self.adaptive_weights: Dict[str, float] = {
            'easy': self.config.difficulty_weights['easy'],
            'medium': self.config.difficulty_weights['medium'],
            'hard': self.config.difficulty_weights['hard']
        }
        
        self.scenario_adaptive_weights: Dict[str, float] = self.config.scenario_weights.copy()
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_current_curriculum(self) -> Tuple[DifficultyLevel, ScenarioType]:
        """Get current curriculum (difficulty and scenario)."""
        return self.current_difficulty, self.current_scenario
    
    def update_performance(
        self,
        joint_performance: float,
        meta_performance: float,
        specialist_performance: Dict[str, float]
    ) -> None:
        """Update performance history for curriculum learning."""
        
        # Update performance histories
        self.performance_history['joint'].append(joint_performance)
        self.performance_history['meta_controller'].append(meta_performance)
        
        for specialist_name, performance in specialist_performance.items():
            if specialist_name in self.performance_history:
                self.performance_history[specialist_name].append(performance)
        
        # Update scenario and difficulty performance
        self.scenario_performance[self.current_scenario].append(joint_performance)
        self.difficulty_performance[self.current_difficulty].append(joint_performance)
        
        # Update episode count
        self.episode_count += 1
        self.difficulty_episode_count += 1
        self.scenario_episode_count += 1
        
        # Check for progression
        self._check_progression()
        
        # Update adaptive weights
        if self.config.adaptive_difficulty:
            self._update_adaptive_weights()
    
    def _check_progression(self) -> None:
        """Check if progression to next difficulty/scenario is warranted."""
        
        # Check difficulty progression
        if self._should_progress_difficulty():
            self._progress_difficulty()
        
        # Check scenario progression
        if self._should_progress_scenario():
            self._progress_scenario()
    
    def _should_progress_difficulty(self) -> bool:
        """Check if should progress to next difficulty level."""
        
        # Minimum episodes requirement
        if self.difficulty_episode_count < self.config.minimum_episodes:
            return False
        
        # Performance requirement
        recent_performance = list(self.difficulty_performance[self.current_difficulty])[-self.config.evaluation_window:]
        if len(recent_performance) < self.config.evaluation_window:
            return False
        
        mean_performance = np.mean(recent_performance)
        if mean_performance < self.config.performance_threshold:
            return False
        
        # Stability requirement
        performance_std = np.std(recent_performance)
        if performance_std > self.config.stability_threshold:
            return False
        
        # Check if there's a next difficulty level
        current_index = self.config.difficulty_levels.index(self.current_difficulty)
        if current_index >= len(self.config.difficulty_levels) - 1:
            return False
        
        return True
    
    def _should_progress_scenario(self) -> bool:
        """Check if should progress to next scenario."""
        
        # Minimum episodes requirement
        if self.scenario_episode_count < self.config.minimum_episodes // 2:
            return False
        
        # Performance requirement
        recent_performance = list(self.scenario_performance[self.current_scenario])[-self.config.evaluation_window:]
        if len(recent_performance) < self.config.evaluation_window:
            return False
        
        mean_performance = np.mean(recent_performance)
        if mean_performance < self.config.performance_threshold:
            return False
        
        # Stability requirement
        performance_std = np.std(recent_performance)
        if performance_std > self.config.stability_threshold:
            return False
        
        return True
    
    def _progress_difficulty(self) -> None:
        """Progress to next difficulty level."""
        
        current_index = self.config.difficulty_levels.index(self.current_difficulty)
        if current_index < len(self.config.difficulty_levels) - 1:
            self.current_difficulty = self.config.difficulty_levels[current_index + 1]
            self.difficulty_episode_count = 0
            self.difficulty_progress = 0.0
            
            # Reset scenario to first scenario of new difficulty
            self._reset_scenario_for_difficulty()
            
            # Log progression
            if self.config.detailed_logging:
                self.logger.info(f"Difficulty progression: {self.current_difficulty.value}")
            
            # Record progression
            self._record_progression('difficulty', self.current_difficulty.value)
    
    def _progress_scenario(self) -> None:
        """Progress to next scenario within current difficulty."""
        
        available_scenarios = self.config.scenarios_per_difficulty[self.current_difficulty.value]
        current_index = available_scenarios.index(self.current_scenario)
        
        if current_index < len(available_scenarios) - 1:
            self.current_scenario = available_scenarios[current_index + 1]
        else:
            # Cycle back to first scenario
            self.current_scenario = available_scenarios[0]
        
        self.scenario_episode_count = 0
        self.scenario_progress = 0.0
        
        # Log progression
        if self.config.detailed_logging:
            self.logger.info(f"Scenario progression: {self.current_scenario.value}")
        
        # Record progression
        self._record_progression('scenario', self.current_scenario.value)
    
    def _reset_scenario_for_difficulty(self) -> None:
        """Reset scenario to first scenario of current difficulty."""
        
        available_scenarios = self.config.scenarios_per_difficulty[self.current_difficulty.value]
        self.current_scenario = available_scenarios[0]
        self.scenario_episode_count = 0
        self.scenario_progress = 0.0
    
    def _update_adaptive_weights(self) -> None:
        """Update adaptive weights based on performance."""
        
        # Update difficulty weights
        for difficulty in self.config.difficulty_levels:
            difficulty_name = difficulty.value
            recent_performance = list(self.difficulty_performance[difficulty])[-50:]
            
            if len(recent_performance) > 0:
                mean_performance = np.mean(recent_performance)
                
                # Adjust weight based on performance
                if mean_performance > 0:
                    self.adaptive_weights[difficulty_name] *= 1.1
                else:
                    self.adaptive_weights[difficulty_name] *= 0.9
                
                # Normalize weights
                total_weight = sum(self.adaptive_weights.values())
                for key in self.adaptive_weights:
                    self.adaptive_weights[key] /= total_weight
        
        # Update scenario weights
        for scenario in ScenarioType:
            scenario_name = scenario.value
            recent_performance = list(self.scenario_performance[scenario])[-50:]
            
            if len(recent_performance) > 0:
                mean_performance = np.mean(recent_performance)
                
                # Adjust weight based on performance
                if mean_performance > 0:
                    self.scenario_adaptive_weights[scenario_name] *= 1.05
                else:
                    self.scenario_adaptive_weights[scenario_name] *= 0.95
        
        # Normalize scenario weights
        total_scenario_weight = sum(self.scenario_adaptive_weights.values())
        for key in self.scenario_adaptive_weights:
            self.scenario_adaptive_weights[key] /= total_scenario_weight
    
    def _record_progression(self, progression_type: str, new_value: str) -> None:
        """Record progression event."""
        
        progression_record = {
            'episode': self.episode_count,
            'type': progression_type,
            'old_value': self.current_difficulty.value if progression_type == 'difficulty' else self.current_scenario.value,
            'new_value': new_value,
            'joint_performance': list(self.performance_history['joint'])[-1] if self.performance_history['joint'] else 0.0,
            'meta_performance': list(self.performance_history['meta_controller'])[-1] if self.performance_history['meta_controller'] else 0.0,
            'specialist_performance': {
                name: list(perf)[-1] if perf else 0.0
                for name, perf in self.performance_history.items()
                if name not in ['joint', 'meta_controller']
            }
        }
        
        self.progression_history.append(progression_record)
    
    def get_scenario_parameters(self) -> Dict[str, Any]:
        """Get parameters for current scenario."""
        
        scenario_params = {
            'difficulty': self.current_difficulty.value,
            'scenario': self.current_scenario.value,
            'episode_count': self.episode_count,
            'difficulty_episode_count': self.difficulty_episode_count,
            'scenario_episode_count': self.scenario_episode_count
        }
        
        # Scenario-specific parameters
        if self.current_scenario == ScenarioType.LOW_VOLATILITY:
            scenario_params.update({
                'volatility_multiplier': 0.5,
                'trend_strength': 0.3,
                'noise_level': 0.1
            })
        elif self.current_scenario == ScenarioType.TRENDING_MARKET:
            scenario_params.update({
                'volatility_multiplier': 0.7,
                'trend_strength': 0.8,
                'noise_level': 0.2
            })
        elif self.current_scenario == ScenarioType.RANGING_MARKET:
            scenario_params.update({
                'volatility_multiplier': 0.6,
                'trend_strength': 0.1,
                'noise_level': 0.3
            })
        elif self.current_scenario == ScenarioType.HIGH_VOLATILITY:
            scenario_params.update({
                'volatility_multiplier': 2.0,
                'trend_strength': 0.4,
                'noise_level': 0.5
            })
        elif self.current_scenario == ScenarioType.CRISIS_MARKET:
            scenario_params.update({
                'volatility_multiplier': 3.0,
                'trend_strength': -0.6,
                'noise_level': 0.8
            })
        elif self.current_scenario == ScenarioType.CORRELATION_BREAKDOWN:
            scenario_params.update({
                'volatility_multiplier': 1.5,
                'trend_strength': 0.2,
                'noise_level': 0.4,
                'correlation_breakdown': True
            })
        elif self.current_scenario == ScenarioType.SECTOR_ROTATION:
            scenario_params.update({
                'volatility_multiplier': 1.2,
                'trend_strength': 0.5,
                'noise_level': 0.3,
                'sector_rotation': True
            })
        elif self.current_scenario == ScenarioType.MACRO_EVENTS:
            scenario_params.update({
                'volatility_multiplier': 1.8,
                'trend_strength': 0.3,
                'noise_level': 0.6,
                'macro_events': True
            })
        
        return scenario_params
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get comprehensive curriculum learning statistics."""
        
        stats = {
            'current_difficulty': self.current_difficulty.value,
            'current_scenario': self.current_scenario.value,
            'episode_count': self.episode_count,
            'difficulty_episode_count': self.difficulty_episode_count,
            'scenario_episode_count': self.scenario_episode_count,
            'difficulty_progress': self.difficulty_progress,
            'scenario_progress': self.scenario_progress,
            'performance_history': {
                name: list(perf) for name, perf in self.performance_history.items()
            },
            'scenario_performance': {
                scenario.value: list(perf) for scenario, perf in self.scenario_performance.items()
            },
            'difficulty_performance': {
                difficulty.value: list(perf) for difficulty, perf in self.difficulty_performance.items()
            },
            'adaptive_weights': self.adaptive_weights.copy(),
            'scenario_adaptive_weights': self.scenario_adaptive_weights.copy(),
            'progression_history': self.progression_history.copy()
        }
        
        return stats
    
    def reset(self) -> None:
        """Reset curriculum learning system."""
        
        # Reset state
        self.current_difficulty = DifficultyLevel.EASY
        self.current_scenario = ScenarioType.LOW_VOLATILITY
        self.difficulty_progress = 0.0
        self.scenario_progress = 0.0
        
        # Reset counters
        self.episode_count = 0
        self.difficulty_episode_count = 0
        self.scenario_episode_count = 0
        
        # Clear histories
        for perf_history in self.performance_history.values():
            perf_history.clear()
        
        for scenario_perf in self.scenario_performance.values():
            scenario_perf.clear()
        
        for difficulty_perf in self.difficulty_performance.values():
            difficulty_perf.clear()
        
        # Clear progression history
        self.progression_history.clear()
        
        # Reset adaptive weights
        self.adaptive_weights = self.config.difficulty_weights.copy()
        self.scenario_adaptive_weights = self.config.scenario_weights.copy()
        
        self.logger.info("Curriculum learning system reset")
