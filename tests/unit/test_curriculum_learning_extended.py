"""
Extended unit tests for curriculum_learning.py to increase coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from collections import deque

from mtquant.agents.training.curriculum_learning import (
    DifficultyLevel,
    ScenarioType,
    CurriculumConfig,
    AdvancedCurriculumLearning
)


class TestDifficultyLevel:
    """Test DifficultyLevel enum."""
    
    def test_difficulty_level_values(self):
        """Test DifficultyLevel enum values."""
        assert DifficultyLevel.EASY.value == "easy"
        assert DifficultyLevel.MEDIUM.value == "medium"
        assert DifficultyLevel.HARD.value == "hard"
        assert DifficultyLevel.EXPERT.value == "expert"
    
    def test_difficulty_level_membership(self):
        """Test DifficultyLevel membership."""
        assert DifficultyLevel.EASY in DifficultyLevel
        assert DifficultyLevel.MEDIUM in DifficultyLevel
        assert DifficultyLevel.HARD in DifficultyLevel
        assert DifficultyLevel.EXPERT in DifficultyLevel


class TestScenarioType:
    """Test ScenarioType enum."""
    
    def test_scenario_type_values(self):
        """Test ScenarioType enum values."""
        assert ScenarioType.LOW_VOLATILITY.value == "low_volatility"
        assert ScenarioType.TRENDING_MARKET.value == "trending_market"
        assert ScenarioType.RANGING_MARKET.value == "ranging_market"
        assert ScenarioType.HIGH_VOLATILITY.value == "high_volatility"
        assert ScenarioType.CRISIS_MARKET.value == "crisis_market"
        assert ScenarioType.CORRELATION_BREAKDOWN.value == "correlation_breakdown"
        assert ScenarioType.SECTOR_ROTATION.value == "sector_rotation"
        assert ScenarioType.MACRO_EVENTS.value == "macro_events"
    
    def test_scenario_type_membership(self):
        """Test ScenarioType membership."""
        assert ScenarioType.LOW_VOLATILITY in ScenarioType
        assert ScenarioType.TRENDING_MARKET in ScenarioType
        assert ScenarioType.RANGING_MARKET in ScenarioType
        assert ScenarioType.HIGH_VOLATILITY in ScenarioType
        assert ScenarioType.CRISIS_MARKET in ScenarioType
        assert ScenarioType.CORRELATION_BREAKDOWN in ScenarioType
        assert ScenarioType.SECTOR_ROTATION in ScenarioType
        assert ScenarioType.MACRO_EVENTS in ScenarioType


class TestCurriculumConfig:
    """Test CurriculumConfig dataclass."""
    
    def test_curriculum_config_defaults(self):
        """Test CurriculumConfig default values."""
        config = CurriculumConfig()
        
        assert config.difficulty_levels == [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        assert config.difficulty_weights == {'easy': 0.4, 'medium': 0.4, 'hard': 0.2}
        assert config.scenarios_per_difficulty == {
            'easy': [ScenarioType.LOW_VOLATILITY, ScenarioType.TRENDING_MARKET],
            'medium': [ScenarioType.RANGING_MARKET, ScenarioType.SECTOR_ROTATION],
            'hard': [ScenarioType.HIGH_VOLATILITY, ScenarioType.CRISIS_MARKET, ScenarioType.CORRELATION_BREAKDOWN]
        }
        assert config.scenario_weights == {
            'low_volatility': 0.3,
            'trending_market': 0.3,
            'ranging_market': 0.2,
            'high_volatility': 0.1,
            'crisis_market': 0.05,
            'correlation_breakdown': 0.03,
            'sector_rotation': 0.02
        }
        assert config.performance_threshold == 0.1
        assert config.stability_threshold == 0.05
        assert config.minimum_episodes == 100
        assert config.evaluation_window == 50
        assert config.adaptive_difficulty is True
        assert config.adaptive_scenarios is True
        assert config.performance_decay == 0.95
        assert config.exploration_bonus == 0.1
        assert config.progress_tracking is True
        assert config.detailed_logging is True
    
    def test_curriculum_config_custom(self):
        """Test CurriculumConfig with custom values."""
        config = CurriculumConfig(
            difficulty_levels=[DifficultyLevel.EASY, DifficultyLevel.HARD],
            difficulty_weights={'easy': 0.6, 'hard': 0.4},
            scenarios_per_difficulty={
                'easy': [ScenarioType.LOW_VOLATILITY],
                'hard': [ScenarioType.CRISIS_MARKET]
            },
            scenario_weights={'low_volatility': 0.5, 'crisis_market': 0.5},
            performance_threshold=0.2,
            stability_threshold=0.1,
            minimum_episodes=200,
            evaluation_window=100,
            adaptive_difficulty=False,
            adaptive_scenarios=False,
            performance_decay=0.9,
            exploration_bonus=0.2,
            progress_tracking=False,
            detailed_logging=False
        )
        
        assert config.difficulty_levels == [DifficultyLevel.EASY, DifficultyLevel.HARD]
        assert config.difficulty_weights == {'easy': 0.6, 'hard': 0.4}
        assert config.scenarios_per_difficulty == {
            'easy': [ScenarioType.LOW_VOLATILITY],
            'hard': [ScenarioType.CRISIS_MARKET]
        }
        assert config.scenario_weights == {'low_volatility': 0.5, 'crisis_market': 0.5}
        assert config.performance_threshold == 0.2
        assert config.stability_threshold == 0.1
        assert config.minimum_episodes == 200
        assert config.evaluation_window == 100
        assert config.adaptive_difficulty is False
        assert config.adaptive_scenarios is False
        assert config.performance_decay == 0.9
        assert config.exploration_bonus == 0.2
        assert config.progress_tracking is False
        assert config.detailed_logging is False


class TestAdvancedCurriculumLearning:
    """Test AdvancedCurriculumLearning class."""
    
    def test_curriculum_learning_initialization(self):
        """Test AdvancedCurriculumLearning initialization."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock(), 'commodities': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        assert curriculum.config == config
        assert curriculum.meta_controller == meta_controller
        assert curriculum.specialists == specialists
        assert curriculum.current_difficulty == DifficultyLevel.EASY
        assert curriculum.current_scenario == ScenarioType.LOW_VOLATILITY
        assert curriculum.difficulty_progress == 0.0
        assert curriculum.scenario_progress == 0.0
        assert curriculum.episode_count == 0
        assert curriculum.difficulty_episode_count == 0
        assert curriculum.scenario_episode_count == 0
        assert isinstance(curriculum.performance_history, dict)
        assert isinstance(curriculum.scenario_performance, dict)
        assert isinstance(curriculum.difficulty_performance, dict)
        assert isinstance(curriculum.progression_history, list)
        assert isinstance(curriculum.adaptive_weights, dict)
        assert isinstance(curriculum.scenario_adaptive_weights, dict)
    
    def test_get_current_curriculum(self):
        """Test getting current curriculum."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        difficulty, scenario = curriculum.get_current_curriculum()
        
        assert difficulty == DifficultyLevel.EASY
        assert scenario == ScenarioType.LOW_VOLATILITY
    
    def test_update_performance(self):
        """Test updating performance."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock(), 'commodities': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        joint_performance = 0.5
        meta_performance = 0.4
        specialist_performance = {'forex': 0.6, 'commodities': 0.7}
        
        curriculum.update_performance(joint_performance, meta_performance, specialist_performance)
        
        assert len(curriculum.performance_history['joint']) == 1
        assert len(curriculum.performance_history['meta_controller']) == 1
        assert len(curriculum.performance_history['forex']) == 1
        assert len(curriculum.performance_history['commodities']) == 1
        assert curriculum.performance_history['joint'][0] == 0.5
        assert curriculum.performance_history['meta_controller'][0] == 0.4
        assert curriculum.performance_history['forex'][0] == 0.6
        assert curriculum.performance_history['commodities'][0] == 0.7
        assert curriculum.episode_count == 1
        assert curriculum.difficulty_episode_count == 1
        assert curriculum.scenario_episode_count == 1
    
    def test_update_performance_with_progression(self):
        """Test updating performance with progression."""
        config = CurriculumConfig(minimum_episodes=1, evaluation_window=1, performance_threshold=0.1)
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Add enough performance data to trigger progression
        for i in range(2):
            curriculum.update_performance(0.5, 0.4, {'forex': 0.6})
        
        # Should have progressed
        assert curriculum.episode_count == 2
    
    def test_should_progress_difficulty_minimum_episodes(self):
        """Test difficulty progression with minimum episodes requirement."""
        config = CurriculumConfig(minimum_episodes=100)
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Should not progress due to minimum episodes
        assert curriculum._should_progress_difficulty() is False
    
    def test_should_progress_difficulty_performance_threshold(self):
        """Test difficulty progression with performance threshold."""
        config = CurriculumConfig(minimum_episodes=1, evaluation_window=1, performance_threshold=0.5)
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Add low performance data
        curriculum.difficulty_performance[DifficultyLevel.EASY].append(0.1)
        
        # Should not progress due to low performance
        assert curriculum._should_progress_difficulty() is False
    
    def test_should_progress_difficulty_stability_threshold(self):
        """Test difficulty progression with stability threshold."""
        config = CurriculumConfig(minimum_episodes=1, evaluation_window=2, performance_threshold=0.1, stability_threshold=0.01)
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Add unstable performance data
        curriculum.difficulty_performance[DifficultyLevel.EASY].extend([0.1, 0.9])
        
        # Should not progress due to instability
        assert curriculum._should_progress_difficulty() is False
    
    def test_should_progress_difficulty_max_level(self):
        """Test difficulty progression at max level."""
        config = CurriculumConfig(minimum_episodes=1, evaluation_window=1, performance_threshold=0.1)
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        curriculum.current_difficulty = DifficultyLevel.HARD  # Max level
        
        # Add good performance data
        curriculum.difficulty_performance[DifficultyLevel.HARD].append(0.5)
        
        # Should not progress as already at max level
        assert curriculum._should_progress_difficulty() is False
    
    def test_should_progress_scenario_minimum_episodes(self):
        """Test scenario progression with minimum episodes requirement."""
        config = CurriculumConfig(minimum_episodes=100)
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Should not progress due to minimum episodes
        assert curriculum._should_progress_scenario() is False
    
    def test_should_progress_scenario_performance_threshold(self):
        """Test scenario progression with performance threshold."""
        config = CurriculumConfig(minimum_episodes=1, evaluation_window=1, performance_threshold=0.5)
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Add low performance data
        curriculum.scenario_performance[ScenarioType.LOW_VOLATILITY].append(0.1)
        
        # Should not progress due to low performance
        assert curriculum._should_progress_scenario() is False
    
    def test_should_progress_scenario_stability_threshold(self):
        """Test scenario progression with stability threshold."""
        config = CurriculumConfig(minimum_episodes=1, evaluation_window=2, performance_threshold=0.1, stability_threshold=0.01)
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Add unstable performance data
        curriculum.scenario_performance[ScenarioType.LOW_VOLATILITY].extend([0.1, 0.9])
        
        # Should not progress due to instability
        assert curriculum._should_progress_scenario() is False
    
    def test_should_progress_scenario_max_scenario(self):
        """Test scenario progression at max scenario."""
        config = CurriculumConfig(minimum_episodes=1, evaluation_window=1, performance_threshold=0.1)
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        curriculum.current_scenario = ScenarioType.TRENDING_MARKET  # Last scenario for easy difficulty
        
        # Add good performance data
        curriculum.scenario_performance[ScenarioType.TRENDING_MARKET].append(0.5)
        
        # Should still progress as scenarios cycle back to first scenario
        assert curriculum._should_progress_scenario() is True
    
    def test_progress_difficulty(self):
        """Test progressing to next difficulty level."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Start at EASY
        assert curriculum.current_difficulty == DifficultyLevel.EASY
        
        curriculum._progress_difficulty()
        
        # Should progress to MEDIUM
        assert curriculum.current_difficulty == DifficultyLevel.MEDIUM
        assert curriculum.difficulty_episode_count == 0
        assert curriculum.difficulty_progress == 0.0
    
    def test_progress_scenario(self):
        """Test progressing to next scenario."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Start at LOW_VOLATILITY
        assert curriculum.current_scenario == ScenarioType.LOW_VOLATILITY
        
        curriculum._progress_scenario()
        
        # Should progress to TRENDING_MARKET
        assert curriculum.current_scenario == ScenarioType.TRENDING_MARKET
        assert curriculum.scenario_episode_count == 0
        assert curriculum.scenario_progress == 0.0
    
    def test_reset_scenario_for_difficulty(self):
        """Test resetting scenario for difficulty."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Change to MEDIUM difficulty
        curriculum.current_difficulty = DifficultyLevel.MEDIUM
        curriculum.current_scenario = ScenarioType.SECTOR_ROTATION
        curriculum.scenario_episode_count = 50
        curriculum.scenario_progress = 0.8
        
        curriculum._reset_scenario_for_difficulty()
        
        # Should reset to first scenario of MEDIUM difficulty
        assert curriculum.current_scenario == ScenarioType.RANGING_MARKET
        assert curriculum.scenario_episode_count == 0
        assert curriculum.scenario_progress == 0.0
    
    def test_update_adaptive_weights(self):
        """Test updating adaptive weights."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Add performance data
        curriculum.difficulty_performance[DifficultyLevel.EASY].extend([0.5, 0.6, 0.7])
        curriculum.scenario_performance[ScenarioType.LOW_VOLATILITY].extend([0.4, 0.5, 0.6])
        
        curriculum._update_adaptive_weights()
        
        # Weights should be updated
        assert isinstance(curriculum.adaptive_weights, dict)
        assert isinstance(curriculum.scenario_adaptive_weights, dict)
    
    def test_record_progression(self):
        """Test recording progression."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        curriculum._record_progression("difficulty", "medium")
        
        assert len(curriculum.progression_history) == 1
        assert curriculum.progression_history[0]['type'] == "difficulty"
        assert curriculum.progression_history[0]['new_value'] == "medium"
        assert curriculum.progression_history[0]['episode'] == 0
    
    def test_get_scenario_parameters(self):
        """Test getting scenario parameters."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Test LOW_VOLATILITY scenario
        params = curriculum.get_scenario_parameters()
        assert isinstance(params, dict)
        assert 'volatility_multiplier' in params
        assert 'trend_strength' in params
        assert 'noise_level' in params
    
    def test_get_scenario_parameters_crisis_market(self):
        """Test getting scenario parameters for crisis market."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        curriculum.current_scenario = ScenarioType.CRISIS_MARKET
        
        params = curriculum.get_scenario_parameters()
        assert isinstance(params, dict)
        assert params['volatility_multiplier'] == 3.0
        assert params['trend_strength'] == -0.6
        assert params['noise_level'] == 0.8
    
    def test_get_scenario_parameters_correlation_breakdown(self):
        """Test getting scenario parameters for correlation breakdown."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        curriculum.current_scenario = ScenarioType.CORRELATION_BREAKDOWN
        
        params = curriculum.get_scenario_parameters()
        assert isinstance(params, dict)
        assert params['volatility_multiplier'] == 1.5
        assert params['trend_strength'] == 0.2
        assert params['noise_level'] == 0.4
        assert params['correlation_breakdown'] is True
    
    def test_get_scenario_parameters_sector_rotation(self):
        """Test getting scenario parameters for sector rotation."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        curriculum.current_scenario = ScenarioType.SECTOR_ROTATION
        
        params = curriculum.get_scenario_parameters()
        assert isinstance(params, dict)
        assert params['volatility_multiplier'] == 1.2
        assert params['trend_strength'] == 0.5
        assert params['noise_level'] == 0.3
        assert params['sector_rotation'] is True
    
    def test_get_scenario_parameters_macro_events(self):
        """Test getting scenario parameters for macro events."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        curriculum.current_scenario = ScenarioType.MACRO_EVENTS
        
        params = curriculum.get_scenario_parameters()
        assert isinstance(params, dict)
        assert params['volatility_multiplier'] == 1.8
        assert params['trend_strength'] == 0.3
        assert params['noise_level'] == 0.6
        assert params['macro_events'] is True
    
    def test_get_curriculum_stats(self):
        """Test getting curriculum statistics."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock(), 'commodities': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Add some performance data
        curriculum.update_performance(0.5, 0.4, {'forex': 0.6, 'commodities': 0.7})
        
        stats = curriculum.get_curriculum_stats()
        
        assert isinstance(stats, dict)
        assert 'current_difficulty' in stats
        assert 'current_scenario' in stats
        assert 'episode_count' in stats
        assert 'difficulty_episode_count' in stats
        assert 'scenario_episode_count' in stats
        assert 'difficulty_progress' in stats
        assert 'scenario_progress' in stats
        assert 'performance_history' in stats
        assert 'scenario_performance' in stats
        assert 'difficulty_performance' in stats
        assert 'adaptive_weights' in stats
        assert 'scenario_adaptive_weights' in stats
        assert 'progression_history' in stats
        
        assert stats['current_difficulty'] == 'easy'
        assert stats['current_scenario'] == 'low_volatility'
        assert stats['episode_count'] == 1
        assert stats['difficulty_episode_count'] == 1
        assert stats['scenario_episode_count'] == 1
        assert stats['difficulty_progress'] == 0.0
        assert stats['scenario_progress'] == 0.0
    
    def test_reset(self):
        """Test resetting curriculum learning system."""
        config = CurriculumConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        curriculum = AdvancedCurriculumLearning(config, meta_controller, specialists)
        
        # Add some data
        curriculum.update_performance(0.5, 0.4, {'forex': 0.6})
        curriculum.current_difficulty = DifficultyLevel.HARD
        curriculum.current_scenario = ScenarioType.CRISIS_MARKET
        curriculum.difficulty_progress = 0.8
        curriculum.scenario_progress = 0.6
        
        curriculum.reset()
        
        # Should be reset to initial state
        assert curriculum.current_difficulty == DifficultyLevel.EASY
        assert curriculum.current_scenario == ScenarioType.LOW_VOLATILITY
        assert curriculum.difficulty_progress == 0.0
        assert curriculum.scenario_progress == 0.0
        assert curriculum.episode_count == 0
        assert curriculum.difficulty_episode_count == 0
        assert curriculum.scenario_episode_count == 0
        assert len(curriculum.progression_history) == 0
        
        # Performance histories should be cleared
        for perf_history in curriculum.performance_history.values():
            assert len(perf_history) == 0
        
        for scenario_perf in curriculum.scenario_performance.values():
            assert len(scenario_perf) == 0
        
        for difficulty_perf in curriculum.difficulty_performance.values():
            assert len(difficulty_perf) == 0
