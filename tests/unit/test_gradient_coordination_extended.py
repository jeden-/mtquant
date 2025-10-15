"""
Extended unit tests for gradient_coordination.py to increase coverage.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from mtquant.agents.training.gradient_coordination import (
    GradientUpdateType,
    GradientCoordinationConfig,
    GradientCoordinationSystem
)


class TestGradientUpdateType:
    """Test GradientUpdateType enum."""
    
    def test_gradient_update_type_values(self):
        """Test GradientUpdateType enum values."""
        assert GradientUpdateType.META_CONTROLLER.value == "meta_controller"
        assert GradientUpdateType.SPECIALIST.value == "specialist"
        assert GradientUpdateType.JOINT.value == "joint"
        assert len(GradientUpdateType) == 3


class TestGradientCoordinationConfig:
    """Test GradientCoordinationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GradientCoordinationConfig()
        
        assert config.meta_update_freq == 1
        assert config.specialist_update_freq == 5
        assert config.joint_update_freq == 10
        assert config.gradient_clipping == 0.5
        assert config.gradient_scaling is True
        assert config.gradient_momentum == 0.9
        assert config.meta_lr_schedule is True
        assert config.specialist_lr_schedule is True
        assert config.joint_lr_schedule is True
        assert config.coordination_weight == 0.5
        assert config.individual_weight == 0.5
        assert config.stability_weight == 0.3
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GradientCoordinationConfig(
            meta_update_freq=2,
            specialist_update_freq=3,
            joint_update_freq=7,
            gradient_clipping=1.0,
            gradient_scaling=False,
            gradient_momentum=0.8,
            meta_lr_schedule=False,
            specialist_lr_schedule=False,
            joint_lr_schedule=False,
            coordination_weight=0.7,
            individual_weight=0.3,
            stability_weight=0.2
        )
        
        assert config.meta_update_freq == 2
        assert config.specialist_update_freq == 3
        assert config.joint_update_freq == 7
        assert config.gradient_clipping == 1.0
        assert config.gradient_scaling is False
        assert config.gradient_momentum == 0.8
        assert config.meta_lr_schedule is False
        assert config.specialist_lr_schedule is False
        assert config.joint_lr_schedule is False
        assert config.coordination_weight == 0.7
        assert config.individual_weight == 0.3
        assert config.stability_weight == 0.2


class TestGradientCoordinationSystem:
    """Test GradientCoordinationSystem class."""
    
    def test_system_initialization(self):
        """Test GradientCoordinationSystem initialization."""
        config = GradientCoordinationConfig()
        
        # Mock meta controller and specialists
        meta_controller = Mock()
        specialists = {
            'forex': Mock(),
            'commodities': Mock(),
            'equity': Mock()
        }
        
        system = GradientCoordinationSystem(config, meta_controller, specialists)
        
        assert system.config == config
        assert system.meta_controller == meta_controller
        assert system.specialists == specialists
        assert 'meta_controller' in system.gradient_history
        assert 'forex' in system.gradient_history
        assert 'commodities' in system.gradient_history
        assert 'equity' in system.gradient_history
        assert 'meta_controller' in system.gradient_norms
        assert 'forex' in system.gradient_norms
        assert 'meta_controller' in system.performance_history
        assert 'joint' in system.performance_history
        assert 'meta_controller' in system.learning_rates
        assert 'forex' in system.learning_rates
        assert 'meta_controller' in system.update_counters
        assert 'joint' in system.update_counters
        assert 'gradient_alignment' in system.coordination_metrics
    
    def test_should_update_meta_controller(self):
        """Test meta controller update decision."""
        config = GradientCoordinationConfig(meta_update_freq=2)
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        system = GradientCoordinationSystem(config, meta_controller, specialists)
        
        # Step 0: should update (step % freq == 0)
        assert system.should_update('meta_controller', 0) is True
        
        # Step 1: should not update
        assert system.should_update('meta_controller', 1) is False
        
        # Step 2: should update
        assert system.should_update('meta_controller', 2) is True
    
    def test_should_update_specialists(self):
        """Test specialists update decision."""
        config = GradientCoordinationConfig(specialist_update_freq=3)
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        system = GradientCoordinationSystem(config, meta_controller, specialists)
        
        # Step 0: should update
        assert system.should_update('forex', 0) is True
        
        # Step 1: should not update
        assert system.should_update('forex', 1) is False
        
        # Step 3: should update
        assert system.should_update('forex', 3) is True
    
    def test_should_update_joint(self):
        """Test joint update decision."""
        config = GradientCoordinationConfig(joint_update_freq=5)
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        system = GradientCoordinationSystem(config, meta_controller, specialists)
        
        # Step 0: should update
        assert system.should_update('joint', 0) is True
        
        # Step 1: should not update
        assert system.should_update('joint', 1) is False
        
        # Step 5: should update
        assert system.should_update('joint', 5) is True
    
    def test_should_update_unknown_agent(self):
        """Test update decision for unknown agent."""
        config = GradientCoordinationConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        system = GradientCoordinationSystem(config, meta_controller, specialists)
        
        # Unknown agent should not update
        assert system.should_update('unknown', 0) is False
    
    def test_coordinate_gradients(self):
        """Test gradient coordination."""
        config = GradientCoordinationConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        system = GradientCoordinationSystem(config, meta_controller, specialists)
        
        # Mock gradients
        gradients = {
            'meta_controller': torch.tensor([1.0, 2.0]),
            'forex': torch.tensor([3.0, 4.0])
        }
        
        coordinated = system.coordinate_gradients(gradients, step=0)
        
        assert isinstance(coordinated, dict)
        assert 'meta_controller' in coordinated
        assert 'forex' in coordinated
    
    def test_coordinate_gradients_empty(self):
        """Test gradient coordination with empty gradients."""
        config = GradientCoordinationConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        system = GradientCoordinationSystem(config, meta_controller, specialists)
        
        coordinated = system.coordinate_gradients({}, step=0)
        
        assert isinstance(coordinated, dict)
        assert len(coordinated) == 0
    
    def test_reset_system(self):
        """Test resetting the system."""
        config = GradientCoordinationConfig()
        meta_controller = Mock()
        specialists = {'forex': Mock()}
        
        system = GradientCoordinationSystem(config, meta_controller, specialists)
        
        # Add some data
        system.gradient_history['meta_controller'].append(torch.tensor([1.0]))
        system.gradient_norms['meta_controller'].append(1.0)
        system.performance_history['meta_controller'].append(0.5)
        system.update_counters['meta_controller'] = 5
        
        system.reset()
        
        # Check that data is cleared
        assert len(system.gradient_history['meta_controller']) == 0
        assert len(system.gradient_norms['meta_controller']) == 0
        assert len(system.performance_history['meta_controller']) == 0
        assert system.update_counters['meta_controller'] == 0
