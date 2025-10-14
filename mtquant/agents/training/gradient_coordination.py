"""
Gradient Coordination System for Joint Training

This module provides gradient coordination between meta-controller and specialists
during Phase 3 joint fine-tuning to ensure stable and effective learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

from ..hierarchical.meta_controller import MetaController
from ..hierarchical.base_specialist import BaseSpecialist


class GradientUpdateType(Enum):
    """Types of gradient updates in joint training."""
    META_CONTROLLER = "meta_controller"
    SPECIALIST = "specialist"
    JOINT = "joint"


@dataclass
class GradientCoordinationConfig:
    """Configuration for gradient coordination system."""
    
    # Update frequencies
    meta_update_freq: int = 1  # Update meta every step
    specialist_update_freq: int = 5  # Update specialists every 5 steps
    joint_update_freq: int = 10  # Joint updates every 10 steps
    
    # Gradient coordination
    gradient_clipping: float = 0.5
    gradient_scaling: bool = True
    gradient_momentum: float = 0.9
    
    # Learning rate scheduling
    meta_lr_schedule: bool = True
    specialist_lr_schedule: bool = True
    joint_lr_schedule: bool = True
    
    # Coordination parameters
    coordination_weight: float = 0.5
    individual_weight: float = 0.5
    stability_weight: float = 0.3
    
    # Monitoring
    gradient_norm_threshold: float = 10.0
    performance_threshold: float = 0.1
    stability_window: int = 100


class GradientCoordinationSystem:
    """
    Gradient coordination system for joint training.
    
    Manages gradient updates between meta-controller and specialists to ensure:
    - Stable learning dynamics
    - Proper coordination between agents
    - Balanced individual vs joint performance
    - Adaptive learning rate scheduling
    """
    
    def __init__(
        self,
        config: GradientCoordinationConfig,
        meta_controller: MetaController,
        specialists: Dict[str, BaseSpecialist]
    ):
        self.config = config
        self.meta_controller = meta_controller
        self.specialists = specialists
        
        # Gradient tracking
        self.gradient_history: Dict[str, deque] = {
            'meta_controller': deque(maxlen=1000),
            **{name: deque(maxlen=1000) for name in specialists.keys()}
        }
        
        self.gradient_norms: Dict[str, List[float]] = {
            'meta_controller': [],
            **{name: [] for name in specialists.keys()}
        }
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {
            'meta_controller': [],
            **{name: [] for name in specialists.keys()},
            'joint': []
        }
        
        # Learning rate tracking
        self.learning_rates: Dict[str, float] = {
            'meta_controller': 0.0003,
            **{name: 0.0003 for name in specialists.keys()}
        }
        
        # Update counters
        self.update_counters: Dict[str, int] = {
            'meta_controller': 0,
            **{name: 0 for name in specialists.keys()},
            'joint': 0
        }
        
        # Coordination metrics
        self.coordination_metrics: Dict[str, List[float]] = {
            'gradient_alignment': [],
            'performance_correlation': [],
            'update_synchronization': [],
            'stability_metric': []
        }
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def should_update(self, agent_name: str, step: int) -> bool:
        """Check if agent should be updated at current step."""
        
        if agent_name == 'meta_controller':
            return step % self.config.meta_update_freq == 0
        elif agent_name in self.specialists:
            return step % self.config.specialist_update_freq == 0
        elif agent_name == 'joint':
            return step % self.config.joint_update_freq == 0
        else:
            return False
    
    def coordinate_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, torch.Tensor]:
        """
        Coordinate gradients between meta-controller and specialists.
        
        Args:
            gradients: Dictionary of gradients for each agent
            step: Current training step
            
        Returns:
            Coordinated gradients
        """
        
        coordinated_gradients = {}
        
        # Process meta-controller gradients
        if 'meta_controller' in gradients:
            meta_gradients = self._process_meta_gradients(gradients['meta_controller'], step)
            coordinated_gradients['meta_controller'] = meta_gradients
        
        # Process specialist gradients
        for specialist_name in self.specialists.keys():
            if specialist_name in gradients:
                specialist_gradients = self._process_specialist_gradients(
                    gradients[specialist_name], specialist_name, step
                )
                coordinated_gradients[specialist_name] = specialist_gradients
        
        # Joint gradient coordination
        if len(coordinated_gradients) > 1:
            coordinated_gradients = self._apply_joint_coordination(coordinated_gradients, step)
        
        # Update coordination metrics
        self._update_coordination_metrics(coordinated_gradients, step)
        
        return coordinated_gradients
    
    def _process_meta_gradients(
        self,
        gradients: torch.Tensor,
        step: int
    ) -> torch.Tensor:
        """Process meta-controller gradients."""
        
        # Clip gradients
        if self.config.gradient_clipping > 0:
            gradients = torch.nn.utils.clip_grad_norm_(
                gradients, self.config.gradient_clipping
            )
        
        # Scale gradients based on performance
        if self.config.gradient_scaling:
            performance_scale = self._get_performance_scale('meta_controller')
            gradients = gradients * performance_scale
        
        # Store gradient history
        self.gradient_history['meta_controller'].append(gradients.clone().detach())
        self.gradient_norms['meta_controller'].append(torch.norm(gradients).item())
        
        # Update learning rate
        if self.config.meta_lr_schedule:
            self._update_learning_rate('meta_controller', step)
        
        return gradients
    
    def _process_specialist_gradients(
        self,
        gradients: torch.Tensor,
        specialist_name: str,
        step: int
    ) -> torch.Tensor:
        """Process specialist gradients."""
        
        # Clip gradients
        if self.config.gradient_clipping > 0:
            gradients = torch.nn.utils.clip_grad_norm_(
                gradients, self.config.gradient_clipping
            )
        
        # Scale gradients based on performance
        if self.config.gradient_scaling:
            performance_scale = self._get_performance_scale(specialist_name)
            gradients = gradients * performance_scale
        
        # Store gradient history
        self.gradient_history[specialist_name].append(gradients.clone().detach())
        self.gradient_norms[specialist_name].append(torch.norm(gradients).item())
        
        # Update learning rate
        if self.config.specialist_lr_schedule:
            self._update_learning_rate(specialist_name, step)
        
        return gradients
    
    def _apply_joint_coordination(
        self,
        gradients: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, torch.Tensor]:
        """Apply joint coordination to gradients."""
        
        coordinated_gradients = gradients.copy()
        
        # Calculate gradient alignment
        gradient_alignment = self._calculate_gradient_alignment(gradients)
        self.coordination_metrics['gradient_alignment'].append(gradient_alignment)
        
        # Apply coordination weighting
        for agent_name, agent_gradients in coordinated_gradients.items():
            # Individual component
            individual_component = agent_gradients * self.config.individual_weight
            
            # Coordination component
            coordination_component = self._calculate_coordination_component(
                agent_gradients, gradients, agent_name
            )
            coordination_component = coordination_component * self.config.coordination_weight
            
            # Stability component
            stability_component = self._calculate_stability_component(
                agent_gradients, agent_name
            )
            stability_component = stability_component * self.config.stability_weight
            
            # Combine components
            coordinated_gradients[agent_name] = (
                individual_component + coordination_component + stability_component
            )
        
        return coordinated_gradients
    
    def _calculate_gradient_alignment(self, gradients: Dict[str, torch.Tensor]) -> float:
        """Calculate alignment between gradients."""
        
        if len(gradients) < 2:
            return 1.0
        
        # Flatten gradients
        flattened_gradients = []
        for agent_gradients in gradients.values():
            flattened_gradients.append(agent_gradients.flatten())
        
        # Calculate pairwise alignments
        alignments = []
        for i in range(len(flattened_gradients)):
            for j in range(i+1, len(flattened_gradients)):
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    flattened_gradients[i], flattened_gradients[j], dim=0
                )
                alignments.append(cos_sim.item())
        
        return np.mean(alignments) if alignments else 1.0
    
    def _calculate_coordination_component(
        self,
        agent_gradients: torch.Tensor,
        all_gradients: Dict[str, torch.Tensor],
        agent_name: str
    ) -> torch.Tensor:
        """Calculate coordination component for agent gradients."""
        
        # Average gradients from other agents
        other_gradients = []
        for name, gradients in all_gradients.items():
            if name != agent_name:
                other_gradients.append(gradients)
        
        if not other_gradients:
            return torch.zeros_like(agent_gradients)
        
        # Calculate average of other gradients
        avg_other_gradients = torch.stack(other_gradients).mean(dim=0)
        
        # Coordination component is the difference between agent and average
        coordination_component = avg_other_gradients - agent_gradients
        
        return coordination_component
    
    def _calculate_stability_component(
        self,
        agent_gradients: torch.Tensor,
        agent_name: str
    ) -> torch.Tensor:
        """Calculate stability component for agent gradients."""
        
        if len(self.gradient_history[agent_name]) < 2:
            return torch.zeros_like(agent_gradients)
        
        # Get recent gradients
        recent_gradients = list(self.gradient_history[agent_name])[-5:]
        
        # Calculate gradient momentum
        gradient_momentum = torch.stack(recent_gradients).mean(dim=0)
        
        # Stability component encourages consistency with recent gradients
        stability_component = gradient_momentum - agent_gradients
        
        return stability_component
    
    def _get_performance_scale(self, agent_name: str) -> float:
        """Get performance-based gradient scaling factor."""
        
        if len(self.performance_history[agent_name]) < 2:
            return 1.0
        
        # Calculate recent performance
        recent_performance = np.mean(self.performance_history[agent_name][-10:])
        
        # Scale based on performance (better performance = higher scale)
        if recent_performance > 0:
            return min(2.0, 1.0 + recent_performance)
        else:
            return max(0.1, 1.0 + recent_performance)
    
    def _update_learning_rate(self, agent_name: str, step: int) -> None:
        """Update learning rate for agent based on performance."""
        
        if len(self.performance_history[agent_name]) < 10:
            return
        
        # Calculate performance trend
        recent_performance = np.mean(self.performance_history[agent_name][-10:])
        older_performance = np.mean(self.performance_history[agent_name][-20:-10])
        
        performance_trend = recent_performance - older_performance
        
        # Adjust learning rate based on trend
        current_lr = self.learning_rates[agent_name]
        
        if performance_trend > 0.01:  # Improving
            new_lr = min(current_lr * 1.1, 0.001)  # Increase LR
        elif performance_trend < -0.01:  # Declining
            new_lr = max(current_lr * 0.9, 0.0001)  # Decrease LR
        else:
            new_lr = current_lr  # Keep same
        
        self.learning_rates[agent_name] = new_lr
    
    def _update_coordination_metrics(
        self,
        gradients: Dict[str, torch.Tensor],
        step: int
    ) -> None:
        """Update coordination metrics."""
        
        # Gradient alignment
        gradient_alignment = self._calculate_gradient_alignment(gradients)
        self.coordination_metrics['gradient_alignment'].append(gradient_alignment)
        
        # Performance correlation
        if len(self.performance_history) > 1:
            performance_correlation = self._calculate_performance_correlation()
            self.coordination_metrics['performance_correlation'].append(performance_correlation)
        
        # Update synchronization
        update_sync = self._calculate_update_synchronization(step)
        self.coordination_metrics['update_synchronization'].append(update_sync)
        
        # Stability metric
        stability_metric = self._calculate_stability_metric()
        self.coordination_metrics['stability_metric'].append(stability_metric)
    
    def _calculate_performance_correlation(self) -> float:
        """Calculate correlation between agent performances."""
        
        if len(self.performance_history) < 2:
            return 0.0
        
        # Get recent performance for all agents
        recent_performance = {}
        for agent_name, performance_history in self.performance_history.items():
            if len(performance_history) > 0:
                recent_performance[agent_name] = performance_history[-10:]
        
        if len(recent_performance) < 2:
            return 0.0
        
        # Calculate pairwise correlations
        correlations = []
        agent_names = list(recent_performance.keys())
        
        for i in range(len(agent_names)):
            for j in range(i+1, len(agent_names)):
                perf1 = recent_performance[agent_names[i]]
                perf2 = recent_performance[agent_names[j]]
                
                if len(perf1) == len(perf2) and len(perf1) > 1:
                    corr = np.corrcoef(perf1, perf2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_update_synchronization(self, step: int) -> float:
        """Calculate synchronization of updates."""
        
        # Check if all agents are being updated at similar frequencies
        update_frequencies = []
        
        for agent_name in ['meta_controller'] + list(self.specialists.keys()):
            if self.should_update(agent_name, step):
                update_frequencies.append(1.0)
            else:
                update_frequencies.append(0.0)
        
        # Synchronization is higher when more agents update together
        synchronization = np.mean(update_frequencies)
        
        return synchronization
    
    def _calculate_stability_metric(self) -> float:
        """Calculate overall stability metric."""
        
        stability_metrics = []
        
        # Gradient norm stability
        for agent_name in ['meta_controller'] + list(self.specialists.keys()):
            if len(self.gradient_norms[agent_name]) > 1:
                recent_norms = self.gradient_norms[agent_name][-10:]
                norm_stability = 1.0 - (np.std(recent_norms) / (np.mean(recent_norms) + 1e-8))
                stability_metrics.append(max(0, norm_stability))
        
        # Performance stability
        for agent_name, performance_history in self.performance_history.items():
            if len(performance_history) > 1:
                recent_performance = performance_history[-10:]
                perf_stability = 1.0 - (np.std(recent_performance) / (np.mean(recent_performance) + 1e-8))
                stability_metrics.append(max(0, perf_stability))
        
        return np.mean(stability_metrics) if stability_metrics else 0.0
    
    def update_performance(self, agent_name: str, performance: float) -> None:
        """Update performance history for agent."""
        
        if agent_name in self.performance_history:
            self.performance_history[agent_name].append(performance)
            
            # Keep only recent history
            if len(self.performance_history[agent_name]) > 1000:
                self.performance_history[agent_name].pop(0)
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get comprehensive coordination statistics."""
        
        stats = {
            'gradient_norms': {name: norms.copy() for name, norms in self.gradient_norms.items()},
            'performance_history': {name: perf.copy() for name, perf in self.performance_history.items()},
            'learning_rates': self.learning_rates.copy(),
            'update_counters': self.update_counters.copy(),
            'coordination_metrics': {name: metrics.copy() for name, metrics in self.coordination_metrics.items()},
            'current_gradient_alignment': self.coordination_metrics['gradient_alignment'][-1] if self.coordination_metrics['gradient_alignment'] else 0.0,
            'current_performance_correlation': self.coordination_metrics['performance_correlation'][-1] if self.coordination_metrics['performance_correlation'] else 0.0,
            'current_stability_metric': self.coordination_metrics['stability_metric'][-1] if self.coordination_metrics['stability_metric'] else 0.0
        }
        
        return stats
    
    def reset(self) -> None:
        """Reset coordination system state."""
        
        # Clear histories
        for agent_name in ['meta_controller'] + list(self.specialists.keys()):
            self.gradient_history[agent_name].clear()
            self.gradient_norms[agent_name].clear()
            self.performance_history[agent_name].clear()
        
        # Reset counters
        self.update_counters = {
            'meta_controller': 0,
            **{name: 0 for name in self.specialists.keys()},
            'joint': 0
        }
        
        # Clear coordination metrics
        for metrics in self.coordination_metrics.values():
            metrics.clear()
        
        self.logger.info("Gradient coordination system reset")
