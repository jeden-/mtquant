"""
Portfolio-Level Reward Functions for Meta-Controller Training

This module provides sophisticated reward functions for training the meta-controller
to optimize portfolio-level performance while managing risk and coordination.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from ...risk_management.portfolio_risk_manager import PortfolioRiskManager
from ...mcp_integration.models.position import Position


class RewardComponent(Enum):
    """Components of the portfolio reward function."""
    PORTFOLIO_RETURN = "portfolio_return"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    DIVERSIFICATION = "diversification"
    ALLOCATION_STABILITY = "allocation_stability"
    SPECIALIST_COORDINATION = "specialist_coordination"
    RISK_MANAGEMENT = "risk_management"
    TRANSACTION_COSTS = "transaction_costs"
    DRAWDOWN_PENALTY = "drawdown_penalty"


@dataclass
class RewardConfig:
    """Configuration for portfolio-level reward function."""
    
    # Component weights
    portfolio_return_weight: float = 1.0
    risk_adjusted_return_weight: float = 2.0
    diversification_weight: float = 0.5
    allocation_stability_weight: float = 0.3
    specialist_coordination_weight: float = 0.4
    risk_management_weight: float = 3.0
    transaction_cost_weight: float = 1.0
    drawdown_penalty_weight: float = 5.0
    
    # Risk parameters
    target_sharpe_ratio: float = 2.0
    max_drawdown_threshold: float = 0.15
    var_confidence_level: float = 0.95
    
    # Diversification parameters
    min_specialist_allocation: float = 0.1
    max_specialist_allocation: float = 0.7
    target_correlation: float = 0.3
    
    # Stability parameters
    allocation_change_threshold: float = 0.2
    max_allocation_volatility: float = 0.1
    
    # Coordination parameters
    performance_correlation_threshold: float = 0.8
    coordination_bonus_threshold: float = 0.6


class PortfolioRewardFunction:
    """
    Sophisticated portfolio-level reward function for meta-controller training.
    
    Combines multiple reward components:
    1. Portfolio return (base performance)
    2. Risk-adjusted return (Sharpe ratio)
    3. Diversification bonus (balanced allocation)
    4. Allocation stability (consistent decisions)
    5. Specialist coordination (team performance)
    6. Risk management (VaR compliance)
    7. Transaction costs (efficiency)
    8. Drawdown penalty (capital preservation)
    """
    
    def __init__(
        self,
        config: RewardConfig,
        portfolio_risk_manager: PortfolioRiskManager
    ):
        self.config = config
        self.portfolio_risk_manager = portfolio_risk_manager
        
        # Performance tracking
        self.portfolio_value_history: List[float] = []
        self.allocation_history: List[np.ndarray] = []
        self.specialist_performance_history: Dict[str, List[float]] = {}
        self.risk_metrics_history: List[Dict[str, float]] = []
        
        # Baseline comparison
        self.equal_weight_baseline: List[float] = []
        self.baseline_returns: List[float] = []
    
    def calculate_reward(
        self,
        portfolio_value: float,
        positions: List[Position],
        allocation: np.ndarray,
        specialist_performance: Dict[str, float],
        transaction_costs: float = 0.0,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive portfolio reward.
        
        Args:
            portfolio_value: Current portfolio value
            positions: List of current positions
            allocation: Allocation to specialists (normalized)
            specialist_performance: Performance metrics per specialist
            transaction_costs: Total transaction costs
            market_data: Optional market data for regime detection
            
        Returns:
            Tuple of (total_reward, component_breakdown)
        """
        
        # Update history
        self._update_history(portfolio_value, allocation, specialist_performance)
        
        # Calculate individual components
        components = {}
        
        # 1. Portfolio return
        components[RewardComponent.PORTFOLIO_RETURN] = self._calculate_portfolio_return_reward()
        
        # 2. Risk-adjusted return
        components[RewardComponent.RISK_ADJUSTED_RETURN] = self._calculate_risk_adjusted_return_reward()
        
        # 3. Diversification
        components[RewardComponent.DIVERSIFICATION] = self._calculate_diversification_reward(allocation)
        
        # 4. Allocation stability
        components[RewardComponent.ALLOCATION_STABILITY] = self._calculate_allocation_stability_reward()
        
        # 5. Specialist coordination
        components[RewardComponent.SPECIALIST_COORDINATION] = self._calculate_coordination_reward()
        
        # 6. Risk management
        components[RewardComponent.RISK_MANAGEMENT] = self._calculate_risk_management_reward(positions)
        
        # 7. Transaction costs
        components[RewardComponent.TRANSACTION_COSTS] = self._calculate_transaction_cost_penalty(transaction_costs)
        
        # 8. Drawdown penalty
        components[RewardComponent.DRAWDOWN_PENALTY] = self._calculate_drawdown_penalty()
        
        # Calculate weighted total reward
        total_reward = sum(
            self._get_component_weight(component) * reward
            for component, reward in components.items()
        )
        
        # Convert to dictionary for logging
        component_dict = {component.value: reward for component, reward in components.items()}
        
        return total_reward, component_dict
    
    def _update_history(
        self,
        portfolio_value: float,
        allocation: np.ndarray,
        specialist_performance: Dict[str, float]
    ) -> None:
        """Update performance history."""
        
        self.portfolio_value_history.append(portfolio_value)
        self.allocation_history.append(allocation.copy())
        
        # Update specialist performance history
        for specialist_name, performance in specialist_performance.items():
            if specialist_name not in self.specialist_performance_history:
                self.specialist_performance_history[specialist_name] = []
            self.specialist_performance_history[specialist_name].append(performance)
    
    def _calculate_portfolio_return_reward(self) -> float:
        """Calculate portfolio return reward."""
        
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        # Calculate portfolio return
        current_value = self.portfolio_value_history[-1]
        previous_value = self.portfolio_value_history[-2]
        portfolio_return = (current_value - previous_value) / previous_value
        
        return portfolio_return
    
    def _calculate_risk_adjusted_return_reward(self) -> float:
        """Calculate risk-adjusted return reward (Sharpe ratio)."""
        
        if len(self.portfolio_value_history) < 20:
            return 0.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(self.portfolio_value_history)):
            ret = (self.portfolio_value_history[i] - self.portfolio_value_history[i-1]) / self.portfolio_value_history[i-1]
            returns.append(ret)
        
        returns = np.array(returns)
        
        # Calculate Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            sharpe_ratio = mean_return / std_return
            # Reward for achieving target Sharpe ratio
            sharpe_reward = sharpe_ratio - self.config.target_sharpe_ratio
            return sharpe_reward
        else:
            return 0.0
    
    def _calculate_diversification_reward(self, allocation: np.ndarray) -> float:
        """Calculate diversification reward."""
        
        # Check allocation bounds
        min_allocation = np.min(allocation)
        max_allocation = np.max(allocation)
        
        # Penalty for extreme allocations
        bounds_penalty = 0.0
        if min_allocation < self.config.min_specialist_allocation:
            bounds_penalty += (self.config.min_specialist_allocation - min_allocation) * 2.0
        if max_allocation > self.config.max_specialist_allocation:
            bounds_penalty += (max_allocation - self.config.max_specialist_allocation) * 2.0
        
        # Entropy-based diversification bonus
        allocation_entropy = -np.sum(allocation * np.log(allocation + 1e-8))
        max_entropy = np.log(len(allocation))
        diversification_score = allocation_entropy / max_entropy
        
        # Combine bounds penalty and diversification bonus
        diversification_reward = diversification_score - bounds_penalty
        
        return diversification_reward
    
    def _calculate_allocation_stability_reward(self) -> float:
        """Calculate allocation stability reward."""
        
        if len(self.allocation_history) < 2:
            return 0.0
        
        # Calculate allocation change
        current_allocation = self.allocation_history[-1]
        previous_allocation = self.allocation_history[-2]
        allocation_change = np.linalg.norm(current_allocation - previous_allocation)
        
        # Reward for stability (inverse of change)
        stability_reward = 1.0 - min(allocation_change / self.config.allocation_change_threshold, 1.0)
        
        # Calculate allocation volatility over time
        if len(self.allocation_history) > 10:
            allocations_array = np.array(self.allocation_history[-10:])
            allocation_volatility = np.mean(np.std(allocations_array, axis=0))
            
            # Penalty for high volatility
            volatility_penalty = max(0, allocation_volatility - self.config.max_allocation_volatility)
            stability_reward -= volatility_penalty
        
        return stability_reward
    
    def _calculate_coordination_reward(self) -> float:
        """Calculate specialist coordination reward."""
        
        if len(self.specialist_performance_history) < 2:
            return 0.0
        
        # Calculate performance correlation between specialists
        specialist_names = list(self.specialist_performance_history.keys())
        if len(specialist_names) < 2:
            return 0.0
        
        # Get recent performance for each specialist
        recent_performance = {}
        for name in specialist_names:
            if len(self.specialist_performance_history[name]) > 0:
                recent_performance[name] = self.specialist_performance_history[name][-20:]  # Last 20 steps
        
        # Calculate pairwise correlations
        correlations = []
        for i, name1 in enumerate(specialist_names):
            for j, name2 in enumerate(specialist_names[i+1:], i+1):
                if name1 in recent_performance and name2 in recent_performance:
                    perf1 = recent_performance[name1]
                    perf2 = recent_performance[name2]
                    
                    if len(perf1) == len(perf2) and len(perf1) > 1:
                        corr = np.corrcoef(perf1, perf2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
        
        if correlations:
            avg_correlation = np.mean(correlations)
            
            # Reward for moderate correlation (not too high, not too low)
            target_corr = self.config.target_correlation
            correlation_reward = 1.0 - abs(avg_correlation - target_corr) / target_corr
            
            # Bonus for coordination (when all specialists perform well)
            all_positive = all(
                np.mean(self.specialist_performance_history[name][-10:]) > 0
                for name in specialist_names
                if len(self.specialist_performance_history[name]) > 0
            )
            
            if all_positive:
                correlation_reward += 0.2
            
            return correlation_reward
        else:
            return 0.0
    
    def _calculate_risk_management_reward(self, positions: List[Position]) -> float:
        """Calculate risk management reward."""
        
        if not positions:
            return 0.0
        
        try:
            # Calculate VaR
            var_result = self.portfolio_risk_manager.calculate_var(
                positions,
                self.portfolio_value_history[-1],
                confidence_level=self.config.var_confidence_level
            )
            
            # Reward for VaR compliance
            var_reward = 0.0
            if var_result.var_pct <= 0.02:  # Within 2% VaR limit
                var_reward = 1.0
            else:
                # Penalty for exceeding VaR
                var_reward = -var_result.var_excess * 10.0
            
            # Check correlation risk
            correlation_check = self.portfolio_risk_manager.check_correlation_risk(
                positions,
                np.eye(len(positions))  # Simplified correlation matrix
            )
            
            correlation_reward = 0.0
            if correlation_check[0]:  # Is safe
                correlation_reward = 1.0
            else:
                # Penalty for high correlation
                correlation_reward = -correlation_check[1] * 5.0
            
            # Check sector allocation
            sector_allocation = self.portfolio_risk_manager.calculate_sector_allocation(positions)
            sector_reward = 0.0
            
            for sector, allocation in sector_allocation.items():
                if allocation <= self.config.max_specialist_allocation:
                    sector_reward += 1.0
                else:
                    sector_reward -= (allocation - self.config.max_specialist_allocation) * 5.0
            
            # Combine risk components
            risk_reward = (var_reward + correlation_reward + sector_reward) / 3.0
            
            return risk_reward
            
        except Exception:
            return 0.0
    
    def _calculate_transaction_cost_penalty(self, transaction_costs: float) -> float:
        """Calculate transaction cost penalty."""
        
        if self.portfolio_value_history:
            cost_ratio = transaction_costs / self.portfolio_value_history[-1]
            return -cost_ratio * 100.0  # Scale up penalty
        else:
            return 0.0
    
    def _calculate_drawdown_penalty(self) -> float:
        """Calculate drawdown penalty."""
        
        if len(self.portfolio_value_history) < 20:
            return 0.0
        
        # Calculate current drawdown
        portfolio_values = np.array(self.portfolio_value_history)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        
        current_drawdown = drawdown[-1]
        
        # Penalty for exceeding drawdown threshold
        if current_drawdown > self.config.max_drawdown_threshold:
            drawdown_penalty = -(current_drawdown - self.config.max_drawdown_threshold) * 10.0
            return drawdown_penalty
        else:
            return 0.0
    
    def _get_component_weight(self, component: RewardComponent) -> float:
        """Get weight for reward component."""
        
        weight_map = {
            RewardComponent.PORTFOLIO_RETURN: self.config.portfolio_return_weight,
            RewardComponent.RISK_ADJUSTED_RETURN: self.config.risk_adjusted_return_weight,
            RewardComponent.DIVERSIFICATION: self.config.diversification_weight,
            RewardComponent.ALLOCATION_STABILITY: self.config.allocation_stability_weight,
            RewardComponent.SPECIALIST_COORDINATION: self.config.specialist_coordination_weight,
            RewardComponent.RISK_MANAGEMENT: self.config.risk_management_weight,
            RewardComponent.TRANSACTION_COSTS: self.config.transaction_cost_weight,
            RewardComponent.DRAWDOWN_PENALTY: self.config.drawdown_penalty_weight
        }
        
        return weight_map.get(component, 1.0)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics."""
        
        metrics = {}
        
        if len(self.portfolio_value_history) > 1:
            # Portfolio metrics
            returns = []
            for i in range(1, len(self.portfolio_value_history)):
                ret = (self.portfolio_value_history[i] - self.portfolio_value_history[i-1]) / self.portfolio_value_history[i-1]
                returns.append(ret)
            
            returns = np.array(returns)
            
            metrics['total_return'] = (self.portfolio_value_history[-1] - self.portfolio_value_history[0]) / self.portfolio_value_history[0]
            metrics['mean_return'] = np.mean(returns)
            metrics['volatility'] = np.std(returns)
            metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Drawdown metrics
            portfolio_values = np.array(self.portfolio_value_history)
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            metrics['max_drawdown'] = np.max(drawdown)
            metrics['current_drawdown'] = drawdown[-1]
        
        # Allocation metrics
        if self.allocation_history:
            allocations_array = np.array(self.allocation_history)
            metrics['allocation_volatility'] = np.mean(np.std(allocations_array, axis=0))
            metrics['allocation_entropy'] = np.mean([-np.sum(alloc * np.log(alloc + 1e-8)) for alloc in allocations_array])
        
        # Specialist performance metrics
        for name, performance_history in self.specialist_performance_history.items():
            if performance_history:
                metrics[f'{name}_mean_performance'] = np.mean(performance_history)
                metrics[f'{name}_sharpe'] = np.mean(performance_history) / np.std(performance_history) if np.std(performance_history) > 0 else 0
        
        return metrics
    
    def reset(self) -> None:
        """Reset reward function state."""
        
        self.portfolio_value_history.clear()
        self.allocation_history.clear()
        self.specialist_performance_history.clear()
        self.risk_metrics_history.clear()
        self.equal_weight_baseline.clear()
        self.baseline_returns.clear()
