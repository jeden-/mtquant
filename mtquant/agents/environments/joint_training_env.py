"""
Joint Training Environment for Phase 3

This environment coordinates training between meta-controller and specialists
in a joint fine-tuning phase where all agents learn together.
"""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .hierarchical_env import BaseHierarchicalEnv, EnvironmentConfig
from ..hierarchical.meta_controller import MetaController
from ..hierarchical.base_specialist import BaseSpecialist
from ..hierarchical.communication import AllocationMessage, PerformanceReport, CoordinationSignal
from ...risk_management.portfolio_risk_manager import PortfolioRiskManager
from ...risk_management.position_sizer import PositionSizer
from ...mcp_integration.models.order import Order
from ...mcp_integration.models.position import Position


@dataclass
class JointTrainingConfig:
    """Configuration for joint training environment."""
    
    # Environment settings
    initial_capital: float = 100000.0
    transaction_cost: float = 0.003
    max_position_size: float = 0.1
    
    # Risk management
    max_portfolio_var: float = 0.02
    max_correlation_exposure: float = 0.7
    max_sector_allocation: float = 0.4
    
    # Joint training parameters
    meta_update_freq: int = 1  # Update meta every step
    specialist_update_freq: int = 5  # Update specialists every 5 steps
    coordination_reward_weight: float = 0.5
    individual_reward_weight: float = 0.5
    
    # Curriculum learning
    curriculum_enabled: bool = True
    difficulty_progression: List[str] = None  # ['easy', 'medium', 'hard']
    scenario_weights: Dict[str, float] = None
    
    # Training
    episode_length: int = 1000
    warmup_steps: int = 50
    
    def __post_init__(self):
        if self.difficulty_progression is None:
            self.difficulty_progression = ['easy', 'medium', 'hard']
        if self.scenario_weights is None:
            self.scenario_weights = {'easy': 0.4, 'medium': 0.4, 'hard': 0.2}


class JointTrainingEnv(BaseHierarchicalEnv):
    """
    Joint training environment for Phase 3.
    
    Coordinates training between:
    - Meta-controller (portfolio-level decisions)
    - Specialists (domain-specific decisions)
    - Full hierarchical system integration
    """
    
    def __init__(
        self,
        config: JointTrainingConfig,
        market_data: Dict[str, pd.DataFrame],
        meta_controller: MetaController,
        specialists: Dict[str, BaseSpecialist],
        communication_hub: Optional[Any] = None,
        portfolio_risk_manager: Optional[PortfolioRiskManager] = None
    ):
        # Set specialists BEFORE calling super().__init__() so _setup_spaces() can access them
        self.specialists = specialists
        self.meta_controller = meta_controller
        self.joint_config = config
        
        # Convert config to EnvironmentConfig for base class
        env_config = EnvironmentConfig(
            instruments=list(market_data.keys()),
            timeframe="1H",
            lookback_window=100,
            initial_capital=config.initial_capital,
            transaction_cost=config.transaction_cost,
            max_position_size=config.max_position_size,
            max_portfolio_var=config.max_portfolio_var,
            max_correlation_exposure=config.max_correlation_exposure,
            stop_loss_pct=0.02,
            risk_penalty_weight=2.0,
            transaction_cost_weight=1.0,
            diversification_bonus_weight=0.5,
            episode_length=config.episode_length,
            warmup_steps=config.warmup_steps
        )
        
        super().__init__(env_config, market_data, communication_hub, portfolio_risk_manager)
        
        # Joint training state
        self.meta_actions_history: List[np.ndarray] = []
        self.specialist_actions_history: Dict[str, List[np.ndarray]] = {
            name: [] for name in specialists.keys()
        }
        self.coordination_rewards: List[float] = []
        self.individual_rewards: Dict[str, List[float]] = {
            name: [] for name in specialists.keys()
        }
        
        # Curriculum learning
        self.current_difficulty = 'easy'
        self.difficulty_progress = 0.0
        self.scenario_history: List[str] = []
        
        # Performance tracking
        self.joint_performance_history: List[float] = []
        self.coordination_metrics: Dict[str, List[float]] = {
            'allocation_efficiency': [],
            'specialist_synchronization': [],
            'risk_coordination': [],
            'performance_correlation': []
        }
        
        # Thread pool for parallel specialist processing
        self.executor = ThreadPoolExecutor(max_workers=max(1, len(specialists)))
        
    def _setup_spaces(self) -> None:
        """Setup action and observation spaces for joint training."""
        
        # Action space: combined meta-controller + specialist actions
        # Meta-controller: [allocation_forex, allocation_commodities, allocation_equity, risk_appetite]
        # Specialists: [signal_per_instrument] for each specialist
        meta_action_dim = 4
        specialist_action_dims = []
        
        for specialist_name, specialist in self.specialists.items():
            instruments = specialist.get_instruments()
            specialist_action_dims.append(len(instruments))
        
        total_action_dim = meta_action_dim + sum(specialist_action_dims)
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_action_dim,),
            dtype=np.float32
        )
        
        # Observation space: combined portfolio + specialist + coordination state
        portfolio_state_dim = 74  # From meta_controller.py
        specialist_state_dim = len(self.specialists) * 20  # 20 features per specialist
        coordination_state_dim = 15  # Coordination metrics
        curriculum_state_dim = 5  # Curriculum learning state
        
        total_obs_dim = (portfolio_state_dim + specialist_state_dim + 
                        coordination_state_dim + curriculum_state_dim)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get observation for joint training."""
        
        # 1. Portfolio state (74 dimensions)
        portfolio_state = self._get_portfolio_state_vector()
        
        # 2. Specialist states
        specialist_states = self._get_specialist_states()
        
        # 3. Coordination state
        coordination_state = self._get_coordination_state()
        
        # 4. Curriculum state
        curriculum_state = self._get_curriculum_state()
        
        # Combine all observations
        observation = np.concatenate([
            portfolio_state,
            specialist_states,
            coordination_state,
            curriculum_state
        ])
        
        return observation.astype(np.float32)
    
    def _get_portfolio_state_vector(self) -> np.ndarray:
        """Extract 74-dimensional portfolio state vector."""
        
        # Get portfolio state
        portfolio_state = self.get_portfolio_state()
        
        # Initialize state vector
        state_vector = np.zeros(74)
        
        # Portfolio metrics (0-9)
        state_vector[0] = portfolio_state['portfolio_value'] / self.config.initial_capital  # Normalized value
        state_vector[1] = portfolio_state['cash'] / self.config.initial_capital  # Cash ratio
        state_vector[2] = portfolio_state['total_exposure'] / self.config.initial_capital  # Exposure ratio
        state_vector[3] = portfolio_state['num_instruments'] / len(self.config.instruments)  # Diversification
        
        # Calculate portfolio returns and volatility
        if len(self.joint_performance_history) > 1:
            returns = np.diff(self.joint_performance_history) / self.joint_performance_history[:-1]
            state_vector[4] = np.mean(returns)  # Mean return
            state_vector[5] = np.std(returns)  # Volatility
            state_vector[6] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0  # Sharpe ratio
        else:
            state_vector[4:7] = 0
        
        # Risk metrics (7-9)
        try:
            portfolio_positions = list(self.positions.values())
            if portfolio_positions:
                var_result = self.portfolio_risk_manager.calculate_var(
                    portfolio_positions,
                    self.portfolio_value,
                    confidence_level=0.95
                )
                state_vector[7] = var_result.var_pct  # VaR percentage
                state_vector[8] = var_result.var_excess  # VaR excess
            else:
                state_vector[7:9] = 0
        except Exception:
            state_vector[7:9] = 0
        
        state_vector[9] = len(self.trade_history) / self.current_step if self.current_step > 0 else 0  # Trade frequency
        
        # Position-level metrics (10-49) - 4 metrics per instrument (10 instruments max)
        instrument_metrics = np.zeros(40)  # 4 * 10 instruments
        for i, instrument in enumerate(self.config.instruments[:10]):  # Limit to 10 instruments
            if instrument in self.positions:
                position = self.positions[instrument]
                instrument_metrics[i*4] = position.quantity / self.config.max_position_size  # Position size
                instrument_metrics[i*4+1] = position.unrealized_pnl / self.portfolio_value  # P&L contribution
                instrument_metrics[i*4+2] = (position.current_price - position.entry_price) / position.entry_price  # Price change
                instrument_metrics[i*4+3] = 1.0  # Position exists
            else:
                instrument_metrics[i*4:i*4+4] = 0
        
        state_vector[10:50] = instrument_metrics
        
        # Market data features (50-73) - 6 features per instrument (4 instruments max)
        market_features = np.zeros(24)  # 6 * 4 instruments
        for i, instrument in enumerate(self.config.instruments[:4]):  # Limit to 4 instruments
            try:
                market_state = self.get_market_state(instrument)
                market_features[i*6] = market_state['volatility']  # Volatility
                market_features[i*6+1] = market_state['volume'] / np.mean(self.market_data[instrument]['volume'])  # Volume ratio
                market_features[i*6+2] = market_state['returns'][-1] if len(market_state['returns']) > 0 else 0  # Latest return
                market_features[i*6+3] = np.mean(market_state['returns']) if len(market_state['returns']) > 0 else 0  # Mean return
                market_features[i*6+4] = np.std(market_state['returns']) if len(market_state['returns']) > 0 else 0  # Return std
                market_features[i*6+5] = 1.0  # Market data available
            except Exception:
                market_features[i*6:i*6+6] = 0
        
        state_vector[50:74] = market_features
        
        return state_vector
    
    def _get_specialist_states(self) -> np.ndarray:
        """Get specialist states for joint training."""
        
        specialist_states = np.zeros(len(self.specialists) * 20)
        
        for i, (name, specialist) in enumerate(self.specialists.items()):
            base_idx = i * 20
            
            # Get specialist's instruments
            instruments = specialist.get_instruments()
            
            # Calculate specialist performance
            specialist_pnl = 0.0
            specialist_trades = 0
            specialist_confidence = 0.5
            
            for instrument in instruments:
                if instrument in self.positions:
                    position = self.positions[instrument]
                    specialist_pnl += position.unrealized_pnl
                    specialist_trades += 1
            
            # Get specialist confidence
            try:
                if instruments:
                    market_state = self.get_market_state(instruments[0])
                    confidence_result = specialist.calculate_confidence(market_state)
                    # Ensure we get a float value, not a Mock
                    if hasattr(confidence_result, '_mock_name'):
                        specialist_confidence = 0.5  # Default if Mock
                    else:
                        specialist_confidence = float(confidence_result)
            except Exception:
                specialist_confidence = 0.5
            
            # Fill specialist state (20 features per specialist)
            specialist_states[base_idx] = specialist_pnl / max(self.portfolio_value, 1.0)  # P&L contribution
            specialist_states[base_idx+1] = specialist_trades / len(instruments)  # Trade ratio
            specialist_states[base_idx+2] = specialist_confidence  # Confidence
            specialist_states[base_idx+3] = len(self.specialist_actions_history[name])  # Action history length
            
            # Recent performance
            if name in self.individual_rewards and self.individual_rewards[name]:
                recent_rewards = self.individual_rewards[name][-10:]
                specialist_states[base_idx+4] = np.mean(recent_rewards)  # Mean recent reward
                specialist_states[base_idx+5] = np.std(recent_rewards)  # Reward volatility
                specialist_states[base_idx+6] = 1.0 if recent_rewards[-1] > 0 else 0.0  # Positive reward flag
            else:
                specialist_states[base_idx+4:base_idx+7] = 0
            
            # Action consistency
            if name in self.specialist_actions_history and self.specialist_actions_history[name]:
                recent_actions = self.specialist_actions_history[name][-10:]
                if recent_actions:
                    actions_array = np.array(recent_actions)
                    specialist_states[base_idx+7] = np.std(actions_array)  # Action volatility
                    specialist_states[base_idx+8] = np.mean(np.abs(actions_array))  # Action magnitude
                else:
                    specialist_states[base_idx+7:base_idx+9] = 0
            else:
                specialist_states[base_idx+7:base_idx+9] = 0
            
            # Coordination metrics
            specialist_states[base_idx+9] = 0.0  # Reserved for coordination
            specialist_states[base_idx+10] = 0.0  # Reserved for coordination
            
            # Fill remaining features with zeros
            specialist_states[base_idx+11:base_idx+20] = 0
        
        return specialist_states
    
    def _get_coordination_state(self) -> np.ndarray:
        """Get coordination state between meta-controller and specialists."""
        
        coordination_state = np.zeros(15)
        
        # 1. Allocation efficiency (0-2)
        if len(self.meta_actions_history) > 0:
            recent_allocations = [action[:3] for action in self.meta_actions_history[-10:]]
            if recent_allocations:
                allocations_array = np.array(recent_allocations)
                # Calculate allocation entropy (diversification)
                allocation_entropy = -np.sum(allocations_array * np.log(allocations_array + 1e-8), axis=1)
                max_entropy = np.log(3)  # 3 specialists
                coordination_state[0] = np.mean(allocation_entropy / max_entropy)  # Allocation efficiency
                coordination_state[1] = np.std(allocation_entropy / max_entropy)  # Allocation stability
                coordination_state[2] = np.mean(allocations_array, axis=0)[0]  # Forex allocation
        
        # 2. Specialist synchronization (3-5)
        if len(self.coordination_rewards) > 0:
            coordination_state[3] = np.mean(self.coordination_rewards[-10:])  # Mean coordination reward
            coordination_state[4] = np.std(self.coordination_rewards[-10:])  # Coordination stability
            coordination_state[5] = 1.0 if self.coordination_rewards[-1] > 0 else 0.0  # Positive coordination
        
        # 3. Risk coordination (6-8)
        try:
            portfolio_positions = list(self.positions.values())
            if portfolio_positions:
                var_result = self.portfolio_risk_manager.calculate_var(
                    portfolio_positions,
                    self.portfolio_value,
                    confidence_level=0.95
                )
                coordination_state[6] = var_result.var_pct  # VaR level
                coordination_state[7] = 1.0 if var_result.var_pct <= 0.02 else 0.0  # VaR compliance
                coordination_state[8] = len(portfolio_positions) / len(self.config.instruments)  # Position coverage
        except Exception:
            coordination_state[6:9] = 0
        
        # 4. Performance correlation (9-11)
        if len(self.individual_rewards) > 1:
            specialist_rewards = []
            for name in self.specialists.keys():
                if name in self.individual_rewards and self.individual_rewards[name]:
                    specialist_rewards.append(self.individual_rewards[name][-10:])
            
            if len(specialist_rewards) > 1:
                # Calculate pairwise correlations
                correlations = []
                for i in range(len(specialist_rewards)):
                    for j in range(i+1, len(specialist_rewards)):
                        if len(specialist_rewards[i]) == len(specialist_rewards[j]) and len(specialist_rewards[i]) > 1:
                            corr = np.corrcoef(specialist_rewards[i], specialist_rewards[j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                
                if correlations:
                    coordination_state[9] = np.mean(correlations)  # Mean correlation
                    coordination_state[10] = np.std(correlations)  # Correlation stability
                    coordination_state[11] = 1.0 if np.mean(correlations) < 0.7 else 0.0  # Good diversification
        
        # 5. Joint performance (12-14)
        if len(self.joint_performance_history) > 1:
            recent_performance = self.joint_performance_history[-10:]
            coordination_state[12] = np.mean(recent_performance)  # Mean joint performance
            coordination_state[13] = np.std(recent_performance)  # Performance stability
            coordination_state[14] = 1.0 if recent_performance[-1] > 0 else 0.0  # Positive performance
        
        return coordination_state
    
    def _get_curriculum_state(self) -> np.ndarray:
        """Get curriculum learning state."""
        
        curriculum_state = np.zeros(5)
        
        # 1. Current difficulty (0)
        difficulty_map = {'easy': 0.0, 'medium': 0.5, 'hard': 1.0}
        curriculum_state[0] = difficulty_map.get(self.current_difficulty, 0.0)
        
        # 2. Difficulty progress (1)
        curriculum_state[1] = self.difficulty_progress
        
        # 3. Scenario history (2)
        if self.scenario_history:
            recent_scenarios = self.scenario_history[-10:]
            curriculum_state[2] = len(set(recent_scenarios)) / len(recent_scenarios)  # Scenario diversity
        
        # 4. Performance vs difficulty (3)
        if len(self.joint_performance_history) > 0:
            recent_performance = np.mean(self.joint_performance_history[-10:])
            # Normalize performance to 0-1 range
            curriculum_state[3] = max(0, min(1, (recent_performance + 1) / 2))
        
        # 5. Curriculum readiness (4)
        # Ready for next difficulty if performance is good and stable
        if (len(self.joint_performance_history) > 20 and 
            np.mean(self.joint_performance_history[-10:]) > 0 and
            np.std(self.joint_performance_history[-10:]) < 0.1):
            curriculum_state[4] = 1.0
        else:
            curriculum_state[4] = 0.0
        
        return curriculum_state
    
    def _execute_action(self, action: np.ndarray) -> List[Order]:
        """Execute joint action (meta-controller + specialists)."""
        
        # Parse action
        meta_action = action[:4]  # Meta-controller action
        specialist_actions = action[4:]  # Specialist actions
        
        # Store actions for history
        self.meta_actions_history.append(meta_action.copy())
        
        # Parse specialist actions
        action_idx = 0
        for specialist_name, specialist in self.specialists.items():
            instruments = specialist.get_instruments()
            specialist_action = specialist_actions[action_idx:action_idx + len(instruments)]
            self.specialist_actions_history[specialist_name].append(specialist_action.copy())
            action_idx += len(instruments)
        
        # Execute meta-controller action
        allocation_raw = meta_action[:3]
        risk_appetite_raw = meta_action[3]
        
        # Normalize allocation using softmax
        allocation = torch.softmax(torch.tensor(allocation_raw), dim=0).numpy()
        
        # Normalize risk appetite using sigmoid
        risk_appetite = 1 / (1 + np.exp(-risk_appetite_raw))
        
        # Create allocation message
        specialist_names = list(self.specialists.keys())
        allocation_message = AllocationMessage(
            message_id=f"joint_alloc_{self.current_step}",
            sender_id="meta_controller",
            timestamp=pd.Timestamp.now(),
            allocations={
                specialist_names[i]: allocation[i] for i in range(len(specialist_names))
            },
            risk_appetite=risk_appetite,
            total_capital=self.portfolio_value
        )
        
        # Send to communication hub
        self.communication_hub.send_message(allocation_message)
        
        # Execute specialist actions
        orders = []
        action_idx = 0
        
        for specialist_name, specialist in self.specialists.items():
            instruments = specialist.get_instruments()
            specialist_action = specialist_actions[action_idx:action_idx + len(instruments)]
            
            # Convert specialist action to orders
            for i, instrument in enumerate(instruments):
                signal = specialist_action[i]
                
                if abs(signal) > 0.1:  # Threshold for trading
                    # Calculate position size based on signal strength and allocation
                    allocation_weight = allocation[list(specialist_names).index(specialist_name)]
                    position_size = self.position_sizer.calculate(
                        signal=signal * allocation_weight,
                        portfolio_equity=self.portfolio_value,
                        instrument_volatility=self._get_instrument_volatility(instrument),
                        method='volatility'
                    )
                    
                    # Create order
                    order = Order(
                        order_id=f"{instrument}_{self.current_step}",
                        agent_id=specialist_name,
                        symbol=instrument,
                        side='buy' if signal > 0 else 'sell',
                        order_type='market',
                        quantity=abs(position_size),
                        price=None,  # Market order
                        status='pending'
                    )
                    
                    orders.append(order)
            
            action_idx += len(instruments)
        
        return orders
    
    def _get_instrument_volatility(self, instrument: str) -> float:
        """Get volatility for position sizing."""
        try:
            market_state = self.get_market_state(instrument)
            returns = market_state['returns']
            if len(returns) > 0:
                return np.std(returns[-20:])  # 20-period volatility
            else:
                return 0.02  # Default 2% volatility
        except Exception:
            return 0.02
    
    def _calculate_reward(self, executed_orders: List[Order]) -> float:
        """Calculate joint reward for meta-controller and specialists."""
        
        if self.current_step == 0:
            return 0.0
        
        # Calculate individual rewards for each specialist
        individual_reward = 0.0
        specialist_rewards = {}
        
        for specialist_name, specialist in self.specialists.items():
            specialist_pnl = 0.0
            for instrument in specialist.get_instruments():
                if instrument in self.positions:
                    specialist_pnl += self.positions[instrument].unrealized_pnl
            
            # Calculate specialist reward
            specialist_reward = specialist_pnl / max(self.portfolio_value, 1.0)
            specialist_rewards[specialist_name] = specialist_reward
            
            # Update individual reward history
            if specialist_name not in self.individual_rewards:
                self.individual_rewards[specialist_name] = []
            self.individual_rewards[specialist_name].append(specialist_reward)
            
            individual_reward += specialist_reward
        
        # Calculate coordination reward
        coordination_reward = self._calculate_coordination_reward(specialist_rewards)
        self.coordination_rewards.append(coordination_reward)
        
        # Calculate joint performance
        joint_performance = (individual_reward + coordination_reward) / 2
        self.joint_performance_history.append(joint_performance)
        
        # Combine individual and coordination rewards
        total_reward = (self.joint_config.individual_reward_weight * individual_reward + 
                       self.joint_config.coordination_reward_weight * coordination_reward)
        
        # Update curriculum learning
        self._update_curriculum_learning(joint_performance)
        
        return total_reward
    
    def _calculate_coordination_reward(self, specialist_rewards: Dict[str, float]) -> float:
        """Calculate coordination reward between meta-controller and specialists."""
        
        coordination_reward = 0.0
        
        # 1. Allocation efficiency
        if len(self.meta_actions_history) > 0:
            recent_allocation = self.meta_actions_history[-1][:3]
            allocation_entropy = -np.sum(recent_allocation * np.log(recent_allocation + 1e-8))
            max_entropy = np.log(3)
            allocation_efficiency = allocation_entropy / max_entropy
            coordination_reward += allocation_efficiency * 0.3
        
        # 2. Specialist synchronization
        if len(specialist_rewards) > 1:
            rewards_array = np.array(list(specialist_rewards.values()))
            # Reward for balanced performance (not all positive, not all negative)
            performance_balance = 1.0 - np.std(rewards_array)
            coordination_reward += performance_balance * 0.2
        
        # 3. Risk coordination
        try:
            portfolio_positions = list(self.positions.values())
            if portfolio_positions:
                var_result = self.portfolio_risk_manager.calculate_var(
                    portfolio_positions,
                    self.portfolio_value,
                    confidence_level=0.95
                )
                # Reward for VaR compliance
                if var_result.var_pct <= 0.02:
                    coordination_reward += 0.3
                else:
                    coordination_reward -= var_result.var_excess * 5.0
        except Exception:
            pass
        
        # 4. Performance correlation
        if len(self.individual_rewards) > 1:
            specialist_reward_histories = []
            for name in self.specialists.keys():
                if name in self.individual_rewards and len(self.individual_rewards[name]) > 1:
                    specialist_reward_histories.append(self.individual_rewards[name][-10:])
            
            if len(specialist_reward_histories) > 1:
                # Calculate average correlation
                correlations = []
                for i in range(len(specialist_reward_histories)):
                    for j in range(i+1, len(specialist_reward_histories)):
                        if len(specialist_reward_histories[i]) == len(specialist_reward_histories[j]):
                            corr = np.corrcoef(specialist_reward_histories[i], specialist_reward_histories[j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    # Reward for moderate correlation (good diversification)
                    if avg_correlation < 0.7:
                        coordination_reward += (0.7 - avg_correlation) * 0.2
        
        return coordination_reward
    
    def _update_curriculum_learning(self, joint_performance: float) -> None:
        """Update curriculum learning based on joint performance."""
        
        if not self.joint_config.curriculum_enabled:
            return
        
        # Update difficulty progress
        self.difficulty_progress = min(1.0, self.difficulty_progress + 0.001)
        
        # Check if ready for next difficulty
        if (len(self.joint_performance_history) > 50 and
            np.mean(self.joint_performance_history[-20:]) > 0 and
            np.std(self.joint_performance_history[-20:]) < 0.1):
            
            # Progress to next difficulty
            difficulty_index = self.joint_config.difficulty_progression.index(self.current_difficulty)
            if difficulty_index < len(self.joint_config.difficulty_progression) - 1:
                self.current_difficulty = self.joint_config.difficulty_progression[difficulty_index + 1]
                self.difficulty_progress = 0.0
                self.logger.info(f"Curriculum progression: {self.current_difficulty}")
        
        # Update scenario history
        self.scenario_history.append(self.current_difficulty)
        if len(self.scenario_history) > 100:
            self.scenario_history.pop(0)
    
    def get_joint_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive joint training statistics."""
        
        stats = {
            'joint_performance': self.joint_performance_history.copy(),
            'coordination_rewards': self.coordination_rewards.copy(),
            'individual_rewards': {name: rewards.copy() for name, rewards in self.individual_rewards.items()},
            'meta_actions': self.meta_actions_history.copy(),
            'specialist_actions': {name: actions.copy() for name, actions in self.specialist_actions_history.items()},
            'coordination_metrics': {name: metrics.copy() for name, metrics in self.coordination_metrics.items()},
            'curriculum_state': {
                'current_difficulty': self.current_difficulty,
                'difficulty_progress': self.difficulty_progress,
                'scenario_history': self.scenario_history.copy()
            }
        }
        
        return stats
    
    def close(self) -> None:
        """Clean up resources."""
        super().close()
        self.executor.shutdown(wait=True)
