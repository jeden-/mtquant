"""
Meta-Controller Training Environment for Phase 2

This environment trains the meta-controller to make portfolio-level decisions
by observing specialist performance and allocating capital accordingly.
"""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from datetime import datetime, timedelta

from .hierarchical_env import BaseHierarchicalEnv, EnvironmentConfig
from ..hierarchical.meta_controller import MetaController
from ..hierarchical.base_specialist import BaseSpecialist
from ..hierarchical.communication import AllocationMessage, PerformanceReport
from ...risk_management.portfolio_risk_manager import PortfolioRiskManager
from ...risk_management.position_sizer import PositionSizer
from ...mcp_integration.models.order import Order
from ...mcp_integration.models.position import Position


@dataclass
class MetaControllerConfig:
    """Configuration for meta-controller training environment."""
    
    # Environment settings
    initial_capital: float = 100000.0
    transaction_cost: float = 0.003
    max_position_size: float = 0.1
    
    # Risk management
    max_portfolio_var: float = 0.02
    max_correlation_exposure: float = 0.7
    max_sector_allocation: float = 0.4
    
    # Reward shaping
    portfolio_return_weight: float = 1.0
    risk_penalty_weight: float = 2.0
    diversification_bonus_weight: float = 0.5
    allocation_stability_weight: float = 0.3
    
    # Training
    episode_length: int = 1000
    warmup_steps: int = 50
    
    # Meta-controller specific
    allocation_update_freq: int = 10  # Update allocations every N steps
    performance_lookback: int = 20  # Lookback for specialist performance
    market_regime_detection: bool = True


class MetaControllerTrainingEnv(BaseHierarchicalEnv):
    """
    Training environment for meta-controller (Phase 2).
    
    The meta-controller learns to:
    - Allocate capital to specialists based on performance
    - Adjust risk appetite based on market conditions
    - Coordinate specialist actions for portfolio optimization
    """
    
    def __init__(
        self,
        config: MetaControllerConfig,
        market_data: Dict[str, pd.DataFrame],
        meta_controller: MetaController,
        specialists: Dict[str, BaseSpecialist],
        communication_hub: Optional[Any] = None,
        portfolio_risk_manager: Optional[PortfolioRiskManager] = None
    ):
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
            risk_penalty_weight=config.risk_penalty_weight,
            transaction_cost_weight=1.0,
            diversification_bonus_weight=config.diversification_bonus_weight,
            episode_length=config.episode_length,
            warmup_steps=config.warmup_steps
        )
        
        super().__init__(env_config, market_data, communication_hub, portfolio_risk_manager)
        
        self.meta_config = config
        self.meta_controller = meta_controller
        self.specialists = specialists
        
        # Meta-controller specific state
        self.specialist_performance_history: Dict[str, List[float]] = {
            name: [] for name in specialists.keys()
        }
        self.allocation_history: List[np.ndarray] = []
        self.risk_appetite_history: List[float] = []
        self.market_regime_history: List[str] = []
        
        # Performance tracking
        self.portfolio_returns_history: List[float] = []
        self.portfolio_volatility_history: List[float] = []
        self.sharpe_ratio_history: List[float] = []
        
        # Baseline comparison
        self.equal_weight_baseline: List[float] = []
        self.baseline_returns: List[float] = []
        
    def _setup_spaces(self) -> None:
        """Setup action and observation spaces for meta-controller."""
        
        # Action space: [allocation_forex, allocation_commodities, allocation_equity, risk_appetite]
        # Allocation is softmax-normalized, risk_appetite is sigmoid-normalized
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: portfolio state (74 dims) + specialist reports + market regime
        portfolio_state_dim = 74  # From meta_controller.py
        specialist_report_dim = len(self.specialists) * 15  # 15 metrics per specialist
        market_regime_dim = 8  # Market regime indicators
        allocation_history_dim = 10  # Recent allocation history
        
        total_obs_dim = (portfolio_state_dim + specialist_report_dim + 
                        market_regime_dim + allocation_history_dim)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get observation for meta-controller."""
        
        # 1. Portfolio state (74 dimensions)
        portfolio_state = self._get_portfolio_state_vector()
        
        # 2. Specialist performance reports
        specialist_reports = self._get_specialist_reports()
        
        # 3. Market regime indicators
        market_regime = self._get_market_regime_indicators()
        
        # 4. Allocation history
        allocation_history = self._get_allocation_history()
        
        # Combine all observations
        observation = np.concatenate([
            portfolio_state,
            specialist_reports,
            market_regime,
            allocation_history
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
        if len(self.portfolio_returns_history) > 1:
            returns = np.array(self.portfolio_returns_history)
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
    
    def _get_specialist_reports(self) -> np.ndarray:
        """Get comprehensive specialist performance reports."""
        
        reports = np.zeros(len(self.specialists) * 15)
        
        for i, (name, specialist) in enumerate(self.specialists.items()):
            # Get specialist's instruments
            instruments = specialist.get_instruments()
            
            # Calculate performance metrics
            specialist_pnl = 0.0
            specialist_trades = 0
            specialist_win_rate = 0.0
            specialist_sharpe = 0.0
            
            for instrument in instruments:
                if instrument in self.positions:
                    position = self.positions[instrument]
                    specialist_pnl += position.unrealized_pnl
                    specialist_trades += 1
            
            # Calculate specialist performance history
            if name in self.specialist_performance_history:
                perf_history = self.specialist_performance_history[name]
                if len(perf_history) > 1:
                    specialist_sharpe = np.mean(perf_history) / np.std(perf_history) if np.std(perf_history) > 0 else 0
                    specialist_win_rate = np.mean([1 if p > 0 else 0 for p in perf_history])
            
            # Get specialist confidence
            try:
                confidence = specialist.calculate_confidence(self.get_market_state(instruments[0]))
            except Exception:
                confidence = 0.5
            
            # Fill report (15 metrics per specialist)
            base_idx = i * 15
            reports[base_idx] = specialist_pnl / self.portfolio_value  # P&L contribution
            reports[base_idx+1] = specialist_trades / len(instruments)  # Trade ratio
            reports[base_idx+2] = confidence  # Confidence
            reports[base_idx+3] = len(self.specialist_performance_history[name])  # History length
            reports[base_idx+4] = np.mean(self.specialist_performance_history[name][-10:]) if self.specialist_performance_history[name] else 0  # Recent performance
            reports[base_idx+5] = np.std(self.specialist_performance_history[name][-10:]) if len(self.specialist_performance_history[name]) > 1 else 0  # Performance volatility
            reports[base_idx+6] = 1.0 if specialist_pnl > 0 else 0.0  # Positive P&L flag
            reports[base_idx+7] = min(specialist_trades, 5) / 5.0  # Activity level
            reports[base_idx+8] = specialist_sharpe  # Sharpe ratio
            reports[base_idx+9] = specialist_win_rate  # Win rate
            reports[base_idx+10] = np.max(self.specialist_performance_history[name][-20:]) if self.specialist_performance_history[name] else 0  # Max recent performance
            reports[base_idx+11] = np.min(self.specialist_performance_history[name][-20:]) if self.specialist_performance_history[name] else 0  # Min recent performance
            reports[base_idx+12] = 0.0  # Reserved
            reports[base_idx+13] = 0.0  # Reserved
            reports[base_idx+14] = 0.0  # Reserved
        
        return reports
    
    def _get_market_regime_indicators(self) -> np.ndarray:
        """Get comprehensive market regime indicators."""
        
        regime_indicators = np.zeros(8)
        
        # Calculate regime indicators from market data
        try:
            # 1. Overall market volatility
            all_returns = []
            for instrument in self.config.instruments:
                market_state = self.get_market_state(instrument)
                all_returns.extend(market_state['returns'])
            
            if all_returns:
                regime_indicators[0] = np.std(all_returns)  # Market volatility
            
            # 2. Correlation regime (simplified)
            if len(self.config.instruments) > 1:
                returns_matrix = []
                for instrument in self.config.instruments[:3]:  # Limit to 3 instruments
                    market_state = self.get_market_state(instrument)
                    if len(market_state['returns']) > 0:
                        returns_matrix.append(market_state['returns'][-20:])  # Last 20 returns
                
                if len(returns_matrix) > 1:
                    # Calculate average correlation
                    correlations = []
                    for i in range(len(returns_matrix)):
                        for j in range(i+1, len(returns_matrix)):
                            if len(returns_matrix[i]) == len(returns_matrix[j]):
                                corr = np.corrcoef(returns_matrix[i], returns_matrix[j])[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(abs(corr))
                    
                    regime_indicators[1] = np.mean(correlations) if correlations else 0
            
            # 3. Trend strength (simplified)
            trend_strength = 0.0
            for instrument in self.config.instruments[:2]:  # Limit to 2 instruments
                market_state = self.get_market_state(instrument)
                if len(market_state['returns']) > 0:
                    recent_returns = market_state['returns'][-10:]
                    trend_strength += abs(np.mean(recent_returns))
            
            regime_indicators[2] = trend_strength / min(len(self.config.instruments), 2)
            
            # 4. Volume regime
            volume_ratio = 0.0
            for instrument in self.config.instruments[:2]:  # Limit to 2 instruments
                market_state = self.get_market_state(instrument)
                avg_volume = np.mean(self.market_data[instrument]['volume'])
                volume_ratio += market_state['volume'] / avg_volume if avg_volume > 0 else 1.0
            
            regime_indicators[3] = volume_ratio / min(len(self.config.instruments), 2)
            
            # 5. Risk-on/Risk-off indicator
            risk_off_score = 0.0
            for instrument in self.config.instruments[:2]:  # Limit to 2 instruments
                market_state = self.get_market_state(instrument)
                if len(market_state['returns']) > 0:
                    recent_returns = market_state['returns'][-5:]
                    risk_off_score += np.mean(recent_returns < 0)  # Percentage of negative returns
            
            regime_indicators[4] = risk_off_score / min(len(self.config.instruments), 2)
            
            # 6. Market regime from history
            if len(self.market_regime_history) > 0:
                recent_regimes = self.market_regime_history[-10:]
                regime_counts = {'bull': 0, 'bear': 0, 'neutral': 0, 'volatile': 0}
                for regime in recent_regimes:
                    if regime in regime_counts:
                        regime_counts[regime] += 1
                
                # Normalize regime counts
                total_regimes = len(recent_regimes)
                regime_indicators[5] = regime_counts['bull'] / total_regimes
                regime_indicators[6] = regime_counts['bear'] / total_regimes
                regime_indicators[7] = regime_counts['volatile'] / total_regimes
            
        except Exception:
            regime_indicators = np.zeros(8)
        
        return regime_indicators
    
    def _get_allocation_history(self) -> np.ndarray:
        """Get recent allocation history."""
        
        allocation_history = np.zeros(10)
        
        if len(self.allocation_history) > 0:
            # Get last 10 allocations
            recent_allocations = self.allocation_history[-10:]
            
            # Calculate allocation statistics
            if len(recent_allocations) > 0:
                allocations_array = np.array(recent_allocations)
                
                # Mean allocation per specialist
                allocation_history[0:3] = np.mean(allocations_array, axis=0)
                
                # Allocation volatility
                allocation_history[3:6] = np.std(allocations_array, axis=0)
                
                # Allocation stability (inverse of volatility)
                allocation_history[6:9] = 1.0 / (1.0 + np.std(allocations_array, axis=0))
                
                # Recent allocation change
                if len(recent_allocations) > 1:
                    allocation_history[9] = np.linalg.norm(recent_allocations[-1] - recent_allocations[-2])
        
        return allocation_history
    
    def _execute_action(self, action: np.ndarray) -> List[Order]:
        """Execute meta-controller action."""
        
        # Parse action
        allocation_raw = action[:3]  # Raw allocation to specialists
        risk_appetite_raw = action[3]  # Raw risk appetite
        
        # Normalize allocation using softmax
        allocation = torch.softmax(torch.tensor(allocation_raw), dim=0).numpy()
        
        # Normalize risk appetite using sigmoid
        risk_appetite = 1 / (1 + np.exp(-risk_appetite_raw))
        
        # Store for history
        self.allocation_history.append(allocation.copy())
        self.risk_appetite_history.append(risk_appetite)
        
        # Create allocation message
        specialist_names = list(self.specialists.keys())
        allocation_message = AllocationMessage(
            message_id=f"alloc_{self.current_step}",
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
        
        # For meta-controller training, we don't execute actual trades
        # Instead, we simulate specialist responses and calculate rewards
        return []  # Empty order list for meta-controller
    
    def _calculate_reward(self, executed_orders: List) -> float:
        """Calculate reward for meta-controller."""
        
        if self.current_step == 0:
            return 0.0
        
        # Base reward from portfolio performance
        if len(self.portfolio_returns_history) > 1:
            portfolio_return = (self.portfolio_returns_history[-1] - self.portfolio_returns_history[-2]) / self.portfolio_returns_history[-2]
        else:
            portfolio_return = 0.0
        
        reward = self.meta_config.portfolio_return_weight * portfolio_return
        
        # Allocation efficiency bonus
        if len(self.allocation_history) > 0:
            current_allocation = self.allocation_history[-1]
            
            # Bonus for balanced allocation (avoid putting everything in one specialist)
            allocation_entropy = -np.sum(current_allocation * np.log(current_allocation + 1e-8))
            max_entropy = np.log(len(self.specialists))
            balance_bonus = allocation_entropy / max_entropy
            reward += self.meta_config.diversification_bonus_weight * balance_bonus * 0.1
        
        # Risk management bonus
        if len(self.risk_appetite_history) > 0:
            current_risk_appetite = self.risk_appetite_history[-1]
            
            # Bonus for appropriate risk appetite based on market conditions
            market_volatility = self._get_market_regime_indicators()[0]
            
            # If market is volatile, prefer lower risk appetite
            if market_volatility > 0.02:  # High volatility threshold
                risk_penalty = current_risk_appetite * 0.1
            else:
                risk_penalty = (1 - current_risk_appetite) * 0.05
            
            reward -= self.meta_config.risk_penalty_weight * risk_penalty
        
        # Allocation stability bonus
        if len(self.allocation_history) > 1:
            allocation_change = np.linalg.norm(self.allocation_history[-1] - self.allocation_history[-2])
            stability_bonus = (1.0 - allocation_change) * 0.01
            reward += self.meta_config.allocation_stability_weight * stability_bonus
        
        # Specialist performance tracking
        specialist_names = list(self.specialists.keys())
        for i, name in enumerate(specialist_names):
            if len(self.allocation_history) > 0:
                allocation_weight = self.allocation_history[-1][i]
                
                # Track specialist performance
                specialist_pnl = 0.0
                for instrument in self.specialists[name].get_instruments():
                    if instrument in self.positions:
                        specialist_pnl += self.positions[instrument].unrealized_pnl
                
                self.specialist_performance_history[name].append(specialist_pnl)
                
                # Reward for good allocation decisions
                if specialist_pnl > 0:
                    reward += allocation_weight * 0.01
                else:
                    reward -= allocation_weight * 0.005
        
        return reward
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for monitoring."""
        return {
            'portfolio_value': self.portfolio_value,
            'allocation_history': self.allocation_history.copy(),
            'risk_appetite_history': self.risk_appetite_history.copy(),
            'specialist_performance': {name: perf.copy() for name, perf in self.specialist_performance_history.items()},
            'portfolio_returns': self.portfolio_returns_history.copy(),
            'market_regime_history': self.market_regime_history.copy()
        }
