"""
Hierarchical Multi-Agent Trading Environment

This module provides the base environment for hierarchical multi-agent trading,
supporting both individual specialist training and meta-controller coordination.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces

from ..hierarchical.meta_controller import MetaController
from ..hierarchical.base_specialist import BaseSpecialist
from ..hierarchical.communication import CommunicationHub, AllocationMessage
from ...risk_management.portfolio_risk_manager import PortfolioRiskManager
from ...risk_management.position_sizer import PositionSizer
from ...mcp_integration.models.order import Order
from ...mcp_integration.models.position import Position


@dataclass
class EnvironmentConfig:
    """Configuration for hierarchical trading environment."""
    
    # Market data
    instruments: List[str]
    timeframe: str = "1H"
    lookback_window: int = 100
    
    # Environment settings
    initial_capital: float = 100000.0
    transaction_cost: float = 0.003  # 0.3% per trade
    max_position_size: float = 0.1  # 10% of portfolio per instrument
    
    # Risk management
    max_portfolio_var: float = 0.02  # 2% VaR limit
    max_correlation_exposure: float = 0.7
    stop_loss_pct: float = 0.02  # 2% stop loss
    
    # Reward shaping
    risk_penalty_weight: float = 1.0
    transaction_cost_weight: float = 1.0
    diversification_bonus_weight: float = 0.5
    
    # Training
    episode_length: int = 1000  # steps per episode
    warmup_steps: int = 50  # steps before trading starts


class BaseHierarchicalEnv(gym.Env, ABC):
    """
    Base class for hierarchical multi-agent trading environments.
    
    This environment supports:
    - Individual specialist training (Phase 1)
    - Meta-controller training (Phase 2) 
    - Joint training (Phase 3)
    """
    
    def __init__(
        self,
        config: EnvironmentConfig,
        market_data: Dict[str, pd.DataFrame],
        communication_hub: Optional[CommunicationHub] = None,
        portfolio_risk_manager: Optional[PortfolioRiskManager] = None
    ):
        super().__init__()
        
        self.config = config
        self.market_data = market_data
        self.communication_hub = communication_hub or CommunicationHub()
        self.portfolio_risk_manager = portfolio_risk_manager or PortfolioRiskManager()
        
        # Validate market data
        self._validate_market_data()
        
        # Initialize state
        self.current_step = 0
        self.episode_start_step = 0
        self.portfolio_value = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Order] = []
        
        # Performance tracking
        self.episode_returns = []
        self.episode_volatility = []
        self.episode_sharpe = []
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Position sizer
        position_sizer_config = {
            'max_position_pct': config.max_position_size,
            'kelly_fraction': 0.25,
            'volatility_lookback': 20
        }
        self.position_sizer = PositionSizer(position_sizer_config)
    
    def _validate_market_data(self) -> None:
        """Validate that all instruments have sufficient market data."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for instrument in self.config.instruments:
            if instrument not in self.market_data:
                raise ValueError(f"Missing market data for {instrument}")
            
            data = self.market_data[instrument]
            if len(data) < self.config.lookback_window + self.config.episode_length:
                raise ValueError(f"Insufficient data for {instrument}: {len(data)} < {self.config.lookback_window + self.config.episode_length}")
            
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing column '{col}' in {instrument} data")
    
    @abstractmethod
    def _setup_spaces(self) -> None:
        """Setup action and observation spaces for specific environment type."""
        pass
    
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current observation based on environment type."""
        pass
    
    @abstractmethod
    def _execute_action(self, action: Union[np.ndarray, Dict[str, np.ndarray]]) -> List[Order]:
        """Execute action and return list of orders."""
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.episode_start_step = 0
        self.portfolio_value = self.config.initial_capital
        self.positions.clear()
        self.trade_history.clear()
        
        # Reset performance tracking
        self.episode_returns.clear()
        self.episode_volatility.clear()
        self.episode_sharpe.clear()
        
        # Reset communication hub
        self.communication_hub.clear_history()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Execute action and get orders
        orders = self._execute_action(action)
        
        # Process orders (simplified - in real implementation would go through broker)
        executed_orders = self._simulate_order_execution(orders)
        
        # Update portfolio state
        self._update_portfolio_state()
        
        # Calculate reward
        reward = self._calculate_reward(executed_orders)
        
        # Update step counter
        self.current_step += 1
        
        # Check if episode is done
        done = self._is_done()
        truncated = False
        
        # Get next observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _simulate_order_execution(self, orders: List[Order]) -> List[Order]:
        """Simulate order execution (simplified version)."""
        executed_orders = []
        
        for order in orders:
            # Get current market price
            instrument_data = self.market_data[order.symbol]
            current_price = instrument_data.iloc[self.current_step]['close']
            
            # Update order with execution price
            order.execution_price = current_price
            order.status = 'filled'
            
            # Apply transaction cost
            transaction_cost = abs(order.quantity) * current_price * self.config.transaction_cost
            order.transaction_cost = transaction_cost
            
            executed_orders.append(order)
            
            # Update position
            self._update_position(order)
        
        return executed_orders
    
    def _update_position(self, order: Order) -> None:
        """Update position based on executed order."""
        symbol = order.symbol
        
        if symbol not in self.positions:
            # Create new position
            self.positions[symbol] = Position(
                position_id=f"{symbol}_{self.current_step}",
                agent_id="hierarchical_env",
                side=order.side,
                entry_price=order.execution_price,
                quantity=order.quantity,
                current_price=order.execution_price,
                unrealized_pnl=0.0,
                opened_at=pd.Timestamp.now()
            )
        else:
            # Update existing position
            position = self.positions[symbol]
            
            if order.side == position.side:
                # Increase position
                total_value = (position.quantity * position.entry_price + 
                             order.quantity * order.execution_price)
                total_quantity = position.quantity + order.quantity
                position.entry_price = total_value / total_quantity
                position.quantity = total_quantity
            else:
                # Close or reduce position
                if abs(order.quantity) >= abs(position.quantity):
                    # Close position
                    del self.positions[symbol]
                else:
                    # Reduce position
                    position.quantity += order.quantity
    
    def _update_portfolio_state(self) -> None:
        """Update portfolio value and position P&L."""
        total_value = 0.0
        
        for symbol, position in self.positions.items():
            # Get current market price
            instrument_data = self.market_data[symbol]
            current_price = instrument_data.iloc[self.current_step]['close']
            
            # Update position
            position.current_price = current_price
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            
            # Add to total value
            total_value += position.quantity * current_price
        
        # Update portfolio value
        self.portfolio_value = total_value
    
    def _calculate_reward(self, executed_orders: List[Order]) -> float:
        """Calculate reward based on portfolio performance and risk."""
        if self.current_step == 0:
            return 0.0
        
        # Calculate portfolio return
        if len(self.episode_returns) > 0:
            prev_value = self.episode_returns[-1]
            current_return = (self.portfolio_value - prev_value) / prev_value
        else:
            current_return = 0.0
        
        self.episode_returns.append(self.portfolio_value)
        
        # Base reward (portfolio return)
        reward = current_return
        
        # Transaction cost penalty
        total_transaction_cost = sum(order.transaction_cost for order in executed_orders)
        transaction_penalty = total_transaction_cost / self.portfolio_value
        reward -= self.config.transaction_cost_weight * transaction_penalty
        
        # Risk penalty (if VaR exceeded)
        if self.portfolio_risk_manager:
            try:
                portfolio_positions = list(self.positions.values())
                risk_check = self.portfolio_risk_manager.check_portfolio_risk(
                    portfolio_positions,
                    self.portfolio_value,
                    self.config.max_portfolio_var
                )
                
                if not risk_check.is_valid:
                    risk_penalty = abs(risk_check.var_excess) * self.config.risk_penalty_weight
                    reward -= risk_penalty
            except Exception:
                # If risk calculation fails, apply small penalty
                reward -= 0.001
        
        # Diversification bonus
        if len(self.positions) > 1:
            diversification_bonus = min(len(self.positions) / len(self.config.instruments), 1.0)
            reward += self.config.diversification_bonus_weight * diversification_bonus * 0.01
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        return self.current_step >= self.config.episode_length
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        return {
            'portfolio_value': self.portfolio_value,
            'num_positions': len(self.positions),
            'current_step': self.current_step,
            'episode_length': self.config.episode_length,
            'instruments': self.config.instruments
        }
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state for meta-controller."""
        return {
            'portfolio_value': self.portfolio_value,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl / (pos.quantity * pos.entry_price) if pos.quantity != 0 else 0
            } for symbol, pos in self.positions.items()},
            'cash': self.portfolio_value - sum(pos.quantity * pos.current_price for pos in self.positions.values()),
            'total_exposure': sum(abs(pos.quantity * pos.current_price) for pos in self.positions.values()),
            'num_instruments': len(self.positions)
        }
    
    def get_market_state(self, instrument: str) -> Dict[str, Any]:
        """Get current market state for specific instrument."""
        if instrument not in self.market_data:
            raise ValueError(f"Unknown instrument: {instrument}")
        
        data = self.market_data[instrument]
        current_idx = self.current_step
        
        if current_idx >= len(data):
            raise ValueError(f"Step {current_idx} exceeds data length for {instrument}")
        
        # Get current and historical data
        current_data = data.iloc[current_idx]
        historical_data = data.iloc[max(0, current_idx - self.config.lookback_window):current_idx + 1]
        
        return {
            'current_price': current_data['close'],
            'open': current_data['open'],
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'returns': historical_data['close'].pct_change().dropna().values,
            'volatility': historical_data['close'].pct_change().std() if len(historical_data) > 1 else 0.0,
            'position': self.positions.get(instrument, None)
        }
    
    def render(self, mode: str = 'human') -> Optional[Any]:
        """Render environment state."""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.config.episode_length}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Positions: {len(self.positions)}")
            
            for symbol, position in self.positions.items():
                print(f"  {symbol}: {position.quantity:.2f} @ ${position.current_price:.2f} "
                      f"(P&L: ${position.unrealized_pnl:.2f})")
        
        return None
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
