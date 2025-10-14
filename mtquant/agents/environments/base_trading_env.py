"""
FinRL-compatible trading environment for MTQuant.

Features:
1. State Space (stationary): Log returns, normalized indicators, position state, risk metrics
2. Action Space: Continuous -1 to 1 (maps to position size via PositionSizer)
3. Reward Function: Sortino ratio - transaction costs
4. Episode metrics tracking (trades, win rate, Sharpe)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

from mtquant.utils.logger import get_logger
from mtquant.risk_management.position_sizer import PositionSizer
from mtquant.data.processors.feature_engineering import (
    add_technical_indicators,
    calculate_log_returns,
    normalize_features
)


@dataclass
class EpisodeMetrics:
    """Episode performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_transaction_costs: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0


class MTQuantTradingEnv(gym.Env):
    """
    MTQuant Trading Environment for RL agents.
    
    State Space (stationary):
    - Log returns (NOT raw prices)
    - Normalized technical indicators (RSI, MACD, Bollinger)
    - Position state (holdings, P&L, age)
    - Risk metrics (volatility, drawdown)
    
    Action Space:
    - Continuous: -1 to 1
    - Maps to position size via PositionSizer
    
    Reward Function:
    - Sortino ratio - transaction costs
    - Penalizes: downside volatility, excessive trading
    - Rewards: risk-adjusted returns
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        data: pd.DataFrame,
        symbol: str = "XAUUSD",
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.003,
        position_sizer: Optional[PositionSizer] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize trading environment.
        
        Args:
            data: OHLCV data with technical indicators
            symbol: Trading symbol
            initial_capital: Starting capital
            transaction_cost: Transaction cost per trade (0.3% default)
            position_sizer: Position sizing strategy
            config: Environment configuration
        """
        super().__init__()
        
        self.logger = get_logger(__name__)
        self.data = data.copy()
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.config = config or {}
        
        # Validate data
        self._validate_data()
        
        # Prepare features
        self._prepare_features()
        
        # Initialize state
        self.current_step = 0
        self.current_capital = initial_capital
        self.current_position = 0.0  # Current position size
        self.position_entry_price = 0.0
        self.position_entry_step = 0
        self.trade_history: List[Dict] = []
        self.episode_metrics = EpisodeMetrics()
        
        # Position sizer
        self.position_sizer = position_sizer or PositionSizer(
            self.config.get('position_sizing', {})
        )
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: features + position state + risk metrics
        n_features = len(self.feature_columns)
        n_position_features = 3  # holdings, unrealized_pnl, position_age
        n_risk_features = 2  # volatility, drawdown
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features + n_position_features + n_risk_features,),
            dtype=np.float32
        )
        
        self.logger.info(
            f"Environment initialized: {symbol}, {len(data)} steps, "
            f"capital: {initial_capital}, features: {n_features}"
        )
    
    def _validate_data(self) -> None:
        """Validate input data."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(self.data) < 100:
            raise ValueError("Data must have at least 100 rows")
        
        self.logger.info(f"Data validated: {len(self.data)} rows, {len(self.data.columns)} columns")
    
    def _prepare_features(self) -> None:
        """Prepare features for RL agent."""
        # Add technical indicators if not present
        if 'rsi' not in self.data.columns:
            self.data = add_technical_indicators(self.data)
        
        # Calculate log returns
        if 'log_returns' not in self.data.columns:
            self.data = calculate_log_returns(self.data)
        
        # Define feature columns (stationary features only)
        self.feature_columns = [
            'log_returns',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'sma_20', 'sma_50', 'atr'
        ]
        
        # Filter existing columns
        self.feature_columns = [
            col for col in self.feature_columns 
            if col in self.data.columns
        ]
        
        # Normalize features
        self.data = normalize_features(self.data, self.feature_columns)
        
        self.logger.info(f"Features prepared: {len(self.feature_columns)} features")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.current_capital = self.initial_capital
        self.current_position = 0.0
        self.position_entry_price = 0.0
        self.position_entry_step = 0
        self.trade_history = []
        self.episode_metrics = EpisodeMetrics()
        
        # Get initial observation
        observation = self._get_state()
        info = self._get_info()
        
        self.logger.info(f"Environment reset: step {self.current_step}")
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Extract action value
        action_value = float(action[0])
        
        # Execute trade if action is significant
        trade_result = self._execute_trade(action_value)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Calculate reward
        reward = self._calculate_reward(trade_result)
        
        # Get next observation
        observation = self._get_state()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _execute_trade(self, action_value: float) -> Dict:
        """Execute trade based on action."""
        trade_result = {
            'action': action_value,
            'position_before': self.current_position,
            'position_after': self.current_position,
            'trade_executed': False,
            'trade_pnl': 0.0,
            'transaction_cost': 0.0
        }
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate target position size (optimized thresholds to reduce over-trading)
        if abs(action_value) > 0.02:  # Increased threshold to reduce over-trading
            target_position = self._calculate_position_size(action_value, current_price)
            
            # Debug logging
            self.logger.debug(f"Action: {action_value:.4f}, Target position: {target_position:.4f}, Current: {self.current_position:.4f}")
            
            # Execute position change
            if abs(target_position - self.current_position) > 0.001:  # Increased minimum position change
                self.logger.debug(f"Executing trade: {self.current_position:.4f} -> {target_position:.4f}")
                trade_result = self._execute_position_change(
                    self.current_position, target_position, current_price
                )
                self.logger.debug(f"Trade executed: P&L={trade_result['trade_pnl']:.4f}, Cost={trade_result['transaction_cost']:.4f}")
            else:
                self.logger.debug(f"Position change too small: {abs(target_position - self.current_position):.6f}")
        else:
            self.logger.debug(f"Action too small: {abs(action_value):.6f}")
        
        return trade_result
    
    def _calculate_position_size(self, action_value: float, current_price: float) -> float:
        """Calculate position size using position sizer."""
        try:
            # Fallback for when position_sizer is None (e.g., in tests)
            if self.position_sizer is None:
                # Simple position sizing: map action value (-1 to 1) to position size (0 to 1)
                position_size = abs(action_value)
                # Scale down for safety
                position_size = position_size * 0.1  # Max 10% of portfolio
                return position_size
            
            # Get current market data for volatility
            if self.current_step >= 20:
                recent_data = self.data.iloc[max(0, self.current_step-20):self.current_step+1]
                volatility = recent_data['atr'].mean() if 'atr' in recent_data.columns else current_price * 0.01
            else:
                volatility = current_price * 0.01
            
            # Calculate position size
            sizing_result = self.position_sizer.calculate(
                signal=action_value,
                portfolio_equity=self.current_capital,
                instrument_volatility=volatility,
                method='volatility'  # Use volatility-based sizing
            )
            
            # Convert to position size (lots)
            position_size = sizing_result.position_size / current_price
            
            # Apply circuit breaker multiplier if available
            if hasattr(self, 'circuit_breaker'):
                position_size *= self.circuit_breaker.get_position_size_multiplier()
            
            return position_size
            
        except Exception as e:
            self.logger.warning(f"Position sizing failed: {e}")
            return 0.0
    
    def _execute_position_change(
        self, 
        from_position: float, 
        to_position: float, 
        current_price: float
    ) -> Dict:
        """Execute position change and calculate P&L."""
        trade_result = {
            'action': (to_position - from_position) / max(abs(from_position), 0.01),
            'position_before': from_position,
            'position_after': to_position,
            'trade_executed': True,
            'trade_pnl': 0.0,
            'transaction_cost': 0.0
        }
        
        # Calculate P&L from previous position
        if from_position != 0:
            position_pnl = from_position * (current_price - self.position_entry_price)
            trade_result['trade_pnl'] = position_pnl
            
            # Update capital
            self.current_capital += position_pnl
            
            # Record trade (closing position)
            self._record_trade(
                position=from_position,
                entry_price=self.position_entry_price,
                exit_price=current_price,
                pnl=position_pnl,
                duration=self.current_step - self.position_entry_step
            )
        
        # Record trade for opening new position (if any)
        if to_position != 0 and from_position == 0:
            # Opening a new position - record as trade with 0 P&L
            self._record_trade(
                position=to_position,
                entry_price=current_price,
                exit_price=current_price,
                pnl=0.0,
                duration=0
            )
        
        # Calculate transaction cost
        position_change = abs(to_position - from_position)
        transaction_cost = position_change * current_price * self.transaction_cost
        trade_result['transaction_cost'] = transaction_cost
        
        # Update capital
        self.current_capital -= transaction_cost
        
        # Update position
        self.current_position = to_position
        if to_position != 0:
            self.position_entry_price = current_price
            self.position_entry_step = self.current_step
        else:
            self.position_entry_price = 0.0
            self.position_entry_step = 0
        
        return trade_result
    
    def _record_trade(
        self, 
        position: float, 
        entry_price: float, 
        exit_price: float, 
        pnl: float, 
        duration: int
    ) -> None:
        """Record trade in history."""
        trade = {
            'step': self.current_step,
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'duration': duration,
            'timestamp': datetime.utcnow()
        }
        
        self.trade_history.append(trade)
        
        # Update episode metrics
        self.episode_metrics.total_trades += 1
        self.episode_metrics.total_pnl += pnl
        
        if pnl > 0:
            self.episode_metrics.winning_trades += 1
        else:
            self.episode_metrics.losing_trades += 1
    
    def _calculate_reward(self, trade_result: Dict) -> float:
        """Calculate sophisticated reward based on trading performance and risk metrics."""
        try:
            trade_pnl = trade_result.get('trade_pnl', 0.0)
            transaction_cost = trade_result.get('transaction_cost', 0.0)
            trade_executed = trade_result.get('trade_executed', False)
            
            reward = 0.0
            
            if trade_executed:
                # P&L-based reward (primary focus on profitability)
                if trade_pnl != 0.0:
                    net_pnl = trade_pnl - transaction_cost
                    if self.current_capital > 0:
                        # Scale P&L reward (more aggressive for better learning)
                        pnl_reward = (net_pnl / self.current_capital) * 200  # Increased multiplier
                        reward += pnl_reward
                        
                        # Strong bonus for profitable trades
                        if net_pnl > 0:
                            reward += 5.0  # Increased bonus
                        # Strong penalty for losing trades
                        else:
                            reward -= 3.0  # Increased penalty
                    else:
                        reward += 0.0
                else:
                    # Penalty for trades with no P&L
                    reward -= 1.0
                
                # Transaction cost penalty (encourages efficient trading)
                if transaction_cost > 0:
                    cost_penalty = (transaction_cost / self.current_capital) * 100  # Increased penalty
                    reward -= cost_penalty
                
                # Over-trading penalty (if too many trades in short time)
                if len(self.trade_history) > 5:  # Reduced threshold
                    recent_trades = self.trade_history[-5:]
                    if len(recent_trades) >= 5:
                        # Check if trades are too frequent
                        time_span = recent_trades[-1]['timestamp'] - recent_trades[0]['timestamp']
                        if time_span.total_seconds() < 1800:  # Less than 30 minutes
                            reward -= 2.0  # Increased penalty for over-trading
                
            else:
                # Small penalty for inaction (reduced to avoid forcing trades)
                reward -= 0.1
                
            # Sharpe ratio bonus (if we have enough trades)
            if len(self.trade_history) >= 5:
                try:
                    returns = [trade['pnl'] / self.current_capital for trade in self.trade_history[-20:]]
                    if len(returns) > 1:
                        mean_return = np.mean(returns)
                        std_return = np.std(returns)
                        if std_return > 0:
                            sharpe = mean_return / std_return
                            # Bonus for good Sharpe ratio
                            if sharpe > 0.5:
                                reward += 1.0
                            elif sharpe < -0.5:
                                reward -= 0.5
                except:
                    pass  # Ignore Sharpe calculation errors
                
            return float(reward)
            
        except Exception as e:
            self.logger.warning(f"Reward calculation failed: {e}")
            return 0.0
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        try:
            # Get market features
            if self.current_step < len(self.data):
                market_features = self.data.iloc[self.current_step][self.feature_columns].values
            else:
                # Use last available data
                market_features = self.data.iloc[-1][self.feature_columns].values
            
            # Position state features
            position_features = np.array([
                self.current_position,  # Current holdings
                self._get_unrealized_pnl(),  # Unrealized P&L
                self._get_position_age()  # Position age
            ])
            
            # Risk metrics
            risk_features = np.array([
                self._get_portfolio_volatility(),  # Portfolio volatility
                self._get_current_drawdown()  # Current drawdown
            ])
            
            # Combine all features
            state = np.concatenate([
                market_features,
                position_features,
                risk_features
            ])
            
            # Handle NaN values
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return state.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"State calculation failed: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _get_unrealized_pnl(self) -> float:
        """Get unrealized P&L of current position."""
        if self.current_position == 0 or self.current_step >= len(self.data):
            return 0.0
        
        current_price = self.data.iloc[self.current_step]['close']
        unrealized_pnl = self.current_position * (current_price - self.position_entry_price)
        
        # Normalize by capital
        return unrealized_pnl / self.current_capital if self.current_capital > 0 else 0.0
    
    def _get_position_age(self) -> float:
        """Get normalized position age."""
        if self.current_position == 0:
            return 0.0
        
        age_steps = self.current_step - self.position_entry_step
        max_age = len(self.data)  # Normalize by total episode length
        
        return min(age_steps / max_age, 1.0)
    
    def _get_portfolio_volatility(self) -> float:
        """Get portfolio volatility."""
        if len(self.trade_history) < 5:
            return 0.0
        
        recent_returns = [trade['pnl'] / self.current_capital for trade in self.trade_history[-20:]]
        return np.std(recent_returns) if recent_returns else 0.0
    
    def _get_current_drawdown(self) -> float:
        """Get current drawdown percentage."""
        if self.current_capital <= self.initial_capital:
            return 0.0
        
        peak_capital = max(self.initial_capital, max(
            [self.initial_capital] + 
            [self.current_capital - sum(trade['pnl'] for trade in self.trade_history[:i+1])
             for i in range(len(self.trade_history))]
        ))
        
        drawdown = (peak_capital - self.current_capital) / peak_capital
        return max(0.0, drawdown)
    
    def _get_info(self) -> Dict:
        """Get environment info."""
        return {
            'current_step': self.current_step,
            'current_capital': self.current_capital,
            'current_position': self.current_position,
            'total_trades': len(self.trade_history),
            'episode_metrics': self.episode_metrics.__dict__
        }
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render environment state."""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Capital: {self.current_capital:.2f}")
            print(f"Position: {self.current_position:.4f}")
            print(f"Trades: {len(self.trade_history)}")
            print(f"Total P&L: {self.episode_metrics.total_pnl:.2f}")
        
        return None
    
    def close(self) -> None:
        """Close environment."""
        self.logger.info("Environment closed")
    
    def get_episode_metrics(self) -> EpisodeMetrics:
        """Get current episode metrics with updated calculations."""
        if self.trade_history and self.episode_metrics.total_trades > 0:
            # Recalculate win rate
            self.episode_metrics.win_rate = (
                self.episode_metrics.winning_trades / self.episode_metrics.total_trades
            )
            
            # Recalculate average win/loss
            if self.episode_metrics.winning_trades > 0:
                winning_trades = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
                self.episode_metrics.avg_win = np.mean(winning_trades)
            
            if self.episode_metrics.losing_trades > 0:
                losing_trades = [t['pnl'] for t in self.trade_history if t['pnl'] < 0]
                self.episode_metrics.avg_loss = abs(np.mean(losing_trades))
            
            # Recalculate profit factor
            if self.episode_metrics.avg_loss > 0 and self.episode_metrics.avg_win > 0:
                self.episode_metrics.profit_factor = (
                    self.episode_metrics.avg_win / self.episode_metrics.avg_loss
                )
            
            # Calculate Sharpe ratio (use absolute P&L values, not returns)
            if len(self.trade_history) > 1:
                pnl_values = [trade['pnl'] for trade in self.trade_history]
                mean_pnl = np.mean(pnl_values)
                std_pnl = np.std(pnl_values)
                
                if std_pnl > 0:
                    # Simple Sharpe ratio (mean P&L / std P&L)
                    self.episode_metrics.sharpe_ratio = mean_pnl / std_pnl
                else:
                    self.episode_metrics.sharpe_ratio = 0.0
            else:
                self.episode_metrics.sharpe_ratio = 0.0
            
            # Calculate Sortino ratio
            if len(self.trade_history) > 1:
                pnl_values = [trade['pnl'] for trade in self.trade_history]
                mean_pnl = np.mean(pnl_values)
                downside_pnl = [p for p in pnl_values if p < 0]
                
                if downside_pnl and len(downside_pnl) > 1:
                    downside_std = np.std(downside_pnl)
                    if downside_std > 0:
                        self.episode_metrics.sortino_ratio = mean_pnl / downside_std
                    else:
                        self.episode_metrics.sortino_ratio = 0.0
                else:
                    self.episode_metrics.sortino_ratio = 0.0
            else:
                self.episode_metrics.sortino_ratio = 0.0
        
        return self.episode_metrics
