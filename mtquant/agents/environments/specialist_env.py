"""
Specialist Training Environments

This module provides individual environments for training each specialist
(Forex, Commodities, Equity) on their domain-specific instruments.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import gymnasium as gym
from gymnasium import spaces

from .hierarchical_env import BaseHierarchicalEnv, EnvironmentConfig
from ..hierarchical.base_specialist import BaseSpecialist
from ..hierarchical.communication import AllocationMessage, PerformanceReport
from ...risk_management.position_sizer import PositionSizer
from ...mcp_integration.models.order import Order


class SpecialistEnv(BaseHierarchicalEnv):
    """
    Environment for training individual specialists.
    
    Each specialist observes:
    - Market data for their instruments
    - Domain-specific features
    - Portfolio state relevant to their domain
    
    Actions:
    - Trading signals for each instrument (-1 to 1)
    - Position sizing decisions
    """
    
    def __init__(
        self,
        config: EnvironmentConfig,
        market_data: Dict[str, np.ndarray],
        specialist: BaseSpecialist,
        communication_hub: Optional[Any] = None,
        portfolio_risk_manager: Optional[Any] = None
    ):
        # Initialize specialist-specific attributes BEFORE calling super().__init__
        self.specialist = specialist
        self.specialist_instruments = specialist.get_instruments()
        
        # Specialist-specific state
        self.domain_features: Dict[str, np.ndarray] = {}
        self.confidence_history: List[float] = []
        self.signal_history: Dict[str, List[float]] = {
            instrument: [] for instrument in self.specialist_instruments
        }
        
        # Performance tracking
        self.specialist_pnl_history: List[float] = []
        self.trade_frequency_history: List[float] = []
        
        # Now call super().__init__ which will call _setup_spaces
        super().__init__(config, market_data, communication_hub, portfolio_risk_manager)
        
    def _setup_spaces(self) -> None:
        """Setup action and observation spaces for specialist."""
        
        # Action space: trading signals for each instrument
        num_instruments = len(self.specialist_instruments)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_instruments,),
            dtype=np.float32
        )
        
        # Observation space: market data + domain features + portfolio state
        market_features_per_instrument = 20  # Technical indicators per instrument
        domain_features_dim = 15  # Domain-specific features
        portfolio_state_dim = 10  # Relevant portfolio state
        
        total_obs_dim = (num_instruments * market_features_per_instrument + 
                        domain_features_dim + 
                        portfolio_state_dim)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get observation for specialist."""
        
        # 1. Market features for each instrument
        market_features = self._get_market_features()
        
        # 2. Domain-specific features
        domain_features = self._get_domain_features()
        
        # 3. Portfolio state relevant to specialist
        portfolio_state = self._get_specialist_portfolio_state()
        
        # Combine all observations
        observation = np.concatenate([
            market_features,
            domain_features,
            portfolio_state
        ])
        
        return observation.astype(np.float32)
    
    def _get_market_features(self) -> np.ndarray:
        """Get market features for all specialist instruments."""
        
        features_per_instrument = 20
        total_features = len(self.specialist_instruments) * features_per_instrument
        market_features = np.zeros(total_features)
        
        for i, instrument in enumerate(self.specialist_instruments):
            try:
                market_state = self.get_market_state(instrument)
                base_idx = i * features_per_instrument
                
                # Price-based features
                current_price = market_state['current_price']
                returns = market_state['returns']
                
                if len(returns) > 0:
                    market_features[base_idx] = returns[-1]  # Latest return
                    market_features[base_idx+1] = np.mean(returns[-5:])  # 5-period mean return
                    market_features[base_idx+2] = np.std(returns[-5:])  # 5-period volatility
                    market_features[base_idx+3] = np.mean(returns[-20:])  # 20-period mean return
                    market_features[base_idx+4] = np.std(returns[-20:])  # 20-period volatility
                    
                    # Momentum indicators
                    market_features[base_idx+5] = np.sum(returns[-5:])  # 5-period momentum
                    market_features[base_idx+6] = np.sum(returns[-10:])  # 10-period momentum
                    market_features[base_idx+7] = np.sum(returns[-20:])  # 20-period momentum
                    
                    # Volatility regime
                    short_vol = np.std(returns[-5:])
                    long_vol = np.std(returns[-20:])
                    market_features[base_idx+8] = short_vol / long_vol if long_vol > 0 else 1.0
                    
                    # Trend strength
                    market_features[base_idx+9] = np.corrcoef(
                        np.arange(len(returns[-10:])), returns[-10:]
                    )[0, 1] if len(returns[-10:]) > 1 else 0
                    
                    # Volume features
                    volume = market_state['volume']
                    avg_volume = np.mean(self.market_data[instrument]['volume'])
                    market_features[base_idx+10] = volume / avg_volume if avg_volume > 0 else 1.0
                    
                    # Price position in recent range
                    recent_data = self.market_data[instrument].iloc[
                        max(0, self.current_step - 20):self.current_step + 1
                    ]
                    if len(recent_data) > 0:
                        high_20 = recent_data['high'].max()
                        low_20 = recent_data['low'].min()
                        market_features[base_idx+11] = (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
                    
                    # Technical indicators (simplified)
                    # RSI approximation
                    gains = returns[returns > 0]
                    losses = -returns[returns < 0]
                    avg_gain = np.mean(gains) if len(gains) > 0 else 0
                    avg_loss = np.mean(losses) if len(losses) > 0 else 0
                    rs = avg_gain / avg_loss if avg_loss > 0 else 100
                    rsi = 100 - (100 / (1 + rs))
                    market_features[base_idx+12] = rsi / 100  # Normalized RSI
                    
                    # MACD approximation
                    ema_12 = np.mean(returns[-12:]) if len(returns) >= 12 else np.mean(returns)
                    ema_26 = np.mean(returns[-26:]) if len(returns) >= 26 else np.mean(returns)
                    macd = ema_12 - ema_26
                    market_features[base_idx+13] = macd
                    
                    # Bollinger Bands approximation
                    bb_upper = np.mean(returns[-20:]) + 2 * np.std(returns[-20:])
                    bb_lower = np.mean(returns[-20:]) - 2 * np.std(returns[-20:])
                    bb_position = (returns[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
                    market_features[base_idx+14] = bb_position
                    
                    # Position information
                    if instrument in self.positions:
                        position = self.positions[instrument]
                        market_features[base_idx+15] = position.quantity / self.config.max_position_size
                        market_features[base_idx+16] = position.unrealized_pnl / self.portfolio_value
                        market_features[base_idx+17] = (position.current_price - position.entry_price) / position.entry_price
                        market_features[base_idx+18] = 1.0  # Position exists
                    else:
                        market_features[base_idx+15:base_idx+19] = 0
                    
                    # Trade frequency
                    market_features[base_idx+19] = len(self.signal_history[instrument]) / max(self.current_step, 1)
                    
            except Exception as e:
                # Fill with zeros if error
                market_features[base_idx:base_idx+features_per_instrument] = 0
        
        return market_features
    
    def _get_domain_features(self) -> np.ndarray:
        """Get domain-specific features."""
        
        domain_features = np.zeros(15)
        
        try:
            # Get domain features from specialist
            if hasattr(self.specialist, 'get_domain_features'):
                domain_features_dict = self.specialist.get_domain_features()
                
                # Convert to array (assuming specialist returns dict with numeric values)
                feature_values = list(domain_features_dict.values())[:15]  # Limit to 15 features
                domain_features[:len(feature_values)] = feature_values
            
            # Add specialist-specific indicators
            if hasattr(self.specialist, 'calculate_confidence'):
                # Get confidence for first instrument
                if self.specialist_instruments:
                    market_state = self.get_market_state(self.specialist_instruments[0])
                    confidence = self.specialist.calculate_confidence(market_state)
                    domain_features[0] = confidence
            
            # Add specialist-specific regime detection
            if hasattr(self.specialist, 'detect_correlation_regime'):
                try:
                    correlation_regime = self.specialist.detect_correlation_regime()
                    domain_features[1] = correlation_regime
                except Exception:
                    domain_features[1] = 0
            
            # Add specialist-specific signals
            if hasattr(self.specialist, 'get_carry_signal'):
                try:
                    carry_signal = self.specialist.get_carry_signal()
                    domain_features[2] = carry_signal
                except Exception:
                    domain_features[2] = 0
            
            # Add specialist-specific schedule checks
            if hasattr(self.specialist, 'check_central_bank_schedule'):
                try:
                    schedule_check = self.specialist.check_central_bank_schedule()
                    domain_features[3] = 1.0 if schedule_check else 0.0
                except Exception:
                    domain_features[3] = 0
            
            # Add specialist-specific inflation regime
            if hasattr(self.specialist, 'detect_inflation_regime'):
                try:
                    inflation_regime = self.specialist.detect_inflation_regime()
                    domain_features[4] = inflation_regime
                except Exception:
                    domain_features[4] = 0
            
            # Add specialist-specific safe haven demand
            if hasattr(self.specialist, 'get_safe_haven_demand'):
                try:
                    safe_haven_demand = self.specialist.get_safe_haven_demand()
                    domain_features[5] = safe_haven_demand
                except Exception:
                    domain_features[5] = 0
            
            # Add specialist-specific sector rotation
            if hasattr(self.specialist, 'detect_sector_rotation'):
                try:
                    sector_rotation = self.specialist.detect_sector_rotation()
                    domain_features[6] = sector_rotation
                except Exception:
                    domain_features[6] = 0
            
            # Add specialist-specific fear/greed index
            if hasattr(self.specialist, 'get_fear_greed_index'):
                try:
                    fear_greed_index = self.specialist.get_fear_greed_index()
                    domain_features[7] = fear_greed_index
                except Exception:
                    domain_features[7] = 0
            
            # Add specialist-specific earnings calendar
            if hasattr(self.specialist, 'check_earnings_calendar'):
                try:
                    earnings_calendar = self.specialist.check_earnings_calendar()
                    domain_features[8] = 1.0 if earnings_calendar else 0.0
                except Exception:
                    domain_features[8] = 0
            
        except Exception:
            domain_features = np.zeros(15)
        
        return domain_features
    
    def _get_specialist_portfolio_state(self) -> np.ndarray:
        """Get portfolio state relevant to specialist."""
        
        portfolio_state = np.zeros(10)
        
        # Specialist-specific portfolio metrics
        specialist_positions = 0
        specialist_pnl = 0.0
        specialist_exposure = 0.0
        
        for instrument in self.specialist_instruments:
            if instrument in self.positions:
                position = self.positions[instrument]
                specialist_positions += 1
                specialist_pnl += position.unrealized_pnl
                specialist_exposure += abs(position.quantity * position.current_price)
        
        portfolio_state[0] = specialist_positions / len(self.specialist_instruments)  # Position ratio
        portfolio_state[1] = specialist_pnl / max(self.portfolio_value, 1.0)  # P&L contribution
        portfolio_state[2] = specialist_exposure / max(self.portfolio_value, 1.0)  # Exposure ratio
        portfolio_state[3] = len(self.specialist_pnl_history) / max(self.current_step, 1)  # History length
        
        # Recent performance
        if len(self.specialist_pnl_history) > 0:
            portfolio_state[4] = np.mean(self.specialist_pnl_history[-10:])  # Recent P&L
            portfolio_state[5] = np.std(self.specialist_pnl_history[-10:])  # P&L volatility
            portfolio_state[6] = 1.0 if self.specialist_pnl_history[-1] > 0 else 0.0  # Positive P&L
        
        # Trade frequency
        if len(self.trade_frequency_history) > 0:
            portfolio_state[7] = np.mean(self.trade_frequency_history[-10:])  # Recent trade frequency
        
        # Confidence history
        if len(self.confidence_history) > 0:
            portfolio_state[8] = np.mean(self.confidence_history[-10:])  # Recent confidence
            portfolio_state[9] = np.std(self.confidence_history[-10:])  # Confidence volatility
        
        return portfolio_state
    
    def _execute_action(self, action: np.ndarray) -> List:
        """Execute specialist action."""
        
        orders = []
        
        # Parse action (trading signals for each instrument)
        for i, instrument in enumerate(self.specialist_instruments):
            signal = action[i]
            
            # Store signal history
            self.signal_history[instrument].append(signal)
            
            # Convert signal to order if significant
            if abs(signal) > 0.1:  # Threshold for trading
                # Calculate position size based on signal strength
                position_size_result = self.position_sizer.calculate(
                    signal=signal,
                    portfolio_equity=self.portfolio_value,
                    instrument_volatility=self._get_instrument_volatility(instrument),
                    method='volatility'
                )
                
                # Extract position size from result
                position_size = position_size_result.position_size
                
                # Create order
                order = Order(
                    order_id=f"{instrument}_{self.current_step}",
                    agent_id=self.specialist.__class__.__name__,
                    symbol=instrument,
                    side='buy' if signal > 0 else 'sell',
                    order_type='market',
                    quantity=abs(position_size),
                    price=None,  # Market order
                    status='pending',
                    signal=signal
                )
                
                orders.append(order)
        
        # Update confidence history
        try:
            if self.specialist_instruments:
                market_state = self.get_market_state(self.specialist_instruments[0])
                confidence = self.specialist.calculate_confidence(market_state)
                self.confidence_history.append(confidence)
        except Exception:
            self.confidence_history.append(0.5)
        
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
    
    def _calculate_reward(self, executed_orders: List) -> np.ndarray:
        """Calculate reward for specialist."""
        
        if self.current_step == 0:
            return np.array([0.0], dtype=np.float32)
        
        # Base reward from specialist P&L
        specialist_pnl = 0.0
        for instrument in self.specialist_instruments:
            if instrument in self.positions:
                specialist_pnl += self.positions[instrument].unrealized_pnl
        
        self.specialist_pnl_history.append(specialist_pnl)
        
        # Calculate P&L-based reward
        if len(self.specialist_pnl_history) > 1:
            pnl_change = self.specialist_pnl_history[-1] - self.specialist_pnl_history[-2]
            reward = pnl_change / max(self.portfolio_value, 1.0)
        else:
            reward = 0.0
        
        # Transaction cost penalty
        total_transaction_cost = sum(order.transaction_cost for order in executed_orders)
        transaction_penalty = total_transaction_cost / max(self.portfolio_value, 1.0)
        reward -= self.config.transaction_cost_weight * transaction_penalty
        
        # Confidence bonus
        if len(self.confidence_history) > 0:
            confidence = self.confidence_history[-1]
            if specialist_pnl > 0:
                reward += confidence * 0.01  # Bonus for confident profitable trades
            else:
                reward -= (1 - confidence) * 0.005  # Penalty for confident losing trades
        
        # Trade frequency penalty (encourage quality over quantity)
        trade_frequency = len(executed_orders) / max(self.current_step, 1)
        self.trade_frequency_history.append(trade_frequency)
        
        if trade_frequency > 0.1:  # More than 10% of steps
            reward -= 0.001  # Small penalty for overtrading
        
        # Diversification bonus
        num_positions = sum(1 for instrument in self.specialist_instruments 
                          if instrument in self.positions)
        diversification_bonus = num_positions / len(self.specialist_instruments)
        reward += self.config.diversification_bonus_weight * diversification_bonus * 0.005
        
        return np.array([reward], dtype=np.float32)
    
    def get_signal_history(self) -> Dict[str, List[float]]:
        """Get signal history for analysis."""
        return {instrument: signals.copy() for instrument, signals in self.signal_history.items()}
    
    def get_confidence_history(self) -> List[float]:
        """Get confidence history for analysis."""
        return self.confidence_history.copy()
    
    def get_specialist_performance(self) -> Dict[str, Any]:
        """Get specialist performance metrics."""
        return {
            'pnl_history': self.specialist_pnl_history.copy(),
            'trade_frequency_history': self.trade_frequency_history.copy(),
            'confidence_history': self.confidence_history.copy(),
            'signal_history': self.get_signal_history(),
            'current_pnl': self.specialist_pnl_history[-1] if self.specialist_pnl_history else 0.0,
            'avg_confidence': np.mean(self.confidence_history) if self.confidence_history else 0.5
        }
