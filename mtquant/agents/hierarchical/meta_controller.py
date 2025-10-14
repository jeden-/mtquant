"""
Meta-Controller for Hierarchical Multi-Agent Trading System

This module implements the portfolio-level decision maker that:
- Observes portfolio state (74 features)
- Allocates capital to 3 specialists (Forex, Commodities, Equity)
- Manages risk appetite (0-1 continuous value)
- Detects market regimes
- Provides value estimates for PPO training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PortfolioState:
    """Portfolio state representation for meta-controller input."""
    portfolio_returns: np.ndarray  # Last 30 days: shape (30,)
    portfolio_volatility: float
    current_drawdown: float
    correlation_matrix: np.ndarray  # Flattened upper triangle: shape (28,)
    specialist_performance: np.ndarray  # 3 specialists × 3 metrics: shape (9,)
    macro_indicators: np.ndarray  # VIX, DXY, rates: shape (5,)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for neural network input."""
        features = np.concatenate([
            self.portfolio_returns,
            [self.portfolio_volatility],
            [self.current_drawdown],
            self.correlation_matrix,
            self.specialist_performance,
            self.macro_indicators
        ])
        return torch.tensor(features, dtype=torch.float32)


class MetaController(nn.Module):
    """
    Meta-Controller for hierarchical trading system.
    
    This is the top-level decision maker that:
    1. Observes portfolio state (74 features)
    2. Allocates capital to 3 specialists
    3. Manages risk appetite
    4. Detects market regimes
    5. Provides value estimates for PPO training
    
    Architecture:
    - Input layer: Linear(74, 256) + ReLU + LayerNorm
    - Hidden: Linear(256, 128) + ReLU + Dropout(0.2)
    - Allocation head: Linear(128, 3) + Softmax
    - Risk head: Linear(128, 1) + Sigmoid
    - Value head: Linear(128, 1)
    """
    
    def __init__(
        self,
        state_dim: int = 74,
        hidden_dim: int = 256,
        hidden_dim_2: int = 128,
        dropout: float = 0.2,
        device: str = "cpu"
    ):
        """
        Initialize MetaController.
        
        Args:
            state_dim: Input state dimension (74 features)
            hidden_dim: First hidden layer dimension
            hidden_dim_2: Second hidden layer dimension
            dropout: Dropout rate
            device: Device to run on
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_2 = hidden_dim_2
        self.dropout = dropout
        self.device = device
        
        # Portfolio state encoder
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Hidden layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Policy heads
        self.allocation_head = nn.Linear(hidden_dim_2, 3)  # 3 specialists
        self.risk_appetite_head = nn.Linear(hidden_dim_2, 1)
        self.value_head = nn.Linear(hidden_dim_2, 1)
        
        # Move to device
        self.to(device)
    
    def forward(self, portfolio_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through meta-controller.
        
        Args:
            portfolio_state: (batch, 74) - Portfolio state tensor
            
        Returns:
            allocations: (batch, 3) - Softmax over specialists
            risk_appetite: (batch, 1) - Sigmoid (0=defensive, 1=aggressive)
            value: (batch, 1) - State value estimate
        """
        # Ensure input is on correct device
        portfolio_state = portfolio_state.to(self.device)
        
        # Encode portfolio state
        encoded = self.input_layer(portfolio_state)
        hidden = self.hidden_layer(encoded)
        
        # Generate outputs
        allocation_logits = self.allocation_head(hidden)
        allocations = F.softmax(allocation_logits, dim=-1)
        
        risk_appetite = torch.sigmoid(self.risk_appetite_head(hidden))
        
        value = self.value_head(hidden)
        
        return allocations, risk_appetite, value
    
    def get_portfolio_state(
        self,
        portfolio: Dict,
        specialists: Dict[str, Dict]
    ) -> torch.Tensor:
        """
        Extract portfolio state from portfolio and specialist data.
        
        Args:
            portfolio: Portfolio data with returns, volatility, drawdown
            specialists: Dict of specialist performance data
            
        Returns:
            portfolio_state: (74,) tensor ready for forward pass
            
        Example:
            >>> portfolio = {
            ...     'returns': np.random.randn(30),
            ...     'volatility': 0.15,
            ...     'drawdown': -0.05,
            ...     'correlation_matrix': np.random.rand(8, 8)
            ... }
            >>> specialists = {
            ...     'forex': {'sharpe': 1.2, 'win_rate': 0.6, 'max_dd': -0.08},
            ...     'commodities': {'sharpe': 0.8, 'win_rate': 0.55, 'max_dd': -0.12},
            ...     'equity': {'sharpe': 1.5, 'win_rate': 0.65, 'max_dd': -0.06}
            ... }
            >>> meta = MetaController()
            >>> state = meta.get_portfolio_state(portfolio, specialists)
            >>> print(state.shape)  # torch.Size([74])
        """
        # Portfolio returns (last 30 days)
        portfolio_returns = np.array(portfolio.get('returns', np.zeros(30)))
        if len(portfolio_returns) != 30:
            # Pad or truncate to 30 days
            if len(portfolio_returns) > 30:
                portfolio_returns = portfolio_returns[-30:]
            else:
                portfolio_returns = np.pad(portfolio_returns, (30 - len(portfolio_returns), 0))
        
        # Portfolio volatility
        portfolio_volatility = portfolio.get('volatility', 0.0)
        
        # Current drawdown
        current_drawdown = portfolio.get('drawdown', 0.0)
        
        # Correlation matrix (flattened upper triangle)
        corr_matrix = portfolio.get('correlation_matrix', np.eye(8))
        if corr_matrix.shape != (8, 8):
            corr_matrix = np.eye(8)  # Default to identity if wrong shape
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = []
        for i in range(8):
            for j in range(i + 1, 8):
                upper_triangle.append(corr_matrix[i, j])
        correlation_matrix = np.array(upper_triangle)  # Shape: (28,)
        
        # Specialist performance metrics (3 specialists × 3 metrics)
        specialist_performance = []
        specialist_order = ['forex', 'commodities', 'equity']
        
        for specialist_name in specialist_order:
            specialist_data = specialists.get(specialist_name, {})
            sharpe = specialist_data.get('sharpe', 0.0)
            win_rate = specialist_data.get('win_rate', 0.5)
            max_drawdown = specialist_data.get('max_drawdown', 0.0)
            
            specialist_performance.extend([sharpe, win_rate, max_drawdown])
        
        specialist_performance = np.array(specialist_performance)  # Shape: (9,)
        
        # Macro indicators (VIX, DXY, rates)
        macro_indicators = np.array([
            portfolio.get('vix', 20.0),  # VIX level
            portfolio.get('dxy', 100.0),  # Dollar Index
            portfolio.get('fed_rate', 5.25),  # Fed funds rate
            portfolio.get('10y_yield', 4.5),  # 10-year Treasury yield
            portfolio.get('credit_spread', 1.5)  # Credit spread
        ])
        
        # Create PortfolioState and convert to tensor
        portfolio_state = PortfolioState(
            portfolio_returns=portfolio_returns,
            portfolio_volatility=portfolio_volatility,
            current_drawdown=current_drawdown,
            correlation_matrix=correlation_matrix,
            specialist_performance=specialist_performance,
            macro_indicators=macro_indicators
        )
        
        return portfolio_state.to_tensor()
    
    def detect_market_regime(self, returns: np.ndarray) -> str:
        """
        Detect market regime based on recent returns.
        
        Args:
            returns: Recent returns array (last 30 days)
            
        Returns:
            regime: 'bull', 'bear', 'neutral', 'volatile'
            
        Example:
            >>> meta = MetaController()
            >>> returns = np.array([0.01, 0.02, -0.01, 0.015, ...])  # 30 days
            >>> regime = meta.detect_market_regime(returns)
            >>> print(regime)  # 'bull'
        """
        if len(returns) == 0:
            return 'neutral'
        
        # Calculate metrics
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Regime detection logic
        if volatility > 0.03:  # High volatility threshold
            return 'volatile'
        elif mean_return > 0.005:  # Positive trend
            return 'bull'
        elif mean_return < -0.005:  # Negative trend
            return 'bear'
        else:
            return 'neutral'
    
    def calculate_kelly_allocation(self, specialist_sharpes: np.ndarray) -> np.ndarray:
        """
        Calculate Kelly Criterion allocation to specialists.
        
        Args:
            specialist_sharpes: Sharpe ratios for each specialist
            
        Returns:
            allocation: Normalized allocation weights
            
        Example:
            >>> meta = MetaController()
            >>> sharpes = np.array([1.2, 0.8, 1.5])  # forex, commodities, equity
            >>> allocation = meta.calculate_kelly_allocation(sharpes)
            >>> print(allocation)  # [0.4, 0.2, 0.4] (normalized)
        """
        if len(specialist_sharpes) != 3:
            raise ValueError("Expected 3 specialist Sharpe ratios")
        
        # Kelly formula: f = (μ - r) / σ²
        # For Sharpe ratio: f ≈ Sharpe / 2 (simplified)
        kelly_weights = specialist_sharpes / 2.0
        
        # Ensure non-negative weights
        kelly_weights = np.maximum(kelly_weights, 0.0)
        
        # Normalize to sum to 1
        if np.sum(kelly_weights) > 0:
            kelly_weights = kelly_weights / np.sum(kelly_weights)
        else:
            # Equal allocation if all negative
            kelly_weights = np.ones(3) / 3.0
        
        return kelly_weights
    
    def get_action_info(self, portfolio_state: torch.Tensor) -> Dict:
        """
        Get detailed action information from meta-controller.
        
        Args:
            portfolio_state: Portfolio state tensor
            
        Returns:
            action_info: Dictionary with allocations, risk appetite, and metadata
        """
        with torch.no_grad():
            allocations, risk_appetite, value = self.forward(portfolio_state)
            
            # Convert to numpy for easier handling
            allocations_np = allocations.cpu().numpy()
            risk_appetite_np = risk_appetite.cpu().numpy()
            value_np = value.cpu().numpy()
            
            return {
                'allocations': allocations_np,
                'risk_appetite': risk_appetite_np,
                'value': value_np,
                'specialist_names': ['forex', 'commodities', 'equity'],
                'allocation_dict': {
                    'forex': float(allocations_np[0]),
                    'commodities': float(allocations_np[1]),
                    'equity': float(allocations_np[2])
                }
            }


# Unit test stubs (to be implemented in test files)
"""
def test_meta_controller_forward():
    '''Test MetaController forward pass produces correct shapes.'''
    meta = MetaController()
    batch_size = 4
    state = torch.randn(batch_size, 74)
    
    allocations, risk_appetite, value = meta.forward(state)
    
    assert allocations.shape == (batch_size, 3)
    assert risk_appetite.shape == (batch_size, 1)
    assert value.shape == (batch_size, 1)
    assert torch.allclose(allocations.sum(dim=1), torch.ones(batch_size))

def test_portfolio_state_extraction():
    '''Test portfolio state extraction from portfolio data.'''
    meta = MetaController()
    
    portfolio = {
        'returns': np.random.randn(30),
        'volatility': 0.15,
        'drawdown': -0.05,
        'correlation_matrix': np.random.rand(8, 8)
    }
    
    specialists = {
        'forex': {'sharpe': 1.2, 'win_rate': 0.6, 'max_drawdown': -0.08},
        'commodities': {'sharpe': 0.8, 'win_rate': 0.55, 'max_drawdown': -0.12},
        'equity': {'sharpe': 1.5, 'win_rate': 0.65, 'max_drawdown': -0.06}
    }
    
    state = meta.get_portfolio_state(portfolio, specialists)
    assert state.shape == (74,)

def test_market_regime_detection():
    '''Test market regime detection logic.'''
    meta = MetaController()
    
    # Bull market
    bull_returns = np.random.normal(0.01, 0.02, 30)
    assert meta.detect_market_regime(bull_returns) == 'bull'
    
    # Bear market
    bear_returns = np.random.normal(-0.01, 0.02, 30)
    assert meta.detect_market_regime(bear_returns) == 'bear'
    
    # Volatile market
    volatile_returns = np.random.normal(0.0, 0.05, 30)
    assert meta.detect_market_regime(volatile_returns) == 'volatile'

def test_kelly_allocation():
    '''Test Kelly Criterion allocation calculation.'''
    meta = MetaController()
    
    sharpes = np.array([1.2, 0.8, 1.5])
    allocation = meta.calculate_kelly_allocation(sharpes)
    
    assert len(allocation) == 3
    assert np.isclose(np.sum(allocation), 1.0)
    assert np.all(allocation >= 0)
"""
