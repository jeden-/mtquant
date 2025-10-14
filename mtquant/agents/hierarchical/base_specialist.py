"""
Base Specialist for Hierarchical Multi-Agent Trading System

This module implements the abstract base class for all specialists:
- Shared functionality for market observation
- Instrument management utilities
- Common neural network components
- Abstract methods that all specialists must implement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class MarketState:
    """Market state representation for specialist input."""
    market_features: torch.Tensor  # Domain-specific market features
    timestamp: float
    regime: str  # Market regime detected by meta-controller


@dataclass
class InstrumentState:
    """Individual instrument state."""
    symbol: str
    observation: torch.Tensor  # Environment observation
    position: Optional[Dict] = None  # Current position info
    performance: Optional[Dict] = None  # Performance metrics


class BaseSpecialist(nn.Module, ABC):
    """
    Abstract base class for all specialists in hierarchical trading system.
    
    Each specialist manages a domain of related instruments:
    - Forex: EURUSD, GBPUSD, USDJPY
    - Commodities: XAUUSD, WTIUSD
    - Equity: SPX500, NAS100, US30
    
    Architecture:
    - Domain encoder: processes global market conditions
    - Instrument heads: individual action heads per instrument
    - Value head: state value estimation
    - Shared utilities: normalization, anomaly detection
    """
    
    def __init__(
        self,
        instruments: List[str],
        market_features_dim: int,
        observation_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        device: str = "cpu"
    ):
        """
        Initialize BaseSpecialist.
        
        Args:
            instruments: List of instrument symbols managed by this specialist
            market_features_dim: Dimension of domain-specific market features
            observation_dim: Dimension of individual instrument observations
            hidden_dim: Hidden layer dimension for instrument heads
            dropout: Dropout rate
            device: Device to run on
        """
        super().__init__()
        
        self.instruments = instruments
        self.market_features_dim = market_features_dim
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        
        # Domain encoder: processes global market conditions
        self.domain_encoder = nn.Sequential(
            nn.Linear(market_features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Value head: shared across all instruments
        self.value_head = nn.Linear(128, 1)
        
        # Move to device
        self.to(device)
    
    @property
    def specialist_type(self) -> str:
        """Return specialist type (to be implemented by subclasses)."""
        return self.__class__.__name__.lower().replace('specialist', '')
    
    @property
    def instrument_count(self) -> int:
        """Return number of instruments managed by this specialist."""
        return len(self.instruments)
    
    @abstractmethod
    def forward(
        self,
        market_state: torch.Tensor,
        instrument_states: Dict[str, torch.Tensor],
        allocation: float
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through specialist.
        
        Args:
            market_state: (batch, market_features_dim) - Global market conditions
            instrument_states: Dict mapping instrument symbols to observations
            allocation: Scalar 0-1 from meta-controller
            
        Returns:
            actions: Dict mapping instrument symbols to action probabilities
            value: (batch, 1) - State value estimate
        """
        pass
    
    @abstractmethod
    def get_instruments(self) -> List[str]:
        """
        Get list of instruments managed by this specialist.
        
        Returns:
            instruments: List of instrument symbols
        """
        return self.instruments
    
    @abstractmethod
    def get_domain_features(self, market_data: Dict[str, Any]) -> torch.Tensor:
        """
        Extract domain-specific features from market data.
        
        Args:
            market_data: Raw market data dictionary
            
        Returns:
            features: (market_features_dim,) tensor
        """
        pass
    
    @abstractmethod
    def calculate_confidence(self, actions: Dict[str, torch.Tensor]) -> float:
        """
        Calculate confidence score for proposed actions.
        
        Args:
            actions: Dict of action probabilities per instrument
            
        Returns:
            confidence: Float 0-1 indicating confidence level
        """
        pass
    
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features to 0-1 range using min-max scaling.
        
        Args:
            features: Input features tensor
            
        Returns:
            normalized_features: Features normalized to [0, 1]
        """
        # Min-max normalization
        min_vals = features.min(dim=0, keepdim=True)[0]
        max_vals = features.max(dim=0, keepdim=True)[0]
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
        
        normalized = (features - min_vals) / range_vals
        return normalized
    
    def detect_anomalies(self, observations: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        """
        Detect anomalous observations for each instrument.
        
        Args:
            observations: Dict of observations per instrument
            
        Returns:
            anomalies: Dict mapping instrument symbols to anomaly flags
        """
        anomalies = {}
        
        for symbol, obs in observations.items():
            # Simple anomaly detection: check for extreme values
            if torch.any(torch.abs(obs) > 5.0):  # More than 5 standard deviations
                anomalies[symbol] = True
            else:
                anomalies[symbol] = False
        
        return anomalies
    
    def scale_actions_by_allocation(
        self,
        actions: Dict[str, torch.Tensor],
        allocation: float
    ) -> Dict[str, torch.Tensor]:
        """
        Scale action probabilities based on allocation from meta-controller.
        
        Args:
            actions: Dict of action probabilities per instrument
            allocation: Allocation amount (0-1)
            
        Returns:
            scaled_actions: Actions scaled by allocation
        """
        scaled_actions = {}
        
        for symbol, action_probs in actions.items():
            # Scale non-hold actions by allocation
            # Hold action (index 1) remains unchanged
            scaled_action = action_probs.clone()
            
            # Scale buy and sell actions
            scaled_action[:, 0] *= allocation  # Buy action
            scaled_action[:, 2] *= allocation  # Sell action
            
            # Renormalize to ensure probabilities sum to 1
            scaled_action = F.softmax(scaled_action, dim=-1)
            
            scaled_actions[symbol] = scaled_action
        
        return scaled_actions
    
    def get_action_info(
        self,
        market_state: torch.Tensor,
        instrument_states: Dict[str, torch.Tensor],
        allocation: float
    ) -> Dict[str, Any]:
        """
        Get detailed action information from specialist.
        
        Args:
            market_state: Market state tensor
            instrument_states: Instrument observations
            allocation: Allocation from meta-controller
            
        Returns:
            action_info: Dictionary with actions, confidence, and metadata
        """
        with torch.no_grad():
            actions, value = self.forward(market_state, instrument_states, allocation)
            confidence = self.calculate_confidence(actions)
            
            # Convert to numpy for easier handling
            actions_np = {}
            for symbol, action_tensor in actions.items():
                actions_np[symbol] = action_tensor.cpu().numpy()
            
            value_np = value.cpu().numpy()
            
            return {
                'actions': actions_np,
                'value': value_np,
                'confidence': confidence,
                'allocation': allocation,
                'specialist_type': self.specialist_type,
                'instruments': self.instruments
            }
    
    def validate_inputs(
        self,
        market_state: torch.Tensor,
        instrument_states: Dict[str, torch.Tensor],
        allocation: float
    ) -> bool:
        """
        Validate input tensors and parameters.
        
        Args:
            market_state: Market state tensor
            instrument_states: Instrument observations
            allocation: Allocation parameter
            
        Returns:
            is_valid: True if inputs are valid
        """
        # Check market state shape
        if market_state.shape[-1] != self.market_features_dim:
            return False
        
        # Check allocation range
        if not (0.0 <= allocation <= 1.0):
            return False
        
        # Check instrument states
        for symbol, obs in instrument_states.items():
            if symbol not in self.instruments:
                return False
            if obs.shape[-1] != self.observation_dim:
                return False
        
        return True
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for this specialist.
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        # This should be implemented by subclasses with actual performance tracking
        return {
            'sharpe_ratio': 0.0,
            'win_rate': 0.5,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'avg_trade_duration': 0.0
        }


# Unit test stubs (to be implemented in test files)
"""
def test_base_specialist_initialization():
    '''Test BaseSpecialist initialization.'''
    # This will fail since BaseSpecialist is abstract
    # But we can test the initialization parameters
    pass

def test_domain_encoder():
    '''Test domain encoder forward pass.'''
    # Test with concrete subclass
    pass

def test_feature_normalization():
    '''Test feature normalization utility.'''
    specialist = ConcreteSpecialist()  # Concrete implementation
    features = torch.randn(10, 5)
    normalized = specialist.normalize_features(features)
    
    assert torch.all(normalized >= 0)
    assert torch.all(normalized <= 1)

def test_anomaly_detection():
    '''Test anomaly detection utility.'''
    specialist = ConcreteSpecialist()
    observations = {
        'EURUSD': torch.randn(10, 20),
        'GBPUSD': torch.randn(10, 20)
    }
    anomalies = specialist.detect_anomalies(observations)
    
    assert isinstance(anomalies, dict)
    assert all(isinstance(v, bool) for v in anomalies.values())

def test_action_scaling():
    '''Test action scaling by allocation.'''
    specialist = ConcreteSpecialist()
    actions = {
        'EURUSD': torch.tensor([[0.3, 0.4, 0.3]]),
        'GBPUSD': torch.tensor([[0.2, 0.6, 0.2]])
    }
    allocation = 0.5
    
    scaled = specialist.scale_actions_by_allocation(actions, allocation)
    
    assert len(scaled) == len(actions)
    for symbol in scaled:
        assert torch.allclose(scaled[symbol].sum(dim=-1), torch.ones(1))

def test_input_validation():
    '''Test input validation.'''
    specialist = ConcreteSpecialist()
    
    # Valid inputs
    market_state = torch.randn(1, specialist.market_features_dim)
    instrument_states = {
        symbol: torch.randn(1, specialist.observation_dim)
        for symbol in specialist.instruments
    }
    allocation = 0.5
    
    assert specialist.validate_inputs(market_state, instrument_states, allocation)
    
    # Invalid allocation
    assert not specialist.validate_inputs(market_state, instrument_states, 1.5)
    
    # Invalid market state dimension
    invalid_market = torch.randn(1, 10)
    assert not specialist.validate_inputs(invalid_market, instrument_states, allocation)
"""
