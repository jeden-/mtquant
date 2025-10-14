"""
Forex Specialist for Hierarchical Multi-Agent Trading System

This module implements the Forex domain specialist that manages:
- EURUSD, GBPUSD, USDJPY currency pairs
- Shared FX market understanding
- Individual action heads per currency pair
- FX-specific feature extraction and analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

from .base_specialist import BaseSpecialist


class ForexSpecialist(BaseSpecialist):
    """
    Forex Specialist for managing EURUSD, GBPUSD, USDJPY.
    
    This specialist understands:
    - Global FX market conditions (DXY, interest rate spreads)
    - Carry trade attractiveness
    - FX volatility patterns
    - Central bank policy impacts
    
    Architecture:
    - FX encoder: processes global FX market state
    - Instrument heads: individual action heads for each currency pair
    - Value head: shared state value estimation
    """
    
    def __init__(
        self,
        instruments: List[str] = None,
        market_features_dim: int = 8,
        observation_dim: int = 50,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        device: str = "cpu"
    ):
        """
        Initialize ForexSpecialist.
        
        Args:
            instruments: List of FX instruments (default: EURUSD, GBPUSD, USDJPY)
            market_features_dim: Dimension of FX market features
            observation_dim: Dimension of individual instrument observations
            hidden_dim: Hidden layer dimension for instrument heads
            dropout: Dropout rate
            device: Device to run on
        """
        if instruments is None:
            instruments = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        super().__init__(
            instruments=instruments,
            market_features_dim=market_features_dim,
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=device
        )
        
        # Individual instrument heads (one per currency pair)
        self.instrument_heads = nn.ModuleDict()
        
        for instrument in self.instruments:
            # Each head takes: FX features (128) + instrument obs (observation_dim)
            input_dim = 128 + observation_dim
            self.instrument_heads[instrument] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 3)  # 3 actions: buy, hold, sell
            )
    
    def forward(
        self,
        market_state: torch.Tensor,
        instrument_states: Dict[str, torch.Tensor],
        allocation: float
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through Forex specialist.
        
        Args:
            market_state: (batch, market_features_dim) - Global FX conditions
            instrument_states: {
                'EURUSD': (batch, obs_dim),
                'GBPUSD': (batch, obs_dim),
                'USDJPY': (batch, obs_dim)
            }
            allocation: Scalar 0-1 from meta-controller
        
        Returns:
            actions: {
                'EURUSD': (batch, 3) probabilities,
                'GBPUSD': (batch, 3) probabilities,
                'USDJPY': (batch, 3) probabilities
            }
            value: (batch, 1)
        """
        # Validate inputs
        if not self.validate_inputs(market_state, instrument_states, allocation):
            raise ValueError("Invalid inputs to ForexSpecialist")
        
        # Encode global FX market state
        fx_features = self.domain_encoder(market_state)  # (batch, 128)
        
        # Generate actions for each instrument
        actions = {}
        for instrument in self.instruments:
            if instrument not in instrument_states:
                raise ValueError(f"Missing instrument state for {instrument}")
            
            # Concatenate FX features with instrument-specific observations
            instrument_obs = instrument_states[instrument]
            combined_input = torch.cat([fx_features, instrument_obs], dim=-1)
            
            # Generate action logits
            action_logits = self.instrument_heads[instrument](combined_input)
            
            # Convert to probabilities
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Scale by allocation
            scaled_probs = self.scale_actions_by_allocation(
                {instrument: action_probs}, allocation
            )[instrument]
            
            actions[instrument] = scaled_probs
        
        # Generate value estimate
        value = self.value_head(fx_features)
        
        return actions, value
    
    def get_instruments(self) -> List[str]:
        """Get list of FX instruments managed by this specialist."""
        return self.instruments
    
    def get_domain_features(self, market_data: Dict[str, Any]) -> torch.Tensor:
        """
        Extract FX-specific features from market data.
        
        Args:
            market_data: Raw market data dictionary
            
        Returns:
            features: (market_features_dim,) tensor with FX market features
        """
        # Extract DXY (Dollar Index)
        dxy = market_data.get('dxy', 100.0)
        
        # Calculate interest rate spreads (US vs EUR, GBP, JPY)
        us_rate = market_data.get('us_rate', 5.25)
        eur_rate = market_data.get('eur_rate', 4.0)
        gbp_rate = market_data.get('gbp_rate', 5.0)
        jpy_rate = market_data.get('jpy_rate', 0.1)
        
        eur_spread = us_rate - eur_rate
        gbp_spread = us_rate - gbp_rate
        jpy_spread = us_rate - jpy_rate
        
        # Compute carry trade attractiveness (average spread)
        carry_attractiveness = (eur_spread + gbp_spread + jpy_spread) / 3.0
        
        # FX volatility index (average ATR across pairs)
        fx_volatility = market_data.get('fx_volatility', 0.01)
        
        # Risk-on/Risk-off indicator (VIX proxy for FX)
        risk_sentiment = market_data.get('risk_sentiment', 0.5)  # 0=risk-off, 1=risk-on
        
        # Central bank policy stance (dovish/hawkish)
        cb_policy_stance = market_data.get('cb_policy_stance', 0.5)  # 0=dovish, 1=hawkish
        
        # Features vector
        features = torch.tensor([
            dxy,
            eur_spread,
            gbp_spread,
            jpy_spread,
            carry_attractiveness,
            fx_volatility,
            risk_sentiment,
            cb_policy_stance
        ], dtype=torch.float32)
        
        return features
    
    def calculate_confidence(self, actions: Dict[str, torch.Tensor]) -> float:
        """
        Calculate confidence score for proposed FX actions.
        
        Args:
            actions: Dict of action probabilities per instrument
            
        Returns:
            confidence: Float 0-1 indicating confidence level
        """
        if not actions:
            return 0.0
        
        # Calculate confidence based on action certainty
        total_confidence = 0.0
        count = 0
        
        for instrument, action_probs in actions.items():
            if instrument not in self.instruments:
                continue
            
            # Confidence = max probability (how certain the model is)
            max_prob = torch.max(action_probs, dim=-1)[0]
            avg_confidence = torch.mean(max_prob).item()
            
            total_confidence += avg_confidence
            count += 1
        
        if count == 0:
            return 0.0
        
        return total_confidence / count
    
    def detect_correlation_regime(self, returns: Dict[str, float]) -> str:
        """
        Detect FX correlation regime.
        
        Args:
            returns: Dict of recent returns per instrument
            
        Returns:
            regime: 'risk-on', 'risk-off', 'neutral'
        """
        if len(returns) < 2:
            return 'neutral'
        
        # Calculate average correlation between pairs
        instruments = list(returns.keys())
        correlations = []
        
        for i in range(len(instruments)):
            for j in range(i + 1, len(instruments)):
                # Simple correlation proxy: similar direction = high correlation
                ret1 = returns[instruments[i]]
                ret2 = returns[instruments[j]]
                
                if ret1 * ret2 > 0:  # Same direction
                    correlations.append(1.0)
                else:  # Opposite direction
                    correlations.append(-1.0)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # Regime detection
        if avg_correlation > 0.3:
            return 'risk-on'  # High positive correlation
        elif avg_correlation < -0.3:
            return 'risk-off'  # High negative correlation
        else:
            return 'neutral'
    
    def get_carry_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Get carry trade signal based on interest rate differentials.
        
        Args:
            market_data: Market data with interest rates
            
        Returns:
            signal: -1 to 1 (negative = short carry, positive = long carry)
        """
        # Get interest rates
        us_rate = market_data.get('us_rate', 5.25)
        eur_rate = market_data.get('eur_rate', 4.0)
        gbp_rate = market_data.get('gbp_rate', 5.0)
        jpy_rate = market_data.get('jpy_rate', 0.1)
        
        # Calculate average spread
        spreads = [us_rate - eur_rate, us_rate - gbp_rate, us_rate - jpy_rate]
        avg_spread = np.mean(spreads)
        
        # Normalize to -1 to 1 range
        # Positive spread = long USD carry trade
        # Negative spread = short USD carry trade
        signal = np.tanh(avg_spread / 2.0)  # Scale by 2% for normalization
        
        return float(signal)
    
    def check_central_bank_schedule(self, current_date: datetime = None) -> Dict[str, bool]:
        """
        Check for upcoming central bank meetings/decisions.
        
        Args:
            current_date: Current date (default: now)
            
        Returns:
            schedule: Dict mapping central banks to upcoming meeting flags
        """
        if current_date is None:
            current_date = datetime.now()
        
        # Simple heuristic: check if we're in the first week of month
        # (when most central banks meet)
        is_first_week = current_date.day <= 7
        
        return {
            'fed_meeting': is_first_week and current_date.month % 3 == 0,  # Quarterly
            'ecb_meeting': is_first_week and current_date.month % 2 == 0,   # Bi-monthly
            'boe_meeting': is_first_week and current_date.month % 2 == 1,   # Bi-monthly
            'boj_meeting': is_first_week and current_date.month % 6 == 0    # Semi-annual
        }
    
    def get_fx_specific_metrics(self) -> Dict[str, float]:
        """
        Get FX-specific performance metrics.
        
        Returns:
            metrics: Dictionary of FX-specific metrics
        """
        base_metrics = self.get_performance_metrics()
        
        # Add FX-specific metrics
        fx_metrics = {
            **base_metrics,
            'carry_trade_pnl': 0.0,
            'fx_volatility_exposure': 0.0,
            'correlation_with_dxy': 0.0,
            'central_bank_impact': 0.0
        }
        
        return fx_metrics
    
    def analyze_fx_market_structure(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze FX market structure and provide insights.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            analysis: Dictionary with FX market analysis
        """
        # Get domain features
        features = self.get_domain_features(market_data)
        
        # Analyze market structure
        analysis = {
            'dxy_level': features[0].item(),
            'interest_rate_spreads': {
                'eur': features[1].item(),
                'gbp': features[2].item(),
                'jpy': features[3].item()
            },
            'carry_attractiveness': features[4].item(),
            'fx_volatility': features[5].item(),
            'risk_sentiment': features[6].item(),
            'cb_policy_stance': features[7].item(),
            'correlation_regime': self.detect_correlation_regime(
                market_data.get('returns', {})
            ),
            'carry_signal': self.get_carry_signal(market_data),
            'cb_schedule': self.check_central_bank_schedule()
        }
        
        return analysis


# Unit test stubs (to be implemented in test files)
"""
def test_forex_specialist_initialization():
    '''Test ForexSpecialist initialization.'''
    specialist = ForexSpecialist()
    
    assert specialist.specialist_type == 'forex'
    assert specialist.instrument_count == 3
    assert specialist.instruments == ['EURUSD', 'GBPUSD', 'USDJPY']
    assert len(specialist.instrument_heads) == 3

def test_forex_specialist_forward():
    '''Test ForexSpecialist forward pass.'''
    specialist = ForexSpecialist()
    batch_size = 2
    
    market_state = torch.randn(batch_size, 8)
    instrument_states = {
        'EURUSD': torch.randn(batch_size, 50),
        'GBPUSD': torch.randn(batch_size, 50),
        'USDJPY': torch.randn(batch_size, 50)
    }
    allocation = 0.5
    
    actions, value = specialist.forward(market_state, instrument_states, allocation)
    
    assert len(actions) == 3
    assert 'EURUSD' in actions
    assert 'GBPUSD' in actions
    assert 'USDJPY' in actions
    
    for instrument, action_probs in actions.items():
        assert action_probs.shape == (batch_size, 3)
        assert torch.allclose(action_probs.sum(dim=-1), torch.ones(batch_size))
    
    assert value.shape == (batch_size, 1)

def test_domain_features_extraction():
    '''Test FX domain features extraction.'''
    specialist = ForexSpecialist()
    
    market_data = {
        'dxy': 105.0,
        'us_rate': 5.25,
        'eur_rate': 4.0,
        'gbp_rate': 5.0,
        'jpy_rate': 0.1,
        'fx_volatility': 0.015,
        'risk_sentiment': 0.7,
        'cb_policy_stance': 0.6
    }
    
    features = specialist.get_domain_features(market_data)
    assert features.shape == (8,)
    assert features[0] == 105.0  # DXY
    assert features[1] == 1.25  # EUR spread

def test_correlation_regime_detection():
    '''Test FX correlation regime detection.'''
    specialist = ForexSpecialist()
    
    # Risk-on scenario (all positive returns)
    risk_on_returns = {'EURUSD': 0.01, 'GBPUSD': 0.02, 'USDJPY': 0.015}
    assert specialist.detect_correlation_regime(risk_on_returns) == 'risk-on'
    
    # Risk-off scenario (mixed returns)
    risk_off_returns = {'EURUSD': -0.01, 'GBPUSD': 0.02, 'USDJPY': -0.015}
    assert specialist.detect_correlation_regime(risk_off_returns) == 'risk-off'

def test_carry_signal():
    '''Test carry trade signal calculation.'''
    specialist = ForexSpecialist()
    
    market_data = {
        'us_rate': 5.25,
        'eur_rate': 4.0,
        'gbp_rate': 5.0,
        'jpy_rate': 0.1
    }
    
    signal = specialist.get_carry_signal(market_data)
    assert -1.0 <= signal <= 1.0
    assert signal > 0  # Positive spread should give positive signal

def test_central_bank_schedule():
    '''Test central bank schedule checking.'''
    specialist = ForexSpecialist()
    
    # Test with specific date
    test_date = datetime(2024, 3, 5)  # First week of March
    schedule = specialist.check_central_bank_schedule(test_date)
    
    assert isinstance(schedule, dict)
    assert 'fed_meeting' in schedule
    assert 'ecb_meeting' in schedule
    assert 'boe_meeting' in schedule
    assert 'boj_meeting' in schedule
"""
