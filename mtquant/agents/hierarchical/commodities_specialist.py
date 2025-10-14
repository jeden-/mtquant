"""
Commodities Specialist for Hierarchical Multi-Agent Trading System

This module implements the Commodities domain specialist that manages:
- XAUUSD (Gold), WTIUSD (Oil) commodity instruments
- Commodity market dynamics understanding
- Inflation, geopolitical risk awareness
- Supply/demand indicators
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

from .base_specialist import BaseSpecialist


class CommoditiesSpecialist(BaseSpecialist):
    """
    Commodities Specialist for managing XAUUSD (Gold) and WTIUSD (Oil).
    
    This specialist understands:
    - Global commodity market conditions
    - Inflation expectations and real yields
    - Geopolitical risk factors
    - Supply/demand dynamics
    - OPEC policy impacts (for oil)
    - Safe-haven demand (for gold)
    
    Architecture:
    - Commodity encoder: processes global commodity conditions
    - Instrument heads: individual action heads for each commodity
    - Value head: shared state value estimation
    """
    
    def __init__(
        self,
        instruments: List[str] = None,
        market_features_dim: int = 6,
        observation_dim: int = 50,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        device: str = "cpu"
    ):
        """
        Initialize CommoditiesSpecialist.
        
        Args:
            instruments: List of commodity instruments (default: XAUUSD, WTIUSD)
            market_features_dim: Dimension of commodity market features
            observation_dim: Dimension of individual instrument observations
            hidden_dim: Hidden layer dimension for instrument heads
            dropout: Dropout rate
            device: Device to run on
        """
        if instruments is None:
            instruments = ['XAUUSD', 'WTIUSD']
        
        super().__init__(
            instruments=instruments,
            market_features_dim=market_features_dim,
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=device
        )
        
        # Individual instrument heads (one per commodity)
        self.instrument_heads = nn.ModuleDict()
        
        for instrument in self.instruments:
            # Each head takes: commodity features (128) + instrument obs (observation_dim)
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
        Forward pass through Commodities specialist.
        
        Args:
            market_state: (batch, market_features_dim) - Global commodity conditions
            instrument_states: {
                'XAUUSD': (batch, obs_dim),
                'WTIUSD': (batch, obs_dim)
            }
            allocation: Scalar 0-1 from meta-controller
        
        Returns:
            actions: {
                'XAUUSD': (batch, 3) probabilities,
                'WTIUSD': (batch, 3) probabilities
            }
            value: (batch, 1)
        """
        # Validate inputs
        if not self.validate_inputs(market_state, instrument_states, allocation):
            raise ValueError("Invalid inputs to CommoditiesSpecialist")
        
        # Encode global commodity market state
        commodity_features = self.domain_encoder(market_state)  # (batch, 128)
        
        # Generate actions for each instrument
        actions = {}
        for instrument in self.instruments:
            if instrument not in instrument_states:
                raise ValueError(f"Missing instrument state for {instrument}")
            
            # Concatenate commodity features with instrument-specific observations
            instrument_obs = instrument_states[instrument]
            combined_input = torch.cat([commodity_features, instrument_obs], dim=-1)
            
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
        value = self.value_head(commodity_features)
        
        return actions, value
    
    def get_instruments(self) -> List[str]:
        """Get list of commodity instruments managed by this specialist."""
        return self.instruments
    
    def get_domain_features(self, market_data: Dict[str, Any]) -> torch.Tensor:
        """
        Extract commodity-specific features from market data.
        
        Args:
            market_data: Raw market data dictionary
            
        Returns:
            features: (market_features_dim,) tensor with commodity market features
        """
        # Get inflation expectations (CPI, PPI)
        cpi_expectation = market_data.get('cpi_expectation', 2.5)  # Annual CPI %
        ppi_expectation = market_data.get('ppi_expectation', 2.0)  # Annual PPI %
        inflation_expectation = (cpi_expectation + ppi_expectation) / 2.0
        
        # Calculate geopolitical risk score (VIX × news sentiment)
        vix_level = market_data.get('vix', 20.0)
        news_sentiment = market_data.get('news_sentiment', 0.5)  # 0=negative, 1=positive
        geopolitical_risk = vix_level * (1.0 - news_sentiment) / 100.0  # Normalize
        
        # Compute supply/demand indicators (inventories, production)
        oil_inventories = market_data.get('oil_inventories', 0.0)  # Change in inventories
        gold_production = market_data.get('gold_production', 0.0)  # Production change
        supply_demand_imbalance = (oil_inventories + gold_production) / 2.0
        
        # Commodity index level (broad commodity basket)
        commodity_index = market_data.get('commodity_index', 100.0)  # CRB or similar
        
        # Real yields (nominal - inflation) - important for gold
        nominal_10y = market_data.get('nominal_10y', 4.5)
        real_yield = nominal_10y - inflation_expectation
        
        # Dollar strength (inverse correlation with commodities)
        dxy_level = market_data.get('dxy', 100.0)
        dollar_strength = (dxy_level - 100.0) / 100.0  # Normalize around 100
        
        # Features vector
        features = torch.tensor([
            inflation_expectation,
            geopolitical_risk,
            supply_demand_imbalance,
            commodity_index / 100.0,  # Normalize
            real_yield,
            dollar_strength
        ], dtype=torch.float32)
        
        return features
    
    def calculate_confidence(self, actions: Dict[str, torch.Tensor]) -> float:
        """
        Calculate confidence score for proposed commodity actions.
        
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
    
    def detect_inflation_regime(self, inflation_data: Dict[str, float]) -> str:
        """
        Detect inflation regime based on various indicators.
        
        Args:
            inflation_data: Dictionary with inflation indicators
            
        Returns:
            regime: 'deflation', 'stable', 'rising', 'hyperinflation'
        """
        cpi = inflation_data.get('cpi', 2.0)
        ppi = inflation_data.get('ppi', 2.0)
        core_cpi = inflation_data.get('core_cpi', 2.0)
        
        # Average inflation rate
        avg_inflation = (cpi + ppi + core_cpi) / 3.0
        
        # Regime detection
        if avg_inflation < 0:
            return 'deflation'
        elif avg_inflation < 1.5:
            return 'stable'
        elif avg_inflation < 5.0:
            return 'rising'
        else:
            return 'hyperinflation'
    
    def get_safe_haven_demand(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate safe-haven demand for gold based on market conditions.
        
        Args:
            market_data: Market data with risk indicators
            
        Returns:
            demand: 0-1 (0=low safe-haven demand, 1=high safe-haven demand)
        """
        # Risk indicators
        vix = market_data.get('vix', 20.0)
        credit_spread = market_data.get('credit_spread', 1.5)
        geopolitical_tension = market_data.get('geopolitical_tension', 0.5)
        
        # Normalize indicators
        vix_normalized = min(vix / 50.0, 1.0)  # Cap at 50 VIX
        credit_spread_normalized = min(credit_spread / 5.0, 1.0)  # Cap at 5%
        
        # Calculate safe-haven demand
        demand = (vix_normalized + credit_spread_normalized + geopolitical_tension) / 3.0
        
        return float(np.clip(demand, 0.0, 1.0))
    
    def check_opec_schedule(self, current_date: datetime = None) -> bool:
        """
        Check for upcoming OPEC meetings/decisions.
        
        Args:
            current_date: Current date (default: now)
            
        Returns:
            opec_meeting: True if OPEC meeting is upcoming
        """
        if current_date is None:
            current_date = datetime.now()
        
        # OPEC typically meets every 6 months (June and December)
        # Check if we're in the month before or during OPEC meeting
        opec_months = [5, 6, 11, 12]  # May, June, November, December
        
        return current_date.month in opec_months
    
    def analyze_commodity_fundamentals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze commodity market fundamentals.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            analysis: Dictionary with commodity market analysis
        """
        # Get domain features
        features = self.get_domain_features(market_data)
        
        # Analyze commodity fundamentals
        analysis = {
            'inflation_expectation': features[0].item(),
            'geopolitical_risk': features[1].item(),
            'supply_demand_imbalance': features[2].item(),
            'commodity_index_level': features[3].item() * 100.0,  # Denormalize
            'real_yield': features[4].item(),
            'dollar_strength': features[5].item(),
            'inflation_regime': self.detect_inflation_regime(
                market_data.get('inflation_data', {})
            ),
            'safe_haven_demand': self.get_safe_haven_demand(market_data),
            'opec_meeting_upcoming': self.check_opec_schedule()
        }
        
        return analysis
    
    def get_commodity_specific_metrics(self) -> Dict[str, float]:
        """
        Get commodity-specific performance metrics.
        
        Returns:
            metrics: Dictionary of commodity-specific metrics
        """
        base_metrics = self.get_performance_metrics()
        
        # Add commodity-specific metrics
        commodity_metrics = {
            **base_metrics,
            'inflation_hedge_pnl': 0.0,
            'safe_haven_pnl': 0.0,
            'supply_demand_pnl': 0.0,
            'geopolitical_risk_exposure': 0.0,
            'real_yield_correlation': 0.0
        }
        
        return commodity_metrics
    
    def get_gold_specific_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get gold-specific analysis and signals.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            analysis: Gold-specific analysis
        """
        # Gold-specific factors
        real_yield = market_data.get('nominal_10y', 4.5) - market_data.get('cpi_expectation', 2.5)
        dollar_strength = (market_data.get('dxy', 100.0) - 100.0) / 100.0
        safe_haven_demand = self.get_safe_haven_demand(market_data)
        
        # Gold signals
        gold_analysis = {
            'real_yield_impact': -real_yield,  # Negative correlation
            'dollar_impact': -dollar_strength,  # Negative correlation
            'safe_haven_signal': safe_haven_demand,
            'inflation_hedge_signal': market_data.get('cpi_expectation', 2.5) / 10.0,  # Normalize
            'central_bank_demand': market_data.get('cb_gold_demand', 0.5),
            'jewelry_demand': market_data.get('jewelry_demand', 0.5),
            'investment_demand': market_data.get('investment_demand', 0.5)
        }
        
        return gold_analysis
    
    def get_oil_specific_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get oil-specific analysis and signals.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            analysis: Oil-specific analysis
        """
        # Oil-specific factors
        inventories = market_data.get('oil_inventories', 0.0)
        production = market_data.get('oil_production', 0.0)
        demand_forecast = market_data.get('oil_demand_forecast', 0.0)
        opec_meeting = self.check_opec_schedule()
        
        # Oil signals
        oil_analysis = {
            'inventory_impact': -inventories,  # Negative correlation with price
            'production_impact': -production,  # Negative correlation
            'demand_impact': demand_forecast,  # Positive correlation
            'opec_meeting_impact': 0.1 if opec_meeting else 0.0,
            'geopolitical_risk_impact': market_data.get('geopolitical_risk', 0.5),
            'seasonal_demand': market_data.get('seasonal_demand', 0.5),
            'refinery_utilization': market_data.get('refinery_utilization', 0.8)
        }
        
        return oil_analysis


# Unit test stubs (to be implemented in test files)
"""
def test_commodities_specialist_initialization():
    '''Test CommoditiesSpecialist initialization.'''
    specialist = CommoditiesSpecialist()
    
    assert specialist.specialist_type == 'commodities'
    assert specialist.instrument_count == 2
    assert specialist.instruments == ['XAUUSD', 'WTIUSD']
    assert len(specialist.instrument_heads) == 2

def test_commodities_specialist_forward():
    '''Test CommoditiesSpecialist forward pass.'''
    specialist = CommoditiesSpecialist()
    batch_size = 2
    
    market_state = torch.randn(batch_size, 6)
    instrument_states = {
        'XAUUSD': torch.randn(batch_size, 50),
        'WTIUSD': torch.randn(batch_size, 50)
    }
    allocation = 0.5
    
    actions, value = specialist.forward(market_state, instrument_states, allocation)
    
    assert len(actions) == 2
    assert 'XAUUSD' in actions
    assert 'WTIUSD' in actions
    
    for instrument, action_probs in actions.items():
        assert action_probs.shape == (batch_size, 3)
        assert torch.allclose(action_probs.sum(dim=-1), torch.ones(batch_size))
    
    assert value.shape == (batch_size, 1)

def test_domain_features_extraction():
    '''Test commodity domain features extraction.'''
    specialist = CommoditiesSpecialist()
    
    market_data = {
        'cpi_expectation': 3.0,
        'ppi_expectation': 2.5,
        'vix': 25.0,
        'news_sentiment': 0.3,
        'oil_inventories': -1.0,
        'gold_production': 0.5,
        'commodity_index': 105.0,
        'nominal_10y': 4.5,
        'dxy': 102.0
    }
    
    features = specialist.get_domain_features(market_data)
    assert features.shape == (6,)
    assert features[0] == 2.75  # Average inflation expectation

def test_inflation_regime_detection():
    '''Test inflation regime detection.'''
    specialist = CommoditiesSpecialist()
    
    # Stable inflation
    stable_inflation = {'cpi': 2.0, 'ppi': 2.0, 'core_cpi': 2.0}
    assert specialist.detect_inflation_regime(stable_inflation) == 'stable'
    
    # Rising inflation
    rising_inflation = {'cpi': 4.0, 'ppi': 3.5, 'core_cpi': 3.0}
    assert specialist.detect_inflation_regime(rising_inflation) == 'rising'

def test_safe_haven_demand():
    '''Test safe-haven demand calculation.'''
    specialist = CommoditiesSpecialist()
    
    # High risk scenario
    high_risk_data = {
        'vix': 40.0,
        'credit_spread': 3.0,
        'geopolitical_tension': 0.8
    }
    
    demand = specialist.get_safe_haven_demand(high_risk_data)
    assert 0.0 <= demand <= 1.0
    assert demand > 0.5  # Should be high in high-risk scenario

def test_opec_schedule():
    '''Test OPEC schedule checking.'''
    specialist = CommoditiesSpecialist()
    
    # Test with OPEC meeting month
    opec_date = datetime(2024, 6, 15)  # June (OPEC meeting month)
    assert specialist.check_opec_schedule(opec_date) == True
    
    # Test with non-OPEC month
    non_opec_date = datetime(2024, 8, 15)  # August (not OPEC month)
    assert specialist.check_opec_schedule(non_opec_date) == False
"""
