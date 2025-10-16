"""
Equity Specialist for Hierarchical Multi-Agent Trading System

This module implements the Equity domain specialist that manages:
- SPX500, NAS100, US30 equity indices
- Equity market sentiment understanding
- Sector rotation, macro trends
- Earnings season, Fed policy impacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

from .base_specialist import BaseSpecialist


class EquitySpecialist(BaseSpecialist):
    """
    Equity Specialist for managing SPX500, NAS100, US30.
    
    This specialist understands:
    - Global equity market conditions
    - Market breadth and sentiment
    - Sector rotation patterns
    - Earnings season dynamics
    - Fed policy impacts
    - P/E ratios and valuations
    
    Architecture:
    - Equity encoder: processes global equity conditions
    - Instrument heads: individual action heads for each index
    - Value head: shared state value estimation
    """
    
    def __init__(
        self,
        instruments: List[str] = None,
        market_features_dim: int = 7,
        observation_dim: int = 50,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        device: str = "cpu"
    ):
        """
        Initialize EquitySpecialist.
        
        Args:
            instruments: List of equity instruments (default: SPX500, NAS100, US30)
            market_features_dim: Dimension of equity market features
            observation_dim: Dimension of individual instrument observations
            hidden_dim: Hidden layer dimension for instrument heads
            dropout: Dropout rate
            device: Device to run on
        """
        if instruments is None:
            instruments = ['SPX500', 'NAS100', 'US30']
        
        super().__init__(
            instruments=instruments,
            market_features_dim=market_features_dim,
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=device
        )
        
        # Individual instrument heads (one per equity index)
        self.instrument_heads = nn.ModuleDict()
        
        for instrument in self.instruments:
            # Each head takes: equity features (128) + instrument obs (observation_dim)
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
        Forward pass through Equity specialist.
        
        Args:
            market_state: (batch, market_features_dim) - Global equity conditions
            instrument_states: {
                'SPX500': (batch, obs_dim),
                'NAS100': (batch, obs_dim),
                'US30': (batch, obs_dim)
            }
            allocation: Scalar 0-1 from meta-controller
        
        Returns:
            actions: {
                'SPX500': (batch, 3) probabilities,
                'NAS100': (batch, 3) probabilities,
                'US30': (batch, 3) probabilities
            }
            value: (batch, 1)
        """
        # Validate inputs
        if not self.validate_inputs(market_state, instrument_states, allocation):
            raise ValueError("Invalid inputs to EquitySpecialist")
        
        # Encode global equity market state
        equity_features = self.domain_encoder(market_state)  # (batch, 128)
        
        # Generate actions for each instrument
        actions = {}
        for instrument in self.instruments:
            if instrument not in instrument_states:
                raise ValueError(f"Missing instrument state for {instrument}")
            
            # Concatenate equity features with instrument-specific observations
            instrument_obs = instrument_states[instrument]
            combined_input = torch.cat([equity_features, instrument_obs], dim=-1)
            
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
        value = self.value_head(equity_features)
        
        return actions, value
    
    def get_instruments(self) -> List[str]:
        """Get list of equity instruments managed by this specialist."""
        return self.instruments
    
    def get_domain_features(self, market_data: Dict[str, Any]) -> torch.Tensor:
        """
        Extract equity-specific features from market data.
        
        Args:
            market_data: Raw market data dictionary
            
        Returns:
            features: (market_features_dim,) tensor with equity market features
        """
        # Calculate market breadth (advance/decline ratio)
        advancing_stocks = market_data.get('advancing_stocks', 1500)
        declining_stocks = market_data.get('declining_stocks', 1500)
        market_breadth = advancing_stocks / (advancing_stocks + declining_stocks)
        
        # Get P/E ratio for SPX500
        spx_pe_ratio = market_data.get('spx500_pe_ratio', 20.0)
        
        # Detect earnings season phase
        earnings_season_phase = market_data.get('earnings_season_phase', 0.5)  # 0=off-season, 1=peak season
        
        # Fed policy stance (dovish/hawkish)
        fed_rate = market_data.get('fed_rate', 5.25)
        fed_rate_change = market_data.get('fed_rate_change', 0.0)  # Recent change
        fed_policy_stance = 0.5 + (fed_rate_change * 10.0)  # Normalize around 0.5
        
        # Equity volatility (VIX level)
        vix_level = market_data.get('vix', 20.0)
        equity_volatility = vix_level / 50.0  # Normalize (cap at 50 VIX)
        
        # Sector rotation indicator
        growth_vs_value = market_data.get('growth_vs_value', 0.5)  # 0=value, 1=growth
        
        # Market sentiment (fear/greed index)
        fear_greed_index = market_data.get('fear_greed_index', 50.0)  # 0-100
        market_sentiment = fear_greed_index / 100.0
        
        # Features vector
        features = torch.tensor([
            market_breadth,
            spx_pe_ratio / 30.0,  # Normalize (cap at 30 P/E)
            earnings_season_phase,
            fed_policy_stance,
            equity_volatility,
            growth_vs_value,
            market_sentiment
        ], dtype=torch.float32)
        
        return features
    
    def calculate_confidence(self, actions: Dict[str, torch.Tensor]) -> float:
        """
        Calculate confidence score for proposed equity actions.
        
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
    
    def detect_sector_rotation(self, sector_data: Dict[str, float]) -> str:
        """
        Detect sector rotation pattern.
        
        Args:
            sector_data: Dictionary with sector performance data
            
        Returns:
            rotation: 'growth', 'value', 'defensive', 'cyclical'
        """
        # Get sector performance (relative to market)
        tech_performance = sector_data.get('technology', 0.0)
        financial_performance = sector_data.get('financials', 0.0)
        defensive_performance = sector_data.get('utilities', 0.0)
        cyclical_performance = sector_data.get('industrials', 0.0)
        
        # Determine rotation pattern
        if tech_performance > 0.02:  # Tech outperforming
            return 'growth'
        elif financial_performance > 0.02:  # Financials outperforming
            return 'value'
        elif defensive_performance > 0.02:  # Defensive outperforming
            return 'defensive'
        elif cyclical_performance > 0.02:  # Cyclical outperforming
            return 'cyclical'
        else:
            return 'neutral'
    
    def get_fear_greed_index(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate fear/greed index based on market indicators.
        
        Args:
            market_data: Market data with sentiment indicators
            
        Returns:
            index: 0-100 (0=extreme fear, 100=extreme greed)
        """
        # Multiple sentiment indicators
        vix = market_data.get('vix', 20.0)
        put_call_ratio = market_data.get('put_call_ratio', 1.0)
        margin_debt = market_data.get('margin_debt', 0.5)  # Normalized
        junk_bond_demand = market_data.get('junk_bond_demand', 0.5)  # Normalized
        
        # Convert to fear/greed components (0-100 each)
        vix_component = max(0, 100 - (vix - 10) * 2)  # Lower VIX = higher greed
        put_call_component = max(0, 100 - (put_call_ratio - 0.5) * 100)  # Lower PCR = higher greed
        margin_component = margin_debt * 100  # Higher margin = higher greed
        junk_bond_component = junk_bond_demand * 100  # Higher demand = higher greed
        
        # Average components
        fear_greed = (vix_component + put_call_component + margin_component + junk_bond_component) / 4.0
        
        return float(np.clip(fear_greed, 0.0, 100.0))
    
    def check_earnings_calendar(self, current_date: datetime = None) -> Dict[str, bool]:
        """
        Check for major earnings announcements this week.
        
        Args:
            current_date: Current date (default: now)
            
        Returns:
            earnings: Dict mapping major companies to earnings this week
        """
        if current_date is None:
            current_date = datetime.now()
        
        # Simple heuristic: check if we're in earnings season
        # Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec
        quarter = (current_date.month - 1) // 3 + 1
        earnings_weeks = [2, 3, 4]  # Weeks 2-4 of each quarter
        
        # Check if current week is in earnings season
        week_of_quarter = (current_date.day - 1) // 7 + 1
        is_earnings_week = week_of_quarter in earnings_weeks
        
        return {
            'aapl_earnings': is_earnings_week and quarter in [1, 2, 3, 4],
            'msft_earnings': is_earnings_week and quarter in [1, 2, 3, 4],
            'googl_earnings': is_earnings_week and quarter in [1, 2, 3, 4],
            'amzn_earnings': is_earnings_week and quarter in [1, 2, 3, 4],
            'tsla_earnings': is_earnings_week and quarter in [1, 2, 3, 4]
        }
    
    def analyze_equity_market_structure(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze equity market structure and provide insights.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            analysis: Dictionary with equity market analysis
        """
        # Get domain features
        features = self.get_domain_features(market_data)
        
        # Analyze market structure
        analysis = {
            'market_breadth': features[0].item(),
            'spx_pe_ratio': features[1].item() * 30.0,  # Denormalize
            'earnings_season_phase': features[2].item(),
            'fed_policy_stance': features[3].item(),
            'equity_volatility': features[4].item() * 50.0,  # Denormalize
            'growth_vs_value': features[5].item(),
            'market_sentiment': features[6].item(),
            'sector_rotation': self.detect_sector_rotation(
                market_data.get('sector_data', {})
            ),
            'fear_greed_index': self.get_fear_greed_index(market_data),
            'earnings_calendar': self.check_earnings_calendar()
        }
        
        return analysis
    
    def get_equity_specific_metrics(self) -> Dict[str, float]:
        """
        Get equity-specific performance metrics.
        
        Returns:
            metrics: Dictionary of equity-specific metrics
        """
        base_metrics = self.get_performance_metrics()
        
        # Add equity-specific metrics
        equity_metrics = {
            **base_metrics,
            'sector_rotation_pnl': 0.0,
            'earnings_season_pnl': 0.0,
            'fed_policy_pnl': 0.0,
            'market_breadth_correlation': 0.0,
            'valuation_timing': 0.0
        }
        
        return equity_metrics
    
    def get_index_specific_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Get index-specific analysis for each equity instrument.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            analysis: Dict mapping instrument symbols to their specific analysis
        """
        analysis = {}
        
        # SPX500 (Broad market, all sectors)
        analysis['SPX500'] = {
            'market_cap_weighted': True,
            'sector_diversification': 'high',
            'valuation_sensitivity': market_data.get('spx_pe_ratio', 20.0) / 25.0,
            'earnings_sensitivity': market_data.get('earnings_season_phase', 0.5),
            'fed_policy_sensitivity': 0.8,  # High sensitivity to Fed policy
            'economic_cycle_sensitivity': 0.9  # High sensitivity to economic cycles
        }
        
        # NAS100 (Tech-heavy, growth-focused)
        analysis['NAS100'] = {
            'market_cap_weighted': True,
            'sector_diversification': 'low',  # Tech-heavy
            'valuation_sensitivity': market_data.get('nasdaq_pe_ratio', 25.0) / 30.0,
            'earnings_sensitivity': 1.0,  # Very high sensitivity to earnings
            'fed_policy_sensitivity': 0.9,  # Very high sensitivity to rates
            'economic_cycle_sensitivity': 0.7,  # Moderate sensitivity
            'growth_stock_sensitivity': 1.0  # High sensitivity to growth sentiment
        }
        
        # US30 (Value stocks, blue-chip)
        analysis['US30'] = {
            'market_cap_weighted': False,  # Price-weighted
            'sector_diversification': 'medium',
            'valuation_sensitivity': market_data.get('dow_pe_ratio', 18.0) / 25.0,
            'earnings_sensitivity': 0.6,  # Lower sensitivity to earnings
            'fed_policy_sensitivity': 0.6,  # Lower sensitivity to rates
            'economic_cycle_sensitivity': 0.8,  # High sensitivity to economic cycles
            'dividend_sensitivity': 0.8  # High sensitivity to dividend yields
        }
        
        return analysis
    
    def calculate_market_regime_score(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate overall market regime score.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            score: 0-1 (0=bear market, 1=bull market)
        """
        # Multiple regime indicators
        vix = market_data.get('vix', 20.0)
        market_breadth = market_data.get('market_breadth', 0.5)
        fear_greed = self.get_fear_greed_index(market_data) / 100.0
        sector_rotation = market_data.get('sector_rotation_strength', 0.5)
        
        # Convert to regime score
        vix_score = max(0, 1.0 - (vix - 15) / 20.0)  # Lower VIX = higher score
        breadth_score = market_breadth
        sentiment_score = fear_greed
        rotation_score = sector_rotation
        
        # Weighted average
        regime_score = (vix_score * 0.3 + breadth_score * 0.3 + 
                       sentiment_score * 0.2 + rotation_score * 0.2)
        
        return float(np.clip(regime_score, 0.0, 1.0))


# Unit test stubs (to be implemented in test files)
"""
def test_equity_specialist_initialization():
    '''Test EquitySpecialist initialization.'''
    specialist = EquitySpecialist()
    
    assert specialist.specialist_type == 'equity'
    assert specialist.instrument_count == 3
    assert specialist.instruments == ['SPX500', 'NAS100', 'US30']
    assert len(specialist.instrument_heads) == 3

def test_equity_specialist_forward():
    '''Test EquitySpecialist forward pass.'''
    specialist = EquitySpecialist()
    batch_size = 2
    
    market_state = torch.randn(batch_size, 7)
    instrument_states = {
        'SPX500': torch.randn(batch_size, 50),
        'NAS100': torch.randn(batch_size, 50),
        'US30': torch.randn(batch_size, 50)
    }
    allocation = 0.5
    
    actions, value = specialist.forward(market_state, instrument_states, allocation)
    
    assert len(actions) == 3
    assert 'SPX500' in actions
    assert 'NAS100' in actions
    assert 'US30' in actions
    
    for instrument, action_probs in actions.items():
        assert action_probs.shape == (batch_size, 3)
        assert torch.allclose(action_probs.sum(dim=-1), torch.ones(batch_size))
    
    assert value.shape == (batch_size, 1)

def test_domain_features_extraction():
    '''Test equity domain features extraction.'''
    specialist = EquitySpecialist()
    
    market_data = {
        'advancing_stocks': 2000,
        'declining_stocks': 1000,
        'spx_pe_ratio': 22.0,
        'earnings_season_phase': 0.8,
        'fed_rate': 5.25,
        'fed_rate_change': 0.25,
        'vix': 18.0,
        'growth_vs_value': 0.7,
        'fear_greed_index': 65.0
    }
    
    features = specialist.get_domain_features(market_data)
    assert features.shape == (7,)
    assert features[0] == 2/3  # Market breadth

def test_sector_rotation_detection():
    '''Test sector rotation detection.'''
    specialist = EquitySpecialist()
    
    # Growth rotation
    growth_sectors = {'technology': 0.03, 'financials': 0.01, 'utilities': -0.01, 'industrials': 0.01}
    assert specialist.detect_sector_rotation(growth_sectors) == 'growth'
    
    # Value rotation
    value_sectors = {'technology': 0.01, 'financials': 0.03, 'utilities': 0.01, 'industrials': 0.01}
    assert specialist.detect_sector_rotation(value_sectors) == 'value'

def test_fear_greed_index():
    '''Test fear/greed index calculation.'''
    specialist = EquitySpecialist()
    
    # Greed scenario
    greed_data = {
        'vix': 15.0,
        'put_call_ratio': 0.6,
        'margin_debt': 0.8,
        'junk_bond_demand': 0.9
    }
    
    index = specialist.get_fear_greed_index(greed_data)
    assert 0.0 <= index <= 100.0
    assert index > 50.0  # Should be greed scenario

def test_earnings_calendar():
    '''Test earnings calendar checking.'''
    specialist = EquitySpecialist()
    
    # Test with earnings week
    earnings_date = datetime(2024, 2, 15)  # Week 2 of Q1
    calendar = specialist.check_earnings_calendar(earnings_date)
    
    assert isinstance(calendar, dict)
    assert 'aapl_earnings' in calendar
    assert 'msft_earnings' in calendar
    assert calendar['aapl_earnings'] == True  # Should be earnings week

def test_market_regime_score():
    '''Test market regime score calculation.'''
    specialist = EquitySpecialist()
    
    # Bull market scenario
    bull_data = {
        'vix': 15.0,
        'market_breadth': 0.7,
        'sector_rotation_strength': 0.8
    }
    
    score = specialist.calculate_market_regime_score(bull_data)
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Should be bull market
"""
