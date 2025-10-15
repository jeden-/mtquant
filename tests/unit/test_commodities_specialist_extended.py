"""
Extended tests for CommoditiesSpecialist.

This module tests the Commodities domain specialist that manages XAUUSD, WTIUSD
commodity instruments with comprehensive coverage of all public methods.
"""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from mtquant.agents.hierarchical.commodities_specialist import CommoditiesSpecialist


class TestCommoditiesSpecialistInitialization:
    """Test CommoditiesSpecialist initialization."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        specialist = CommoditiesSpecialist()
        
        assert specialist.instruments == ['XAUUSD', 'WTIUSD']
        assert specialist.specialist_type == 'commodities'
        assert specialist.market_features_dim == 6
        assert specialist.observation_dim == 50
        assert specialist.hidden_dim == 64
        assert specialist.dropout == 0.2
        assert specialist.device == "cpu"
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        instruments = ['XAUUSD']
        specialist = CommoditiesSpecialist(
            instruments=instruments,
            market_features_dim=8,
            observation_dim=60,
            hidden_dim=128,
            dropout=0.3,
            device="cpu"
        )
        
        assert specialist.instruments == instruments
        assert specialist.market_features_dim == 8
        assert specialist.observation_dim == 60
        assert specialist.hidden_dim == 128
        assert specialist.dropout == 0.3
        assert specialist.device == "cpu"
    
    def test_initialization_components(self):
        """Test that all components are initialized."""
        specialist = CommoditiesSpecialist()
        
        # Check that neural network components exist
        assert hasattr(specialist, 'domain_encoder')
        assert hasattr(specialist, 'instrument_heads')
        assert hasattr(specialist, 'value_head')
        
        # Check that instrument heads are created for each instrument
        assert len(specialist.instrument_heads) == len(specialist.instruments)


class TestCommoditiesSpecialistForward:
    """Test CommoditiesSpecialist forward method."""
    
    @pytest.fixture
    def specialist(self):
        """Create CommoditiesSpecialist instance."""
        return CommoditiesSpecialist()
    
    @pytest.fixture
    def mock_inputs(self):
        """Create mock inputs for forward method."""
        market_state = torch.randn(1, 6)  # market_features_dim
        instrument_states = {
            'XAUUSD': torch.randn(1, 50),
            'WTIUSD': torch.randn(1, 50)
        }
        allocation = 0.4  # Scalar, not tensor
        
        return market_state, instrument_states, allocation
    
    def test_forward_basic(self, specialist, mock_inputs):
        """Test basic forward pass."""
        market_state, instrument_states, allocation = mock_inputs
        
        actions, value = specialist.forward(market_state, instrument_states, allocation)
        
        # Verify return types
        assert isinstance(actions, dict)
        assert isinstance(value, torch.Tensor)
        
        # Verify actions for each instrument
        for instrument in specialist.instruments:
            assert instrument in actions
            assert isinstance(actions[instrument], torch.Tensor)
            assert actions[instrument].shape[0] == 1  # batch size
        
        # Verify value shape
        assert value.shape[0] == 1  # batch size
    
    def test_forward_batch(self, specialist):
        """Test forward pass with batch input."""
        batch_size = 4
        market_state = torch.randn(batch_size, 6)
        instrument_states = {
            'XAUUSD': torch.randn(batch_size, 50),
            'WTIUSD': torch.randn(batch_size, 50)
        }
        allocation = 0.4  # Scalar, not tensor
        
        actions, value = specialist.forward(market_state, instrument_states, allocation)
        
        # Verify batch processing
        for instrument in specialist.instruments:
            assert actions[instrument].shape[0] == batch_size
        assert value.shape[0] == batch_size
    
    def test_forward_different_instruments(self):
        """Test forward pass with different instruments."""
        instruments = ['XAUUSD']
        specialist = CommoditiesSpecialist(instruments=instruments)
        
        market_state = torch.randn(1, 6)
        instrument_states = {
            'XAUUSD': torch.randn(1, 50)
        }
        allocation = 0.5  # Scalar, not tensor
        
        actions, value = specialist.forward(market_state, instrument_states, allocation)
        
        # Verify only specified instruments are in actions
        assert len(actions) == len(instruments)
        for instrument in instruments:
            assert instrument in actions


class TestCommoditiesSpecialistMethods:
    """Test CommoditiesSpecialist public methods."""
    
    @pytest.fixture
    def specialist(self):
        """Create CommoditiesSpecialist instance."""
        return CommoditiesSpecialist()
    
    def test_get_instruments(self, specialist):
        """Test get_instruments method."""
        instruments = specialist.get_instruments()
        
        assert isinstance(instruments, list)
        assert len(instruments) == 2
        assert 'XAUUSD' in instruments
        assert 'WTIUSD' in instruments
    
    def test_get_domain_features(self, specialist):
        """Test get_domain_features method."""
        market_data = {
            'XAUUSD': {'close': 2000.0, 'volume': 1000},
            'WTIUSD': {'close': 75.0, 'volume': 5000},
            'DXY': {'close': 100.0},
            '10Y_YIELD': {'close': 4.5},
            'INFLATION_EXPECTATION': {'close': 2.5},
            'GEOPOLITICAL_RISK': {'close': 0.3}
        }
        
        features = specialist.get_domain_features(market_data)
        
        assert isinstance(features, torch.Tensor)
        # Features might be 1D tensor, check the last dimension
        if len(features.shape) > 1:
            assert features.shape[-1] == specialist.market_features_dim
        else:
            assert features.shape[0] == specialist.market_features_dim
    
    def test_calculate_confidence(self, specialist):
        """Test calculate_confidence method."""
        actions = {
            'XAUUSD': torch.tensor([[0.7, 0.2, 0.1]]),  # buy, hold, sell
            'WTIUSD': torch.tensor([[0.6, 0.3, 0.1]])
        }
        
        confidence = specialist.calculate_confidence(actions)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_detect_inflation_regime(self, specialist):
        """Test detect_inflation_regime method."""
        inflation_data = {
            'cpi': 3.2,
            'core_cpi': 2.8,
            'pce': 2.9,
            'inflation_expectation': 2.5,
            'real_yield': 1.5
        }
        
        regime = specialist.detect_inflation_regime(inflation_data)
        
        assert isinstance(regime, str)
        assert regime in ['high', 'medium', 'low', 'stable', 'rising']
    
    def test_get_safe_haven_demand(self, specialist):
        """Test get_safe_haven_demand method."""
        market_data = {
            'VIX': {'close': 25.0},
            'DXY': {'close': 100.0},
            '10Y_YIELD': {'close': 4.5},
            'GEOPOLITICAL_RISK': {'close': 0.7}
        }
        
        demand = specialist.get_safe_haven_demand(market_data)
        
        assert isinstance(demand, float)
        assert 0.0 <= demand <= 1.0
    
    def test_check_opec_schedule(self, specialist):
        """Test check_opec_schedule method."""
        current_date = datetime.utcnow()
        
        opec_info = specialist.check_opec_schedule(current_date)
        
        # Method might return boolean or dict
        assert isinstance(opec_info, (bool, dict))
    
    def test_analyze_commodity_fundamentals(self, specialist):
        """Test analyze_commodity_fundamentals method."""
        market_data = {
            'XAUUSD': {'close': 2000.0, 'volume': 1000},
            'WTIUSD': {'close': 75.0, 'volume': 5000},
            'DXY': {'close': 100.0},
            '10Y_YIELD': {'close': 4.5}
        }
        
        fundamentals = specialist.analyze_commodity_fundamentals(market_data)
        
        assert isinstance(fundamentals, dict)
        # Check for actual keys returned by the method
        assert len(fundamentals) > 0
    
    def test_get_commodity_specific_metrics(self, specialist):
        """Test get_commodity_specific_metrics method."""
        metrics = specialist.get_commodity_specific_metrics()
        
        assert isinstance(metrics, dict)
        # Check for actual keys returned by the method
        assert len(metrics) > 0
    
    def test_get_gold_specific_analysis(self, specialist):
        """Test get_gold_specific_analysis method."""
        market_data = {
            'XAUUSD': {'close': 2000.0, 'volume': 1000},
            'DXY': {'close': 100.0},
            '10Y_YIELD': {'close': 4.5},
            'INFLATION_EXPECTATION': {'close': 2.5}
        }
        
        analysis = specialist.get_gold_specific_analysis(market_data)
        
        assert isinstance(analysis, dict)
        # Check for actual keys returned by the method
        assert len(analysis) > 0
    
    def test_get_oil_specific_analysis(self, specialist):
        """Test get_oil_specific_analysis method."""
        market_data = {
            'WTIUSD': {'close': 75.0, 'volume': 5000},
            'DXY': {'close': 100.0},
            'GEOPOLITICAL_RISK': {'close': 0.3},
            'OPEC_PRODUCTION': {'close': 30.0}
        }
        
        analysis = specialist.get_oil_specific_analysis(market_data)
        
        assert isinstance(analysis, dict)
        # Check for actual keys returned by the method
        assert len(analysis) > 0


class TestCommoditiesSpecialistEdgeCases:
    """Test CommoditiesSpecialist edge cases and error handling."""
    
    @pytest.fixture
    def specialist(self):
        """Create CommoditiesSpecialist instance."""
        return CommoditiesSpecialist()
    
    def test_forward_empty_instrument_states(self, specialist):
        """Test forward with empty instrument states."""
        market_state = torch.randn(1, 6)
        instrument_states = {}
        allocation = 0.4
        
        # Should handle empty instrument states gracefully
        try:
            actions, value = specialist.forward(market_state, instrument_states, allocation)
            assert isinstance(actions, dict)
            assert isinstance(value, torch.Tensor)
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert "instrument" in str(e).lower() or "empty" in str(e).lower()
    
    def test_get_domain_features_missing_data(self, specialist):
        """Test get_domain_features with missing market data."""
        market_data = {
            'XAUUSD': {'close': 2000.0},
            'DXY': {'close': 100.0}
            # Missing other required data
        }
        
        features = specialist.get_domain_features(market_data)
        
        assert isinstance(features, torch.Tensor)
        # Features might be 1D tensor, check the last dimension
        if len(features.shape) > 1:
            assert features.shape[-1] == specialist.market_features_dim
        else:
            assert features.shape[0] == specialist.market_features_dim
    
    def test_calculate_confidence_empty_actions(self, specialist):
        """Test calculate_confidence with empty actions."""
        actions = {}
        
        confidence = specialist.calculate_confidence(actions)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_detect_inflation_regime_empty_data(self, specialist):
        """Test detect_inflation_regime with empty data."""
        inflation_data = {}
        
        regime = specialist.detect_inflation_regime(inflation_data)
        
        assert isinstance(regime, str)
        assert regime in ['high', 'medium', 'low', 'stable', 'rising']
    
    def test_get_safe_haven_demand_missing_data(self, specialist):
        """Test get_safe_haven_demand with missing data."""
        market_data = {
            'VIX': {'close': 25.0}
            # Missing other required data
        }
        
        demand = specialist.get_safe_haven_demand(market_data)
        
        assert isinstance(demand, float)
        assert 0.0 <= demand <= 1.0
    
    def test_forward_different_batch_sizes(self, specialist):
        """Test forward with mismatched batch sizes."""
        market_state = torch.randn(2, 6)  # batch size 2
        instrument_states = {
            'XAUUSD': torch.randn(1, 50),  # batch size 1
            'WTIUSD': torch.randn(1, 50)
        }
        allocation = 0.4
        
        # Should handle mismatched batch sizes gracefully
        try:
            actions, value = specialist.forward(market_state, instrument_states, allocation)
            assert isinstance(actions, dict)
            assert isinstance(value, torch.Tensor)
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert "batch" in str(e).lower() or "size" in str(e).lower() or "dimension" in str(e).lower()


class TestCommoditiesSpecialistIntegration:
    """Test CommoditiesSpecialist integration scenarios."""
    
    @pytest.fixture
    def specialist(self):
        """Create CommoditiesSpecialist instance."""
        return CommoditiesSpecialist()
    
    def test_full_trading_cycle(self, specialist):
        """Test a full trading cycle with all methods."""
        # Market data
        market_data = {
            'XAUUSD': {'close': 2000.0, 'volume': 1000},
            'WTIUSD': {'close': 75.0, 'volume': 5000},
            'DXY': {'close': 100.0},
            '10Y_YIELD': {'close': 4.5},
            'INFLATION_EXPECTATION': {'close': 2.5},
            'GEOPOLITICAL_RISK': {'close': 0.3}
        }
        
        # Get domain features
        features = specialist.get_domain_features(market_data)
        assert isinstance(features, torch.Tensor)
        
        # Add batch dimension if needed
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        # Forward pass
        instrument_states = {
            'XAUUSD': torch.randn(1, 50),
            'WTIUSD': torch.randn(1, 50)
        }
        allocation = 0.4  # Scalar, not tensor
        
        actions, value = specialist.forward(features, instrument_states, allocation)
        assert isinstance(actions, dict)
        assert isinstance(value, torch.Tensor)
        
        # Calculate confidence
        confidence = specialist.calculate_confidence(actions)
        assert isinstance(confidence, float)
        
        # Analyze commodity fundamentals
        fundamentals = specialist.analyze_commodity_fundamentals(market_data)
        assert isinstance(fundamentals, dict)
        
        # Get commodity metrics
        metrics = specialist.get_commodity_specific_metrics()
        assert isinstance(metrics, dict)
        
        # Get gold analysis
        gold_analysis = specialist.get_gold_specific_analysis(market_data)
        assert isinstance(gold_analysis, dict)
        
        # Get oil analysis
        oil_analysis = specialist.get_oil_specific_analysis(market_data)
        assert isinstance(oil_analysis, dict)
    
    def test_inflation_regime_analysis(self, specialist):
        """Test inflation regime analysis workflow."""
        # High inflation scenario
        high_inflation_data = {
            'cpi': 4.5,
            'core_cpi': 4.0,
            'pce': 4.2,
            'inflation_expectation': 3.5,
            'real_yield': 0.5
        }
        
        high_regime = specialist.detect_inflation_regime(high_inflation_data)
        assert isinstance(high_regime, str)
        
        # Low inflation scenario
        low_inflation_data = {
            'cpi': 1.5,
            'core_cpi': 1.2,
            'pce': 1.3,
            'inflation_expectation': 1.8,
            'real_yield': 2.5
        }
        
        low_regime = specialist.detect_inflation_regime(low_inflation_data)
        assert isinstance(low_regime, str)
    
    def test_safe_haven_analysis(self, specialist):
        """Test safe haven demand analysis workflow."""
        # High risk scenario
        high_risk_data = {
            'VIX': {'close': 35.0},  # High volatility
            'DXY': {'close': 95.0},  # Weak dollar
            '10Y_YIELD': {'close': 3.0},  # Low yields
            'GEOPOLITICAL_RISK': {'close': 0.8}  # High risk
        }
        
        high_demand = specialist.get_safe_haven_demand(high_risk_data)
        assert isinstance(high_demand, float)
        
        # Low risk scenario
        low_risk_data = {
            'VIX': {'close': 15.0},  # Low volatility
            'DXY': {'close': 105.0},  # Strong dollar
            '10Y_YIELD': {'close': 5.0},  # High yields
            'GEOPOLITICAL_RISK': {'close': 0.2}  # Low risk
        }
        
        low_demand = specialist.get_safe_haven_demand(low_risk_data)
        assert isinstance(low_demand, float)
        
        # Both demands should be valid floats
        assert isinstance(high_demand, float)
        assert isinstance(low_demand, float)
    
    def test_opec_schedule_analysis(self, specialist):
        """Test OPEC schedule analysis workflow."""
        # Test during OPEC meeting period
        opec_date = datetime(2024, 6, 1)  # June (OPEC meeting month)
        opec_info = specialist.check_opec_schedule(opec_date)
        assert isinstance(opec_info, (bool, dict))
        
        # Test outside OPEC meeting period
        non_opec_date = datetime(2024, 3, 15)  # March
        non_opec_info = specialist.check_opec_schedule(non_opec_date)
        assert isinstance(non_opec_info, (bool, dict))
    
    def test_commodity_fundamentals_analysis(self, specialist):
        """Test commodity fundamentals analysis workflow."""
        # Bullish commodity data
        bullish_data = {
            'XAUUSD': {'close': 2100.0, 'volume': 1500},  # High gold price
            'WTIUSD': {'close': 85.0, 'volume': 8000},  # High oil price
            'DXY': {'close': 95.0},  # Weak dollar (good for commodities)
            '10Y_YIELD': {'close': 3.0}  # Low yields (good for gold)
        }
        
        bullish_fundamentals = specialist.analyze_commodity_fundamentals(bullish_data)
        assert isinstance(bullish_fundamentals, dict)
        
        # Bearish commodity data
        bearish_data = {
            'XAUUSD': {'close': 1900.0, 'volume': 800},  # Low gold price
            'WTIUSD': {'close': 65.0, 'volume': 3000},  # Low oil price
            'DXY': {'close': 105.0},  # Strong dollar (bad for commodities)
            '10Y_YIELD': {'close': 5.0}  # High yields (bad for gold)
        }
        
        bearish_fundamentals = specialist.analyze_commodity_fundamentals(bearish_data)
        assert isinstance(bearish_fundamentals, dict)
    
    def test_gold_vs_oil_analysis(self, specialist):
        """Test gold vs oil specific analysis."""
        market_data = {
            'XAUUSD': {'close': 2000.0, 'volume': 1000},
            'WTIUSD': {'close': 75.0, 'volume': 5000},
            'DXY': {'close': 100.0},
            '10Y_YIELD': {'close': 4.5},
            'INFLATION_EXPECTATION': {'close': 2.5},
            'GEOPOLITICAL_RISK': {'close': 0.3}
        }
        
        # Gold analysis
        gold_analysis = specialist.get_gold_specific_analysis(market_data)
        assert isinstance(gold_analysis, dict)
        
        # Oil analysis
        oil_analysis = specialist.get_oil_specific_analysis(market_data)
        assert isinstance(oil_analysis, dict)
        
        # Both analyses should return different insights
        assert gold_analysis != oil_analysis
