"""
Extended tests for EquitySpecialist.

This module tests the Equity domain specialist that manages SPX500, NAS100, US30
equity indices with comprehensive coverage of all public methods.
"""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from mtquant.agents.hierarchical.equity_specialist import EquitySpecialist


class TestEquitySpecialistInitialization:
    """Test EquitySpecialist initialization."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        specialist = EquitySpecialist()
        
        assert specialist.instruments == ['SPX500', 'NAS100', 'US30']
        assert specialist.specialist_type == 'equity'
        assert specialist.market_features_dim == 7
        assert specialist.observation_dim == 50
        assert specialist.hidden_dim == 64
        assert specialist.dropout == 0.2
        assert specialist.device == "cpu"
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        instruments = ['SPX500', 'NAS100']
        specialist = EquitySpecialist(
            instruments=instruments,
            market_features_dim=10,
            observation_dim=60,
            hidden_dim=128,
            dropout=0.3,
            device="cpu"
        )
        
        assert specialist.instruments == instruments
        assert specialist.market_features_dim == 10
        assert specialist.observation_dim == 60
        assert specialist.hidden_dim == 128
        assert specialist.dropout == 0.3
        assert specialist.device == "cpu"
    
    def test_initialization_components(self):
        """Test that all components are initialized."""
        specialist = EquitySpecialist()
        
        # Check that neural network components exist
        assert hasattr(specialist, 'domain_encoder')
        assert hasattr(specialist, 'instrument_heads')
        assert hasattr(specialist, 'value_head')
        
        # Check that instrument heads are created for each instrument
        assert len(specialist.instrument_heads) == len(specialist.instruments)


class TestEquitySpecialistForward:
    """Test EquitySpecialist forward method."""
    
    @pytest.fixture
    def specialist(self):
        """Create EquitySpecialist instance."""
        return EquitySpecialist()
    
    @pytest.fixture
    def mock_inputs(self):
        """Create mock inputs for forward method."""
        market_state = torch.randn(1, 7)  # market_features_dim
        instrument_states = {
            'SPX500': torch.randn(1, 50),
            'NAS100': torch.randn(1, 50),
            'US30': torch.randn(1, 50)
        }
        allocation = torch.tensor([0.4])
        
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
        market_state = torch.randn(batch_size, 7)
        instrument_states = {
            'SPX500': torch.randn(batch_size, 50),
            'NAS100': torch.randn(batch_size, 50),
            'US30': torch.randn(batch_size, 50)
        }
        allocation = 0.4  # Scalar, not tensor
        
        actions, value = specialist.forward(market_state, instrument_states, allocation)
        
        # Verify batch processing
        for instrument in specialist.instruments:
            assert actions[instrument].shape[0] == batch_size
        assert value.shape[0] == batch_size
    
    def test_forward_different_instruments(self):
        """Test forward pass with different instruments."""
        instruments = ['SPX500', 'NAS100']
        specialist = EquitySpecialist(instruments=instruments)
        
        market_state = torch.randn(1, 7)
        instrument_states = {
            'SPX500': torch.randn(1, 50),
            'NAS100': torch.randn(1, 50)
        }
        allocation = 0.5  # Scalar, not tensor
        
        actions, value = specialist.forward(market_state, instrument_states, allocation)
        
        # Verify only specified instruments are in actions
        assert len(actions) == len(instruments)
        for instrument in instruments:
            assert instrument in actions


class TestEquitySpecialistMethods:
    """Test EquitySpecialist public methods."""
    
    @pytest.fixture
    def specialist(self):
        """Create EquitySpecialist instance."""
        return EquitySpecialist()
    
    def test_get_instruments(self, specialist):
        """Test get_instruments method."""
        instruments = specialist.get_instruments()
        
        assert isinstance(instruments, list)
        assert len(instruments) == 3
        assert 'SPX500' in instruments
        assert 'NAS100' in instruments
        assert 'US30' in instruments
    
    def test_get_domain_features(self, specialist):
        """Test get_domain_features method."""
        market_data = {
            'SPX500': {'close': 4000.0, 'volume': 1000000},
            'NAS100': {'close': 12000.0, 'volume': 800000},
            'US30': {'close': 35000.0, 'volume': 600000},
            'VIX': {'close': 20.0},
            'DXY': {'close': 100.0},
            '10Y_YIELD': {'close': 4.5},
            'FED_RATE': {'close': 5.25}
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
            'SPX500': torch.tensor([[0.7, 0.2, 0.1]]),  # buy, hold, sell
            'NAS100': torch.tensor([[0.6, 0.3, 0.1]]),
            'US30': torch.tensor([[0.5, 0.4, 0.1]])
        }
        
        confidence = specialist.calculate_confidence(actions)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_detect_sector_rotation(self, specialist):
        """Test detect_sector_rotation method."""
        sector_performance = {
            'technology': 0.05,
            'healthcare': 0.02,
            'financials': -0.01,
            'energy': -0.03,
            'utilities': 0.01
        }
        
        rotation = specialist.detect_sector_rotation(sector_performance)
        
        assert isinstance(rotation, str)
        assert rotation in ['tech_heavy', 'defensive', 'cyclical', 'balanced', 'growth', 'neutral']
    
    def test_get_fear_greed_index(self, specialist):
        """Test get_fear_greed_index method."""
        market_data = {
            'VIX': {'close': 20.0},
            'SPX500': {'close': 4000.0},
            'PUT_CALL_RATIO': {'close': 0.8},
            'ADVANCE_DECLINE': {'close': 1.2}
        }
        
        index = specialist.get_fear_greed_index(market_data)
        
        assert isinstance(index, float)
        assert 0.0 <= index <= 100.0
    
    def test_check_earnings_calendar(self, specialist):
        """Test check_earnings_calendar method."""
        current_date = datetime.utcnow()
        
        earnings_info = specialist.check_earnings_calendar(current_date)
        
        assert isinstance(earnings_info, dict)
        # Check for actual keys returned by the method
        assert len(earnings_info) > 0
    
    def test_analyze_equity_market_structure(self, specialist):
        """Test analyze_equity_market_structure method."""
        market_data = {
            'SPX500': {'close': 4000.0, 'volume': 1000000},
            'NAS100': {'close': 12000.0, 'volume': 800000},
            'US30': {'close': 35000.0, 'volume': 600000}
        }
        
        structure = specialist.analyze_equity_market_structure(market_data)
        
        assert isinstance(structure, dict)
        # Check for actual keys returned by the method
        assert len(structure) > 0
    
    def test_get_equity_specific_metrics(self, specialist):
        """Test get_equity_specific_metrics method."""
        market_data = {
            'SPX500': {'close': 4000.0, 'volume': 1000000},
            'NAS100': {'close': 12000.0, 'volume': 800000},
            'US30': {'close': 35000.0, 'volume': 600000},
            'VIX': {'close': 20.0},
            'DXY': {'close': 100.0}
        }
        
        metrics = specialist.get_equity_specific_metrics()
        
        assert isinstance(metrics, dict)
        # Check for actual keys returned by the method
        assert len(metrics) > 0
    
    def test_get_index_specific_analysis(self, specialist):
        """Test get_index_specific_analysis method."""
        market_data = {
            'SPX500': {'close': 4000.0, 'volume': 1000000},
            'NAS100': {'close': 12000.0, 'volume': 800000},
            'US30': {'close': 35000.0, 'volume': 600000}
        }
        
        analysis = specialist.get_index_specific_analysis(market_data)
        
        assert isinstance(analysis, dict)
        for instrument in specialist.instruments:
            assert instrument in analysis
            assert isinstance(analysis[instrument], dict)
    
    def test_calculate_market_regime_score(self, specialist):
        """Test calculate_market_regime_score method."""
        market_data = {
            'SPX500': {'close': 4000.0, 'volume': 1000000},
            'NAS100': {'close': 12000.0, 'volume': 800000},
            'US30': {'close': 35000.0, 'volume': 600000},
            'VIX': {'close': 20.0},
            'DXY': {'close': 100.0}
        }
        
        score = specialist.calculate_market_regime_score(market_data)
        
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


class TestEquitySpecialistEdgeCases:
    """Test EquitySpecialist edge cases and error handling."""
    
    @pytest.fixture
    def specialist(self):
        """Create EquitySpecialist instance."""
        return EquitySpecialist()
    
    def test_forward_empty_instrument_states(self, specialist):
        """Test forward with empty instrument states."""
        market_state = torch.randn(1, 7)
        instrument_states = {}
        allocation = torch.tensor([0.4])
        
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
            'SPX500': {'close': 4000.0},
            'VIX': {'close': 20.0}
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
    
    def test_detect_sector_rotation_empty_performance(self, specialist):
        """Test detect_sector_rotation with empty performance data."""
        sector_performance = {}
        
        rotation = specialist.detect_sector_rotation(sector_performance)
        
        assert isinstance(rotation, str)
        assert rotation in ['tech_heavy', 'defensive', 'cyclical', 'balanced', 'growth', 'neutral']
    
    def test_get_fear_greed_index_missing_data(self, specialist):
        """Test get_fear_greed_index with missing data."""
        market_data = {
            'VIX': {'close': 20.0}
            # Missing other required data
        }
        
        index = specialist.get_fear_greed_index(market_data)
        
        assert isinstance(index, float)
        assert 0.0 <= index <= 100.0
    
    def test_forward_different_batch_sizes(self, specialist):
        """Test forward with mismatched batch sizes."""
        market_state = torch.randn(2, 7)  # batch size 2
        instrument_states = {
            'SPX500': torch.randn(1, 50),  # batch size 1
            'NAS100': torch.randn(1, 50),
            'US30': torch.randn(1, 50)
        }
        allocation = torch.tensor([0.4])
        
        # Should handle mismatched batch sizes gracefully
        try:
            actions, value = specialist.forward(market_state, instrument_states, allocation)
            assert isinstance(actions, dict)
            assert isinstance(value, torch.Tensor)
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert "batch" in str(e).lower() or "size" in str(e).lower() or "dimension" in str(e).lower()


class TestEquitySpecialistIntegration:
    """Test EquitySpecialist integration scenarios."""
    
    @pytest.fixture
    def specialist(self):
        """Create EquitySpecialist instance."""
        return EquitySpecialist()
    
    def test_full_trading_cycle(self, specialist):
        """Test a full trading cycle with all methods."""
        # Market data
        market_data = {
            'SPX500': {'close': 4000.0, 'volume': 1000000},
            'NAS100': {'close': 12000.0, 'volume': 800000},
            'US30': {'close': 35000.0, 'volume': 600000},
            'VIX': {'close': 20.0},
            'DXY': {'close': 100.0},
            '10Y_YIELD': {'close': 4.5},
            'FED_RATE': {'close': 5.25}
        }
        
        # Get domain features
        features = specialist.get_domain_features(market_data)
        assert isinstance(features, torch.Tensor)
        
        # Add batch dimension if needed
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        # Forward pass
        instrument_states = {
            'SPX500': torch.randn(1, 50),
            'NAS100': torch.randn(1, 50),
            'US30': torch.randn(1, 50)
        }
        allocation = 0.4  # Scalar, not tensor
        
        actions, value = specialist.forward(features, instrument_states, allocation)
        assert isinstance(actions, dict)
        assert isinstance(value, torch.Tensor)
        
        # Calculate confidence
        confidence = specialist.calculate_confidence(actions)
        assert isinstance(confidence, float)
        
        # Analyze market structure
        structure = specialist.analyze_equity_market_structure(market_data)
        assert isinstance(structure, dict)
        
        # Get equity metrics
        metrics = specialist.get_equity_specific_metrics()
        assert isinstance(metrics, dict)
        
        # Calculate market regime
        regime_score = specialist.calculate_market_regime_score(market_data)
        assert isinstance(regime_score, float)
    
    def test_sector_rotation_analysis(self, specialist):
        """Test sector rotation analysis workflow."""
        sector_performance = {
            'technology': 0.05,
            'healthcare': 0.02,
            'financials': -0.01,
            'energy': -0.03,
            'utilities': 0.01
        }
        
        rotation = specialist.detect_sector_rotation(sector_performance)
        assert isinstance(rotation, str)
        
        # Test different sector performance scenarios
        tech_heavy_performance = {
            'technology': 0.08,
            'healthcare': 0.01,
            'financials': -0.02,
            'energy': -0.05,
            'utilities': -0.01
        }
        
        tech_rotation = specialist.detect_sector_rotation(tech_heavy_performance)
        assert isinstance(tech_rotation, str)
    
    def test_earnings_season_analysis(self, specialist):
        """Test earnings season analysis workflow."""
        # Test during earnings season
        earnings_date = datetime(2024, 1, 15)  # Mid-January (earnings season)
        earnings_info = specialist.check_earnings_calendar(earnings_date)
        assert isinstance(earnings_info, dict)
        
        # Test outside earnings season
        non_earnings_date = datetime(2024, 6, 15)  # Mid-June
        non_earnings_info = specialist.check_earnings_calendar(non_earnings_date)
        assert isinstance(non_earnings_info, dict)
    
    def test_fear_greed_analysis(self, specialist):
        """Test fear and greed index analysis workflow."""
        # Bullish market data
        bullish_data = {
            'VIX': {'close': 15.0},  # Low volatility
            'SPX500': {'close': 4200.0},  # High prices
            'PUT_CALL_RATIO': {'close': 0.6},  # Low put/call ratio
            'ADVANCE_DECLINE': {'close': 1.5}  # More advancing stocks
        }
        
        bullish_index = specialist.get_fear_greed_index(bullish_data)
        assert isinstance(bullish_index, float)
        
        # Bearish market data
        bearish_data = {
            'VIX': {'close': 35.0},  # High volatility
            'SPX500': {'close': 3800.0},  # Low prices
            'PUT_CALL_RATIO': {'close': 1.2},  # High put/call ratio
            'ADVANCE_DECLINE': {'close': 0.7}  # More declining stocks
        }
        
        bearish_index = specialist.get_fear_greed_index(bearish_data)
        assert isinstance(bearish_index, float)
        
        # Both indices should be valid floats
        assert isinstance(bullish_index, float)
        assert isinstance(bearish_index, float)
