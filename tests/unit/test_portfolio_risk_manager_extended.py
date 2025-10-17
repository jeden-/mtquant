"""
Extended tests for PortfolioRiskManager.

This module tests additional methods in PortfolioRiskManager that were not
covered in the original tests.
"""

import pytest
import numpy as np
import tempfile
import os
import yaml
from datetime import datetime
from unittest.mock import Mock, patch

from mtquant.risk_management.portfolio_risk_manager import (
    PortfolioRiskManager, Portfolio, RiskLimits, CorrelationTracker
)
from mtquant.mcp_integration.models.position import Position


class TestPortfolioRiskManagerExtended:
    """Test additional PortfolioRiskManager methods."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create PortfolioRiskManager for testing."""
        return PortfolioRiskManager()
    
    @pytest.fixture
    def sample_positions(self):
        """Create sample positions for testing."""
        return [
            Position(
                position_id="pos_1",
                agent_id="test_agent",
                symbol="EURUSD",
                side="long",
                quantity=0.1,
                entry_price=1.1,
                current_price=1.1
            ),
            Position(
                position_id="pos_2",
                agent_id="test_agent",
                symbol="XAUUSD",
                side="long",
                quantity=0.05,
                entry_price=2000.0,
                current_price=2000.0
            ),
            Position(
                position_id="pos_3",
                agent_id="test_agent",
                symbol="SPX500",
                side="long",
                quantity=1.0,
                entry_price=4000.0,
                current_price=4000.0
            )
        ]
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio for testing."""
        return Portfolio(
            equity=100000.0,
            margin_used=5000.0,
            margin_available=95000.0,
            positions=[],
            returns_history=np.random.randn(30, 8),
            correlation_matrix=np.eye(8),
            sector_allocation={'forex': 0.3, 'commodities': 0.2, 'equity': 0.5},
            last_updated=datetime.utcnow()
        )
    
    def test_load_config_valid_file(self, risk_manager):
        """Test loading configuration from valid YAML file."""
        # Create temporary config file
        config_data = {
            'portfolio_risk': {
                'max_portfolio_var': 0.03,
                'max_correlation_exposure': 0.8,
                'max_sector_allocation': 0.5,
                'correlation_window': 150,
                'var_confidence': 0.99,
                'margin_buffer': 0.15
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Test loading config
            config = risk_manager._load_config(config_path)
            
            assert isinstance(config, RiskLimits)
            assert config.max_portfolio_var == 0.03
            assert config.max_correlation_exposure == 0.8
            assert config.max_sector_allocation == 0.5
            assert config.var_calculation_window == 150
            assert config.var_confidence_level == 0.99
            assert config.margin_buffer == 0.15
        finally:
            os.unlink(config_path)
    
    def test_load_config_invalid_file(self, risk_manager):
        """Test loading configuration from invalid file."""
        # Create temporary invalid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            # Test loading invalid config
            config = risk_manager._load_config(config_path)
            
            # Should return default RiskLimits
            assert isinstance(config, RiskLimits)
            assert config.max_portfolio_var == 0.02  # Default value
        finally:
            os.unlink(config_path)
    
    def test_load_config_nonexistent_file(self, risk_manager):
        """Test loading configuration from nonexistent file."""
        config = risk_manager._load_config("nonexistent_file.yaml")
        
        # Should return default RiskLimits
        assert isinstance(config, RiskLimits)
        assert config.max_portfolio_var == 0.02  # Default value
    
    def test_calculate_sector_allocation(self, risk_manager, sample_positions):
        """Test calculate_sector_allocation method."""
        allocation = risk_manager.calculate_sector_allocation(sample_positions)
        
        assert isinstance(allocation, dict)
        assert 'forex' in allocation
        assert 'commodities' in allocation
        assert 'equity' in allocation
        
        # Check that allocations sum to reasonable values
        total_allocation = sum(allocation.values())
        assert total_allocation > 0
    
    def test_calculate_sector_allocation_empty_positions(self, risk_manager):
        """Test calculate_sector_allocation with empty positions."""
        allocation = risk_manager.calculate_sector_allocation([])
        
        assert isinstance(allocation, dict)
        assert 'forex' in allocation
        assert 'commodities' in allocation
        assert 'equity' in allocation
        
        # All allocations should be 0
        for sector, value in allocation.items():
            assert value == 0.0
    
    def test_calculate_sector_allocation_unknown_instrument(self, risk_manager):
        """Test calculate_sector_allocation with unknown instrument."""
        positions = [
            Position(
                position_id="pos_1",
                agent_id="test_agent",
                symbol="UNKNOWN",
                side="long",
                quantity=0.1,
                entry_price=1.0,
                current_price=1.0
            )
        ]
        
        allocation = risk_manager.calculate_sector_allocation(positions)
        
        assert isinstance(allocation, dict)
        # Should handle unknown instruments gracefully
        assert 'forex' in allocation
        assert 'commodities' in allocation
        assert 'equity' in allocation
    
    def test_check_margin_requirement_sufficient(self, risk_manager, sample_positions):
        """Test check_margin_requirement with sufficient margin."""
        available_margin = 100000.0
        
        result = risk_manager.check_margin_requirement(sample_positions, available_margin)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
        
        # Should pass with sufficient margin
        assert result[0] == True
        assert result[1] > 0  # Required margin
    
    def test_check_margin_requirement_insufficient(self, risk_manager, sample_positions):
        """Test check_margin_requirement with insufficient margin."""
        available_margin = 1.0  # Very low margin
        
        result = risk_manager.check_margin_requirement(sample_positions, available_margin)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
        
        # Should fail with insufficient margin
        assert result[0] == False
        assert result[1] > available_margin  # Required margin > available
    
    def test_check_margin_requirement_empty_positions(self, risk_manager):
        """Test check_margin_requirement with empty positions."""
        available_margin = 100000.0
        
        result = risk_manager.check_margin_requirement([], available_margin)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
        
        # Should pass with no positions
        assert result[0] == True
        assert result[1] == 0.0  # No margin required
    
    def test_check_margin_requirement_zero_margin(self, risk_manager, sample_positions):
        """Test check_margin_requirement with zero available margin."""
        available_margin = 0.0
        
        result = risk_manager.check_margin_requirement(sample_positions, available_margin)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
        
        # Should fail with zero margin
        assert result[0] == False
        assert result[1] > 0  # Some margin required
    
    def test_correlation_tracker_get_max_correlation_exposure(self, risk_manager, sample_positions):
        """Test CorrelationTracker.get_max_correlation_exposure method."""
        # Add some returns to correlation tracker
        returns = {
            'EURUSD': 0.01,
            'GBPUSD': 0.02,
            'USDJPY': -0.01,
            'XAUUSD': 0.03,
            'WTIUSD': -0.02,
            'SPX500': 0.015,
            'NAS100': 0.02,
            'US30': 0.01
        }
        
        for _ in range(10):  # Add multiple returns
            risk_manager.correlation_tracker.update(returns)
        
        # Test method
        max_exposure = risk_manager.correlation_tracker.get_max_correlation_exposure(sample_positions)
        
        assert isinstance(max_exposure, float)
        assert 0.0 <= max_exposure <= 1.0
    
    def test_correlation_tracker_get_max_correlation_exposure_empty(self, risk_manager):
        """Test get_max_correlation_exposure with empty positions."""
        max_exposure = risk_manager.correlation_tracker.get_max_correlation_exposure([])
        
        assert isinstance(max_exposure, float)
        assert max_exposure == 0.0
    
    def test_correlation_tracker_get_max_correlation_exposure_zero_value(self, risk_manager):
        """Test get_max_correlation_exposure with zero position values."""
        positions = [
            Position(
                position_id="pos_1",
                agent_id="test_agent",
                symbol="EURUSD",
                side="long",
                quantity=0.0,  # Zero quantity
                entry_price=1.1,
                current_price=1.1
            )
        ]
        
        max_exposure = risk_manager.correlation_tracker.get_max_correlation_exposure(positions)
        
        assert isinstance(max_exposure, float)
        assert max_exposure == 0.0
    
    def test_correlation_tracker_visualize_correlation_heatmap(self, risk_manager):
        """Test visualize_correlation_heatmap method."""
        # Add some returns to correlation tracker
        returns = {
            'EURUSD': 0.01,
            'GBPUSD': 0.02,
            'USDJPY': -0.01,
            'XAUUSD': 0.03,
            'WTIUSD': -0.02,
            'SPX500': 0.015,
            'NAS100': 0.02,
            'US30': 0.01
        }
        
        for _ in range(10):  # Add multiple returns
            risk_manager.correlation_tracker.update(returns)
        
        # Test method
        result = risk_manager.correlation_tracker.visualize_correlation_heatmap()
        
        # Should return None if matplotlib is not available, or Figure if available
        assert result is None or hasattr(result, 'savefig')
    
    def test_correlation_tracker_detect_regime_change(self, risk_manager):
        """Test detect_regime_change method."""
        # Add initial returns
        returns = {
            'EURUSD': 0.01,
            'GBPUSD': 0.02,
            'USDJPY': -0.01,
            'XAUUSD': 0.03,
            'WTIUSD': -0.02,
            'SPX500': 0.015,
            'NAS100': 0.02,
            'US30': 0.01
        }
        
        # Add enough returns for regime detection
        for _ in range(15):
            risk_manager.correlation_tracker.update(returns)
        
        # Test method
        regime = risk_manager.correlation_tracker.detect_regime_change()
        
        # Should return None, 'correlation_spike', or 'correlation_breakdown'
        assert regime is None or regime in ['correlation_spike', 'correlation_breakdown']
    
    def test_correlation_tracker_detect_regime_change_insufficient_data(self, risk_manager):
        """Test detect_regime_change with insufficient data."""
        # Add only a few returns
        returns = {
            'EURUSD': 0.01,
            'GBPUSD': 0.02,
            'USDJPY': -0.01,
            'XAUUSD': 0.03,
            'WTIUSD': -0.02,
            'SPX500': 0.015,
            'NAS100': 0.02,
            'US30': 0.01
        }
        
        for _ in range(5):  # Not enough for regime detection
            risk_manager.correlation_tracker.update(returns)
        
        # Test method
        regime = risk_manager.correlation_tracker.detect_regime_change()
        
        # Should return None with insufficient data
        assert regime is None
    
    def test_correlation_tracker_get_current_correlations(self, risk_manager):
        """Test get_current_correlations method."""
        # Add some returns to correlation tracker
        returns = {
            'EURUSD': 0.01,
            'GBPUSD': 0.02,
            'USDJPY': -0.01,
            'XAUUSD': 0.03,
            'WTIUSD': -0.02,
            'SPX500': 0.015,
            'NAS100': 0.02,
            'US30': 0.01
        }
        
        for _ in range(10):
            risk_manager.correlation_tracker.update(returns)
        
        # Test method
        correlations = risk_manager.correlation_tracker.get_current_correlations()
        
        assert isinstance(correlations, np.ndarray)
        assert correlations.shape == (8, 8)  # 8x8 correlation matrix
        assert np.allclose(correlations, correlations.T)  # Should be symmetric
    
    def test_correlation_tracker_calculate_average_correlation(self, risk_manager):
        """Test _calculate_average_correlation method."""
        # Create test correlation matrix
        corr_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
        
        # Test method
        avg_corr = risk_manager.correlation_tracker._calculate_average_correlation(corr_matrix)
        
        assert isinstance(avg_corr, float)
        assert 0.0 <= avg_corr <= 1.0
        
        # Should be average of upper triangle (excluding diagonal)
        expected = (0.5 + 0.3 + 0.2) / 3
        assert abs(avg_corr - expected) < 1e-10
    
    def test_correlation_tracker_calculate_average_correlation_identity(self, risk_manager):
        """Test _calculate_average_correlation with identity matrix."""
        # Create identity matrix
        corr_matrix = np.eye(3)
        
        # Test method
        avg_corr = risk_manager.correlation_tracker._calculate_average_correlation(corr_matrix)
        
        assert isinstance(avg_corr, float)
        assert avg_corr == 0.0  # All off-diagonal elements are 0
    
    def test_correlation_tracker_calculate_average_correlation_single_element(self, risk_manager):
        """Test _calculate_average_correlation with single element matrix."""
        # Create 1x1 matrix
        corr_matrix = np.array([[1.0]])
        
        # Test method
        avg_corr = risk_manager.correlation_tracker._calculate_average_correlation(corr_matrix)
        
        assert isinstance(avg_corr, float)
        assert avg_corr == 0.0  # No off-diagonal elements




