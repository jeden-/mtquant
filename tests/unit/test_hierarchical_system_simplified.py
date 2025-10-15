"""
Simplified comprehensive tests for HierarchicalTradingSystem.

This module tests the main orchestrator of the hierarchical multi-agent trading system,
focusing on core functionality with proper mocking.
"""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from mtquant.agents.hierarchical.hierarchical_system import (
    HierarchicalTradingSystem, SystemState
)
from mtquant.agents.hierarchical.meta_controller import MetaController
from mtquant.agents.hierarchical.base_specialist import BaseSpecialist
from mtquant.agents.hierarchical.communication import (
    CommunicationHub, AllocationMessage, PerformanceReport
)
from mtquant.risk_management.portfolio_risk_manager import (
    PortfolioRiskManager, Portfolio, RiskLimits
)
from mtquant.mcp_integration.models.position import Position
from mtquant.mcp_integration.models.order import Order


class TestHierarchicalTradingSystemInitialization:
    """Test HierarchicalTradingSystem initialization."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        meta_controller = Mock(spec=MetaController)
        specialists = {
            'forex': Mock(spec=BaseSpecialist),
            'commodities': Mock(spec=BaseSpecialist),
            'equity': Mock(spec=BaseSpecialist)
        }
        
        # Mock specialist attributes
        for specialist in specialists.values():
            specialist.instruments = ['EURUSD', 'XAUUSD', 'SPX500']
            specialist.specialist_type = 'test'
        
        portfolio_risk_manager = Mock(spec=PortfolioRiskManager)
        communication_hub = Mock(spec=CommunicationHub)
        
        return {
            'meta_controller': meta_controller,
            'specialists': specialists,
            'portfolio_risk_manager': portfolio_risk_manager,
            'communication_hub': communication_hub
        }
    
    def test_initialization_basic(self, mock_components):
        """Test basic system initialization."""
        system = HierarchicalTradingSystem(
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager'],
            communication_hub=mock_components['communication_hub']
        )
        
        assert system.meta_controller == mock_components['meta_controller']
        assert system.specialists == mock_components['specialists']
        assert system.portfolio_risk_manager == mock_components['portfolio_risk_manager']
        assert system.communication_hub == mock_components['communication_hub']
        assert system.system_id == "hierarchical_system"
        assert system.current_state is None
        assert len(system.state_history) == 0
        assert system.training_mode is False
        assert system.risk_enabled is True
        assert system.logging_enabled is True
    
    def test_initialization_with_custom_id(self, mock_components):
        """Test initialization with custom system ID."""
        system = HierarchicalTradingSystem(
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager'],
            communication_hub=mock_components['communication_hub'],
            system_id="custom_system"
        )
        
        assert system.system_id == "custom_system"
    
    def test_initialization_stats(self, mock_components):
        """Test initialization statistics."""
        system = HierarchicalTradingSystem(
            meta_controller=mock_components['meta_controller'],
            specialists=mock_components['specialists'],
            portfolio_risk_manager=mock_components['portfolio_risk_manager'],
            communication_hub=mock_components['communication_hub']
        )
        
        assert system.stats['total_decisions'] == 0
        assert system.stats['orders_executed'] == 0
        assert system.stats['orders_rejected'] == 0
        assert system.stats['risk_violations'] == 0
        assert system.stats['avg_decision_time_ms'] == 0.0
        assert system.stats['last_decision_time'] is None


class TestHierarchicalTradingSystemConfiguration:
    """Test system configuration methods."""
    
    @pytest.fixture
    def system_with_mocks(self):
        """Create system with mocked components."""
        meta_controller = Mock(spec=MetaController)
        specialists = {'forex': Mock(spec=BaseSpecialist)}
        
        # Mock specialist attributes
        specialists['forex'].instruments = ['EURUSD']
        specialists['forex'].specialist_type = 'forex'
        
        portfolio_risk_manager = Mock(spec=PortfolioRiskManager)
        communication_hub = Mock(spec=CommunicationHub)
        
        system = HierarchicalTradingSystem(
            meta_controller=meta_controller,
            specialists=specialists,
            portfolio_risk_manager=portfolio_risk_manager,
            communication_hub=communication_hub
        )
        
        return system
    
    def test_enable_training_mode(self, system_with_mocks):
        """Test enable_training_mode method."""
        system = system_with_mocks
        
        system.enable_training_mode()
        assert system.training_mode is True
    
    def test_disable_training_mode(self, system_with_mocks):
        """Test disable_training_mode method."""
        system = system_with_mocks
        
        system.disable_training_mode()
        assert system.training_mode is False
    
    def test_enable_risk_management(self, system_with_mocks):
        """Test enable_risk_management method."""
        system = system_with_mocks
        
        system.enable_risk_management()
        assert system.risk_enabled is True
    
    def test_disable_risk_management(self, system_with_mocks):
        """Test disable_risk_management method."""
        system = system_with_mocks
        
        system.disable_risk_management()
        assert system.risk_enabled is False
    
    def test_get_system_status(self, system_with_mocks):
        """Test get_system_status method."""
        system = system_with_mocks
        
        # Set some state
        system.stats['total_decisions'] = 10
        system.stats['orders_executed'] = 8
        system.stats['orders_rejected'] = 2
        
        # Test method
        status = system.get_system_status()
        
        # Verify status
        assert isinstance(status, dict)
        assert 'system_id' in status
        assert 'training_mode' in status
        assert 'risk_enabled' in status
        assert 'statistics' in status
        assert status['statistics']['total_decisions'] == 10
        assert status['statistics']['orders_executed'] == 8
        assert status['statistics']['orders_rejected'] == 2
    
    def test_get_decision_history(self, system_with_mocks):
        """Test get_decision_history method."""
        system = system_with_mocks
        
        # Add some state history
        state1 = Mock(spec=SystemState)
        state1.timestamp = datetime.utcnow()
        state2 = Mock(spec=SystemState)
        state2.timestamp = datetime.utcnow()
        
        system.state_history = [state1, state2]
        
        # Test method
        history = system.get_decision_history()
        
        # Verify history
        assert isinstance(history, list)
        assert len(history) == 2
    
    def test_export_system_state(self, system_with_mocks):
        """Test export_system_state method."""
        system = system_with_mocks
        
        # Set current state
        current_state = Mock(spec=SystemState)
        current_state.timestamp = datetime.utcnow()
        current_state.market_data = {'EURUSD': {'close': 1.1000}}
        current_state.portfolio = Mock(spec=Portfolio)
        current_state.positions = []
        current_state.meta_decisions = {'allocations': [0.4, 0.3, 0.3]}
        current_state.specialist_actions = {'forex': {'EURUSD': {'signal': 0.8}}}
        current_state.risk_validation = {'passed': True}
        current_state.approved_orders = []
        current_state.rejected_orders = []
        current_state.system_metrics = {'performance': 0.05}
        current_state.to_dict = Mock(return_value={'test': 'data'})
        
        system.current_state = current_state
        
        # Test method
        exported = system.export_system_state('test_export.json')
        
        # Verify export
        assert exported is None  # Method returns None


class TestHierarchicalTradingSystemPortfolioState:
    """Test portfolio state management."""
    
    @pytest.fixture
    def system_with_mocks(self):
        """Create system with mocked components."""
        meta_controller = Mock(spec=MetaController)
        specialists = {'forex': Mock(spec=BaseSpecialist)}
        
        # Mock specialist attributes
        specialists['forex'].instruments = ['EURUSD']
        specialists['forex'].get_domain_features.return_value = torch.randn(50)
        
        portfolio_risk_manager = Mock(spec=PortfolioRiskManager)
        communication_hub = Mock(spec=CommunicationHub)
        
        system = HierarchicalTradingSystem(
            meta_controller=meta_controller,
            specialists=specialists,
            portfolio_risk_manager=portfolio_risk_manager,
            communication_hub=communication_hub
        )
        
        return system, {
            'meta_controller': meta_controller,
            'specialists': specialists,
            'portfolio_risk_manager': portfolio_risk_manager,
            'communication_hub': communication_hub
        }
    
    def test_get_portfolio_state(self, system_with_mocks):
        """Test get_portfolio_state method."""
        system, mocks = system_with_mocks
        
        # Mock portfolio
        portfolio = Mock(spec=Portfolio)
        portfolio.equity = 100000.0
        portfolio.equity = 100000.0
        portfolio.returns_history = np.random.randn(30, 8)
        portfolio.correlation_matrix = np.eye(8)
        
        # Mock meta controller
        expected_state = torch.randn(74)
        mocks['meta_controller'].get_portfolio_state.return_value = expected_state
        
        # Test method
        result = system.get_portfolio_state(portfolio, mocks['specialists'])
        
        # Verify result
        assert torch.equal(result, expected_state)
        mocks['meta_controller'].get_portfolio_state.assert_called_once()
    
    def test_get_specialist_states(self, system_with_mocks):
        """Test get_specialist_states method."""
        system, mocks = system_with_mocks
        
        # Mock market data with proper structure (single values, not dicts)
        market_data = {'EURUSD': 1.1000}
        
        # Test method
        result = system.get_specialist_states(market_data, mocks['specialists'])
        
        # Verify result
        assert isinstance(result, dict)
        assert 'forex' in result
        assert 'market_state' in result['forex']
        assert 'instrument_states' in result['forex']
        
        # Verify specialist was called
        mocks['specialists']['forex'].get_domain_features.assert_called_once()


class TestHierarchicalTradingSystemRiskManagement:
    """Test risk management integration."""
    
    @pytest.fixture
    def system_with_mocks(self):
        """Create system with mocked components."""
        meta_controller = Mock(spec=MetaController)
        specialists = {'forex': Mock(spec=BaseSpecialist)}
        
        # Mock specialist attributes
        specialists['forex'].instruments = ['EURUSD']
        
        portfolio_risk_manager = Mock(spec=PortfolioRiskManager)
        communication_hub = Mock(spec=CommunicationHub)
        
        system = HierarchicalTradingSystem(
            meta_controller=meta_controller,
            specialists=specialists,
            portfolio_risk_manager=portfolio_risk_manager,
            communication_hub=communication_hub
        )
        
        return system, {
            'meta_controller': meta_controller,
            'specialists': specialists,
            'portfolio_risk_manager': portfolio_risk_manager,
            'communication_hub': communication_hub
        }
    
    def test_scale_down_to_risk_limit(self, system_with_mocks):
        """Test scale_down_to_risk_limit method."""
        system, mocks = system_with_mocks
        
        # Mock orders (not specialist actions)
        orders = [
            Order(
                order_id="test_1",
                agent_id="forex",
                symbol="EURUSD",
                side="buy",
                order_type="market",
                quantity=0.1,
                signal=0.8
            ),
            Order(
                order_id="test_2",
                agent_id="commodities",
                symbol="XAUUSD",
                side="sell",
                order_type="market",
                quantity=0.05,
                signal=-0.3
            )
        ]
        
        # Mock portfolio
        portfolio = Mock(spec=Portfolio)
        portfolio.equity = 100000.0
        
        # Test method
        result = system.scale_down_to_risk_limit(orders, portfolio)
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) <= len(orders)  # Should be scaled down or same
    
    def test_risk_disabled(self, system_with_mocks):
        """Test behavior when risk management is disabled."""
        system, mocks = system_with_mocks
        system.risk_enabled = False
        
        # Mock orders
        orders = [
            Order(
                order_id="test_1",
                agent_id="forex",
                symbol="EURUSD",
                side="buy",
                order_type="market",
                quantity=0.1,
                signal=0.8
            )
        ]
        
        # Mock portfolio
        portfolio = Mock(spec=Portfolio)
        portfolio.equity = 100000.0
        
        # Test method
        result = system.scale_down_to_risk_limit(orders, portfolio)
        
        # Verify actions are returned unchanged when risk is disabled
        assert result == orders
        
        # Verify risk manager was not called
        mocks['portfolio_risk_manager'].check_portfolio_risk.assert_not_called()


class TestHierarchicalTradingSystemEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def system_with_mocks(self):
        """Create system with mocked components."""
        meta_controller = Mock(spec=MetaController)
        specialists = {'forex': Mock(spec=BaseSpecialist)}
        
        # Mock specialist attributes
        specialists['forex'].instruments = ['EURUSD']
        
        portfolio_risk_manager = Mock(spec=PortfolioRiskManager)
        communication_hub = Mock(spec=CommunicationHub)
        
        system = HierarchicalTradingSystem(
            meta_controller=meta_controller,
            specialists=specialists,
            portfolio_risk_manager=portfolio_risk_manager,
            communication_hub=communication_hub
        )
        
        return system, {
            'meta_controller': meta_controller,
            'specialists': specialists,
            'portfolio_risk_manager': portfolio_risk_manager,
            'communication_hub': communication_hub
        }
    
    def test_step_with_none_portfolio(self, system_with_mocks):
        """Test step with None portfolio."""
        system, mocks = system_with_mocks
        
        market_data = {'EURUSD': {'close': 1.1000}}
        
        # Mock current positions
        current_positions = []
        
        # Test method
        result = system.step(market_data, None, current_positions)
        
        # Verify graceful handling
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_step_with_empty_market_data(self, system_with_mocks):
        """Test step with empty market data."""
        system, mocks = system_with_mocks
        
        market_data = {}
        portfolio = Mock(spec=Portfolio)
        
        # Mock current positions
        current_positions = []
        
        # Test method
        result = system.step(market_data, portfolio, current_positions)
        
        # Verify graceful handling
        assert isinstance(result, list)
    
    def test_step_with_meta_controller_error(self, system_with_mocks):
        """Test step with meta controller error."""
        system, mocks = system_with_mocks
        
        market_data = {'EURUSD': {'close': 1.1000}}
        portfolio = Mock(spec=Portfolio)
        
        # Mock meta controller error
        mocks['meta_controller'].get_portfolio_state.side_effect = Exception("Meta controller error")
        
        # Mock current positions
        current_positions = []
        
        # Test method
        result = system.step(market_data, portfolio, current_positions)
        
        # Verify graceful error handling
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_step_with_communication_error(self, system_with_mocks):
        """Test step with communication hub error."""
        system, mocks = system_with_mocks
        
        market_data = {'EURUSD': {'close': 1.1000}}
        portfolio = Mock(spec=Portfolio)
        
        # Mock portfolio state
        portfolio_state = torch.randn(74)
        mocks['meta_controller'].get_portfolio_state.return_value = portfolio_state
        
        # Mock meta controller decisions
        allocations = torch.tensor([0.4, 0.3, 0.3])
        risk_appetite = torch.tensor([0.7])
        value = torch.tensor([0.5])
        mocks['meta_controller'].forward.return_value = (allocations, risk_appetite, value)
        
        # Mock communication error
        mocks['communication_hub'].send_allocation.side_effect = Exception("Communication error")
        
        # Mock current positions
        current_positions = []
        
        # Test method
        result = system.step(market_data, portfolio, current_positions)
        
        # Verify graceful error handling
        assert isinstance(result, list)
    
    def test_flatten_specialist_actions_empty(self, system_with_mocks):
        """Test flatten_specialist_actions with empty actions."""
        system, mocks = system_with_mocks
        
        specialist_actions = {}
        
        # Mock current positions
        current_positions = []
        
        # Test method
        result = system.flatten_specialist_actions(specialist_actions, current_positions)
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_flatten_specialist_actions_none(self, system_with_mocks):
        """Test flatten_specialist_actions with None actions."""
        system, mocks = system_with_mocks
        
        # Mock current positions
        current_positions = []
        
        # Test method - should handle None gracefully
        try:
            result = system.flatten_specialist_actions(None, current_positions)
            # If it doesn't raise an exception, verify result
            assert isinstance(result, list)
        except AttributeError:
            # Expected behavior - None has no 'items' attribute
            pass


class TestSystemState:
    """Test SystemState dataclass."""
    
    def test_system_state_initialization(self):
        """Test SystemState initialization."""
        timestamp = datetime.utcnow()
        market_data = {'EURUSD': {'close': 1.1000}}
        portfolio = Mock(spec=Portfolio)
        positions = []
        meta_decisions = {'allocations': [0.4, 0.3, 0.3]}
        specialist_actions = {'forex': {'EURUSD': {'signal': 0.8}}}
        risk_validation = {'passed': True}
        approved_orders = []
        rejected_orders = []
        system_metrics = {'performance': 0.05}
        
        state = SystemState(
            timestamp=timestamp,
            market_data=market_data,
            portfolio=portfolio,
            positions=positions,
            meta_decisions=meta_decisions,
            specialist_actions=specialist_actions,
            risk_validation=risk_validation,
            approved_orders=approved_orders,
            rejected_orders=rejected_orders,
            system_metrics=system_metrics
        )
        
        assert state.timestamp == timestamp
        assert state.market_data == market_data
        assert state.portfolio == portfolio
        assert state.positions == positions
        assert state.meta_decisions == meta_decisions
        assert state.specialist_actions == specialist_actions
        assert state.risk_validation == risk_validation
        assert state.approved_orders == approved_orders
        assert state.rejected_orders == rejected_orders
        assert state.system_metrics == system_metrics
    
    def test_system_state_default_values(self):
        """Test SystemState with default values."""
        timestamp = datetime.utcnow()
        market_data = {'EURUSD': {'close': 1.1000}}
        portfolio = Mock(spec=Portfolio)
        
        state = SystemState(
            timestamp=timestamp,
            market_data=market_data,
            portfolio=portfolio,
            positions=[],
            meta_decisions={},
            specialist_actions={},
            risk_validation={},
            approved_orders=[],
            rejected_orders=[],
            system_metrics={}
        )
        
        assert state.timestamp == timestamp
        assert state.market_data == market_data
        assert state.portfolio == portfolio
        assert state.positions == []
        assert state.meta_decisions == {}
        assert state.specialist_actions == {}
        assert state.risk_validation == {}
        assert state.approved_orders == []
        assert state.rejected_orders == []
        assert state.system_metrics == {}
