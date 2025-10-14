"""
Additional comprehensive tests for hierarchical multi-agent system components.

Tests more functionality to increase code coverage further.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from datetime import datetime

# Import hierarchical components
from mtquant.agents.hierarchical import (
    MetaController,
    ForexSpecialist,
    CommoditiesSpecialist,
    EquitySpecialist,
    CommunicationHub,
    HierarchicalTradingSystem,
    SystemState,
    AllocationMessage,
    PerformanceReport,
    CoordinationSignal,
    AlertMessage,
    SpecialistRegistry,
    get_specialist_registry,
    create_specialist
)

from mtquant.risk_management import PortfolioRiskManager, CorrelationTracker
from mtquant.mcp_integration.models.position import Position
from mtquant.mcp_integration.models.order import Order


class TestMetaControllerComprehensive:
    """Comprehensive tests for MetaController functionality."""
    
    def test_forward_pass_shape(self):
        """Test forward pass output shapes."""
        controller = MetaController()
        
        # Create portfolio state tensor
        portfolio_state = torch.randn(74)
        
        allocations, risk_appetite, value = controller.forward(portfolio_state)
        
        assert allocations.shape == (3,)  # 3 specialists
        assert risk_appetite.shape == (1,)
        assert value.shape == (1,)  # Single value
        
        # Check softmax normalization
        assert torch.allclose(allocations.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.all(allocations >= 0)  # All positive
        
        # Check sigmoid bounds
        assert 0 <= risk_appetite.item() <= 1
    
    def test_batch_forward_pass(self):
        """Test forward pass with batch input."""
        controller = MetaController()
        
        # Create batch of portfolio states
        batch_size = 5
        portfolio_states = torch.randn(batch_size, 74)
        
        allocations, risk_appetite, value = controller.forward(portfolio_states)
        
        assert allocations.shape == (batch_size, 3)
        assert risk_appetite.shape == (batch_size, 1)
        assert value.shape == (batch_size, 1)
        
        # Check softmax normalization for each batch item
        for i in range(batch_size):
            assert torch.allclose(allocations[i].sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_get_portfolio_state_edge_cases(self):
        """Test portfolio state calculation with edge cases."""
        controller = MetaController()
        
        # Test with empty portfolio
        empty_portfolio = {}
        empty_specialists = {}
        
        state = controller.get_portfolio_state(empty_portfolio, empty_specialists)
        assert isinstance(state, torch.Tensor)
        assert state.shape[0] == 74
        
        # Test with missing data
        partial_portfolio = {'returns': np.array([0.01, 0.02])}  # Only 2 days
        partial_specialists = {'forex': {'sharpe': 1.2}}  # Missing other specialists
        
        state = controller.get_portfolio_state(partial_portfolio, partial_specialists)
        assert isinstance(state, torch.Tensor)
        assert state.shape[0] == 74
    
    def test_detect_market_regime_edge_cases(self):
        """Test market regime detection with edge cases."""
        controller = MetaController()
        
        # Empty returns
        regime = controller.detect_market_regime(np.array([]))
        assert regime in ['bull', 'bear', 'sideways', 'volatile', 'neutral']
        
        # Single return
        regime = controller.detect_market_regime(np.array([0.01]))
        assert regime in ['bull', 'bear', 'sideways', 'volatile', 'neutral']
        
        # All zeros
        regime = controller.detect_market_regime(np.zeros(10))
        assert regime in ['bull', 'bear', 'sideways', 'volatile', 'neutral']
        
        # Very volatile
        volatile_returns = np.array([0.1, -0.1, 0.15, -0.15, 0.2, -0.2])
        regime = controller.detect_market_regime(volatile_returns)
        assert regime in ['bull', 'bear', 'sideways', 'volatile', 'neutral']
    
    def test_calculate_kelly_allocation_edge_cases(self):
        """Test Kelly allocation with edge cases."""
        controller = MetaController()
        
        # All negative Sharpe ratios
        negative_sharpes = np.array([-0.5, -0.3, -0.8])
        allocations = controller.calculate_kelly_allocation(negative_sharpes)
        assert isinstance(allocations, np.ndarray)
        assert allocations.shape == (3,)
        assert np.allclose(allocations.sum(), 1.0, atol=1e-6)
        
        # All zeros
        zero_sharpes = np.zeros(3)
        allocations = controller.calculate_kelly_allocation(zero_sharpes)
        assert isinstance(allocations, np.ndarray)
        assert allocations.shape == (3,)
        assert np.allclose(allocations.sum(), 1.0, atol=1e-6)
        
        # Wrong number of specialists
        with pytest.raises(ValueError):
            controller.calculate_kelly_allocation(np.array([1.0, 2.0]))  # Only 2


class TestSpecialistsComprehensive:
    """Comprehensive tests for Specialist functionality."""
    
    def test_forex_specialist_forward_pass(self):
        """Test ForexSpecialist forward pass."""
        specialist = ForexSpecialist()
        
        # Mock inputs with correct dimensions (add batch dimension)
        market_state = torch.randn(1, 8)  # batch_size=1, market_features_dim
        instrument_states = {
            'EURUSD': torch.randn(1, 50),  # batch_size=1, observation_dim
            'GBPUSD': torch.randn(1, 50),
            'USDJPY': torch.randn(1, 50)
        }
        allocation = 0.4  # Scalar allocation
        
        actions, value = specialist.forward(market_state, instrument_states, allocation)
        
        assert isinstance(actions, dict)
        assert isinstance(value, torch.Tensor)
        assert value.shape == (1, 1)  # batch_size=1, value_dim=1
        assert 'EURUSD' in actions
        assert 'GBPUSD' in actions
        assert 'USDJPY' in actions
    
    def test_commodities_specialist_forward_pass(self):
        """Test CommoditiesSpecialist forward pass."""
        specialist = CommoditiesSpecialist()
        
        # Mock inputs with correct dimensions (add batch dimension)
        market_state = torch.randn(1, 6)  # batch_size=1, market_features_dim for commodities
        instrument_states = {
            'XAUUSD': torch.randn(1, 50),  # batch_size=1, observation_dim
            'WTIUSD': torch.randn(1, 50)
        }
        allocation = 0.3  # Scalar allocation
        
        actions, value = specialist.forward(market_state, instrument_states, allocation)
        
        assert isinstance(actions, dict)
        assert isinstance(value, torch.Tensor)
        assert value.shape == (1, 1)  # batch_size=1, value_dim=1
        assert 'XAUUSD' in actions
        assert 'WTIUSD' in actions
    
    def test_equity_specialist_forward_pass(self):
        """Test EquitySpecialist forward pass."""
        specialist = EquitySpecialist()
        
        # Mock inputs with correct dimensions (add batch dimension)
        market_state = torch.randn(1, 7)  # batch_size=1, market_features_dim for equity
        instrument_states = {
            'SPX500': torch.randn(1, 50),  # batch_size=1, observation_dim
            'NAS100': torch.randn(1, 50),
            'US30': torch.randn(1, 50)
        }
        allocation = 0.5  # Scalar allocation
        
        actions, value = specialist.forward(market_state, instrument_states, allocation)
        
        assert isinstance(actions, dict)
        assert isinstance(value, torch.Tensor)
        assert value.shape == (1, 1)  # batch_size=1, value_dim=1
        assert 'SPX500' in actions
        assert 'NAS100' in actions
        assert 'US30' in actions
    
    def test_specialist_get_instruments(self):
        """Test specialist instrument retrieval."""
        forex = ForexSpecialist()
        commodities = CommoditiesSpecialist()
        equity = EquitySpecialist()
        
        assert forex.get_instruments() == ['EURUSD', 'GBPUSD', 'USDJPY']
        assert commodities.get_instruments() == ['XAUUSD', 'WTIUSD']
        assert equity.get_instruments() == ['SPX500', 'NAS100', 'US30']
    
    def test_specialist_calculate_confidence(self):
        """Test specialist confidence calculation."""
        forex = ForexSpecialist()
        
        # Mock actions (should be tensor, not dict)
        actions = {
            'EURUSD': torch.tensor([0.3, 0.4, 0.3]),  # buy, hold, sell probabilities
            'GBPUSD': torch.tensor([0.2, 0.5, 0.3]),
            'USDJPY': torch.tensor([0.4, 0.3, 0.3])
        }
        
        confidence = forex.calculate_confidence(actions)
        
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


class TestCommunicationHubComprehensive:
    """Comprehensive tests for CommunicationHub functionality."""
    
    def test_message_types(self):
        """Test different message types."""
        hub = CommunicationHub()
        
        # Test AllocationMessage
        alloc_msg = AllocationMessage(
            specialist_id="forex",
            allocation=0.4,
            risk_appetite=0.6,
            market_regime="bull"
        )
        hub.send_message(alloc_msg)
        
        # Test PerformanceReport
        perf_msg = PerformanceReport(
            specialist_id="forex",
            confidence_score=0.8,
            realized_pnl=100.0,
            unrealized_pnl=50.0,
            sharpe_ratio=1.5,
            win_rate=0.6,
            risk_utilization=0.7
        )
        hub.send_message(perf_msg)
        
        # Test CoordinationSignal
        coord_msg = CoordinationSignal(
            from_specialist="meta_controller",
            to_specialist="forex",
            signal_type="reduce_exposure",
            data={"target_specialists": ["forex", "commodities"]},
            priority="high"
        )
        hub.send_message(coord_msg)
        
        # Test AlertMessage
        alert_msg = AlertMessage(
            alert_type="risk_limit_exceeded",
            severity="critical",
            message="Portfolio VaR exceeded",
            data={"affected_agents": ["forex"]}
        )
        hub.send_message(alert_msg)
        
        assert len(hub.message_history) == 4
    
    def test_message_filtering(self):
        """Test message filtering by agent."""
        hub = CommunicationHub()
        
        # Send messages from different agents
        msg1 = AllocationMessage(
            specialist_id="forex",
            allocation=0.4,
            risk_appetite=0.6,
            market_regime="bull"
        )
        msg2 = AllocationMessage(
            specialist_id="commodities",
            allocation=0.3,
            risk_appetite=0.5,
            market_regime="bull"
        )
        msg3 = PerformanceReport(
            specialist_id="forex",
            confidence_score=0.8,
            realized_pnl=100.0,
            unrealized_pnl=50.0,
            sharpe_ratio=1.5,
            win_rate=0.6,
            risk_utilization=0.7
        )
        
        hub.send_message(msg1)
        hub.send_message(msg2)
        hub.send_message(msg3)
        
        # Filter by agent
        forex_messages = hub.get_messages_by_agent("forex")
        commodities_messages = hub.get_messages_by_agent("commodities")
        
        assert len(forex_messages) == 2  # msg1 and msg3
        assert len(commodities_messages) == 1  # msg2
    
    def test_message_statistics(self):
        """Test message statistics tracking."""
        hub = CommunicationHub()
        
        # Send various messages
        for i in range(10):
            msg = AllocationMessage(
                specialist_id=f"specialist_{i % 3}",
                allocation=0.3,
                risk_appetite=0.5,
                market_regime="bull"
            )
            hub.send_message(msg)
        
        stats = hub.get_statistics()
        
        assert stats['total_messages'] == 10
        assert 'messages_by_type' in stats
        assert 'messages_by_agent' in stats
        # Check that messages are counted (key might be class name or type)
        assert len(stats['messages_by_type']) > 0


class TestPortfolioRiskManagerComprehensive:
    """Comprehensive tests for PortfolioRiskManager functionality."""
    
    def test_correlation_tracker_integration(self):
        """Test correlation tracker integration."""
        risk_manager = PortfolioRiskManager()
        
        # Test correlation tracker exists
        assert hasattr(risk_manager, 'correlation_tracker')
        assert isinstance(risk_manager.correlation_tracker, CorrelationTracker)
    
    def test_var_calculation_methods(self):
        """Test different VaR calculation methods."""
        risk_manager = PortfolioRiskManager()
        
        # Mock positions
        positions = [
            Position(
                position_id="1",
                agent_id="forex",
                symbol="EURUSD",
                side="long",
                quantity=0.1,
                entry_price=1.2,
                current_price=1.25,
                unrealized_pnl=5.0,
                opened_at=Mock(),
                broker_id="test"
            )
        ]
        
        # Mock returns history
        returns_history = np.random.normal(0, 0.02, (100, 8))
        
        # Test different methods
        methods = ['variance_covariance', 'historical', 'monte_carlo']
        
        for method in methods:
            var_result = risk_manager.calculate_var(positions, returns_history, method=method)
            assert isinstance(var_result, dict)
            assert 'var' in var_result
            assert var_result['var'] > 0
    
    def test_sector_allocation_calculation(self):
        """Test sector allocation calculation."""
        risk_manager = PortfolioRiskManager()
        
        # Mock positions from different sectors
        positions = [
            Position(
                position_id="1",
                agent_id="forex",
                symbol="EURUSD",
                side="long",
                quantity=0.1,
                entry_price=1.2,
                current_price=1.25,
                unrealized_pnl=5.0,
                opened_at=Mock(),
                broker_id="test"
            ),
            Position(
                position_id="2",
                agent_id="commodities",
                symbol="XAUUSD",
                side="long",
                quantity=0.05,
                entry_price=2050.0,
                current_price=2055.0,
                unrealized_pnl=2.5,
                opened_at=Mock(),
                broker_id="test"
            )
        ]
        
        sector_allocation = risk_manager.calculate_sector_allocation(positions)
        
        assert isinstance(sector_allocation, dict)
        assert 'forex' in sector_allocation
        assert 'commodities' in sector_allocation
    
    def test_margin_requirement_check(self):
        """Test margin requirement checking."""
        risk_manager = PortfolioRiskManager()
        
        # Mock positions
        positions = [
            Position(
                position_id="1",
                agent_id="forex",
                symbol="EURUSD",
                side="long",
                quantity=0.1,
                entry_price=1.2,
                current_price=1.25,
                unrealized_pnl=5.0,
                opened_at=Mock(),
                broker_id="test"
            )
        ]
        
        # Mock account info
        account_info = {
            'balance': 10000.0,
            'equity': 10050.0,
            'margin': 100.0,
            'free_margin': 9950.0
        }
        
        margin_check = risk_manager.check_margin_requirement(positions, account_info['free_margin'])
        
        assert isinstance(margin_check, tuple)
        assert len(margin_check) == 2
        assert isinstance(margin_check[0], bool)  # is_valid
        assert isinstance(margin_check[1], float)  # required_margin


class TestCorrelationTrackerComprehensive:
    """Comprehensive tests for CorrelationTracker functionality."""
    
    def test_correlation_matrix_initialization(self):
        """Test correlation matrix initialization."""
        instruments = ['EURUSD', 'GBPUSD', 'XAUUSD', 'SPX500']
        tracker = CorrelationTracker(instruments)
        
        assert tracker.instruments == instruments
        assert tracker.n_instruments == len(instruments)
        assert tracker.correlation_matrix.shape == (len(instruments), len(instruments))
        assert np.allclose(np.diag(tracker.correlation_matrix), 1.0)  # Diagonal should be 1.0
    
    def test_correlation_update(self):
        """Test correlation matrix updates."""
        instruments = ['EURUSD', 'GBPUSD', 'XAUUSD']
        tracker = CorrelationTracker(instruments, window=5)  # Small window for testing
        
        # Add returns data
        for i in range(10):
            returns = {
                'EURUSD': 0.01 + i * 0.001,
                'GBPUSD': 0.005 + i * 0.001,
                'XAUUSD': 0.02 + i * 0.001
            }
            tracker.update(returns)
        
        # Check that correlation matrix is updated
        assert tracker.correlation_matrix is not None
        assert tracker.correlation_matrix.shape == (3, 3)
    
    def test_correlation_regime_detection(self):
        """Test correlation regime change detection."""
        instruments = ['EURUSD', 'GBPUSD', 'XAUUSD']
        tracker = CorrelationTracker(instruments, window=5)
        
        # Add some returns
        for i in range(10):
            returns = {
                'EURUSD': 0.01 + i * 0.001,
                'GBPUSD': 0.005 + i * 0.001,
                'XAUUSD': 0.02 + i * 0.001
            }
            tracker.update(returns)
        
        regime_change = tracker.detect_regime_change()
        
        # Should return None or a string description
        assert regime_change is None or isinstance(regime_change, str)
    
    def test_get_current_correlations(self):
        """Test getting current correlations."""
        instruments = ['EURUSD', 'GBPUSD', 'XAUUSD']
        tracker = CorrelationTracker(instruments)
        
        correlations = tracker.get_current_correlations()
        
        assert correlations is not None
        assert correlations.shape == (len(instruments), len(instruments))
        assert np.allclose(np.diag(correlations), 1.0)


class TestHierarchicalTradingSystemComprehensive:
    """Comprehensive tests for HierarchicalTradingSystem functionality."""
    
    def test_system_initialization(self):
        """Test system initialization."""
        # Create components
        meta_controller = MetaController()
        specialists = {
            'forex': ForexSpecialist(),
            'commodities': CommoditiesSpecialist(),
            'equity': EquitySpecialist()
        }
        portfolio_risk_manager = PortfolioRiskManager()
        communication_hub = CommunicationHub()
        
        system = HierarchicalTradingSystem(
            meta_controller=meta_controller,
            specialists=specialists,
            portfolio_risk_manager=portfolio_risk_manager,
            communication_hub=communication_hub
        )
        
        assert system.meta_controller == meta_controller
        assert system.specialists == specialists
        assert system.portfolio_risk_manager == portfolio_risk_manager
        assert system.communication_hub == communication_hub
        assert system.system_id == "hierarchical_system"
    
    def test_system_step_with_positions(self):
        """Test system step with existing positions."""
        # Create components
        meta_controller = MetaController()
        specialists = {
            'forex': ForexSpecialist(),
            'commodities': CommoditiesSpecialist(),
            'equity': EquitySpecialist()
        }
        portfolio_risk_manager = PortfolioRiskManager()
        communication_hub = CommunicationHub()
        
        system = HierarchicalTradingSystem(
            meta_controller=meta_controller,
            specialists=specialists,
            portfolio_risk_manager=portfolio_risk_manager,
            communication_hub=communication_hub
        )
        
        # Mock market data
        market_data = {
            'EURUSD': {'close': 1.25, 'volume': 1000},
            'GBPUSD': {'close': 1.35, 'volume': 800},
            'USDJPY': {'close': 150.0, 'volume': 1200},
            'XAUUSD': {'close': 2045.0, 'volume': 500},
            'WTIUSD': {'close': 75.0, 'volume': 300},
            'SPX500': {'close': 4500.0, 'volume': 2000},
            'NAS100': {'close': 15000.0, 'volume': 1500},
            'US30': {'close': 35000.0, 'volume': 1000}
        }
        
        # Mock account info
        account_info = {
            'balance': 10000.0,
            'equity': 10050.0,
            'margin': 100.0,
            'free_margin': 9950.0
        }
        
        # Mock existing positions
        positions = [
            Position(
                position_id="1",
                agent_id="forex",
                symbol="EURUSD",
                side="long",
                quantity=0.1,
                entry_price=1.2,
                current_price=1.25,
                unrealized_pnl=5.0,
                opened_at=Mock(),
                broker_id="test"
            )
        ]
        
        # Execute step
        result = system.step(market_data, account_info, positions)
        
        # The step method returns a list of actions
        assert isinstance(result, list)
    
    def test_system_error_handling(self):
        """Test system error handling."""
        # Create components
        meta_controller = MetaController()
        specialists = {
            'forex': ForexSpecialist(),
            'commodities': CommoditiesSpecialist(),
            'equity': EquitySpecialist()
        }
        portfolio_risk_manager = PortfolioRiskManager()
        communication_hub = CommunicationHub()
        
        system = HierarchicalTradingSystem(
            meta_controller=meta_controller,
            specialists=specialists,
            portfolio_risk_manager=portfolio_risk_manager,
            communication_hub=communication_hub
        )
        
        # Test with invalid market data
        invalid_market_data = {}  # Empty market data
        
        account_info = {
            'balance': 10000.0,
            'equity': 10050.0,
            'margin': 100.0,
            'free_margin': 9950.0
        }
        
        positions = []
        
        # Should handle errors gracefully
        result = system.step(invalid_market_data, account_info, positions)
        
        # Should still return a result (even if empty)
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
