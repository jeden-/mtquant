"""
Simplified unit tests for hierarchical multi-agent system components.

Tests basic functionality without complex initialization.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

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
    AlertMessage
)

from mtquant.risk_management import PortfolioRiskManager, CorrelationTracker
from mtquant.mcp_integration.models.position import Position
from mtquant.mcp_integration.models.order import Order


class TestMetaController:
    """Test MetaController functionality."""
    
    def test_meta_controller_initialization(self):
        """Test MetaController initialization."""
        controller = MetaController()
        
        assert controller.state_dim == 74
        assert controller.hidden_dim == 256
        assert controller.hidden_dim_2 == 128
        assert controller.input_layer is not None
        assert controller.hidden_layer is not None
        assert controller.allocation_head is not None
        assert controller.risk_appetite_head is not None
        assert controller.value_head is not None
    
    def test_meta_controller_forward_pass(self):
        """Test MetaController forward pass."""
        controller = MetaController()
        
        # Create sample portfolio state
        batch_size = 2
        portfolio_state = torch.randn(batch_size, 74)
        
        # Forward pass
        allocations, risk_appetite, value = controller(portfolio_state)
        
        # Check output shapes
        assert allocations.shape == (batch_size, 3)
        assert risk_appetite.shape == (batch_size, 1)
        assert value.shape == (batch_size, 1)
        
        # Check allocations sum to 1 (softmax)
        assert torch.allclose(allocations.sum(dim=1), torch.ones(batch_size), atol=1e-6)
        
        # Check risk appetite is between 0 and 1 (sigmoid)
        assert torch.all(risk_appetite >= 0)
        assert torch.all(risk_appetite <= 1)


class TestForexSpecialist:
    """Test ForexSpecialist functionality."""
    
    def test_forex_specialist_initialization(self):
        """Test ForexSpecialist initialization."""
        specialist = ForexSpecialist()
        
        assert specialist.instruments == ['EURUSD', 'GBPUSD', 'USDJPY']
        assert specialist.domain_encoder is not None
        assert specialist.instrument_heads is not None
        assert specialist.value_head is not None
    
    def test_forex_specialist_forward_pass(self):
        """Test ForexSpecialist forward pass."""
        specialist = ForexSpecialist()
        
        # Create sample domain features and instrument states
        batch_size = 2
        market_state = torch.randn(batch_size, specialist.market_features_dim)
        instrument_states = {
            'EURUSD': torch.randn(batch_size, specialist.observation_dim),
            'GBPUSD': torch.randn(batch_size, specialist.observation_dim),
            'USDJPY': torch.randn(batch_size, specialist.observation_dim)
        }
        allocation = 0.4
        
        # Forward pass
        actions, value = specialist(market_state, instrument_states, allocation)
        
        # Check output shapes
        assert isinstance(actions, dict)
        assert len(actions) == 3  # 3 instruments
        assert value.shape == (batch_size, 1)


class TestCommoditiesSpecialist:
    """Test CommoditiesSpecialist functionality."""
    
    def test_commodities_specialist_initialization(self):
        """Test CommoditiesSpecialist initialization."""
        specialist = CommoditiesSpecialist()
        
        assert specialist.instruments == ['XAUUSD', 'WTIUSD']
        assert specialist.domain_encoder is not None
        assert specialist.instrument_heads is not None
        assert specialist.value_head is not None


class TestEquitySpecialist:
    """Test EquitySpecialist functionality."""
    
    def test_equity_specialist_initialization(self):
        """Test EquitySpecialist initialization."""
        specialist = EquitySpecialist()
        
        assert specialist.instruments == ['SPX500', 'NAS100', 'US30']
        assert specialist.domain_encoder is not None
        assert specialist.instrument_heads is not None
        assert specialist.value_head is not None


class TestCommunicationHub:
    """Test CommunicationHub functionality."""
    
    def test_communication_hub_initialization(self):
        """Test CommunicationHub initialization."""
        hub = CommunicationHub()
        
        assert len(hub.message_history) == 0
        assert len(hub.registered_agents) == 0
        assert hub.message_stats['total_messages'] == 0
    
    def test_send_message(self):
        """Test message sending."""
        hub = CommunicationHub()
        
        # Create test message
        message = AllocationMessage(
            specialist_id="forex_specialist",
            allocation=0.4,
            risk_appetite=0.6,
            market_regime="bull",
            timestamp=Mock(),
            metadata={"risk_level": "medium"}
        )
        
        hub.send_message(message)
        
        assert len(hub.message_history) == 1
        # Check that message was stored (it's wrapped in a dict with id and timestamp)
        stored_message = hub.message_history[0]
        assert stored_message['message'] == message
    
    def test_get_messages_for_agent(self):
        """Test message retrieval for specific agent."""
        hub = CommunicationHub()
        
        # Create test messages
        message1 = AllocationMessage(
            specialist_id="forex_specialist",
            allocation=0.4,
            risk_appetite=0.6,
            market_regime="bull",
            timestamp=Mock(),
            metadata={}
        )
        
        message2 = AllocationMessage(
            specialist_id="commodities_specialist",
            allocation=0.3,
            risk_appetite=0.5,
            market_regime="bull",
            timestamp=Mock(),
            metadata={}
        )
        
        hub.send_message(message1)
        hub.send_message(message2)
        
        # Get messages for forex_specialist
        forex_messages = hub.get_messages_by_agent("forex_specialist")
        assert len(forex_messages) == 1
        assert forex_messages[0]['message'] == message1
        
        # Get messages for commodities_specialist
        commodities_messages = hub.get_messages_by_agent("commodities_specialist")
        assert len(commodities_messages) == 1
        assert commodities_messages[0]['message'] == message2


class TestHierarchicalTradingSystem:
    """Test HierarchicalTradingSystem functionality."""
    
    def test_hierarchical_system_initialization(self):
        """Test HierarchicalTradingSystem initialization."""
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
        
        assert system.meta_controller is not None
        assert len(system.specialists) == 3
        assert 'forex' in system.specialists
        assert 'commodities' in system.specialists
        assert 'equity' in system.specialists
        assert system.communication_hub is not None
        assert system.portfolio_risk_manager is not None
        assert system.system_id == "hierarchical_system"


class TestPortfolioRiskManager:
    """Test PortfolioRiskManager functionality."""
    
    def test_portfolio_risk_manager_initialization(self):
        """Test PortfolioRiskManager initialization."""
        risk_manager = PortfolioRiskManager()
        
        assert risk_manager.config is not None
        assert len(risk_manager.instruments) == 8
        assert risk_manager.correlation_tracker is not None


class TestCorrelationTracker:
    """Test CorrelationTracker functionality."""
    
    def test_correlation_tracker_initialization(self):
        """Test CorrelationTracker initialization."""
        tracker = CorrelationTracker(
            instruments=['EURUSD', 'GBPUSD', 'XAUUSD']
        )
        
        assert tracker.instruments == ['EURUSD', 'GBPUSD', 'XAUUSD']
        assert tracker.n_instruments == 3
        assert tracker.window == 100
        assert tracker.correlation_matrix is not None  # Initialized as identity matrix
    
    def test_update_prices(self):
        """Test price update."""
        tracker = CorrelationTracker(
            instruments=['EURUSD', 'GBPUSD', 'XAUUSD']
        )
        
        # Update returns
        returns = {'EURUSD': 0.01, 'GBPUSD': 0.005, 'XAUUSD': 0.02}
        tracker.update(returns)
        
        assert len(tracker.returns_history) == 1
        # Correlation matrix should still be identity matrix since we only have 1 observation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
