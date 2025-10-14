"""
Unit tests for hierarchical multi-agent system components.

Tests:
- MetaController
- BaseSpecialist and implementations
- Communication system
- HierarchicalTradingSystem
- PortfolioRiskManager
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
        controller = MetaController(
            state_dim=74,
            hidden_dim=256,
            hidden_dim_2=128,
            dropout=0.2,
            device="cpu"
        )
        
        assert controller.state_dim == 74
        assert controller.hidden_dim == 256
        assert controller.hidden_dim_2 == 128
        assert controller.dropout == 0.2
        assert controller.device == "cpu"
        assert controller.input_layer is not None
        assert controller.hidden_layer is not None
        assert controller.allocation_head is not None
        assert controller.risk_head is not None
        assert controller.value_head is not None
    
    def test_meta_controller_forward_pass(self):
        """Test MetaController forward pass."""
        controller = MetaController(
            state_dim=74,
            hidden_dim=256,
            hidden_dim_2=128,
            dropout=0.2,
            device="cpu"
        )
        
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
    
    def test_get_portfolio_state(self):
        """Test portfolio state calculation."""
        controller = MetaController(
            portfolio_dim=50,
            hidden_dim=128,
            num_specialists=3
        )
        
        # Mock positions
        positions = [
            Position(
                position_id="1",
                agent_id="forex",
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
                side="short",
                quantity=0.05,
                entry_price=2050.0,
                current_price=2045.0,
                unrealized_pnl=2.5,
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
        
        # Mock market data
        market_data = {
            'EURUSD': {'close': 1.25, 'volume': 1000},
            'XAUUSD': {'close': 2045.0, 'volume': 500}
        }
        
        portfolio_state = controller.get_portfolio_state(
            positions=positions,
            account_info=account_info,
            market_data=market_data
        )
        
        assert isinstance(portfolio_state, torch.Tensor)
        assert portfolio_state.shape[0] == 50  # portfolio_dim
    
    def test_detect_market_regime(self):
        """Test market regime detection."""
        controller = MetaController(
            portfolio_dim=50,
            hidden_dim=128,
            num_specialists=3
        )
        
        # Mock market data
        market_data = {
            'EURUSD': {'close': 1.25, 'volume': 1000},
            'XAUUSD': {'close': 2045.0, 'volume': 500},
            'SPX500': {'close': 4500.0, 'volume': 2000}
        }
        
        regime = controller.detect_market_regime(market_data)
        
        assert regime in ['bull', 'bear', 'sideways', 'volatile']
    
    def test_calculate_kelly_allocation(self):
        """Test Kelly allocation calculation."""
        controller = MetaController(
            portfolio_dim=50,
            hidden_dim=128,
            num_specialists=3
        )
        
        # Mock specialist performance
        specialist_performance = {
            'forex': {'win_rate': 0.6, 'avg_return': 0.02, 'volatility': 0.15},
            'commodities': {'win_rate': 0.55, 'avg_return': 0.025, 'volatility': 0.18},
            'equity': {'win_rate': 0.65, 'avg_return': 0.03, 'volatility': 0.12}
        }
        
        allocations = controller.calculate_kelly_allocation(specialist_performance)
        
        assert isinstance(allocations, torch.Tensor)
        assert allocations.shape == (3,)  # num_specialists
        assert torch.allclose(allocations.sum(), torch.tensor(1.0), atol=1e-6)


class TestForexSpecialist:
    """Test ForexSpecialist functionality."""
    
    def test_forex_specialist_initialization(self):
        """Test ForexSpecialist initialization."""
        specialist = ForexSpecialist(
            domain_dim=30,
            hidden_dim=64,
            instruments=['EURUSD', 'GBPUSD', 'USDJPY']
        )
        
        assert specialist.domain_dim == 30
        assert specialist.hidden_dim == 64
        assert specialist.instruments == ['EURUSD', 'GBPUSD', 'USDJPY']
        assert specialist.domain_encoder is not None
        assert specialist.instrument_heads is not None
        assert specialist.value_head is not None
    
    def test_forex_specialist_forward_pass(self):
        """Test ForexSpecialist forward pass."""
        specialist = ForexSpecialist(
            domain_dim=30,
            hidden_dim=64,
            instruments=['EURUSD', 'GBPUSD', 'USDJPY']
        )
        
        # Create sample domain features
        batch_size = 2
        domain_features = torch.randn(batch_size, 30)
        
        # Forward pass
        actions, confidence, value = specialist(domain_features)
        
        # Check output shapes
        assert actions.shape == (batch_size, 3)  # 3 instruments
        assert confidence.shape == (batch_size, 1)
        assert value.shape == (batch_size, 1)
        
        # Check confidence is between 0 and 1
        assert torch.all(confidence >= 0)
        assert torch.all(confidence <= 1)
    
    def test_get_domain_features(self):
        """Test domain-specific feature extraction."""
        specialist = ForexSpecialist(
            domain_dim=30,
            hidden_dim=64,
            instruments=['EURUSD', 'GBPUSD', 'USDJPY']
        )
        
        # Mock market data
        market_data = {
            'EURUSD': {'close': 1.25, 'volume': 1000, 'rsi': 60.0},
            'GBPUSD': {'close': 1.35, 'volume': 800, 'rsi': 55.0},
            'USDJPY': {'close': 150.0, 'volume': 1200, 'rsi': 65.0}
        }
        
        features = specialist.get_domain_features(market_data)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == 30  # domain_dim
    
    def test_detect_correlation_regime(self):
        """Test correlation regime detection."""
        specialist = ForexSpecialist(
            domain_dim=30,
            hidden_dim=64,
            instruments=['EURUSD', 'GBPUSD', 'USDJPY']
        )
        
        # Mock correlation matrix
        correlation_matrix = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
        
        regime = specialist.detect_correlation_regime(correlation_matrix)
        
        assert regime in ['high', 'medium', 'low']
    
    def test_get_carry_signal(self):
        """Test carry trade signal calculation."""
        specialist = ForexSpecialist(
            domain_dim=30,
            hidden_dim=64,
            instruments=['EURUSD', 'GBPUSD', 'USDJPY']
        )
        
        # Mock interest rates
        interest_rates = {
            'EUR': 0.02,
            'GBP': 0.03,
            'USD': 0.05,
            'JPY': 0.01
        }
        
        signal = specialist.get_carry_signal(interest_rates)
        
        assert isinstance(signal, float)
        assert -1 <= signal <= 1


class TestCommoditiesSpecialist:
    """Test CommoditiesSpecialist functionality."""
    
    def test_commodities_specialist_initialization(self):
        """Test CommoditiesSpecialist initialization."""
        specialist = CommoditiesSpecialist(
            domain_dim=25,
            hidden_dim=64,
            instruments=['XAUUSD', 'WTIUSD']
        )
        
        assert specialist.domain_dim == 25
        assert specialist.hidden_dim == 64
        assert specialist.instruments == ['XAUUSD', 'WTIUSD']
        assert specialist.domain_encoder is not None
        assert specialist.instrument_heads is not None
        assert specialist.value_head is not None
    
    def test_get_domain_features(self):
        """Test commodity-specific feature extraction."""
        specialist = CommoditiesSpecialist(
            domain_dim=25,
            hidden_dim=64,
            instruments=['XAUUSD', 'WTIUSD']
        )
        
        # Mock market data
        market_data = {
            'XAUUSD': {'close': 2045.0, 'volume': 500, 'rsi': 55.0},
            'WTIUSD': {'close': 75.0, 'volume': 300, 'rsi': 60.0}
        }
        
        features = specialist.get_domain_features(market_data)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == 25  # domain_dim
    
    def test_detect_inflation_regime(self):
        """Test inflation regime detection."""
        specialist = CommoditiesSpecialist(
            domain_dim=25,
            hidden_dim=64,
            instruments=['XAUUSD', 'WTIUSD']
        )
        
        # Mock inflation data
        inflation_data = {
            'cpi': 0.03,
            'ppi': 0.025,
            'gold_price': 2045.0,
            'oil_price': 75.0
        }
        
        regime = specialist.detect_inflation_regime(inflation_data)
        
        assert regime in ['high', 'medium', 'low']


class TestEquitySpecialist:
    """Test EquitySpecialist functionality."""
    
    def test_equity_specialist_initialization(self):
        """Test EquitySpecialist initialization."""
        specialist = EquitySpecialist(
            domain_dim=35,
            hidden_dim=64,
            instruments=['SPX500', 'NAS100', 'US30']
        )
        
        assert specialist.domain_dim == 35
        assert specialist.hidden_dim == 64
        assert specialist.instruments == ['SPX500', 'NAS100', 'US30']
        assert specialist.domain_encoder is not None
        assert specialist.instrument_heads is not None
        assert specialist.value_head is not None
    
    def test_get_domain_features(self):
        """Test equity-specific feature extraction."""
        specialist = EquitySpecialist(
            domain_dim=35,
            hidden_dim=64,
            instruments=['SPX500', 'NAS100', 'US30']
        )
        
        # Mock market data
        market_data = {
            'SPX500': {'close': 4500.0, 'volume': 2000, 'rsi': 58.0},
            'NAS100': {'close': 15000.0, 'volume': 1500, 'rsi': 62.0},
            'US30': {'close': 35000.0, 'volume': 1000, 'rsi': 55.0}
        }
        
        features = specialist.get_domain_features(market_data)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == 35  # domain_dim
    
    def test_detect_sector_rotation(self):
        """Test sector rotation detection."""
        specialist = EquitySpecialist(
            domain_dim=35,
            hidden_dim=64,
            instruments=['SPX500', 'NAS100', 'US30']
        )
        
        # Mock sector data
        sector_data = {
            'technology': 0.25,
            'healthcare': 0.20,
            'financials': 0.15,
            'energy': 0.10,
            'utilities': 0.08,
            'materials': 0.07,
            'industrials': 0.10,
            'consumer_discretionary': 0.05
        }
        
        rotation = specialist.detect_sector_rotation(sector_data)
        
        assert rotation in ['tech_heavy', 'defensive', 'cyclical', 'balanced']


class TestCommunicationHub:
    """Test CommunicationHub functionality."""
    
    def test_communication_hub_initialization(self):
        """Test CommunicationHub initialization."""
        hub = CommunicationHub()
        
        assert hub.message_history == []
        assert hub.active_alerts == []
        assert hub.routing_table == {}
    
    def test_send_message(self):
        """Test message sending."""
        hub = CommunicationHub()
        
        # Create test message
        message = AllocationMessage(
            sender_id="meta_controller",
            recipient_id="forex_specialist",
            allocations={"forex": 0.4, "commodities": 0.3, "equity": 0.3},
            timestamp=Mock(),
            metadata={"risk_level": "medium"}
        )
        
        hub.send_message(message)
        
        assert len(hub.message_history) == 1
        assert hub.message_history[0] == message
    
    def test_get_messages_for_agent(self):
        """Test message retrieval for specific agent."""
        hub = CommunicationHub()
        
        # Create test messages
        message1 = AllocationMessage(
            sender_id="meta_controller",
            recipient_id="forex_specialist",
            allocations={"forex": 0.4},
            timestamp=Mock(),
            metadata={}
        )
        
        message2 = AllocationMessage(
            sender_id="meta_controller",
            recipient_id="commodities_specialist",
            allocations={"commodities": 0.3},
            timestamp=Mock(),
            metadata={}
        )
        
        hub.send_message(message1)
        hub.send_message(message2)
        
        # Get messages for forex_specialist
        forex_messages = hub.get_messages_for_agent("forex_specialist")
        assert len(forex_messages) == 1
        assert forex_messages[0] == message1
        
        # Get messages for commodities_specialist
        commodities_messages = hub.get_messages_for_agent("commodities_specialist")
        assert len(commodities_messages) == 1
        assert commodities_messages[0] == message2
    
    def test_create_alert(self):
        """Test alert creation."""
        hub = CommunicationHub()
        
        alert = hub.create_alert(
            alert_type="risk_violation",
            severity="high",
            message="Position size exceeds limits",
            agent_id="forex_specialist",
            metadata={"position_size": 0.15, "limit": 0.10}
        )
        
        assert alert.alert_type == "risk_violation"
        assert alert.severity == "high"
        assert alert.message == "Position size exceeds limits"
        assert alert.agent_id == "forex_specialist"
        assert len(hub.active_alerts) == 1


class TestHierarchicalTradingSystem:
    """Test HierarchicalTradingSystem functionality."""
    
    def test_hierarchical_system_initialization(self):
        """Test HierarchicalTradingSystem initialization."""
        system = HierarchicalTradingSystem(
            meta_controller_config={
                'portfolio_dim': 50,
                'hidden_dim': 128,
                'num_specialists': 3
            },
            specialist_configs={
                'forex': {
                    'domain_dim': 30,
                    'hidden_dim': 64,
                    'instruments': ['EURUSD', 'GBPUSD', 'USDJPY']
                },
                'commodities': {
                    'domain_dim': 25,
                    'hidden_dim': 64,
                    'instruments': ['XAUUSD', 'WTIUSD']
                },
                'equity': {
                    'domain_dim': 35,
                    'hidden_dim': 64,
                    'instruments': ['SPX500', 'NAS100', 'US30']
                }
            }
        )
        
        assert system.meta_controller is not None
        assert len(system.specialists) == 3
        assert 'forex' in system.specialists
        assert 'commodities' in system.specialists
        assert 'equity' in system.specialists
        assert system.communication_hub is not None
        assert system.risk_manager is not None
        assert system.state == SystemState.INITIALIZED
    
    def test_system_step(self):
        """Test system step execution."""
        system = HierarchicalTradingSystem(
            meta_controller_config={
                'portfolio_dim': 50,
                'hidden_dim': 128,
                'num_specialists': 3
            },
            specialist_configs={
                'forex': {
                    'domain_dim': 30,
                    'hidden_dim': 64,
                    'instruments': ['EURUSD', 'GBPUSD', 'USDJPY']
                },
                'commodities': {
                    'domain_dim': 25,
                    'hidden_dim': 64,
                    'instruments': ['XAUUSD', 'WTIUSD']
                },
                'equity': {
                    'domain_dim': 35,
                    'hidden_dim': 64,
                    'instruments': ['SPX500', 'NAS100', 'US30']
                }
            }
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
        
        # Mock positions
        positions = []
        
        # Execute step
        result = system.step(market_data, account_info, positions)
        
        assert isinstance(result, dict)
        assert 'allocations' in result
        assert 'actions' in result
        assert 'risk_status' in result
        assert 'alerts' in result


class TestPortfolioRiskManager:
    """Test PortfolioRiskManager functionality."""
    
    def test_portfolio_risk_manager_initialization(self):
        """Test PortfolioRiskManager initialization."""
        risk_manager = PortfolioRiskManager(
            max_portfolio_var=0.05,
            max_correlation=0.7,
            max_sector_allocation=0.3
        )
        
        assert risk_manager.max_portfolio_var == 0.05
        assert risk_manager.max_correlation == 0.7
        assert risk_manager.max_sector_allocation == 0.3
        assert risk_manager.correlation_tracker is not None
    
    def test_check_portfolio_risk(self):
        """Test portfolio risk checking."""
        risk_manager = PortfolioRiskManager(
            max_portfolio_var=0.05,
            max_correlation=0.7,
            max_sector_allocation=0.3
        )
        
        # Mock positions
        positions = [
            Position(
                position_id="1",
                agent_id="forex",
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
                side="short",
                quantity=0.05,
                entry_price=2050.0,
                current_price=2045.0,
                unrealized_pnl=2.5,
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
        
        risk_status = risk_manager.check_portfolio_risk(positions, account_info)
        
        assert isinstance(risk_status, dict)
        assert 'is_safe' in risk_status
        assert 'var' in risk_status
        assert 'correlation_risk' in risk_status
        assert 'sector_allocation' in risk_status
    
    def test_calculate_var(self):
        """Test VaR calculation."""
        risk_manager = PortfolioRiskManager(
            max_portfolio_var=0.05,
            max_correlation=0.7,
            max_sector_allocation=0.3
        )
        
        # Mock returns data
        returns = np.random.normal(0, 0.02, 100)
        
        var = risk_manager.calculate_var(returns, confidence_level=0.95)
        
        assert isinstance(var, float)
        assert var > 0  # VaR should be positive (loss)


class TestCorrelationTracker:
    """Test CorrelationTracker functionality."""
    
    def test_correlation_tracker_initialization(self):
        """Test CorrelationTracker initialization."""
        tracker = CorrelationTracker(
            instruments=['EURUSD', 'GBPUSD', 'XAUUSD'],
            window_size=20
        )
        
        assert tracker.instruments == ['EURUSD', 'GBPUSD', 'XAUUSD']
        assert tracker.window_size == 20
        assert len(tracker.price_history) == 3
        assert tracker.correlation_matrix is None
    
    def test_update_prices(self):
        """Test price update."""
        tracker = CorrelationTracker(
            instruments=['EURUSD', 'GBPUSD', 'XAUUSD'],
            window_size=20
        )
        
        # Update prices
        prices = {'EURUSD': 1.25, 'GBPUSD': 1.35, 'XAUUSD': 2045.0}
        tracker.update_prices(prices)
        
        assert len(tracker.price_history['EURUSD']) == 1
        assert len(tracker.price_history['GBPUSD']) == 1
        assert len(tracker.price_history['XAUUSD']) == 1
    
    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        tracker = CorrelationTracker(
            instruments=['EURUSD', 'GBPUSD', 'XAUUSD'],
            window_size=20
        )
        
        # Add some price history
        for i in range(25):
            prices = {
                'EURUSD': 1.25 + i * 0.001,
                'GBPUSD': 1.35 + i * 0.001,
                'XAUUSD': 2045.0 + i * 0.1
            }
            tracker.update_prices(prices)
        
        correlation_matrix = tracker.calculate_correlation_matrix()
        
        assert correlation_matrix is not None
        assert correlation_matrix.shape == (3, 3)
        assert np.allclose(np.diag(correlation_matrix), 1.0)  # Diagonal should be 1.0
    
    def test_detect_correlation_regime(self):
        """Test correlation regime detection."""
        tracker = CorrelationTracker(
            instruments=['EURUSD', 'GBPUSD', 'XAUUSD'],
            window_size=20
        )
        
        # Add price history
        for i in range(25):
            prices = {
                'EURUSD': 1.25 + i * 0.001,
                'GBPUSD': 1.35 + i * 0.001,
                'XAUUSD': 2045.0 + i * 0.1
            }
            tracker.update_prices(prices)
        
        regime = tracker.detect_correlation_regime()
        
        assert regime in ['high', 'medium', 'low']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
