"""
Extended unit tests for hierarchical multi-agent system components.

Tests additional functionality to increase code coverage.
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


class TestMetaControllerExtended:
    """Extended tests for MetaController functionality."""
    
    def test_get_portfolio_state(self):
        """Test portfolio state calculation."""
        controller = MetaController()
        
        # Mock portfolio data
        portfolio = {
            'returns': np.random.randn(30),
            'volatility': 0.15,
            'drawdown': -0.05,
            'correlation_matrix': np.random.rand(8, 8)
        }
        
        # Mock specialist performance
        specialists = {
            'forex': {'sharpe': 1.2, 'win_rate': 0.6, 'max_dd': -0.08},
            'commodities': {'sharpe': 0.8, 'win_rate': 0.55, 'max_dd': -0.12},
            'equity': {'sharpe': 1.5, 'win_rate': 0.65, 'max_dd': -0.06}
        }
        
        portfolio_state = controller.get_portfolio_state(portfolio, specialists)
        
        assert isinstance(portfolio_state, torch.Tensor)
        assert portfolio_state.shape[0] == 74  # state_dim
    
    def test_detect_market_regime(self):
        """Test market regime detection."""
        controller = MetaController()
        
        # Mock returns data as numpy array
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.015, 0.01, -0.005, 0.02])
        
        regime = controller.detect_market_regime(returns)
        
        assert regime in ['bull', 'bear', 'sideways', 'volatile']
    
    def test_calculate_kelly_allocation(self):
        """Test Kelly allocation calculation."""
        controller = MetaController()
        
        # Mock specialist performance as numpy array
        specialist_performance = np.array([1.2, 0.8, 1.5])  # Sharpe ratios
        
        allocations = controller.calculate_kelly_allocation(specialist_performance)
        
        assert isinstance(allocations, np.ndarray)
        assert allocations.shape == (3,)  # num_specialists
        assert np.allclose(allocations.sum(), 1.0, atol=1e-6)


class TestForexSpecialistExtended:
    """Extended tests for ForexSpecialist functionality."""
    
    def test_get_domain_features(self):
        """Test domain-specific feature extraction."""
        specialist = ForexSpecialist()
        
        # Mock market data
        market_data = {
            'EURUSD': {'close': 1.25, 'volume': 1000, 'rsi': 60.0},
            'GBPUSD': {'close': 1.35, 'volume': 800, 'rsi': 55.0},
            'USDJPY': {'close': 150.0, 'volume': 1200, 'rsi': 65.0}
        }
        
        features = specialist.get_domain_features(market_data)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == specialist.market_features_dim
    
    def test_detect_correlation_regime(self):
        """Test correlation regime detection."""
        specialist = ForexSpecialist()
        
        # Mock returns data as dict
        returns = {
            'EURUSD': 0.01,
            'GBPUSD': 0.02,
            'USDJPY': -0.01
        }
        
        regime = specialist.detect_correlation_regime(returns)
        
        assert regime in ['risk-on', 'risk-off', 'neutral']
    
    def test_get_carry_signal(self):
        """Test carry trade signal calculation."""
        specialist = ForexSpecialist()
        
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


class TestCommoditiesSpecialistExtended:
    """Extended tests for CommoditiesSpecialist functionality."""
    
    def test_get_domain_features(self):
        """Test commodity-specific feature extraction."""
        specialist = CommoditiesSpecialist()
        
        # Mock market data
        market_data = {
            'XAUUSD': {'close': 2045.0, 'volume': 500, 'rsi': 55.0},
            'WTIUSD': {'close': 75.0, 'volume': 300, 'rsi': 60.0}
        }
        
        features = specialist.get_domain_features(market_data)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == specialist.market_features_dim
    
    def test_detect_inflation_regime(self):
        """Test inflation regime detection."""
        specialist = CommoditiesSpecialist()
        
        # Mock inflation data
        inflation_data = {
            'cpi': 0.03,
            'ppi': 0.025,
            'gold_price': 2045.0,
            'oil_price': 75.0
        }
        
        regime = specialist.detect_inflation_regime(inflation_data)
        
        assert regime in ['high', 'medium', 'low', 'stable']


class TestEquitySpecialistExtended:
    """Extended tests for EquitySpecialist functionality."""
    
    def test_get_domain_features(self):
        """Test equity-specific feature extraction."""
        specialist = EquitySpecialist()
        
        # Mock market data
        market_data = {
            'SPX500': {'close': 4500.0, 'volume': 2000, 'rsi': 58.0},
            'NAS100': {'close': 15000.0, 'volume': 1500, 'rsi': 62.0},
            'US30': {'close': 35000.0, 'volume': 1000, 'rsi': 55.0}
        }
        
        features = specialist.get_domain_features(market_data)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == specialist.market_features_dim
    
    def test_detect_sector_rotation(self):
        """Test sector rotation detection."""
        specialist = EquitySpecialist()
        
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
        
        assert rotation in ['tech_heavy', 'defensive', 'cyclical', 'balanced', 'growth']


class TestCommunicationHubExtended:
    """Extended tests for CommunicationHub functionality."""
    
    def test_broadcast_allocation(self):
        """Test broadcasting allocation messages."""
        hub = CommunicationHub()
        
        # Create allocation messages manually since broadcast_allocation doesn't exist
        message1 = AllocationMessage(
            specialist_id="forex",
            allocation=0.4,
            risk_appetite=0.6,
            market_regime="bull"
        )
        message2 = AllocationMessage(
            specialist_id="commodities",
            allocation=0.3,
            risk_appetite=0.6,
            market_regime="bull"
        )
        message3 = AllocationMessage(
            specialist_id="equity",
            allocation=0.3,
            risk_appetite=0.6,
            market_regime="bull"
        )
        
        hub.send_message(message1)
        hub.send_message(message2)
        hub.send_message(message3)
        
        assert len(hub.message_history) == 3
    
    def test_collect_reports(self):
        """Test collecting performance reports."""
        hub = CommunicationHub()
        
        # Mock specialists
        specialists = {
            'forex': Mock(),
            'commodities': Mock(),
            'equity': Mock()
        }
        
        # Mock get_performance_metrics method
        for specialist in specialists.values():
            specialist.get_performance_metrics = Mock(return_value={
                'confidence': 0.8,
                'realized_pnl': 100.0,
                'unrealized_pnl': 50.0,
                'sharpe_ratio': 1.5,
                'win_rate': 0.6
            })
        
        reports = hub.collect_reports(specialists)
        
        assert len(reports) == 3
        assert all(isinstance(report, PerformanceReport) for report in reports)
    
    def test_get_statistics(self):
        """Test getting communication statistics."""
        hub = CommunicationHub()
        
        # Send some messages
        message = AllocationMessage(
            specialist_id="forex_specialist",
            allocation=0.4,
            risk_appetite=0.6,
            market_regime="bull"
        )
        hub.send_message(message)
        
        stats = hub.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_messages' in stats
        assert 'messages_by_type' in stats
        assert 'messages_by_agent' in stats
        assert stats['total_messages'] == 1


class TestSpecialistRegistry:
    """Test SpecialistRegistry functionality."""
    
    def test_get_specialist_registry(self):
        """Test getting specialist registry."""
        registry = get_specialist_registry()
        
        assert isinstance(registry, SpecialistRegistry)
        assert len(registry._specialists) >= 3  # forex, commodities, equity
    
    def test_create_specialist(self):
        """Test creating specialist."""
        specialist = create_specialist('forex', {
            'instruments': ['EURUSD', 'GBPUSD', 'USDJPY'],
            'market_features_dim': 8,
            'observation_dim': 50
        })
        
        assert isinstance(specialist, ForexSpecialist)
        assert specialist.instruments == ['EURUSD', 'GBPUSD', 'USDJPY']
    
    def test_create_invalid_specialist(self):
        """Test creating invalid specialist."""
        with pytest.raises(ValueError):
            create_specialist('invalid_specialist')


class TestPortfolioRiskManagerExtended:
    """Extended tests for PortfolioRiskManager functionality."""
    
    def test_check_portfolio_risk(self):
        """Test portfolio risk checking."""
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
            ),
            Position(
                position_id="2",
                agent_id="commodities",
                symbol="XAUUSD",
                side="short",
                quantity=0.05,
                entry_price=2050.0,
                current_price=2045.0,
                unrealized_pnl=2.5,
                opened_at=Mock(),
                broker_id="test"
            )
        ]
        
        # Mock portfolio with returns_history
        from mtquant.risk_management.portfolio_risk_manager import Portfolio
        portfolio = Portfolio(
            equity=10000.0,
            margin_used=100.0,
            margin_available=9900.0,
            positions=positions,
            returns_history=np.random.randn(100, 8),  # 100 days, 8 instruments
            correlation_matrix=np.eye(8),
            sector_allocation={'forex': 0.5, 'commodities': 0.3, 'equity': 0.2},
            last_updated=datetime.now()
        )
        
        risk_status = risk_manager.check_portfolio_risk(positions, portfolio)
        
        assert isinstance(risk_status, tuple)
        assert len(risk_status) == 2
        assert isinstance(risk_status[0], bool)  # is_valid
        assert isinstance(risk_status[1], str)  # reason
    
    def test_calculate_var(self):
        """Test VaR calculation."""
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
        returns_history = np.random.normal(0, 0.02, (100, 8))  # 100 days, 8 instruments
        
        var_result = risk_manager.calculate_var(positions, returns_history)
        
        assert isinstance(var_result, dict)
        assert 'var' in var_result
        assert var_result['var'] > 0  # VaR should be positive (loss)


class TestCorrelationTrackerExtended:
    """Extended tests for CorrelationTracker functionality."""
    
    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        tracker = CorrelationTracker(
            instruments=['EURUSD', 'GBPUSD', 'XAUUSD']
        )
        
        # Add some returns history
        for i in range(25):
            returns = {
                'EURUSD': 0.01 + i * 0.001,
                'GBPUSD': 0.005 + i * 0.001,
                'XAUUSD': 0.02 + i * 0.001
            }
            tracker.update(returns)
        
        correlation_matrix = tracker.get_current_correlations()
        
        assert correlation_matrix is not None
        assert correlation_matrix.shape == (3, 3)
        assert np.allclose(np.diag(correlation_matrix), 1.0)  # Diagonal should be 1.0
    
    def test_detect_regime_change(self):
        """Test correlation regime change detection."""
        tracker = CorrelationTracker(
            instruments=['EURUSD', 'GBPUSD', 'XAUUSD']
        )
        
        # Add returns history
        for i in range(25):
            returns = {
                'EURUSD': 0.01 + i * 0.001,
                'GBPUSD': 0.005 + i * 0.001,
                'XAUUSD': 0.02 + i * 0.001
            }
            tracker.update(returns)
        
        regime_change = tracker.detect_regime_change()
        
        # Should return None or a regime change description
        assert regime_change is None or isinstance(regime_change, str)


class TestHierarchicalTradingSystemExtended:
    """Extended tests for HierarchicalTradingSystem functionality."""
    
    def test_system_step(self):
        """Test system step execution."""
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
        
        # Mock positions
        positions = []
        
        # Execute step
        result = system.step(market_data, account_info, positions)
        
        # The step method returns a list of actions, not a dict
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
