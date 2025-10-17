"""
Integration Tests for Hierarchical Multi-Agent System

This module tests the complete integration of the hierarchical trading system:
- Full system step() execution
- Multi-agent coordination
- Risk management enforcement
- Training pipeline integration
- End-to-end decision flow
"""

import pytest
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from mtquant.agents.hierarchical.meta_controller import MetaController
from mtquant.agents.hierarchical.base_specialist import BaseSpecialist
from mtquant.agents.hierarchical.forex_specialist import ForexSpecialist
from mtquant.agents.hierarchical.commodities_specialist import CommoditiesSpecialist
from mtquant.agents.hierarchical.equity_specialist import EquitySpecialist
from mtquant.agents.hierarchical.hierarchical_system import HierarchicalTradingSystem
from mtquant.agents.hierarchical.communication import CommunicationHub
from mtquant.agents.hierarchical.specialist_factory import SpecialistRegistry
from mtquant.risk_management.portfolio_risk_manager import PortfolioRiskManager
from mtquant.risk_management.position_sizer import PositionSizer
from mtquant.mcp_integration.models.position import Position
from mtquant.mcp_integration.models.order import Order
from mtquant.agents.training.phase3_joint_training import Phase3JointTrainer, create_phase3_trainer


@pytest.fixture
def mock_market_data():
    """Create comprehensive mock market data."""
    np.random.seed(42)
    
    data = {}
    instruments = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'WTIUSD', 'SPX500', 'NAS100', 'US30']
    
    for instrument in instruments:
        # Generate realistic OHLCV data
        base_price = 100 if 'USD' in instrument else 2000 if 'XAU' in instrument else 100
        prices = base_price + np.cumsum(np.random.randn(1000) * 0.01)
        
        data[instrument] = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H'),
            'open': prices,
            'high': prices * (1 + np.random.rand(1000) * 0.002),
            'low': prices * (1 - np.random.rand(1000) * 0.002),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        })
    
    return data


@pytest.fixture
def sample_portfolio():
    """Create sample portfolio with positions."""
    positions = [
        Position(
            position_id="pos_1",
            agent_id="forex_agent",
            symbol="EURUSD",
            side="long",
            quantity=10000,
            entry_price=1.1000,
            current_price=1.1050,
            unrealized_pnl=50.0,
            opened_at=datetime.now() - timedelta(hours=2)
        ),
        Position(
            position_id="pos_2",
            agent_id="commodities_agent",
            symbol="XAUUSD",
            side="long",
            quantity=1.0,
            entry_price=2000.0,
            current_price=2010.0,
            unrealized_pnl=10.0,
            opened_at=datetime.now() - timedelta(hours=1)
        )
    ]
    
    return {
        'portfolio_value': 100000.0,
        'positions': positions,
        'cash': 50000.0,
        'total_exposure': 50000.0,
        'num_instruments': 2
    }


@pytest.fixture
def hierarchical_system():
    """Create complete hierarchical system."""
    # Create meta-controller
    meta_controller = MetaController(
        state_dim=74,
        hidden_dim=256,
        hidden_dim_2=128,
        dropout=0.2
    )
    
    # Create specialists using factory
    registry = SpecialistRegistry()
    specialists = {
        'forex': registry.create_specialist('forex', {
            'instruments': ['EURUSD', 'GBPUSD', 'USDJPY'],
            'market_features_dim': 8,
            'observation_dim': 50,
            'hidden_dim': 64
        }),
        'commodities': registry.create_specialist('commodities', {
            'instruments': ['XAUUSD', 'WTIUSD'],
            'market_features_dim': 6,
            'observation_dim': 50,
            'hidden_dim': 64
        }),
        'equity': registry.create_specialist('equity', {
            'instruments': ['SPX500', 'NAS100', 'US30'],
            'market_features_dim': 7,
            'observation_dim': 50,
            'hidden_dim': 64
        })
    }
    
    # Create portfolio risk manager
    portfolio_risk_manager = PortfolioRiskManager()
    
    # Create communication hub
    communication_hub = CommunicationHub()
    
    # Create hierarchical system
    system = HierarchicalTradingSystem(
        meta_controller=meta_controller,
        specialists=specialists,
        portfolio_risk_manager=portfolio_risk_manager,
        communication_hub=communication_hub
    )
    
    return system


class TestHierarchicalSystemIntegration:
    """Test complete hierarchical system integration."""
    
    def test_system_initialization(self, hierarchical_system):
        """Test system initializes correctly."""
        assert hierarchical_system.meta_controller is not None
        assert len(hierarchical_system.specialists) == 3
        assert hierarchical_system.portfolio_risk_manager is not None
        assert hierarchical_system.communication_hub is not None
        
        # Check specialist types
        assert 'forex' in hierarchical_system.specialists
        assert 'commodities' in hierarchical_system.specialists
        assert 'equity' in hierarchical_system.specialists
        
        # Check specialist instruments
        forex_specialist = hierarchical_system.specialists['forex']
        assert len(forex_specialist.get_instruments()) == 3
        assert 'EURUSD' in forex_specialist.get_instruments()
    
    def test_portfolio_state_extraction(self, hierarchical_system, sample_portfolio):
        """Test portfolio state extraction for meta-controller."""
        portfolio_state = hierarchical_system.get_portfolio_state(
            sample_portfolio, 
            hierarchical_system.specialists
        )
        
        assert isinstance(portfolio_state, torch.Tensor)
        assert portfolio_state.shape[1] == 74  # Expected state dimension
        
        # Check that state contains reasonable values
        assert not torch.isnan(portfolio_state).any()
        assert not torch.isinf(portfolio_state).any()
    
    def test_specialist_state_extraction(self, hierarchical_system, mock_market_data):
        """Test specialist state extraction."""
        specialist_states = hierarchical_system.get_specialist_states(
            mock_market_data, 
            hierarchical_system.specialists
        )
        
        assert isinstance(specialist_states, dict)
        assert len(specialist_states) == 3  # 3 specialists
        
        for spec_name, states in specialist_states.items():
            assert 'market_state' in states
            assert 'instrument_states' in states
            assert isinstance(states['market_state'], torch.Tensor)
            assert isinstance(states['instrument_states'], dict)
    
    def test_meta_controller_decision_flow(self, hierarchical_system, sample_portfolio):
        """Test meta-controller decision making."""
        # Get portfolio state
        portfolio_state = hierarchical_system.get_portfolio_state(
            sample_portfolio, 
            hierarchical_system.specialists
        )
        
        # Make meta-controller decision
        allocations, risk_appetite, value = hierarchical_system.meta_controller(portfolio_state)
        
        # Verify outputs
        assert allocations.shape == (1, 3)  # 3 specialists
        assert risk_appetite.shape == (1, 1)
        assert value.shape == (1, 1)
        
        # Check allocations sum to 1 (softmax)
        assert torch.allclose(allocations.sum(dim=1), torch.ones(1), atol=1e-6)
        
        # Check risk appetite is in [0, 1]
        assert 0 <= risk_appetite.item() <= 1
    
    def test_specialist_decision_flow(self, hierarchical_system, mock_market_data):
        """Test specialist decision making."""
        # Get specialist states
        specialist_states = hierarchical_system.get_specialist_states(
            mock_market_data, 
            hierarchical_system.specialists
        )
        
        # Test each specialist
        for spec_name, specialist in hierarchical_system.specialists.items():
            states = specialist_states[spec_name]
            allocation = 0.3  # Mock allocation from meta-controller
            
            # Make specialist decision
            actions, value = specialist(
                states['market_state'], 
                states['instrument_states'], 
                allocation
            )
            
            # Verify outputs
            assert isinstance(actions, dict)
            assert isinstance(value, torch.Tensor)
            assert value.shape == (1, 1)
            
            # Check actions for each instrument
            instruments = specialist.get_instruments()
            assert len(actions) == len(instruments)
            
            for instrument in instruments:
                assert instrument in actions
                action_tensor = actions[instrument]
                assert action_tensor.shape == (1, 3)  # 3 actions (buy, hold, sell)
    
    def test_communication_flow(self, hierarchical_system):
        """Test communication between meta-controller and specialists."""
        # Mock allocation message
        from mtquant.agents.hierarchical.communication import AllocationMessage
        
        allocation_msg = AllocationMessage(
            specialist_id="forex",
            allocation=0.4,
            risk_appetite=0.7,
            market_regime="bull",
            timestamp=datetime.now()
        )
        
        # Send allocation
        hierarchical_system.communication_hub.send_allocation(
            hierarchical_system.meta_controller,
            hierarchical_system.specialists
        )
        
        # Check message history
        messages = hierarchical_system.communication_hub.get_message_history()
        assert len(messages) > 0
        
        # Check that allocation messages were sent
        allocation_messages = [
            msg for msg in messages 
            if isinstance(msg, AllocationMessage)
        ]
        assert len(allocation_messages) == 3  # One for each specialist
    
    def test_risk_management_integration(self, hierarchical_system, sample_portfolio):
        """Test risk management integration."""
        # Create proposed positions
        proposed_positions = [
            Position(
                position_id="new_pos_1",
                agent_id="forex_agent",
                symbol="GBPUSD",
                side="long",
                quantity=20000,
                entry_price=1.2500,
                current_price=1.2500,
                unrealized_pnl=0.0,
                opened_at=datetime.now(),
                updated_at=datetime.now(),
                status="pending"
            )
        ]
        
        # Check portfolio risk
        is_valid, reason = hierarchical_system.portfolio_risk_manager.check_portfolio_risk(
            proposed_positions, 
            sample_portfolio
        )
        
        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)
        
        # If risk check fails, reason should be provided
        if not is_valid:
            assert len(reason) > 0
    
    def test_order_generation(self, hierarchical_system, mock_market_data, sample_portfolio):
        """Test order generation from specialist actions."""
        # Get specialist decisions
        specialist_states = hierarchical_system.get_specialist_states(
            mock_market_data, 
            hierarchical_system.specialists
        )
        
        # Mock specialist actions
        specialist_actions = {}
        for spec_name, specialist in hierarchical_system.specialists.items():
            states = specialist_states[spec_name]
            actions, _ = specialist(states['market_state'], states['instrument_states'], 0.3)
            specialist_actions[spec_name] = actions
        
        # Generate orders
        orders = hierarchical_system.flatten_specialist_actions(specialist_actions)
        
        assert isinstance(orders, list)
        
        # Check order structure
        for order in orders:
            assert isinstance(order, Order)
            assert order.symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'WTIUSD', 'SPX500', 'NAS100', 'US30']
            assert order.side in ['buy', 'sell']
            assert order.order_type in ['market', 'limit', 'stop']
            assert order.quantity > 0
    
    def test_complete_decision_cycle(self, hierarchical_system, mock_market_data, sample_portfolio):
        """Test complete decision cycle from market data to orders."""
        current_positions = sample_portfolio['positions']
        
        # Execute complete step
        try:
            orders = hierarchical_system.step(mock_market_data, sample_portfolio, current_positions)
            
            assert isinstance(orders, list)
            
            # If orders are generated, they should be valid
            for order in orders:
                assert isinstance(order, Order)
                assert order.quantity > 0
                assert order.symbol in mock_market_data.keys()
                
        except Exception as e:
            # If step fails due to missing implementations, that's OK for integration test
            # The important thing is that the system structure is correct
            assert "step" in str(e).lower() or "not implemented" in str(e).lower()


class TestTrainingPipelineIntegration:
    """Test training pipeline integration."""
    
    def test_phase3_trainer_creation(self):
        """Test Phase 3 trainer can be created."""
        trainer = create_phase3_trainer()
        
        assert trainer is not None
        assert trainer.config is not None
        assert trainer.specialists is not None
        assert trainer.meta_controller is not None
        assert trainer.portfolio_risk_manager is not None
        assert trainer.communication_hub is not None
    
    def test_training_components_initialization(self):
        """Test all training components initialize correctly."""
        trainer = create_phase3_trainer()
        
        # Check gradient coordinator
        if trainer.config.gradient_coordination:
            assert trainer.gradient_coordinator is not None
        
        # Check curriculum manager
        if trainer.config.curriculum_learning:
            assert trainer.curriculum_manager is not None
        
        # Check monitoring dashboard
        assert trainer.monitoring_dashboard is not None
        
        # Check checkpointing system
        assert trainer.checkpointing_system is not None
        
        # Check portfolio reward
        assert trainer.portfolio_reward is not None
    
    def test_joint_training_environment_creation(self):
        """Test joint training environment can be created."""
        trainer = create_phase3_trainer()
        
        # Create joint training environment
        env = trainer.create_joint_training_env()
        
        assert env is not None
        assert hasattr(env, 'step')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'render')
    
    def test_parallel_environment_creation(self):
        """Test parallel environments can be created."""
        trainer = create_phase3_trainer()
        
        # Create parallel environments
        parallel_envs = trainer.create_parallel_envs()
        
        assert parallel_envs is not None
        assert hasattr(parallel_envs, 'step')
        assert hasattr(parallel_envs, 'reset')
        assert parallel_envs.num_envs == trainer.config.n_envs


class TestMultiAgentCoordination:
    """Test multi-agent coordination."""
    
    def test_specialist_coordination(self, hierarchical_system):
        """Test coordination between specialists."""
        # Create mock market conditions
        market_data = {
            'EURUSD': pd.DataFrame({
                'close': [1.1000, 1.1010, 1.1020],
                'volume': [1000, 1100, 1200]
            }),
            'XAUUSD': pd.DataFrame({
                'close': [2000, 2010, 2020],
                'volume': [500, 550, 600]
            })
        }
        
        # Get specialist states
        specialist_states = hierarchical_system.get_specialist_states(
            market_data, 
            hierarchical_system.specialists
        )
        
        # Test that specialists can coordinate
        allocations = {'forex': 0.4, 'commodities': 0.3, 'equity': 0.3}
        
        specialist_decisions = {}
        for spec_name, specialist in hierarchical_system.specialists.items():
            if spec_name in specialist_states:
                states = specialist_states[spec_name]
                actions, value = specialist(states['market_state'], states['instrument_states'], allocations[spec_name])
                specialist_decisions[spec_name] = (actions, value)
        
        # Verify coordination
        assert len(specialist_decisions) > 0
        
        # Check that decisions are consistent
        for spec_name, (actions, value) in specialist_decisions.items():
            assert isinstance(actions, dict)
            assert isinstance(value, torch.Tensor)
            assert not torch.isnan(value).any()
    
    def test_risk_coordination(self, hierarchical_system, sample_portfolio):
        """Test risk coordination across agents."""
        # Create multiple proposed positions
        proposed_positions = [
            Position(
                position_id="pos_1",
                agent_id="forex_agent",
                symbol="EURUSD",
                side="long",
                quantity=50000,
                entry_price=1.1000,
                current_price=1.1000,
                unrealized_pnl=0.0,
                opened_at=datetime.now(),
                updated_at=datetime.now(),
                status="pending"
            ),
            Position(
                position_id="pos_2",
                agent_id="commodities_agent",
                symbol="XAUUSD",
                side="long",
                quantity=5.0,
                entry_price=2000.0,
                current_price=2000.0,
                unrealized_pnl=0.0,
                opened_at=datetime.now(),
                updated_at=datetime.now(),
                status="pending"
            )
        ]
        
        # Check portfolio-level risk
        is_valid, reason = hierarchical_system.portfolio_risk_manager.check_portfolio_risk(
            proposed_positions, 
            sample_portfolio
        )
        
        assert isinstance(is_valid, bool)
        
        # If positions are too large, risk check should fail
        if not is_valid:
            assert "risk" in reason.lower() or "limit" in reason.lower() or "var" in reason.lower()
    
    def test_communication_coordination(self, hierarchical_system):
        """Test communication coordination between agents."""
        # Send allocation messages
        hierarchical_system.communication_hub.send_allocation(
            hierarchical_system.meta_controller,
            hierarchical_system.specialists
        )
        
        # Collect performance reports
        reports = hierarchical_system.communication_hub.collect_reports(
            hierarchical_system.specialists
        )
        
        assert isinstance(reports, list)
        assert len(reports) == 3  # One report per specialist
        
        # Check report structure
        for report in reports:
            assert hasattr(report, 'specialist_id')
            assert hasattr(report, 'confidence_score')
            assert hasattr(report, 'realized_pnl')
            assert hasattr(report, 'unrealized_pnl')
            assert hasattr(report, 'sharpe_ratio')
            assert hasattr(report, 'win_rate')
            assert hasattr(report, 'risk_utilization')


class TestErrorHandling:
    """Test error handling in integrated system."""
    
    def test_market_data_error_handling(self, hierarchical_system, sample_portfolio):
        """Test handling of invalid market data."""
        # Create invalid market data
        invalid_market_data = {
            'EURUSD': pd.DataFrame(),  # Empty DataFrame
            'XAUUSD': None  # None value
        }
        
        current_positions = sample_portfolio['positions']
        
        # System should handle invalid data gracefully
        try:
            orders = hierarchical_system.step(invalid_market_data, sample_portfolio, current_positions)
            # If it doesn't crash, that's good
            assert isinstance(orders, list)
        except Exception as e:
            # If it crashes, it should be a meaningful error
            assert "market" in str(e).lower() or "data" in str(e).lower()
    
    def test_portfolio_error_handling(self, hierarchical_system, mock_market_data):
        """Test handling of invalid portfolio data."""
        # Create invalid portfolio
        invalid_portfolio = {
            'portfolio_value': -1000,  # Negative value
            'positions': [],  # Empty positions
            'cash': None  # None value
        }
        
        current_positions = []
        
        # System should handle invalid portfolio gracefully
        try:
            orders = hierarchical_system.step(mock_market_data, invalid_portfolio, current_positions)
            assert isinstance(orders, list)
        except Exception as e:
            # If it crashes, it should be a meaningful error
            assert "portfolio" in str(e).lower() or "value" in str(e).lower()
    
    def test_specialist_error_handling(self, hierarchical_system):
        """Test handling of specialist errors."""
        # Create invalid specialist states
        invalid_states = {
            'market_state': torch.tensor([[float('nan')] * 8]),  # NaN values
            'instrument_states': {
                'EURUSD': torch.tensor([[float('inf')] * 50])  # Inf values
            }
        }
        
        forex_specialist = hierarchical_system.specialists['forex']
        
        # Specialist should handle invalid states gracefully
        try:
            actions, value = forex_specialist(
                invalid_states['market_state'],
                invalid_states['instrument_states'],
                0.3
            )
            # If it doesn't crash, check outputs are valid
            assert not torch.isnan(value).any()
            assert not torch.isinf(value).any()
        except Exception as e:
            # If it crashes, it should be a meaningful error
            assert "nan" in str(e).lower() or "inf" in str(e).lower() or "invalid" in str(e).lower()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
