"""
Hierarchical Trading System for Multi-Agent Trading

This module implements the main orchestrator that integrates:
- Meta-Controller for portfolio-level decisions
- Specialists for domain-specific expertise
- Portfolio Risk Manager for risk validation
- Communication Hub for inter-agent messaging

The system coordinates the full decision pipeline from market observation
to order execution with comprehensive risk management.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from .meta_controller import MetaController
from .base_specialist import BaseSpecialist
from .communication import CommunicationHub, AllocationMessage, PerformanceReport
from ...mcp_integration.models.position import Position
from ...mcp_integration.models.order import Order
from ...risk_management.portfolio_risk_manager import PortfolioRiskManager, Portfolio


@dataclass
class SystemState:
    """Current state of the hierarchical trading system."""
    timestamp: datetime
    market_data: Dict[str, Any]
    portfolio: Portfolio
    positions: List[Position]
    meta_decisions: Dict[str, Any]
    specialist_actions: Dict[str, Dict[str, Any]]
    risk_validation: Dict[str, Any]
    approved_orders: List[Order]
    rejected_orders: List[Order]
    system_metrics: Dict[str, float]


class HierarchicalTradingSystem:
    """
    Main orchestrator for hierarchical multi-agent trading system.
    
    This class integrates all components and manages the full decision pipeline:
    1. Meta observes portfolio state
    2. Meta decides allocations + risk appetite
    3. Meta broadcasts to specialists
    4. Specialists observe market + propose actions
    5. Specialists report back to meta
    6. Portfolio risk check
    7. If risk OK: execute orders
    8. If risk violated: scale down or reject
    """
    
    def __init__(
        self,
        meta_controller: MetaController,
        specialists: Dict[str, BaseSpecialist],
        portfolio_risk_manager: PortfolioRiskManager,
        communication_hub: CommunicationHub,
        system_id: str = "hierarchical_system"
    ):
        """
        Initialize HierarchicalTradingSystem.
        
        Args:
            meta_controller: Meta-controller instance
            specialists: Dictionary of specialist instances
            portfolio_risk_manager: Portfolio risk manager
            communication_hub: Communication hub for messaging
            system_id: Unique system identifier
        """
        self.meta_controller = meta_controller
        self.specialists = specialists
        self.portfolio_risk_manager = portfolio_risk_manager
        self.communication_hub = communication_hub
        self.system_id = system_id
        
        # System state
        self.current_state: Optional[SystemState] = None
        self.state_history: List[SystemState] = []
        
        # Configuration
        self.training_mode = False
        self.risk_enabled = True
        self.logging_enabled = True
        
        # Statistics
        self.stats = {
            'total_decisions': 0,
            'orders_executed': 0,
            'orders_rejected': 0,
            'risk_violations': 0,
            'avg_decision_time_ms': 0.0,
            'last_decision_time': None
        }
        
        # Logger
        self.logger = logging.getLogger(__name__)
        if self.logging_enabled:
            self.logger.setLevel(logging.INFO)
        
        # Register system with communication hub
        self.communication_hub.register_agent(
            system_id, 
            'hierarchical_system',
            ['allocation', 'performance_report', 'coordination_signal', 'alert']
        )
        
        # Register specialists
        for specialist_id, specialist in specialists.items():
            self.communication_hub.register_agent(
                specialist_id,
                'specialist',
                ['allocation', 'coordination_signal']
            )
    
    def step(
        self,
        market_data: Dict[str, Any],
        portfolio: Portfolio,
        current_positions: List[Position]
    ) -> List[Order]:
        """
        Execute one decision cycle of the hierarchical system.
        
        Args:
            market_data: Current market data for all instruments
            portfolio: Current portfolio state
            current_positions: List of current positions
            
        Returns:
            approved_orders: List of approved orders to execute
        """
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Meta observes portfolio state
            portfolio_state = self.get_portfolio_state(portfolio, self.specialists)
            
            # Step 2: Meta decides allocations + risk appetite
            with torch.no_grad():
                allocations, risk_appetite, value = self.meta_controller.forward(portfolio_state)
            
            # Convert to numpy and extract values
            allocations_np = allocations.cpu().numpy().flatten()
            risk_appetite_np = risk_appetite.cpu().numpy().item()
            
            # Step 3: Meta broadcasts to specialists
            specialist_ids = list(self.specialists.keys())
            allocation_dict = {
                specialist_id: float(allocations_np[i]) 
                for i, specialist_id in enumerate(specialist_ids)
            }
            
            market_regime = self.meta_controller.detect_market_regime(
                portfolio.returns_history[-30:] if len(portfolio.returns_history) >= 30 else portfolio.returns_history
            )
            
            self.communication_hub.send_allocation(
                meta_controller_id=self.system_id,
                specialists=self.specialists,
                allocations=allocation_dict,
                risk_appetite=risk_appetite_np,
                market_regime=market_regime
            )
            
            # Step 4: Specialists observe market + propose actions
            specialist_states = self.get_specialist_states(market_data, self.specialists)
            specialist_actions = {}
            
            for specialist_id, specialist in self.specialists.items():
                allocation = allocation_dict.get(specialist_id, 0.0)
                market_state = specialist_states[specialist_id]['market_state']
                instrument_states = specialist_states[specialist_id]['instrument_states']
                
                with torch.no_grad():
                    actions, value = specialist.forward(market_state, instrument_states, allocation)
                
                # Convert to numpy
                actions_np = {}
                for instrument, action_tensor in actions.items():
                    actions_np[instrument] = action_tensor.cpu().numpy()
                
                specialist_actions[specialist_id] = {
                    'actions': actions_np,
                    'value': value.cpu().numpy().item(),
                    'allocation': allocation,
                    'confidence': specialist.calculate_confidence(actions)
                }
            
            # Step 5: Specialists report back to meta
            performance_reports = self.communication_hub.collect_reports(self.specialists)
            
            # Step 6: Convert specialist actions to orders
            proposed_orders = self.flatten_specialist_actions(specialist_actions, current_positions)
            
            # Step 7: Portfolio risk check
            approved_orders = []
            rejected_orders = []
            
            if self.risk_enabled:
                # Check portfolio risk
                is_valid, reason = self.portfolio_risk_manager.check_portfolio_risk(
                    proposed_orders, portfolio
                )
                
                if is_valid:
                    approved_orders = proposed_orders
                else:
                    # Step 8: Scale down to risk limit
                    approved_orders = self.scale_down_to_risk_limit(proposed_orders, portfolio)
                    rejected_orders = [order for order in proposed_orders if order not in approved_orders]
                    
                    # Log risk violation
                    self.stats['risk_violations'] += 1
                    self.communication_hub.broadcast_alert(
                        'risk_violation',
                        'warning',
                        f"Risk limit exceeded: {reason}. Orders scaled down.",
                        {'rejected_count': len(rejected_orders), 'approved_count': len(approved_orders)}
                    )
            else:
                approved_orders = proposed_orders
            
            # Step 9: Update system state
            decision_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.current_state = SystemState(
                timestamp=datetime.utcnow(),
                market_data=market_data,
                portfolio=portfolio,
                positions=current_positions,
                meta_decisions={
                    'allocations': allocation_dict,
                    'risk_appetite': risk_appetite_np,
                    'market_regime': market_regime,
                    'value': value.cpu().numpy().item()
                },
                specialist_actions=specialist_actions,
                risk_validation={
                    'risk_enabled': self.risk_enabled,
                    'orders_checked': len(proposed_orders),
                    'orders_approved': len(approved_orders),
                    'orders_rejected': len(rejected_orders)
                },
                approved_orders=approved_orders,
                rejected_orders=rejected_orders,
                system_metrics={
                    'decision_time_ms': decision_time,
                    'total_specialists': len(self.specialists),
                    'total_instruments': sum(len(spec.instruments) for spec in self.specialists.values())
                }
            )
            
            # Update statistics
            self.stats['total_decisions'] += 1
            self.stats['orders_executed'] += len(approved_orders)
            self.stats['orders_rejected'] += len(rejected_orders)
            self.stats['last_decision_time'] = datetime.utcnow()
            
            # Update average decision time
            if self.stats['total_decisions'] == 1:
                self.stats['avg_decision_time_ms'] = decision_time
            else:
                self.stats['avg_decision_time_ms'] = (
                    (self.stats['avg_decision_time_ms'] * (self.stats['total_decisions'] - 1) + decision_time) /
                    self.stats['total_decisions']
                )
            
            # Store state history
            self.state_history.append(self.current_state)
            if len(self.state_history) > 1000:  # Keep last 1000 states
                self.state_history.pop(0)
            
            # Log decision
            if self.logging_enabled:
                self.logger.info(
                    f"Decision cycle completed: {len(approved_orders)} orders approved, "
                    f"{len(rejected_orders)} rejected, {decision_time:.2f}ms"
                )
            
            return approved_orders
            
        except Exception as e:
            # Handle errors gracefully
            self.logger.error(f"Error in decision cycle: {e}")
            self.communication_hub.broadcast_alert(
                'system_error',
                'error',
                f"Decision cycle failed: {str(e)}",
                {'error_type': type(e).__name__}
            )
            return []
    
    def get_portfolio_state(
        self,
        portfolio: Portfolio,
        specialists: Dict[str, BaseSpecialist]
    ) -> torch.Tensor:
        """
        Extract 74-dim portfolio state for meta-controller.
        
        Args:
            portfolio: Current portfolio state
            specialists: Dictionary of specialist instances
            
        Returns:
            portfolio_state: 74-dimensional tensor
        """
        # Get specialist performance metrics
        specialist_performance = {}
        for specialist_id, specialist in specialists.items():
            metrics = getattr(specialist, 'get_performance_metrics', lambda: {})()
            specialist_performance[specialist_id] = {
                'sharpe': metrics.get('sharpe_ratio', 0.0),
                'win_rate': metrics.get('win_rate', 0.5),
                'max_drawdown': metrics.get('max_drawdown', 0.0)
            }
        
        # Use meta-controller's method to extract state
        portfolio_data = {
            'returns': portfolio.returns_history.flatten() if len(portfolio.returns_history) > 0 else np.zeros(30),
            'volatility': np.std(portfolio.returns_history) if len(portfolio.returns_history) > 0 else 0.0,
            'drawdown': portfolio.equity * 0.05,  # Simplified drawdown calculation
            'correlation_matrix': portfolio.correlation_matrix,
            'vix': 20.0,  # Placeholder - should come from market data
            'dxy': 100.0,  # Placeholder
            'fed_rate': 5.25,  # Placeholder
            '10y_yield': 4.5,  # Placeholder
            'credit_spread': 1.5  # Placeholder
        }
        
        return self.meta_controller.get_portfolio_state(portfolio_data, specialist_performance)
    
    def get_specialist_states(
        self,
        market_data: Dict[str, Any],
        specialists: Dict[str, BaseSpecialist]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract market + instrument states per specialist.
        
        Args:
            market_data: Raw market data
            specialists: Dictionary of specialist instances
            
        Returns:
            specialist_states: Dictionary mapping specialist IDs to their states
        """
        specialist_states = {}
        
        for specialist_id, specialist in specialists.items():
            # Get domain-specific market features
            market_state = specialist.get_domain_features(market_data)
            
            # Get instrument-specific observations
            instrument_states = {}
            for instrument in specialist.instruments:
                if instrument in market_data:
                    # Convert market data to observation tensor
                    # This is simplified - in practice, this would be more complex
                    observation = torch.tensor(market_data[instrument], dtype=torch.float32)
                    instrument_states[instrument] = observation
                else:
                    # Default observation if data not available
                    observation = torch.zeros(50, dtype=torch.float32)  # Default observation dim
                    instrument_states[instrument] = observation
            
            specialist_states[specialist_id] = {
                'market_state': market_state,
                'instrument_states': instrument_states
            }
        
        return specialist_states
    
    def flatten_specialist_actions(
        self,
        specialist_actions: Dict[str, Dict[str, Any]],
        current_positions: List[Position]
    ) -> List[Order]:
        """
        Convert specialist actions to order list.
        
        Args:
            specialist_actions: Dictionary of specialist actions
            current_positions: Current positions
            
        Returns:
            orders: List of proposed orders
        """
        orders = []
        
        for specialist_id, actions_data in specialist_actions.items():
            actions = actions_data['actions']
            allocation = actions_data['allocation']
            
            for instrument, action_probs in actions.items():
                # Get most likely action
                action_idx = np.argmax(action_probs)
                
                # Skip hold actions (index 1)
                if action_idx == 1:  # Hold
                    continue
                
                # Determine order side
                side = 'buy' if action_idx == 0 else 'sell'
                
                # Calculate position size based on allocation and confidence
                confidence = actions_data['confidence']
                base_size = 0.1  # Base position size
                position_size = base_size * allocation * confidence
                
                # Create order
                order = Order(
                    order_id=f"{specialist_id}_{instrument}_{datetime.utcnow().timestamp()}",
                    agent_id=specialist_id,
                    symbol=instrument,
                    side=side,
                    order_type='market',
                    quantity=position_size,
                    timestamp=datetime.utcnow()
                )
                
                orders.append(order)
        
        return orders
    
    def scale_down_to_risk_limit(
        self,
        orders: List[Order],
        portfolio: Portfolio
    ) -> List[Order]:
        """
        Proportionally reduce order sizes if risk exceeded.
        
        Args:
            orders: List of proposed orders
            portfolio: Current portfolio state
            
        Returns:
            scaled_orders: List of orders scaled to risk limits
        """
        if not orders:
            return orders
        
        # Calculate total order value
        total_order_value = sum(order.quantity * 1000 for order in orders)  # Simplified
        
        # Calculate risk limit (simplified)
        risk_limit = portfolio.equity * 0.02  # 2% of equity
        
        if total_order_value <= risk_limit:
            return orders
        
        # Calculate scaling factor
        scaling_factor = risk_limit / total_order_value
        
        # Scale down orders
        scaled_orders = []
        for order in orders:
            scaled_order = Order(
                order_id=order.order_id,
                agent_id=order.agent_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity * scaling_factor,
                timestamp=order.timestamp
            )
            scaled_orders.append(scaled_order)
        
        return scaled_orders
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            status: Dictionary with system status information
        """
        return {
            'system_id': self.system_id,
            'timestamp': datetime.utcnow().isoformat(),
            'training_mode': self.training_mode,
            'risk_enabled': self.risk_enabled,
            'statistics': self.stats,
            'specialists': {
                specialist_id: {
                    'type': specialist.specialist_type,
                    'instruments': specialist.instruments,
                    'status': 'active'
                }
                for specialist_id, specialist in self.specialists.items()
            },
            'communication_stats': self.communication_hub.get_statistics(),
            'current_state': self.current_state.to_dict() if self.current_state else None
        }
    
    def enable_training_mode(self) -> None:
        """Enable training mode for gradient computation."""
        self.training_mode = True
        self.logger.info("Training mode enabled")
    
    def disable_training_mode(self) -> None:
        """Disable training mode for inference only."""
        self.training_mode = False
        self.logger.info("Training mode disabled")
    
    def enable_risk_management(self) -> None:
        """Enable portfolio risk management."""
        self.risk_enabled = True
        self.logger.info("Risk management enabled")
    
    def disable_risk_management(self) -> None:
        """Disable portfolio risk management (dangerous!)."""
        self.risk_enabled = False
        self.logger.warning("Risk management disabled - this is dangerous!")
    
    def get_decision_history(self, limit: Optional[int] = None) -> List[SystemState]:
        """
        Get decision history.
        
        Args:
            limit: Maximum number of states to return
            
        Returns:
            history: List of system states
        """
        if limit is None:
            return self.state_history.copy()
        else:
            return self.state_history[-limit:]
    
    def export_system_state(self, filepath: str) -> None:
        """
        Export current system state to file.
        
        Args:
            filepath: Path to export file
        """
        export_data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'system_status': self.get_system_status(),
            'current_state': self.current_state.to_dict() if self.current_state else None,
            'recent_history': [
                state.to_dict() for state in self.state_history[-10:]  # Last 10 states
            ]
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"System state exported to {filepath}")


# Unit test stubs (to be implemented in test files)
"""
def test_hierarchical_system_initialization():
    '''Test HierarchicalTradingSystem initialization.'''
    # Create mock components
    meta_controller = MetaController()
    specialists = {
        'forex': ForexSpecialist(),
        'commodities': CommoditiesSpecialist(),
        'equity': EquitySpecialist()
    }
    risk_manager = PortfolioRiskManager()
    comm_hub = CommunicationHub()
    
    # Initialize system
    system = HierarchicalTradingSystem(
        meta_controller=meta_controller,
        specialists=specialists,
        portfolio_risk_manager=risk_manager,
        communication_hub=comm_hub
    )
    
    assert system.meta_controller == meta_controller
    assert len(system.specialists) == 3
    assert system.risk_enabled == True
    assert system.training_mode == False

def test_decision_cycle():
    '''Test full decision cycle.'''
    # Create system with mock components
    system = HierarchicalTradingSystem(...)
    
    # Create test data
    market_data = {
        'EURUSD': [1.1000, 1.1001, 1.0999],
        'XAUUSD': [2000.0, 2001.0, 1999.0],
        'SPX500': [4000.0, 4001.0, 3999.0]
    }
    
    portfolio = Portfolio(
        equity=100000.0,
        margin_used=0.0,
        margin_available=100000.0,
        positions=[],
        returns_history=np.random.randn(100, 8) * 0.01,
        correlation_matrix=np.eye(8),
        sector_allocation={'forex': 0.0, 'commodities': 0.0, 'equity': 0.0},
        last_updated=datetime.utcnow()
    )
    
    # Execute decision cycle
    orders = system.step(market_data, portfolio, [])
    
    assert isinstance(orders, list)
    assert system.stats['total_decisions'] == 1
    assert system.current_state is not None

def test_portfolio_state_extraction():
    '''Test portfolio state extraction.'''
    system = HierarchicalTradingSystem(...)
    
    portfolio = Portfolio(...)
    specialists = {...}
    
    state = system.get_portfolio_state(portfolio, specialists)
    assert state.shape == (74,)
    assert isinstance(state, torch.Tensor)

def test_risk_scaling():
    '''Test risk limit scaling.'''
    system = HierarchicalTradingSystem(...)
    
    orders = [
        Order(order_id='1', agent_id='test', symbol='EURUSD', side='buy', order_type='market', quantity=1.0),
        Order(order_id='2', agent_id='test', symbol='XAUUSD', side='buy', order_type='market', quantity=0.1)
    ]
    
    portfolio = Portfolio(equity=10000.0, ...)
    
    scaled_orders = system.scale_down_to_risk_limit(orders, portfolio)
    assert len(scaled_orders) == len(orders)
    assert all(order.quantity <= original_order.quantity for order, original_order in zip(scaled_orders, orders))
"""
