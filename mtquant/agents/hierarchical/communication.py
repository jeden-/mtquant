"""
Agent Communication System for Hierarchical Multi-Agent Trading System

This module implements inter-agent messaging system with:
- Message types for different communication patterns
- CommunicationHub for centralized message routing
- Message history and querying
- Correlation ID tracking for debugging
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from collections import deque
import uuid
import json
import logging
from enum import Enum


class MessageType(Enum):
    """Message types for different communication patterns."""
    ALLOCATION = "allocation"
    PERFORMANCE_REPORT = "performance_report"
    COORDINATION_SIGNAL = "coordination_signal"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"


@dataclass
class AllocationMessage:
    """
    Top-down message: Meta-Controller → Specialist
    
    Contains allocation decisions and market regime information
    from the meta-controller to individual specialists.
    """
    specialist_id: str
    allocation: float  # 0-1 capital allocation
    risk_appetite: float  # 0-1 risk appetite (0=defensive, 1=aggressive)
    market_regime: str  # 'bull', 'bear', 'neutral', 'volatile'
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': MessageType.ALLOCATION.value,
            'specialist_id': self.specialist_id,
            'allocation': self.allocation,
            'risk_appetite': self.risk_appetite,
            'market_regime': self.market_regime,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AllocationMessage':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class PerformanceReport:
    """
    Bottom-up message: Specialist → Meta-Controller
    
    Contains performance metrics and confidence scores
    from specialists back to the meta-controller.
    """
    specialist_id: str
    confidence_score: float  # 0-1 confidence in current actions
    realized_pnl: float  # Realized P&L since last report
    unrealized_pnl: float  # Current unrealized P&L
    sharpe_ratio: float  # Recent Sharpe ratio
    win_rate: float  # Win rate (0-1)
    risk_utilization: float  # % of allocated risk used (0-1)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': MessageType.PERFORMANCE_REPORT.value,
            'specialist_id': self.specialist_id,
            'confidence_score': self.confidence_score,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'risk_utilization': self.risk_utilization,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceReport':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class CoordinationSignal:
    """
    Horizontal message: Specialist ↔ Specialist
    
    Contains coordination signals between specialists
    for hedging opportunities, correlation alerts, etc.
    """
    from_specialist: str
    to_specialist: str
    signal_type: str  # 'hedge_opportunity', 'correlation_alert', 'risk_warning'
    data: Dict[str, Any]  # Signal-specific data
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: str = "normal"  # 'low', 'normal', 'high', 'critical'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': MessageType.COORDINATION_SIGNAL.value,
            'from_specialist': self.from_specialist,
            'to_specialist': self.to_specialist,
            'signal_type': self.signal_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoordinationSignal':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class AlertMessage:
    """
    System-wide alert message.
    
    Used for circuit breaker activations, system errors,
    and other critical notifications.
    """
    alert_type: str  # 'circuit_breaker', 'system_error', 'risk_violation'
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str  # Human-readable message
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': MessageType.ALERT.value,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertMessage':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class CommunicationHub:
    """
    Central message broker for hierarchical trading system.
    
    Features:
    - Route messages between agents
    - Maintain message history
    - Broadcast alerts
    - Correlation ID tracking
    - Message querying and filtering
    """
    
    def __init__(
        self,
        message_history_size: int = 1000,
        correlation_id_enabled: bool = True,
        log_all_messages: bool = True
    ):
        """
        Initialize CommunicationHub.
        
        Args:
            message_history_size: Maximum number of messages to store
            correlation_id_enabled: Enable correlation ID tracking
            log_all_messages: Log all messages for debugging
        """
        self.message_history_size = message_history_size
        self.correlation_id_enabled = correlation_id_enabled
        self.log_all_messages = log_all_messages
        
        # Message storage
        self.message_history: deque = deque(maxlen=message_history_size)
        self.correlation_tracker: Dict[str, List[str]] = {}  # correlation_id -> message_ids
        
        # Agent registrations
        self.registered_agents: Dict[str, str] = {}  # agent_id -> agent_type
        self.agent_subscriptions: Dict[str, List[str]] = {}  # agent_id -> message_types
        
        # Statistics
        self.message_stats = {
            'total_messages': 0,
            'messages_by_type': {msg_type.value: 0 for msg_type in MessageType},
            'messages_by_agent': {},
            'correlation_chains': 0
        }
        
        # Logger
        self.logger = logging.getLogger(__name__)
        if log_all_messages:
            self.logger.setLevel(logging.INFO)
    
    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        subscriptions: Optional[List[str]] = None
    ) -> None:
        """
        Register an agent with the communication hub.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent ('meta_controller', 'specialist', etc.)
            subscriptions: List of message types to subscribe to
        """
        self.registered_agents[agent_id] = agent_type
        self.agent_subscriptions[agent_id] = subscriptions or []
        self.message_stats['messages_by_agent'][agent_id] = 0
        
        self.logger.info(f"Registered agent {agent_id} of type {agent_type}")
    
    def send_message(self, message: Union[AllocationMessage, PerformanceReport, CoordinationSignal, AlertMessage]) -> str:
        """
        Send a message through the communication hub.
        
        Args:
            message: Message to send
            
        Returns:
            message_id: Unique message identifier
        """
        message_id = str(uuid.uuid4())
        
        # Add to history
        message_entry = {
            'id': message_id,
            'message': message,
            'timestamp': datetime.utcnow()
        }
        self.message_history.append(message_entry)
        
        # Update correlation tracking
        if self.correlation_id_enabled and hasattr(message, 'correlation_id'):
            if message.correlation_id not in self.correlation_tracker:
                self.correlation_tracker[message.correlation_id] = []
            self.correlation_tracker[message.correlation_id].append(message_id)
        
        # Update statistics
        self.message_stats['total_messages'] += 1
        message_type = self._get_message_type(message)
        self.message_stats['messages_by_type'][message_type] += 1
        
        # Log message
        if self.log_all_messages:
            self.logger.info(f"Sent {message_type} message: {message_id}")
        
        return message_id
    
    def send_allocation(
        self,
        meta_controller_id: str,
        specialists: Dict[str, Any],
        allocations: Dict[str, float],
        risk_appetite: float,
        market_regime: str
    ) -> List[str]:
        """
        Meta broadcasts allocation to all specialists.
        
        Args:
            meta_controller_id: Meta-controller identifier
            specialists: Dictionary of specialist instances
            allocations: Allocation amounts per specialist
            risk_appetite: Risk appetite level
            market_regime: Current market regime
            
        Returns:
            message_ids: List of sent message IDs
        """
        message_ids = []
        
        for specialist_id, specialist in specialists.items():
            allocation = allocations.get(specialist_id, 0.0)
            
            message = AllocationMessage(
                specialist_id=specialist_id,
                allocation=allocation,
                risk_appetite=risk_appetite,
                market_regime=market_regime,
                metadata={
                    'from_agent': meta_controller_id,
                    'total_specialists': len(specialists)
                }
            )
            
            message_id = self.send_message(message)
            message_ids.append(message_id)
            
            # Update agent stats
            if specialist_id in self.message_stats['messages_by_agent']:
                self.message_stats['messages_by_agent'][specialist_id] += 1
        
        return message_ids
    
    def collect_reports(
        self,
        specialists: Dict[str, Any]
    ) -> List[PerformanceReport]:
        """
        Meta collects performance from all specialists.
        
        Args:
            specialists: Dictionary of specialist instances
            
        Returns:
            reports: List of performance reports
        """
        reports = []
        
        for specialist_id, specialist in specialists.items():
            # Get performance metrics from specialist
            metrics = getattr(specialist, 'get_performance_metrics', lambda: {})()
            
            report = PerformanceReport(
                specialist_id=specialist_id,
                confidence_score=metrics.get('confidence', 0.5),
                realized_pnl=metrics.get('realized_pnl', 0.0),
                unrealized_pnl=metrics.get('unrealized_pnl', 0.0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                win_rate=metrics.get('win_rate', 0.5),
                risk_utilization=metrics.get('risk_utilization', 0.0),
                metadata={
                    'specialist_type': getattr(specialist, 'specialist_type', 'unknown'),
                    'instruments': getattr(specialist, 'instruments', [])
                }
            )
            
            self.send_message(report)
            reports.append(report)
        
        return reports
    
    def broadcast_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        System-wide alert (e.g., circuit breaker activated).
        
        Args:
            alert_type: Type of alert
            severity: Alert severity level
            message: Human-readable message
            data: Additional alert data
            
        Returns:
            message_id: Sent message ID
        """
        alert = AlertMessage(
            alert_type=alert_type,
            severity=severity,
            message=message,
            data=data or {}
        )
        
        message_id = self.send_message(alert)
        
        # Log critical alerts
        if severity in ['error', 'critical']:
            self.logger.error(f"Critical alert: {message}")
        
        return message_id
    
    def check_coordination_opportunities(
        self,
        specialists: Dict[str, Any]
    ) -> List[CoordinationSignal]:
        """
        Detect hedge opportunities, correlation alerts.
        
        Args:
            specialists: Dictionary of specialist instances
            
        Returns:
            signals: List of coordination signals
        """
        signals = []
        
        # Check for correlation opportunities
        specialist_list = list(specialists.items())
        for i, (id1, spec1) in enumerate(specialist_list):
            for j, (id2, spec2) in enumerate(specialist_list[i+1:], i+1):
                # Check for hedging opportunities
                if self._detect_hedge_opportunity(spec1, spec2):
                    signal = CoordinationSignal(
                        from_specialist=id1,
                        to_specialist=id2,
                        signal_type='hedge_opportunity',
                        data={
                            'hedge_ratio': 0.5,  # Simplified
                            'correlation': -0.8  # Simplified
                        },
                        priority='normal'
                    )
                    signals.append(signal)
                    self.send_message(signal)
                
                # Check for correlation alerts
                if self._detect_correlation_alert(spec1, spec2):
                    signal = CoordinationSignal(
                        from_specialist=id1,
                        to_specialist=id2,
                        signal_type='correlation_alert',
                        data={
                            'correlation_level': 0.9,  # Simplified
                            'risk_level': 'high'
                        },
                        priority='high'
                    )
                    signals.append(signal)
                    self.send_message(signal)
        
        return signals
    
    def _detect_hedge_opportunity(self, spec1: Any, spec2: Any) -> bool:
        """Detect hedging opportunity between two specialists."""
        # Simplified logic - in practice, this would analyze positions and correlations
        return False  # Placeholder
    
    def _detect_correlation_alert(self, spec1: Any, spec2: Any) -> bool:
        """Detect correlation alert between two specialists."""
        # Simplified logic - in practice, this would analyze correlation levels
        return False  # Placeholder
    
    def get_messages_by_type(
        self,
        message_type: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages by type.
        
        Args:
            message_type: Message type to filter by
            limit: Maximum number of messages to return
            
        Returns:
            messages: List of message entries
        """
        filtered_messages = [
            entry for entry in self.message_history
            if self._get_message_type(entry['message']) == message_type
        ]
        
        if limit:
            filtered_messages = filtered_messages[-limit:]
        
        return filtered_messages
    
    def get_messages_by_agent(
        self,
        agent_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages by agent.
        
        Args:
            agent_id: Agent identifier
            limit: Maximum number of messages to return
            
        Returns:
            messages: List of message entries
        """
        filtered_messages = []
        
        for entry in self.message_history:
            message = entry['message']
            if hasattr(message, 'specialist_id') and message.specialist_id == agent_id:
                filtered_messages.append(entry)
            elif hasattr(message, 'from_specialist') and message.from_specialist == agent_id:
                filtered_messages.append(entry)
            elif hasattr(message, 'to_specialist') and message.to_specialist == agent_id:
                filtered_messages.append(entry)
        
        if limit:
            filtered_messages = filtered_messages[-limit:]
        
        return filtered_messages
    
    def get_correlation_chain(self, correlation_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages in a correlation chain.
        
        Args:
            correlation_id: Correlation identifier
            
        Returns:
            messages: List of messages in the chain
        """
        if correlation_id not in self.correlation_tracker:
            return []
        
        message_ids = self.correlation_tracker[correlation_id]
        messages = []
        
        for entry in self.message_history:
            if entry['id'] in message_ids:
                messages.append(entry)
        
        return sorted(messages, key=lambda x: x['timestamp'])
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get communication hub statistics.
        
        Returns:
            stats: Dictionary with statistics
        """
        return {
            **self.message_stats,
            'registered_agents': len(self.registered_agents),
            'active_correlations': len(self.correlation_tracker),
            'message_history_size': len(self.message_history)
        }
    
    def _get_message_type(self, message: Union[AllocationMessage, PerformanceReport, CoordinationSignal, AlertMessage]) -> str:
        """Get message type string."""
        if isinstance(message, AllocationMessage):
            return MessageType.ALLOCATION.value
        elif isinstance(message, PerformanceReport):
            return MessageType.PERFORMANCE_REPORT.value
        elif isinstance(message, CoordinationSignal):
            return MessageType.COORDINATION_SIGNAL.value
        elif isinstance(message, AlertMessage):
            return MessageType.ALERT.value
        else:
            return 'unknown'
    
    def clear_history(self) -> None:
        """Clear message history."""
        self.message_history.clear()
        self.correlation_tracker.clear()
        self.logger.info("Message history cleared")
    
    def export_messages(self, filepath: str) -> None:
        """
        Export message history to JSON file.
        
        Args:
            filepath: Path to export file
        """
        export_data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'statistics': self.get_statistics(),
            'messages': [
                {
                    'id': entry['id'],
                    'timestamp': entry['timestamp'].isoformat(),
                    'message': entry['message'].to_dict()
                }
                for entry in self.message_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(self.message_history)} messages to {filepath}")


# Unit test stubs (to be implemented in test files)
"""
def test_allocation_message():
    '''Test AllocationMessage creation and serialization.'''
    message = AllocationMessage(
        specialist_id='forex_specialist',
        allocation=0.4,
        risk_appetite=0.7,
        market_regime='bull'
    )
    
    assert message.specialist_id == 'forex_specialist'
    assert message.allocation == 0.4
    assert message.risk_appetite == 0.7
    assert message.market_regime == 'bull'
    
    # Test serialization
    data = message.to_dict()
    assert data['type'] == 'allocation'
    assert data['specialist_id'] == 'forex_specialist'
    
    # Test deserialization
    restored = AllocationMessage.from_dict(data)
    assert restored.specialist_id == message.specialist_id

def test_performance_report():
    '''Test PerformanceReport creation and serialization.'''
    report = PerformanceReport(
        specialist_id='commodities_specialist',
        confidence_score=0.8,
        realized_pnl=100.0,
        unrealized_pnl=50.0,
        sharpe_ratio=1.5,
        win_rate=0.6,
        risk_utilization=0.3
    )
    
    assert report.specialist_id == 'commodities_specialist'
    assert report.confidence_score == 0.8
    assert report.sharpe_ratio == 1.5
    
    # Test serialization
    data = report.to_dict()
    assert data['type'] == 'performance_report'

def test_coordination_signal():
    '''Test CoordinationSignal creation and serialization.'''
    signal = CoordinationSignal(
        from_specialist='forex_specialist',
        to_specialist='commodities_specialist',
        signal_type='hedge_opportunity',
        data={'hedge_ratio': 0.5},
        priority='high'
    )
    
    assert signal.from_specialist == 'forex_specialist'
    assert signal.to_specialist == 'commodities_specialist'
    assert signal.signal_type == 'hedge_opportunity'
    assert signal.priority == 'high'

def test_communication_hub():
    '''Test CommunicationHub functionality.'''
    hub = CommunicationHub(message_history_size=100)
    
    # Register agents
    hub.register_agent('meta_controller', 'meta_controller')
    hub.register_agent('forex_specialist', 'specialist')
    hub.register_agent('commodities_specialist', 'specialist')
    
    # Send allocation message
    message = AllocationMessage(
        specialist_id='forex_specialist',
        allocation=0.4,
        risk_appetite=0.7,
        market_regime='bull'
    )
    
    message_id = hub.send_message(message)
    assert message_id is not None
    
    # Check statistics
    stats = hub.get_statistics()
    assert stats['total_messages'] == 1
    assert stats['messages_by_type']['allocation'] == 1

def test_message_filtering():
    '''Test message filtering by type and agent.'''
    hub = CommunicationHub()
    
    # Send different message types
    allocation_msg = AllocationMessage('forex_specialist', 0.4, 0.7, 'bull')
    performance_msg = PerformanceReport('forex_specialist', 0.8, 100.0, 50.0, 1.5, 0.6, 0.3)
    
    hub.send_message(allocation_msg)
    hub.send_message(performance_msg)
    
    # Filter by type
    allocation_messages = hub.get_messages_by_type('allocation')
    assert len(allocation_messages) == 1
    
    performance_messages = hub.get_messages_by_type('performance_report')
    assert len(performance_messages) == 1

def test_correlation_tracking():
    '''Test correlation ID tracking.'''
    hub = CommunicationHub(correlation_id_enabled=True)
    
    # Send messages with same correlation ID
    correlation_id = str(uuid.uuid4())
    
    msg1 = AllocationMessage('forex_specialist', 0.4, 0.7, 'bull')
    msg1.correlation_id = correlation_id
    
    msg2 = PerformanceReport('forex_specialist', 0.8, 100.0, 50.0, 1.5, 0.6, 0.3)
    msg2.correlation_id = correlation_id
    
    hub.send_message(msg1)
    hub.send_message(msg2)
    
    # Check correlation chain
    chain = hub.get_correlation_chain(correlation_id)
    assert len(chain) == 2
"""
