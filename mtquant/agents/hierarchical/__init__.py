"""
Hierarchical Multi-Agent Trading System

This module implements a hierarchical trading architecture with:
- Meta-Controller: Portfolio-level decision maker
- Specialists: Domain experts (Forex, Commodities, Equity)
- Communication: Inter-agent messaging system
- Risk Management: Portfolio-level risk controls

Architecture:
Level 1: Meta-Controller (Portfolio Manager)
Level 2: Specialists (Domain Experts)
Level 3: Instrument Agents (Execution)
"""

from .meta_controller import MetaController
from .base_specialist import BaseSpecialist
from .forex_specialist import ForexSpecialist
from .commodities_specialist import CommoditiesSpecialist
from .equity_specialist import EquitySpecialist
from .specialist_factory import SpecialistRegistry, get_specialist_registry, create_specialist
from .communication import (
    AllocationMessage,
    PerformanceReport,
    CoordinationSignal,
    AlertMessage,
    CommunicationHub,
    MessageType
)

from .hierarchical_system import HierarchicalTradingSystem, SystemState

__all__ = [
    'MetaController',
    'BaseSpecialist',
    'ForexSpecialist',
    'CommoditiesSpecialist',
    'EquitySpecialist',
    'SpecialistRegistry',
    'get_specialist_registry',
    'create_specialist',
    'AllocationMessage',
    'PerformanceReport',
    'CoordinationSignal',
    'AlertMessage',
    'CommunicationHub',
    'MessageType',
    'HierarchicalTradingSystem',
    'SystemState'
]
