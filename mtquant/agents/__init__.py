"""
MTQuant Agents Module

Contains RL trading agents, environments, policies, and training infrastructure.

Components:
- environments/: Trading environments for RL training
- policies/: RL algorithms (PPO, SAC, TD3)
- training/: Training scripts and utilities
- agent_manager.py: Agent lifecycle management

Each agent is responsible for a single instrument and generates trading signals
based on market data and learned patterns.
"""

__version__ = "0.1.0"

from .agent_manager import AgentManager

__all__ = ["AgentManager"]
