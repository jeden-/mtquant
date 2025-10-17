"""
Agent Manager - Centralized Agent Lifecycle Management.

This module provides comprehensive agent lifecycle management including:
- AgentLifecycleManager: State management and transitions
- AgentScheduler: Task scheduling and execution
- AgentRegistry: Active agent tracking and monitoring

Author: MTQuant Development Team
Date: October 15, 2025
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict
import json

from mtquant.utils.logger import get_logger
from mtquant.utils.exceptions import (
    MTQuantError,
    AgentError,
    StateTransitionError
)


logger = get_logger(__name__)


class AgentState(str, Enum):
    """Agent lifecycle states."""
    INITIALIZED = "initialized"
    TRAINING = "training"
    PAPER_TRADING = "paper_trading"
    LIVE = "live"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_win_rate(self) -> None:
        """Calculate and update win rate."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        else:
            self.win_rate = 0.0


@dataclass
class AgentInfo:
    """Agent information and state."""
    agent_id: str
    agent_type: str  # 'specialist', 'meta_controller', 'ppo'
    symbol: Optional[str] = None
    state: AgentState = AgentState.INITIALIZED
    created_at: datetime = field(default_factory=datetime.now)
    last_state_change: datetime = field(default_factory=datetime.now)
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    config: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'symbol': self.symbol,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'last_state_change': self.last_state_change.isoformat(),
            'metrics': {
                'total_trades': self.metrics.total_trades,
                'winning_trades': self.metrics.winning_trades,
                'losing_trades': self.metrics.losing_trades,
                'total_pnl': self.metrics.total_pnl,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'max_drawdown': self.metrics.max_drawdown,
                'win_rate': self.metrics.win_rate,
                'last_updated': self.metrics.last_updated.isoformat()
            },
            'config': self.config,
            'error_message': self.error_message
        }


class AgentLifecycleManager:
    """
    Manages agent lifecycle states and transitions.
    
    Valid state transitions:
    - INITIALIZED → TRAINING
    - TRAINING → PAPER_TRADING
    - PAPER_TRADING → LIVE
    - Any state → PAUSED
    - PAUSED → previous state
    - Any state → ERROR
    - Any state → STOPPED
    
    Example:
        >>> manager = AgentLifecycleManager()
        >>> manager.register_agent('agent_1', 'specialist', symbol='XAUUSD')
        >>> manager.transition_state('agent_1', AgentState.TRAINING)
        >>> manager.get_agent_state('agent_1')
        AgentState.TRAINING
    """
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        AgentState.INITIALIZED: [AgentState.TRAINING, AgentState.PAUSED, AgentState.STOPPED],
        AgentState.TRAINING: [AgentState.PAPER_TRADING, AgentState.PAUSED, AgentState.ERROR, AgentState.STOPPED],
        AgentState.PAPER_TRADING: [AgentState.LIVE, AgentState.PAUSED, AgentState.ERROR, AgentState.STOPPED],
        AgentState.LIVE: [AgentState.PAUSED, AgentState.ERROR, AgentState.STOPPED],
        AgentState.PAUSED: [AgentState.TRAINING, AgentState.PAPER_TRADING, AgentState.LIVE, AgentState.STOPPED],
        AgentState.ERROR: [AgentState.INITIALIZED, AgentState.STOPPED],
        AgentState.STOPPED: [AgentState.INITIALIZED]
    }
    
    def __init__(self):
        """Initialize the lifecycle manager."""
        self._agents: Dict[str, AgentInfo] = {}
        self._state_history: Dict[str, List[tuple]] = defaultdict(list)
        self.logger = get_logger(__name__)
    
    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        symbol: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentInfo:
        """
        Register a new agent.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent ('specialist', 'meta_controller', 'ppo')
            symbol: Trading symbol (optional)
            config: Agent configuration (optional)
            
        Returns:
            AgentInfo object
            
        Raises:
            AgentError: If agent already registered
        """
        if agent_id in self._agents:
            raise AgentError(f"Agent {agent_id} already registered")
        
        agent_info = AgentInfo(
            agent_id=agent_id,
            agent_type=agent_type,
            symbol=symbol,
            config=config or {}
        )
        
        self._agents[agent_id] = agent_info
        self._state_history[agent_id].append((AgentState.INITIALIZED, datetime.now()))
        
        self.logger.info(f"Registered agent {agent_id} (type: {agent_type}, symbol: {symbol})")
        return agent_info
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent.
        
        Args:
            agent_id: Agent identifier
            
        Raises:
            AgentError: If agent not found
        """
        if agent_id not in self._agents:
            raise AgentError(f"Agent {agent_id} not found")
        
        # Transition to STOPPED before unregistering
        if self._agents[agent_id].state != AgentState.STOPPED:
            self.transition_state(agent_id, AgentState.STOPPED)
        
        del self._agents[agent_id]
        self.logger.info(f"Unregistered agent {agent_id}")
    
    def transition_state(self, agent_id: str, new_state: AgentState, error_message: Optional[str] = None) -> None:
        """
        Transition agent to a new state.
        
        Args:
            agent_id: Agent identifier
            new_state: Target state
            error_message: Error message if transitioning to ERROR state
            
        Raises:
            AgentError: If agent not found
            StateTransitionError: If transition is invalid
        """
        if agent_id not in self._agents:
            raise AgentError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        current_state = agent.state
        
        # Check if transition is valid
        if new_state not in self.VALID_TRANSITIONS[current_state]:
            raise StateTransitionError(
                f"Invalid state transition for agent {agent_id}: "
                f"{current_state.value} → {new_state.value}"
            )
        
        # Update state
        agent.state = new_state
        agent.last_state_change = datetime.now()
        
        if new_state == AgentState.ERROR:
            agent.error_message = error_message
        else:
            agent.error_message = None
        
        # Record in history
        self._state_history[agent_id].append((new_state, agent.last_state_change))
        
        self.logger.info(
            f"Agent {agent_id} transitioned: {current_state.value} → {new_state.value}"
            + (f" (error: {error_message})" if error_message else "")
        )
    
    def get_agent_state(self, agent_id: str) -> AgentState:
        """
        Get current agent state.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Current AgentState
            
        Raises:
            AgentError: If agent not found
        """
        if agent_id not in self._agents:
            raise AgentError(f"Agent {agent_id} not found")
        
        return self._agents[agent_id].state
    
    def get_agent_info(self, agent_id: str) -> AgentInfo:
        """
        Get complete agent information.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            AgentInfo object
            
        Raises:
            AgentError: If agent not found
        """
        if agent_id not in self._agents:
            raise AgentError(f"Agent {agent_id} not found")
        
        return self._agents[agent_id]
    
    def get_all_agents(self) -> Dict[str, AgentInfo]:
        """Get all registered agents."""
        return self._agents.copy()
    
    def get_agents_by_state(self, state: AgentState) -> List[AgentInfo]:
        """Get all agents in a specific state."""
        return [agent for agent in self._agents.values() if agent.state == state]
    
    def get_agents_by_type(self, agent_type: str) -> List[AgentInfo]:
        """Get all agents of a specific type."""
        return [agent for agent in self._agents.values() if agent.agent_type == agent_type]
    
    def update_metrics(self, agent_id: str, metrics: AgentMetrics) -> None:
        """
        Update agent metrics.
        
        Args:
            agent_id: Agent identifier
            metrics: New metrics
            
        Raises:
            AgentError: If agent not found
        """
        if agent_id not in self._agents:
            raise AgentError(f"Agent {agent_id} not found")
        
        self._agents[agent_id].metrics = metrics
        self._agents[agent_id].metrics.last_updated = datetime.now()
        
        self.logger.debug(f"Updated metrics for agent {agent_id}")
    
    def get_state_history(self, agent_id: str) -> List[tuple]:
        """
        Get state transition history for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of (state, timestamp) tuples
            
        Raises:
            AgentError: If agent not found
        """
        if agent_id not in self._agents:
            raise AgentError(f"Agent {agent_id} not found")
        
        return self._state_history[agent_id].copy()


@dataclass
class ScheduledTask:
    """Scheduled task information."""
    task_id: str
    agent_id: str
    task_func: Callable
    schedule: str  # cron-like: "0 9 * * *" or interval: "every 1h"
    next_run: datetime
    last_run: Optional[datetime] = None
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'schedule': self.schedule,
            'next_run': self.next_run.isoformat(),
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'enabled': self.enabled
        }


class AgentScheduler:
    """
    Task scheduler for agents.
    
    Supports cron-like scheduling and interval-based scheduling.
    
    Example:
        >>> scheduler = AgentScheduler()
        >>> scheduler.schedule_task('task_1', 'agent_1', train_func, 'every 1h')
        >>> await scheduler.start()
    """
    
    def __init__(self):
        """Initialize the scheduler."""
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self.logger = get_logger(__name__)
    
    def schedule_task(
        self,
        task_id: str,
        agent_id: str,
        task_func: Callable,
        schedule: str,
        start_time: Optional[datetime] = None
    ) -> ScheduledTask:
        """
        Schedule a task.
        
        Args:
            task_id: Unique task identifier
            agent_id: Agent identifier
            task_func: Async function to execute
            schedule: Schedule string (e.g., "every 1h", "every 30m")
            start_time: First execution time (default: now)
            
        Returns:
            ScheduledTask object
            
        Raises:
            ValueError: If task already exists
        """
        if task_id in self._tasks:
            raise ValueError(f"Task {task_id} already exists")
        
        next_run = start_time or datetime.now()
        
        task = ScheduledTask(
            task_id=task_id,
            agent_id=agent_id,
            task_func=task_func,
            schedule=schedule,
            next_run=next_run
        )
        
        self._tasks[task_id] = task
        self.logger.info(f"Scheduled task {task_id} for agent {agent_id}: {schedule}")
        
        return task
    
    def unschedule_task(self, task_id: str) -> None:
        """
        Unschedule a task.
        
        Args:
            task_id: Task identifier
            
        Raises:
            ValueError: If task not found
        """
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        
        del self._tasks[task_id]
        self.logger.info(f"Unscheduled task {task_id}")
    
    def enable_task(self, task_id: str) -> None:
        """Enable a task."""
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        
        self._tasks[task_id].enabled = True
        self.logger.info(f"Enabled task {task_id}")
    
    def disable_task(self, task_id: str) -> None:
        """Disable a task."""
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        
        self._tasks[task_id].enabled = False
        self.logger.info(f"Disabled task {task_id}")
    
    def get_task(self, task_id: str) -> ScheduledTask:
        """Get task information."""
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        
        return self._tasks[task_id]
    
    def get_all_tasks(self) -> Dict[str, ScheduledTask]:
        """Get all scheduled tasks."""
        return self._tasks.copy()
    
    def _parse_interval(self, schedule: str) -> timedelta:
        """
        Parse interval string to timedelta.
        
        Args:
            schedule: Interval string (e.g., "every 1h", "every 30m")
            
        Returns:
            timedelta object
        """
        # Simple parser for "every Xh" or "every Xm"
        if schedule.startswith("every "):
            interval_str = schedule.replace("every ", "").strip()
            
            if interval_str.endswith("h"):
                hours = int(interval_str[:-1])
                return timedelta(hours=hours)
            elif interval_str.endswith("m"):
                minutes = int(interval_str[:-1])
                return timedelta(minutes=minutes)
            elif interval_str.endswith("d"):
                days = int(interval_str[:-1])
                return timedelta(days=days)
        
        raise ValueError(f"Invalid schedule format: {schedule}")
    
    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            self.logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        self.logger.info("Scheduler started")
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return
        
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Scheduler stopped")
    
    async def _run_scheduler(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.now()
                
                # Check all tasks
                for task in list(self._tasks.values()):
                    if not task.enabled:
                        continue
                    
                    # Check if task should run
                    if now >= task.next_run:
                        # Execute task
                        try:
                            self.logger.info(f"Executing task {task.task_id} for agent {task.agent_id}")
                            
                            if asyncio.iscoroutinefunction(task.task_func):
                                await task.task_func()
                            else:
                                task.task_func()
                            
                            task.last_run = now
                            
                            # Calculate next run time
                            interval = self._parse_interval(task.schedule)
                            task.next_run = now + interval
                            
                            self.logger.info(
                                f"Task {task.task_id} completed. Next run: {task.next_run}"
                            )
                        
                        except Exception as e:
                            self.logger.error(
                                f"Error executing task {task.task_id}: {e}",
                                exc_info=True
                            )
                
                # Sleep for 1 second before next check
                await asyncio.sleep(1)
            
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}", exc_info=True)
                await asyncio.sleep(5)


class AgentRegistry:
    """
    Central registry for active agents with health monitoring.
    
    Tracks all active agents and provides health status monitoring.
    
    Example:
        >>> registry = AgentRegistry()
        >>> registry.register('agent_1', agent_instance)
        >>> health = registry.get_health_status('agent_1')
    """
    
    def __init__(self, lifecycle_manager: AgentLifecycleManager):
        """
        Initialize the registry.
        
        Args:
            lifecycle_manager: AgentLifecycleManager instance
        """
        self._agents: Dict[str, Any] = {}  # agent_id → agent instance
        self._lifecycle_manager = lifecycle_manager
        self._last_heartbeat: Dict[str, datetime] = {}
        self.logger = get_logger(__name__)
    
    def register(self, agent_id: str, agent_instance: Any) -> None:
        """
        Register an agent instance.
        
        Args:
            agent_id: Agent identifier
            agent_instance: Agent instance
            
        Raises:
            AgentError: If agent not in lifecycle manager
        """
        # Verify agent exists in lifecycle manager
        try:
            self._lifecycle_manager.get_agent_info(agent_id)
        except AgentError:
            raise AgentError(
                f"Agent {agent_id} not found in lifecycle manager. "
                "Register with lifecycle manager first."
            )
        
        self._agents[agent_id] = agent_instance
        self._last_heartbeat[agent_id] = datetime.now()
        
        self.logger.info(f"Registered agent instance {agent_id}")
    
    def unregister(self, agent_id: str) -> None:
        """
        Unregister an agent instance.
        
        Args:
            agent_id: Agent identifier
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            del self._last_heartbeat[agent_id]
            self.logger.info(f"Unregistered agent instance {agent_id}")
    
    def get_agent(self, agent_id: str) -> Any:
        """
        Get agent instance.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance
            
        Raises:
            AgentError: If agent not found
        """
        if agent_id not in self._agents:
            raise AgentError(f"Agent instance {agent_id} not found in registry")
        
        return self._agents[agent_id]
    
    def get_all_agents(self) -> Dict[str, Any]:
        """Get all registered agent instances."""
        return self._agents.copy()
    
    def heartbeat(self, agent_id: str) -> None:
        """
        Record agent heartbeat.
        
        Args:
            agent_id: Agent identifier
        """
        if agent_id in self._agents:
            self._last_heartbeat[agent_id] = datetime.now()
    
    def get_health_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent health status.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Health status dictionary
            
        Raises:
            AgentError: If agent not found
        """
        if agent_id not in self._agents:
            raise AgentError(f"Agent {agent_id} not found in registry")
        
        agent_info = self._lifecycle_manager.get_agent_info(agent_id)
        last_heartbeat = self._last_heartbeat.get(agent_id)
        
        # Check if agent is healthy (heartbeat within last 60 seconds)
        is_healthy = False
        if last_heartbeat:
            time_since_heartbeat = (datetime.now() - last_heartbeat).total_seconds()
            is_healthy = time_since_heartbeat < 60
        
        return {
            'agent_id': agent_id,
            'state': agent_info.state.value,
            'is_healthy': is_healthy,
            'last_heartbeat': last_heartbeat.isoformat() if last_heartbeat else None,
            'uptime_seconds': (datetime.now() - agent_info.created_at).total_seconds(),
            'error_message': agent_info.error_message
        }
    
    def get_all_health_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all agents."""
        return {
            agent_id: self.get_health_status(agent_id)
            for agent_id in self._agents.keys()
        }


# Convenience function to create a complete agent management system
def create_agent_management_system() -> tuple:
    """
    Create a complete agent management system.
    
    Returns:
        Tuple of (lifecycle_manager, scheduler, registry)
    """
    lifecycle_manager = AgentLifecycleManager()
    scheduler = AgentScheduler()
    registry = AgentRegistry(lifecycle_manager)
    
    return lifecycle_manager, scheduler, registry



