"""
Comprehensive tests for AgentManager to achieve >85% coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import json

from mtquant.agents.agent_manager import (
    AgentLifecycleManager, AgentScheduler, AgentRegistry,
    AgentState, AgentInfo, AgentMetrics
)
from mtquant.agents.hierarchical.forex_specialist import ForexSpecialist
from mtquant.agents.hierarchical.commodities_specialist import CommoditiesSpecialist
from mtquant.agents.hierarchical.equity_specialist import EquitySpecialist
from mtquant.utils.exceptions import StateTransitionError, AgentError


class TestAgentLifecycleManagerComprehensive:
    """Comprehensive tests for AgentLifecycleManager."""
    
    @pytest.fixture
    def sample_agent_info(self):
        """Sample agent information."""
        return AgentInfo(
            agent_id="test_forex_001",
            agent_type="forex_specialist",
            symbol="EURUSD",
            config={
                "instruments": ["EURUSD", "GBPUSD", "USDJPY"],
                "initial_capital": 100000.0,
                "risk_limits": {
                    "max_position_size": 0.1,
                    "max_daily_loss": 0.05,
                    "stop_loss_pct": 0.02
                },
                "trading_params": {
                    "transaction_cost": 0.003,
                    "slippage": 0.0001
                }
            }
        )
    
    @pytest.fixture
    def lifecycle_manager(self):
        """Create AgentLifecycleManager instance."""
        return AgentLifecycleManager()
    
    def test_lifecycle_manager_initialization(self, lifecycle_manager):
        """Test AgentLifecycleManager initialization."""
        assert lifecycle_manager.agents == {}
        assert lifecycle_manager.agent_registry is not None
        assert lifecycle_manager.scheduler is not None
        assert lifecycle_manager.logger is not None
    
    def test_register_agent_success(self, lifecycle_manager, sample_agent_info):
        """Test successful agent registration."""
        agent = lifecycle_manager.register_agent(
            agent_id="test_forex_001",
            agent_type="forex_specialist",
            symbol="EURUSD",
            config=sample_agent_info.config
        )
        
        assert agent is not None
        assert agent.agent_id == "test_forex_001"
        assert agent.state == AgentState.INITIALIZED
        assert "test_forex_001" in lifecycle_manager._agents
    
    @pytest.mark.asyncio
    async def test_create_agent_duplicate_id(self, lifecycle_manager, sample_agent_config):
        """Test creating agent with duplicate ID."""
        # Create first agent
        await lifecycle_manager.create_agent(sample_agent_config)
        
        # Try to create second agent with same ID
        with pytest.raises(AgentError):
            await lifecycle_manager.create_agent(sample_agent_config)
    
    @pytest.mark.asyncio
    async def test_start_agent_success(self, lifecycle_manager, sample_agent_config):
        """Test successful agent start."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        
        await lifecycle_manager.start_agent("test_forex_001")
        
        assert agent.state == AgentState.RUNNING
    
    @pytest.mark.asyncio
    async def test_start_agent_not_found(self, lifecycle_manager):
        """Test starting non-existent agent."""
        with pytest.raises(AgentError):
            await lifecycle_manager.start_agent("non_existent")
    
    @pytest.mark.asyncio
    async def test_start_agent_invalid_state(self, lifecycle_manager, sample_agent_config):
        """Test starting agent in invalid state."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        agent.state = AgentState.ERROR  # Set to error state
        
        with pytest.raises(StateTransitionError):
            await lifecycle_manager.start_agent("test_forex_001")
    
    @pytest.mark.asyncio
    async def test_stop_agent_success(self, lifecycle_manager, sample_agent_config):
        """Test successful agent stop."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        await lifecycle_manager.start_agent("test_forex_001")
        
        await lifecycle_manager.stop_agent("test_forex_001")
        
        assert agent.state == AgentState.STOPPED
    
    @pytest.mark.asyncio
    async def test_pause_agent_success(self, lifecycle_manager, sample_agent_config):
        """Test successful agent pause."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        await lifecycle_manager.start_agent("test_forex_001")
        
        await lifecycle_manager.pause_agent("test_forex_001")
        
        assert agent.state == AgentState.PAUSED
    
    @pytest.mark.asyncio
    async def test_resume_agent_success(self, lifecycle_manager, sample_agent_config):
        """Test successful agent resume."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        await lifecycle_manager.start_agent("test_forex_001")
        await lifecycle_manager.pause_agent("test_forex_001")
        
        await lifecycle_manager.resume_agent("test_forex_001")
        
        assert agent.state == AgentState.RUNNING
    
    @pytest.mark.asyncio
    async def test_restart_agent_success(self, lifecycle_manager, sample_agent_config):
        """Test successful agent restart."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        await lifecycle_manager.start_agent("test_forex_001")
        
        await lifecycle_manager.restart_agent("test_forex_001")
        
        assert agent.state == AgentState.RUNNING
    
    @pytest.mark.asyncio
    async def test_remove_agent_success(self, lifecycle_manager, sample_agent_config):
        """Test successful agent removal."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        
        await lifecycle_manager.remove_agent("test_forex_001")
        
        assert "test_forex_001" not in lifecycle_manager.agents
        assert agent.state == AgentState.REMOVED
    
    @pytest.mark.asyncio
    async def test_get_agent_success(self, lifecycle_manager, sample_agent_config):
        """Test getting existing agent."""
        created_agent = await lifecycle_manager.create_agent(sample_agent_config)
        
        retrieved_agent = await lifecycle_manager.get_agent("test_forex_001")
        
        assert retrieved_agent == created_agent
    
    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, lifecycle_manager):
        """Test getting non-existent agent."""
        with pytest.raises(AgentError):
            await lifecycle_manager.get_agent("non_existent")
    
    @pytest.mark.asyncio
    async def test_list_agents(self, lifecycle_manager, sample_agent_config):
        """Test listing all agents."""
        # Create multiple agents
        config1 = sample_agent_config
        config2 = AgentConfig(
            agent_id="test_commodities_001",
            agent_type="commodities_specialist",
            instruments=["XAUUSD", "WTIUSD"],
            initial_capital=100000.0,
            risk_limits={},
            trading_params={}
        )
        
        await lifecycle_manager.create_agent(config1)
        await lifecycle_manager.create_agent(config2)
        
        agents = await lifecycle_manager.list_agents()
        
        assert len(agents) == 2
        assert "test_forex_001" in agents
        assert "test_commodities_001" in agents
    
    @pytest.mark.asyncio
    async def test_get_agent_status(self, lifecycle_manager, sample_agent_config):
        """Test getting agent status."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        await lifecycle_manager.start_agent("test_forex_001")
        
        status = await lifecycle_manager.get_agent_status("test_forex_001")
        
        assert status["agent_id"] == "test_forex_001"
        assert status["state"] == AgentState.RUNNING
        assert "uptime" in status
        assert "performance" in status
    
    @pytest.mark.asyncio
    async def test_update_agent_config(self, lifecycle_manager, sample_agent_config):
        """Test updating agent configuration."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        
        new_config = AgentConfig(
            agent_id="test_forex_001",
            agent_type="forex_specialist",
            instruments=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
            initial_capital=150000.0,
            risk_limits={
                "max_position_size": 0.15,
                "max_daily_loss": 0.03,
                "stop_loss_pct": 0.015
            },
            trading_params={
                "transaction_cost": 0.002,
                "slippage": 0.0002
            }
        )
        
        await lifecycle_manager.update_agent_config("test_forex_001", new_config)
        
        assert agent.config.instruments == ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        assert agent.config.initial_capital == 150000.0
    
    @pytest.mark.asyncio
    async def test_health_check_all_agents(self, lifecycle_manager, sample_agent_config):
        """Test health check for all agents."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        await lifecycle_manager.start_agent("test_forex_001")
        
        health_status = await lifecycle_manager.health_check()
        
        assert isinstance(health_status, dict)
        assert "test_forex_001" in health_status
        assert health_status["test_forex_001"]["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_emergency_stop_all_agents(self, lifecycle_manager, sample_agent_config):
        """Test emergency stop for all agents."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        await lifecycle_manager.start_agent("test_forex_001")
        
        await lifecycle_manager.emergency_stop()
        
        assert agent.state == AgentState.EMERGENCY_STOPPED
    
    @pytest.mark.asyncio
    async def test_get_system_metrics(self, lifecycle_manager, sample_agent_config):
        """Test getting system metrics."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        await lifecycle_manager.start_agent("test_forex_001")
        
        metrics = await lifecycle_manager.get_system_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_agents" in metrics
        assert "running_agents" in metrics
        assert "system_uptime" in metrics
        assert metrics["total_agents"] == 1
        assert metrics["running_agents"] == 1
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, lifecycle_manager, sample_agent_config):
        """Test agent error handling."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        await lifecycle_manager.start_agent("test_forex_001")
        
        # Simulate agent error
        agent.state = AgentState.ERROR
        agent.error_message = "Test error"
        
        status = await lifecycle_manager.get_agent_status("test_forex_001")
        
        assert status["state"] == AgentState.ERROR
        assert "error_message" in status
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, lifecycle_manager, sample_agent_config):
        """Test concurrent operations on agents."""
        agent = await lifecycle_manager.create_agent(sample_agent_config)
        
        # Start multiple concurrent operations
        tasks = [
            lifecycle_manager.start_agent("test_forex_001"),
            lifecycle_manager.pause_agent("test_forex_001"),
            lifecycle_manager.get_agent_status("test_forex_001")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least one operation should succeed
        assert len([r for r in results if not isinstance(r, Exception)]) >= 1


class TestAgentSchedulerComprehensive:
    """Comprehensive tests for AgentScheduler."""
    
    @pytest.fixture
    def scheduler_config(self):
        """Sample scheduler configuration."""
        return SchedulingConfig(
            max_concurrent_agents=5,
            task_timeout=30.0,
            retry_attempts=3,
            retry_delay=1.0
        )
    
    @pytest.fixture
    def scheduler(self, scheduler_config):
        """Create AgentScheduler instance."""
        return AgentScheduler(scheduler_config)
    
    def test_scheduler_initialization(self, scheduler, scheduler_config):
        """Test AgentScheduler initialization."""
        assert scheduler.config == scheduler_config
        assert scheduler.active_tasks == {}
        assert scheduler.task_queue is not None
        assert scheduler.logger is not None
    
    @pytest.mark.asyncio
    async def test_schedule_task_success(self, scheduler):
        """Test successful task scheduling."""
        async def test_task():
            return "task_result"
        
        task_id = await scheduler.schedule_task(test_task, priority=1)
        
        assert task_id is not None
        assert task_id in scheduler.active_tasks
    
    @pytest.mark.asyncio
    async def test_schedule_task_with_timeout(self, scheduler):
        """Test task scheduling with timeout."""
        async def slow_task():
            await asyncio.sleep(2.0)
            return "slow_result"
        
        # Set short timeout
        scheduler.config.task_timeout = 0.5
        
        task_id = await scheduler.schedule_task(slow_task, priority=1)
        
        # Wait for timeout
        await asyncio.sleep(1.0)
        
        # Task should be cancelled due to timeout
        assert task_id not in scheduler.active_tasks
    
    @pytest.mark.asyncio
    async def test_cancel_task_success(self, scheduler):
        """Test successful task cancellation."""
        async def test_task():
            await asyncio.sleep(1.0)
            return "task_result"
        
        task_id = await scheduler.schedule_task(test_task, priority=1)
        
        success = await scheduler.cancel_task(task_id)
        
        assert success is True
        assert task_id not in scheduler.active_tasks
    
    @pytest.mark.asyncio
    async def test_cancel_task_not_found(self, scheduler):
        """Test cancelling non-existent task."""
        success = await scheduler.cancel_task("non_existent_task")
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, scheduler):
        """Test getting task status."""
        async def test_task():
            await asyncio.sleep(0.1)
            return "task_result"
        
        task_id = await scheduler.schedule_task(test_task, priority=1)
        
        status = await scheduler.get_task_status(task_id)
        
        assert status["task_id"] == task_id
        assert "status" in status
        assert "created_at" in status
    
    @pytest.mark.asyncio
    async def test_list_active_tasks(self, scheduler):
        """Test listing active tasks."""
        async def test_task():
            await asyncio.sleep(0.1)
            return "task_result"
        
        task_id1 = await scheduler.schedule_task(test_task, priority=1)
        task_id2 = await scheduler.schedule_task(test_task, priority=2)
        
        active_tasks = await scheduler.list_active_tasks()
        
        assert len(active_tasks) == 2
        assert task_id1 in active_tasks
        assert task_id2 in active_tasks
    
    @pytest.mark.asyncio
    async def test_priority_scheduling(self, scheduler):
        """Test priority-based task scheduling."""
        results = []
        
        async def low_priority_task():
            results.append("low")
            return "low_result"
        
        async def high_priority_task():
            results.append("high")
            return "high_result"
        
        # Schedule low priority first
        await scheduler.schedule_task(low_priority_task, priority=1)
        await scheduler.schedule_task(high_priority_task, priority=5)
        
        # Wait for tasks to complete
        await asyncio.sleep(0.5)
        
        # High priority should execute first
        assert results[0] == "high"
        assert results[1] == "low"
    
    @pytest.mark.asyncio
    async def test_max_concurrent_agents_limit(self, scheduler):
        """Test maximum concurrent agents limit."""
        # Set low limit
        scheduler.config.max_concurrent_agents = 2
        
        async def long_task():
            await asyncio.sleep(1.0)
            return "long_result"
        
        # Schedule more tasks than limit
        task_ids = []
        for i in range(5):
            task_id = await scheduler.schedule_task(long_task, priority=1)
            task_ids.append(task_id)
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Only 2 tasks should be active
        active_tasks = await scheduler.list_active_tasks()
        assert len(active_tasks) <= 2


class TestAgentRegistryComprehensive:
    """Comprehensive tests for AgentRegistry."""
    
    @pytest.fixture
    def agent_registry(self):
        """Create AgentRegistry instance."""
        return AgentRegistry()
    
    def test_registry_initialization(self, agent_registry):
        """Test AgentRegistry initialization."""
        assert agent_registry.agents == {}
        assert agent_registry.agent_types == {}
        assert agent_registry.logger is not None
    
    def test_register_agent_success(self, agent_registry):
        """Test successful agent registration."""
        agent = Mock()
        agent.agent_id = "test_agent_001"
        agent.agent_type = "forex_specialist"
        
        agent_registry.register_agent(agent)
        
        assert "test_agent_001" in agent_registry.agents
        assert agent_registry.agents["test_agent_001"] == agent
        assert "forex_specialist" in agent_registry.agent_types
        assert "test_agent_001" in agent_registry.agent_types["forex_specialist"]
    
    def test_register_agent_duplicate_id(self, agent_registry):
        """Test registering agent with duplicate ID."""
        agent1 = Mock()
        agent1.agent_id = "test_agent_001"
        agent1.agent_type = "forex_specialist"
        
        agent2 = Mock()
        agent2.agent_id = "test_agent_001"
        agent2.agent_type = "commodities_specialist"
        
        agent_registry.register_agent(agent1)
        
        with pytest.raises(AgentError):
            agent_registry.register_agent(agent2)
    
    def test_unregister_agent_success(self, agent_registry):
        """Test successful agent unregistration."""
        agent = Mock()
        agent.agent_id = "test_agent_001"
        agent.agent_type = "forex_specialist"
        
        agent_registry.register_agent(agent)
        agent_registry.unregister_agent("test_agent_001")
        
        assert "test_agent_001" not in agent_registry.agents
        assert "test_agent_001" not in agent_registry.agent_types["forex_specialist"]
    
    def test_unregister_agent_not_found(self, agent_registry):
        """Test unregistering non-existent agent."""
        with pytest.raises(AgentError):
            agent_registry.unregister_agent("non_existent")
    
    def test_get_agent_success(self, agent_registry):
        """Test getting existing agent."""
        agent = Mock()
        agent.agent_id = "test_agent_001"
        agent.agent_type = "forex_specialist"
        
        agent_registry.register_agent(agent)
        
        retrieved_agent = agent_registry.get_agent("test_agent_001")
        
        assert retrieved_agent == agent
    
    def test_get_agent_not_found(self, agent_registry):
        """Test getting non-existent agent."""
        with pytest.raises(AgentError):
            agent_registry.get_agent("non_existent")
    
    def test_list_agents_by_type(self, agent_registry):
        """Test listing agents by type."""
        # Register agents of different types
        forex_agent = Mock()
        forex_agent.agent_id = "forex_001"
        forex_agent.agent_type = "forex_specialist"
        
        commodities_agent = Mock()
        commodities_agent.agent_id = "commodities_001"
        commodities_agent.agent_type = "commodities_specialist"
        
        agent_registry.register_agent(forex_agent)
        agent_registry.register_agent(commodities_agent)
        
        forex_agents = agent_registry.list_agents_by_type("forex_specialist")
        commodities_agents = agent_registry.list_agents_by_type("commodities_specialist")
        
        assert len(forex_agents) == 1
        assert forex_agents[0] == forex_agent
        assert len(commodities_agents) == 1
        assert commodities_agents[0] == commodities_agent
    
    def test_list_all_agents(self, agent_registry):
        """Test listing all agents."""
        # Register multiple agents
        for i in range(3):
            agent = Mock()
            agent.agent_id = f"test_agent_{i:03d}"
            agent.agent_type = "forex_specialist"
            agent_registry.register_agent(agent)
        
        all_agents = agent_registry.list_all_agents()
        
        assert len(all_agents) == 3
        for i in range(3):
            assert f"test_agent_{i:03d}" in [a.agent_id for a in all_agents]
    
    def test_get_agent_count_by_type(self, agent_registry):
        """Test getting agent count by type."""
        # Register agents of different types
        for i in range(2):
            agent = Mock()
            agent.agent_id = f"forex_{i:03d}"
            agent.agent_type = "forex_specialist"
            agent_registry.register_agent(agent)
        
        for i in range(3):
            agent = Mock()
            agent.agent_id = f"commodities_{i:03d}"
            agent.agent_type = "commodities_specialist"
            agent_registry.register_agent(agent)
        
        forex_count = agent_registry.get_agent_count_by_type("forex_specialist")
        commodities_count = agent_registry.get_agent_count_by_type("commodities_specialist")
        
        assert forex_count == 2
        assert commodities_count == 3
    
    def test_get_registry_stats(self, agent_registry):
        """Test getting registry statistics."""
        # Register agents of different types
        for i in range(2):
            agent = Mock()
            agent.agent_id = f"forex_{i:03d}"
            agent.agent_type = "forex_specialist"
            agent_registry.register_agent(agent)
        
        for i in range(3):
            agent = Mock()
            agent.agent_id = f"commodities_{i:03d}"
            agent.agent_type = "commodities_specialist"
            agent_registry.register_agent(agent)
        
        stats = agent_registry.get_registry_stats()
        
        assert stats["total_agents"] == 5
        assert stats["agent_types"] == 2
        assert stats["forex_specialist"] == 2
        assert stats["commodities_specialist"] == 3
