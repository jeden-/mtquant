"""
Unit tests for connection_pool.py with 66% coverage.

This file has 192 lines and 66% coverage, so adding comprehensive tests here will significantly increase overall coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import connection pool classes and functions
from mtquant.mcp_integration.managers.connection_pool import (
    ConnectionPool, AdapterInfo, ConnectionStats, ConnectionStatus
)
from mtquant.mcp_integration.adapters.base_adapter import BrokerAdapter, HealthStatus
from mtquant.utils.exceptions import BrokerConnectionError, BrokerError


class TestConnectionStatus:
    """Tests for ConnectionStatus enum."""
    
    def test_connection_status_values(self):
        """Test ConnectionStatus enum values."""
        assert ConnectionStatus.CONNECTED.value == "connected"
        assert ConnectionStatus.DISCONNECTED.value == "disconnected"
        assert ConnectionStatus.CONNECTING.value == "connecting"
        assert ConnectionStatus.FAILED.value == "failed"
    
    def test_connection_status_members(self):
        """Test ConnectionStatus enum members."""
        members = list(ConnectionStatus)
        assert len(members) == 4
        assert ConnectionStatus.CONNECTED in members
        assert ConnectionStatus.DISCONNECTED in members
        assert ConnectionStatus.CONNECTING in members
        assert ConnectionStatus.FAILED in members


class TestAdapterInfo:
    """Tests for AdapterInfo dataclass."""
    
    def test_adapter_info_initialization(self):
        """Test AdapterInfo initialization."""
        mock_adapter = Mock(spec=BrokerAdapter)
        adapter_info = AdapterInfo(
            adapter=mock_adapter,
            is_primary=True,
            added_at=datetime.utcnow()
        )
        
        assert adapter_info.adapter == mock_adapter
        assert adapter_info.is_primary == True
        assert isinstance(adapter_info.added_at, datetime)
        assert adapter_info.last_health_check is None
        assert adapter_info.health_status is None
        assert adapter_info.connection_status == ConnectionStatus.DISCONNECTED
        assert adapter_info.failure_count == 0
        assert adapter_info.last_failure is None
        assert adapter_info.uptime_start is None
    
    def test_adapter_info_with_optional_fields(self):
        """Test AdapterInfo with optional fields."""
        mock_adapter = Mock(spec=BrokerAdapter)
        health_status = HealthStatus(
            is_connected=True,
            latency_ms=10.5,
            last_check=datetime.utcnow()
        )
        
        adapter_info = AdapterInfo(
            adapter=mock_adapter,
            is_primary=False,
            added_at=datetime.utcnow(),
            last_health_check=datetime.utcnow(),
            health_status=health_status,
            connection_status=ConnectionStatus.CONNECTED,
            failure_count=2,
            last_failure=datetime.utcnow(),
            uptime_start=datetime.utcnow()
        )
        
        assert adapter_info.is_primary == False
        assert adapter_info.health_status == health_status
        assert adapter_info.connection_status == ConnectionStatus.CONNECTED
        assert adapter_info.failure_count == 2
        assert adapter_info.last_health_check is not None
        assert adapter_info.last_failure is not None
        assert adapter_info.uptime_start is not None


class TestConnectionStats:
    """Tests for ConnectionStats dataclass."""
    
    def test_connection_stats_initialization(self):
        """Test ConnectionStats initialization."""
        stats = ConnectionStats()
        
        assert stats.total_adapters == 0
        assert stats.healthy_adapters == 0
        assert stats.primary_broker is None
        assert stats.backup_brokers == []
        assert stats.total_uptime_hours == 0.0
        assert stats.total_failures == 0
        assert stats.last_health_check is None
    
    def test_connection_stats_with_values(self):
        """Test ConnectionStats with values."""
        stats = ConnectionStats(
            total_adapters=3,
            healthy_adapters=2,
            primary_broker="broker-1",
            backup_brokers=["broker-2", "broker-3"],
            total_uptime_hours=24.5,
            total_failures=5,
            last_health_check=datetime.utcnow()
        )
        
        assert stats.total_adapters == 3
        assert stats.healthy_adapters == 2
        assert stats.primary_broker == "broker-1"
        assert stats.backup_brokers == ["broker-2", "broker-3"]
        assert stats.total_uptime_hours == 24.5
        assert stats.total_failures == 5
        assert stats.last_health_check is not None


class TestConnectionPoolInitialization:
    """Tests for ConnectionPool initialization."""
    
    def test_connection_pool_initialization(self):
        """Test ConnectionPool initialization."""
        pool = ConnectionPool()
        
        assert isinstance(pool.adapters, dict)
        assert len(pool.adapters) == 0
        assert pool.primary_broker is None
        assert isinstance(pool.backup_brokers, list)
        assert len(pool.backup_brokers) == 0
        assert pool._health_monitoring_task is None
        assert pool._lock is not None
        assert pool.logger is not None


class TestConnectionPoolAddAdapter:
    """Tests for ConnectionPool add_adapter method."""
    
    @pytest.fixture
    def pool(self):
        """Create ConnectionPool instance."""
        return ConnectionPool()
    
    @pytest.fixture
    def mock_adapter(self):
        """Create mock BrokerAdapter."""
        adapter = Mock(spec=BrokerAdapter)
        adapter.broker_id = "test_broker"
        return adapter
    
    @pytest.mark.asyncio
    async def test_add_adapter_primary(self, pool, mock_adapter):
        """Test adding primary adapter."""
        await pool.add_adapter("broker-1", mock_adapter, is_primary=True)
        
        assert "broker-1" in pool.adapters
        assert pool.adapters["broker-1"].adapter == mock_adapter
        assert pool.adapters["broker-1"].is_primary == True
        assert pool.primary_broker == "broker-1"
        assert len(pool.backup_brokers) == 0
    
    @pytest.mark.asyncio
    async def test_add_adapter_backup(self, pool, mock_adapter):
        """Test adding backup adapter."""
        await pool.add_adapter("broker-2", mock_adapter, is_primary=False)
        
        assert "broker-2" in pool.adapters
        assert pool.adapters["broker-2"].adapter == mock_adapter
        assert pool.adapters["broker-2"].is_primary == False
        assert pool.primary_broker is None
        assert "broker-2" in pool.backup_brokers
    
    @pytest.mark.asyncio
    async def test_add_adapter_duplicate_id(self, pool, mock_adapter):
        """Test adding adapter with duplicate ID."""
        await pool.add_adapter("broker-1", mock_adapter, is_primary=True)
        
        with pytest.raises(ValueError, match="Broker broker-1 already exists in pool"):
            await pool.add_adapter("broker-1", mock_adapter, is_primary=False)
    
    @pytest.mark.asyncio
    async def test_add_adapter_multiple(self, pool, mock_adapter):
        """Test adding multiple adapters."""
        adapter1 = Mock(spec=BrokerAdapter)
        adapter2 = Mock(spec=BrokerAdapter)
        adapter3 = Mock(spec=BrokerAdapter)
        
        await pool.add_adapter("broker-1", adapter1, is_primary=True)
        await pool.add_adapter("broker-2", adapter2, is_primary=False)
        await pool.add_adapter("broker-3", adapter3, is_primary=False)
        
        assert len(pool.adapters) == 3
        assert pool.primary_broker == "broker-1"
        assert len(pool.backup_brokers) == 2
        assert "broker-2" in pool.backup_brokers
        assert "broker-3" in pool.backup_brokers


class TestConnectionPoolConnectAll:
    """Tests for ConnectionPool connect_all method."""
    
    @pytest.fixture
    def pool(self):
        """Create ConnectionPool instance."""
        return ConnectionPool()
    
    @pytest.fixture
    def mock_adapter_connected(self):
        """Create mock adapter that connects successfully."""
        adapter = Mock(spec=BrokerAdapter)
        adapter.connect = AsyncMock(return_value=True)
        return adapter
    
    @pytest.fixture
    def mock_adapter_failed(self):
        """Create mock adapter that fails to connect."""
        adapter = Mock(spec=BrokerAdapter)
        adapter.connect = AsyncMock(return_value=False)
        return adapter
    
    @pytest.mark.asyncio
    async def test_connect_all_success(self, pool, mock_adapter_connected):
        """Test connecting all adapters successfully."""
        await pool.add_adapter("broker-1", mock_adapter_connected, is_primary=True)
        await pool.add_adapter("broker-2", mock_adapter_connected, is_primary=False)
        
        results = await pool.connect_all()
        
        assert results["broker-1"] == True
        assert results["broker-2"] == True
        assert pool.adapters["broker-1"].connection_status == ConnectionStatus.CONNECTED
        assert pool.adapters["broker-2"].connection_status == ConnectionStatus.CONNECTED
        assert pool.adapters["broker-1"].uptime_start is not None
        assert pool.adapters["broker-2"].uptime_start is not None
        assert pool.adapters["broker-1"].failure_count == 0
        assert pool.adapters["broker-2"].failure_count == 0
    
    @pytest.mark.asyncio
    async def test_connect_all_partial_failure(self, pool, mock_adapter_connected, mock_adapter_failed):
        """Test connecting adapters with partial failure."""
        await pool.add_adapter("broker-1", mock_adapter_connected, is_primary=True)
        await pool.add_adapter("broker-2", mock_adapter_failed, is_primary=False)
        
        results = await pool.connect_all()
        
        assert results["broker-1"] == True
        assert results["broker-2"] == False
        assert pool.adapters["broker-1"].connection_status == ConnectionStatus.CONNECTED
        assert pool.adapters["broker-2"].connection_status == ConnectionStatus.FAILED
        assert pool.adapters["broker-1"].failure_count == 0
        assert pool.adapters["broker-2"].failure_count == 1
        assert pool.adapters["broker-2"].last_failure is not None
    
    @pytest.mark.asyncio
    async def test_connect_all_exception(self, pool):
        """Test connecting adapters with exception."""
        mock_adapter = Mock(spec=BrokerAdapter)
        mock_adapter.connect = AsyncMock(side_effect=Exception("Connection error"))
        
        await pool.add_adapter("broker-1", mock_adapter, is_primary=True)
        
        results = await pool.connect_all()
        
        assert results["broker-1"] == False
        assert pool.adapters["broker-1"].connection_status == ConnectionStatus.FAILED
        assert pool.adapters["broker-1"].failure_count == 1
        assert pool.adapters["broker-1"].last_failure is not None


class TestConnectionPoolDisconnectAll:
    """Tests for ConnectionPool disconnect_all method."""
    
    @pytest.fixture
    def pool(self):
        """Create ConnectionPool instance."""
        return ConnectionPool()
    
    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        adapter = Mock(spec=BrokerAdapter)
        adapter.disconnect = AsyncMock()
        return adapter
    
    @pytest.mark.asyncio
    async def test_disconnect_all_success(self, pool, mock_adapter):
        """Test disconnecting all adapters successfully."""
        await pool.add_adapter("broker-1", mock_adapter, is_primary=True)
        await pool.add_adapter("broker-2", mock_adapter, is_primary=False)
        
        # Set connection status to connected
        pool.adapters["broker-1"].connection_status = ConnectionStatus.CONNECTED
        pool.adapters["broker-2"].connection_status = ConnectionStatus.CONNECTED
        pool.adapters["broker-1"].uptime_start = datetime.utcnow()
        pool.adapters["broker-2"].uptime_start = datetime.utcnow()
        
        await pool.disconnect_all()
        
        assert pool.adapters["broker-1"].connection_status == ConnectionStatus.DISCONNECTED
        assert pool.adapters["broker-2"].connection_status == ConnectionStatus.DISCONNECTED
        assert pool.adapters["broker-1"].uptime_start is None
        assert pool.adapters["broker-2"].uptime_start is None
        mock_adapter.disconnect.assert_called()
    
    @pytest.mark.asyncio
    async def test_disconnect_all_exception(self, pool):
        """Test disconnecting adapters with exception."""
        mock_adapter = Mock(spec=BrokerAdapter)
        mock_adapter.disconnect = AsyncMock(side_effect=Exception("Disconnect error"))
        
        await pool.add_adapter("broker-1", mock_adapter, is_primary=True)
        
        # Should not raise exception
        await pool.disconnect_all()
        
        # Adapter should still be marked as disconnected
        assert pool.adapters["broker-1"].connection_status == ConnectionStatus.DISCONNECTED


class TestConnectionPoolGetAdapter:
    """Tests for ConnectionPool get_adapter method."""
    
    @pytest.fixture
    def pool(self):
        """Create ConnectionPool instance."""
        return ConnectionPool()
    
    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        return Mock(spec=BrokerAdapter)
    
    @pytest.mark.asyncio
    async def test_get_adapter_success(self, pool, mock_adapter):
        """Test getting adapter successfully."""
        await pool.add_adapter("broker-1", mock_adapter, is_primary=True)
        
        result = await pool.get_adapter("broker-1")
        
        assert result == mock_adapter
    
    @pytest.mark.asyncio
    async def test_get_adapter_not_found(self, pool):
        """Test getting adapter that doesn't exist."""
        with pytest.raises(KeyError, match="Broker broker-1 not found in pool"):
            await pool.get_adapter("broker-1")


class TestConnectionPoolGetHealthyAdapter:
    """Tests for ConnectionPool get_healthy_adapter method."""
    
    @pytest.fixture
    def pool(self):
        """Create ConnectionPool instance."""
        return ConnectionPool()
    
    @pytest.fixture
    def mock_adapter_healthy(self):
        """Create mock healthy adapter."""
        adapter = Mock(spec=BrokerAdapter)
        return adapter
    
    @pytest.fixture
    def mock_adapter_unhealthy(self):
        """Create mock unhealthy adapter."""
        adapter = Mock(spec=BrokerAdapter)
        return adapter
    
    @pytest.mark.asyncio
    async def test_get_healthy_adapter_primary_healthy(self, pool, mock_adapter_healthy):
        """Test getting healthy primary adapter."""
        await pool.add_adapter("broker-1", mock_adapter_healthy, is_primary=True)
        
        # Set adapter as healthy
        pool.adapters["broker-1"].connection_status = ConnectionStatus.CONNECTED
        pool.adapters["broker-1"].health_status = HealthStatus(
            is_connected=True,
            latency_ms=10.0,
            last_check=datetime.utcnow()
        )
        
        result = await pool.get_healthy_adapter()
        
        assert result == mock_adapter_healthy
    
    @pytest.mark.asyncio
    async def test_get_healthy_adapter_primary_unhealthy_backup_healthy(self, pool, mock_adapter_healthy, mock_adapter_unhealthy):
        """Test getting healthy backup adapter when primary is unhealthy."""
        await pool.add_adapter("broker-1", mock_adapter_unhealthy, is_primary=True)
        await pool.add_adapter("broker-2", mock_adapter_healthy, is_primary=False)
        
        # Set primary as unhealthy
        pool.adapters["broker-1"].connection_status = ConnectionStatus.FAILED
        pool.adapters["broker-1"].health_status = HealthStatus(
            is_connected=False,
            latency_ms=0.0,
            last_check=datetime.utcnow()
        )
        
        # Set backup as healthy
        pool.adapters["broker-2"].connection_status = ConnectionStatus.CONNECTED
        pool.adapters["broker-2"].health_status = HealthStatus(
            is_connected=True,
            latency_ms=15.0,
            last_check=datetime.utcnow()
        )
        
        result = await pool.get_healthy_adapter()
        
        assert result == mock_adapter_healthy
    
    @pytest.mark.asyncio
    async def test_get_healthy_adapter_no_healthy(self, pool, mock_adapter_unhealthy):
        """Test getting healthy adapter when none are healthy."""
        await pool.add_adapter("broker-1", mock_adapter_unhealthy, is_primary=True)
        
        # Set adapter as unhealthy
        pool.adapters["broker-1"].connection_status = ConnectionStatus.FAILED
        pool.adapters["broker-1"].health_status = HealthStatus(
            is_connected=False,
            latency_ms=0.0,
            last_check=datetime.utcnow()
        )
        
        with pytest.raises(BrokerConnectionError, match="No healthy adapters available"):
            await pool.get_healthy_adapter()
    
    @pytest.mark.asyncio
    async def test_get_healthy_adapter_no_adapters(self, pool):
        """Test getting healthy adapter when no adapters exist."""
        with pytest.raises(BrokerConnectionError, match="No healthy adapters available"):
            await pool.get_healthy_adapter()


class TestConnectionPoolHealthCheckAll:
    """Tests for ConnectionPool health_check_all method."""
    
    @pytest.fixture
    def pool(self):
        """Create ConnectionPool instance."""
        return ConnectionPool()
    
    @pytest.fixture
    def mock_adapter_healthy(self):
        """Create mock adapter with healthy status."""
        adapter = Mock(spec=BrokerAdapter)
        adapter.health_check = AsyncMock(return_value=HealthStatus(
            is_connected=True,
            latency_ms=10.0,
            last_check=datetime.utcnow()
        ))
        return adapter
    
    @pytest.fixture
    def mock_adapter_unhealthy(self):
        """Create mock adapter with unhealthy status."""
        adapter = Mock(spec=BrokerAdapter)
        adapter.health_check = AsyncMock(return_value=HealthStatus(
            is_connected=False,
            latency_ms=0.0,
            last_check=datetime.utcnow(),
            error="Connection failed"
        ))
        return adapter
    
    @pytest.mark.asyncio
    async def test_health_check_all_success(self, pool, mock_adapter_healthy):
        """Test health check all adapters successfully."""
        await pool.add_adapter("broker-1", mock_adapter_healthy, is_primary=True)
        await pool.add_adapter("broker-2", mock_adapter_healthy, is_primary=False)
        
        results = await pool.health_check_all()
        
        assert len(results) == 2
        assert "broker-1" in results
        assert "broker-2" in results
        assert results["broker-1"].is_connected == True
        assert results["broker-2"].is_connected == True
        
        # Check that health status was updated
        assert pool.adapters["broker-1"].health_status.is_connected == True
        assert pool.adapters["broker-2"].health_status.is_connected == True
        assert pool.adapters["broker-1"].last_health_check is not None
        assert pool.adapters["broker-2"].last_health_check is not None
    
    @pytest.mark.asyncio
    async def test_health_check_all_with_failures(self, pool, mock_adapter_healthy, mock_adapter_unhealthy):
        """Test health check all adapters with some failures."""
        await pool.add_adapter("broker-1", mock_adapter_healthy, is_primary=True)
        await pool.add_adapter("broker-2", mock_adapter_unhealthy, is_primary=False)
        
        # Set initial connection status to CONNECTED for broker-2 to test transition
        pool.adapters["broker-2"].connection_status = ConnectionStatus.CONNECTED
        
        results = await pool.health_check_all()
        
        assert len(results) == 2
        assert results["broker-1"].is_connected == True
        assert results["broker-2"].is_connected == False
        
        # Check that connection status was updated for broker-2 (CONNECTED -> FAILED)
        assert pool.adapters["broker-1"].connection_status == ConnectionStatus.DISCONNECTED  # No change
        assert pool.adapters["broker-2"].connection_status == ConnectionStatus.FAILED  # Changed from CONNECTED
        assert pool.adapters["broker-2"].failure_count == 1
        assert pool.adapters["broker-2"].last_failure is not None
    
    @pytest.mark.asyncio
    async def test_health_check_all_exception(self, pool):
        """Test health check all adapters with exception."""
        mock_adapter = Mock(spec=BrokerAdapter)
        mock_adapter.health_check = AsyncMock(side_effect=Exception("Health check error"))
        
        await pool.add_adapter("broker-1", mock_adapter, is_primary=True)
        
        results = await pool.health_check_all()
        
        assert len(results) == 1
        assert results["broker-1"].is_connected == False
        assert "Health check error" in results["broker-1"].error
        
        # Check that adapter was marked as failed
        assert pool.adapters["broker-1"].connection_status == ConnectionStatus.FAILED
        assert pool.adapters["broker-1"].failure_count == 1
        assert pool.adapters["broker-1"].last_failure is not None


class TestConnectionPoolHealthMonitoring:
    """Tests for ConnectionPool health monitoring methods."""
    
    @pytest.fixture
    def pool(self):
        """Create ConnectionPool instance."""
        return ConnectionPool()
    
    @pytest.mark.asyncio
    async def test_start_health_monitoring(self, pool):
        """Test starting health monitoring."""
        await pool.start_health_monitoring(interval=5)
        
        assert pool._health_monitoring_task is not None
        assert not pool._health_monitoring_task.done()
    
    @pytest.mark.asyncio
    async def test_start_health_monitoring_already_running(self, pool):
        """Test starting health monitoring when already running."""
        await pool.start_health_monitoring(interval=5)
        task1 = pool._health_monitoring_task
        
        # Try to start again
        await pool.start_health_monitoring(interval=10)
        
        # Should be the same task
        assert pool._health_monitoring_task == task1
    
    @pytest.mark.asyncio
    async def test_stop_health_monitoring(self, pool):
        """Test stopping health monitoring."""
        await pool.start_health_monitoring(interval=5)
        
        await pool.stop_health_monitoring()
        
        assert pool._health_monitoring_task is None
    
    @pytest.mark.asyncio
    async def test_stop_health_monitoring_not_running(self, pool):
        """Test stopping health monitoring when not running."""
        # Should not raise exception
        await pool.stop_health_monitoring()
        
        assert pool._health_monitoring_task is None


class TestConnectionPoolFailover:
    """Tests for ConnectionPool failover methods."""
    
    @pytest.fixture
    def pool(self):
        """Create ConnectionPool instance."""
        return ConnectionPool()
    
    @pytest.fixture
    def mock_adapter_healthy(self):
        """Create mock healthy adapter."""
        adapter = Mock(spec=BrokerAdapter)
        return adapter
    
    @pytest.fixture
    def mock_adapter_unhealthy(self):
        """Create mock unhealthy adapter."""
        adapter = Mock(spec=BrokerAdapter)
        return adapter
    
    @pytest.mark.asyncio
    async def test_failover_to_backup_success(self, pool, mock_adapter_healthy, mock_adapter_unhealthy):
        """Test successful failover to backup."""
        await pool.add_adapter("broker-1", mock_adapter_unhealthy, is_primary=True)
        await pool.add_adapter("broker-2", mock_adapter_healthy, is_primary=False)
        
        # Set primary as unhealthy
        pool.adapters["broker-1"].connection_status = ConnectionStatus.FAILED
        pool.adapters["broker-1"].health_status = HealthStatus(
            is_connected=False,
            latency_ms=0.0,
            last_check=datetime.utcnow()
        )
        
        # Set backup as healthy
        pool.adapters["broker-2"].connection_status = ConnectionStatus.CONNECTED
        pool.adapters["broker-2"].health_status = HealthStatus(
            is_connected=True,
            latency_ms=15.0,
            last_check=datetime.utcnow()
        )
        
        result = await pool.failover_to_backup()
        
        assert result == "broker-2"
        assert pool.primary_broker == "broker-2"
        assert "broker-1" in pool.backup_brokers
        assert "broker-2" not in pool.backup_brokers
    
    @pytest.mark.asyncio
    async def test_failover_to_backup_no_healthy_backup(self, pool, mock_adapter_unhealthy):
        """Test failover when no healthy backup available."""
        await pool.add_adapter("broker-1", mock_adapter_unhealthy, is_primary=True)
        await pool.add_adapter("broker-2", mock_adapter_unhealthy, is_primary=False)
        
        # Set both as unhealthy
        for broker_id in ["broker-1", "broker-2"]:
            pool.adapters[broker_id].connection_status = ConnectionStatus.FAILED
            pool.adapters[broker_id].health_status = HealthStatus(
                is_connected=False,
                latency_ms=0.0,
                last_check=datetime.utcnow()
            )
        
        with pytest.raises(BrokerConnectionError, match="No healthy backup brokers available"):
            await pool.failover_to_backup()
    
    @pytest.mark.asyncio
    async def test_failover_to_backup_no_backups(self, pool, mock_adapter_unhealthy):
        """Test failover when no backup brokers exist."""
        await pool.add_adapter("broker-1", mock_adapter_unhealthy, is_primary=True)
        
        with pytest.raises(BrokerConnectionError, match="No healthy backup brokers available"):
            await pool.failover_to_backup()


class TestConnectionPoolGetConnectionStats:
    """Tests for ConnectionPool get_connection_stats method."""
    
    @pytest.fixture
    def pool(self):
        """Create ConnectionPool instance."""
        return ConnectionPool()
    
    @pytest.fixture
    def mock_adapter_healthy(self):
        """Create mock healthy adapter."""
        return Mock(spec=BrokerAdapter)
    
    @pytest.fixture
    def mock_adapter_unhealthy(self):
        """Create mock unhealthy adapter."""
        return Mock(spec=BrokerAdapter)
    
    @pytest.mark.asyncio
    async def test_get_connection_stats_empty_pool(self, pool):
        """Test getting connection stats for empty pool."""
        stats = pool.get_connection_stats()
        
        assert stats.total_adapters == 0
        assert stats.healthy_adapters == 0
        assert stats.primary_broker is None
        assert stats.backup_brokers == []
        assert stats.total_uptime_hours == 0.0
        assert stats.total_failures == 0
        assert stats.last_health_check is not None
    
    @pytest.mark.asyncio
    async def test_get_connection_stats_with_adapters(self, pool, mock_adapter_healthy, mock_adapter_unhealthy):
        """Test getting connection stats with adapters."""
        await pool.add_adapter("broker-1", mock_adapter_healthy, is_primary=True)
        await pool.add_adapter("broker-2", mock_adapter_unhealthy, is_primary=False)
        
        # Set broker-1 as healthy
        pool.adapters["broker-1"].connection_status = ConnectionStatus.CONNECTED
        pool.adapters["broker-1"].health_status = HealthStatus(
            is_connected=True,
            latency_ms=10.0,
            last_check=datetime.utcnow()
        )
        pool.adapters["broker-1"].uptime_start = datetime.utcnow() - timedelta(hours=1)
        pool.adapters["broker-1"].failure_count = 0
        
        # Set broker-2 as unhealthy
        pool.adapters["broker-2"].connection_status = ConnectionStatus.FAILED
        pool.adapters["broker-2"].health_status = HealthStatus(
            is_connected=False,
            latency_ms=0.0,
            last_check=datetime.utcnow()
        )
        pool.adapters["broker-2"].failure_count = 3
        
        stats = pool.get_connection_stats()
        
        assert stats.total_adapters == 2
        assert stats.healthy_adapters == 1
        assert stats.primary_broker == "broker-1"
        assert stats.backup_brokers == ["broker-2"]
        assert stats.total_uptime_hours > 0
        assert stats.total_failures == 3
        assert stats.last_health_check is not None
    
    @pytest.mark.asyncio
    async def test_get_connection_stats_no_uptime(self, pool, mock_adapter_healthy):
        """Test getting connection stats with no uptime."""
        await pool.add_adapter("broker-1", mock_adapter_healthy, is_primary=True)
        
        # Set adapter as healthy but no uptime_start
        pool.adapters["broker-1"].connection_status = ConnectionStatus.CONNECTED
        pool.adapters["broker-1"].health_status = HealthStatus(
            is_connected=True,
            latency_ms=10.0,
            last_check=datetime.utcnow()
        )
        pool.adapters["broker-1"].uptime_start = None
        
        stats = pool.get_connection_stats()
        
        assert stats.total_adapters == 1
        assert stats.healthy_adapters == 1
        assert stats.total_uptime_hours == 0.0


class TestConnectionPoolRepr:
    """Tests for ConnectionPool __repr__ method."""
    
    @pytest.fixture
    def pool(self):
        """Create ConnectionPool instance."""
        return ConnectionPool()
    
    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        return Mock(spec=BrokerAdapter)
    
    @pytest.mark.asyncio
    async def test_connection_pool_repr_empty(self, pool):
        """Test ConnectionPool string representation when empty."""
        repr_str = repr(pool)
        
        assert "ConnectionPool" in repr_str
        assert "adapters=0" in repr_str
        assert "healthy=0" in repr_str
        assert "primary=None" in repr_str
    
    @pytest.mark.asyncio
    async def test_connection_pool_repr_with_adapters(self, pool, mock_adapter):
        """Test ConnectionPool string representation with adapters."""
        await pool.add_adapter("broker-1", mock_adapter, is_primary=True)
        await pool.add_adapter("broker-2", mock_adapter, is_primary=False)
        
        # Set one as healthy
        pool.adapters["broker-1"].connection_status = ConnectionStatus.CONNECTED
        pool.adapters["broker-1"].health_status = HealthStatus(
            is_connected=True,
            latency_ms=10.0,
            last_check=datetime.utcnow()
        )
        
        repr_str = repr(pool)
        
        assert "ConnectionPool" in repr_str
        assert "adapters=2" in repr_str
        assert "healthy=1" in repr_str
        assert "primary=broker-1" in repr_str


class TestConnectionPoolEdgeCases:
    """Tests for edge cases in ConnectionPool."""
    
    @pytest.fixture
    def pool(self):
        """Create ConnectionPool instance."""
        return ConnectionPool()
    
    @pytest.mark.asyncio
    async def test_health_monitoring_loop_cancellation(self, pool):
        """Test health monitoring loop cancellation."""
        # Start health monitoring
        await pool.start_health_monitoring(interval=1)
        
        # Stop it immediately
        await pool.stop_health_monitoring()
        
        # Task should be cancelled
        assert pool._health_monitoring_task is None
    
    @pytest.mark.asyncio
    async def test_health_monitoring_loop_exception(self, pool):
        """Test health monitoring loop with exception."""
        mock_adapter = Mock(spec=BrokerAdapter)
        mock_adapter.health_check = AsyncMock(side_effect=Exception("Health check error"))
        
        await pool.add_adapter("broker-1", mock_adapter, is_primary=True)
        
        # Start health monitoring with short interval
        await pool.start_health_monitoring(interval=0.1)
        
        # Wait a bit for the loop to run
        await asyncio.sleep(0.2)
        
        # Stop monitoring
        await pool.stop_health_monitoring()
        
        # Should handle exceptions gracefully
        assert pool._health_monitoring_task is None
    
    @pytest.mark.asyncio
    async def test_connect_all_empty_pool(self, pool):
        """Test connecting all adapters when pool is empty."""
        results = await pool.connect_all()
        
        assert results == {}
    
    @pytest.mark.asyncio
    async def test_health_check_all_empty_pool(self, pool):
        """Test health check all adapters when pool is empty."""
        results = await pool.health_check_all()
        
        assert results == {}
    
    @pytest.mark.asyncio
    async def test_disconnect_all_empty_pool(self, pool):
        """Test disconnecting all adapters when pool is empty."""
        # Should not raise exception
        await pool.disconnect_all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
