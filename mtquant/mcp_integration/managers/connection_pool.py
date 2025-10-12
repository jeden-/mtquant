"""
Connection Pool for managing multiple broker connections.

This module provides a connection pool for managing multiple broker adapters
with health monitoring, automatic failover, and connection statistics tracking.

Features:
1. Pool of broker adapters (MT5, MT4 in future)
2. Health monitoring (periodic checks every 30s)
3. Automatic failover to backup broker
4. Connection statistics tracking

Example:
    # Create connection pool
    pool = ConnectionPool()
    
    # Add adapters
    await pool.add_adapter("primary", mt5_adapter, is_primary=True)
    await pool.add_adapter("backup", mt5_adapter_backup)
    
    # Connect all
    await pool.connect_all()
    
    # Get healthy adapter
    adapter = await pool.get_healthy_adapter()
    
    # Start health monitoring
    await pool.start_health_monitoring(interval=30)
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from mtquant.mcp_integration.adapters.base_adapter import BrokerAdapter, HealthStatus
from mtquant.utils.exceptions import BrokerConnectionError, BrokerError
from mtquant.utils.logger import get_logger

logger = get_logger(__name__)


class ConnectionStatus(Enum):
    """Connection status enumeration."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    FAILED = "failed"


@dataclass
class AdapterInfo:
    """Information about a broker adapter in the pool."""
    adapter: BrokerAdapter
    is_primary: bool
    added_at: datetime
    last_health_check: Optional[datetime] = None
    health_status: Optional[HealthStatus] = None
    connection_status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    uptime_start: Optional[datetime] = None


@dataclass
class ConnectionStats:
    """Connection statistics for the pool."""
    total_adapters: int = 0
    healthy_adapters: int = 0
    primary_broker: Optional[str] = None
    backup_brokers: List[str] = field(default_factory=list)
    total_uptime_hours: float = 0.0
    total_failures: int = 0
    last_health_check: Optional[datetime] = None


class ConnectionPool:
    """
    Connection pool for managing multiple broker adapters.
    
    This class provides a centralized way to manage multiple broker connections
    with health monitoring, automatic failover, and connection statistics.
    
    Features:
    - Pool of broker adapters (MT5, MT4 in future)
    - Health monitoring (periodic checks every 30s)
    - Automatic failover to backup broker
    - Connection statistics tracking
    - Thread-safe operations
    """
    
    def __init__(self):
        """Initialize connection pool."""
        self.adapters: Dict[str, AdapterInfo] = {}
        self.primary_broker: Optional[str] = None
        self.backup_brokers: List[str] = []
        self._health_monitoring_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self.logger = get_logger(__name__)
        
        self.logger.info("Connection pool initialized")
    
    async def add_adapter(
        self,
        broker_id: str,
        adapter: BrokerAdapter,
        is_primary: bool = False
    ) -> None:
        """
        Add broker adapter to pool.
        
        Args:
            broker_id: Unique broker identifier
            adapter: BrokerAdapter instance
            is_primary: Mark as primary broker for routing
            
        Raises:
            ValueError: If broker_id already exists
        """
        async with self._lock:
            if broker_id in self.adapters:
                raise ValueError(f"Broker {broker_id} already exists in pool")
            
            adapter_info = AdapterInfo(
                adapter=adapter,
                is_primary=is_primary,
                added_at=datetime.utcnow()
            )
            
            self.adapters[broker_id] = adapter_info
            
            if is_primary:
                self.primary_broker = broker_id
            else:
                self.backup_brokers.append(broker_id)
            
            self.logger.info(f"Added adapter {broker_id} to pool (primary: {is_primary})")
    
    async def connect_all(self) -> Dict[str, bool]:
        """
        Connect all adapters in pool.
        
        Returns:
            Dict of broker_id -> connection_status
        """
        results = {}
        
        async with self._lock:
            for broker_id, adapter_info in self.adapters.items():
                try:
                    self.logger.info(f"Connecting adapter {broker_id}")
                    adapter_info.connection_status = ConnectionStatus.CONNECTING
                    
                    connected = await adapter_info.adapter.connect()
                    
                    if connected:
                        adapter_info.connection_status = ConnectionStatus.CONNECTED
                        adapter_info.uptime_start = datetime.utcnow()
                        adapter_info.failure_count = 0
                        adapter_info.last_failure = None
                        results[broker_id] = True
                        self.logger.info(f"Successfully connected adapter {broker_id}")
                    else:
                        adapter_info.connection_status = ConnectionStatus.FAILED
                        adapter_info.failure_count += 1
                        adapter_info.last_failure = datetime.utcnow()
                        results[broker_id] = False
                        self.logger.warning(f"Failed to connect adapter {broker_id}")
                
                except Exception as e:
                    adapter_info.connection_status = ConnectionStatus.FAILED
                    adapter_info.failure_count += 1
                    adapter_info.last_failure = datetime.utcnow()
                    results[broker_id] = False
                    self.logger.error(f"Error connecting adapter {broker_id}: {e}")
        
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect all adapters cleanly."""
        async with self._lock:
            for broker_id, adapter_info in self.adapters.items():
                try:
                    await adapter_info.adapter.disconnect()
                    adapter_info.connection_status = ConnectionStatus.DISCONNECTED
                    adapter_info.uptime_start = None
                    self.logger.info(f"Disconnected adapter {broker_id}")
                except Exception as e:
                    self.logger.error(f"Error disconnecting adapter {broker_id}: {e}")
    
    async def get_adapter(self, broker_id: str) -> BrokerAdapter:
        """
        Get specific adapter by ID.
        
        Args:
            broker_id: Broker identifier
            
        Returns:
            BrokerAdapter instance
            
        Raises:
            KeyError: If broker_id not found
        """
        async with self._lock:
            if broker_id not in self.adapters:
                raise KeyError(f"Broker {broker_id} not found in pool")
            
            return self.adapters[broker_id].adapter
    
    async def get_healthy_adapter(self) -> BrokerAdapter:
        """
        Get first healthy adapter.
        
        Priority: primary broker -> backup brokers
        
        Returns:
            First healthy BrokerAdapter
            
        Raises:
            BrokerConnectionError: If no healthy adapters available
        """
        async with self._lock:
            # Check primary broker first
            if self.primary_broker:
                primary_info = self.adapters[self.primary_broker]
                if (primary_info.connection_status == ConnectionStatus.CONNECTED and
                    primary_info.health_status and
                    primary_info.health_status.is_connected):
                    return primary_info.adapter
            
            # Check backup brokers
            for broker_id in self.backup_brokers:
                backup_info = self.adapters[broker_id]
                if (backup_info.connection_status == ConnectionStatus.CONNECTED and
                    backup_info.health_status and
                    backup_info.health_status.is_connected):
                    return backup_info.adapter
            
            # No healthy adapters found
            raise BrokerConnectionError("No healthy adapters available")
    
    async def health_check_all(self) -> Dict[str, HealthStatus]:
        """
        Check health of all adapters.
        
        Updates internal health_status dict.
        Returns current health status.
        
        Returns:
            Dict of broker_id -> HealthStatus
        """
        results = {}
        
        async with self._lock:
            for broker_id, adapter_info in self.adapters.items():
                try:
                    health = await adapter_info.adapter.health_check()
                    adapter_info.health_status = health
                    adapter_info.last_health_check = datetime.utcnow()
                    
                    # Update connection status based on health
                    if health.is_connected:
                        if adapter_info.connection_status == ConnectionStatus.FAILED:
                            adapter_info.connection_status = ConnectionStatus.CONNECTED
                            adapter_info.uptime_start = datetime.utcnow()
                            self.logger.info(f"Adapter {broker_id} recovered")
                    else:
                        if adapter_info.connection_status == ConnectionStatus.CONNECTED:
                            adapter_info.connection_status = ConnectionStatus.FAILED
                            adapter_info.failure_count += 1
                            adapter_info.last_failure = datetime.utcnow()
                            self.logger.warning(f"Adapter {broker_id} failed health check")
                    
                    results[broker_id] = health
                    
                except Exception as e:
                    self.logger.error(f"Health check failed for {broker_id}: {e}")
                    adapter_info.connection_status = ConnectionStatus.FAILED
                    adapter_info.failure_count += 1
                    adapter_info.last_failure = datetime.utcnow()
                    
                    # Create failed health status
                    failed_health = HealthStatus(
                        is_connected=False,
                        latency_ms=0.0,
                        last_check=datetime.utcnow(),
                        error=str(e)
                    )
                    results[broker_id] = failed_health
        
        return results
    
    async def start_health_monitoring(self, interval: int = 30) -> None:
        """
        Start background task for periodic health checks.
        
        Args:
            interval: Seconds between health checks
        """
        if self._health_monitoring_task and not self._health_monitoring_task.done():
            self.logger.warning("Health monitoring already running")
            return
        
        self._health_monitoring_task = asyncio.create_task(
            self._health_monitoring_loop(interval)
        )
        self.logger.info(f"Started health monitoring (interval: {interval}s)")
    
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring task."""
        if self._health_monitoring_task:
            self._health_monitoring_task.cancel()
            try:
                await self._health_monitoring_task
            except asyncio.CancelledError:
                pass
            self._health_monitoring_task = None
            self.logger.info("Stopped health monitoring")
    
    async def failover_to_backup(self) -> str:
        """
        Failover to backup broker.
        
        Returns:
            broker_id of new primary broker
            
        Raises:
            BrokerConnectionError: If no backup available
        """
        async with self._lock:
            # Find first healthy backup
            for broker_id in self.backup_brokers:
                backup_info = self.adapters[broker_id]
                if (backup_info.connection_status == ConnectionStatus.CONNECTED and
                    backup_info.health_status and
                    backup_info.health_status.is_connected):
                    
                    # Update primary broker
                    old_primary = self.primary_broker
                    self.primary_broker = broker_id
                    self.backup_brokers = [b for b in self.backup_brokers if b != broker_id]
                    if old_primary:
                        self.backup_brokers.append(old_primary)
                    
                    self.logger.warning(f"Failover: {old_primary} -> {broker_id}")
                    return broker_id
            
            raise BrokerConnectionError("No healthy backup brokers available")
    
    def get_connection_stats(self) -> ConnectionStats:
        """
        Return connection statistics.
        
        Returns:
            ConnectionStats object with current statistics
        """
        healthy_count = 0
        total_failures = 0
        total_uptime = 0.0
        
        for adapter_info in self.adapters.values():
            if (adapter_info.connection_status == ConnectionStatus.CONNECTED and
                adapter_info.health_status and
                adapter_info.health_status.is_connected):
                healthy_count += 1
            
            total_failures += adapter_info.failure_count
            
            if adapter_info.uptime_start:
                uptime = datetime.utcnow() - adapter_info.uptime_start
                total_uptime += uptime.total_seconds() / 3600  # Convert to hours
        
        return ConnectionStats(
            total_adapters=len(self.adapters),
            healthy_adapters=healthy_count,
            primary_broker=self.primary_broker,
            backup_brokers=self.backup_brokers.copy(),
            total_uptime_hours=total_uptime,
            total_failures=total_failures,
            last_health_check=datetime.utcnow()
        )
    
    async def _health_monitoring_loop(self, interval: int) -> None:
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(interval)
                await self.health_check_all()
                
                # Check if failover is needed
                if self.primary_broker:
                    primary_info = self.adapters[self.primary_broker]
                    if (primary_info.connection_status == ConnectionStatus.FAILED and
                        primary_info.failure_count >= 3):  # Fail after 3 consecutive failures
                        try:
                            await self.failover_to_backup()
                        except BrokerConnectionError:
                            self.logger.error("Failover failed - no backup available")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def __repr__(self) -> str:
        """String representation of connection pool."""
        stats = self.get_connection_stats()
        return (f"ConnectionPool(adapters={stats.total_adapters}, "
                f"healthy={stats.healthy_adapters}, "
                f"primary={stats.primary_broker})")
