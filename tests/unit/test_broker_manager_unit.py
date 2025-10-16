"""Test BrokerManager.register_broker() functionality."""
import pytest
from unittest.mock import Mock, AsyncMock
from mtquant.mcp_integration.managers.broker_manager import BrokerManager
from mtquant.utils.exceptions import BrokerError


class TestBrokerManager:
    """Test BrokerManager functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.broker_manager = BrokerManager()
        self.mock_client = Mock()
        self.mock_client.broker_id = "test_broker"
    
    @pytest.mark.asyncio
    async def test_register_broker_success(self):
        """Test successful broker registration."""
        # Arrange
        broker_id = "mt5_12345"
        client = self.mock_client
        
        # Act
        await self.broker_manager.register_broker(broker_id, client)
        
        # Assert
        assert broker_id in self.broker_manager._brokers
        assert self.broker_manager._brokers[broker_id] == client
        assert broker_id in self.broker_manager.list_brokers()
    
    @pytest.mark.asyncio
    async def test_get_broker_success(self):
        """Test getting registered broker."""
        # Arrange
        broker_id = "mt5_12345"
        client = self.mock_client
        await self.broker_manager.register_broker(broker_id, client)
        
        # Act
        retrieved_client = self.broker_manager.get_broker(broker_id)
        
        # Assert
        assert retrieved_client == client
    
    def test_get_broker_not_found(self):
        """Test getting non-existent broker."""
        # Act
        retrieved_client = self.broker_manager.get_broker("non_existent")
        
        # Assert
        assert retrieved_client is None
    
    def test_list_brokers_empty(self):
        """Test listing brokers when none are registered."""
        # Act
        brokers = self.broker_manager.list_brokers()
        
        # Assert
        assert brokers == []
    
    @pytest.mark.asyncio
    async def test_list_brokers_multiple(self):
        """Test listing multiple brokers."""
        # Arrange
        broker1_id = "mt5_12345"
        broker2_id = "mt4_67890"
        client1 = Mock()
        client2 = Mock()
        
        await self.broker_manager.register_broker(broker1_id, client1)
        await self.broker_manager.register_broker(broker2_id, client2)
        
        # Act
        brokers = self.broker_manager.list_brokers()
        
        # Assert
        assert len(brokers) == 2
        assert broker1_id in brokers
        assert broker2_id in brokers
    
    @pytest.mark.asyncio
    async def test_unregister_broker_success(self):
        """Test successful broker unregistration."""
        # Arrange
        broker_id = "mt5_12345"
        client = self.mock_client
        await self.broker_manager.register_broker(broker_id, client)
        
        # Act
        self.broker_manager.unregister_broker(broker_id)
        
        # Assert
        assert broker_id not in self.broker_manager._brokers
        assert broker_id not in self.broker_manager.list_brokers()
    
    def test_unregister_broker_not_found(self):
        """Test unregistering non-existent broker."""
        # Act & Assert - should not raise exception
        self.broker_manager.unregister_broker("non_existent")
    
    @pytest.mark.asyncio
    async def test_register_broker_overwrite(self):
        """Test registering broker with same ID overwrites previous."""
        # Arrange
        broker_id = "mt5_12345"
        client1 = Mock()
        client2 = Mock()
        
        await self.broker_manager.register_broker(broker_id, client1)
        
        # Act
        await self.broker_manager.register_broker(broker_id, client2)
        
        # Assert
        assert self.broker_manager._brokers[broker_id] == client2
        assert len(self.broker_manager.list_brokers()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
