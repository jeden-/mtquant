"""Test MockMT5Client functionality."""
import pytest
import asyncio
from mtquant.mcp_integration.clients.mock_mt5_client import MockMT5Client


class TestMockMT5Client:
    """Test MockMT5Client functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = {
            'account': 12345,
            'password': 'test_password',
            'server': 'TEST-SERVER'
        }
        self.client = MockMT5Client("test_broker", self.config)
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        # Act
        result = await self.client.connect()
        
        # Assert
        assert result is True
        assert self.client.is_connected is True
    
    @pytest.mark.asyncio
    async def test_get_account_info(self):
        """Test getting account info."""
        # Arrange
        await self.client.connect()
        
        # Act
        account_info = await self.client.get_account_info()
        
        # Assert
        assert isinstance(account_info, dict)
        assert 'balance' in account_info
        assert 'equity' in account_info
        assert 'margin' in account_info
        assert 'free_margin' in account_info
        assert 'profit' in account_info
        assert 'leverage' in account_info
        assert 'login' in account_info
        assert 'server' in account_info
        
        # Check values
        assert account_info['login'] == 12345
        assert account_info['server'] == 'TEST-SERVER'
        assert account_info['balance'] == 100000.0
        assert account_info['leverage'] == 100
    
    @pytest.mark.asyncio
    async def test_health_check_connected(self):
        """Test health check when connected."""
        # Arrange
        await self.client.connect()
        
        # Act
        health = await self.client.health_check()
        
        # Assert
        assert health is True
    
    @pytest.mark.asyncio
    async def test_health_check_not_connected(self):
        """Test health check when not connected."""
        # Act
        health = await self.client.health_check()
        
        # Assert
        assert health is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection."""
        # Arrange
        await self.client.connect()
        assert self.client.is_connected is True
        
        # Act
        await self.client.disconnect()
        
        # Assert
        assert self.client.is_connected is False
    
    @pytest.mark.asyncio
    async def test_get_account_info_not_connected(self):
        """Test getting account info when not connected."""
        # Act
        account_info = await self.client.get_account_info()
        
        # Assert - should still return mock data
        assert isinstance(account_info, dict)
        assert 'balance' in account_info
    
    def test_is_connected_property(self):
        """Test is_connected property."""
        # Initially not connected
        assert self.client.is_connected is False
        
        # After connect (sync)
        self.client._connected = True
        assert self.client.is_connected is True
        
        # After disconnect (sync)
        self.client._connected = False
        assert self.client.is_connected is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
