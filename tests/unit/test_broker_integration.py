"""Integration test for broker connection flow."""
import pytest
import pytest_asyncio
import asyncio
import asyncpg
from unittest.mock import patch, Mock
from mtquant.mcp_integration.managers.broker_manager import BrokerManager
from mtquant.mcp_integration.clients.mock_mt5_client import MockMT5Client


class TestBrokerIntegration:
    """Integration test for broker connection flow."""
    
    @pytest_asyncio.fixture
    async def db_connection(self):
        """Create test database connection."""
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            database='mtquantum',
            user='postgres',
            password='MARiusz@!2025'
        )
        yield conn
        await conn.close()
    
    @pytest.fixture
    def broker_manager(self):
        """Create broker manager."""
        return BrokerManager()
    
    @pytest.fixture
    def mock_mt5_client(self):
        """Create mock MT5 client."""
        config = {
            'account': 12345,
            'password': 'test_password',
            'server': 'TEST-SERVER'
        }
        return MockMT5Client("test_broker", config)
    
    @pytest.mark.asyncio
    async def test_full_broker_connection_flow(self, broker_manager, mock_mt5_client, db_connection):
        """Test complete broker connection flow."""
        # Arrange
        broker_id = "integration_test_mt5"
        
        # Clean up any existing test data
        await db_connection.execute(
            "DELETE FROM broker_connections WHERE broker_id = $1",
            broker_id
        )
        
        # Act 1: Connect client
        connection_result = await mock_mt5_client.connect()
        assert connection_result is True
        
        # Act 2: Get account info
        account_info = await mock_mt5_client.get_account_info()
        assert account_info is not None
        
        # Act 3: Register with broker manager
        await broker_manager.register_broker(broker_id, mock_mt5_client)
        
        # Act 4: Save to database
        await db_connection.execute("""
            INSERT INTO broker_connections 
            (broker_id, broker_type, account, password_encrypted, server, last_connected_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (broker_id) 
            DO UPDATE SET last_connected_at = NOW(), is_active = TRUE
        """, broker_id, 'mt5', 12345, 'test_password', 'TEST-SERVER')
        
        # Assert 1: Client is connected
        assert mock_mt5_client.is_connected is True
        
        # Assert 2: Broker is registered in manager
        retrieved_client = broker_manager.get_broker(broker_id)
        assert retrieved_client == mock_mt5_client
        assert broker_id in broker_manager.list_brokers()
        
        # Assert 3: Broker is saved in database
        db_result = await db_connection.fetchrow(
            "SELECT * FROM broker_connections WHERE broker_id = $1",
            broker_id
        )
        assert db_result is not None
        assert db_result['broker_id'] == broker_id
        assert db_result['broker_type'] == 'mt5'
        assert db_result['account'] == 12345
        assert db_result['is_active'] is True
        
        # Act 4: Disconnect
        await mock_mt5_client.disconnect()
        
        # Assert 4: Client is disconnected
        assert mock_mt5_client.is_connected is False
        
        # Cleanup
        await db_connection.execute(
            "DELETE FROM broker_connections WHERE broker_id = $1",
            broker_id
        )
    
    @pytest.mark.asyncio
    async def test_broker_persistence_after_restart(self, db_connection):
        """Test that broker persists in database after 'restart'."""
        # Arrange
        broker_id = "persistence_test_mt5"
        
        # Clean up any existing test data
        await db_connection.execute(
            "DELETE FROM broker_connections WHERE broker_id = $1",
            broker_id
        )
        
        # Act 1: Save broker to database
        await db_connection.execute("""
            INSERT INTO broker_connections 
            (broker_id, broker_type, account, password_encrypted, server, last_connected_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
        """, broker_id, 'mt5', 12345, 'test_password', 'TEST-SERVER')
        
        # Act 2: Simulate restart - create new broker manager
        new_broker_manager = BrokerManager()
        
        # Act 3: Load brokers from database
        db_brokers = await db_connection.fetch("""
            SELECT broker_id, broker_type, account, server, last_connected_at, is_active
            FROM broker_connections
            WHERE is_active = TRUE
            ORDER BY last_connected_at DESC
        """)
        
        # Assert: Broker is in database but not in memory
        assert len(db_brokers) >= 1
        broker_found = any(row['broker_id'] == broker_id for row in db_brokers)
        assert broker_found is True
        
        # Broker should not be in memory (simulating restart)
        assert broker_id not in new_broker_manager.list_brokers()
        
        # Cleanup
        await db_connection.execute(
            "DELETE FROM broker_connections WHERE broker_id = $1",
            broker_id
        )
    
    @pytest.mark.asyncio
    async def test_multiple_brokers_management(self, broker_manager, db_connection):
        """Test managing multiple brokers."""
        # Arrange
        broker1_id = "multi_test_mt5_1"
        broker2_id = "multi_test_mt4_1"
        
        # Clean up any existing test data
        await db_connection.execute(
            "DELETE FROM broker_connections WHERE broker_id IN ($1, $2)",
            broker1_id, broker2_id
        )
        
        # Create mock clients
        config1 = {'account': 12345, 'password': 'pass1', 'server': 'SERVER1'}
        config2 = {'account': 12346, 'password': 'pass2', 'server': 'SERVER2'}
        client1 = MockMT5Client(broker1_id, config1)
        client2 = MockMT5Client(broker2_id, config2)
        
        # Act 1: Connect both clients
        await client1.connect()
        await client2.connect()
        
        # Act 2: Register both with manager
        await broker_manager.register_broker(broker1_id, client1)
        await broker_manager.register_broker(broker2_id, client2)
        
        # Act 3: Save both to database
        await db_connection.execute("""
            INSERT INTO broker_connections 
            (broker_id, broker_type, account, password_encrypted, server, last_connected_at)
            VALUES ($1, 'mt5', $2, $3, $4, NOW()),
                   ($5, 'mt4', $6, $7, $8, NOW())
        """, broker1_id, 12345, 'pass1', 'SERVER1',
             broker2_id, 12346, 'pass2', 'SERVER2')
        
        # Assert 1: Both brokers in memory
        assert len(broker_manager.list_brokers()) == 2
        assert broker1_id in broker_manager.list_brokers()
        assert broker2_id in broker_manager.list_brokers()
        
        # Assert 2: Both brokers in database
        db_brokers = await db_connection.fetch("""
            SELECT broker_id FROM broker_connections 
            WHERE broker_id IN ($1, $2) AND is_active = TRUE
        """, broker1_id, broker2_id)
        assert len(db_brokers) == 2
        
        # Act 3: Disconnect one broker
        await client1.disconnect()
        broker_manager.unregister_broker(broker1_id)
        
        # Assert 3: One broker removed from memory
        assert len(broker_manager.list_brokers()) == 1
        assert broker1_id not in broker_manager.list_brokers()
        assert broker2_id in broker_manager.list_brokers()
        
        # Cleanup
        await db_connection.execute(
            "DELETE FROM broker_connections WHERE broker_id IN ($1, $2)",
            broker1_id, broker2_id
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
