"""Test database persistence for broker connections."""
import pytest
import pytest_asyncio
import asyncio
import asyncpg
from unittest.mock import Mock, patch


class TestBrokerPersistence:
    """Test broker persistence to PostgreSQL."""
    
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
    
    @pytest.mark.asyncio
    async def test_broker_connections_table_exists(self, db_connection):
        """Test that broker_connections table exists."""
        # Act
        result = await db_connection.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name = 'broker_connections'
        """)
        
        # Assert
        assert len(result) == 1
        assert result[0]['table_name'] == 'broker_connections'
    
    @pytest.mark.asyncio
    async def test_insert_broker_connection(self, db_connection):
        """Test inserting broker connection."""
        # Arrange
        broker_id = "test_mt5_12345"
        broker_type = "mt5"
        account = 12345
        password = "test_password"
        server = "TEST-SERVER"
        
        # Clean up any existing test data
        await db_connection.execute(
            "DELETE FROM broker_connections WHERE broker_id = $1",
            broker_id
        )
        
        # Act
        await db_connection.execute("""
            INSERT INTO broker_connections 
            (broker_id, broker_type, account, password_encrypted, server, last_connected_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
        """, broker_id, broker_type, account, password, server)
        
        # Assert
        result = await db_connection.fetchrow(
            "SELECT * FROM broker_connections WHERE broker_id = $1",
            broker_id
        )
        
        assert result is not None
        assert result['broker_id'] == broker_id
        assert result['broker_type'] == broker_type
        assert result['account'] == account
        assert result['password_encrypted'] == password
        assert result['server'] == server
        assert result['is_active'] is True
        
        # Cleanup
        await db_connection.execute(
            "DELETE FROM broker_connections WHERE broker_id = $1",
            broker_id
        )
    
    @pytest.mark.asyncio
    async def test_upsert_broker_connection(self, db_connection):
        """Test upsert (INSERT ... ON CONFLICT) functionality."""
        # Arrange
        broker_id = "test_mt5_upsert"
        broker_type = "mt5"
        account = 12345
        password = "test_password"
        server = "TEST-SERVER"
        
        # Clean up any existing test data
        await db_connection.execute(
            "DELETE FROM broker_connections WHERE broker_id = $1",
            broker_id
        )
        
        # Act - First insert
        await db_connection.execute("""
            INSERT INTO broker_connections 
            (broker_id, broker_type, account, password_encrypted, server, last_connected_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (broker_id) 
            DO UPDATE SET last_connected_at = NOW(), is_active = TRUE
        """, broker_id, broker_type, account, password, server)
        
        # Act - Second insert (should update)
        await db_connection.execute("""
            INSERT INTO broker_connections 
            (broker_id, broker_type, account, password_encrypted, server, last_connected_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (broker_id) 
            DO UPDATE SET last_connected_at = NOW(), is_active = TRUE
        """, broker_id, broker_type, account, password, server)
        
        # Assert - Should still be only one record
        result = await db_connection.fetch(
            "SELECT * FROM broker_connections WHERE broker_id = $1",
            broker_id
        )
        
        assert len(result) == 1
        assert result[0]['broker_id'] == broker_id
        
        # Cleanup
        await db_connection.execute(
            "DELETE FROM broker_connections WHERE broker_id = $1",
            broker_id
        )
    
    @pytest.mark.asyncio
    async def test_get_active_brokers(self, db_connection):
        """Test getting active brokers."""
        # Arrange
        broker1_id = "test_mt5_active1"
        broker2_id = "test_mt5_active2"
        broker3_id = "test_mt5_inactive"
        
        # Clean up any existing test data
        await db_connection.execute(
            "DELETE FROM broker_connections WHERE broker_id IN ($1, $2, $3)",
            broker1_id, broker2_id, broker3_id
        )
        
        # Insert test data
        await db_connection.execute("""
            INSERT INTO broker_connections 
            (broker_id, broker_type, account, password_encrypted, server, is_active)
            VALUES ($1, 'mt5', 12345, 'pass1', 'SERVER1', TRUE),
                   ($2, 'mt5', 12346, 'pass2', 'SERVER2', TRUE),
                   ($3, 'mt5', 12347, 'pass3', 'SERVER3', FALSE)
        """, broker1_id, broker2_id, broker3_id)
        
        # Act
        result = await db_connection.fetch("""
            SELECT broker_id, broker_type, account, server, last_connected_at, is_active
            FROM broker_connections
            WHERE is_active = TRUE AND broker_id IN ($1, $2, $3)
            ORDER BY last_connected_at DESC
        """, broker1_id, broker2_id, broker3_id)
        
        # Assert
        assert len(result) == 2
        broker_ids = [row['broker_id'] for row in result]
        assert broker1_id in broker_ids
        assert broker2_id in broker_ids
        assert broker3_id not in broker_ids
        
        # Cleanup
        await db_connection.execute(
            "DELETE FROM broker_connections WHERE broker_id IN ($1, $2, $3)",
            broker1_id, broker2_id, broker3_id
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
