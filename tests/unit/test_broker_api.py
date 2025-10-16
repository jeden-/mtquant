"""Test API endpoints for broker management."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json

# Import the FastAPI app
from api.main import app


class TestBrokerAPI:
    """Test broker API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_broker_manager(self):
        """Create mock broker manager."""
        manager = Mock()
        manager.register_broker = AsyncMock()
        manager.get_broker = Mock(return_value=None)
        manager.list_brokers = Mock(return_value=[])
        return manager
    
    @pytest.fixture
    def mock_mt5_client(self):
        """Create mock MT5 client."""
        client = Mock()
        client.connect = AsyncMock(return_value=True)
        client.get_account_info = AsyncMock(return_value={
            'balance': 100000.0,
            'equity': 100000.0,
            'margin': 0.0,
            'free_margin': 100000.0,
            'profit': 0.0,
            'leverage': 100,
            'login': 12345,
            'server': 'TEST-SERVER'
        })
        client.is_connected = True
        return client
    
    @patch('api.routes.brokers.broker_manager')
    @patch('api.routes.brokers.MockMT5Client')
    @patch('api.routes.brokers.asyncpg.connect')
    def test_connect_broker_success(self, mock_asyncpg, mock_mt5_client_class, mock_broker_manager, client):
        """Test successful broker connection."""
        # Arrange
        mock_mt5_client = Mock()
        mock_mt5_client.connect = AsyncMock(return_value=True)
        mock_mt5_client.get_account_info = AsyncMock(return_value={
            'balance': 100000.0,
            'equity': 100000.0,
            'margin': 0.0,
            'free_margin': 100000.0,
            'profit': 0.0,
            'leverage': 100,
            'login': 12345,
            'server': 'TEST-SERVER'
        })
        mock_mt5_client_class.return_value = mock_mt5_client
        
        # Mock database connection
        mock_conn = Mock()
        mock_conn.execute = AsyncMock()
        mock_conn.close = AsyncMock()
        mock_asyncpg.return_value = mock_conn
        
        # Mock broker manager
        mock_broker_manager.register_broker = AsyncMock()
        mock_broker_manager.get_broker = Mock(return_value=mock_mt5_client)
        mock_broker_manager.list_brokers = Mock(return_value=['mt5_12345'])
        
        request_data = {
            "broker_type": "mt5",
            "account": 12345,
            "password": "test_password",
            "server": "TEST-SERVER"
        }
        
        # Act
        response = client.post("/api/brokers/connect", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert 'broker_id' in data
        assert data['broker_id'] == 'mt5_12345'
        assert 'account_info' in data
        assert data['account_info']['balance'] == 100000.0
        
        # Verify mock calls
        mock_mt5_client_class.assert_called_once()
        mock_mt5_client.connect.assert_called_once()
        mock_mt5_client.get_account_info.assert_called_once()
        mock_broker_manager.register_broker.assert_called_once()
        mock_conn.execute.assert_called_once()
    
    @patch('api.routes.brokers.broker_manager')
    def test_connect_broker_invalid_type(self, mock_broker_manager, client):
        """Test broker connection with invalid broker type."""
        # Arrange
        request_data = {
            "broker_type": "invalid",
            "account": 12345,
            "password": "test_password",
            "server": "TEST-SERVER"
        }
        
        # Act
        response = client.post("/api/brokers/connect", json=request_data)
        
        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Unsupported broker type" in data['detail']
    
    @patch('api.routes.brokers.broker_manager')
    def test_connect_broker_missing_fields(self, mock_broker_manager, client):
        """Test broker connection with missing required fields."""
        # Arrange
        request_data = {
            "broker_type": "mt5",
            "account": 12345
            # Missing password and server
        }
        
        # Act
        response = client.post("/api/brokers/connect", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error
    
    @patch('api.routes.brokers.broker_manager')
    @patch('api.routes.brokers.asyncpg.connect')
    def test_list_brokers_success(self, mock_asyncpg, mock_broker_manager, client):
        """Test listing brokers."""
        # Arrange
        mock_conn = Mock()
        mock_conn.fetch = AsyncMock(return_value=[
            {
                'broker_id': 'mt5_12345',
                'broker_type': 'mt5',
                'account': 12345,
                'server': 'TEST-SERVER',
                'last_connected_at': None,
                'is_active': True
            }
        ])
        mock_conn.close = AsyncMock()
        mock_asyncpg.return_value = mock_conn
        
        mock_broker_manager.get_broker = Mock(return_value=None)
        
        # Act
        response = client.get("/api/brokers/list")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert 'brokers' in data
        assert len(data['brokers']) == 1
        assert data['brokers'][0]['broker_id'] == 'mt5_12345'
        assert data['brokers'][0]['connected'] is False
        assert 'account_info' in data['brokers'][0]
    
    @patch('api.routes.brokers.broker_manager')
    def test_get_broker_status_not_found(self, mock_broker_manager, client):
        """Test getting status of non-existent broker."""
        # Arrange
        mock_broker_manager.get_broker = Mock(return_value=None)
        
        # Act
        response = client.get("/api/brokers/non_existent/status")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['connected'] is False
        assert data['broker_id'] == 'non_existent'
    
    @patch('api.routes.brokers.broker_manager')
    def test_get_broker_status_found(self, mock_broker_manager, client):
        """Test getting status of existing broker."""
        # Arrange
        mock_client = Mock()
        mock_client.is_connected = True
        mock_client.get_account_info = AsyncMock(return_value={
            'balance': 100000.0,
            'equity': 100000.0,
            'margin': 0.0,
            'free_margin': 100000.0,
            'profit': 0.0,
            'leverage': 100,
            'login': 12345,
            'server': 'TEST-SERVER'
        })
        mock_broker_manager.get_broker = Mock(return_value=mock_client)
        
        # Act
        response = client.get("/api/brokers/mt5_12345/status")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['connected'] is True
        assert data['broker_id'] == 'mt5_12345'
        assert 'account_info' in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
