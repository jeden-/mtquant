"""Test script to check broker_manager state."""
import requests
import json

def test_broker_manager():
    """Test broker manager state."""
    
    # Test 1: Connect broker
    print("=== TEST 1: CONNECT BROKER ===")
    connect_data = {
        "broker_type": "mt5",
        "account": 62675178,
        "password": "9Rb!Z8*K",
        "server": "OANDATMS-MT5"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/brokers/connect",
            json=connect_data,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: List brokers
    print("\n=== TEST 2: LIST BROKERS ===")
    try:
        response = requests.get(
            "http://localhost:8000/api/brokers/list",
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Check broker status
    print("\n=== TEST 3: BROKER STATUS ===")
    try:
        response = requests.get(
            "http://localhost:8000/api/brokers/mt5_62675178/status",
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_broker_manager()
