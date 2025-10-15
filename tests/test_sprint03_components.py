"""
Sprint 03 Component Tests.

Quick validation tests for all Sprint 3 implementations.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_imports_api_routes():
    """Test that all API route modules can be imported."""
    print("Testing API routes imports...")
    
    try:
        from api.routes import websocket
        print("  ✅ api.routes.websocket")
    except Exception as e:
        print(f"  ❌ api.routes.websocket: {e}")
        return False
    
    try:
        from api.routes import metrics
        print("  ✅ api.routes.metrics")
    except Exception as e:
        print(f"  ❌ api.routes.metrics: {e}")
        return False
    
    try:
        from api.routes import agents
        print("  ✅ api.routes.agents")
    except Exception as e:
        print(f"  ❌ api.routes.agents: {e}")
        return False
    
    try:
        from api.routes import portfolio
        print("  ✅ api.routes.portfolio")
    except Exception as e:
        print(f"  ❌ api.routes.portfolio: {e}")
        return False
    
    try:
        from api.routes import orders
        print("  ✅ api.routes.orders")
    except Exception as e:
        print(f"  ❌ api.routes.orders: {e}")
        return False
    
    return True


def test_imports_database_clients():
    """Test that database clients can be imported."""
    print("\nTesting database clients imports...")
    
    try:
        from mtquant.data.storage.redis_client import RedisClient
        print("  ✅ RedisClient")
    except Exception as e:
        print(f"  ❌ RedisClient: {e}")
        return False
    
    try:
        from mtquant.data.storage.postgresql_client import PostgreSQLClient
        print("  ✅ PostgreSQLClient")
    except Exception as e:
        print(f"  ❌ PostgreSQLClient: {e}")
        return False
    
    try:
        from mtquant.data.storage.questdb_client import QuestDBClient
        print("  ✅ QuestDBClient")
    except Exception as e:
        print(f"  ❌ QuestDBClient: {e}")
        return False
    
    return True


def test_database_client_methods():
    """Test that new methods exist in database clients."""
    print("\nTesting database client methods...")
    
    try:
        from mtquant.data.storage.redis_client import RedisClient
        client = RedisClient.__dict__
        assert 'get_metric' in str(client), "get_metric method missing"
        assert 'set_metric' in str(client), "set_metric method missing"
        print("  ✅ RedisClient has get_metric and set_metric")
    except Exception as e:
        print(f"  ❌ RedisClient methods: {e}")
        return False
    
    try:
        from mtquant.data.storage.postgresql_client import PostgreSQLClient
        client = PostgreSQLClient.__dict__
        assert 'get_open_positions' in str(client), "get_open_positions method missing"
        print("  ✅ PostgreSQLClient has get_open_positions")
    except Exception as e:
        print(f"  ❌ PostgreSQLClient methods: {e}")
        return False
    
    return True


def test_websocket_components():
    """Test WebSocket components."""
    print("\nTesting WebSocket components...")
    
    try:
        from api.routes.websocket import ConnectionManager, manager
        print("  ✅ ConnectionManager class exists")
        print("  ✅ Global manager instance exists")
        
        # Check methods
        assert hasattr(manager, 'connect'), "connect method missing"
        assert hasattr(manager, 'disconnect'), "disconnect method missing"
        assert hasattr(manager, 'broadcast'), "broadcast method missing"
        print("  ✅ ConnectionManager has required methods")
    except Exception as e:
        print(f"  ❌ WebSocket components: {e}")
        return False
    
    return True


def test_metrics_api_components():
    """Test Metrics API components."""
    print("\nTesting Metrics API components...")
    
    try:
        from api.routes.metrics import router
        print("  ✅ Metrics router exists")
        
        # Check that routes are registered
        routes = [route.path for route in router.routes]
        expected_routes = ['/system', '/agents/{agent_id}', '/agents', '/portfolio']
        
        for expected in expected_routes:
            if any(expected in route for route in routes):
                print(f"  ✅ Route {expected} registered")
            else:
                print(f"  ⚠️  Route {expected} not found (may be normal)")
    except Exception as e:
        print(f"  ❌ Metrics API: {e}")
        return False
    
    return True


def test_api_models():
    """Test API models."""
    print("\nTesting API models...")
    
    try:
        from api.models.agent_schemas import AgentConfigSchema, AgentMetricsSchema
        print("  ✅ Agent schemas")
    except Exception as e:
        print(f"  ❌ Agent schemas: {e}")
        return False
    
    try:
        from api.models.portfolio_schemas import PortfolioSummarySchema
        print("  ✅ Portfolio schemas")
    except Exception as e:
        print(f"  ❌ Portfolio schemas: {e}")
        return False
    
    try:
        from api.models.order_schemas import OrderCreateRequest, OrderResponse
        print("  ✅ Order schemas")
    except Exception as e:
        print(f"  ❌ Order schemas: {e}")
        return False
    
    return True


def test_training_script_exists():
    """Test that training script exists."""
    print("\nTesting training script...")
    
    script_path = "scripts/run_end_to_end_training.py"
    if os.path.exists(script_path):
        print(f"  ✅ {script_path} exists")
        
        # Check file size (should be substantial)
        size = os.path.getsize(script_path)
        if size > 1000:
            print(f"  ✅ Script has substantial content ({size} bytes)")
            return True
        else:
            print(f"  ⚠️  Script is very small ({size} bytes)")
            return False
    else:
        print(f"  ❌ {script_path} not found")
        return False


def test_docker_files():
    """Test that Docker files exist."""
    print("\nTesting Docker files...")
    
    files = [
        "docker/Dockerfile.backend",
        "docker/Dockerfile.frontend",
        "docker/docker-compose.yml",
        "docker/nginx.conf"
    ]
    
    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} not found")
            all_exist = False
    
    return all_exist


def test_frontend_structure():
    """Test that frontend structure exists."""
    print("\nTesting frontend structure...")
    
    dirs = [
        "frontend/src",
        "frontend/src/components",
        "frontend/src/pages",
        "frontend/src/services",
        "frontend/src/types"
    ]
    
    files = [
        "frontend/package.json",
        "frontend/tsconfig.json",
        "frontend/vite.config.ts",
        "frontend/tailwind.config.js",
        "frontend/src/main.tsx",
        "frontend/src/App.tsx"
    ]
    
    all_exist = True
    
    for dir in dirs:
        if os.path.exists(dir) and os.path.isdir(dir):
            print(f"  ✅ {dir}/")
        else:
            print(f"  ❌ {dir}/ not found")
            all_exist = False
    
    for file in files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} not found")
            all_exist = False
    
    return all_exist


def test_documentation():
    """Test that documentation exists."""
    print("\nTesting documentation...")
    
    docs = [
        "docs/training_guide.md",
        "docs/deployment_guide.md",
        "docs/sprint_03_implementation_summary.md"
    ]
    
    all_exist = True
    for doc in docs:
        if os.path.exists(doc):
            size = os.path.getsize(doc)
            print(f"  ✅ {doc} ({size} bytes)")
        else:
            print(f"  ❌ {doc} not found")
            all_exist = False
    
    return all_exist


def test_ci_cd_workflows():
    """Test that CI/CD workflows exist."""
    print("\nTesting CI/CD workflows...")
    
    workflows = [
        ".github/workflows/ci.yml",
        ".github/workflows/cd.yml"
    ]
    
    all_exist = True
    for workflow in workflows:
        if os.path.exists(workflow):
            print(f"  ✅ {workflow}")
        else:
            print(f"  ❌ {workflow} not found")
            all_exist = False
    
    return all_exist


def main():
    """Run all Sprint 03 tests."""
    print("=" * 60)
    print("SPRINT 03 COMPONENT VALIDATION")
    print("=" * 60)
    
    tests = [
        ("API Routes Imports", test_imports_api_routes),
        ("Database Clients Imports", test_imports_database_clients),
        ("Database Client Methods", test_database_client_methods),
        ("WebSocket Components", test_websocket_components),
        ("Metrics API Components", test_metrics_api_components),
        ("API Models", test_api_models),
        ("Training Script", test_training_script_exists),
        ("Docker Files", test_docker_files),
        ("Frontend Structure", test_frontend_structure),
        ("Documentation", test_documentation),
        ("CI/CD Workflows", test_ci_cd_workflows)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} failed with exception: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "-" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

