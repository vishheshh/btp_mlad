"""
API Integration Test - Test REST API and WebSocket endpoints
"""
import requests
import time
import json


BASE_URL = "http://localhost:5000/api"


def test_api_endpoints():
    """Test all REST API endpoints."""
    print("="*60)
    print("API INTEGRATION TEST")
    print("="*60)
    
    # Test 1: Get status (before starting)
    print("\n1. Testing GET /api/status...")
    response = requests.get(f"{BASE_URL}/status")
    assert response.status_code == 200
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Simulation running: {data['simulation']['is_running']}")
    
    # Test 2: Start simulation
    print("\n2. Testing POST /api/simulation/start...")
    response = requests.post(f"{BASE_URL}/simulation/start", json={"speed": 100.0})
    assert response.status_code == 200
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Message: {data['message']}")
    
    # Wait a bit for simulation to process some data
    print("\n3. Waiting for simulation to process data...")
    time.sleep(2)
    
    # Test 3: Get status (after starting)
    print("\n4. Testing GET /api/status (after start)...")
    response = requests.get(f"{BASE_URL}/status")
    assert response.status_code == 200
    data = response.json()
    print(f"   Simulation running: {data['simulation']['is_running']}")
    print(f"   Speed: {data['simulation']['speed']}")
    print(f"   Data store hours: {data['data_store']['total_hours']}")
    
    # Test 4: Get forecast data
    print("\n5. Testing GET /api/forecast...")
    response = requests.get(f"{BASE_URL}/forecast?hours=10")
    assert response.status_code == 200
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Data points: {len(data['data'])}")
    if len(data['data']) > 0:
        print(f"   Sample: {data['data'][0]}")
    
    # Test 5: Get detections
    print("\n6. Testing GET /api/detections...")
    response = requests.get(f"{BASE_URL}/detections?limit=10")
    assert response.status_code == 200
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Detections: {data['count']}")
    
    # Test 6: Get alerts
    print("\n7. Testing GET /api/alerts...")
    response = requests.get(f"{BASE_URL}/alerts?limit=10")
    assert response.status_code == 200
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Alerts: {data['count']}")
    
    # Test 7: Get metrics
    print("\n8. Testing GET /api/metrics...")
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    data = response.json()
    print(f"   Status: {data['status']}")
    if 'metrics' in data and 'throughput_hours_per_sec' in data['metrics']:
        print(f"   Throughput: {data['metrics']['throughput_hours_per_sec']:.2f} hours/sec")
    
    # Test 8: Inject attack
    print("\n9. Testing POST /api/simulation/inject_attack...")
    attack_data = {
        "type": "PULSE",
        "start_hour": 5,
        "duration": 2,
        "magnitude": 0.2
    }
    response = requests.post(f"{BASE_URL}/simulation/inject_attack", json=attack_data)
    assert response.status_code == 200
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Message: {data['message']}")
    
    # Test 9: Set speed
    print("\n10. Testing POST /api/simulation/speed...")
    response = requests.post(f"{BASE_URL}/simulation/speed", json={"speed": 50.0})
    assert response.status_code == 200
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Speed: {data['speed']}")
    
    # Wait a bit more
    time.sleep(2)
    
    # Test 10: Stop simulation
    print("\n11. Testing POST /api/simulation/stop...")
    response = requests.post(f"{BASE_URL}/simulation/stop")
    assert response.status_code == 200
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Message: {data['message']}")
    
    print("\n" + "="*60)
    print("API INTEGRATION TEST COMPLETE")
    print("="*60)
    print("\nAll endpoints tested successfully!")
    print("\nNote: WebSocket testing requires a WebSocket client.")
    print("You can test WebSocket connections using a browser console or WebSocket client.")


if __name__ == "__main__":
    print("Make sure the API server is running:")
    print("  python api/api_server.py")
    print("\nPress Enter to start testing...")
    input()
    
    try:
        test_api_endpoints()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API server.")
        print("Please start the server first: python api/api_server.py")
    except AssertionError as e:
        print(f"\nERROR: Test failed - {e}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

