#!/usr/bin/env python3
"""
Test script for Trading Bot Web Interface
Tests the Flask API endpoints and basic functionality
"""

import requests
import json
import time

def test_api_endpoints():
    """Test basic API endpoints"""
    base_url = "http://localhost:8080"
    
    print("🤖 Testing Trading Bot Web Interface")
    print("=" * 50)
    
    # Test 1: Dashboard data
    print("\n📊 Testing Dashboard API...")
    try:
        response = requests.get(f"{base_url}/api/dashboard", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Dashboard API working")
            print(f"   Active Models: {data.get('activeModels', 'N/A')}")
            print(f"   ML Accuracy: {data.get('mlAccuracy', 'N/A')}%")
            print(f"   Last P&L: ${data.get('lastPnl', 'N/A')}")
        else:
            print(f"❌ Dashboard API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard API error: {e}")
    
    # Test 2: Datasets
    print("\n💾 Testing Datasets API...")
    try:
        response = requests.get(f"{base_url}/api/datasets", timeout=5)
        if response.status_code == 200:
            datasets = response.json()
            print(f"✅ Datasets API working")
            print(f"   Found {len(datasets)} datasets")
            for dataset in datasets[:3]:  # Show first 3
                print(f"   - {dataset.get('name', 'Unknown')}: {dataset.get('size', 'N/A')}")
        else:
            print(f"❌ Datasets API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Datasets API error: {e}")
    
    # Test 3: Models
    print("\n🧠 Testing Models API...")
    try:
        response = requests.get(f"{base_url}/api/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Models API working")
            print(f"   Found {len(models)} models")
            for model in models[:2]:  # Show first 2
                print(f"   - {model.get('name', 'Unknown')}: Active={model.get('active', False)}")
        else:
            print(f"❌ Models API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Models API error: {e}")
    
    # Test 4: System Status
    print("\n🔧 Testing System Status API...")
    try:
        response = requests.get(f"{base_url}/api/system/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ System Status API working")
            print(f"   System: {status.get('system', 'N/A')}")
            print(f"   Running Processes: {status.get('running_processes', 'N/A')}")
        else:
            print(f"❌ System Status API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ System Status API error: {e}")
    
    # Test 5: Web Interface (HTML)
    print("\n🌐 Testing Web Interface...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200 and 'Trading Bot' in response.text:
            print("✅ Web interface is accessible")
            print("   HTML page loads correctly")
        else:
            print(f"❌ Web interface failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Web interface error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")
    print(f"📱 Web interface available at: {base_url}")

if __name__ == "__main__":
    # Wait a moment for server to fully start
    time.sleep(2)
    test_api_endpoints()