"""
Test script for the web application
"""

import requests
import json
import os
import sys

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    print("✓ Health check passed")

def test_main_page():
    """Test main page loads"""
    print("Testing main page...")
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert "Activity Recognition" in response.text
    print("✓ Main page loads correctly")

def test_static_files():
    """Test static files are served"""
    print("Testing static files...")
    css_response = requests.get(f"{BASE_URL}/static/css/style.css")
    assert css_response.status_code == 200
    assert "body" in css_response.text.lower()
    
    js_response = requests.get(f"{BASE_URL}/static/js/main.js")
    assert js_response.status_code == 200
    assert "function" in js_response.text.lower() or "const" in js_response.text.lower()
    print("✓ Static files served correctly")

def test_invalid_prediction():
    """Test prediction endpoint with invalid data"""
    print("Testing invalid prediction request...")
    response = requests.post(
        f"{BASE_URL}/predict/3dcnn_simple",
        json={"filename": "nonexistent.mp4"}
    )
    assert response.status_code == 404
    print("✓ Invalid prediction handled correctly")

def test_invalid_model():
    """Test prediction with invalid model"""
    print("Testing invalid model type...")
    response = requests.post(
        f"{BASE_URL}/predict/invalid_model",
        json={"filename": "test.mp4"}
    )
    assert response.status_code == 400
    print("✓ Invalid model type handled correctly")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Web Application")
    print("=" * 60)
    
    try:
        test_health()
        test_main_page()
        test_static_files()
        test_invalid_prediction()
        test_invalid_model()
        
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Make sure Flask app is running on port 5000")
        sys.exit(1)
    except AssertionError as e:
        print(f"ERROR: Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)









