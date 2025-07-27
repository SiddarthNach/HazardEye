#!/usr/bin/env python3
"""
Test script for the Road Danger Detection App
Run this to verify the app works before deploying
"""

import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from lane_analysis import FindLaneLines
        print("✅ Lane analysis imported successfully")
    except ImportError as e:
        print(f"❌ Lane analysis import failed: {e}")
        return False
    
    try:
        from utils import analyze_road_safety
        print("✅ Utils imported successfully")
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test if required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'app.py',
        'utils.py',
        'lane_analysis.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_temp_directory():
    """Test if temp directory can be created"""
    print("\n📂 Testing temp directory...")
    
    try:
        os.makedirs("temp", exist_ok=True)
        print("✅ Temp directory created/verified")
        return True
    except Exception as e:
        print(f"❌ Failed to create temp directory: {e}")
        return False

def main():
    print("🚀 Road Danger Detection App - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_file_structure),
        ("Temp Directory Test", test_temp_directory)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your app is ready to run.")
        print("\nTo start the app, run:")
        print("streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please fix the issues above before running the app.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
