#!/usr/bin/env python3
"""
Simple test script to verify HazardEye deployment
"""
import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing HazardEye imports...")
    
    try:
        import streamlit as st
        print("✅ streamlit - OK")
    except ImportError as e:
        print(f"❌ streamlit - FAILED: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv-python - OK")
    except ImportError as e:
        print(f"❌ opencv-python - FAILED: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy - OK")
    except ImportError as e:
        print(f"❌ numpy - FAILED: {e}")
        return False
    
    try:
        import boto3
        print("✅ boto3 - OK")
    except ImportError as e:
        print(f"❌ boto3 - FAILED: {e}")
        return False
    
    # Test utils import (the critical one)
    try:
        try:
            from utils_fallback import analyze_road_safety, save_uploaded_file, create_user_table, add_user, login_user
            print("✅ utils_fallback - OK (preferred)")
        except ImportError:
            from utils import analyze_road_safety, save_uploaded_file, create_user_table, add_user, login_user
            print("✅ utils - OK (fallback)")
    except ImportError as e:
        print(f"❌ utils/utils_fallback - FAILED: {e}")
        return False
    
    # Test optional imports
    try:
        from ultralytics import YOLO
        print("✅ ultralytics - OK (YOLO available)")
    except ImportError:
        print("⚠️ ultralytics - Not available (will use OpenCV fallback)")
    
    try:
        from aws_config import AWSConfig
        print("✅ aws_config - OK")
    except ImportError:
        print("⚠️ aws_config - Not available (S3 disabled)")
    
    try:
        from lane_analysis import FindLaneLines
        print("✅ lane_analysis - OK")
    except ImportError as e:
        print(f"⚠️ lane_analysis - Not available: {e}")
    
    return True

def test_file_structure():
    """Test that required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'app.py',
        'requirements.txt'
    ]
    
    optional_files = [
        'utils.py',
        'utils_fallback.py',
        'lane_analysis.py',
        'aws_config.py',
        'users.db'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - OK")
        else:
            print(f"❌ {file} - MISSING (CRITICAL)")
            return False
    
    for file in optional_files:
        if os.path.exists(file):
            print(f"✅ {file} - OK")
        else:
            print(f"⚠️ {file} - Missing (optional)")
    
    return True

def test_environment():
    """Test environment variables"""
    print("\n🔧 Testing environment...")
    
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    env_vars = ['AWS_DEFAULT_REGION', 'S3_BUCKET_NAME', 'PORT', 'PYTHONPATH']
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        print(f"{var}: {value}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 HazardEye Deployment Test")
    print("=" * 40)
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_file_structure():
        success = False
    
    test_environment()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ ALL TESTS PASSED - Deployment should work!")
    else:
        print("❌ SOME TESTS FAILED - Check errors above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
