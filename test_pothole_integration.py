#!/usr/bin/env python3
"""
Test script for pothole detection integration
"""

import os
import sys
sys.path.append('/Users/siddarthnachannagari/HazardEye')

from utils import load_pothole_model, detect_potholes_in_frame
import cv2
import numpy as np

def test_pothole_model():
    """Test if the pothole model loads correctly"""
    print("🔧 Testing pothole model loading...")
    
    model = load_pothole_model()
    if model is not None:
        print("✅ Pothole model loaded successfully!")
        print(f"📝 Model type: {type(model)}")
        return True
    else:
        print("❌ Failed to load pothole model")
        return False

def test_pothole_detection():
    """Test pothole detection on a dummy frame"""
    print("\n🔧 Testing pothole detection function...")
    
    model = load_pothole_model()
    if model is None:
        print("❌ Cannot test detection - model not loaded")
        return False
    
    # Create a dummy frame (640x480 RGB)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        pothole_count, annotated_frame = detect_potholes_in_frame(dummy_frame, model)
        print(f"✅ Detection function works! Found {pothole_count} potholes in dummy frame")
        print(f"📝 Annotated frame shape: {annotated_frame.shape}")
        return True
    except Exception as e:
        print(f"❌ Detection function failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting pothole detection integration test...\n")
    
    # Change to the correct directory
    os.chdir('/Users/siddarthnachannagari/HazardEye')
    
    # Test model loading
    model_test = test_pothole_model()
    
    # Test detection function
    detection_test = test_pothole_detection()
    
    print(f"\n📊 Test Results:")
    print(f"   Model Loading: {'✅ PASS' if model_test else '❌ FAIL'}")
    print(f"   Detection Function: {'✅ PASS' if detection_test else '❌ FAIL'}")
    
    if model_test and detection_test:
        print("\n🎉 All tests passed! Pothole detection is ready to use.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
