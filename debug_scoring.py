#!/usr/bin/env python3
"""
Debug script to test the safety scoring system
"""

import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils import analyze_road_safety, calculate_safety_score

def test_scoring_system():
    """Test the scoring system with sample data"""
    print("🧪 Testing Safety Scoring System...")
    
    # Test calculate_safety_score with sample metrics
    test_metrics = {
        'lane_detection_score': 800,  # 800 successful frames
        'curve_warnings': 50,
        'center_deviation_warnings': 30,
        'total_frames': 1000,
        'processed_frames': 1000,
        'average_curvature': [1000, 1200, 800, 1500, 900],  # Sample curvature values
        'average_position_deviation': [0.2, 0.5, 0.8, 0.3, 0.1]  # Sample deviation values
    }
    
    print("\n📊 Sample metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")
    
    # Test the scoring function
    try:
        score_result = calculate_safety_score(test_metrics)
        print(f"\n✅ Scoring calculation successful!")
        print(f"📈 Results:")
        for key, value in score_result.items():
            print(f"  {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Scoring calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_video():
    """Test with an actual video file if available"""
    print("\n🎬 Testing with actual video...")
    
    # Look for available video files
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        import glob
        video_files.extend(glob.glob(ext))
    
    if not video_files:
        print("❌ No video files found for testing")
        return False
    
    test_video = video_files[0]
    print(f"📹 Testing with: {test_video}")
    
    try:
        # Test the full analysis pipeline
        result = analyze_road_safety(test_video)
        print(f"✅ Analysis completed: {result['analysis_complete']}")
        if result['analysis_complete']:
            print(f"📊 Safety Score: {result['safety_score']}")
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        return result['analysis_complete']
    except Exception as e:
        print(f"❌ Video analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 Safety Scoring Debug Suite")
    print("=" * 50)
    
    # Test 1: Scoring calculation
    scoring_test = test_scoring_system()
    
    # Test 2: Video analysis (if available)
    video_test = test_with_video()
    
    print("\n" + "=" * 50)
    print("📊 Debug Results:")
    print(f"  Scoring System: {'✅ PASSED' if scoring_test else '❌ FAILED'}")
    print(f"  Video Analysis: {'✅ PASSED' if video_test else '❌ FAILED'}")
    
    if scoring_test and video_test:
        print("\n🎉 All tests passed! The scoring system should work correctly.")
    elif scoring_test:
        print("\n⚠️ Scoring works but video analysis has issues. Check video file format or lane detection.")
    else:
        print("\n🚨 Critical issues found. Check the error messages above.")

if __name__ == "__main__":
    main()
