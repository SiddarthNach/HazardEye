#!/usr/bin/env python3
"""
Test the new pothole-weighted scoring algorithm
"""

import sys
import os
sys.path.append('/Users/siddarthnachannagari/HazardEye')
os.chdir('/Users/siddarthnachannagari/HazardEye')

from utils import calculate_safety_score

def test_scoring_with_37_potholes():
    """Test scoring with the user's example: 37 potholes detected"""
    
    # Simulate metrics similar to user's test case
    test_metrics = {
        'lane_detection_score': 2701,  # Perfect lane detection
        'curve_warnings': 0,
        'center_deviation_warnings': 0,
        'pothole_detections': 37,  # 37 potholes detected
        'pothole_frames': 38,  # 1.4% of frames
        'total_frames': 2701,
        'processed_frames': 2701,
        'average_curvature': [1000, 1200, 1500],  # Good curvature
        'average_position_deviation': [0.1, 0.2, 0.15],  # Good positioning
        'risk_level': 'LOW'  # Will be overridden
    }
    
    print("🧪 Testing New Scoring Algorithm")
    print("=" * 50)
    print(f"Input Metrics:")
    print(f"  📹 Total frames: {test_metrics['total_frames']}")
    print(f"  🛣️ Lane detection success: {test_metrics['lane_detection_score']} ({test_metrics['lane_detection_score']/test_metrics['total_frames']*100:.1f}%)")
    print(f"  🕳️ Potholes detected: {test_metrics['pothole_detections']}")
    print(f"  📊 Pothole frame rate: {test_metrics['pothole_frames']/test_metrics['total_frames']*100:.1f}%")
    print(f"  ⚠️ Curve warnings: {test_metrics['curve_warnings']}")
    print(f"  🚗 Position warnings: {test_metrics['center_deviation_warnings']}")
    
    score_data = calculate_safety_score(test_metrics)
    
    print(f"\n📊 Results:")
    print(f"  🏆 Final Safety Score: {score_data['score']}/100")
    print(f"  ⚠️ Risk Level: {score_data['risk_level']}")
    print(f"  🛣️ Lane Component (30%): {score_data['lane_component_score']}/100")
    print(f"  🕳️ Pothole Component (70%): {score_data['pothole_component_score']}/100")
    print(f"  🔢 Weighted Calculation: ({score_data['lane_component_score']} × 0.3) + ({score_data['pothole_component_score']} × 0.7) = {score_data['score']}")
    
    print(f"\n✅ Expected Behavior:")
    print(f"  - With 37 potholes, score should be LOW (< 50)")
    print(f"  - Risk level should be HIGH or CRITICAL")
    print(f"  - Pothole component should dominate the score")
    
    return score_data

if __name__ == "__main__":
    test_scoring_with_37_potholes()
