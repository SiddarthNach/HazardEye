#!/usr/bin/env python3

"""
Simple script to run lane detection on videoplayback2.mp4 using the working pipeline.
"""

from lane_detection_pipeline import FindLaneLines

def main():
    print("🎬 Running lane detection on videoplayback2.mp4...")
    
    # Initialize the lane detection pipeline
    try:
        lane_detector = FindLaneLines()
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        return
    
    # Process the video
    input_video = "../videoplayback2.mp4"
    output_video = "videoplayback2_lane_detection_output.mp4"
    
    try:
        lane_detector.process_video(input_video, output_video)
        print(f"✅ SUCCESS: Video processed and saved to {output_video}")
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
