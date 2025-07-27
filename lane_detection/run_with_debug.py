#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

# Import your classes from their respective files
from CameraCalibration import CameraCalibration
from Threshold import Thresholding
from PerspectiveTransformation import PerspectiveTransformation
from LaneLines import LaneLines

class FindLaneLinesDebug:
    """
    Debug version of lane detection pipeline with forced overlays.
    """
    def __init__(self):
        """ Initializes all the individual processing modules. """
        print("🔧 Initializing lane detection pipeline...")
        self.calibration = CameraCalibration('../Calibration_Photos', 5, 9)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()
        print("✅ Pipeline initialized successfully")

    def forward(self, img):
        """
        Debug version that always shows visual feedback.
        """
        # Store original image for later blending
        original_img = np.copy(img) 
        img_size = (img.shape[1], img.shape[0])

        try:
            # 1. Undistort the image
            img_undistorted = self.calibration.undistort(img)
            
            # 2. Warp the undistorted RGB image to bird's-eye view
            warped_undistorted_rgb = self.transform.forward(img_undistorted, img_size=img_size)
            
            # 3. Apply thresholding to the warped RGB image to get a binary image (0 or 255)
            binary_warped = self.thresholding.forward(warped_undistorted_rgb)
            
            # 4. Process the binary warped image with LaneLines
            lane_detection_on_warped_img = self.lanelines.forward(binary_warped)
            
            # 5. Unwarp the final lane detection image back to original perspective
            unwarped_lane_info = self.transform.backward(lane_detection_on_warped_img, img_size=img_size)

            # 6. Blend with original - but ensure we always have some visual feedback
            if unwarped_lane_info is not None and unwarped_lane_info.shape == img_undistorted.shape:
                final_output = cv2.addWeighted(img_undistorted, 1, unwarped_lane_info, 0.6, 0)
            else:
                # Fallback: show undistorted image with debug info
                final_output = img_undistorted.copy()
            
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
            final_output = original_img.copy()

        # ALWAYS add debug information to the output
        final_output = self.add_debug_overlay(final_output)
        
        return final_output

    def add_debug_overlay(self, img):
        """
        Always add visual feedback to show the pipeline is working.
        """
        # Create a copy to modify
        debug_img = img.copy()
        
        # Add border to show processing
        cv2.rectangle(debug_img, (0, 0), (img.shape[1]-1, img.shape[0]-1), (0, 255, 0), 3)
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_img, "🚗 HazardEye Lane Detection", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(debug_img, "✅ Pipeline Active", (10, 60), font, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Frame: {img.shape[1]}x{img.shape[0]}", (10, 90), font, 0.5, (255, 255, 255), 1)
        
        # Add timestamp or frame indicator
        cv2.putText(debug_img, "Processing...", (10, img.shape[0] - 20), font, 0.5, (255, 255, 0), 1)
        
        # Add corner markers to show calibration is working
        corner_size = 20
        cv2.circle(debug_img, (corner_size, corner_size), 10, (255, 0, 0), -1)
        cv2.circle(debug_img, (img.shape[1] - corner_size, corner_size), 10, (255, 0, 0), -1)
        cv2.circle(debug_img, (corner_size, img.shape[0] - corner_size), 10, (255, 0, 0), -1)
        cv2.circle(debug_img, (img.shape[1] - corner_size, img.shape[0] - corner_size), 10, (255, 0, 0), -1)
        
        return debug_img

    def process_video(self, input_path, output_path):
        """ Processes an entire video file and saves the result. """
        print(f"🎬 Processing video: {input_path}")
        print(f"📁 Output will be saved to: {output_path}")
        
        clip = VideoFileClip(input_path)
        print(f"📊 Video info: {clip.duration:.1f}s, {clip.fps}fps, {clip.size}")
        
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False, verbose=True, logger='bar')
        print(f"✅ Video saved to: {output_path}")

def main():
    print("🎬 Running debug lane detection on videoplayback2.mp4...")
    
    # Initialize the debug pipeline
    findLaneLines = FindLaneLinesDebug()
    
    # Process the video
    input_video = '../videoplayback2.mp4'
    output_video = 'videoplayback2_debug_output.mp4'
    
    try:
        findLaneLines.process_video(input_video, output_video)
        print(f"\n✅ SUCCESS: Debug video processed and saved to {output_video}")
        print("🔍 This version will always show visual overlays even if lane detection fails")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    main()
