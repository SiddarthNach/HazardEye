import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from IPython.display import HTML
from IPython.core.display import Video
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Threshold import Thresholding
from PerspectiveTransformation import *
from LaneLines import *

class FindLaneLines:
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('../Calibration_Photos', 5, 9)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)

def main():
    # Use the working pipeline approach instead
    print("Using the working lane_detection_pipeline.py approach...")
    print("Please run: python lane_detection_pipeline.py")
    print("The pipeline is already configured for videoplayback3.mp4")
    print("To change to videoplayback2.mp4, edit the VIDEO_PATH in lane_detection_pipeline.py")

if __name__ == "__main__":
    main()