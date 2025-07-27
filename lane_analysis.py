# Advanced Lane Detection System
# Based on Dt-Pham/Advanced-Lane-Lines repository

import cv2
import numpy as np
import sys
import os

def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)
    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255

class Thresholding:
    def __init__(self):
        pass

    def forward(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        v_channel = hsv[:,:,2]

        right_lane = threshold_rel(l_channel, 0.8, 1.0)
        right_lane[:,:750] = 0

        left_lane = threshold_abs(h_channel, 20, 30)
        left_lane &= threshold_rel(v_channel, 0.7, 1.0)
        left_lane[:,550:] = 0

        img2 = left_lane | right_lane
        return img2

class PerspectiveTransformation:
    def __init__(self):
        # Fixed coordinates for standard lane detection
        self.src = np.float32([(550, 460), (150, 720), (1200, 720), (770, 460)])
        self.dst = np.float32([(100, 0), (100, 720), (1100, 720), (1100, 0)])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, flags=cv2.INTER_LINEAR):
        height, width = img.shape[:2]
        # Use the original image size for transformation
        return cv2.warpPerspective(img, self.M, (width, height), flags=flags)

    def backward(self, img, flags=cv2.INTER_LINEAR):
        height, width = img.shape[:2]
        # Use the original image size for transformation
        return cv2.warpPerspective(img, self.M_inv, (width, height), flags=flags)

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []
        self.nwindows = 9
        self.margin = 100
        self.minpix = 50

    def forward(self, img):
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)
        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx&condy], self.nonzeroy[condx&condy]

    def extract_features(self, img):
        self.img = img
        self.window_height = int(img.shape[0]//self.nwindows)
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        assert(len(img.shape) == 2)
        out_img = np.dstack((img, img, img))
        histogram = hist(img)
        midpoint = histogram.shape[0]//2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height//2
        leftx, lefty, rightx, righty = [], [], [], []

        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)
            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)
            if len(good_left_x) > self.minpix:
                leftx_current = int(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = int(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)
        
        # Lower threshold for more forgiving detection
        if len(lefty) > 500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        elif self.left_fit is None:
            # Default left lane if none found
            self.left_fit = np.array([0.0002, -0.3, 200])
            
        if len(righty) > 500:
            self.right_fit = np.polyfit(righty, rightx, 2)
        elif self.right_fit is None:
            # Default right lane if none found
            self.right_fit = np.array([0.0002, -0.3, 1080])

        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))
        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))

        ploty = np.linspace(miny, maxy, img.shape[0])
        
        # Safe polynomial evaluation with null checks
        if self.left_fit is not None:
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        else:
            left_fitx = np.full_like(ploty, 200)  # Default left position
            
        if self.right_fit is not None:
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        else:
            right_fitx = np.full_like(ploty, 1080)  # Default right position

        for i, y in enumerate(ploty):
            l = int(np.clip(left_fitx[i], 0, img.shape[1]-1))
            r = int(np.clip(right_fitx[i], 0, img.shape[1]-1))
            y = int(y)
            cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

        return out_img

    def plot(self, out_img):
        # Only proceed if we have valid lane fits
        if self.left_fit is None or self.right_fit is None:
            # Just draw a simple message if no lanes detected
            cv2.putText(out_img, "Lane Detection Initializing...", org=(10, 50), 
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 0), thickness=2)
            return out_img
            
        lR, rR, pos = self.measure_curvature()
        value = None
        if abs(self.left_fit[0]) > abs(self.right_fit[0]):
            value = self.left_fit[0]
        else:
            value = self.right_fit[0]

        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')
        
        if len(self.dir) > 10:
            self.dir.pop(0)

        W = min(400, out_img.shape[1])  # Ensure widget doesn't exceed image width
        H = min(500, out_img.shape[0])  # Ensure widget doesn't exceed image height
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0,:] = [0, 0, 255]
        widget[-1,:] = [0, 0, 255]
        widget[:,0] = [0, 0, 255]
        widget[:,-1] = [0, 0, 255]
        out_img[:H, :W] = widget

        direction = max(set(self.dir), key = self.dir.count) if self.dir else 'F'
        msg = "Keep Straight Ahead"
        curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
        if direction == 'L':
            msg = "Left Curve Ahead"
        if direction == 'R':
            msg = "Right Curve Ahead"

        cv2.putText(out_img, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.putText(out_img, "Good Lane Keeping", org=(10, 400), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0, 255, 0), thickness=2)
        cv2.putText(out_img, "Vehicle is {:.2f} m away from center".format(pos), org=(10, 450), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.66, color=(255, 255, 255), thickness=2)
        return out_img

    def measure_curvature(self):
        # Return safe defaults if no lane fits available
        if self.left_fit is None or self.right_fit is None:
            return 1000, 1000, 0.0  # Large radius (straight), centered
            
        ym = 30/720
        xm = 3.7/700
        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym
        
        # Safe division with checks for zero
        left_denom = np.absolute(2*left_fit[0])
        right_denom = np.absolute(2*right_fit[0])
        
        if left_denom < 1e-6:
            left_curveR = 10000  # Very large radius (essentially straight)
        else:
            left_curveR = ((1 + (2*left_fit[0] *y_eval + left_fit[1])**2)**1.5) / left_denom
            
        if right_denom < 1e-6:
            right_curveR = 10000  # Very large radius (essentially straight)
        else:
            right_curveR = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / right_denom
            
        xl = np.dot(self.left_fit, [700**2, 700, 1])
        xr = np.dot(self.right_fit, [700**2, 700, 1])
        pos = (1280//2 - (xl+xr)//2)*xm
        return left_curveR, right_curveR, pos

class FindLaneLines:
    def __init__(self):
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        
        # Process the image through the pipeline
        processed_img = self.transform.forward(img)
        processed_img = self.thresholding.forward(processed_img)
        processed_img = self.lanelines.forward(processed_img)
        processed_img = self.transform.backward(processed_img)
        
        # Ensure both images have same dimensions and channels
        if len(processed_img.shape) == 2:  # Grayscale to RGB
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        
        # Make sure dimensions match
        if processed_img.shape != out_img.shape:
            processed_img = cv2.resize(processed_img, (out_img.shape[1], out_img.shape[0]))
        
        out_img = cv2.addWeighted(out_img, 1, processed_img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_video(self, input_path, output_path):
        print(f"🎬 Processing video: {input_path}")
        print(f"📁 Output will be saved to: {output_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {input_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        print(f"📊 Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.forward(frame_rgb)
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            out.write(processed_frame_bgr)
            
            frame_count += 1
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        cap.release()
        out.release()
        print(f"✅ Lane detection video saved to: {output_path}")
        return output_path

def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = 'videoplayback3.mp4'
    
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at {input_path}")
        return
    
    base_name = os.path.splitext(input_path)[0]
    output_path = base_name + '_lane_detection.mp4'
    
    print(f"📹 Advanced Lane Detection System")
    print(f"🎯 Based on Dt-Pham/Advanced-Lane-Lines repository")
    print(f"📂 Input: {input_path}")
    print(f"📂 Output: {output_path}")
    
    findLaneLines = FindLaneLines()
    
    try:
        findLaneLines.process_video(input_path, output_path)
        print(f"\n✅ SUCCESS: Lane detection completed!")
        print(f"🎬 Processed video saved to: {output_path}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
