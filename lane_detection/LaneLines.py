import numpy as np
import cv2
import matplotlib.image as mpimg
# No need for matplotlib.pyplot here as it's not used directly in the class methods anymore

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    """ Class containing information about detected lane lines.

    Attributes:
        left_fit (np.array): Coefficients of a polynomial that fit left lane line
        right_fit (np.array): Coefficients of a polynomial that fit right lane line
        binary (np.array): The last processed binary image
        nonzero (tuple): Result of img.nonzero(), (array of y_indices, array of x_indices)
        nonzerox (np.array): x coordinates of all non-zero pixels
        nonzeroy (np.array): y coordinates of all non-zero pixels
        clear_visibility (bool): Flag indicating if lane lines were clearly detected
        dir (list): Stores recent turn directions ('F', 'L', 'R') for smoothing
        left_curve_img (np.array): Image for left turn indicator
        right_curve_img (np.array): Image for right turn indicator
        keep_straight_img (np.array): Image for straight ahead indicator
        nwindows (int): Number of sliding windows for lane detection
        margin (int): Width of the sliding windows
        minpix (int): Minimum number of pixels to recenter a window
    """
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []

        try:
            # Assuming these are in the same directory as where the main script is run from
            self.left_curve_img = mpimg.imread('left_turn.png')
            self.right_curve_img = mpimg.imread('right_turn.png')
            self.keep_straight_img = mpimg.imread('straight.png')

            # Convert to 8-bit unsigned for drawing with OpenCV
            self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.keep_straight_img = cv2.normalize(src=self.keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        except FileNotFoundError as e:
            print(f"Warning: Indicator image not found ({e}). Ensure 'left_turn.png', 'right_turn.png', and 'straight.png' are in the script's execution directory.")
            # Create dummy images to prevent errors later
            # For 4-channel (RGBA) images, make sure dummy also has 4 channels
            self.left_curve_img = np.zeros((100, 100, 4), dtype=np.uint8)
            self.right_curve_img = np.zeros((100, 100, 4), dtype=np.uint8)
            self.keep_straight_img = np.zeros((100, 100, 4), dtype=np.uint8)


        # HYPERPARAMETERS
        self.nwindows = 9
        self.margin = 100
        self.minpix = 50

    def forward(self, img):
        # img here is the BINARY WARPED image
        self.extract_features(img)
        
        # fit_poly will take the binary image, internally create a 3-channel image to draw on,
        # and return that 3-channel image with the lane area and lines drawn.
        out_img_with_lanes = self.fit_poly(img)
        
        # plot will take this 3-channel image (with lanes drawn) and add the dashboard info.
        # It assumes the image passed to it is the warped image where calculations were made.
        final_warped_output = self.plot(out_img_with_lanes)
        
        return final_warped_output


    def pixels_in_window(self, center, margin, height):
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx&condy], self.nonzeroy[condx&condy]

    def extract_features(self, img):
        self.img = img # Storing the binary image
        self.window_height = np.int32(img.shape[0]//self.nwindows)

        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        assert(len(img.shape) == 2) # Ensure it's a 2D (binary) image

        # out_img here is for visualization of sliding windows; it's 3-channel
        out_img = np.dstack((img, img, img)) * 255 

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
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        leftx = np.array(leftx)
        lefty = np.array(lefty)
        rightx = np.array(rightx)
        righty = np.array(righty)

        return leftx, lefty, rightx, righty, out_img # out_img for debug visualization only

    def fit_poly(self, img):
        leftx, lefty, rightx, righty, _ = self.find_lane_pixels(img) # Discard out_img from find_lane_pixels

        # Create an RGB image to draw on
        # If img is 0/1, dstack creates 0/0/0 or 1/1/1, then multiply by 255.
        color_output = np.dstack((img, img, img)) * 255

        if len(lefty) > self.minpix * 5:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        elif self.left_fit is None:
            print("Warning: Not enough left lane pixels to fit. Using previous fit if available.")
            self.clear_visibility = False

        if len(righty) > self.minpix * 5:
            self.right_fit = np.polyfit(righty, rightx, 2)
        elif self.right_fit is None:
            print("Warning: Not enough right lane pixels to fit. Using previous fit if available.")
            self.clear_visibility = False

        if self.left_fit is None or self.right_fit is None:
            print("Error: Could not fit both lane lines. Returning blank image.")
            return np.zeros_like(color_output) # Return a black image if fitting failed

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        # Create an image for the lane area, then fill it
        warp_zero = np.zeros_like(img).astype(np.uint8)
        color_warp_fill = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp_fill, np.int_([pts]), (0,255, 0)) # Green lane area

        # Overlay the green filled lane onto the color_output (which is just the binary image made RGB)
        # This will make the background black and the lane area green.
        # Alternatively, if you want to keep the original binary image's context:
        # result = cv2.addWeighted(color_output, 1, color_warp_fill, 0.3, 0)
        # For this setup, we just want the green fill on a black background
        # and then we'll add the lines/points.
        color_output = color_warp_fill # Now color_output is the green filled lane area

        # Draw the lane lines themselves (yellow for left, blue for right)
        for i in range(len(ploty)):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(ploty[i])
            if 0 <= l < img.shape[1]:
                cv2.circle(color_output, (l, y), 3, (255, 255, 0), -1) # Yellow dot
            if 0 <= r < img.shape[1]:
                cv2.circle(color_output, (r, y), 3, (0, 0, 255), -1) # Blue dot

        return color_output # This image now has green fill and colored lines/points

    def plot(self, out_img):
        # out_img here is the warped image with lanes drawn (from fit_poly)
        np.set_printoptions(precision=6, suppress=True)

        if self.left_fit is None or self.right_fit is None:
            print("Cannot measure curvature: Lane fits are not available.")
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

        # Create a widget area (black background with red border)
        W = out_img.shape[1] // 3
        H = out_img.shape[0] // 3

        overlay = out_img.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
        alpha = 0.5
        out_img = cv2.addWeighted(overlay, alpha, out_img, 1 - alpha, 0)
        cv2.rectangle(out_img, (0, 0), (W-1, H-1), (0, 0, 255), 2)

        direction = max(set(self.dir), key = self.dir.count)
        msg = "Keep Straight Ahead"
        curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))

        # Ensure paste location for icons is within image bounds
        offset_x = W // 2 - 50 # Centered horizontally, 50 is half of 100x100 default dummy image
        offset_y = 10 # Some padding from top

        if direction == 'L':
            if self.left_curve_img.shape[2] == 4: # RGBA
                y_idx, x_idx = self.left_curve_img[:,:,3].nonzero()
                out_img[y_idx + offset_y, x_idx + offset_x] = self.left_curve_img[y_idx, x_idx, :3]
            msg = "Left Curve Ahead"
        elif direction == 'R':
            if self.right_curve_img.shape[2] == 4: # RGBA
                y_idx, x_idx = self.right_curve_img[:,:,3].nonzero()
                out_img[y_idx + offset_y, x_idx + offset_x] = self.right_curve_img[y_idx, x_idx, :3]
            msg = "Right Curve Ahead"
        elif direction == 'F':
            if self.keep_straight_img.shape[2] == 4: # RGBA
                y_idx, x_idx = self.keep_straight_img[:,:,3].nonzero()
                out_img[y_idx + offset_y, x_idx + offset_x] = self.keep_straight_img[y_idx, x_idx, :3]

        cv2.putText(out_img, msg, org=(10, H // 2 + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)
        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, org=(10, H // 2 + 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)

        cv2.putText(
            out_img,
            "Good Lane Keeping",
            org=(10, H - 70),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 255, 0),
            thickness=2)

        cv2.putText(
            out_img,
            "Vehicle is {:.2f} m away from center".format(pos),
            org=(10, H - 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=2)

        return out_img

    def measure_curvature(self):
        if self.left_fit is None or self.right_fit is None:
            print("Warning: Cannot measure curvature, lane fits are missing.")
            return 0, 0, 0

        ym_per_pix = 30/720
        xm_per_pix = 3.7/700

        left_fit_cr = self.left_fit.copy()
        right_fit_cr = self.right_fit.copy()

        y_eval_pixel = 700 # Pixel value near bottom of image
        y_eval_meters = y_eval_pixel * ym_per_pix # Convert pixel to meters

        left_curveR = ((1 + (2*left_fit_cr[0]*y_eval_meters + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curveR = ((1 + (2*right_fit_cr[0]*y_eval_meters + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        image_center_pixel = 1280 / 2

        xl_bottom = self.left_fit[0]*y_eval_pixel**2 + self.left_fit[1]*y_eval_pixel + self.left_fit[2]
        xr_bottom = self.right_fit[0]*y_eval_pixel**2 + self.right_fit[1]*y_eval_pixel + self.right_fit[2]

        lane_center_position_pixel = (xl_bottom + xr_bottom) / 2
        pos_from_center = (image_center_pixel - lane_center_position_pixel) * xm_per_pix
        
        return left_curveR, right_curveR, pos_from_center