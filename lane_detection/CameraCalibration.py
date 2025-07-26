import numpy as np
import cv2
import glob
import matplotlib.image as mpimg

class CameraCalibration():
    """ Class that calibrates camera using chessboard images.

    Attributes:
        mtx (np.array): Camera matrix
        dist (np.array): Distortion coefficients
    """
    def __init__(self, image_dir, nx, ny, debug=False):
        fnames = glob.glob("{}/*".format(image_dir))
        objpoints = []
        imgpoints = []

        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        img_shape = None

        for fname in fnames:
            img = mpimg.imread(fname)
            if img_shape is None:
                img_shape = (img.shape[1], img.shape[0])

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                if debug:
                    # In a modular setup, displaying plots directly from __init__ might be less ideal
                    # but kept for consistency if debugging locally.
                    import matplotlib.pyplot as plt
                    img_copy = np.copy(img)
                    cv2.drawChessboardCorners(img_copy, (nx, ny), corners, ret)
                    plt.imshow(img_copy)
                    plt.title(f'Corners found in {fname.split("/")[-1]}')
                    plt.show()

        if img_shape is None:
            raise Exception("No images found or processed to determine image shape for calibration.")

        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

        if not ret:
            raise Exception("Camera calibration failed.")
        else:
            print("\n--- Camera Calibration Results ---")
            print(f"Camera matrix:\n{self.mtx}")
            print(f"Distortion coefficients:\n{self.dist}")

    def undistort(self, img):
        if not hasattr(self, 'mtx') or not hasattr(self, 'dist'):
            raise Exception("Camera is not calibrated. Please call the CameraCalibration constructor with calibration images first.")
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)