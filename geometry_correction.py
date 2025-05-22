import numpy as np
import cv2

class GeometryCorrector:
    def __init__(self, k1=0.2, k2=0.13):
        self.k1 = k1
        self.k2 = k2

    def apply_fisheye_correction(self, image):
        h, w = image.shape[:2]
        camera_matrix = np.array([[w / 2, 0, w / 2],
                                  [0, h / 2, h / 2],
                                  [0, 0, 1]], dtype=np.float32)

        dist_coeffs = np.array([self.k1, self.k2, 0, 0], dtype=np.float32)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), camera_matrix, (w, h), cv2.CV_16SC2
        )

        corrected_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        return corrected_image