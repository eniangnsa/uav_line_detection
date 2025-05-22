import cv2
import numpy as np

class ContourDetector:
    def detect(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

class LineDetector:
    def detect(self, edges, threshold=100, min_line_length=150, max_line_gap=30):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold,
                                minLineLength=min_line_length, maxLineGap=max_line_gap)
        return lines