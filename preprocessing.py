import cv2
import numpy as np

class Preprocessor:
    def convert_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_blur(self, image, kernel_size=(15, 15), sigma=0):
        return cv2.GaussianBlur(image, kernel_size, sigma)

    def apply_binary_threshold(self, gray_image):
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

class MorphologyProcessor:
    def __init__(self, kernel_size=(3, 3)):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    def apply_morphological_opening(self, binary_image):
        return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, self.kernel)