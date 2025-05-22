import cv2
import numpy as np

class Drawer:
    def __init__(self, neutral_color=(128, 128, 128)):
        self.neutral_color = neutral_color

    def prepare_canvas(self, image):
        """
        Prepares a canvas with extra space above the image.
        The canvas is twice the height of the original image.
        """
        h, w = image.shape[:2]
        canvas = np.full((h * 2, w, 3), self.neutral_color, dtype=np.uint8)
        canvas[h:, :] = image  # Place the original image at the bottom
        return canvas

    def plot_extrapolated_line(self, canvas, x1, y1, x2, y2, color=(0, 0, 255), thickness=4):
        """
        Plots a line on the original image and extends it into the upper half of the canvas.
        """
        h, w = canvas.shape[:2]
        original_height = h // 2  # Height of the original image

        # Adjust y-coordinates to account for the shifted image
        y1_shifted = y1 + original_height
        y2_shifted = y2 + original_height

        # Draw the original line segment on the lower part of the canvas
        cv2.line(canvas, (x1, y1_shifted), (x2, y2_shifted), color, thickness)

        # Calculate slope and intercept of the line
        if x2 != x1:  # Avoid division by zero
            slope = (y2_shifted - y1_shifted) / (x2 - x1)
            intercept = y1_shifted - slope * x1

            # Find intersection points with the boundaries
            y_top = 0  # Top boundary of the canvas
            x_top = int((y_top - intercept) / slope) if slope != 0 else x1

            y_bottom = h  # Bottom boundary of the upper half
            x_bottom = int((y_bottom - intercept) / slope) if slope != 0 else x1

        else:  # Handle vertical lines
            # For vertical lines, x remains constant
            x_top = x1
            x_bottom = x1
            y_bottom = h
            y_top = 0

        # Ensure the extrapolated points are within the canvas bounds
        x_top = max(0, min(w - 1, x_top))
        x_bottom = max(0, min(w - 1, x_bottom))

        # Draw the extrapolated line in the upper half
        cv2.line(canvas, (x_bottom, y_bottom), (x_top, y_top), (0, 200, 0), 1)

        return canvas

    def draw_contours(self, canvas, contours, color=(255, 0, 0), thickness=2, aspect_ratio_threshold=1.5):
        """
        Draws contours on the shifted image in the bottom half of the canvas.
        """
        h, w = canvas.shape[:2]
        original_height = h // 2  # Height of the original image

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                if w_rect < 100 and h_rect > 50:
                    continue
                aspect_ratio = float(w_rect) / float(h_rect) if h_rect > 0 else 0
                # Approximate the contour
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if aspect_ratio > 5:
                    continue

                # filter by number of vertices in the approximated polygon
                if (4 <= len(approx) <= 6):
                    continue

                # Adjust the contour's y-coordinates to account for the vertical shift
                shifted_contour = contour.copy()
                shifted_contour[:, :, 1] += original_height  # Shift y-coordinates downward

                # Draw the adjusted contour on the canvas
                cv2.drawContours(canvas, [shifted_contour], -1, color, thickness)

        return canvas