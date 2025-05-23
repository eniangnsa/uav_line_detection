import cv2
import numpy as np

class ContourDetector:
    """
    Detects contours in a binary image.
    """
    def detect(self, binary):
        """
        Finds external contours in a binary image.

        Args:
            binary (numpy.ndarray): The binary input image.

        Returns:
            list: A list of detected contours.
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

class LineDetector:
    """
    Detects lines in an edge map using the Probabilistic Hough Line Transform.
    """
    def detect(self, edges, threshold=100, min_line_length=150, max_line_gap=30):
        """
        Applies Hough Line Transform to detect lines.

        Args:
            edges (numpy.ndarray): The edge map image.
            threshold (int): Accumulator threshold parameter.
            min_line_length (int): Minimum line length. Line segments shorter than this are rejected.
            max_line_gap (int): Maximum allowed gap between points on the same line to link them.

        Returns:
            numpy.ndarray: An array of lines in the format [[x1, y1, x2, y2]], or None if no lines are found.
        """
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold,
                                minLineLength=min_line_length, maxLineGap=max_line_gap)
        return lines

class ContourSelector:
    """
    Selects the 'best' contour from a collection based on specific criteria,
    and fits a line to the selected contour.
    """
    def __init__(self, min_area=1000, max_aspect_ratio=5, min_height_ratio=0.8,
                 max_approx_vertices=6, min_approx_vertices=4,
                 top_bottom_tolerance=10):
        """
        Initializes the ContourSelector with filtering criteria.

        Args:
            min_area (int): Minimum contour area to consider.
            max_aspect_ratio (float): Maximum aspect ratio (width/height) to consider.
            min_height_ratio (float): Minimum ratio of contour height to image height for 'top-to-bottom' check.
            max_approx_vertices (int): Maximum number of vertices in approximated polygon to discard.
            min_approx_vertices (int): Minimum number of vertices in approximated polygon to discard.
            top_bottom_tolerance (int): Pixel tolerance for a contour to be considered 'top-to-bottom'.
        """
        self.min_area = min_area
        self.max_aspect_ratio = max_aspect_ratio
        self.min_height_ratio = min_height_ratio
        self.max_approx_vertices = max_approx_vertices
        self.min_approx_vertices = min_approx_vertices
        self.top_bottom_tolerance = top_bottom_tolerance

    def select_and_process_contour(self, contours, image_height, image_width):
        """
        Selects the best contour based on criteria, fits a line to it, and extends the line.

        Criteria:
        1. Discard contours based on size, aspect ratio, and approximated polygon vertices.
        2. Find contours that run all the way from top to bottom.
        3. If multiple such contours, select the narrowest.
        4. If none run from top to bottom, return None.
        5. If a contour is returned, fit a line to it and extend that line.

        Args:
            contours (list): A list of contours detected by cv2.findContours.
            image_height (int): The height of the original image.
            image_width (int): The width of the original image.

        Returns:
            tuple: (best_contour, fitted_line_points) where best_contour is the selected
                   contour and fitted_line_points is a tuple (x1, y1, x2, y2) representing
                   the extended line, or (None, None) if no suitable contour is found.
        """
        candidate_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w_rect, h_rect = cv2.boundingRect(contour)

            # Gate 1: Discard based on specific width/height ranges (from original code)
            if w_rect < 100 and h_rect > 50:
                continue

            aspect_ratio = float(w_rect) / float(h_rect) if h_rect > 0 else 0

            # Gate 2: Discard based on aspect ratio
            if aspect_ratio > self.max_aspect_ratio:
                continue

            # Approximate the contour (used for vertex count gate)
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Gate 3: Discard by number of vertices in the approximated polygon
            if self.min_approx_vertices <= len(approx) <= self.max_approx_vertices:
                continue

            # Check for top-to-bottom spanning
            is_top_to_bottom = (y <= self.top_bottom_tolerance and
                                (y + h_rect) >= (image_height - self.top_bottom_tolerance))

            if is_top_to_bottom:
                candidate_contours.append((contour, w_rect)) # Store contour and its width

        if not candidate_contours:
            return None, None

        # Select the narrowest contour among candidates
        candidate_contours.sort(key=lambda x: x[1]) # Sort by width
        best_contour = candidate_contours[0][0]

        # Fit a line to the best contour
        # [vx, vy, x, y] where (x,y) is a point on the line, (vx,vy) is a unit vector along the line
        [vx, vy, x, y] = cv2.fitLine(best_contour, cv2.DIST_L2, 0, 0.01, 0.01)

        # Extend the line to the top and bottom of the image
        # Handle vertical lines separately to avoid division by zero
        if vy != 0:
            y_top = 0
            x_top = int(x - vx * (y - y_top) / vy)

            y_bottom = image_height - 1
            x_bottom = int(x + vx * (y_bottom - y) / vy)
        else: # Vertical line
            x_top = int(x)
            x_bottom = int(x)
            y_top = 0
            y_bottom = image_height - 1

        # Ensure x-coordinates are within image bounds
        x_top = max(0, min(image_width - 1, x_top))
        x_bottom = max(0, min(image_width - 1, x_bottom))

        fitted_line_points = (x_top, y_top, x_bottom, y_bottom)

        return best_contour, fitted_line_points