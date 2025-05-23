# Configuration dictionary
CONFIG = {
    "fisheye_correction": {"enabled": True, "k1": 0.2, "k2": 0.13},
    "preprocessing": {
        "grayscale": {"enabled": True},
        "binary_threshold": {"enabled": True},
        "morphological_opening": {"enabled": True, "kernel_size": (3, 3)}
    },
    "line_detection": {
        "enabled": True,
        "threshold": 100,
        "min_line_length": 150,
        "max_line_gap": 30
    },
    "contour_detection": {"enabled": True},
    "contour_selection": { # New section for contour selection parameters
        "enabled": True,
        "min_area": 1000,
        "max_aspect_ratio": 5.0,
        "min_height_ratio": 0.8, # Contour height must be at least 80% of image height
        "max_approx_vertices": 6, # Discard if too many vertices
        "min_approx_vertices": 4, # Discard if too few vertices
        "top_bottom_tolerance": 10 # Pixels from top/bottom to consider 'spanning'
    },
    "drawing": {
        "neutral_color": (128, 128, 128),
        "lines": {"enabled": True, "color": (30, 200, 30), "thickness": 2}, # For Hough lines
        "contours": {"enabled": True, "color": (255, 0, 0), "thickness": 2, "aspect_ratio_threshold": 3}, # For selected contour outline
        "contour_fitted_line": {"enabled": True, "color": (0, 255, 255), "thickness": 3} # For line fitted to selected contour
    }
}