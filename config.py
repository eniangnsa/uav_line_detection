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
    "drawing": {
        "lines": {"enabled": True, "color": (30, 200, 30), "thickness": 2},
        "contours": {"enabled": True, "color": (255, 0, 0), "thickness": 2, "aspect_ratio_threshold": 3}
    }
}
