import cv2

class Visualizer:
    def __init__(self, window_name="Video Stream"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)

    def display(self, image, frame_number=0, timestamp=0):
        if timestamp is None:
            timestamp = 0

        cv2.setWindowTitle(self.window_name, f'Frame: {frame_number}, Time: {timestamp:.2f}s')
        cv2.imshow(self.window_name, image)
        # cv2.imwrite(f"result_{frame_number}.png", image)

    def wait_key(self, delay=10):
        return cv2.waitKey(delay)