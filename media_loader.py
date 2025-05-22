import os
import cv2

class MediaLoader:
    def __init__(self, input_path):
        self.input_path = input_path
        self.is_video = os.path.isfile(input_path)
        self.frame_count = 0
        self.image_files = []

        if self.is_video:
            # Video file
            self.cap = cv2.VideoCapture(input_path)
            if not self.cap.isOpened():
                raise ValueError(f"Error opening video file: {input_path}")
        else:
            # Image folder
            self.image_files = sorted(
                [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            )
            if not self.image_files:
                raise ValueError(f"No valid image files found in folder: {input_path}")

    def get_next_frame(self):
        if self.is_video:
            # Process video frame
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
                return frame, self.frame_count, timestamp
            else:
                return None, None, None
        else:
            # Process image file
            if self.frame_count < len(self.image_files):
                image_path = os.path.join(self.input_path, self.image_files[self.frame_count])
                frame = cv2.imread(image_path)

                if frame is None:
                    raise ValueError(f"Error reading image file: {image_path}")

                # Resize image if necessary
                h, w = frame.shape[:2]
                if h > 500 or w > 800:
                    scale_factor = 500 / h  # Calculate scale factor to make height 500
                    new_width = int(w * scale_factor)
                    new_height = 500
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                self.frame_count += 1
                return frame, self.frame_count, None  # No timestamp for images
            else:
                return None, None, None

    def release(self):
        if self.is_video:
            self.cap.release()