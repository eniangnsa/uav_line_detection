import os
import argparse
import cv2
from config import CONFIG
from media_loader import MediaLoader
from geometry_correction import GeometryCorrector
from preprocessing import Preprocessor, MorphologyProcessor
from contour_detection import ContourDetector, LineDetector
from drawing import Drawer
from visualization import Visualizer

def main():
    parser = argparse.ArgumentParser(description="Process video or image data.")
    parser.add_argument("input_path", help="Path to the input video file or image directory.")
    parser.add_argument("--output_dir", default="output", help="Directory to save output frames (if needed).")
    parser.add_argument("--save_video", action="store_true", help="Save the processed output as a video file.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize components
    media_loader = MediaLoader(args.input_path)
    geometry_corrector = GeometryCorrector(k1=CONFIG["fisheye_correction"]["k1"],
                                         k2=CONFIG["fisheye_correction"]["k2"])
    preprocessor = Preprocessor()
    morphology_processor = MorphologyProcessor(kernel_size=CONFIG["preprocessing"]["morphological_opening"]["kernel_size"])
    line_detector = LineDetector()
    contour_detector = ContourDetector()
    drawer = Drawer(neutral_color=CONFIG["drawing"].get("neutral_color", (128, 128, 128)))
    visualizer = Visualizer()

    # Initialize video writer if save_video is True and input is a video
    video_writer = None
    if args.save_video and media_loader.is_video:
        input_filename = os.path.basename(args.input_path)
        name, ext = os.path.splitext(input_filename)
        output_video_path = os.path.join(args.output_dir, f"{name}_processed.avi")
        fps = media_loader.cap.get(cv2.CAP_PROP_FPS)
        print(f"Input video FPS: {fps}") # Verify the FPS
        if fps <= 0:
            fps = 30.0 # Set a default FPS if the input FPS is invalid
            print(f"Warning: Invalid FPS in input video, setting FPS to {fps}")
        frame_width = int(media_loader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(media_loader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Try 'MJPG' or 'AVC1' if 'XVID' doesn't work
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        print(f"Saving processed video to: {output_video_path} with FPS: {fps} and dimensions: ({frame_width}, {frame_height})")
    elif args.save_video and not media_loader.is_video:
        print("Warning: --save_video is enabled, but the input is not a video. No video will be saved.")

    while True:
        frame, frame_number, timestamp = media_loader.get_next_frame()

        if frame is None:
            break

        # Apply fisheye correction
        if CONFIG["fisheye_correction"]["enabled"]:
            frame = geometry_corrector.apply_fisheye_correction(frame)

        # Preprocessing steps
        gray = preprocessor.convert_to_grayscale(frame) if CONFIG["preprocessing"]["grayscale"]["enabled"] else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        binary = gray
        if CONFIG["preprocessing"]["binary_threshold"]["enabled"]:
            blur_image = preprocessor.apply_blur(gray)
            binary = preprocessor.apply_binary_threshold(blur_image)

        if CONFIG["preprocessing"]["morphological_opening"]["enabled"]:
            binary = morphology_processor.apply_morphological_opening(binary)

        # Edge detection
        edges = cv2.Canny(binary, 30, 100)

        # Detect lines and contours
        lines = line_detector.detect(edges,
                                     threshold=CONFIG["line_detection"]["threshold"],
                                     min_line_length=CONFIG["line_detection"]["min_line_length"],
                                     max_line_gap=CONFIG["line_detection"]["max_line_gap"]) if CONFIG["line_detection"]["enabled"] else None

        contours = contour_detector.detect(binary) if CONFIG["contour_detection"]["enabled"] else None

        # Prepare canvas with extra space above the image
        canvas = drawer.prepare_canvas(frame)

        # Draw detected elements
        if CONFIG["drawing"]["lines"]["enabled"] and lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                canvas = drawer.plot_extrapolated_line(
                    canvas, x1, y1, x2, y2,
                    color=CONFIG["drawing"]["lines"]["color"],
                    thickness=CONFIG["drawing"]["lines"]["thickness"]
                )

        if CONFIG["drawing"]["contours"]["enabled"] and contours is not None:
            canvas = drawer.draw_contours(canvas, contours,
                                         color=CONFIG["drawing"]["contours"]["color"],
                                         thickness=CONFIG["drawing"]["contours"]["thickness"],
                                         aspect_ratio_threshold=CONFIG["drawing"]["contours"]["aspect_ratio_threshold"])

        # Save the processed frame (original size with overlays) to the video
        if video_writer is not None:
            h = canvas.shape[0] // 2  # Original height
            original_frame_with_overlay = canvas[h:, :]
            video_writer.write(original_frame_with_overlay)

        # Display the result
        visualizer.display(canvas, frame_number, timestamp)

        # Check for exit condition
        key = visualizer.wait_key(10)
        if key & 0xFF == ord('q') or key == 27:
            break

        # Add key wait for images
        if not media_loader.is_video:
            cv2.waitKey(0)

    # Cleanup
    media_loader.release()
    cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()
        print("Processed video saved successfully.")

if __name__ == "__main__":
    main()