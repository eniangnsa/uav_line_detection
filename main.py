import os
import argparse
import cv2
import numpy as np # Added for general numpy operations if needed

# Import classes from their respective files
from config import CONFIG
from media_loader import MediaLoader
from geometry_correction import GeometryCorrector
from preprocessing import Preprocessor, MorphologyProcessor # Preprocessor classs
from contour_detection import ContourDetector, ContourSelector, LineDetector # ContourDetector and new ContourSelector

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
    
    # Initialize ContourDetector and the new ContourSelector
    contour_detector = ContourDetector()
    contour_selector = ContourSelector(
        min_area=CONFIG["contour_selection"]["min_area"],
        max_aspect_ratio=CONFIG["contour_selection"]["max_aspect_ratio"],
        min_height_ratio=CONFIG["contour_selection"]["min_height_ratio"],
        max_approx_vertices=CONFIG["contour_selection"]["max_approx_vertices"],
        min_approx_vertices=CONFIG["contour_selection"]["min_approx_vertices"],
        top_bottom_tolerance=CONFIG["contour_selection"]["top_bottom_tolerance"]
    )

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

        # Get frame dimensions for contour selection
        frame_height, frame_width = frame.shape[:2]

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

        # Detect lines using Hough Transform
        hough_lines = None
        if CONFIG["line_detection"]["enabled"]:
            hough_lines = line_detector.detect(edges,
                                             threshold=CONFIG["line_detection"]["threshold"],
                                             min_line_length=CONFIG["line_detection"]["min_line_length"],
                                             max_line_gap=CONFIG["line_detection"]["max_line_gap"])

        # Detect and select the best contour
        all_contours = None
        best_contour = None
        fitted_line_from_contour = None

        if CONFIG["contour_detection"]["enabled"]:
            all_contours = contour_detector.detect(binary)
            if all_contours: # Only try to select if contours were found
                best_contour, fitted_line_from_contour = contour_selector.select_and_process_contour(
                    all_contours, frame_height, frame_width
                )

        # Prepare canvas with extra space above the image
        canvas = drawer.prepare_canvas(frame)

        # Draw detected elements
        # Draw Hough lines if enabled
        if CONFIG["drawing"]["lines"]["enabled"] and hough_lines is not None:
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                canvas = drawer.plot_extrapolated_line(
                    canvas, x1, y1, x2, y2,
                    color=CONFIG["drawing"]["lines"]["color"],
                    thickness=CONFIG["drawing"]["lines"]["thickness"]
                )

        # Draw the selected contour and its fitted line if found
        if CONFIG["drawing"]["contours"]["enabled"] and best_contour is not None:
            # Draw the selected contour
            # Adjust the contour's y-coordinates to account for the vertical shift on canvas
            original_height_on_canvas = canvas.shape[0] // 2
            shifted_best_contour = best_contour.copy()
            shifted_best_contour[:, :, 1] += original_height_on_canvas
            cv2.drawContours(canvas, [shifted_best_contour], -1, 
                             CONFIG["drawing"]["contours"]["color"], 
                             CONFIG["drawing"]["contours"]["thickness"])

            # Draw the fitted line from the contour
            if fitted_line_from_contour is not None:
                x1, y1, x2, y2 = fitted_line_from_contour
                # Since the fitted line is calculated for the original frame dimensions,
                # we need to adjust its y-coordinates for the canvas display.
                canvas = drawer.plot_extrapolated_line(
                    canvas, x1, y1, x2, y2,
                    color=CONFIG["drawing"]["contour_fitted_line"]["color"], # Use a distinct color for fitted line
                    thickness=CONFIG["drawing"]["contour_fitted_line"]["thickness"]
                )
        
        # Save the processed frame (original size with overlays) to the video
        if video_writer is not None:
            # The canvas has double the height, we only want the original frame area with overlays
            h_original_frame = frame.shape[0] # This is the original frame height
            processed_frame_for_video = canvas[h_original_frame:, :] # Get the bottom half of the canvas
            
            # Ensure the frame being written matches the dimensions initialized for video_writer
            # If the original frame was resized by MediaLoader, the video_writer should match that size.
            # Here, we assume frame_width and frame_height from media_loader.cap.get() are correct.
            if processed_frame_for_video.shape[1] != frame_width or processed_frame_for_video.shape[0] != frame_height:
                # This case should ideally not happen if MediaLoader's resizing is consistent
                # with initial video_writer dimensions, but adding a resize as a safeguard.
                processed_frame_for_video = cv2.resize(processed_frame_for_video, (frame_width, frame_height))
            
            video_writer.write(processed_frame_for_video)


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