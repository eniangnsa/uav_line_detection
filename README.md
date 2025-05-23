# UAV LINE DETECTION
* This project implements a pipeline for line and contour detection in video or image inputs using OpenCV and NumPy.
* Follow the instructions below to set up and run the project.


## 1. Create a virtual environment
run the code below to create a virtual environment:
* python -m venv .venv

## 2. Activate the Virtual Environment
* on Windows:
.venv\Scripts\activate
* on Mac and Linux
source .venv/bin/activate

## 3. Install Dependencies
* pip install opencv-python numpy

## 4. Project Structure
The project is organized into the following files within the uav_line_detection:
* |------config.py
* |------contour_detection.py
* |------drawing.py
* |------geometry_correction.py
* |------main.py
* |------media_loader.py
* |------morphology.py
* |------preprocessing.py
* |______visualization.py
## 5. Configuration (config.py)
The config.py file contains a dictionary (CONFIG) where you can adjust various parameters of the processing pipeline. 
Modify these parameters as needed for your specific video or image data.

## Execution
To run the line and contour detection pipeline, navigate to the uav_line_detection directory in your terminal and execute the main.py script, providing the path to your input video file or image directory as a command-line argument.

cd uav_line_detection
python main.py <path_to_input_video_or_image_directory> [options]

### Arguments
* <path_to_input_video_or_image_directory>: **Required**. Specifies the path to either a single video file (e.g., data/input videos/video.avi) or a directory containing image files (e.g., data/images). 
If a directory is provided, the script will process all .png, .jpg, .jpeg, .bmp, and .tiff files within it.
### Options
* --output_dir <output_directory>: Optional. Specifies the directory where output frames (if you choose to save individual frames within the code) and the processed video (if --save_video is used) will be saved. Defaults to output.
* --save_video: Optional. If this flag is included, and if the input is a video file, the processed output will be saved as a new video file in the output directory (named <original_filename>_processed.avi).


## Examples
* Process a single video file:
** python main.py "data/input videos/my_drone_footage.mp4"

* Process a video and save the output:
** python main.py "video.avi" --save_video

* Process a video and specify a custom output directory:
python main.py "video.avi" --output_dir "processed_output" --save_video

Output
The script will display the processed video or images in a window. If the --save_video flag is used with a video input, a new processed video file will be saved in the specified output directory.
