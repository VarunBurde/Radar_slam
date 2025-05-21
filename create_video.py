import cv2
import os
import re

"""
Script to create a video from the sequence of SLAM visualization frames.
This takes all PNG images from the radar_slam_imgs folder, sorts them by frame number,
and combines them into an MP4 video.
"""

# Configuration
img_folder = "radar_slam_imgs"
output_video = "videos/radar_slam_video.mp4"
fps = 15  # Frames per second in output video

# Get list of image files
images = [img for img in os.listdir(img_folder) if img.endswith(".png") or img.endswith(".jpg")]

# Extract frame number using regex and sort numerically
def get_frame_number(filename):
    """Extract numeric frame ID from filename using regex"""
    match = re.search(r'frame_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

# Sort images by frame number
images.sort(key=get_frame_number)

if not images:
    print(f"Error: No images found in {img_folder}")
    exit()

# Read the first image to get dimensions for the video
frame = cv2.imread(os.path.join(img_folder, images[0]))
if frame is None:
    print(f"Error: Could not read first image {images[0]}")
    exit()
    
height, width, layers = frame.shape

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Add each image to the video
total_images = len(images)
for i, image in enumerate(images):
    img_path = os.path.join(img_folder, image)
    frame = cv2.imread(img_path)
    if frame is not None:
        video.write(frame)
        print(f"Added image {i+1}/{total_images}: {image}")
    else:
        print(f"Warning: Failed to read {image}")

# Release the video writer
video.release()
print(f"Video successfully created at {output_video}")