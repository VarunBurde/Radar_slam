import cv2
import os
import re

img_folder = "/media/varun/Vision_projects/projects/lawn_mower/radar_slam_imgs"
output_video = "/media/varun/Vision_projects/projects/lawn_mower/radar_slam_video.mp4"

# Get list of image files
images = [img for img in os.listdir(img_folder) if img.endswith(".png") or img.endswith(".jpg")]

# Extract frame number and sort numerically
def get_frame_number(filename):
    match = re.search(r'frame_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

# Sort images by frame number
images.sort(key=get_frame_number)

if not images:
    print(f"No images found in {img_folder}")
    exit()

# Read the first image to get dimensions
frame = cv2.imread(os.path.join(img_folder, images[0]))
height, width, layers = frame.shape

# Define video writer
fps = 15  # Frames per second (adjust as needed)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Add each image to the video
for image in images:
    img_path = os.path.join(img_folder, image)
    frame = cv2.imread(img_path)
    if frame is not None:
        video.write(frame)
        print(f"Added {image} to video")
    else:
        print(f"Failed to read {image}")

# Release the video writer
video.release()
print(f"Video created at {output_video}")