# File paths
ROSBAG_PATH = "/media/varun/Vision_projects/projects/lawn_mower/smaut_calibration_data/rviz_auto_bag_20250409_143636_0.db3"
REALSENSE_BAG_PATH = "/media/varun/Vision_projects/projects/lawn_mower/smaut_calibration_data/realsense_record_20250409_143552.bag"
OUTPUT_FOLDER = "/media/varun/Vision_projects/projects/lawn_mower/calibration_output"

# Calibration parameters
MAX_TIME_DIFF_MS = 1000  # Maximum allowed time difference for matching frames in milliseconds
MAX_RGB_FRAMES_TO_EXTRACT = 1 # Max RGB frames to extract from realsense bag
MAX_LIDAR_MESSAGES_TO_EXTRACT = 5 # Max Lidar messages to extract from rosbag
