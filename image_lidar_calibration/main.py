import os
import rclpy
import traceback

# Project local imports
import config
import ros_utils
import realsense_utils
import calibration_utils

def main():
    rclpy.init()
    
    try:
        # Step 1: Extract camera intrinsics from RealSense bag file
        print("=== STEP 1: EXTRACTING CAMERA INTRINSICS ===")
        camera_intrinsics = realsense_utils.extract_realsense_intrinsics(config.REALSENSE_BAG_PATH)
        if camera_intrinsics:
            print("Successfully extracted camera intrinsics.")
        else:
            print("Failed to extract camera intrinsics. Using defaults in JSON.")

        # Step 2: Extract a single RGB image from RealSense bag file
        print("\n=== STEP 2: EXTRACTING RGB IMAGE ===")
        rgb_frames = realsense_utils.read_realsense_rgb(
            config.REALSENSE_BAG_PATH, 
            output_folder=None, 
            max_frames=1 # Ensure we only get one frame as per typical config
        )
        
        if not rgb_frames:
            print("No RGB frames were extracted. Exiting.")
            return
        selected_rgb_frame = rgb_frames[0]
        print(f"Successfully extracted 1 RGB image. Timestamp: {selected_rgb_frame['timestamp']:.0f}ms.")
        
        # Step 3: Extract point cloud messages from ROS2 bag
        print("\n=== STEP 3: EXTRACTING POINT CLOUD MESSAGES ===")
        # MAX_LIDAR_MESSAGES_TO_EXTRACT in config controls how many are read for merging
        pointcloud_messages, _ = ros_utils.read_rosbag(
            config.ROSBAG_PATH, 
            target_topic=None, 
            message_type_filter="PointCloud",
            max_messages_to_extract=config.MAX_LIDAR_MESSAGES_TO_EXTRACT 
        )
        
        if not pointcloud_messages:
            print("No point cloud data found in the ROS2 bag. Exiting.")
            return
        print(f"Successfully extracted {len(pointcloud_messages)} point cloud message(s) for merging.")

        # Step 4: Merge extracted point cloud messages
        print("\n=== STEP 4: MERGING POINT CLOUDS ===")
        merged_pointcloud_container = calibration_utils.create_merged_pointcloud_message(pointcloud_messages)

        if not merged_pointcloud_container:
            print("Failed to merge point clouds or no points found. Exiting.")
            return
        num_original_clouds = len(pointcloud_messages)
        num_merged_points = merged_pointcloud_container['data'].width * merged_pointcloud_container['data'].height
        print(f"Successfully merged {num_original_clouds} point clouds into one with {num_merged_points} points.")
        print(f"Merged point cloud representative timestamp (from first original cloud): {merged_pointcloud_container['timestamp']} ns.")
        
        # Step 5: Create calibration folder with the selected RGB and merged point cloud
        print("\n=== STEP 5: CREATING CALIBRATION DATA FOLDER ===")
        os.makedirs(config.OUTPUT_FOLDER, exist_ok=True) 
        
        calibration_folder_path = calibration_utils.create_calibration_folder(
            config.OUTPUT_FOLDER, 
            selected_rgb_frame, 
            merged_pointcloud_container, 
            camera_intrinsics
        )
        
        print(f"\nProcess complete! Calibration data saved to: {calibration_folder_path}")
        print("Key files created in the folder:")
        print(f"  - Image: 0.png")
        print(f"  - Point Cloud: 0.pcd")
        print(f"  - Intrinsic JSON: center_camera-intrinsic.json")
        print(f"  - Extrinsic JSON: top_center_lidar-to-center_camera-extrinsic.json")
        print(f"  - Summary JSON: summary.json")
        
    except Exception as e:
        print(f"An error occurred during the calibration process: {e}")
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        print("ROS2 shutdown.")

if __name__ == "__main__":
    main()
