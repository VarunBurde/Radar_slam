# Standard library imports
import os
import json
import yaml
import shutil
import struct
import subprocess
import traceback
from datetime import datetime

# Third-party imports 
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

# ROS2 related imports
import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py.point_cloud2 import read_points

# File paths
rosbag_path = "/media/varun/Vision_projects/projects/lawn_mower/smaut_calibration_data/rviz_auto_bag_20250409_143636_0.db3"
realsense_path = "/media/varun/Vision_projects/projects/lawn_mower/smaut_calibration_data/realsense_record_20250409_143552.bag"


def read_rosbag(bag_path, target_topic=None, message_type=None):
    # Initialize the reader
    reader = rosbag2_py.SequentialReader()
    
    # Create storage options and converter options
    storage_options = rosbag2_py._storage.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py._storage.ConverterOptions('', '')
    
    # Open the rosbag
    reader.open(storage_options, converter_options)
    
    # Get topic metadata
    topic_types = {}
    for topic_metadata in reader.get_all_topics_and_types():
        topic_types[topic_metadata.name] = topic_metadata.type
    
    print("\nAvailable topics:")
    for topic_name, topic_type in topic_types.items():
        print(f" - {topic_name} [{topic_type}]")
    
    # Find point cloud topics if none specified
    pointcloud_topics = []
    if not target_topic:
        for topic_name, topic_type in topic_types.items():
            if "PointCloud" in topic_type:
                pointcloud_topics.append(topic_name)
        
        if pointcloud_topics:
            print(f"\nDetected point cloud topics: {pointcloud_topics}")
            if len(pointcloud_topics) > 0:
                target_topic = pointcloud_topics[0]
                print(f"Using first point cloud topic: {target_topic}")
    
    # Read and display messages for the specified topic (or all if none specified)
    print(f"\nReading messages" + (f" from topic: {target_topic}" if target_topic else ""))
    
    messages = []
    count = 0
    
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        
        # Skip if not the target topic
        if target_topic and topic_name != target_topic:
            continue
            
        # Skip if not the target message type
        if message_type and message_type not in topic_types[topic_name]:
            continue
            
        msg_type = get_message(topic_types[topic_name])
        msg = deserialize_message(data, msg_type)
        
        # Store the message
        messages.append({
            'topic': topic_name,
            'timestamp': timestamp,
            'data': msg
        })
        
        # Print message info (without full point cloud data to avoid console flooding)
        print(f"\nTopic: {topic_name}")
        print(f"Timestamp: {timestamp}")
        if "PointCloud" in topic_types[topic_name]:
            print(f"Point Cloud message with {len(msg.data) if hasattr(msg, 'data') else '?'} points")
        else:
            print(f"Message: {msg}")
        
        count += 1
        # Limit the number of messages displayed but keep collecting data
        if count >= 5:  # Just get a few point clouds
            print("\n... More messages available but not displayed")
            break
    
    return messages, topic_types

def process_pointcloud_data(messages):
    """Process point cloud data from collected messages"""
    if not messages:
        print("No point cloud messages found.")
        return
    
    print(f"\nProcessing {len(messages)} point cloud messages")
    
    # Example: Print basic info about each point cloud
    for i, msg_container in enumerate(messages):
        msg = msg_container['data']
        timestamp = msg_container['timestamp']
        
        # Format timestamp as readable time
        seconds = timestamp // 10**9
        nanoseconds = timestamp % 10**9
        
        print(f"\nPoint Cloud {i+1}:")
        print(f"  - Timestamp: {seconds}.{nanoseconds:09d}")
        
        # Extract point cloud information - adapt to your specific message type
        try:
            if hasattr(msg, 'height') and hasattr(msg, 'width'):
                print(f"  - Dimensions: {msg.height} x {msg.width}")
            if hasattr(msg, 'point_step') and hasattr(msg, 'row_step'):
                print(f"  - Point step: {msg.point_step}, Row step: {msg.row_step}")
            if hasattr(msg, 'fields'):
                print(f"  - Fields: {[field.name for field in msg.fields]}")
            if hasattr(msg, 'data'):
                print(f"  - Data size: {len(msg.data)} bytes")
        except AttributeError as e:
            print(f"  - Error parsing point cloud data: {e}")
    
    return messages

def read_realsense_rgb(bag_file_path, output_folder=None, max_frames=10):
    """
    Extract RGB images from a RealSense bag file
    
    Args:
        bag_file_path: Path to the .bag file
        output_folder: Optional folder to save images
        max_frames: Maximum number of frames to extract
    """
    # Create output folder if specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        
    print(f"Reading RGB images from: {bag_file_path}")
    
    # Check if the bag file exists
    if not os.path.exists(bag_file_path):
        print(f"Error: Bag file not found at {bag_file_path}")
        return []
    
    try:
        # Try method 1: Using rs.pipeline directly
        try:
            print("Attempting to read bag file using pipeline method...")
            # Create pipeline and config
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Tell config to read from the bag file
            rs.config.enable_device_from_file(config, bag_file_path)
            
            # Enable the color stream - RGB - try without specifying format and fps
            config.enable_stream(rs.stream.color)
            
            # Start streaming from file
            profile = pipeline.start(config)
            
            # Get playback device to control playback
            playback = profile.get_device().as_playback()
            playback.set_real_time(False)  # Don't wait for frames according to their timestamps
            
            # Collect frames
            rgb_frames = []
            frame_count = 0
            
            # Process frames until max_frames is reached or file ends
            while frame_count < max_frames:
                try:
                    # Wait for a frameset
                    frames = pipeline.wait_for_frames(timeout_ms=5000)  # 5 seconds timeout
                    
                    # Get color frame
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    
                    # Get timestamp (in milliseconds)
                    timestamp = frames.get_timestamp()
                    
                    # Convert to numpy array
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # RealSense uses RGB format, convert to BGR for OpenCV
                    color_image_bgr = color_image
                    
                    # Save the frame information
                    rgb_frames.append({
                        'timestamp': timestamp,
                        'image': color_image_bgr
                    })
                    
                    # Save image to file if output_folder is specified
                    if output_folder:
                        filename = f"{output_folder}/rgb_{frame_count:04d}_{int(timestamp)}.png"
                        cv2.imwrite(filename, color_image_bgr)
                        print(f"Saved: {filename}")
                    
                    frame_count += 1
                    print(f"Extracted RGB frame {frame_count}/{max_frames} - timestamp: {timestamp/1000.0:.3f}s")
                    
                except RuntimeError as e:
                    print(f"Error reading frame: {e}")
                    break
            
            if rgb_frames:
                print(f"Extracted {len(rgb_frames)} RGB frames using pipeline method")
                return rgb_frames
                
        except RuntimeError as e:
            print(f"Pipeline method failed: {e}")
            # Fall through to try alternative method
        
        # Method 2: Using rs.playback directly with a context
        print("Attempting to read bag file using context method...")
        ctx = rs.context()
        dev = ctx.load_device(bag_file_path)
        playback = dev.as_playback()
        playback.set_real_time(False)
        
        sensors = dev.query_sensors()
        rgb_sensor = None
        for sensor in sensors:
            if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                rgb_sensor = sensor
                break
        
        if not rgb_sensor:
            print("Could not find RGB camera in the bag file")
            return []
        
        # Configure RGB sensor
        rgb_sensor.open(rs.stream.color)
        
        # Collect frames
        rgb_frames = []
        frame_count = 0
        
        while frame_count < max_frames:
            try:
                # Wait for next frame
                frame = playback.get_next_frame()
                
                # Get color frame
                color_frame = frame.get_color_frame()
                if not color_frame:
                    continue
                
                # Get timestamp
                timestamp = color_frame.get_timestamp()
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # RealSense uses RGB format, convert to BGR for OpenCV
                color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
                # Save the frame information
                rgb_frames.append({
                    'timestamp': timestamp,
                    'image': color_image_bgr
                })
                
                # Save image to file if output_folder is specified
                if output_folder:
                    filename = f"{output_folder}/rgb_{frame_count:04d}_{int(timestamp)}.png"
                    cv2.imwrite(filename, color_image_bgr)
                    print(f"Saved: {filename}")
                
                frame_count += 1
                print(f"Extracted RGB frame {frame_count}/{max_frames} - timestamp: {timestamp/1000.0:.3f}s")
                
            except RuntimeError as e:
                print(f"Error reading frame: {e}")
                break
        
        return rgb_frames
        
    except Exception as e:
        print(f"Error reading RealSense bag file: {e}")
        print("Falling back to realsense-viewer method (manual extraction)...")
        
        # Method 3: Suggest using realsense-viewer as a fallback
        print("\nAlternative approach:")
        print("1. Open the bag file in realsense-viewer: realsense-viewer -f " + bag_file_path)
        print("2. Export frames manually from the viewer")
        print("3. Use the exported images for your calibration")
        
        # Try to check if realsense-viewer is available
        try:
            result = subprocess.run(['which', 'realsense-viewer'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            if result.returncode == 0:
                print("\nRealsense-viewer is available. Run:")
                print(f"realsense-viewer -f {bag_file_path}")
            else:
                print("\nRealsense-viewer might not be installed. Install it with:")
                print("sudo apt install librealsense2-utils")
        except:
            print("\nCouldn't check for realsense-viewer.")
        
        return []
        
    finally:
        # Stop the pipeline if it exists
        try:
            pipeline.stop()
        except:
            pass

def find_closest_messages(reference_frames, target_messages, max_time_diff_ms=100):
    """
    Find the closest target message for each reference frame based on timestamps
    
    Args:
        reference_frames: List of dictionaries with 'timestamp' keys (RGB frames)
        target_messages: List of dictionaries with 'timestamp' keys (point clouds)
        max_time_diff_ms: Maximum allowed time difference in milliseconds
        
    Returns:
        List of tuples containing (reference_frame, closest_message, time_difference_ms)
    """
    if not reference_frames or not target_messages:
        return []
    
    # Convert ROS timestamps (nanoseconds) to milliseconds for comparison
    target_timestamps = [msg['timestamp'] / 1000000 for msg in target_messages]
    
    matched_pairs = []
    
    # For each reference frame (RGB image)
    for ref_frame in reference_frames:
        ref_timestamp_ms = ref_frame['timestamp']  # RealSense timestamps are already in milliseconds
        
        # Find the closest target message
        closest_idx = -1
        min_diff = float('inf')
        
        for i, target_ts in enumerate(target_timestamps):
            time_diff = abs(ref_timestamp_ms - target_ts)
            if time_diff < min_diff:
                min_diff = time_diff
                closest_idx = i
        
        # Check if the time difference is within acceptable range
        if closest_idx >= 0 and min_diff <= max_time_diff_ms:
            matched_pairs.append((
                ref_frame, 
                target_messages[closest_idx],
                min_diff  # Time difference in milliseconds
            ))
            print(f"Matched: RGB ts={ref_timestamp_ms:.3f}ms with point cloud ts={target_timestamps[closest_idx]:.3f}ms (diff: {min_diff:.3f}ms)")
        else:
            print(f"No match found for RGB frame at timestamp {ref_timestamp_ms:.3f}ms (closest diff: {min_diff:.3f}ms)")
    
    return matched_pairs

def save_synchronized_data(matched_pairs, output_folder):
    """Save the synchronized RGB and point cloud data"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare data for saving
    synchronized_data = []
    for idx, (rgb_frame, pointcloud_msg, time_diff) in enumerate(matched_pairs):
        # Create unique identifier for this pair
        pair_id = f"pair_{idx:04d}"
        
        # Create path for JSON metadata
        json_path = os.path.join(output_folder, f"{pair_id}_metadata.json")
        
        # Extract necessary information
        rgb_timestamp = rgb_frame['timestamp']
        pc_timestamp = pointcloud_msg['timestamp'] / 1000000  # Convert from ns to ms
        
        # Save RGB image if not already saved
        rgb_image_filename = f"rgb_{idx:04d}_{int(rgb_timestamp)}.png"
        rgb_image_path = os.path.join(output_folder, rgb_image_filename)
        
        if 'image' in rgb_frame and not os.path.exists(rgb_image_path):
            import cv2
            cv2.imwrite(rgb_image_path, rgb_frame['image'])
        
        # Create metadata for this pair
        metadata = {
            'pair_id': pair_id,
            'rgb_timestamp_ms': float(rgb_timestamp),
            'pointcloud_timestamp_ms': float(pc_timestamp),
            'time_difference_ms': float(time_diff),
            'rgb_image_path': rgb_image_filename,
            'pointcloud_topic': pointcloud_msg['topic'],
            'synchronized_at': import_datetime_and_return_now(),
        }
        
        # Save metadata to JSON file
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved synchronized pair {idx+1}/{len(matched_pairs)}: {json_path}")
        synchronized_data.append(metadata)
    
    # Save overall summary
    summary_path = os.path.join(output_folder, "synchronization_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'total_synchronized_pairs': len(synchronized_data),
            'pairs': synchronized_data
        }, f, indent=2)
    
    print(f"Synchronized data summary saved to: {summary_path}")
    return synchronized_data

def import_datetime_and_return_now():
    """Helper function to get current timestamp as string"""
    from datetime import datetime
    return datetime.now().isoformat()

def save_pointcloud_as_pcd(pointcloud_msg, output_path):
    """
    Save a ROS PointCloud2 message as a PCD file
    
    Args:
        pointcloud_msg: Dictionary containing the point cloud message data
        output_path: Path to save the PCD file
    """
    try:
        import numpy as np
        import struct
        from sensor_msgs_py.point_cloud2 import read_points
        
        pc_msg = pointcloud_msg['data']
        
        # Convert the point cloud to a numpy array
        pc_data = np.array(list(read_points(pc_msg)))
        
        if len(pc_data) == 0:
            print(f"Error: Point cloud has no points")
            return False
            
        # Extract x, y, z coordinates
        if 'x' in pc_data.dtype.names:
            x = pc_data['x']
            y = pc_data['y']
            z = pc_data['z']
        else:
            # Assume the first three columns are x, y, z if not named
            x = pc_data[:, 0]
            y = pc_data[:, 1]
            z = pc_data[:, 2]
        
        # Extract intensity if available (optional)
        intensity = None
        if 'intensity' in pc_data.dtype.names:
            intensity = pc_data['intensity']
        elif pc_data.shape[1] > 3:
            # Assume the 4th column is intensity
            intensity = pc_data[:, 3]
        
        # Create the PCD file
        with open(output_path, 'w') as f:
            # Write header
            f.write("# .PCD v0.7 - Point Cloud Data\n")
            f.write("VERSION 0.7\n")
            
            if intensity is not None:
                f.write("FIELDS x y z intensity\n")
                f.write("SIZE 4 4 4 4\n")
                f.write("TYPE F F F F\n")
                f.write("COUNT 1 1 1 1\n")
            else:
                f.write("FIELDS x y z\n")
                f.write("SIZE 4 4 4\n")
                f.write("TYPE F F F\n")
                f.write("COUNT 1 1 1\n")
            
            f.write(f"WIDTH {len(x)}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {len(x)}\n")
            f.write("DATA ascii\n")
            
            # Write point cloud data
            for i in range(len(x)):
                if intensity is not None:
                    f.write(f"{x[i]} {y[i]} {z[i]} {intensity[i]}\n")
                else:
                    f.write(f"{x[i]} {y[i]} {z[i]}\n")
        
        print(f"Saved point cloud to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving point cloud as PCD: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_realsense_intrinsics(bag_file_path):
    """
    Extract intrinsic parameters from a RealSense bag file
    
    Args:
        bag_file_path: Path to the RealSense .bag file
        
    Returns:
        dict: Camera intrinsic parameters or None if extraction fails
    """
    import pyrealsense2 as rs
    
    print("Extracting camera intrinsics from RealSense bag file...")
    
    try:
        # Create a context and load the device from file
        ctx = rs.context()
        device = ctx.load_device(bag_file_path)
        
        # Get the list of sensors in the device
        sensors = device.query_sensors()
        
        # Find the RGB/color sensor
        color_sensor = None
        for sensor in sensors:
            if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                color_sensor = sensor
                break
        
        if not color_sensor:
            print("Could not find RGB camera in the bag file")
            return None
        
        # Get the color sensor's stream profile
        stream_profiles = color_sensor.get_stream_profiles()
        color_profile = None
        
        for profile in stream_profiles:
            if profile.stream_type() == rs.stream.color:
                video_profile = profile.as_video_stream_profile()
                color_profile = profile
                break
        
        if not color_profile:
            print("Could not find color stream profile")
            return None
        
        # Get the intrinsics
        video_profile = color_profile.as_video_stream_profile()
        intrinsics = video_profile.get_intrinsics()
        
        # Extract resolution
        width = intrinsics.width
        height = intrinsics.height
        
        # Extract intrinsic matrix parameters
        fx = intrinsics.fx
        fy = intrinsics.fy
        ppx = intrinsics.ppx
        ppy = intrinsics.ppy
        
        # Extract distortion parameters
        distortion_model = intrinsics.model
        distortion_coeffs = intrinsics.coeffs
        
        print(f"Successfully extracted camera intrinsics:")
        print(f"Resolution: {width}x{height}")
        print(f"Focal lengths: fx={fx}, fy={fy}")
        print(f"Principal point: ppx={ppx}, ppy={ppy}")
        print(f"Distortion model: {distortion_model}")
        print(f"Distortion coefficients: {distortion_coeffs}")
        
        # Return the intrinsic parameters in a structured format
        return {
            "resolution": {
                "width": width,
                "height": height
            },
            "intrinsic_matrix": {
                "fx": fx,
                "fy": fy,
                "ppx": ppx,
                "ppy": ppy
            },
            "distortion": {
                "model": str(distortion_model),
                "coeffs": distortion_coeffs
            }
        }
        
    except Exception as e:
        print(f"Error extracting camera intrinsics: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_calibration_json_files(output_folder, camera_intrinsics=None):
    """
    Create calibration JSON files in the required format
    
    Args:
        output_folder: Folder to save the JSON files
        camera_intrinsics: Optional camera intrinsic parameters extracted from RealSense
        
    Returns:
        List of paths to the created JSON files
    """
    import json
    import os
    
    # Create intrinsic calibration JSON
    intrinsic_data = {
        "center_camera-intrinsic": {
            "sensor_name": "center_camera",
            "target_sensor_name": "center_camera",
            "device_type": "camera",
            "param_type": "intrinsic",
            "param": {
                "img_dist_w": 1920,
                "img_dist_h": 1080,
                "cam_K": {
                    "rows": 3,
                    "cols": 3,
                    "type": 6,
                    "continuous": True,
                    "data": [
                        [2109.75, 0, 949.828],
                        [0, 2071.72, 576.237],
                        [0, 0, 1]
                    ]
                },
                "cam_dist": {
                    "rows": 1,
                    "cols": 4,
                    "type": 6,
                    "continuous": True,
                    "data": [
                        [-0.10814499855041504, 0.1386680006980896, -0.0037975700106471777, -0.004841269925236702]
                    ]
                }
            }
        }
    }
    
    # Update intrinsic parameters if available from RealSense
    if camera_intrinsics:
        # Update resolution
        if "resolution" in camera_intrinsics:
            intrinsic_data["center_camera-intrinsic"]["param"]["img_dist_w"] = camera_intrinsics["resolution"]["width"]
            intrinsic_data["center_camera-intrinsic"]["param"]["img_dist_h"] = camera_intrinsics["resolution"]["height"]
        
        # Update camera matrix
        if "intrinsic_matrix" in camera_intrinsics:
            matrix = camera_intrinsics["intrinsic_matrix"]
            intrinsic_data["center_camera-intrinsic"]["param"]["cam_K"]["data"] = [
                [matrix["fx"], 0, matrix["ppx"]],
                [0, matrix["fy"], matrix["ppy"]],
                [0, 0, 1]
            ]
        
        # Update distortion coefficients
        if "distortion" in camera_intrinsics and "coeffs" in camera_intrinsics["distortion"]:
            coeffs = camera_intrinsics["distortion"]["coeffs"]
            # Convert to the required format - assuming Brown-Conrady model with k1, k2, p1, p2
            if len(coeffs) >= 4:
                intrinsic_data["center_camera-intrinsic"]["param"]["cam_dist"]["data"] = [
                    [coeffs[0], coeffs[1], coeffs[2], coeffs[3]]
                ]
            elif len(coeffs) == 5:  # Include k3 if available
                intrinsic_data["center_camera-intrinsic"]["param"]["cam_dist"]["data"] = [
                    [coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]]
                ]
    

    extrinsics = np.eye(4, dtype=float)  # Default to identity matrix
    extrinsics_rotation = R.from_euler('zyx', [-30,0,90], degrees=True)  # Example rotation around Z-axis
    extrinsics_translation = np.array([0.00, -0.09, -0.13]) 
    extrinsics[:3, :3] = extrinsics_rotation.as_matrix()  # Set rotation
    extrinsics[:3, 3] = extrinsics_translation  # Set translation

    
    # Create extrinsic calibration JSON with values from physical measurements
    extrinsic_data = {
        "top_center_lidar-to-center_camera-extrinsic": {
            "sensor_name": "top_center_lidar",
            "target_sensor_name": "center_camera",
            "device_type": "relational",
            "param_type": "extrinsic",
            "param": {
                "time_lag": 0,
                "sensor_calib": {
                    "rows": 4,
                    "cols": 4,
                    "type": 6,
                    "continuous": True,
                    "data": extrinsics.tolist()
                }
            }
        }
    }
    
    # File paths
    intrinsic_path = os.path.join(output_folder, "center_camera-intrinsic.json")
    extrinsic_path = os.path.join(output_folder, "top_center_lidar-to-center_camera-extrinsic.json")
    
    # Write files with pretty formatting
    with open(intrinsic_path, 'w') as f:
        json.dump(intrinsic_data, f, indent=4)
    print(f"Created intrinsic calibration file: {intrinsic_path}")
    
    with open(extrinsic_path, 'w') as f:
        json.dump(extrinsic_data, f, indent=4)
    print(f"Created extrinsic calibration file: {extrinsic_path}")
    
    return [intrinsic_path, extrinsic_path]

def create_calibration_folder(output_folder, rgb_frame=None, pointcloud_msg=None, camera_intrinsics=None):
    """
    Create a folder with calibration data (one image, one point cloud, and two JSON files)
    
    Args:
        output_folder: Base folder for output
        rgb_frame: Dictionary with RGB image data
        pointcloud_msg: Dictionary with point cloud data
        camera_intrinsics: Optional camera intrinsic parameters
    """
    import os
    import json
    import shutil
    from datetime import datetime
    
    # Create calibration folder name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calib_folder = os.path.join(output_folder, f"calibration_data_{timestamp}")
    os.makedirs(calib_folder, exist_ok=True)
    
    print(f"\nCreating calibration folder: {calib_folder}")
    
    # Create calibration JSON files directly in the folder
    json_files = create_calibration_json_files(calib_folder, camera_intrinsics)
    
    # Save image and point cloud
    result_files = {}
    
    if rgb_frame and 'image' in rgb_frame:
        import cv2
        img_path = os.path.join(calib_folder, "0.png")
        cv2.imwrite(img_path, rgb_frame['image'])
        print(f"Saved RGB image to {img_path}")
        result_files['image'] = img_path
    
    if pointcloud_msg and 'data' in pointcloud_msg:
        pcd_path = os.path.join(calib_folder, "0.pcd")
        if save_pointcloud_as_pcd(pointcloud_msg, pcd_path):
            result_files['pointcloud'] = pcd_path
    
    # Create a summary file
    summary = {
        "created_at": timestamp,
        "rgb_timestamp_ms": float(rgb_frame['timestamp']) if rgb_frame else None,
        "pointcloud_timestamp_ms": float(pointcloud_msg['timestamp'] / 1000000) if pointcloud_msg else None,
        "files": list(result_files.values()) + json_files
    }
    
    summary_path = os.path.join(calib_folder, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Created calibration data summary at {summary_path}")
    return calib_folder

if __name__ == "__main__":
    # Initialize ROS2
    rclpy.init()
    
    # Set output folder
    output_folder = "/media/varun/Vision_projects/projects/lawn_mower/calibration_output"
    
    try:
        # Step 1: Extract single RGB image from RealSense bag file and camera intrinsics
        print("=== EXTRACTING RGB IMAGE AND CAMERA INTRINSICS ===")
        
        # Extract camera intrinsics
        camera_intrinsics = extract_realsense_intrinsics(realsense_path)
        
        # Extract RGB frames
        rgb_frames = read_realsense_rgb(realsense_path, None, max_frames=1)
        
        if not rgb_frames:
            print("No RGB frames were extracted from the RealSense bag file")
            exit(1)
        
        print(f"Successfully extracted RGB image with timestamp: {rgb_frames[0]['timestamp']} ms")
        
        # Step 2: Extract single point cloud from ROS2 bag
        print("\n=== EXTRACTING POINT CLOUD ===")
        point_cloud_topic = None  # Auto-detect point cloud topics
        pointcloud_messages, topic_types = read_rosbag(rosbag_path, point_cloud_topic, "PointCloud")
        
        if not pointcloud_messages:
            print("No point cloud data found in the ROS2 bag")
            exit(1)
        
        print(f"Successfully extracted point cloud with timestamp: {pointcloud_messages[0]['timestamp']} ns")
        
        # Step 3: Find the closest matching pair
        print("\n=== FINDING BEST MATCH ===")
        best_match = find_closest_messages(rgb_frames, pointcloud_messages, max_time_diff_ms=1000)
        
        if not best_match:
            print("No matching RGB and point cloud pairs were found within the time threshold")
            # Use the first available point cloud if no good match
            print("Using first available RGB image and point cloud")
            rgb_frame = rgb_frames[0]
            pointcloud_msg = pointcloud_messages[0]
        else:
            # Get the best match
            rgb_frame, pointcloud_msg, time_diff = best_match[0]
            print(f"Found best match with time difference: {time_diff:.2f} ms")
        
        # Step 4: Create calibration folder with all required files
        print("\n=== CREATING CALIBRATION DATA FOLDER ===")
        calibration_folder = create_calibration_folder(output_folder, rgb_frame, pointcloud_msg, camera_intrinsics)
        
        print(f"\nComplete! Calibration data saved to {calibration_folder}")
        print("Files saved:")
        print(f"  - Image: {os.path.join(calibration_folder, '0.png')}")
        print(f"  - Point Cloud: {os.path.join(calibration_folder, '0.pcd')}")
        print(f"  - Extrinsic JSON: {os.path.join(calibration_folder, 'top_center_lidar-to-center_camera-extrinsic.json')}")
        print(f"  - Intrinsic JSON: {os.path.join(calibration_folder, 'center_camera-intrinsic.json')} (updated with RealSense parameters)")
        print(f"  - Summary: {os.path.join(calibration_folder, 'summary.json')}")
        
    except Exception as e:
        print(f"Error processing bag files: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()