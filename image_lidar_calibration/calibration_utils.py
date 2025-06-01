import os
import json
import shutil
import struct
from datetime import datetime

import numpy as np
import cv2 # For saving images if not already saved by read_realsense_rgb
from scipy.spatial.transform import Rotation as R
from sensor_msgs_py import point_cloud2 # For read_points, create_cloud
from sensor_msgs.msg import PointField # For defining PointCloud2 fields

def find_closest_messages(reference_frames, target_messages, max_time_diff_ms=100):
    """
    Find the closest target message for each reference frame based on timestamps.
    Reference frames timestamps are expected in ms.
    Target messages timestamps are expected in ns.
    """
    if not reference_frames or not target_messages:
        print("Warning: No reference frames or target messages to match.")
        return []
    
    # Convert target timestamps (nanoseconds) to milliseconds for comparison
    target_timestamps_ms = [{'original_ts_ns': msg['timestamp'], 
                             'ts_ms': msg['timestamp'] / 1_000_000, 
                             'original_index': i} 
                            for i, msg in enumerate(target_messages)]
    
    matched_pairs = []
    
    for ref_frame in reference_frames:
        ref_timestamp_ms = ref_frame['timestamp']  # RealSense timestamps are in milliseconds
        
        closest_target_info = None
        min_diff = float('inf')
        
        for target_info in target_timestamps_ms:
            time_diff = abs(ref_timestamp_ms - target_info['ts_ms'])
            if time_diff < min_diff:
                min_diff = time_diff
                closest_target_info = target_info
        
        if closest_target_info and min_diff <= max_time_diff_ms:
            matched_pairs.append((
                ref_frame, 
                target_messages[closest_target_info['original_index']],
                min_diff  # Time difference in milliseconds
            ))
            # print(f"Matched: RGB ts={ref_timestamp_ms:.3f}ms with target ts={closest_target_info['ts_ms']:.3f}ms (diff: {min_diff:.3f}ms)")
        # else:
            # closest_diff_val = min_diff if closest_target_info else "N/A"
            # print(f"No suitable match found for RGB frame at ts {ref_timestamp_ms:.3f}ms (closest diff: {closest_diff_val}ms, threshold: {max_time_diff_ms}ms)")
            
    # Sort by time difference to get the best match first if multiple reference frames were processed
    matched_pairs.sort(key=lambda x: x[2])
    return matched_pairs

def create_merged_pointcloud_message(pointcloud_messages_list):
    """
    Merges multiple PointCloud2 messages into a single PointCloud2 message.
    Assumes all input point clouds are in the same frame and robot is stationary.
    """
    if not pointcloud_messages_list:
        print("Warning: No point cloud messages provided for merging.")
        return None

    # Check if the original point clouds have intensity data
    first_pc_msg = pointcloud_messages_list[0]['data']
    field_names = [field.name for field in first_pc_msg.fields]
    has_intensity = 'intensity' in field_names
    
    print(f"Merging {len(pointcloud_messages_list)} point clouds")
    print(f"Original fields: {field_names}")
    print(f"Has intensity: {has_intensity}")

    all_points = []
    for msg_container in pointcloud_messages_list:
        pc_msg = msg_container['data']  # This is a sensor_msgs.msg.PointCloud2 object
        
        if has_intensity:
            # Read points with intensity
            points_from_this_msg = list(point_cloud2.read_points(pc_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        else:
            # Read points without intensity
            points_from_this_msg = list(point_cloud2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True))
        
        all_points.extend(points_from_this_msg)

    if not all_points:
        print("Warning: No valid points found in the provided point clouds to merge.")
        return None

    # Use the header from the first point cloud message as a template for the merged cloud.
    header = pointcloud_messages_list[0]['data'].header

    # Define fields for the new PointCloud2 message
    if has_intensity:
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
    else:
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

    # Create the merged PointCloud2 message object
    merged_pc2_object = point_cloud2.create_cloud(header, fields, all_points)
    
    representative_timestamp_ns = pointcloud_messages_list[0]['timestamp']

    return {
        'topic': 'merged_pointcloud', # Indicate it's a merged cloud
        'timestamp': representative_timestamp_ns, 
        'data': merged_pc2_object, # The actual sensor_msgs.msg.PointCloud2 object
        'num_merged': len(pointcloud_messages_list) # Store how many clouds were merged
    }

def save_pointcloud_as_pcd(pointcloud_msg_container, output_path):
    """
    Save a ROS PointCloud2 message (from the container structure) as a PCD file.
    """
    try:
        from sensor_msgs_py.point_cloud2 import read_points
        
        pc_msg = pointcloud_msg_container['data'] # Actual PointCloud2 message
        
        # Debug: Check the actual message fields
        field_names = [field.name for field in pc_msg.fields]
        print(f"Available fields in point cloud: {field_names}")
        print(f"Point cloud from topic: {pointcloud_msg_container.get('topic', 'unknown')}")
        
        # Convert the point cloud to a numpy array
        pc_data = np.array(list(read_points(pc_msg)))
        
        if len(pc_data) == 0:
            print(f"Error: Point cloud has no points")
            return False
            
        print(f"Point cloud data shape: {pc_data.shape}")
        print(f"Point cloud data dtype: {pc_data.dtype}")
        if hasattr(pc_data.dtype, 'names'):
            print(f"Field names in data: {pc_data.dtype.names}")
            
        # Extract x, y, z coordinates
        if hasattr(pc_data.dtype, 'names') and pc_data.dtype.names is not None:
            print(f"Structured array with fields: {pc_data.dtype.names}")
            if 'x' in pc_data.dtype.names:
                x = pc_data['x']
                y = pc_data['y']
                z = pc_data['z']
            else:
                print(f"Error: Point cloud structured array missing required x, y, z fields")
                return False
        else:
            # Assume the first three columns are x, y, z if not named
            print(f"Non-structured array, using column indices")
            x = pc_data[:, 0]
            y = pc_data[:, 1]
            z = pc_data[:, 2]
        
        # Extract intensity if available (optional)
        intensity = None
        if hasattr(pc_data.dtype, 'names') and pc_data.dtype.names is not None:
            if 'intensity' in pc_data.dtype.names:
                intensity = pc_data['intensity']
                print(f"Found intensity field with range: {np.min(intensity)} to {np.max(intensity)}")
            elif 'i' in pc_data.dtype.names:
                intensity = pc_data['i']
                print(f"Found 'i' field with range: {np.min(intensity)} to {np.max(intensity)}")
        elif pc_data.shape[1] > 3:
            # Assume the 4th column is intensity
            intensity = pc_data[:, 3]
            print(f"Using column 3 as intensity with range: {np.min(intensity)} to {np.max(intensity)}")
        else:
            print("No intensity field found")
        
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
        if intensity is not None:
            print(f"Included intensity values in PCD file")
        else:
            print(f"No intensity values included in PCD file")
        return True
        
    except Exception as e:
        print(f"Error saving point cloud as PCD: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_calibration_json_files(output_dir, camera_intrinsics=None):
    """
    Create intrinsic and extrinsic calibration JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Default intrinsic data (can be overridden by RealSense data)
    intrinsic_data = {
        "center_camera-intrinsic": {
            "sensor_name": "center_camera", "target_sensor_name": "center_camera",
            "device_type": "camera", "param_type": "intrinsic",
            "param": {
                "img_dist_w": 1920, "img_dist_h": 1080,
                "cam_K": {"rows": 3, "cols": 3, "type": 6, "continuous": True, 
                          "data": [[2109.75, 0, 949.828], [0, 2071.72, 576.237], [0, 0, 1]]},
                "cam_dist": {"rows": 1, "cols": 4, "type": 6, "continuous": True, 
                             "data": [[-0.10814499855041504, 0.1386680006980896, 
                                       -0.0037975700106471777, -0.004841269925236702]]}
            }
        }
    }
    
    if camera_intrinsics:
        param = intrinsic_data["center_camera-intrinsic"]["param"]
        if "resolution" in camera_intrinsics:
            param["img_dist_w"] = camera_intrinsics["resolution"]["width"]
            param["img_dist_h"] = camera_intrinsics["resolution"]["height"]
        if "intrinsic_matrix" in camera_intrinsics:
            matrix = camera_intrinsics["intrinsic_matrix"]
            param["cam_K"]["data"] = [[matrix["fx"], 0, matrix["ppx"]],
                                      [0, matrix["fy"], matrix["ppy"]],
                                      [0, 0, 1]]
        if "distortion" in camera_intrinsics and "coeffs" in camera_intrinsics["distortion"]:
            coeffs = camera_intrinsics["distortion"]["coeffs"]
            # Ensure cam_dist data matches the number of coeffs, e.g. k1,k2,p1,p2,(k3)
            param["cam_dist"]["data"] = [coeffs[:5]] # Take up to 5 coeffs
            param["cam_dist"]["cols"] = len(param["cam_dist"]["data"][0])


    # Extrinsic data (example values, should be measured/calibrated)
    extrinsics_matrix = np.eye(4, dtype=float)
    # Example: ZYX Euler angles: Yaw(-30 deg), Pitch(0 deg), Roll(90 deg)
    # This order might need adjustment based on convention (e.g., extrinsic vs intrinsic rotations)
    # And axis definition (e.g. ROS vs OpenCV camera coordinates)
    # rotation = R.from_euler('zyx', [-30, 0, 90], degrees=True) # Original
    # Common for camera: X (roll), Y (pitch), Z (yaw) relative to camera frame
    # Assuming transformation from Lidar to Camera:
    # Lidar: X-forward, Y-left, Z-up
    # Camera: Z-forward, X-right, Y-down
    # Rotation: Lidar Z maps to Camera -Y, Lidar Y maps to Camera -X, Lidar X maps to Camera Z
    # This implies a +90 deg rotation around Lidar X, then +90 deg around new Lidar Y (original Z)
    # Or, from camera to lidar: R_cl = R_lc.T
    # Let's use a common setup: Lidar pointing forward, camera slightly above and forward.
    # Extrinsics T_camera_lidar (transform points from Lidar frame to Camera frame)
    # Rotation part: align Lidar axes (e.g. x-fwd, y-left, z-up) to Camera axes (z-fwd, x-right, y-down)
    # R_cam_lidar = R_cam_optical @ R_optical_lidar_body @ R_lidar_body_lidar_sensor
    # Simplified: Assume Lidar X forward, Y left, Z up. Camera Z forward, X right, Y down.
    # Rotation from Lidar to Camera:
    # Lidar X -> Camera Z
    # Lidar Y -> Camera -X
    # Lidar Z -> Camera -Y
    # This corresponds to: R = [[0, -1,  0], [0,  0, -1], [1,  0,  0]]
    # Or Euler: e.g. pitch by -90 deg (around Y), then yaw by -90 deg (around new Z)
    # For simplicity, using the provided Euler angles, but noting they are application-specific.
    rotation = R.from_euler('zyx', [-30, 0, 90], degrees=True) 
    translation = np.array([0.00, -0.09, -0.13]) # meters; (X, Y, Z) translation of Lidar origin in Camera frame
    
    extrinsics_matrix[:3, :3] = rotation.as_matrix()
    extrinsics_matrix[:3, 3] = translation
    
    extrinsic_data = {
        "top_center_lidar-to-center_camera-extrinsic": {
            "sensor_name": "top_center_lidar", "target_sensor_name": "center_camera",
            "device_type": "relational", "param_type": "extrinsic",
            "param": {
                "time_lag": 0, # in seconds
                "sensor_calib": {"rows": 4, "cols": 4, "type": 6, "continuous": True, 
                                 "data": extrinsics_matrix.tolist()}
            }
        }
    }
    
    intrinsic_path = os.path.join(output_dir, "center_camera-intrinsic.json")
    extrinsic_path = os.path.join(output_dir, "top_center_lidar-to-center_camera-extrinsic.json")
    
    with open(intrinsic_path, 'w') as f:
        json.dump(intrinsic_data, f, indent=4)
    print(f"Created intrinsic calibration file: {intrinsic_path}")
    
    with open(extrinsic_path, 'w') as f:
        json.dump(extrinsic_data, f, indent=4)
    print(f"Created extrinsic calibration file: {extrinsic_path}")
    
    return [intrinsic_path, extrinsic_path]

def create_calibration_folder(base_output_folder, rgb_frame, pointcloud_msg_container, camera_intrinsics=None):
    """
    Create a folder with one image, one point cloud, and calibration JSON files.
    pointcloud_msg_container can be a single or merged point cloud.
    """
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    calib_folder_name = f"calibration_data_{timestamp_str}"
    calib_folder_path = os.path.join(base_output_folder, calib_folder_name)
    os.makedirs(calib_folder_path, exist_ok=True)
    
    print(f"\nCreating calibration folder: {calib_folder_path}")
    
    json_files = create_calibration_json_files(calib_folder_path, camera_intrinsics)
    
    saved_files = {}
    
    # Save RGB image (0.png)
    if rgb_frame and 'image' in rgb_frame:
        img_path = os.path.join(calib_folder_path, "0.png")
        try:
            cv2.imwrite(img_path, rgb_frame['image'])
            print(f"Saved RGB image to {img_path}")
            saved_files['image'] = "0.png"
        except Exception as e:
            print(f"Error saving RGB image: {e}")
    
    # Save PointCloud (0.pcd)
    if pointcloud_msg_container and 'data' in pointcloud_msg_container:
        pcd_path = os.path.join(calib_folder_path, "0.pcd")
        if save_pointcloud_as_pcd(pointcloud_msg_container, pcd_path):
            print(f"Saved point cloud to {pcd_path}") 
            saved_files['pointcloud'] = "0.pcd"
        else:
            print(f"Failed to save point cloud to {pcd_path}")

    # Create a summary file
    pc_timestamp_ns = int(pointcloud_msg_container['timestamp']) if pointcloud_msg_container and 'timestamp' in pointcloud_msg_container else None
    pc_topic = pointcloud_msg_container.get('topic') if pointcloud_msg_container else None
    num_merged = pointcloud_msg_container.get('num_merged', 1) # Default to 1 if not specified (i.e. not merged)

    summary_data = {
        "folder_created_at": datetime.now().isoformat(),
        "rgb_timestamp_ms": float(rgb_frame['timestamp']) if rgb_frame and 'timestamp' in rgb_frame else None,
        "pointcloud_representative_timestamp_ns": pc_timestamp_ns,
        "pointcloud_topic_or_type": pc_topic,
        "pointclouds_merged_count": num_merged,
        "files_generated": {
            "image": saved_files.get('image'),
            "pointcloud": saved_files.get('pointcloud'),
            "intrinsic_json": os.path.basename(json_files[0]) if len(json_files) > 0 else None,
            "extrinsic_json": os.path.basename(json_files[1]) if len(json_files) > 1 else None,
        },
        "camera_intrinsics_source": "RealSense Bag" if camera_intrinsics else "Defaults"
    }
    
    summary_path = os.path.join(calib_folder_path, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Created calibration data summary at {summary_path}")
    
    return calib_folder_path
