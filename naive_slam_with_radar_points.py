import pandas as pd
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

"""
Simple SLAM implementation using radar point clouds:
1. Load radar data binned by time
2. For each consecutive frame, perform point cloud registration (ICP)
3. Accumulate transformations to build a global map
4. Visualize the trajectory and point cloud map
"""

# Load radar point cloud data with time-binned frames
df = pd.read_csv("csv_data/side_radars_binned_051ms.csv")

# Get frame ID range
frame_ids = df["frame_id"]
frame_min = frame_ids.min()
frame_max = frame_ids.max()

# Initialize variables for SLAM
transform_frames = []  # List to store coordinate frames for visualization
world_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
transform_frames.append(world_coordinate_frame)  # Add world coordinate frame as reference

# Robot pose in world coordinates (starting with identity matrix)
robot_position = np.eye(4)  
last_transform = np.eye(4)  # Previous frame-to-frame transform for initialization

# Initialize global point cloud map
pcd_main = o3d.geometry.PointCloud()
pcd_main.points = o3d.utility.Vector3dVector([])

# Process all frames (except the last one)
unique_frames = df["frame_id"].unique()
for i in range(0, len(unique_frames)-1):
    print(f"Processing frame {i}/{len(unique_frames)-1}")
    
    # STEP 1: Get point cloud from current frame
    filtered_df_1 = df[df["frame_id"].isin([i])]
    points_1 = filtered_df_1[["meas_pos_x", "meas_pos_y", "meas_pos_z"]].values
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points_1)

    # STEP 2: Get point cloud from next frame
    filtered_df_2 = df[df["frame_id"].isin([i+1])]
    points_2 = filtered_df_2[["meas_pos_x", "meas_pos_y", "meas_pos_z"]].values
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points_2)

    # STEP 3: Perform point cloud registration (ICP)
    # Find transformation that aligns current frame with previous frame
    threshold = 1.0  # Maximum correspondence distance
    trans_init = last_transform  # Use previous transform as initial guess
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # STEP 4: Update SLAM state
    # Create a coordinate frame to visualize the robot's pose
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    # Update the transformation matrices
    last_transform = reg_p2p.transformation  # Save current transform for next iteration
    # Update the global robot position by composing the inverse of the current transform
    # (We use inverse because ICP gives us source->target, but we need target->source for SLAM)
    robot_position = robot_position @ np.linalg.inv(reg_p2p.transformation)

    # STEP 5: Transform point cloud to global coordinates
    pcd2.transform(robot_position)
    
    # STEP 6: Update visualization elements
    coordinate_frame.transform(robot_position)  # Move coordinate frame to current robot pose
    pcd_main.points.extend(pcd2.points)  # Add current frame points to global map
    transform_frames.append(coordinate_frame)  # Add current frame's coordinate system
    print(f"Robot position matrix:\n{robot_position}")

# Set color for the global point cloud
pcd_main.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
transform_frames.append(pcd_main)

# Visualize the complete SLAM result
o3d.visualization.draw_geometries(transform_frames)



