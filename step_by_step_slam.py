import pandas as pd
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

"""
Advanced SLAM implementation with visualization of each step:
1. Load radar data binned by time
2. Perform frame-by-frame ICP registration
3. Build a global point cloud map
4. Capture images for each step to create a visualization
"""

# Load radar point cloud data with time-binned frames
df = pd.read_csv("csv_data/side_radars_binned_051ms.csv")

# Get frame ID range information
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

# Initialize point cloud for matching visualization
pcd_matching = o3d.geometry.PointCloud()
pcd_matching.points = o3d.utility.Vector3dVector([])

# Process frames with visualization
for i in range(0, 180):  # Process first 180 frames
    print(f"Processing frame {i}/180")
    
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

    # Reset matching point cloud for visualization
    pcd_matching.points = o3d.utility.Vector3dVector([])

    # STEP 4: Create coordinate frame for visualization
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # STEP 5: Update SLAM state
    last_transform = reg_p2p.transformation  # Save current transform for next iteration
    # Update global robot position (using inverse because ICP gives source->target transform)
    robot_position = robot_position @ np.linalg.inv(reg_p2p.transformation)

    # STEP 6: Transform the point cloud to global coordinates
    pcd2.transform(robot_position)

    # STEP 7: Update visualization elements
    coordinate_frame.transform(robot_position)  # Transform coordinate frame to current pose
    pcd_main.points.extend(pcd2.points)  # Add current frame points to global map
    
    # Add points to matching visualization (for debugging)
    pcd_matching.points.extend(pcd1.points)
    pcd_matching.points.extend(pcd2.points)

    # Set colors for visualization
    pcd_main.paint_uniform_color([1, 0, 0])       # Red for global map
    pcd_matching.paint_uniform_color([0, 1, 0])   # Green for matching points
    pcd2.paint_uniform_color([0, 0, 1])           # Blue for current frame

    # Ensure coordinate frame has proper visual properties
    if isinstance(coordinate_frame, o3d.geometry.TriangleMesh):
        coordinate_frame.compute_vertex_normals()
    
    # Prepare geometries for visualization
    geometries = [pcd_main, pcd2]
    geometries.extend(transform_frames)

    # STEP 8: Create visualization and capture frame
    vis = o3d.visualization.Visualizer()
    vis.create_window()
  
    # Add all geometries to visualizer
    for geom in geometries:
        vis.add_geometry(geom)
        
    # Update geometry and render
    for geom in geometries:
        vis.update_geometry(geom)
    
    vis.poll_events()
    vis.update_renderer()
    
    # Capture frame image
    vis.capture_screen_image(f"radar_slam_imgs/frame_{i}.png")
    
    # Clean up visualizer
    vis.destroy_window()

    # Add current coordinate frame to trajectory visualization
    transform_frames.append(coordinate_frame)

# Add the complete point cloud to final visualization
transform_frames.append(pcd_main)

# Display final SLAM result with complete trajectory
o3d.visualization.draw_geometries(transform_frames)



