import pandas as pd
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Load radar data with time-binned frame IDs
df = pd.read_csv("csv_data/side_radars_binned_05ms.csv")

# Get frame ID range information
frame_ids = df["frame_id"]
frame_min = frame_ids.min()
frame_max = frame_ids.max()

# ---------- Point Cloud Registration Example ----------

# STEP 1: Extract points from frame 9 (source point cloud)
filtered_df = df[df["frame_id"].isin([9])]
points = filtered_df[["meas_pos_x", "meas_pos_y", "meas_pos_z"]].values
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points)
pcd1.paint_uniform_color([1, 0, 0])  # Red color

# STEP 2: Extract points from frame 10 (target point cloud)
filtered_df = df[df["frame_id"].isin([10])]
points = filtered_df[["meas_pos_x", "meas_pos_y", "meas_pos_z"]].values
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points)
pcd2.paint_uniform_color([0, 1, 0])  # Green color

# STEP 3: Perform point cloud registration using ICP (Iterative Closest Point)
# Find transformation that aligns pcd1 with pcd2
threshold = 1.0  # Maximum correspondence distance
trans_init = np.eye(4)  # Initial transformation (identity matrix)
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

# STEP 4: Visualize transformation with coordinate frame
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
# Apply the estimated transformation to the coordinate frame
coordinate_frame.transform(reg_p2p.transformation)

# STEP 5: Extract points from frame 11 for additional comparison
filtered_df = df[df["frame_id"].isin([11])]
points = filtered_df[["meas_pos_x", "meas_pos_y", "meas_pos_z"]].values
pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(points)
pcd3.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red color

# Perform registration between frame 9 and frame 11
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1, pcd3, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

# Create another coordinate frame to visualize this transformation
coordinate_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
coordinate_frame2.transform(reg_p2p.transformation)

# STEP 6: Visualize all point clouds and coordinate frames
o3d.visualization.draw_geometries(
    [pcd3, coordinate_frame, coordinate_frame2],
    window_name="3D Radar Point Cloud with Transformations",
    width=1280,
    height=720,
    point_show_normal=False
)

