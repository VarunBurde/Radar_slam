# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load your dataset
# df = pd.read_csv("side_radars_binned_20ms.csv")

# # Normalize frame IDs to a range (0 to 1) for color mapping
# frame_ids = df["frame_id"]
# frame_id_min = frame_ids.min()
# frame_id_max = frame_ids.max()
# normed_frame_ids = (frame_ids - frame_id_min) / (frame_id_max - frame_id_min)

# # Create a colormap (e.g., 'viridis' or 'hsv')
# cmap = plt.cm.get_cmap("hsv")

# # Map normalized frame_id values to colors
# colors = cmap(normed_frame_ids)

# # Plot all points with color based on their frame ID
# plt.figure(figsize=(12, 8))
# plt.scatter(df["meas_pos_x"], df["meas_pos_y"], c=colors, s=3, alpha=0.6)

# plt.title("Radar Points by Frame ID")
# plt.xlabel("X (meters)")
# plt.ylabel("Y (meters)")
# plt.axis("equal")
# plt.grid(True)
# # plt.colorbar(label="Normalized Frame ID")
# plt.tight_layout()
# plt.show()


import pandas as pd
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Load your binned radar data
df = pd.read_csv("side_radars_binned_05ms.csv")

# Normalize frame_id values to range [0, 1] for coloring
frame_ids = df["frame_id"]
frame_min = frame_ids.min()
frame_max = frame_ids.max()
# frame_norm = (frame_ids - frame_min) / (frame_max - frame_min)

# Use a colormap (e.g., hsv) to assign a color to each point based on frame_id
# cmap = plt.get_cmap("hsv")
# colors = cmap(frame_norm)[:, :3]  # Ignore alpha channel

# Filter points for frame_id 0 and 1
filtered_df = df[df["frame_id"].isin([9])]

# Extract XYZ points
points = filtered_df[["meas_pos_x", "meas_pos_y", "meas_pos_z"]].values

# Create the point cloud
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

pcd1.paint_uniform_color([1, 0, 0])

# Filter points for frame_id 0 and 1
filtered_df = df[df["frame_id"].isin([10])]

# Extract XYZ points
points = filtered_df[["meas_pos_x", "meas_pos_y", "meas_pos_z"]].values

# Create the point cloud
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

pcd2.paint_uniform_color([0, 1, 0])

threshold = 1
trans_init = np.eye(4)
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())


# Create a coordinate frame to visualize the transformation
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

# Apply the transformation to the coordinate frame
coordinate_frame.transform(reg_p2p.transformation)

# Filter points for frame_id 0 and 1
filtered_df = df[df["frame_id"].isin([11])]

# Extract XYZ points
points = filtered_df[["meas_pos_x", "meas_pos_y", "meas_pos_z"]].values

# Create the point cloud
pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(points)
pcd3.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

# pcd3.paint_uniform_color([0, 0, 1])

threshold = 1
trans_init = np.eye(4)
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1, pcd3, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())


# Create a coordinate frame to visualize the transformation
coordinate_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# Apply the transformation to the coordinate frame
coordinate_frame2.transform(reg_p2p.transformation)


#Launch Open3D visualizer
o3d.visualization.draw_geometries(
    [pcd3,coordinate_frame,coordinate_frame2],
    window_name="3D Radar Point Cloud - Colored by Frame ID",
    width=1280,
    height=720,
    point_show_normal=False
)

