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
df = pd.read_csv("side_radars_binned_051ms.csv")

# Normalize frame_id values to range [0, 1] for coloring
frame_ids = df["frame_id"]
frame_min = frame_ids.min()
frame_max = frame_ids.max()

transform_frames = []
world_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
transform_frames.append(world_coordinate_frame)
robot_position = np.eye(4)
last_transform = np.eye(4)
pcd_main = o3d.geometry.PointCloud()
pcd_main.points = o3d.utility.Vector3dVector([])

for i in range(0, len(df["frame_id"].unique())-1):
    print(i)
    # Filter points for frame_id 0 and 1
    filtered_df_1 = df[df["frame_id"].isin([i])]

    # Extract XYZ points
    points_1 = filtered_df_1[["meas_pos_x", "meas_pos_y", "meas_pos_z"]].values

    # Create the point cloud
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points_1)

    # Filter points for frame_id 0 and 1
    filtered_df_2 = df[df["frame_id"].isin([i+1])]

    # Extract XYZ points
    points_2 = filtered_df_2[["meas_pos_x", "meas_pos_y", "meas_pos_z"]].values

    # Create the point cloud
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points_2)

    threshold = 1.0
    trans_init = last_transform
    reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # Create a coordinate frame to visualize the transformation
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    last_transform = reg_p2p.transformation
    robot_position = robot_position @ np.linalg.inv(reg_p2p.transformation)

    pcd2.transform(robot_position)

    coordinate_frame.transform(robot_position)
    pcd_main.points.extend(pcd2.points)

    transform_frames.append(coordinate_frame)
    print(robot_position)



pcd_main.paint_uniform_color([1.0, 0.0, 0.0])  # Set color for the main point cloud
transform_frames.append(pcd_main)
o3d.visualization.draw_geometries(transform_frames)



