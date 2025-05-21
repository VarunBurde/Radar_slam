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

pcd_matching = o3d.geometry.PointCloud()
pcd_matching.points = o3d.utility.Vector3dVector([])

for i in range(0, 180):


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

    # make pcd_matching empty
    pcd_matching.points = o3d.utility.Vector3dVector([])

    # Create a coordinate frame to visualize the transformation
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    last_transform = reg_p2p.transformation
    robot_position = robot_position @ np.linalg.inv(reg_p2p.transformation)
    # print(reg_p2p)

    pcd2.transform(robot_position)

    coordinate_frame.transform(robot_position)
    pcd_main.points.extend(pcd2.points)
    pcd_matching.points.extend(pcd1.points)
    pcd_matching.points.extend(pcd2.points)

    pcd_main.paint_uniform_color([1, 0, 0])
    pcd_matching.paint_uniform_color([0, 1, 0])
    pcd2.paint_uniform_color([0, 0, 1])

  # For coordinate frames, ensure they have material properties
    if isinstance(coordinate_frame, o3d.geometry.TriangleMesh):
        coordinate_frame.compute_vertex_normals()
    
    geometries = [pcd_main, pcd2]
    geometries.extend(transform_frames)

    # Replace the draw_geometries call with this visualization and capture code
    vis = o3d.visualization.Visualizer()
    vis.create_window()
  
    # Add all geometries
    for geom in geometries:
        vis.add_geometry(geom)
        
    # Update each geometry individually
    for geom in geometries:
        vis.update_geometry(geom)
    
    vis.poll_events()
    vis.update_renderer()
    
    # Capture the frame
    vis.capture_screen_image(f"/media/varun/Vision_projects/projects/lawn_mower/radar_slam_imgs/frame_{i}.png")
    
    # Close the window when done
    vis.destroy_window()

    transform_frames.append(coordinate_frame)

transform_frames.append(pcd_main)
o3d.visualization.draw_geometries(transform_frames)



