import pandas as pd
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import Imu
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import time

"""
ROS2 Node to publish radar point clouds as PointCloud2 messages.
This simulates a LiDAR/radar sensor by publishing:
1. Point cloud data from radar frames
2. IMU messages with default values
3. TF transforms between sensor frames

This allows visualization and processing with standard ROS tools.
"""

class PCDPublisher(Node):
    def __init__(self):
        """Initialize the node and set up publishers and data sources"""
        super().__init__('pcd_publisher')
        
        # Create publishers
        self.publisher_ = self.create_publisher(PointCloud2, '/unilidar/cloud', 10)
        self.publisher_imu = self.create_publisher(Imu, '/unilidar/imu', 10)

        # Create TF broadcaster for coordinate frames
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Load the binned radar data
        self.df = pd.read_csv("csv_data/side_radars_binned_051ms.csv")
        
        # Set up playback parameters
        self.frame_index = 0
        self.max_frames = 180  # Limit to first 180 frames
        self.completed = False
        
        # Create timer for publishing at 10Hz (0.1 seconds)
        self.timer = self.create_timer(0.1, self.publish_pointcloud)
        self.get_logger().info("PCD Publisher initialized. Starting playback...")

    def publish_tf(self):
        """
        Publish TF transforms between coordinate frames:
        - unilidar_imu_initial -> unilidar_imu (static)
        - unilidar_imu -> unilidar_lidar (static)
        """
        current_time = self.get_clock().now().to_msg()
        
        # First transform: unilidar_imu_initial -> unilidar_imu
        imu_transform = TransformStamped()
        imu_transform.header.stamp = current_time
        imu_transform.header.frame_id = "unilidar_imu_initial"
        imu_transform.child_frame_id = "unilidar_imu"
        
        # Identity transform (no rotation or translation)
        imu_transform.transform.rotation.w = 1.0
        imu_transform.transform.rotation.x = 0.0
        imu_transform.transform.rotation.y = 0.0
        imu_transform.transform.rotation.z = 0.0
        imu_transform.transform.translation.x = 0.0
        imu_transform.transform.translation.y = 0.0
        imu_transform.transform.translation.z = 0.0
        
        # Second transform: unilidar_imu -> unilidar_lidar
        lidar_transform = TransformStamped()
        lidar_transform.header.stamp = current_time
        lidar_transform.header.frame_id = "unilidar_imu"
        lidar_transform.child_frame_id = "unilidar_lidar"
        
        # Identity transform (no rotation or translation)
        lidar_transform.transform.rotation.w = 1.0
        lidar_transform.transform.rotation.x = 0.0
        lidar_transform.transform.rotation.y = 0.0
        lidar_transform.transform.rotation.z = 0.0
        lidar_transform.transform.translation.x = 0.0
        lidar_transform.transform.translation.y = 0.0
        lidar_transform.transform.translation.z = 0.0
        
        # Publish both transforms together
        self.tf_broadcaster.sendTransform([imu_transform, lidar_transform])
        
    def publish_pointcloud(self):
        """
        Publish the current frame's point cloud data and IMU message.
        Advances to the next frame after publishing.
        """
        # Check if we've completed all frames
        if self.frame_index >= self.max_frames:
            self.get_logger().info("Published all frames. Shutting down...")
            self.completed = True
            rclpy.shutdown()
            return
            
        # Get current frame
        i = self.frame_index
        self.get_logger().info(f"Publishing frame {i} of {self.max_frames}")
        
        # Get points for current frame
        filtered_df_1 = self.df[self.df["frame_id"].isin([i])]
        
        # Publish coordinate transforms
        self.publish_tf()
        
        # Extract XYZ points
        points = filtered_df_1[["meas_pos_x", "meas_pos_y", "meas_pos_z"]].values
        
        # Convert to numpy array
        xyz_points = np.asarray(points)
        
        # Create array with 6 dimensions for PointCloud2 fields
        num_points = xyz_points.shape[0]
        points_array = np.zeros((num_points, 6), dtype=np.float32)
        
        # Fill in point data
        points_array[:, 0:3] = xyz_points         # XYZ coordinates
        points_array[:, 3] = 100.0                # intensity (default value)
        points_array[:, 4] = 0.0                  # ring (default value)
        points_array[:, 5] = 0.0                  # time (default value)
        
        # Create PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "unilidar_lidar"
        
        # Define point fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='ring', offset=16, datatype=PointField.UINT16, count=1),
            PointField(name='time', offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Create and publish point cloud message
        msg = pc2.create_cloud(header, fields, points_array.tolist())
        msg.is_dense = True
        self.publisher_.publish(msg)

        # Create and publish IMU message
        header_imu = Header()
        header_imu.stamp = self.get_clock().now().to_msg()
        header_imu.frame_id = "unilidar_imu"
        imu_msg = Imu()
        imu_msg.header = header_imu

        # Set default IMU values (no motion)
        # Orientation (identity quaternion)
        imu_msg.orientation.w = 1.0
        imu_msg.orientation.x = 0.0
        imu_msg.orientation.y = 0.0
        imu_msg.orientation.z = 0.0

        # No angular velocity
        imu_msg.angular_velocity.x = 0.0
        imu_msg.angular_velocity.y = 0.0
        imu_msg.angular_velocity.z = 0.0

        # No linear acceleration
        imu_msg.linear_acceleration.x = 0.0
        imu_msg.linear_acceleration.y = 0.0
        imu_msg.linear_acceleration.z = 0.0

        # Set covariance matrices to "unknown" (-1)
        imu_msg.orientation_covariance = [-1.0] * 9
        imu_msg.angular_velocity_covariance = [-1.0] * 9
        imu_msg.linear_acceleration_covariance = [-1.0] * 9

        # Publish IMU message
        self.publisher_imu.publish(imu_msg)

        # Advance to next frame
        self.frame_index += 1

if __name__ == '__main__':
    # Initialize ROS
    rclpy.init()
    
    # Create and start publisher node
    pcd_publisher = PCDPublisher()
    
    try:
        # Spin node until shutdown
        rclpy.spin(pcd_publisher)
    except KeyboardInterrupt:
        # Handle Ctrl+C
        pass
    finally:
        # Clean up
        if not pcd_publisher.completed:
            pcd_publisher.get_logger().info("Interrupted. Shutting down...")
        pcd_publisher.destroy_node()
        if not rclpy.ok():
            rclpy.shutdown()