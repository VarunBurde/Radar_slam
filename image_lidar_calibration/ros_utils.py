import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py.point_cloud2 import read_points # Keep if process_pointcloud_data is used with it

def read_rosbag(bag_path, target_topic=None, message_type_filter=None, max_messages_to_extract=5):
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
    
    # Find point cloud topics if none specified and filter is for PointCloud
    pointcloud_topics = []
    if not target_topic and message_type_filter and "PointCloud" in message_type_filter:
        for topic_name, topic_type_val in topic_types.items():
            if "PointCloud" in topic_type_val: # Check if "PointCloud" is in the actual type string
                pointcloud_topics.append(topic_name)
        
        if pointcloud_topics:
            print(f"\nDetected point cloud topics: {pointcloud_topics}")
            if len(pointcloud_topics) > 0:
                target_topic = pointcloud_topics[0] # Default to the first detected
                print(f"Using first point cloud topic: {target_topic}")
    
    print(f"\nReading messages" + (f" from topic: {target_topic}" if target_topic else "") + 
          (f" of type containing: {message_type_filter}" if message_type_filter else ""))
    
    messages = []
    count = 0
    
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        
        # Skip if not the target topic (if a target_topic is specified)
        if target_topic and topic_name != target_topic:
            continue
            
        # Skip if not the target message type (if a message_type_filter is specified)
        current_topic_type = topic_types.get(topic_name, "")
        if message_type_filter and message_type_filter not in current_topic_type:
            continue
            
        msg_type_name = topic_types[topic_name]
        msg_type_class = get_message(msg_type_name)
        msg = deserialize_message(data, msg_type_class)
        
        messages.append({
            'topic': topic_name,
            'timestamp': timestamp, # nanoseconds
            'data': msg
        })
        
        print(f"\nTopic: {topic_name}")
        print(f"Timestamp: {timestamp}")
        if message_type_filter and "PointCloud" in message_type_filter and hasattr(msg, 'data'):
            print(f"Point Cloud message with {len(msg.data) if hasattr(msg, 'data') else '?'} points")
        else:
            print(f"Message: {msg}") # Could be large, consider summarizing
        
        count += 1
        if count >= max_messages_to_extract:
            print(f"\nReached max_messages_to_extract ({max_messages_to_extract}).")
            break
            
    if not messages:
        print(f"No messages found for topic '{target_topic}' and type filter '{message_type_filter}'.")

    return messages, topic_types

def process_pointcloud_data(messages):
    """Process point cloud data from collected messages (example: print info)"""
    if not messages:
        print("No point cloud messages found to process.")
        return
    
    print(f"\nProcessing {len(messages)} point cloud messages (displaying info):")
    
    for i, msg_container in enumerate(messages):
        msg = msg_container['data']
        timestamp = msg_container['timestamp'] # nanoseconds
        
        seconds = timestamp // 10**9
        nanoseconds = timestamp % 10**9
        
        print(f"\nPoint Cloud {i+1}:")
        print(f"  - Timestamp (ns): {timestamp} ({seconds}.{nanoseconds:09d} s)")
        
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
    
    # This function primarily prints info; actual data is in 'messages'
    return messages
