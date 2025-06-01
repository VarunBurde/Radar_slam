import os
import subprocess
import numpy as np
import cv2
import pyrealsense2 as rs

def read_realsense_rgb(bag_file_path, output_folder=None, max_frames=10):
    """
    Extract RGB images from a RealSense bag file
    
    Args:
        bag_file_path: Path to the .bag file
        output_folder: Optional folder to save images
        max_frames: Maximum number of frames to extract
    """
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        
    print(f"Reading RGB images from: {bag_file_path}")
    
    if not os.path.exists(bag_file_path):
        print(f"Error: Bag file not found at {bag_file_path}")
        return []

    pipeline = None # Initialize pipeline to None for finally block
    rgb_frames = []

    try:
        # Method 1: Using rs.pipeline directly
        print("Attempting to read bag file using pipeline method...")
        pipeline = rs.pipeline()
        config = rs.config()
        
        rs.config.enable_device_from_file(config, bag_file_path, repeat_playback=False)
        config.enable_stream(rs.stream.color) # Try to enable any color stream
        
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        
        frame_count = 0
        while frame_count < max_frames:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=10000) # Increased timeout
                if not frames:
                    print("No frames received from pipeline, bag might have ended or stream not available.")
                    break
            except RuntimeError as e:
                print(f"RuntimeError waiting for frames: {e}. Assuming end of bag or stream issue.")
                break # End of bag or stream not available

            color_frame = frames.get_color_frame()
            if not color_frame:
                print(f"No color frame in frameset at count {frame_count}. Skipping.")
                # Check if other frames exist to see if it's just a missing color frame
                depth_frame = frames.get_depth_frame()
                if not depth_frame and not frames.first_or_default(rs.stream.infrared):
                     print("No other frames found either. Likely end of bag.")
                     break # No other frames, likely end of bag
                continue

            timestamp = color_frame.get_timestamp() # Milliseconds
            color_image = np.asanyarray(color_frame.get_data())
            
            # RealSense SDK typically provides BGR8 data for color frames when read as numpy array.
            # If it's RGB, conversion to BGR for OpenCV would be:
            # color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            # However, usually it's already BGR.
            color_image_bgr = color_image 
            
            rgb_frames.append({'timestamp': timestamp, 'image': color_image_bgr})
            
            if output_folder:
                filename = f"{output_folder}/rgb_{frame_count:04d}_{int(timestamp)}.png"
                cv2.imwrite(filename, color_image_bgr)
                print(f"Saved: {filename}")
            
            frame_count += 1
            print(f"Extracted RGB frame {frame_count}/{max_frames} - timestamp: {timestamp/1000.0:.3f}s")

        if rgb_frames:
            print(f"Extracted {len(rgb_frames)} RGB frames using pipeline method.")
            return rgb_frames
        else:
            print("Pipeline method did not extract any frames.")

    except RuntimeError as e:
        print(f"Pipeline method failed: {e}")
        # Fall through to alternative methods if necessary, or handle error
    except Exception as e:
        print(f"An unexpected error occurred during pipeline processing: {e}")
    finally:
        if pipeline:
            try:
                pipeline.stop()
            except Exception as e:
                print(f"Error stopping pipeline: {e}")
    
    # If pipeline method failed or yielded no frames, suggest alternatives
    print("\nCould not extract frames using the pipeline method.")
    print("Consider checking bag file integrity or stream configuration.")
    print("Alternative: Use realsense-viewer to export frames manually:")
    print(f"  realsense-viewer -f {bag_file_path}")
    try:
        result = subprocess.run(['which', 'realsense-viewer'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"\nRealsense-viewer is available. Run: realsense-viewer -f {bag_file_path}")
        else:
            print("\nRealsense-viewer might not be installed. Install it with: sudo apt install librealsense2-utils")
    except FileNotFoundError:
        print("\nCouldn't check for realsense-viewer (command 'which' not found).")
        
    return []


def extract_realsense_intrinsics(bag_file_path):
    """
    Extract intrinsic parameters from a RealSense bag file.
    Assumes the bag contains a color stream.
    """
    print(f"Extracting camera intrinsics from RealSense bag file: {bag_file_path}")
    
    pipeline = None
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_file_path, repeat_playback=False)
        
        # Attempt to start the pipeline to get active streams
        # We don't need to enable specific streams if we just want to query profiles
        # However, sometimes starting with a generic color stream helps initialize device info
        try:
            config.enable_stream(rs.stream.color) 
        except RuntimeError as e:
            print(f"Could not enable generic color stream for intrinsics: {e}. Trying to proceed without.")

        profile = pipeline.start(config) # Start pipeline to access device and streams
        
        # Iterate over all streams to find a color stream and get its intrinsics
        for stream_profile in profile.get_streams():
            if stream_profile.stream_type() == rs.stream.color:
                video_profile = stream_profile.as_video_stream_profile()
                if video_profile:
                    intrinsics = video_profile.get_intrinsics()
                    
                    width = intrinsics.width
                    height = intrinsics.height
                    fx = intrinsics.fx
                    fy = intrinsics.fy
                    ppx = intrinsics.ppx
                    ppy = intrinsics.ppy
                    distortion_model = intrinsics.model
                    distortion_coeffs = intrinsics.coeffs
                    
                    print(f"Successfully extracted camera intrinsics:")
                    print(f"  Resolution: {width}x{height}")
                    print(f"  Focal lengths: fx={fx}, fy={fy}")
                    print(f"  Principal point: ppx={ppx}, ppy={ppy}")
                    print(f"  Distortion model: {distortion_model}")
                    print(f"  Distortion coefficients: {distortion_coeffs}")
                    
                    return {
                        "resolution": {"width": width, "height": height},
                        "intrinsic_matrix": {"fx": fx, "fy": fy, "ppx": ppx, "ppy": ppy},
                        "distortion": {"model": str(distortion_model), "coeffs": list(distortion_coeffs)}
                    }
        
        print("Could not find a color video stream profile in the bag file.")
        return None
        
    except RuntimeError as e:
        print(f"RuntimeError extracting camera intrinsics: {e}")
        print("This might happen if the bag file doesn't contain a recognizable color stream or is corrupted.")
        return None
    except Exception as e:
        print(f"Unexpected error extracting camera intrinsics: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if pipeline:
            try:
                pipeline.stop()
            except Exception as e:
                print(f"Error stopping pipeline during intrinsics extraction: {e}")
