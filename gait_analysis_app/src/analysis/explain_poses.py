# Explanation of Pose Data Structure
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.pose.extractor import PoseExtractor

import numpy as np





def explain_poses_structure():
    """
    Demonstrate the structure of pose data extracted from a video.
    """
    # Path to a sample video
    video_path = "data/raw/reference_gaits/16_35.mpg"
    
    # Extract poses using our PoseExtractor
    extractor = PoseExtractor()
    poses, fps = extractor.process_video(video_path)
    
    # Let's break down the structure of 'poses'
    print("Pose Data Explanation:")
    print("--------------------")
    print(f"1. Total number of frames: {poses.shape[0]}")
    print(f"2. Number of landmarks per frame: {poses.shape[1]}")
    print(f"3. Data for each landmark: {poses.shape[2]} values")
    print("\nExample Landmark Structure:")
    print("Each landmark is [x, y, z, visibility]")
    print("- x, y, z: Spatial coordinates (0-1 range)")
    print("- visibility: Confidence of landmark detection (0-1)")
    
    # Let's look at the first frame, first landmark
    first_frame_first_landmark = poses[0, 0]
    print("\nFirst Frame, First Landmark:")
    print(first_frame_first_landmark)
    
    # Demonstrate coordinate ranges
    print("\nCoordinate Ranges:")
    print(f"X range: {poses[:,:,0].min()} - {poses[:,:,0].max()}")
    print(f"Y range: {poses[:,:,1].min()} - {poses[:,:,1].max()}")
    print(f"Z range: {poses[:,:,2].min()} - {poses[:,:,2].max()}")
    print(f"Visibility range: {poses[:,:,3].min()} - {poses[:,:,3].max()}")

# Run the explanation when the script is executed
if __name__ == "__main__":
    explain_poses_structure()