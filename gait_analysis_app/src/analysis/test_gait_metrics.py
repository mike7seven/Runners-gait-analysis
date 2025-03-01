"""
Test script for gait metrics calculation.

This script processes a video, extracts pose landmarks, and calculates gait metrics.
"""


import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from src.pose.extractor import PoseExtractor
from src.analysis.metrics import analyze_gait

import numpy as np

def test_gait_metrics(video_path):
    """
    Test gait metrics calculation on a video file.
    
    Args:
        video_path: Path to the video file
    """
    print(f"Testing gait metrics with: {video_path}")
    
    # Step 1: Extract poses from video
    extractor = PoseExtractor(model_complexity=2)  # Use highest accuracy
    poses, fps = extractor.process_video(video_path)
    
    if len(poses) == 0:
        print("❌ No poses detected in video")
        return
    
    print(f"✅ Successfully extracted {len(poses)} frames of pose data at {fps} fps")
    
    # Step 2: Calculate gait metrics
    metrics = analyze_gait(poses, fps)
    
    # Step 3: Display results
    print("\nGait Analysis Results:")
    print("-" * 25)
    print(f"Stride Length: {metrics['stride_length']:.2f} (relative units)")
    print(f"Cadence: {metrics['cadence']:.2f} steps/minute")
    print(f"Symmetry Score: {metrics['symmetry']:.2f}%")
    print()
    print("Joint Angles:")
    print(f"  Left Knee (avg/max): {metrics['avg_left_knee_angle']:.2f}° / {metrics['max_left_knee_angle']:.2f}°")
    print(f"  Right Knee (avg/max): {metrics['avg_right_knee_angle']:.2f}° / {metrics['max_right_knee_angle']:.2f}°")
    print(f"  Left Hip (avg): {metrics['avg_left_hip_angle']:.2f}°")
    print(f"  Right Hip (avg): {metrics['avg_right_hip_angle']:.2f}°")
    
    # Check for potential issues in the metrics
    identify_potential_issues(metrics)
    
    return metrics

def identify_potential_issues(metrics):
    """
    Identify potential gait issues based on calculated metrics.
    
    Args:
        metrics: Dictionary of gait metrics
    """
    print("\nPotential Observations:")
    
    # Check symmetry
    if metrics['symmetry'] < 80:
        print("- Low symmetry score: Possible uneven gait pattern")
    
    # Compare left/right knee angles
    knee_diff = abs(metrics['avg_left_knee_angle'] - metrics['avg_right_knee_angle'])
    if knee_diff > 10:
        print(f"- Knee angle difference: {knee_diff:.2f}° (possible uneven flexion)")
    
    # Compare left/right hip angles
    hip_diff = abs(metrics['avg_left_hip_angle'] - metrics['avg_right_hip_angle'])
    if hip_diff > 10:
        print(f"- Hip angle difference: {hip_diff:.2f}° (possible uneven hip movement)")
    
    # Cadence check (very general)
    if metrics['cadence'] < 100:
        print("- Low cadence: Slower than typical walking pace")
    elif metrics['cadence'] > 130:
        print("- High cadence: Faster than typical walking pace")

if __name__ == "__main__":
    # Use one of the MPG files in the reference_gaits folder
    video_path = "data/raw/reference_gaits/16_35.mpg"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Check that you're running this script from the 'gait_analysis_app' directory")
    else:
        test_gait_metrics(video_path)