import numpy as np

# MediaPipe landmark indices for important joints
# These indices correspond to specific body parts in the MediaPipe pose model
LANDMARK_INDICES = {
    'left_ankle': 27,
    'right_ankle': 28,
    'left_knee': 25,
    'right_knee': 26,
    'left_hip': 23,
    'right_hip': 24,
    'left_shoulder': 11,
    'right_shoulder': 12
}

def calculate_stride_length(poses):
    """
    Calculate the average stride length from a sequence of pose landmarks.
    
    Stride length is the distance between successive placements of the same foot.
    We'll measure this by tracking the horizontal movement of the ankles.
    
    Args:
        poses: Array of pose landmarks from MediaPipe [frames, landmarks, coordinates]
        
    Returns:
        stride_length: Average stride length in relative units
    """
    # Make sure we have enough frames to calculate stride
    if len(poses) < 10:
        return 0.0
    
    # Extract ankle positions over time
    left_ankle_x = poses[:, LANDMARK_INDICES['left_ankle'], 0]
    right_ankle_x = poses[:, LANDMARK_INDICES['right_ankle'], 0]
    
    # Find local maxima for each ankle (when the foot is extended forward)
    # We'll use a simple method: a point is a maximum if it's larger than its neighbors
    left_peaks = []
    right_peaks = []
    
    # For the left ankle
    for i in range(1, len(left_ankle_x) - 1):
        if left_ankle_x[i] > left_ankle_x[i-1] and left_ankle_x[i] > left_ankle_x[i+1]:
            left_peaks.append(i)
    
    # For the right ankle
    for i in range(1, len(right_ankle_x) - 1):
        if right_ankle_x[i] > right_ankle_x[i-1] and right_ankle_x[i] > right_ankle_x[i+1]:
            right_peaks.append(i)
    
    # If we couldn't find enough peaks, return 0
    if len(left_peaks) < 2 and len(right_peaks) < 2:
        return 0.0
    
    # Calculate average distance between consecutive peaks for each foot
    stride_lengths = []
    
    # Left foot strides
    if len(left_peaks) >= 2:
        for i in range(1, len(left_peaks)):
            # Get the x positions of the ankle at each peak
            pos1 = left_ankle_x[left_peaks[i-1]]
            pos2 = left_ankle_x[left_peaks[i]]
            # Calculate the distance
            stride_lengths.append(abs(pos2 - pos1))
    
    # Right foot strides
    if len(right_peaks) >= 2:
        for i in range(1, len(right_peaks)):
            # Get the x positions of the ankle at each peak
            pos1 = right_ankle_x[right_peaks[i-1]]
            pos2 = right_ankle_x[right_peaks[i]]
            # Calculate the distance
            stride_lengths.append(abs(pos2 - pos1))
    
    # Return the average stride length, if we found any
    if stride_lengths:
        return np.mean(stride_lengths)
    else:
        return 0.0

def calculate_cadence(poses, fps):
    """
    Calculate cadence (steps per minute) from pose sequence.
    
    Cadence is the rate at which a person walks, expressed in steps per minute.
    
    Args:
        poses: Array of pose landmarks from MediaPipe [frames, landmarks, coordinates]
        fps: Frames per second of the video
        
    Returns:
        cadence: Steps per minute
    """
    # We'll use the same approach as for stride length - find peaks in ankle movement
    if len(poses) < 10:
        return 0.0
    
    # Extract ankle y positions (vertical movement)
    # Using Y position here as it often gives clearer steps than X
    left_ankle_y = poses[:, LANDMARK_INDICES['left_ankle'], 1]
    right_ankle_y = poses[:, LANDMARK_INDICES['right_ankle'], 1]
    
    # Find local minima (when foot touches ground)
    left_steps = []
    right_steps = []
    
    # For the left ankle (lower y value means foot is down)
    for i in range(1, len(left_ankle_y) - 1):
        if left_ankle_y[i] < left_ankle_y[i-1] and left_ankle_y[i] < left_ankle_y[i+1]:
            left_steps.append(i)
    
    # For the right ankle
    for i in range(1, len(right_ankle_y) - 1):
        if right_ankle_y[i] < right_ankle_y[i-1] and right_ankle_y[i] < right_ankle_y[i+1]:
            right_steps.append(i)
    
    # Total number of steps detected
    total_steps = len(left_steps) + len(right_steps)
    
    if total_steps < 2:
        return 0.0
    
    # Calculate time elapsed
    total_frames = len(poses)
    time_seconds = total_frames / fps
    
    # Calculate steps per minute
    steps_per_minute = (total_steps / time_seconds) * 60
    
    return steps_per_minute

def calculate_symmetry(poses):
    """
    Calculate gait symmetry as a percentage.
    
    A perfectly symmetrical gait would have 100% symmetry.
    We'll measure symmetry by comparing the movement patterns of left and right sides.
    
    Args:
        poses: Array of pose landmarks from MediaPipe [frames, landmarks, coordinates]
        
    Returns:
        symmetry: Gait symmetry as a percentage (0-100)
    """
    if len(poses) < 10:
        return 0.0
    
    # We'll calculate symmetry by comparing:
    # 1. Stride lengths of left and right feet
    # 2. Range of motion of left and right knees
    # 3. Hip movement patterns
    
    # 1. Stride length symmetry
    left_ankle_x = poses[:, LANDMARK_INDICES['left_ankle'], 0]
    right_ankle_x = poses[:, LANDMARK_INDICES['right_ankle'], 0]
    
    left_range = np.max(left_ankle_x) - np.min(left_ankle_x)
    right_range = np.max(right_ankle_x) - np.min(right_ankle_x)
    
    # Calculate ratio (smaller / larger) to get a value between 0 and 1
    stride_symmetry = min(left_range, right_range) / max(left_range, right_range) if max(left_range, right_range) > 0 else 0
    
    # 2. Knee range of motion symmetry
    left_knee_angles = calculate_joint_angles(poses, 'left_knee')
    right_knee_angles = calculate_joint_angles(poses, 'right_knee')
    
    left_knee_range = np.max(left_knee_angles) - np.min(left_knee_angles)
    right_knee_range = np.max(right_knee_angles) - np.min(right_knee_angles)
    
    knee_symmetry = min(left_knee_range, right_knee_range) / max(left_knee_range, right_knee_range) if max(left_knee_range, right_knee_range) > 0 else 0
    
    # 3. Hip movement symmetry
    left_hip_y = poses[:, LANDMARK_INDICES['left_hip'], 1]
    right_hip_y = poses[:, LANDMARK_INDICES['right_hip'], 1]
    
    left_hip_range = np.max(left_hip_y) - np.min(left_hip_y)
    right_hip_range = np.max(right_hip_y) - np.min(right_hip_y)
    
    hip_symmetry = min(left_hip_range, right_hip_range) / max(left_hip_range, right_hip_range) if max(left_hip_range, right_hip_range) > 0 else 0
    
    # Combine the symmetry scores (weighted average)
    overall_symmetry = (0.4 * stride_symmetry + 0.4 * knee_symmetry + 0.2 * hip_symmetry) * 100
    
    return overall_symmetry

def calculate_joint_angles(poses, joint_name):
    """
    Calculate the angle of a joint over time.
    
    For knees: measures angle between hip, knee, and ankle
    For hips: measures angle between shoulder, hip, and knee
    
    Args:
        poses: Array of pose landmarks from MediaPipe [frames, landmarks, coordinates]
        joint_name: Name of the joint ('left_knee', 'right_knee', 'left_hip', 'right_hip')
        
    Returns:
        angles: Array of angles in degrees for each frame
    """
    angles = []
    
    if joint_name == 'left_knee':
        hip_idx = LANDMARK_INDICES['left_hip']
        knee_idx = LANDMARK_INDICES['left_knee']
        ankle_idx = LANDMARK_INDICES['left_ankle']
    elif joint_name == 'right_knee':
        hip_idx = LANDMARK_INDICES['right_hip']
        knee_idx = LANDMARK_INDICES['right_knee']
        ankle_idx = LANDMARK_INDICES['right_ankle']
    elif joint_name == 'left_hip':
        shoulder_idx = LANDMARK_INDICES['left_shoulder']
        hip_idx = LANDMARK_INDICES['left_hip']
        knee_idx = LANDMARK_INDICES['left_knee']
    elif joint_name == 'right_hip':
        shoulder_idx = LANDMARK_INDICES['right_shoulder']
        hip_idx = LANDMARK_INDICES['right_hip']
        knee_idx = LANDMARK_INDICES['right_knee']
    else:
        return np.zeros(len(poses))
    
    for frame in range(len(poses)):
        if joint_name.endswith('knee'):
            # For knee, calculate angle between hip, knee, and ankle
            hip = poses[frame, hip_idx, :2]  # x, y coordinates
            knee = poses[frame, knee_idx, :2]
            ankle = poses[frame, ankle_idx, :2]
            
            # Calculate vectors
            vector1 = hip - knee
            vector2 = ankle - knee
            
            # Calculate angle between vectors
            angle = calculate_angle_between_vectors(vector1, vector2)
            angles.append(angle)
        else:
            # For hip, calculate angle between shoulder, hip, and knee
            shoulder = poses[frame, shoulder_idx, :2]
            hip = poses[frame, hip_idx, :2]
            knee = poses[frame, knee_idx, :2]
            
            # Calculate vectors
            vector1 = shoulder - hip
            vector2 = knee - hip
            
            # Calculate angle between vectors
            angle = calculate_angle_between_vectors(vector1, vector2)
            angles.append(angle)
    
    return np.array(angles)

def calculate_angle_between_vectors(v1, v2):
    """
    Calculate the angle between two 2D vectors in degrees.
    
    Args:
        v1: First vector [x, y]
        v2: Second vector [x, y]
        
    Returns:
        angle: Angle in degrees
    """
    # Dot product
    dot = np.dot(v1, v2)
    
    # Calculate magnitudes
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    # Calculate cosine of the angle
    cos_angle = dot / (mag1 * mag2) if (mag1 * mag2) > 0 else 0
    
    # Clamp to ensure it's within valid range for arccos
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    
    # Calculate angle in degrees
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    return angle

def analyze_gait(poses, fps):
    """
    Perform a comprehensive gait analysis on a sequence of poses.
    
    Args:
        poses: Array of pose landmarks from MediaPipe [frames, landmarks, coordinates]
        fps: Frames per second of the video
        
    Returns:
        metrics: Dictionary of gait metrics
    """
    # Calculate all our metrics
    stride_length = calculate_stride_length(poses)
    cadence = calculate_cadence(poses, fps)
    symmetry = calculate_symmetry(poses)
    
    # Calculate joint angles
    left_knee_angles = calculate_joint_angles(poses, 'left_knee')
    right_knee_angles = calculate_joint_angles(poses, 'right_knee')
    left_hip_angles = calculate_joint_angles(poses, 'left_hip')
    right_hip_angles = calculate_joint_angles(poses, 'right_hip')
    
    # Assemble results
    metrics = {
        'stride_length': stride_length,
        'cadence': cadence,
        'symmetry': symmetry,
        'avg_left_knee_angle': np.mean(left_knee_angles),
        'avg_right_knee_angle': np.mean(right_knee_angles),
        'max_left_knee_angle': np.max(left_knee_angles),
        'max_right_knee_angle': np.max(right_knee_angles),
        'avg_left_hip_angle': np.mean(left_hip_angles),
        'avg_right_hip_angle': np.mean(right_hip_angles)
    }
    
    return metrics

# # src/analysis/metrics.py
# import numpy as np

# def calculate_stride_length(poses):
#     """
#     Calculate stride length from pose landmarks.
    
#     Stride length is the distance between heel strikes of alternate feet.
    
#     Args:
#         poses (np.array): 3D array of pose landmarks [frames, landmarks, [x,y,z,visibility]]
    
#     Returns:
#         float: Estimated stride length in relative units
    
#     Key Machine Learning Concept: Feature Extraction
#     - We're extracting meaningful biomechanical information from raw pose data
#     - Uses spatial relationships between body landmarks
#     """
#     # MediaPipe landmark indices for key points
#     LEFT_HEEL_INDEX = 29
#     RIGHT_HEEL_INDEX = 30

#     # Check if we have enough frames and landmarks
#     if len(poses) < 2 or poses.shape[1] <= max(LEFT_HEEL_INDEX, RIGHT_HEEL_INDEX):
#         return 0.0

#     # Calculate heel positions across frames
#     left_heel_positions = poses[:, LEFT_HEEL_INDEX, :2]  # x, y coordinates
#     right_heel_positions = poses[:, RIGHT_HEEL_INDEX, :2]  # x, y coordinates

#     # Calculate distances between consecutive heel positions
#     left_heel_distances = np.sqrt(np.sum(np.diff(left_heel_positions, axis=0)**2, axis=1))
#     right_heel_distances = np.sqrt(np.sum(np.diff(right_heel_positions, axis=0)**2, axis=1))

#     # Average the maximum distances as an estimate of stride length
#     # This captures the maximum lateral movement of heels
#     stride_length = np.mean([
#         np.max(left_heel_distances) if len(left_heel_distances) > 0 else 0,
#         np.max(right_heel_distances) if len(right_heel_distances) > 0 else 0
#     ])

#     return stride_length

# def calculate_cadence(poses, fps):
#     """
#     Calculate walking cadence (steps per minute).
    
#     Cadence is the number of steps taken in a given time period.
    
#     Args:
#         poses (np.array): 3D array of pose landmarks
#         fps (float): Frames per second of the video
    
#     Returns:
#         float: Estimated cadence in steps per minute
    
#     Key Machine Learning Concept: Time Series Analysis
#     - We're analyzing temporal patterns in movement data
#     - Extracting rhythmic characteristics of walking
#     """
#     # MediaPipe landmark indices
#     LEFT_HEEL_INDEX = 29
#     RIGHT_HEEL_INDEX = 30

#     # Check if we have enough frames and landmarks
#     if len(poses) < 2 or poses.shape[1] <= max(LEFT_HEEL_INDEX, RIGHT_HEEL_INDEX):
#         return 0.0

#     # Detect heel strikes (significant vertical movement)
#     left_heel_y = poses[:, LEFT_HEEL_INDEX, 1]
#     right_heel_y = poses[:, RIGHT_HEEL_INDEX, 1]

#     # Simple heel strike detection (when heel moves significantly)
#     def detect_heel_strikes(heel_positions):
#         # Calculate vertical changes between frames
#         heel_changes = np.diff(heel_positions)
        
#         # Identify significant downward movements (heel strikes)
#         heel_strikes = np.where(heel_changes < -0.02)[0]
#         return heel_strikes

#     left_strikes = detect_heel_strikes(left_heel_y)
#     right_strikes = detect_heel_strikes(right_heel_y)

#     # Combine and sort heel strikes
#     all_strikes = np.sort(np.concatenate([left_strikes, right_strikes]))

#     # Calculate steps (number of strikes)
#     num_steps = len(all_strikes)

#     # Calculate cadence (steps per minute)
#     # Total time in minutes = number of frames / (fps * 60)
#     # Cadence = number of steps / total time
#     total_time_minutes = len(poses) / (fps * 60)
#     cadence = num_steps / total_time_minutes if total_time_minutes > 0 else 0

#     return cadence

# def calculate_symmetry(poses):
#     """
#     Calculate gait symmetry by comparing left and right side movements.
    
#     Args:
#         poses (np.array): 3D array of pose landmarks
    
#     Returns:
#         float: Symmetry score (percentage, 100 = perfect symmetry)
    
#     Key Machine Learning Concept: Comparative Feature Analysis
#     - Comparing movement patterns between body sides
#     - Quantifying biomechanical balance
#     """
#     # Key landmark indices for comparison
#     LEFT_HIP = 23
#     RIGHT_HIP = 24
#     LEFT_KNEE = 25
#     RIGHT_KNEE = 26
#     LEFT_ANKLE = 27
#     RIGHT_ANKLE = 28

#     # Extract key joint positions
#     left_hip_positions = poses[:, LEFT_HIP, :2]
#     right_hip_positions = poses[:, RIGHT_HIP, :2]
#     left_knee_positions = poses[:, LEFT_KNEE, :2]
#     right_knee_positions = poses[:, RIGHT_KNEE, :2]
#     left_ankle_positions = poses[:, LEFT_ANKLE, :2]
#     right_ankle_positions = poses[:, RIGHT_ANKLE, :2]

#     # Calculate movement ranges for each side
#     def calculate_movement_range(positions):
#         return np.ptp(positions, axis=0)  # Peak to peak - max displacement

#     # Compare movement ranges
#     hip_range_diff = np.abs(calculate_movement_range(left_hip_positions) - 
#                              calculate_movement_range(right_hip_positions))
#     knee_range_diff = np.abs(calculate_movement_range(left_knee_positions) - 
#                               calculate_movement_range(right_knee_positions))
#     ankle_range_diff = np.abs(calculate_movement_range(left_ankle_positions) - 
#                                calculate_movement_range(right_ankle_positions))

#     # Calculate symmetry (lower difference means more symmetry)
#     # Normalize and convert to a percentage
#     symmetry_score = max(0, 100 - (np.mean([
#         np.linalg.norm(hip_range_diff),
#         np.linalg.norm(knee_range_diff),
#         np.linalg.norm(ankle_range_diff)
#     ]) * 100))

#     return symmetry_score

# # Optional: Main block for testing
# if __name__ == "__main__":
#     # You can add test code here to verify metrics calculation
#     # Example: load a sample video and print metrics
#     from ..pose.extractor import PoseExtractor

#     # Path to a sample video
#     video_path = "../data/raw/reference_gaits/16_35.mpg"
    
#     # Extract poses
#     extractor = PoseExtractor()
#     poses, fps = extractor.process_video(video_path)
    
#     # Calculate metrics
#     stride_length = calculate_stride_length(poses)
#     cadence = calculate_cadence(poses, fps)
#     symmetry = calculate_symmetry(poses)
    
#     print(f"Stride Length: {stride_length:.4f}")
#     print(f"Cadence: {cadence:.2f} steps/minute")
#     print(f"Symmetry: {symmetry:.2f}%")