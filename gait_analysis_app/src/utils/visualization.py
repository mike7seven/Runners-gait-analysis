import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mediapipe as mp

def visualize_pose(image, landmarks):
    """Visualize pose landmarks on an image."""
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # Create a copy of the image
    annotated_image = image.copy()
    
    # Draw the pose landmarks
    mp_drawing.draw_landmarks(
        annotated_image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )
    
    return annotated_image

def plot_joint_angles(angles, joint_names, title="Joint Angles Over Time"):
    """Plot joint angles over time."""
    plt.figure(figsize=(12, 6))
    for i, joint in enumerate(joint_names):
        plt.plot(angles[:, i], label=joint)
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def animate_poses(poses, fps=30):
    """Create an animation of pose keypoints over time."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set plot limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()  # Invert y-axis to match image coordinates
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], s=10)
    
    # Animation update function
    def update(frame):
        frame_data = poses[frame]
        x = [point[0] for point in frame_data]
        y = [point[1] for point in frame_data]
        scatter.set_offsets(np.column_stack([x, y]))
        return scatter,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(poses), interval=1000/fps, blit=True)
    
    return anim
