import cv2
import mediapipe as mp
import numpy as np

class PoseExtractor:
    def __init__(self, model_complexity=2):  # Using full complexity for M3 Mac
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def process_video(self, video_path):
        """Extract pose landmarks from video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        poses = []
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            # Convert to RGB & process
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Extract landmark coordinates
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    frame_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                poses.append(frame_landmarks)
            
        cap.release()
        return np.array(poses), fps
