import numpy as np

class GaitEmbeddingGenerator:
    def __init__(self, embedding_dim=256):
        self.embedding_dim = embedding_dim
    
    def generate_embedding(self, poses):
        """
        Create a simple embedding directly from MediaPipe pose landmarks.
        
        Args:
            poses: Array of pose landmarks from MediaPipe [frames, landmarks, coordinates]
            
        Returns:
            embedding: A normalized vector representation
        """
        # If no poses detected, return a zero vector
        if len(poses) == 0:
            return np.zeros(self.embedding_dim)
            
        # Process the landmark data
        # For simplicity, we'll use basic statistical features of the landmarks
        
        # 1. Calculate mean pose across all frames
        mean_pose = np.mean(poses, axis=0)
        # 2. Calculate standard deviation to capture variation in movement
        std_pose = np.std(poses, axis=0)
        
        # 3. Flatten these statistics into vectors
        mean_flat = mean_pose.flatten()
        std_flat = std_pose.flatten()
        
        # 4. Concatenate them to form our feature vector
        features = np.concatenate([mean_flat, std_flat])
        
        # 5. Resize to our target embedding dimension
        if len(features) > self.embedding_dim:
            # If too large, sample down to embedding_dim
            indices = np.linspace(0, len(features)-1, self.embedding_dim, dtype=int)
            embedding = features[indices]
        else:
            # If too small, pad with zeros
            embedding = np.pad(features, (0, self.embedding_dim - len(features)))
        
        # 6. Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding# import numpy as np

# class GaitEmbeddingGenerator:
#     def __init__(self, embedding_dim=512):
#         self.embedding_dim = embedding_dim
    
#     def extract_gait_features(self, poses):
#         """Extract key gait features from pose sequence."""
#         # Example features (expand based on your specific needs):
        
#         # 1. Joint angles over time (hip, knee, ankle)
#         hip_angles = self._calculate_joint_angles(poses, 'hip')
#         knee_angles = self._calculate_joint_angles(poses, 'knee')
#         ankle_angles = self._calculate_joint_angles(poses, 'ankle')
        
#         # 2. Stride metrics
#         stride_length = self._calculate_stride_length(poses)
#         stride_time = self._calculate_stride_time(poses)
        
#         # 3. Stability metrics
#         center_of_mass_variation = self._calculate_com_variation(poses)
        
#         # Combine features
#         features = np.concatenate([
#             hip_angles, knee_angles, ankle_angles,
#             stride_length, stride_time, center_of_mass_variation
#         ])
        
#         return features
        
#     def generate_embedding(self, poses):
#         """Create a fixed-length embedding from variable-length pose sequence."""
#         # Extract meaningful features
#         features = self.extract_gait_features(poses)
        
#         # Create fixed-length embedding (simplified example)
#         # In practice, you might use a neural network here
#         if len(features) > self.embedding_dim:
#             # Downsample if too many features
#             indices = np.linspace(0, len(features)-1, self.embedding_dim, dtype=int)
#             embedding = features[indices]
#         else:
#             # Pad if too few features
#             embedding = np.pad(features, (0, self.embedding_dim - len(features)))
            
#         # Normalize embedding
#         embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
#         return embedding
    
#     # Helper methods - placeholder implementations
#     def _calculate_joint_angles(self, poses, joint_type):
#         # Placeholder function, implement actual calculation based on pose landmarks
#         return np.zeros(10)  # Return dummy data for now
        
#     def _calculate_stride_length(self, poses):
#         # Placeholder function
#         return np.zeros(5)
        
#     def _calculate_stride_time(self, poses):
#         # Placeholder function
#         return np.zeros(5)
        
#     def _calculate_com_variation(self, poses):
#         # Placeholder function for center of mass variation
#         return np.zeros(5)
