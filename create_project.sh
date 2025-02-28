#!/bin/bash
# Script to create the gait analysis project structure

# Create main project directory
mkdir -p gait_analysis_app

# Create source code structure
mkdir -p gait_analysis_app/src/{pose,embeddings,database,utils}

# Create empty __init__.py files to make directories into Python packages
touch gait_analysis_app/src/__init__.py
touch gait_analysis_app/src/pose/__init__.py
touch gait_analysis_app/src/embeddings/__init__.py
touch gait_analysis_app/src/database/__init__.py
touch gait_analysis_app/src/utils/__init__.py

# Create notebooks directory
mkdir -p gait_analysis_app/notebooks

# Create data directories
mkdir -p gait_analysis_app/data/{raw,processed,embeddings}

# Create config directory
mkdir -p gait_analysis_app/config

# Create main application files
touch gait_analysis_app/app.py
touch gait_analysis_app/requirements.txt
touch gait_analysis_app/README.md
touch gait_analysis_app/config/settings.py

# Create source code files
cat > gait_analysis_app/src/pose/extractor.py << 'EOF'
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
EOF

cat > gait_analysis_app/src/embeddings/generator.py << 'EOF'
import numpy as np

class GaitEmbeddingGenerator:
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim
    
    def extract_gait_features(self, poses):
        """Extract key gait features from pose sequence."""
        # Example features (expand based on your specific needs):
        
        # 1. Joint angles over time (hip, knee, ankle)
        hip_angles = self._calculate_joint_angles(poses, 'hip')
        knee_angles = self._calculate_joint_angles(poses, 'knee')
        ankle_angles = self._calculate_joint_angles(poses, 'ankle')
        
        # 2. Stride metrics
        stride_length = self._calculate_stride_length(poses)
        stride_time = self._calculate_stride_time(poses)
        
        # 3. Stability metrics
        center_of_mass_variation = self._calculate_com_variation(poses)
        
        # Combine features
        features = np.concatenate([
            hip_angles, knee_angles, ankle_angles,
            stride_length, stride_time, center_of_mass_variation
        ])
        
        return features
        
    def generate_embedding(self, poses):
        """Create a fixed-length embedding from variable-length pose sequence."""
        # Extract meaningful features
        features = self.extract_gait_features(poses)
        
        # Create fixed-length embedding (simplified example)
        # In practice, you might use a neural network here
        if len(features) > self.embedding_dim:
            # Downsample if too many features
            indices = np.linspace(0, len(features)-1, self.embedding_dim, dtype=int)
            embedding = features[indices]
        else:
            # Pad if too few features
            embedding = np.pad(features, (0, self.embedding_dim - len(features)))
            
        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        return embedding
    
    # Helper methods - placeholder implementations
    def _calculate_joint_angles(self, poses, joint_type):
        # Placeholder function, implement actual calculation based on pose landmarks
        return np.zeros(10)  # Return dummy data for now
        
    def _calculate_stride_length(self, poses):
        # Placeholder function
        return np.zeros(5)
        
    def _calculate_stride_time(self, poses):
        # Placeholder function
        return np.zeros(5)
        
    def _calculate_com_variation(self, poses):
        # Placeholder function for center of mass variation
        return np.zeros(5)
EOF

cat > gait_analysis_app/src/database/pinecone_db.py << 'EOF'
import pinecone

class GaitDatabase:
    def __init__(self, api_key, environment, index_name, dimension=512):
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
            
        self.index = pinecone.Index(index_name)
        
    def store_gait_embedding(self, id, embedding, metadata=None):
        """Store a gait embedding in Pinecone."""
        self.index.upsert(vectors=[(id, embedding.tolist(), metadata)])
        
    def search_similar_gaits(self, query_embedding, top_k=5):
        """Find similar gait patterns."""
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return results
EOF

cat > gait_analysis_app/src/utils/visualization.py << 'EOF'
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
EOF

# Create a sample notebook
cat > gait_analysis_app/notebooks/pose_extraction.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Gait Analysis: Pose Extraction\n",
    "\n",
    "This notebook demonstrates how to extract pose data from videos using MediaPipe."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import sys\n",
    "import os\n",
    "\n",
    "# Add the parent directory to the path to import from src\n",
    "sys.path.append(os.path.abspath('..'))\n",
    "\n",
    "import cv2\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from src.pose.extractor import PoseExtractor\n",
    "from src.utils.visualization import visualize_pose, animate_poses"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Initialize the pose extractor\n",
    "extractor = PoseExtractor(model_complexity=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Path to your video file\n",
    "video_path = '../data/raw/sample_walking.mp4'  # Replace with your video file\n",
    "\n",
    "# Process the video\n",
    "poses, fps = extractor.process_video(video_path)\n",
    "print(f\"Extracted {len(poses)} frames of pose data at {fps} fps\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Visualize a sample frame\n",
    "# This would require implementing a function to reconstruct MediaPipe landmarks\n",
    "# from our extracted data, which we'll add later\n",
    "\n",
    "# For now, just print the shape of our data\n",
    "print(f\"Pose data shape: {poses.shape}\")\n",
    "print(f\"Each frame has {poses.shape[1]} landmarks with {poses.shape[2]} values (x, y, z, visibility)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Save the extracted poses for later use\n",
    "np.save('../data/processed/sample_poses.npy', poses)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
EOF

cat > gait_analysis_app/notebooks/embedding_generation.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Gait Analysis: Embedding Generation\n",
    "\n",
    "This notebook demonstrates how to generate embeddings from pose data and store them in Pinecone."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import sys\n",
    "import os\n",
    "\n",
    "# Add the parent directory to the path to import from src\n",
    "sys.path.append(os.path.abspath('..'))\n",
    "\n",
    "import numpy as np\n",
    "from src.embeddings.generator import GaitEmbeddingGenerator\n",
    "from src.database.pinecone_db import GaitDatabase"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load the processed pose data\n",
    "poses = np.load('../data/processed/sample_poses.npy')\n",
    "print(f\"Loaded pose data with shape: {poses.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Initialize the embedding generator\n",
    "embedding_generator = GaitEmbeddingGenerator(embedding_dim=256)  # Using a smaller dimension for testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Generate an embedding\n",
    "embedding = embedding_generator.generate_embedding(poses)\n",
    "print(f\"Generated embedding with shape: {embedding.shape}\")\n",
    "print(f\"Embedding norm: {np.linalg.norm(embedding)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Initialize Pinecone client\n",
    "# You'll need to set these values\n",
    "api_key = \"your-pinecone-api-key\"\n",
    "environment = \"your-environment\"\n",
    "index_name = \"gait-analysis\"\n",
    "\n",
    "db = GaitDatabase(api_key, environment, index_name, dimension=256)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Store the embedding\n",
    "metadata = {\n",
    "    \"source_video\": \"sample_walking.mp4\",\n",
    "    \"subject_id\": \"test-subject-001\",\n",
    "    \"recording_date\": \"2023-06-15\"\n",
    "}\n",
    "\n",
    "db.store_gait_embedding(\"sample-001\", embedding, metadata)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Test searching for similar gaits\n",
    "# In a real scenario, you'd use a different embedding as a query\n",
    "results = db.search_similar_gaits(embedding, top_k=5)\n",
    "print(\"Search results:\")\n",
    "print(results)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
EOF

# Create requirements.txt
cat > gait_analysis_app/requirements.txt << 'EOF'
# Core libraries
numpy>=1.20.0
opencv-python>=4.5.0
mediapipe>=0.8.10
pinecone-client>=2.0.0

# Visualization
matplotlib>=3.4.0

# Jupyter notebook
jupyter>=1.0.0
ipykernel>=6.0.0

# Data manipulation
pandas>=1.3.0

# ML utilities (optional)
scikit-learn>=1.0.0
EOF

# Create a basic settings file
cat > gait_analysis_app/config/settings.py << 'EOF'
"""
Configuration settings for the gait analysis application.
"""

# Pinecone settings
PINECONE_API_KEY = "your-api-key-here"  # Replace with your actual API key
PINECONE_ENVIRONMENT = "your-environment"  # Replace with your environment
PINECONE_INDEX_NAME = "gait-analysis"

# MediaPipe settings
MODEL_COMPLEXITY = 2  # 0, 1, or 2 (higher is more accurate but slower)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Embedding settings
EMBEDDING_DIMENSION = 512

# Processing settings
TARGET_FPS = 30  # Target frames per second for processing
PROCESS_EVERY_N_FRAMES = 2  # Process every Nth frame to reduce computation
EOF

# Create a basic README
cat > gait_analysis_app/README.md << 'EOF'
# Gait Analysis Application

This application analyzes human gait patterns from video data using MediaPipe for pose estimation. It extracts pose landmarks, generates embeddings, and stores them in a Pinecone vector database for similarity search.

## Project Structure

- `src/`: Source code
  - `pose/`: Pose extraction using MediaPipe
  - `embeddings/`: Gait feature extraction and embedding generation
  - `database/`: Pinecone database integration
  - `utils/`: Visualization and helper utilities
- `notebooks/`: Jupyter notebooks for exploration and testing
- `data/`: Data storage
- `config/`: Configuration files

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Update Pinecone API key in `config/settings.py`

3. Place sample videos in `data/raw/`

4. Run the notebooks in the `notebooks/` directory to experiment

## Running the Application

```
python app.py
```

## Features

- Pose extraction from video using MediaPipe
- Feature extraction for gait analysis
- Vector embeddings for similarity search
- Storage and retrieval using Pinecone
EOF

# Create a simple app.py
cat > gait_analysis_app/app.py << 'EOF'
"""
Main application entry point.
"""
import os
from src.pose.extractor import PoseExtractor
from src.embeddings.generator import GaitEmbeddingGenerator
from src.database.pinecone_db import GaitDatabase
from config.settings import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION,
    MODEL_COMPLEXITY
)

def process_video(video_path):
    """Process a video file and store the embedding in Pinecone."""
    # Extract the filename to use as ID
    video_filename = os.path.basename(video_path)
    video_id = os.path.splitext(video_filename)[0]
    
    print(f"Processing video: {video_filename}")
    
    # Extract poses
    extractor = PoseExtractor(model_complexity=MODEL_COMPLEXITY)
    poses, fps = extractor.process_video(video_path)
    
    print(f"Extracted {len(poses)} frames of pose data")
    
    # Generate embedding
    generator = GaitEmbeddingGenerator(embedding_dim=EMBEDDING_DIMENSION)
    embedding = generator.generate_embedding(poses)
    
    print(f"Generated embedding with dimension {embedding.shape}")
    
    # Save to Pinecone
    db = GaitDatabase(
        PINECONE_API_KEY, 
        PINECONE_ENVIRONMENT, 
        PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION
    )
    
    metadata = {
        "source_video": video_filename,
        "fps": fps,
        "frame_count": len(poses)
    }
    
    db.store_gait_embedding(video_id, embedding, metadata)
    print(f"Stored embedding with ID: {video_id}")
    
    return video_id

def find_similar_gaits(video_path, top_k=5):
    """Find videos with similar gait patterns."""
    # First process the query video
    query_id = process_video(video_path)
    
    # Extract the embedding from Pinecone
    # Note: In a real application, you would process the video again
    # rather than retrieving it, but this is for demonstration
    
    video_filename = os.path.basename(video_path)
    print(f"Finding videos with similar gait patterns to {video_filename}")
    
    # Generate embedding (repeating the process for simplicity)
    extractor = PoseExtractor(model_complexity=MODEL_COMPLEXITY)
    poses, _ = extractor.process_video(video_path)
    
    generator = GaitEmbeddingGenerator(embedding_dim=EMBEDDING_DIMENSION)
    query_embedding = generator.generate_embedding(poses)
    
    # Search for similar gaits
    db = GaitDatabase(
        PINECONE_API_KEY, 
        PINECONE_ENVIRONMENT, 
        PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION
    )
    
    results = db.search_similar_gaits(query_embedding, top_k=top_k)
    
    print("Similar gait patterns found:")
    for match in results['matches']:
        print(f"ID: {match['id']}, Score: {match['score']:.4f}")
        for key, value in match['metadata'].items():
            print(f"  {key}: {value}")
        print()
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Gait Analysis Application")
    print("------------------------")
    print(
        "Usage examples:\n"
        "1. process_video('data/raw/sample.mp4')\n"
        "2. find_similar_gaits('data/raw/query.mp4', top_k=5)"
    )
EOF

# Make directories and files readable/executable
chmod -R 755 gait_analysis_app

echo "Project structure created successfully in ./gait_analysis_app"
echo "Run 'pip install -r gait_analysis_app/requirements.txt' to install dependencies"