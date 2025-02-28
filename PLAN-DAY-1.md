# Day 1 Tasks: Core ML Pipeline Verification

## Morning Tasks (3-4 hours)

### 1. Verify Pinecone Connection (1 hour)

First, let's make sure your Pinecone integration works correctly:

1. Update `config/settings.py` with your current Pinecone environment:
   ```python
   # Update this value with your actual environment (e.g., "gcp-starter")
   PINECONE_ENVIRONMENT = "gcp-starter"  # or the environment from your Pinecone dashboard
   ```

2. Create a simple test script to verify Pinecone connection:
   ```python
   # test_pinecone.py
   from src.database.pinecone_db import GaitDatabase
   from config.settings import PINECONE_API_KEY, PINECONE_ENVIRONMENT
   
   def test_pinecone_connection():
       print("Testing Pinecone connection...")
       try:
           db = GaitDatabase(
               api_key=PINECONE_API_KEY,
               environment=PINECONE_ENVIRONMENT,
               index_name="gait-analysis",
               dimension=256
           )
           print("✅ Successfully connected to Pinecone!")
           return True
       except Exception as e:
           print(f"❌ Failed to connect to Pinecone: {e}")
           return False
   
   if __name__ == "__main__":
       test_pinecone_connection()
   ```

3. Run the test script:
   ```bash
   cd gait_analysis_app
   python test_pinecone.py
   ```

### 2. Test MPG Video Processing (1-2 hours)

Verify that your code can process the `.mpg` videos in your dataset:

1. Check if `.mpg` is in the allowed extensions in `populate_database.py`:
   ```python
   # In populate_database.py, find this line:
   video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mpg']
   # Ensure '.mpg' is in the list
   ```

2. Create a test script to process a single MPG file:
   ```python
   # test_mpg_processing.py
   import os
   from src.pose.extractor import PoseExtractor
   
   def test_mpg_processing(video_path):
       print(f"Testing MPG processing with: {video_path}")
       
       if not os.path.exists(video_path):
           print(f"❌ Video file not found: {video_path}")
           return False
       
       try:
           extractor = PoseExtractor()
           poses, fps = extractor.process_video(video_path)
           
           print(f"✅ Successfully processed video!")
           print(f"   - Extracted {len(poses)} frames of pose data")
           print(f"   - Video FPS: {fps}")
           
           if len(poses) == 0:
               print("⚠️ Warning: No poses were detected!")
           
           return len(poses) > 0
       except Exception as e:
           print(f"❌ Failed to process video: {e}")
           return False
   
   if __name__ == "__main__":
       # Use one of your MPG files from the reference_gaits folder
       video_path = "data/raw/reference_gaits/16_35.mpg"
       test_mpg_processing(video_path)
   ```

3. Run the test script:
   ```bash
   cd gait_analysis_app
   python test_mpg_processing.py
   ```

### 3. Set Up Cursor IDE for Team Development (1 hour)

1. Create the `.cursor` folder with the settings files provided earlier:
   - `.cursor/settings.json`
   - `.cursor/extensions.json`
   - `.cursor/tasks.json`

2. Create your team's Git branches:
   ```bash
   git checkout -b develop
   git push -u origin develop
   
   # Create feature branches as needed
   git checkout -b feature/python-fastapi
   ```

## Afternoon Tasks (3-4 hours)

### 4. Create FastAPI Integration (2 hours)

1. Create a new file `fastapi_server.py` in the `gait_analysis_app` directory with the code from the earlier template

2. Add FastAPI to your requirements.txt:
   ```
   # Add these to requirements.txt
   fastapi>=0.68.0
   python-multipart>=0.0.5
   uvicorn>=0.15.0
   ```

3. Install the new dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Test the FastAPI server:
   ```bash
   cd gait_analysis_app
   python -m uvicorn fastapi_server:app --reload --port 8000
   ```
   
5. Open http://localhost:8000/docs to see the Swagger documentation

### 5. Implement Basic Gait Metrics (2 hours)

1. Create a new file `src/analysis/metrics.py` with the gait metrics code from the earlier template

2. Create a simple test script for the metrics:
   ```python
   # test_gait_metrics.py
   import numpy as np
   from src.pose.extractor import PoseExtractor
   from src.analysis.metrics import calculate_stride_length, calculate_cadence, calculate_symmetry
   
   def test_gait_metrics(video_path):
       print(f"Testing gait metrics with: {video_path}")
       
       # Extract poses
       extractor = PoseExtractor()
       poses, fps = extractor.process_video(video_path)
       
       if len(poses) == 0:
           print("❌ No poses detected in video")
           return
       
       # Calculate metrics
       stride_length = calculate_stride_length(poses)
       cadence = calculate_cadence(poses, fps)
       symmetry = calculate_symmetry(poses)
       
       print("Gait Metrics Results:")
       print(f"Stride Length: {stride_length:.2f} (relative units)")
       print(f"Cadence: {cadence:.2f} steps/minute")
       print(f"Symmetry Score: {symmetry:.2f}%")
   
   if __name__ == "__main__":
       # Use one of your MPG files
       video_path = "data/raw/reference_gaits/16_35.mpg"
       test_gait_metrics(video_path)
   ```

3. Run the test script:
   ```bash
   cd gait_analysis_app
   python test_gait_metrics.py
   ```

## End of Day 1 Checklist

- [ ] Pinecone connection verified 
- [ ] MPG video processing tested
- [ ] FastAPI server created and running
- [ ] Basic gait metrics implemented
- [ ] Cursor IDE configured for team development
- [ ] Git branches created for team workflow

## Resources for Python Learning

If you encounter issues with any Python concepts, use Cursor's AI:

1. Select the code you don't understand
2. Press `Cmd+K` (Mac) or `Ctrl+K` (Windows)
3. Ask "Explain this Python code" or "How does this work?"