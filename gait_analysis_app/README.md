# Gait Analysis Application

This application analyzes human gait patterns from video data using MediaPipe for pose estimation. It extracts pose landmarks, generates embeddings, and stores them in a Pinecone vector database for similarity search.

## System Architecture

![System Architecture](../docs/Architecture.svg)

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
