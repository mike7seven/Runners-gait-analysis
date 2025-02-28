import os
from src.pose.extractor import PoseExtractor
from src.embeddings.generator import GaitEmbeddingGenerator
from src.database.pinecone_db import GaitDatabase
from config.settings import PINECONE_API_KEY, PINECONE_ENVIRONMENT, EMBEDDING_DIMENSION

def process_video_to_database(video_path, metadata=None):
    """
    Process a video and add its embedding to Pinecone.

    Args:
        video_path: Path to the video file
        metadata: Dictionary of metadata about the video

    Returns:
        id: The ID of the stored embedding
    """
    # Extract the filename to use as ID
    video_id = os.path.basename(video_path).split('.')[0]

    print(f"Processing video: {video_path}")

    # 1. Extract poses using MediaPipe
    extractor = PoseExtractor()
    poses, fps = extractor.process_video(video_path)
    print(f"Extracted {len(poses)} frames of pose data")

    # 2. Generate embedding
    generator = GaitEmbeddingGenerator(embedding_dim=256)
    embedding = generator.generate_embedding(poses)
    print(f"Generated embedding of length {len(embedding)}")

    # 3. Store in Pinecone
    if metadata is None:
        metadata = {}

    # Add some basic video metadata
    metadata.update({
        "source_file": os.path.basename(video_path),
        "fps": fps,
        "frame_count": len(poses)
    })

    # # Initialize Pinecone connection
    db = GaitDatabase(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
        index_name="gait-analysis",
        dimension=EMBEDDING_DIMENSION
    )

    # Store the embedding
    db.store_gait_embedding(video_id, embedding, metadata)
    print(f"Stored embedding in Pinecone with ID: {video_id}")

    return video_id

def process_directory_to_database(directory_path, metadata_prefix=None):
    """
    Process all videos in a directory and add them to Pinecone.

    Args:
        directory_path: Path to directory containing videos
        metadata_prefix: Dictionary of metadata to apply to all videos

    Returns:
        ids: List of IDs of the stored embeddings
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mpg']
    ids = []

    # Process each video file
    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(directory_path, filename)

            # Create metadata for this video
            metadata = {}
            if metadata_prefix:
                metadata.update(metadata_prefix)

            # Process the video
            video_id = process_video_to_database(video_path, metadata)
            ids.append(video_id)

    print(f"Processed {len(ids)} videos from {directory_path}")
    return ids

# Example usage
if __name__ == "__main__":
    # Make sure your Pinecone API key is set in config/settings.py

    # Process a single video
    # process_video_to_database("data/raw/sample_walk.mp4", {"gait_type": "normal"})

    # Process a directory of reference "good gait" videos
    process_directory_to_database(
        "data/raw/reference_gaits", 
        metadata_prefix={"category": "reference", "gait_quality": "good"}
    )