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
