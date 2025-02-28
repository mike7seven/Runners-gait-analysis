"""
Configuration settings for the gait analysis application.
"""

# Pinecone settings
PINECONE_API_KEY = "pcsk_75u8m8_L5iBMApTv4Tt8FrbZeypQCbhHsZjLpCibgcNBuzZ1ZfwQLbqmu58uC6XKQyQpj3"  # Replace with your actual API key
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
