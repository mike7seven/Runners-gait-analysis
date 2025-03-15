"""
REST API for the Gait Analysis Application.
Exposes the core functionality as HTTP endpoints using FastAPI.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import os
import tempfile
import uuid
import shutil
import glob
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

# Import core functionality
from app import process_video, find_similar_gaits
from src.database.pinecone_db import GaitDatabase
from config.settings import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION
)

# Define response models
class VideoMetadata(BaseModel):
    source_video: str
    fps: float
    frame_count: int
    upload_date: Optional[str] = None
    processed_date: Optional[str] = None

class VideoInfo(BaseModel):
    id: str
    metadata: VideoMetadata

class ProcessVideoResponse(BaseModel):
    status: str
    video_id: str
    message: str

class SimilarityMatch(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]

class FindSimilarResponse(BaseModel):
    status: str
    query_id: str
    matches: List[SimilarityMatch]

class StatsResponse(BaseModel):
    total_videos: int
    total_frames_processed: int
    average_frames_per_video: float
    video_formats: Dict[str, int]

app = FastAPI(
    title="Runners Gait Analysis API",
    description="API for analyzing runner gait patterns from video",
    version="1.0.0",
    docs_url=None,  # Disable default docs to customize
    redoc_url=None  # Disable default redoc to customize
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure upload folder
UPLOAD_FOLDER = 'data/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure raw video folder
RAW_FOLDER = 'data/raw'
if not os.path.exists(RAW_FOLDER):
    os.makedirs(RAW_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db():
    """Get database connection."""
    db = GaitDatabase(
        PINECONE_API_KEY, 
        PINECONE_ENVIRONMENT, 
        PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION
    )
    return db

# Custom OpenAPI and Swagger UI
@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(
        title="Runners Gait Analysis API",
        version="1.0.0",
        description="API for analyzing runner gait patterns from video",
        routes=app.routes,
    )

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Runners Gait Analysis API",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Runners Gait Analysis API - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.post("/api/process-video", response_model=ProcessVideoResponse)
async def api_process_video(video: UploadFile = File(...)):
    """
    Process a video file and store its gait embedding.
    
    Expects a multipart/form-data request with a 'video' file.
    Returns the video ID and processing status.
    """
    if not video:
        raise HTTPException(status_code=400, detail="No video file provided")
    
    if video.filename == '':
        raise HTTPException(status_code=400, detail="No video file selected")
    
    if not allowed_file(video.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    # Generate a unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4()}_{video.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # Save the uploaded file
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    try:
        # Process the video
        video_id = process_video(filepath)
        
        # Also save a copy to the raw folder for retrieval
        raw_path = os.path.join(RAW_FOLDER, f"{video_id}.mp4")
        shutil.copy(filepath, raw_path)
        
        return {
            "status": "success",
            "video_id": video_id,
            "message": "Video processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/find-similar", response_model=FindSimilarResponse)
async def api_find_similar(
    video: UploadFile = File(...),
    top_k: int = Query(5, description="Number of similar videos to return")
):
    """
    Find videos with similar gait patterns.
    
    Expects a multipart/form-data request with a 'video' file.
    Optional query parameter 'top_k' for number of results (default: 5).
    Returns the similar videos with their similarity scores.
    """
    if not video:
        raise HTTPException(status_code=400, detail="No video file provided")
    
    if video.filename == '':
        raise HTTPException(status_code=400, detail="No video file selected")
    
    if not allowed_file(video.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    # Generate a unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4()}_{video.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # Save the uploaded file
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    try:
        # Find similar gaits
        results = find_similar_gaits(filepath, top_k=top_k)
        
        # Clean up the results for JSON serialization
        matches = []
        for match in results['matches']:
            matches.append({
                "id": match['id'],
                "score": match['score'],
                "metadata": match['metadata']
            })
        
        return {
            "status": "success",
            "query_id": os.path.basename(filepath),
            "matches": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/{video_id}")
async def get_video(video_id: str):
    """
    Retrieve a video by ID.
    
    This endpoint returns the video file for the given ID.
    """
    video_path = f"{RAW_FOLDER}/{video_id}.mp4"
    
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    else:
        raise HTTPException(status_code=404, detail="Video not found")

@app.get("/api/videos", response_model=List[VideoInfo])
async def list_videos(db: GaitDatabase = Depends(get_db)):
    """
    List all processed videos.
    
    Returns a list of all videos that have been processed and stored in the database.
    """
    try:
        # This is a placeholder - in a real implementation, you would fetch
        # the list of videos from your database
        # For now, we'll just list the video files in the raw folder
        videos = []
        for video_file in glob.glob(f"{RAW_FOLDER}/*.mp4"):
            video_id = os.path.basename(video_file).split('.')[0]
            
            # Get metadata from database if available
            try:
                # This is a simplified example - you would need to implement
                # proper metadata retrieval from your database
                metadata = {
                    "source_video": f"{video_id}.mp4",
                    "fps": 30.0,  # Placeholder
                    "frame_count": 300,  # Placeholder
                    "upload_date": datetime.now().isoformat()
                }
            except Exception:
                metadata = {
                    "source_video": f"{video_id}.mp4",
                    "fps": 30.0,
                    "frame_count": 0
                }
            
            videos.append({
                "id": video_id,
                "metadata": metadata
            })
        
        return videos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get statistics about the processed videos.
    
    Returns statistics such as total number of videos, total frames processed, etc.
    """
    try:
        # Count videos in the raw folder
        video_files = glob.glob(f"{RAW_FOLDER}/*.mp4")
        total_videos = len(video_files)
        
        # Count video formats
        video_formats = {}
        for ext in ALLOWED_EXTENSIONS:
            count = len(glob.glob(f"{RAW_FOLDER}/*.{ext}"))
            if count > 0:
                video_formats[ext] = count
        
        # Placeholder for total frames and average frames
        # In a real implementation, you would get this from your database
        total_frames = total_videos * 300  # Placeholder
        avg_frames = total_frames / total_videos if total_videos > 0 else 0
        
        return {
            "total_videos": total_videos,
            "total_frames_processed": total_frames,
            "average_frames_per_video": avg_frames,
            "video_formats": video_formats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for the web dashboard
# This allows serving the React frontend from the same server
# app.mount("/", StaticFiles(directory="../web-dashboard/build", html=True), name="static")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 