# Runners Gait Analysis API

This API provides endpoints for analyzing runner gait patterns from video data. It uses FastAPI to expose the core functionality of the gait analysis application.

## API Endpoints

### Health Check
```
GET /api/health
```
Returns the status of the API.

### Process Video
```
POST /api/process-video
```
Processes a video file and stores its gait embedding in the database.

**Request:**
- Multipart form with a `video` file (supported formats: mp4, mov, avi)

**Response:**
```json
{
  "status": "success",
  "video_id": "unique_video_id",
  "message": "Video processed successfully"
}
```

### Find Similar Gaits
```
POST /api/find-similar?top_k=5
```
Finds videos with similar gait patterns to the uploaded video.

**Request:**
- Multipart form with a `video` file (supported formats: mp4, mov, avi)
- Query parameter `top_k` (optional, default: 5): Number of similar videos to return

**Response:**
```json
{
  "status": "success",
  "query_id": "filename",
  "matches": [
    {
      "id": "video_id",
      "score": 0.95,
      "metadata": {
        "source_video": "original_filename.mp4",
        "fps": 30,
        "frame_count": 300
      }
    },
    ...
  ]
}
```

### Get Video
```
GET /api/videos/{video_id}
```
Retrieves a video by ID.

**Response:**
- Video file (media type: video/mp4)

### List Videos
```
GET /api/videos
```
Lists all processed videos in the database.

**Response:**
```json
[
  {
    "id": "video_id_1",
    "metadata": {
      "source_video": "original_filename_1.mp4",
      "fps": 30,
      "frame_count": 300,
      "upload_date": "2023-03-15T12:34:56.789Z"
    }
  },
  {
    "id": "video_id_2",
    "metadata": {
      "source_video": "original_filename_2.mp4",
      "fps": 30,
      "frame_count": 250,
      "upload_date": "2023-03-16T10:11:12.345Z"
    }
  },
  ...
]
```

### Get Statistics
```
GET /api/stats
```
Returns statistics about the processed videos.

**Response:**
```json
{
  "total_videos": 10,
  "total_frames_processed": 3000,
  "average_frames_per_video": 300.0,
  "video_formats": {
    "mp4": 8,
    "mov": 2
  }
}
```

## Running the API

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the API server:
```bash
python api.py
```

The API will be available at http://localhost:8000.

## API Documentation with Swagger UI

FastAPI automatically generates interactive API documentation using Swagger UI:

- **Swagger UI**: http://localhost:8000/docs
  - Interactive documentation that allows you to:
  - Explore all available endpoints
  - See request/response schemas
  - Test endpoints directly from the browser
  - Understand parameter requirements

- **ReDoc**: http://localhost:8000/redoc
  - Alternative documentation view with a different layout

### Using Swagger UI

1. Navigate to http://localhost:8000/docs in your browser
2. Click on any endpoint to expand it
3. Click "Try it out" to test the endpoint
4. Fill in the required parameters
5. Click "Execute" to send the request
6. View the response below

This makes it easy to test and understand the API without writing any code.

## Integration with React Frontend

To connect your React frontend to this API:

1. Set up your React app to make API calls to the endpoints listed above
2. Use the proxy setting in your package.json to avoid CORS issues during development:
   ```json
   "proxy": "http://localhost:8000"
   ```
3. For production, you can either:
   - Deploy the API and frontend separately and configure CORS appropriately
   - Build the React app and serve it from the FastAPI server by uncommenting the static files mount in api.py 