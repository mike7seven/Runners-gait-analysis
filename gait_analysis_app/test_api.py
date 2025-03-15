"""
Test script for the Gait Analysis API.
This script tests the API endpoints by making requests to the running API server.
"""
import requests
import os
import sys
import time
import json

# API base URL
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    response = requests.get(f"{API_BASE_URL}/api/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_process_video(video_path):
    """Test the process-video endpoint."""
    print(f"Testing process-video endpoint with {video_path}...")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return None
    
    with open(video_path, "rb") as video_file:
        files = {"video": (os.path.basename(video_path), video_file, "video/mp4")}
        response = requests.post(f"{API_BASE_URL}/api/process-video", files=files)
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    
    if response.status_code == 200:
        return response.json().get("video_id")
    return None

def test_find_similar(video_path, top_k=3):
    """Test the find-similar endpoint."""
    print(f"Testing find-similar endpoint with {video_path}...")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return
    
    with open(video_path, "rb") as video_file:
        files = {"video": (os.path.basename(video_path), video_file, "video/mp4")}
        response = requests.post(
            f"{API_BASE_URL}/api/find-similar?top_k={top_k}", 
            files=files
        )
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_list_videos():
    """Test the list videos endpoint."""
    print("Testing list videos endpoint...")
    response = requests.get(f"{API_BASE_URL}/api/videos")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_get_stats():
    """Test the stats endpoint."""
    print("Testing stats endpoint...")
    response = requests.get(f"{API_BASE_URL}/api/stats")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_get_video(video_id):
    """Test the get video endpoint."""
    print(f"Testing get video endpoint with ID {video_id}...")
    response = requests.get(f"{API_BASE_URL}/api/videos/{video_id}")
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print(f"Successfully retrieved video. Content type: {response.headers.get('Content-Type')}")
        print(f"Content length: {len(response.content)} bytes")
    else:
        print(f"Response: {response.text}")
    print()

def main():
    """Run the API tests."""
    # Check if the API is running
    try:
        test_health()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the API server is running at http://localhost:8000")
        sys.exit(1)
    
    # Test the stats endpoint
    test_get_stats()
    
    # Test the list videos endpoint
    test_list_videos()
    
    # Check if a video path was provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
        # Test the process-video endpoint
        video_id = test_process_video(video_path)
        
        if video_id:
            # Wait a moment for processing to complete
            time.sleep(2)
            
            # Test the get video endpoint
            test_get_video(video_id)
            
            # Test the find-similar endpoint
            test_find_similar(video_path)
    else:
        print("No video path provided. Skipping video processing tests.")
        print("Usage: python test_api.py <path_to_video_file>")

if __name__ == "__main__":
    main() 