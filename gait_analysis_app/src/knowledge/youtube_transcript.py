from youtube_transcript_api import YouTubeTranscriptApi
from typing import Dict, List, Optional
import os

class YoutubeTranscriptDownloader:
    def __init__(self, language: str = "en"):
        self.language = language

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract the video ID from a YouTube URL"""
        if "v=" in url:
            return url.split("v=")[1][:11]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1][:11]
        return None
    
    def get_transcript(self, video_id: str) -> Optional[List[Dict]]:
        """Download YouTube transcript"""
        if "youtube.com" in video_id or "youtu.be" in video_id:
            video_id = self.extract_video_id(video_id)

        if not video_id:
            print("Invalid video ID or URL")
            return None

        print(f"Downloading transcript for video ID: {video_id}")
        
        try:
            return YouTubeTranscriptApi.get_transcript(video_id, languages=[self.language])
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
        
    def save_transcript(self, transcript: List[Dict], filename: str) -> bool:
        """Save transcript to a file"""
        if not transcript:
            return False
        
        os.makedirs("../../data/transcripts", exist_ok=True)

        filename = f"../../data/transcripts/{filename}.txt"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in transcript:
                    f.write(f"{entry['text']}\n")
            return True
        except Exception as e:
            print(f"Error saving transcript: {str(e)}")
            return False
    
def main(video_url, print_transcript = False):
    downloader = YoutubeTranscriptDownloader()

    # get transcript
    transcript = downloader.get_transcript(video_url)

    if transcript:
        # save transcript
        # downloader.save_transcript(transcript, video_url)
        video_id = downloader.extract_video_id(video_url)

        if downloader.save_transcript(transcript, video_id):
            print(f"Transcript saved successfully to ../data/transcripts/{video_id}.txt")
            # print transcript if true
            if print_transcript:
                for entry in transcript:
                    print(f"{entry['text']}")
        else:
            print("Failed to save transcript")
    
    else:
        print("Failed to get transcript")

    return None

if __name__ == "__main__":
    video_id = "https://www.youtube.com/watch?v=_kGESn8ArrU"
    transcript = main(video_id, print_transcript=True)