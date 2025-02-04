from config import VIDEO_PATH, YOUTUBE_URL, DETECTION_OBJECT
from video_downloader import download_video
from video_processor import VideoProcessor
from model_handler import ModelHandler
import os

if os.name == 'nt':
    vips_path = r'vips-dev-8.16\bin' 
    ffmpeg_path = r'ffmpeg-7.1-essentials_build\bin'
    current_path = os.path.dirname(os.path.abspath(__file__))
    vips_path = os.path.join(current_path, vips_path)
    ffmpeg_path = os.path.join(current_path, ffmpeg_path)

    os.environ['PATH'] = vips_path + os.pathsep + ffmpeg_path + os.pathsep + os.environ.get('PATH', '')


def process_single_video(video_path: str, model_handler: ModelHandler):
    # Initialize video processor
    processor = VideoProcessor(video_path)
    
    # Extract keyframes
    if not processor.extract_keyframes():
        print(f"Failed to extract keyframes from video: {video_path}")
        return
        
    # Process keyframes
    processor.process_keyframes(model_handler)
    
    # Save COCO format annotations
    processor.save_coco_annotations()
    
    print(f"Video keyframe processing completed: {video_path}")

def main():
    # Download video if needed
    download_video(YOUTUBE_URL, VIDEO_PATH)
    
    # Get the directory path from VIDEO_PATH
    video_dir = os.path.dirname(VIDEO_PATH) if os.path.dirname(VIDEO_PATH) else "."
    
    # Initialize model
    model_handler = ModelHandler()
    model_handler.load_model()
    
    # Process all video files in the directory
    for filename in os.listdir(video_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_dir, filename)
            print(f"\nProcessing video: {video_path}")
            process_single_video(video_path, model_handler)

    print("\nAll videos have been processed.")

if __name__ == "__main__":
    main() 