import cv2
from tqdm import tqdm
from typing import Generator, Tuple, List
import subprocess
import os
from config import *
from utils import *

class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.total_frames = 0
        self.keyframes_dir = os.path.join(os.path.dirname(video_path), 'keyframes', os.path.splitext(os.path.basename(video_path))[0])
        
    def extract_keyframes(self, scene_threshold: float = 0.3) -> bool:
        """Extract keyframes from video using FFmpeg with scene detection
        
        Args:
            scene_threshold (float): Threshold for scene change detection (0.0 to 1.0).
                                   Lower values extract more frames. Default: 0.3
        """
        try:
            # Create directory for saving keyframes
            os.makedirs(self.keyframes_dir, exist_ok=True)
            
            # Build FFmpeg command to extract keyframes using both I-frames and scene detection
            cmd = [
                'ffmpeg', '-i', self.video_path,
                '-vf', f"select='eq(pict_type,I) + gt(scene,{scene_threshold})'",
                '-vsync', '0',
                '-frame_pts', '1',
                '-q:v', '2',  # High quality JPEG
                os.path.join(self.keyframes_dir, '%d.jpg')
            ]
            
            # Execute FFmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg keyframe extraction failed: {result.stderr}")
                return False
                
            # Check if any frames were extracted
            keyframes = [f for f in os.listdir(self.keyframes_dir) if f.endswith('.jpg')]
            if not keyframes:
                print("No keyframes were extracted. Trying with lower scene threshold...")
                # Retry with lower threshold if no frames were extracted
                return self.extract_keyframes(scene_threshold=max(0.1, scene_threshold - 0.1))
                
            print(f"Successfully extracted {len(keyframes)} keyframes to: {self.keyframes_dir}")
            return True
            
        except Exception as e:
            print(f"Error occurred while extracting keyframes: {str(e)}")
            return False
            
    def process_keyframes(self, model_handler) -> None:
        """Process extracted keyframes"""
        if not os.path.exists(self.keyframes_dir):
            print("Keyframes directory does not exist")
            return
            
        keyframes = [f for f in os.listdir(self.keyframes_dir) if f.endswith('.jpg')]
        
        for keyframe in tqdm(keyframes, desc="Processing keyframes"):
            frame_path = os.path.join(self.keyframes_dir, keyframe)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame_num = int(keyframe.split('_')[1].split('.')[0])
                self.process_frame(frame_num, frame, model_handler)
        
    def open_video(self) -> bool:

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Cannot open video file: {self.video_path}")
            return False
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {self.total_frames}")
        return True
        
    def close_video(self):

        if self.cap:
            self.cap.release()
            
    def read_frames(self) -> Generator[Tuple[int, List], None, None]:

        if not self.cap:
            return
            
        frame_batch = []
        frame_counter = 0
        
        with tqdm(total=self.total_frames, desc="Processing frames", unit="frame") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    if frame_batch:
                        yield frame_counter, frame_batch
                    break
                    
                frame_counter += 1
                frame_batch.append((frame_counter, frame))
                pbar.update(1)
                
                if len(frame_batch) == BATCH_SIZE:
                    yield frame_counter, frame_batch
                    frame_batch = []
                    
    def process_frame(self, frame_num: int, frame, model_handler) -> None:

        pil_image = cv2_to_pil(frame)
        detection_results = model_handler.detect_objects(pil_image, DETECTION_OBJECT)
        
        if not detection_results:
            return
            
        image_with_boxes = draw_boxes(pil_image, detection_results)
        save_path = save_frame(self.video_path, image_with_boxes, frame_num)
        # print(f"Saved processed frame {frame_num} to {save_path}")