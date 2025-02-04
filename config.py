import os

# Video settings
VIDEO_PATH = "Video"
YOUTUBE_URL = ["https://www.youtube.com/watch?v=gLCBXi9d_Ao"]
SAVE_DIR = "box"
BATCH_SIZE = 20

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Model settings
MODEL_NAME = "vikhyatk/moondream2"
MODEL_REVISION = "2025-01-09"
DEVICE = "cuda"

# Detection settings
DETECTION_OBJECT = "robot"
BOX_COLOR = "red"
BOX_WIDTH = 3 

SAVE_FRAME = True
SCENE_THRESHOLD = 0.3