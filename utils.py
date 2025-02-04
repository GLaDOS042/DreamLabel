import cv2
from PIL import Image, ImageDraw
import os
from config import BOX_COLOR, BOX_WIDTH, SAVE_DIR

def cv2_to_pil(cv2_frame) -> Image.Image:

    rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)

def draw_boxes(image: Image.Image, detection_results: list) -> Image.Image:

    if not detection_results:
        return image
        
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    image_width, image_height = image.size
    
    for obj in detection_results:
        if not all(key in obj for key in ("x_min", "y_min", "x_max", "y_max")):
            continue
            
        x_min = int(obj["x_min"] * image_width)
        y_min = int(obj["y_min"] * image_height)
        x_max = int(obj["x_max"] * image_width)
        y_max = int(obj["y_max"] * image_height)
        
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline=BOX_COLOR,
            width=BOX_WIDTH
        )
    
    return image_with_boxes

def save_frame(video_path: str, image: Image.Image, frame_num: int) -> str:
    """Save a frame to disk with proper path handling
    
    Args:
        video_path: Full path to the source video file
        image: PIL Image to save
        frame_num: Frame number for the filename
    """
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create save directory if it doesn't exist
    save_dir = os.path.join(SAVE_DIR, video_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create save path with sanitized filename
    save_path = os.path.join(save_dir, f"{frame_num:06d}.png")
    
    # Save the image
    image.save(save_path)
    return save_path 