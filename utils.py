import cv2
from PIL import Image, ImageDraw
import os
import json
from datetime import datetime
from typing import Dict, List
from config import BOX_COLOR, BOX_WIDTH, SAVE_DIR, DETECTION_OBJECT

def cv2_to_pil(cv2_frame) -> Image.Image:
    rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)

def get_video_name(video_path: str) -> str:
    """Extract video name without extension from path"""
    return os.path.splitext(os.path.basename(video_path))[0]

def create_directory(dir_path: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(dir_path, exist_ok=True)

def convert_to_absolute_coords(obj: Dict, image_width: int, image_height: int) -> Dict[str, int]:
    """Convert relative coordinates to absolute pixels"""
    x_min = int(obj["x_min"] * image_width)
    y_min = int(obj["y_min"] * image_height)
    x_max = int(obj["x_max"] * image_width)
    y_max = int(obj["y_max"] * image_height)
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max
    }

def draw_boxes(image: Image.Image, detection_results: list) -> Image.Image:
    if not detection_results:
        return image
        
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    image_width, image_height = image.size
    
    for obj in detection_results:
        if not all(key in obj for key in ("x_min", "y_min", "x_max", "y_max")):
            continue
            
        coords = convert_to_absolute_coords(obj, image_width, image_height)
        draw.rectangle(
            [(coords["x_min"], coords["y_min"]), (coords["x_max"], coords["y_max"])],
            outline=BOX_COLOR,
            width=BOX_WIDTH
        )
    
    return image_with_boxes

def save_frame(video_path: str, image: Image.Image, frame_num: int) -> str:
    """Save a frame to disk with proper path handling"""
    video_name = get_video_name(video_path)
    save_dir = os.path.join(SAVE_DIR, video_name)
    create_directory(save_dir)
    
    save_path = os.path.join(save_dir, f"{video_name}_{frame_num}.png")
    image.save(save_path)
    return save_path

def create_coco_annotation_base() -> Dict:
    """Create base COCO format annotation structure"""
    return {
        "info": {
            "description": "Video Object Detection Dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "----",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": DETECTION_OBJECT, "supercategory": "object"}]
    }

def create_image_info(frame_num: int, image: Image.Image) -> Dict:
    """Create COCO format image info"""
    width, height = image.size
    return {
        "id": frame_num,
        "file_name": f"{frame_num:06d}.png",
        "width": width,
        "height": height
    }

def create_annotation(annotation_id: int, frame_num: int, obj: Dict, image_width: int, image_height: int) -> Dict:
    """Create COCO format annotation for a detection result"""
    coords = convert_to_absolute_coords(obj, image_width, image_height)
    width = coords["x_max"] - coords["x_min"]
    height = coords["y_max"] - coords["y_min"]
    
    return {
        "id": annotation_id,
        "image_id": frame_num,
        "category_id": 1,
        "bbox": [coords["x_min"], coords["y_min"], width, height],
        "area": width * height,
        "iscrowd": 0
    }

def save_coco_annotations(annotations: Dict, video_path: str) -> None:
    video_name = get_video_name(video_path)
    annotation_dir = os.path.join(os.path.dirname(video_path), 'annotations')
    create_directory(annotation_dir)
    
    annotation_path = os.path.join(annotation_dir, f'{video_name}_annotations.json')
    with open(annotation_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"Saved COCO format annotations to: {annotation_path}") 