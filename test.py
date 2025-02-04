import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM
import yt_dlp  # Import yt-dlp for downloading YouTube videos
import os

temp_path = r'C:\Users\markc\Dev\Dream\vips-dev-8.16\bin'

os.environ['PATH'] = temp_path + os.pathsep + os.environ.get('PATH', '')

print(os.environ['PATH'])

# ----------------------------
# Download YouTube video and set video_path
# ----------------------------
youtube_url = "https://www.youtube.com/watch?v=QkWtB4iUcGM"  # Replace with your YouTube video URL
ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Use the specified format string
    'outtmpl': 'downloaded_video.%(ext)s',  # Output filename template
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(youtube_url, download=True)
    video_path = ydl.prepare_filename(info_dict)
print(f"Video downloaded and saved at: {video_path}")

# ----------------------------
# Setup output directory
# ----------------------------
save_dir = "box"
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# Load the model and send it to GPU if available.
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
).to("cuda")

# ----------------------------
# Open the video file.
# ----------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

# ----------------------------
# Helper function: Convert OpenCV BGR frame to PIL Image.
# ----------------------------
def cv2_to_pil(cv2_frame):
    rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)

# ----------------------------
# Process the video in batches of 20 frames.
# ----------------------------
batch_size = 20
frame_batch = []  # To store (frame_number, frame)
global_frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        # Process any remaining frames in the last batch.
        if frame_batch:
            for frame_num, cv_frame in frame_batch:
                pil_image = cv2_to_pil(cv_frame)
                image_width, image_height = pil_image.size

                # Run object detection for "robot".
                detection_results = model.detect(pil_image, "robot")["objects"]
                if detection_results:
                    # Create a copy to draw boxes.
                    image_with_boxes = pil_image.copy()
                    draw = ImageDraw.Draw(image_with_boxes)
                    for obj in detection_results:
                        if all(key in obj for key in ("x_min", "y_min", "x_max", "y_max")):
                            x_min = int(obj["x_min"] * image_width)
                            y_min = int(obj["y_min"] * image_height)
                            x_max = int(obj["x_max"] * image_width)
                            y_max = int(obj["y_max"] * image_height)
                            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)
                        else:
                            print(f"No bounding box info found for object in frame {frame_num}: {obj}")
                    # Save the processed frame.
                    save_path = os.path.join(save_dir, f"detected_frame_{frame_num:06d}.png")
                    image_with_boxes.save(save_path)
                    print(f"Saved processed frame {frame_num} with bounding boxes to {save_path}")
            frame_batch = []
        break

    global_frame_counter += 1
    frame_batch.append((global_frame_counter, frame))

    if len(frame_batch) == batch_size:
        for frame_num, cv_frame in frame_batch:
            pil_image = cv2_to_pil(cv_frame)
            image_width, image_height = pil_image.size

            # Run object detection for "robot".
            detection_results = model.detect(pil_image, "robot")["objects"]
            if detection_results:
                # Create a copy to draw boxes.
                image_with_boxes = pil_image.copy()
                draw = ImageDraw.Draw(image_with_boxes)
                for obj in detection_results:
                    if all(key in obj for key in ("x_min", "y_min", "x_max", "y_max")):
                        x_min = int(obj["x_min"] * image_width)
                        y_min = int(obj["y_min"] * image_height)
                        x_max = int(obj["x_max"] * image_width)
                        y_max = int(obj["y_max"] * image_height)
                        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)
                    else:
                        print(f"No bounding box info found for object in frame {frame_num}: {obj}")

                # Save the processed frame.
                save_path = os.path.join(save_dir, f"detected_frame_{frame_num:06d}.png")
                image_with_boxes.save(save_path)
                print(f"Saved processed frame {frame_num} with bounding boxes to {save_path}")

        # Clear the batch.
        frame_batch = []

cap.release()
print("Finished processing video.")
