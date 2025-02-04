# YouTube Video Object Detection Tool

A tool that downloads YouTube videos and performs object detection. This tool uses the Moondream2 model to identify and mark specific objects in videos.

## Features

- YouTube video downloading
- Object detection using Moondream2 AI model
- Automatic bounding box drawing around detected objects
- Batch processing support
- Customizable detection targets and visual effects

## System Requirements

- Python 3.x
- CUDA-enabled GPU (for model inference)
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The following settings can be adjusted in `config.py`:

- `VIDEO_PATH`: Video storage path
- `YOUTUBE_URL`: List of YouTube video URLs to process
- `SAVE_DIR`: Directory for processed videos
- `BATCH_SIZE`: Batch processing size
- `MODEL_NAME`: AI model name to use
- `DEVICE`: Computing device (cuda/cpu)
- `DETECTION_OBJECT`: Target object to detect
- `BOX_COLOR`: Bounding box color
- `BOX_WIDTH`: Bounding box width

## Usage

1. Set desired YouTube video URLs in `config.py`
2. Run the program using one of these methods:

   ### Using Python directly:
   ```bash
   python main.py
   ```

   ### Using Windows Batch File (Recommended for Windows users):
   Simply double-click the `start.bat` file in the project directory. This batch file will:
   - Activate the virtual environment automatically
   - Run the program
   - Keep the window open to show any error messages
   - Close automatically when processing is complete

## Notes

- Ensure sufficient disk space for storing downloaded videos
- Stable internet connection required for YouTube video downloads
- Processing time depends on video length and hardware performance

## License

[License details]

## Contributing

Issues and Pull Requests are welcome! 