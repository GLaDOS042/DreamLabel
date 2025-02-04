# YouTube Video Object Detection Tool

A tool that downloads YouTube videos and performs object detection. This tool uses the Moondream2 model to identify and mark specific objects in videos.

## Quick Start for Windows Users

Simply double-click the `start.bat` file in the project directory to run the program. The batch file will:
- Automatically activate the virtual environment
- Install all required dependencies
- Run the program
- Keep the window open to show any error messages
- Close automatically when processing is complete

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
git clone [https://github.com/GLaDOS042/DreamLabel.git]
```

2. Install PyTorch dependencies:
- Visit https://pytorch.org/get-started/locally/ to get the correct PyTorch installation command for your system
- Select your preferences (stable, OS, package manager, CUDA version etc.)
- Run the generated command (e.g., `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)

3. Install other dependencies:

Windows users:
- Simply run `start.bat` (Recommended)

Linux/Mac users:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --pre "yt-dlp[default]"
sudo apt-get install ffmpeg
sudo apt-get install vips
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
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate  # Windows
   python main.py
   ```
## Notes

- Ensure sufficient disk space for storing downloaded videos
- Stable internet connection required for YouTube video downloads
- Processing time depends on video length and hardware performance

## License

[License details]

## Contributing

Issues and Pull Requests are welcome! 