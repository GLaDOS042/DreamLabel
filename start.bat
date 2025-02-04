@echo off
echo Creating virtual environment...
python -m venv venv

echo Upgrading pip in the virtual environment...
venv\Scripts\pip.exe install --upgrade pip

echo Installing packages from requirements.txt...
venv\Scripts\pip.exe install -r requirements.txt

echo Installing pre-release version of yt-dlp[default]...
venv\Scripts\pip.exe install --pre "yt-dlp[default]"


echo Downloading vips
curl -O -L https://github.com/libvips/build-win64-mxe/releases/download/v8.16.0/vips-dev-w64-all-8.16.0.zip

echo Unzipping vips
tar -xf vips-dev-w64-all-8.16.0.zip

echo Downloading ffmpeg
curl -O -L https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.7z

echo Unzipping ffmpeg
tar -xf ffmpeg-release-essentials.7z

echo Removing zip files
del vips-dev-w64-all-8.16.0.zip
del ffmpeg-release-essentials.7z

echo Installation complete

echo Please visit https://pytorch.org/get-started/locally/ to install PyTorch with CUDA support
echo Please activate the virtual environment by running "venv\Scripts\activate"
pause

