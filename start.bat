@echo off
echo Creating virtual environment...
python -m venv venv

echo Upgrading pip in the virtual environment...
venv\Scripts\pip.exe install --upgrade pip

echo Installing packages from requirements.txt...
venv\Scripts\pip.exe install -r requirements.txt

echo Installing pre-release version of yt-dlp[default]...
venv\Scripts\pip.exe install --pre "yt-dlp[default]"

echo downloading vips
curl -O -L https://github.com/libvips/build-win64-mxe/releases/download/v8.16.0/vips-dev-w64-all-8.16.0.zip

echo unzipping vips
tar -xf vips-dev-w64-all-8.16.0.zip

echo removing zip file
del vips-dev-w64-all-8.16.0.zip

echo Installation complete
pause
