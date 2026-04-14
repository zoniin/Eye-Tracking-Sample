@echo off
echo Setting up Visual Studio environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

echo Adding CMake to PATH...
set "PATH=C:\Users\jarre\cmake-4.2.3-windows-x86_64\bin;%PATH%"

echo Verifying CMake...
cmake --version

echo Installing Python dependencies...
cd /d "C:\Users\jarre\Eye-Tracking-Sample"
pip install dlib opencv-python numpy imutils

echo Done!
