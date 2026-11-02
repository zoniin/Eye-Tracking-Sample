@echo off
REM Optional: build/install dlib on Windows (only needed for `eye_tracker.py --backend dlib`).
REM Requires "Visual Studio Build Tools" (C++ workload) and CMake on PATH.
REM Replaces the old install_deps.bat / install_deps2.bat / install_dlib.bat,
REM which were hard-coded to one person's paths.

cd /d "%~dp0.."

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if exist "%VCVARS%" (
    echo Setting up Visual Studio 2019 build environment...
    call "%VCVARS%" x64
) else (
    echo NOTE: %VCVARS% not found - run this from a "x64 Native Tools Command Prompt" instead.
)

where cmake >nul 2>&1
if errorlevel 1 (
    echo ERROR: cmake not found on PATH. Install it from https://cmake.org/download/ and retry.
    exit /b 1
)

pip install -r requirements.txt
pip install dlib
if errorlevel 1 (
    echo If the build failed with a generator error, try:  set CMAKE_GENERATOR=Visual Studio 16 2019
    exit /b 1
)
echo Done. Download the model with scripts\download_dlib_model.sh (or from http://dlib.net/files/).
