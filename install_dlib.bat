@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set PATH=C:\Users\jarre\cmake-4.2.3-windows-x86_64\bin;%PATH%
cd C:\Users\jarre\Eye-Tracking-Sample
pip install -r requirements.txt
