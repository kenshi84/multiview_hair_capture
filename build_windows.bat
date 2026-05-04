@echo off
setlocal enabledelayedexpansion

:: ---- Find Visual Studio with vswhere ----
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found. Is Visual Studio installed?
    exit /b 1
)

for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VS_PATH=%%i"
if not defined VS_PATH (
    echo ERROR: No Visual Studio installation found.
    exit /b 1
)

set "VCVARSALL=%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat"
if not exist "%VCVARSALL%" (
    echo ERROR: vcvarsall.bat not found at %VCVARSALL%
    exit /b 1
)

:: ---- Set up MSVC environment ----
echo Setting up Visual Studio environment...
call "%VCVARSALL%" x64
if errorlevel 1 (
    echo ERROR: vcvarsall.bat failed
    exit /b 1
)

:: ---- Check prerequisites ----
where cmake >nul 2>&1 || (echo ERROR: cmake not found in PATH && exit /b 1)
where ninja >nul 2>&1 || (echo ERROR: ninja not found in PATH. Install via: winget install Ninja-build.Ninja && exit /b 1)
where nvcc  >nul 2>&1 || (echo ERROR: nvcc not found in PATH. Is CUDA Toolkit installed? && exit /b 1)

:: ---- Initialize submodules if needed ----
cd /d "%~dp0"
if not exist "third_party\nanoflann\include\nanoflann.hpp" (
    echo Initializing git submodules...
    git submodule update --init --recursive
    if errorlevel 1 (
        echo ERROR: git submodule update failed
        exit /b 1
    )
)

:: ---- Configure and build ----
if not exist build mkdir build
cd build

echo.
echo Configuring...
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .. 2>&1
if errorlevel 1 (
    echo ERROR: CMake configure failed
    exit /b 1
)

echo.
echo Building...
cmake --build . 2>&1
if errorlevel 1 (
    echo ERROR: Build failed
    exit /b 1
)

echo.
echo Build successful: %cd%\hair_recon.exe
