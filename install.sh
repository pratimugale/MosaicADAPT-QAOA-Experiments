#!/bin/bash
set -e

REPO_URL="https://github.com/pratimugale/TetrisADAPT.jl"
DIR_NAME="TetrisADAPT.jl"

# Clean up previous install if it exists to ensure a fresh clone
if [ -d "$DIR_NAME" ]; then
    echo "Directory $DIR_NAME already exists. Removing it..."
    rm -rf "$DIR_NAME"
fi

echo "Cloning $REPO_URL..."
git clone "$REPO_URL"

cd "$DIR_NAME"

echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Export JULIA_PYTHON to ensure PyCall uses the venv python
export JULIA_PYTHON="$(pwd)/venv/bin/python"
echo "Set JULIA_PYTHON to $JULIA_PYTHON"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Running make install..."
make install

echo "Running make install-kamis..."
make install-kamis

cd TetrisADAPT.jl/external/KaMIS && ./compile_withcmake.sh

echo "Running make smoke..."
cd TetrisADAPT.jl && make smoke

# Check if smoke-kamis target exists before running
if grep -q "smoke-kamis:" Makefile; then
    echo "Running make smoke-kamis..."
    cd TetrisADAPT.jl && make smoke-kamis
else
    echo "Warning: Make target 'smoke-kamis' not found in Makefile. Installation partially confirmed (smoke passed, smoke-kamis missing)."
fi

echo "Installation Script Finished"
