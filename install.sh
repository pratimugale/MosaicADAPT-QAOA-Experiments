#!/bin/bash
set -e

# Configuration
REPO_URL="https://github.com/pratimugale/TetrisADAPT.jl.git"
DIR_NAME="TetrisADAPT.jl"
VENV_DIR="venv"

echo "Starting installation for mis-tetris-adapt..."

# 1. Clone TetrisADAPT.jl if it doesn't exist
if [ ! -d "$DIR_NAME" ]; then
    echo "Cloning $REPO_URL..."
    git clone "$REPO_URL"
else
    echo "$DIR_NAME already exists. Skipping clone."
fi

# 2. Setup Python Virtual Environment
echo "Setting up Python virtual environment..."
python3 -m venv "$VENV_DIR"

# Activate venv for the script execution
source "$VENV_DIR/bin/activate"

# 3. Install Python Dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Build Julia Dependencies (PyCall)
echo "Building Julia dependencies with PyCall linked to venv..."
# Explicitly set PYTHON to the absolute path of the venv python
export PYTHON="$(pwd)/$VENV_DIR/bin/python3"
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.develop(path="TetrisADAPT.jl"); Pkg.build("PyCall")'

echo "Installation script finished successfully."
