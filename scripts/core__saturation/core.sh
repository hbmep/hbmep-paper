#! /bin/bash
REPO="$HOME/repos/hbmep-paper"
PYTHON_ENV="$REPO/.venv/bin/activate"

# Change to the directory where the script is located
cd $REPO/notebooks/simulations/

# Activate the virtual environment
source $PYTHON_ENV

# Run the script
python -m core__saturation $1 $2
