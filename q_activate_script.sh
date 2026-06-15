#!/bin/bash

# Load the user's bashrc configurations
source ~/.bashrc

# Activate the conda environment
conda activate diffuse

# Launch JupyterLab
python -m jupyterlab --ip 0.0.0.0
