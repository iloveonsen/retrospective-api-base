#!/bin/bash
# Optionally include NVIDIA's entrypoint logic here, if needed

# Activate the Conda environment
source /miniconda3/etc/profile.d/conda.sh
conda activate fastapi

# Execute the passed command
exec "$@"