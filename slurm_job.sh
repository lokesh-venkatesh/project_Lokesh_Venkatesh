#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=mitoNN
#SBATCH --output=logs/%J.out
#SBATCH --error=logs/%J.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lokesh.venkatesh@students.iiserpune.ac.in

# Load conda setup script (IMPORTANT!)
source /home/madhu/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate mito_env

# Define paths
SCRIPT_PATH="$1"
OUTPUT_DIR="/storage/madhu/lokesh/project_Lokesh_Venkatesh"
mkdir -p "$OUTPUT_DIR"

# Extract script filename (not used but okay to keep)
SCRIPT_NAME=$(basename "$SCRIPT_PATH")

# Run the script with args
python "$SCRIPT_PATH" --output_dir "$OUTPUT_DIR" --config config.yaml
