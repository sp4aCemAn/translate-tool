#!/bin/bash
set -e # Exit immediately if a command fails

echo "--- Setting up Python virtual environment ---"
sudo apt-get update && sudo apt-get install python3-venv -y
python3 -m venv converter_env
source converter_env/bin/activate
pip install --upgrade pip
pip install ctranslate2 transformers sentencepiece torch

echo "--- Preparing directories ---"
mkdir -p ~/models

# Define variables
export HF_MODEL="facebook/nllb-200-3.3B"
export CT2_MODEL_DIR="~/models/nllb-200-3.3B-ct2-int8"

echo "--- Starting model download and conversion (This will be slow!) ---"
echo "Source: ${HF_MODEL}"
echo "Destination: ${CT2_MODEL_DIR}"

ct2-transformers-converter \
  --model ${HF_MODEL} \
  --output_dir ${CT2_MODEL_DIR} \
  --quantization int8 \
  --force

echo "--- Cleaning up ---"
deactivate

echo "----------------------------------------------------"
echo "SUCCESS: Model is downloaded, converted, and ready!"
echo "Location: ${CT2_MODEL_DIR}"
echo "You can now delete the 'converter_env' directory if you want."
echo "----------------------------------------------------"
