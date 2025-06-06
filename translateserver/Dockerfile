# Use a CUDA base image (adjust tag based on your driver/CUDA version if needed)
# Check NVIDIA documentation for recommended base images
# Using CUDA 12.1.1 runtime as an example
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables FOR ENGLISH TO JAPANESE
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CT2_CONVERTED_MODEL=/app/models/nllb-200-3.3B-ct2-int8 \ 
    HF_MODEL_NAME=facebook/nllb-200-3.3B \ 
    PIP_DEFAULT_TIMEOUT=100 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1k

# Install system dependencies
# Added wget just in case, git for potential future use
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and related tools first to avoid potential issues finding packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python libraries
# Ensure torch version matches the CUDA version in the base image (cu121 for CUDA 12.1)
RUN pip3 install --no-cache-dir \
    ctranslate2 \
    transformers \
    sentencepiece \
    torch 

# Set working directory
WORKDIR /app

# Download the original Hugging Face model and convert it to CTranslate2 INT8 format
# This happens during the image build process using the ENV variables defined above
RUN echo "Downloading and converting model: ${HF_MODEL_NAME}" \
	&& python3 -c "import os; from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
		       print(f'Downloading model {os.environ.get(\"HF_MODEL_NAME\")}...'); \
		       model = AutoModelForSeq2SeqLM.from_pretrained(os.environ.get('HF_MODEL_NAME'), token=False); \
		       print(f'Downloading tokenizer {os.environ.get(\"HF_MODEL_NAME\")}...'); \
		       tokenizer = AutoTokenizer.from_pretrained(os.environ.get('HF_MODEL_NAME'), token=False); \
		       print('Saving model and tokenizer temporarily...'); \
		       model.save_pretrained('./hf_model_temp'); \
		       tokenizer.save_pretrained('./hf_model_temp'); \
		       print('Download complete.')" \
	    && echo "Converting model to CTranslate2 format: ${CT2_CONVERTED_MODEL}" \
	    && ct2-transformers-converter --model ./hf_model_temp --output_dir ${CT2_CONVERTED_MODEL} --quantization int8 --force \
	    && rm -rf ./hf_model_temp \
	    && echo "Model conversion complete."# Copy the CLI application file
	COPY app.py .

# Command to run the CLI application
CMD ["python3", "app.py"]
