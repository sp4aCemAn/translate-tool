# =========================================================================
# Stage 1: The "Builder"
# This stage's only job is to download and convert the model.
# =========================================================================
FROM python:3.10-slim as builder

# Install only the dependencies needed for conversion
RUN pip install --no-cache-dir ctranslate2 transformers sentencepiece

# Set environment variables for the model names
ARG HF_MODEL_NAME="facebook/nllb-200-3.3B"
ARG CT2_MODEL_DIR="/converted_model"

# Run the conversion process. The resulting model is saved inside this stage.
RUN ct2-transformers-converter \
    --model ${HF_MODEL_NAME} \
    --output_dir ${CT2_MODEL_DIR} \
    --quantization int8 \
    --force

# =========================================================================
# Stage 2: The Final Application Image
# This is the slim, final image that will actually run your app.
# =========================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
# The model path inside THIS container.
ENV CT2_CONVERTED_MODEL=/app/models/nllb-200-3.3B-ct2-int8

# Install Python and dependencies needed for running the app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch ctranslate2 transformers sentencepiece

# Set the working directory
WORKDIR /app

# The MAGIC: Copy the converted model from the "builder" stage
# This copies the result of Stage 1 into our final image.
COPY --from=builder /converted_model ${CT2_CONVERTED_MODEL}

# Copy your application code
COPY app.py .

# Command to run the application
CMD ["python3", "app.py"]
