# musictrain — reproducible training/inference image.
# Build:   docker build -t musictrain .
# Run CLI: docker run --rm -v $PWD:/work musictrain eval --seeds 3
# Run API: docker run --rm -p 8000:8000 -v $PWD:/work musictrain serve --port 8000
#
# Optional HF cache warm-up (#13): bake the model weights into the image so the
# first generation doesn't download them at runtime:
#   docker build --build-arg WARM_MODEL=facebook/musicgen-small -t musictrain .
#
# CUDA image for training on a GPU host (e.g. the Ubuntu workstation):
#   docker build --build-arg WITH_CUDA=1 --build-arg TORCH_INDEX=cu124 -t musictrain:cuda .
#   docker run --rm --gpus all -v $PWD:/work musictrain:cuda finetune --steps 5

FROM python:3.11-slim

ARG WITH_CUDA=0
ARG TORCH_INDEX=cu124

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKDIR=/work \
    HF_HOME=/cache/huggingface

# System deps: ffmpeg for audio decode/normalize, curl for health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY pyproject.toml ./
COPY musictrain ./musictrain

# Install the package without heavy optional deps first (fast, cacheable layer),
# then pull the full model-serving stack on demand via the extras below.
RUN pip install --upgrade pip \
    && pip install . \
    && pip install fastapi uvicorn huggingface_hub

# Heavyweight training deps. Default build keeps the image lean (CPU torch);
# --build-arg WITH_CUDA=1 swaps in CUDA builds from the pytorch index so
# finetune/train-lyrics run on the GPU (auto-detected by the code).
RUN if [ "$WITH_CUDA" = "1" ]; then \
      pip install torch torchaudio --index-url https://download.pytorch.org/whl/$TORCH_INDEX; \
    else \
      pip install torch torchaudio; \
    fi \
    && pip install transformers accelerate peft librosa soundfile scikit-learn streamlit mlflow

# Optional cache warm-up (#13): pre-pull model weights (defaults to nothing).
ARG WARM_MODEL=""
RUN if [ -n "$WARM_MODEL" ]; then \
      python -m musictrain.cli warm-cache --model "$WARM_MODEL" || true; \
    fi

EXPOSE 8000 8501

ENTRYPOINT ["python", "-m", "musictrain.cli"]
