# musictrain — reproducible training/inference image.
# Build:   docker build -t musictrain .
# Run CLI: docker run --rm -v $PWD:/work musictrain eval --seeds 3
# Run API: docker run --rm -p 8000:8000 -v $PWD:/work musictrain serve --port 8000

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKDIR=/work

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
    && pip install fastapi uvicorn

# Optional heavyweight deps (uncomment if you need them in-image):
# RUN pip install torch torchaudio transformers accelerate \
#     librosa soundfile scikit-learn streamlit mlflow

EXPOSE 8000 8501

ENTRYPOINT ["python", "-m", "musictrain.cli"]
