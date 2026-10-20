# GPU model runtime: the service's runtime app with the trained encoder behind it (FALCON_ENCODER=production).
# The build context is the repository root, so the model code, its configuration and the weight manifest are the
# very files the submission image uses: src/reid, configs/final_model.yaml, configs/weights.json and
# requirements-inference.txt. A threshold or a checkpoint changes in one place for both.
# Build from the repository root:  docker buildx build --platform linux/amd64 -f service/deploy/runtime.Dockerfile -t <image> .
# docker-compose.model.yml builds it the same way; runtime.Dockerfile.dockerignore keeps the context to these files.
# The weights are fetched at build time from their public source and verified by sha256; the container never goes
# online and refuses to start if a file is missing, altered, or no GPU is visible.
FROM python:3.13.7-slim@sha256:5f55cdf0c5d9dc1a415637a5ccc4a9e18663ad203673173b8cda8f8dcacef689
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# The model stack first: torch with CUDA 12.6 and the exact versions the submission was produced with.
RUN pip install --no-cache-dir --no-deps torch==2.10.0+cu126 torchvision==0.25.0+cu126 \
      --index-url https://download.pytorch.org/whl/cu126
COPY requirements-inference.txt /app/requirements-inference.txt
RUN pip install --no-cache-dir -r /app/requirements-inference.txt --extra-index-url https://download.pytorch.org/whl/cu126
# Then the service's runtime app; its lock only moves idna 3.19 -> 3.20.
COPY service/requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

# Weights before code, so a code change does not download 1.7 GB again.
COPY service/runtime/fetch_weights.py /app/runtime/
COPY configs/weights.json /app/runtime/weights.json
RUN python /app/runtime/fetch_weights.py --out /models

COPY service/backend /app/backend
COPY src/reid /app/runtime/reid
COPY configs/final_model.yaml /app/runtime/final_model.yaml
RUN groupadd --gid 10001 falcon && useradd --uid 10001 --gid falcon --create-home falcon \
    && chown -R root:root /models && chmod -R a+rX /models

# /tmp is the only writable place under the stack's read-only root, so every cache points there.
ENV PYTHONPATH=/app/backend:/app/runtime \
    FALCON_ENCODER=production \
    FALCON_MODEL_MANIFEST=/models/runtime-bundle.json \
    FALCON_MODEL_CONFIG=/app/runtime/final_model.yaml \
    FALCON_MODEL_DEVICE=cuda \
    FALCON_PORT=8001 \
    FALCON_HEALTH_PATH=/health/live \
    HF_HUB_OFFLINE=1 \
    HOME=/tmp XDG_CACHE_HOME=/tmp/.cache TORCH_HOME=/tmp/.cache/torch TRITON_CACHE_DIR=/tmp/.cache/triton
USER falcon
EXPOSE 8001
# Loading 1.7 GB and a warm-up forward take a while; health turns green only after the model has embedded once.
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('FALCON_PORT','8001')+os.getenv('FALCON_HEALTH_PATH','/health/live'),timeout=3)"
CMD ["uvicorn", "falcon_service.model_runtime:app", "--host", "0.0.0.0", "--port", "8001", "--no-server-header", "--no-access-log"]
