# Offline inference image: builds the test submission from the dataset mounted at /data.
# Weights are fetched at build time from a public source and verified by sha256; at runtime there is no network.
FROM python:3.13.7-slim@sha256:5f55cdf0c5d9dc1a415637a5ccc4a9e18663ad203673173b8cda8f8dcacef689 AS dependencies

WORKDIR /app
RUN pip install --no-cache-dir --no-deps torch==2.10.0+cu126 torchvision==0.25.0+cu126 --index-url https://download.pytorch.org/whl/cu126
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt --extra-index-url https://download.pytorch.org/whl/cu126

FROM dependencies AS inference
COPY src/ src/
COPY configs/ configs/
COPY scripts/docker_fetch_checkpoint.py scripts/entrypoint.sh scripts/
ENV PYTHONPATH=/app/src HF_HOME=/opt/hf PYTHONDONTWRITEBYTECODE=1

# Public weights pinned by sha256 in configs/weights.json: the build needs network but no credentials.
RUN --mount=type=cache,id=yslfalcon-checkpoints,target=/var/cache/reid,sharing=locked \
    python scripts/docker_fetch_checkpoint.py
ENV HF_HUB_OFFLINE=1

RUN useradd --uid 1000 --create-home falcon && mkdir -p /app/artifacts && chown falcon:falcon /app/artifacts
USER falcon
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["submit", "configs/final_model.yaml"]
