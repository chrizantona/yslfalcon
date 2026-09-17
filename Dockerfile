# Offline inference image: builds the test submission from the dataset mounted at /data.
# Not verified yet: the development pod has no Docker.
FROM python:3.13.7-slim@sha256:5f55cdf0c5d9dc1a415637a5ccc4a9e18663ad203673173b8cda8f8dcacef689 AS dependencies

WORKDIR /app
RUN pip install --no-cache-dir --no-deps torch==2.10.0+cu126 torchvision==0.25.0+cu126 --index-url https://download.pytorch.org/whl/cu126
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt --extra-index-url https://download.pytorch.org/whl/cu126

FROM dependencies AS inference
COPY src/ src/
COPY configs/ configs/
COPY scripts/docker_fetch_checkpoint.py scripts/docker_fetch_checkpoint.py
ENV PYTHONPATH=/app/src HF_HOME=/opt/hf PYTHONDONTWRITEBYTECODE=1

# Bake the trained checkpoint into the image, then forbid HF downloads at runtime.
ARG WEIGHTS_URL=https://api.github.com/repos/chrizantona/yslfalcon/releases/assets/569756756
ARG WEIGHTS_SHA256=b1a937546b14b693d35196d3dab23fcbae48b1e7657fe30743307a4a8b125bde
ARG CHECKPOINT_NAME=exp024.pt
RUN --mount=type=secret,id=github_token python scripts/docker_fetch_checkpoint.py
ENV HF_HUB_OFFLINE=1

RUN useradd --uid 1000 --create-home falcon && mkdir -p /app/artifacts && chown falcon:falcon /app/artifacts
USER falcon
ENTRYPOINT ["python", "-m", "reid.submit", "--data-config", "configs/data_docker.yaml", "--config"]
CMD ["configs/submit_exp024.yaml"]
