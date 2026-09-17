# Offline inference image: builds the test submission from the dataset mounted at /data.
# Not verified yet: the development pod has no Docker.
FROM python:3.13.7-slim

WORKDIR /app
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY scripts/docker_fetch_checkpoint.py scripts/docker_fetch_checkpoint.py
ENV PYTHONPATH=/app/src HF_HOME=/opt/hf PYTHONDONTWRITEBYTECODE=1

# bake the pretrained weights into the image, then forbid network access at runtime
ARG WEIGHTS_URL
ARG WEIGHTS_SHA256=b1a937546b14b693d35196d3dab23fcbae48b1e7657fe30743307a4a8b125bde
RUN --mount=type=secret,id=github_token python scripts/docker_fetch_checkpoint.py
ENV HF_HUB_OFFLINE=1

RUN useradd --uid 1000 --create-home falcon && mkdir -p /app/artifacts && chown falcon:falcon /app/artifacts
USER falcon
CMD ["python", "-m", "reid.submit", "--data-config", "configs/data_docker.yaml", "--config", "configs/submit_exp024.yaml"]
