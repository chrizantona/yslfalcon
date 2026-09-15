# Offline inference image: builds the test submission from the dataset mounted at /data.
# Not verified yet: the development pod has no Docker.
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
ENV PYTHONPATH=/app/src HF_HOME=/opt/hf

# bake the pretrained weights into the image, then forbid network access at runtime
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')"
ENV HF_HUB_OFFLINE=1

CMD ["python", "-m", "reid.submit", "--data-config", "configs/data_docker.yaml", "--config", "configs/exp000_clip_zeroshot.yaml"]
