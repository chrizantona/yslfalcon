FROM python:3.13-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26 AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.lock /app/requirements.lock
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r /app/requirements.lock \
    && groupadd --gid 10001 falcon \
    && useradd --uid 10001 --gid falcon --create-home falcon \
    && mkdir -p /data /models \
    && chown falcon:falcon /data /models
COPY --chown=falcon:falcon backend /app/backend
ENV PYTHONPATH=/app/backend FALCON_STORAGE_ROOT=/data FALCON_PORT=8000
USER falcon
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('FALCON_PORT','8000')+os.getenv('FALCON_HEALTH_PATH','/api/v1/health/ready'),timeout=3)"
CMD ["uvicorn", "falcon_service.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--no-access-log"]
