FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# QuTiP may require native deps in some environments.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gfortran \
      git \
      curl \
      ca-certificates \
      libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

# uv (Python package manager)
RUN pip install --no-cache-dir uv

