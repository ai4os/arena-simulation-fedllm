FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY arena-fedllm /app/arena-fedllm

WORKDIR /app/arena-fedllm

RUN python -m pip install --upgrade pip \
    && pip install -e .

CMD ["flwr", "run", ".", "--stream"]
