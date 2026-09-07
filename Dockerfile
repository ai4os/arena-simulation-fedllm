FROM python:3.10-slim

LABEL maintainer='Judith Sáinz-Pardo '
LABEL version='0.0.1'

ARG branch=main

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        nano \
        tini \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 \
    SHELL=/bin/bash

RUN mkdir -p /srv \
    && git clone https://github.com/ai4os/deep-start /srv/.deep-start \
    && ln -s /srv/.deep-start/deep-start.sh /usr/local/bin/deep-start

COPY arena-fedllm /app/arena-fedllm

WORKDIR /app/arena-fedllm

RUN python -m pip install --upgrade pip \
    && pip install -e .

EXPOSE 8888

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["deep-start", "-j"]
