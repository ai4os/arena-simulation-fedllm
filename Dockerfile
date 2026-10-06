FROM python:3.10-bookworm

LABEL maintainer='Judith Sáinz-Pardo '
LABEL version='0.8'

ARG branch=main

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    SHELL=/bin/bash

RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        curl \
        wget \
        git \
        nano \
        tini \
        procps \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/tsl0922/ttyd/releases/download/1.7.4/ttyd.x86_64 -O /usr/bin/ttyd \
    && chmod +x /usr/bin/ttyd

RUN mkdir -p /srv \
    && git clone https://github.com/ai4os/deep-start /srv/.deep-start \
    && ln -s /srv/.deep-start/deep-start.sh /usr/local/bin/deep-start

COPY arena-fedllm /srv/arena-fedllm
WORKDIR /srv/arena-fedllm
RUN python -m pip install --upgrade pip \
    && pip install -e .

EXPOSE 5000 6006 8888

WORKDIR /srv

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["ttyd", "-p", "6006", "bash", "-lc", "mkdir -p /storage/logs && cd /srv/arena-fedllm && flwr run . --stream 2>&1 | tee /storage/logs/run_$(date +%F_%H-%M-%S).log; cp -r /root/.flwr /storage/logs/flwr-home 2>/dev/null; sleep infinity"]