ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.4.1
#ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUNTIME_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_RUNTIME_CONTAINER}
WORKDIR /app

# Install dependencies
RUN set -xe \
 && apt update \
 && apt install -y python3-pip python3-dev git ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt .
RUN set -xe \
 && pip install --no-cache -r requirements.txt \
 && pip install setuptools psutil torch \
 && pip install flash-attn --no-build-isolation \
 && pip cache purge

# Copy sources
COPY . .

ENTRYPOINT ["/app/entrypoint.sh"]
