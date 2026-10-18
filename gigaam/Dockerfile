# GigaAM ASR server, NVIDIA GPUs (CUDA 12.8 wheels: Blackwell sm_120 and everything older).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DO_NOT_TRACK=1

# ffmpeg decodes every upload (gigaam.load_audio shells out to it), git installs GigaAM
# from a pinned commit, curl is the container healthcheck.
RUN set -xe \
 && apt-get update -q \
 && apt-get install -fyq --no-install-recommends ffmpeg git curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.10.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --index-url ${TORCH_INDEX} \
      "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}"

# gigaam on PyPI is stuck at 0.1.0 and has no v3 checkpoints, so install upstream at a
# pinned commit. asr.py rides on internals (AudioDataset, model._decode), do not float it.
ARG GIGAAM_REF=559d88d6b72541412743929f633a6ae7c9950b85
RUN pip install "gigaam @ git+https://github.com/salute-developers/GigaAM.git@${GIGAAM_REF}"

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY asr.py main.py /app/

ENV GIGAAM_DEVICE=cuda \
    GIGAAM_CACHE_DIR=/models \
    APP_PORT=8770
EXPOSE 8770

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${APP_PORT}"]
