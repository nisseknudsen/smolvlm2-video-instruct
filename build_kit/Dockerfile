FROM python:latest AS model
RUN pip install -U pip && \
    pip install huggingface_hub[cli]

RUN huggingface-cli download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --exclude onnx/

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

ARG VIRTUAL_ENV=/make87/venv
ARG DEBIAN_FRONTEND=noninteractive

# copy folder from `model` image in `/root/.cache/huggingface/hub` to `/root/.cache/huggingface/hub` in `base` image
COPY --from=model /root/.cache/huggingface/hub /root/.cache/huggingface/hub


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv python install 3.11 && \
    uv venv ${VIRTUAL_ENV} && \
    sh ${VIRTUAL_ENV}/bin/activate && \
    uv pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY . .

RUN sh ${VIRTUAL_ENV}/bin/activate && \
    uv pip install . && \
    uv pip install --no-build-isolation flash-attn

ENTRYPOINT ["/make87/venv/bin/python3", "-m", "app.main"]
