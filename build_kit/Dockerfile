FROM python:latest AS model
RUN pip install -U pip && \
    pip install huggingface_hub[cli]

RUN huggingface-cli download HuggingFaceTB/SmolVLM2-500M-Video-Instruct onnx/decoder_model_merged.onnx onnx/embed_tokens.onnx onnx/vision_encoder.onnx

FROM ghcr.io/make87/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
# copy folder from `model` image in `/root/.cache/huggingface/hub` to `/root/.cache/huggingface/hub` in `base` image
COPY --from=model /root/.cache/huggingface/hub /root/.cache/huggingface/hub

