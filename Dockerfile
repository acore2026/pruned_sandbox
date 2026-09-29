# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nvidia/cuda:13.0.1-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG PYPI_MIRROR_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HOME=/home/sandbox \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PATH=/opt/venv/bin:${PATH} \
    ASR_MODEL=/models/asr/whisper-large-v3 \
    ASR_DOWNLOAD_ROOT=/models/asr \
    ASR_DEVICE=cuda \
    ASR_COMPUTE_TYPE=float16 \
    INTENT_BACKEND=hybrid \
    INTENT_MODEL=/models/intent/Qwen2.5-0.5B-Instruct \
    YOLO_MODEL=/models/yolo/yolov8s-worldv2.pt \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_DEVICE=0 \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        iproute2 \
        python3.12 \
        python3.12-venv \
        supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

WORKDIR /app
COPY requirements.txt ./
COPY services/asr/requirements.txt /app/services/asr/requirements.txt
COPY services/intent/requirements.txt /app/services/intent/requirements.txt
COPY services/video/requirements.txt /app/services/video/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install "cuda-bindings==13.0.3" --index-url "${PYPI_MIRROR_URL}" \
    && pip install \
        "torch==${TORCH_VERSION}+cu130" "torchvision==${TORCHVISION_VERSION}+cu130" \
        --index-url "${PYPI_MIRROR_URL}" \
        --extra-index-url "${TORCH_INDEX_URL}" \
    && pip install -r requirements.txt --index-url "${PYPI_MIRROR_URL}"

# 三套模型和 CLIP 权重均在运行期由 docker-compose.yml 以只读卷挂载。
# 因此模型权重不会进入镜像层，也不会在运行阶段联网下载；clip Python 包随镜像提供。

COPY services /app/services
COPY services/video/vendor/clip /opt/venv/lib/python3.12/site-packages/clip
COPY services/video/vendor/ftfy /opt/venv/lib/python3.12/site-packages/ftfy
COPY services/video/vendor/wcwidth /opt/venv/lib/python3.12/site-packages/wcwidth
COPY deploy/supervisord.conf /etc/supervisor/conf.d/sandbox.conf
COPY deploy/entrypoint.sh /app/deploy/entrypoint.sh
RUN useradd --create-home --uid 10001 sandbox \
    && mkdir -p /tmp/sandbox-asr /tmp/ultralytics /home/sandbox/.cache/clip /models \
    && chown -R sandbox:sandbox /app /tmp/sandbox-asr /tmp/ultralytics /home/sandbox /models \
    && chmod 0755 /app/deploy/entrypoint.sh

EXPOSE 9004 28501 28502
ENTRYPOINT ["/app/deploy/entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/sandbox.conf"]
