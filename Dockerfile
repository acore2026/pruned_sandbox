ARG BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:${PATH} \
    ASR_MODEL=/models/asr/whisper-large-v3 \
    ASR_DOWNLOAD_ROOT=/models/asr \
    ASR_DEVICE=cuda \
    ASR_COMPUTE_TYPE=float16 \
    INTENT_BACKEND=hybrid \
    INTENT_MODEL=/models/intent/Qwen2.5-0.5B-Instruct \
    YOLO_MODEL=/models/yolo/yolov8s-worldv2.pt \
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
    && python3.12 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt ./
COPY services/asr/requirements.txt /app/services/asr/requirements.txt
COPY services/intent/requirements.txt /app/services/intent/requirements.txt
COPY services/video/requirements.txt /app/services/video/requirements.txt
RUN pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}" \
    && pip install --no-cache-dir -r requirements.txt

# 这三个命名构建上下文由 docker-compose.yml 指向 sandbox-demo 的原模型目录。
# 模型直接进入镜像，不在构建或运行阶段联网下载。
COPY --from=asr_model / /models/asr/whisper-large-v3/
COPY --from=intent_model / /models/intent/Qwen2.5-0.5B-Instruct/
COPY --from=yolo_models / /models/yolo/
RUN test -s /models/asr/whisper-large-v3/model.bin \
    && test -s /models/asr/whisper-large-v3/config.json \
    && test -s /models/intent/Qwen2.5-0.5B-Instruct/model.safetensors \
    && test -s /models/intent/Qwen2.5-0.5B-Instruct/config.json \
    && test -s /models/yolo/yolov8s-worldv2.pt \
    && test -s /models/yolo/box0612.pt \
    && test -s /models/yolo/toy.pt \
    && test -s /models/yolo/bottles.pt

COPY services /app/services
COPY deploy/supervisord.conf /etc/supervisor/conf.d/sandbox.conf
COPY deploy/entrypoint.sh /app/deploy/entrypoint.sh
RUN useradd --create-home --uid 10001 sandbox \
    && mkdir -p /tmp/sandbox-asr /models \
    && chown -R sandbox:sandbox /app /tmp/sandbox-asr /models \
    && chmod 0755 /app/deploy/entrypoint.sh

EXPOSE 9004 8011 28500
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl --noproxy '*' -fsS http://127.0.0.1:9004/health >/dev/null \
        && curl --noproxy '*' -fsS http://127.0.0.1:8011/health >/dev/null \
        && curl --noproxy '*' -fsS http://127.0.0.1:28500/healthz >/dev/null \
        || exit 1

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/sandbox.conf"]
