# Sandbox Lite

## 快速开始

### 1. 准备配置

```bash
cd /home/aicore/pruned_sandbox
test -f sandbox.env || cp sandbox.env.example sandbox.env
tailscale ip -4
```

将`sandbox.env`中的`VIDEO_PUBLIC_IP`改为本机联调地址；使用Tailscale时可通过
`tailscale ip -4`查询。

### 2. 使用容器启动（推荐）

```bash
cd /home/aicore/pruned_sandbox
set -a
. ./sandbox.env
set +a
bash scripts/check_model_sources.sh
docker compose --env-file sandbox.env up -d --build
```

首次构建完成后，再次启动可省略`--build`：

```bash
cd /home/aicore/pruned_sandbox
docker compose --env-file sandbox.env up -d
```

### 3. 查看状态和日志

```bash
cd /home/aicore/pruned_sandbox
docker compose --env-file sandbox.env ps
docker compose --env-file sandbox.env logs -f sandbox
```

宿主机滚动日志：

```bash
cd /home/aicore/pruned_sandbox
tail -f logs/asr.log logs/intent.log logs/sandbox.log
```

### 4. 检查接口

```bash
cd /home/aicore/pruned_sandbox
curl --noproxy '*' http://127.0.0.1:28501/healthz
curl --noproxy '*' http://127.0.0.1:28502/healthz
curl --noproxy '*' http://127.0.0.1:9004/health
docker exec sandbox-lite curl --noproxy '*' -fsS http://127.0.0.1:9005/health
docker exec sandbox-lite curl --noproxy '*' -fsS http://127.0.0.1:8011/health
```

### 5. 测试语音识别

```bash
cd /home/aicore/pruned_sandbox
curl --noproxy '*' -X POST http://127.0.0.1:9004/api/v1/transcribe \
  -F file=@test_audio/patrol-area-a.mp3 \
  -F request_id=asr-001 \
  -F language=zh
```

停止容器：

```bash
cd /home/aicore/pruned_sandbox
docker compose --env-file sandbox.env down
```

### 本地运行（不使用容器）

首次准备虚拟环境：

```bash
cd /home/aicore/pruned_sandbox
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

使用真实模型启动：

```bash
cd /home/aicore/pruned_sandbox
SANDBOX_REAL_MODELS=true ./scripts/run_local.sh
```

按`Ctrl-C`统一停止服务。无GPU的Mock模式执行：

```bash
cd /home/aicore/pruned_sandbox
./scripts/run_local.sh
```

## 运行架构

服务端口：

| 地址 | 用途 |
| --- | --- |
| `http://127.0.0.1:28501` | CMF管理面 |
| `http://127.0.0.1:28502` | N6用户面 |
| `http://{host}:9004` | 对外独立任务 ASR（眼镜首句语音） |
| `http://127.0.0.1:9005` | 内部运行期 ASR（仅Sandbox调用） |
| `http://127.0.0.1:8011` | Intent内部服务 |

容器在运行期以只读方式挂载宿主机的Whisper、Qwen和YOLO权重；模型不复制进镜像。默认目录如下：

```text
/home/aicore/pruned_sandbox/models/
├── whisper-models/whisper-large-v3/model.bin
├── semantic-models/Qwen/Qwen2.5-0.5B-Instruct/model.safetensors
└── yolo-models/yolov8s-worldv2.pt
```

Sandbox作为当前GPU机器上的独立容器运行，使用宿主机网络并通过Tailscale与上下游
通信，不依赖上游Docker网络。容器模式需要Docker Compose、NVIDIA Container
Toolkit和可用NVIDIA GPU；`9005`与`8011`只监听本机回环接口。

一个镜像、一个容器、四个职责目录和三个独立进程：

| 目录 | 进程 | 端口 | 职责 |
| --- | --- | ---: | --- |
| `services/asr` | `python -m services.asr.main` | 0.0.0.0:9004（任务发现）、127.0.0.1:9005（运行期） | 一份faster-whisper模型，双ASR监听端口 |
| `services/intent` | `python -m services.intent.main` | 127.0.0.1:8011（内部） | 文字语义与意图分类 |
| `services/sandbox` | `python -m services.sandbox.main` | 0.0.0.0:28501（CMF管理面）、0.0.0.0:28502（N6用户面） | Sandbox标准接口与运行时装配 |
| `services/video` | 由Sandbox进程加载 | - | WebRTC、YOLO与处理视频输出 |

Supervisor 同时启动并分别守护三个Python进程。Sandbox进程提供两个隔离的
HTTP入口，但共享`binding_ref`、幂等记录、WebRTC PeerConnection、识别目标
和动作状态，避免接口层与媒体层产生两套状态。ASR是眼镜直接调用的独立辅助服务，
Intent是Sandbox内部推理能力；
算网会话申请、算力分配和PDU策略仍由Sandbox外部组件管理。
容器对外发布CMF管理接口`28501`、N6用户面接口`28502`和独立ASR接口`9004`；
`9005`及`8011`仅供容器内部调用。管理路径不会注册到用户面端口，用户路径也不会注册到
管理面端口。

## 算网标准接口

CMF管理面（`28501`）：

- `POST /management/v1/compute-session-bindings:bind`
- `GET /management/v1/compute-session-bindings/{binding_ref}`
- `GET /management/v1/compute-session-bindings/{binding_ref}/media-state`
- `POST /management/v1/compute-session-bindings/{binding_ref}:unbind`

N6用户面（`28502`）：

- `POST /v1/media-connections`
- `DELETE /v1/media-connections/{media_connection_id}`
- `PUT/GET /v1/recognition-targets/{compute_service_session_id}`
- `POST /v1/audio-control-actions`（兼容入口；只返回文本和意图）

管理接口使用`FREE6GC_COMPUTING_SANDBOX_MANAGEMENT_TOKEN`配置Bearer Token。
用户面请求必须携带完整`computing_context`，服务按`binding_ref`校验会话、
实例、角色及Agent。producer和consumer均由终端提交Offer，Sandbox返回
非Trickle ICE Answer。

Sandbox不向机器狗发送控制指令。旧的`POST /v1/control-actions`与
`GET /v1/control-actions/{action_id}`保留路由但固定返回`410 Gone`，避免旧调用方误认为
Sandbox仍会执行设备动作。视频识别仍使用`U-RECOGNITION`维护持续识别目标。

## 模型

容器启动时从当前仓库`models`目录只读挂载权重，不在线下载，也不将权重写入镜像层：

- `whisper-large-v3`
- `Qwen2.5-0.5B-Instruct`
- `yolov8s-worldv2.pt`

默认模型源目录由`sandbox.env.example`中的`ASR_MODEL_SOURCE`、`INTENT_MODEL_SOURCE`
和`YOLO_MODEL_SOURCE`配置。这三个目录是容器启动前的必要条件；`bash scripts/check_model_sources.sh`
可检查它们是否完整。

## 运行配置说明

容器使用宿主机网络，上下游通过`VIDEO_PUBLIC_IP`访问`28501`、`28502`和`9004`。
容器需要Docker Compose、NVIDIA Container Toolkit及可用GPU。默认基础镜像为
`nvidia/cuda:13.0.1-runtime-ubuntu24.04`；它不包含CUDA编译工具，可显著减小镜像。
可通过`BASE_IMAGE`覆盖为兼容的本地镜像。

日志同时输出到终端并滚动保存到`logs/asr.log`、`logs/intent.log`和
`logs/sandbox.log`。默认单文件上限10 MiB、保留5份历史记录，可通过
`LOG_DIR`、`LOG_MAX_BYTES`和`LOG_BACKUP_COUNT`调整。

本地轻量模式关闭ASR和YOLO推理，并使用规则意图分类，适合接口Mock；真实模型模式
会从当前仓库`models/`目录加载Whisper和Qwen。具体命令见顶部“快速开始”。

## ASR

ASR是独立辅助服务，不属于十个Sandbox标准接口。对外接口为`POST /api/v1/transcribe`，
使用`multipart/form-data`提交`file`、`request_id`和可选`language`：

```bash
cd /home/aicore/pruned_sandbox
curl --noproxy '*' -X POST http://127.0.0.1:9004/api/v1/transcribe \
  -F file=@speech.wav \
  -F request_id=asr-001 \
  -F language=zh
```

响应只包含`request_id`、`text`和`intent`；任务发现实例额外返回`required_skills`。
同一个ASR进程通过监听端口区分两类请求：`9004`的`discovery`用于Sandbox拉起前的首句任务，
严格按场景文档映射“巡逻、巡检”为`TASK`和`["patrol", "camera"]`，“实时画面、
查看现场”为`VIDEO_TASK`和相同技能，“可疑物识别”为`OBJECT_RECOGNITION`和
`["camera"]`。仅本机可访问的`9005`为`runtime`，用于已建立算力会话后的Sandbox；“威吓歹徒”和“驱逐歹徒”
均返回`executor=robot dog`、`intent=movement`、`direction=forward`。任意音频都返回
转写文本；任务发现未命中时为`intent.type=UNKNOWN`，运行期未命中时为
`intent.matched=false`。

`POST /v1/audio-control-actions`仅为已绑定会话保留兼容入口；它校验
`computing_context`后调用本机`9005`，返回文本和运行期动作意图，不创建动作任务，也不控制机器狗。

支持 `wav/mp3/m4a/flac/ogg/webm`，默认最大 50 MiB。默认加载
`whisper-large-v3`，使用 CUDA `float16`。可通过`ASR_INITIAL_PROMPT`、`ASR_HOTWORDS`、
`ASR_DISCOVERY_HOST`、`ASR_DISCOVERY_PORT`、`ASR_RUNTIME_HOST`和`ASR_RUNTIME_PORT`配置监听地址。详细交接契约见
`AR眼镜语音识别与意图接口定义.md`。

## 意图分类

接口：

- `GET /health`
- `POST /api/v1/intent`
- `POST /api/v1/semantic/route`（兼容原入口）

```bash
cd /home/aicore/pruned_sandbox
curl http://127.0.0.1:8011/api/v1/intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"帮我找黄色的狗"}'
```

默认候选意图包含场景映射的`patrol`、`video_task`、`object_recognition`、`defense`、
`movement`，以及兼容原语义路由的`find_object`、`grab`和`other`，可通过
`INTENT_CANDIDATES`配置。默认
`INTENT_BACKEND=hybrid`：使用宿主机挂载的原 Qwen 模型，模型失败时回退规则。
响应中的`executor`是可直接用于 Agent Discovery `required_skills`的 skill；当前机器狗
相关意图返回`robot dog`，未命中`other`时返回`null`。

## Orange兼容WebRTC接口

以下旧接口暂时保留用于已有Orange联调；新算网流程应使用上述
`/v1/media-connections`，不再依赖旧的服务端producer Offer流程。

完整流程：

```text
核心网分配 session，并向 Orange SDK 返回媒体端点
        │
        ▼
Video Server 创建 recvonly Offer
        │ HTTP 响应
        ▼
手机设置 Remote Offer，发送摄像头轨道，创建 Answer
        │ HTTP POST
        ▼
Video Server 收到第一帧 → SOURCE_CONNECTED
        │
        ├── 最新帧队列（容量 1）→ YOLO → session 独立处理画面
        │
        ▼
消费端创建 recvonly Offer → Server 返回 Answer → 接收 YOLO 视频
```

Offer 是 Video Server 对手机发起的视频协商，但通过手机主动发出的 HTTP
请求响应返回；手机端不需要 HTTP 服务端点。

`session_id` 第一次出现在 source 或 processed 媒体请求中时，服务只为它
创建进程内 WebRTC/YOLO 管线。这个本地状态用于关联上下行媒体，不表示服务
创建或管理了核心网卸载会话。

### Orange 媒体接口

| 方法 | 路径 | 鉴权 | 功能 |
| --- | --- | --- | --- |
| GET | `/healthz` | 无 | Orange 健康检查 |
| GET | `/debug/v1/sessions` | 无 | 查看会话状态 |
| POST | `/video/v1/sessions/{session_id}/source` | 无 | 创建源 Offer或提交源 Answer |
| POST | `/video/v1/sessions/{session_id}/source/stop` | 无 | 停止源和相关 PeerConnection |
| POST | `/video/v1/sessions/{session_id}/processed` | 无 | 提交消费端 Offer并获得 YOLO 视频 Answer |

服务不提供 `/compute/v1/offloading-sessions`、`.../consumers` 或强制删除会话
接口，也不生成 producer token、consumer ticket 或校验 Bearer。核心网分配
session 时应返回指向上述媒体路径的 `producer` 和 `processed_stream`。

```json
{
  "producer": {
    "video_server_ip": "<sandbox-tailscale-ip>",
    "source_start_url": "http://<sandbox-tailscale-ip>:28502/video/v1/sessions/{session_id}/source",
    "source_stop_url": "http://<sandbox-tailscale-ip>:28502/video/v1/sessions/{session_id}/source/stop"
  },
  "processed_stream": {
    "video_server_ip": "<sandbox-tailscale-ip>",
    "offer_url": "http://<sandbox-tailscale-ip>:28502/video/v1/sessions/{session_id}/processed",
    "protocol": "webrtc",
    "signaling": "non-trickle"
  }
}
```

源端第一次请求 Offer：

```http
POST /video/v1/sessions/{session_id}/source
Content-Type: application/json

{"action":"create_offer"}
```

返回的 SDP 使用嵌套格式：

```json
{
  "session_id": "core-session-...",
  "state": "WAITING_FOR_SOURCE",
  "sdp_offer": {
    "type": "offer",
    "sdp": "v=0\r\n..."
  }
}
```

手机创建 Answer 后再次 POST 同一路径：

```json
{
  "sdp_answer": {
    "type": "answer",
    "sdp": "v=0\r\n..."
  }
}
```

只有服务端收到第一帧后才返回 `SOURCE_CONNECTED`；默认等待 12 秒。消费端
不需要等待 source，可以提前用核心网返回的 `processed_stream.offer_url`
创建 `recvonly` Offer：

```http
POST /video/v1/sessions/{session_id}/processed
Content-Type: application/json

{
  "sdp_offer": {
    "type": "offer",
    "sdp": "v=0\r\n..."
  }
}
```

响应中的 `sdp_answer` 用于消费端的 Remote Description。source 尚未连接时
响应状态为 `SOURCE_PENDING` 并持续输出占位帧；YOLO 首帧就绪后在同一个远端
Track 内切换，不重新协商。

### H.264 与 MTU

- B→Server 只协商 H.264 High/Constrained High（`64001f`/`640c1f`，
  `packetization-mode=1`），并校验 Answer，禁止静默回退。
- Server→消费端优先 H.264 Baseline，与 aiortc 实际编码能力一致。
- 与当前 mock 一样，H.264 FU-A 的 RTP payload 上限默认是 **1150 bytes**，
  可通过 `VIDEO_H264_RTP_PAYLOAD_BYTES` 在 `1100..1150` 内调整。这不是网卡 MTU；
  它是为 **1280-byte Android TUN MTU** 的 RTP 扩展、SRTP、UDP 和 IP 头预留
  空间后的媒体 payload 上限。
- 上下行连接都会在必要时主动请求关键帧，减少启动关键帧丢失导致的首帧等待。

处理流默认固定为 `640x480@30fps`；不同尺寸的源帧会等比例缩放并 letterbox，
占位帧切到 YOLO 帧时不会改变输出分辨率。

### YOLO 扩展接口

- `GET /api/v1/detection/classes`：查询当前检测目标。
- `POST /api/v1/detection/classes`：动态设置检测目标。
- `GET /health`、`GET /api/health`：YOLO 和帧处理详细状态。

```bash
cd /home/aicore/pruned_sandbox
curl -X POST http://127.0.0.1:28502/api/v1/detection/classes \
  -H 'Content-Type: application/json' \
  -d '{"classes":["person","dog"]}'
```

默认且仅使用原通用 `yolov8s-worldv2.pt`。YOLO 模型进程内只加载一次，
每个核心网 session ID 使用独立帧管线，多个观看端不会重复推理。

## 测试

```bash
cd /home/aicore/pruned_sandbox
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖三个独立服务、原模型默认值、无会话管理接口，以及真实 aiortc
端到端链路（消费端先建链、占位帧、H.264 High 源 Answer、YOLO 帧无重协商
切换、RTP payload 限制和停止媒体）。
