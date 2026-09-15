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
docker exec sandbox-lite curl --noproxy '*' -fsS http://127.0.0.1:8011/health
```

### 5. 测试语音识别

```bash
cd /home/aicore/pruned_sandbox
curl --noproxy '*' -X POST http://127.0.0.1:9004/api/v1/transcribe \
  -F file=@test_audio/patrol-area-a.mp3 \
  -F session_id=patrol-test \
  -F task_id=asr-001 \
  -F source=glasses \
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
| `http://127.0.0.1:9004` | ASR语音转文字 |
| `http://127.0.0.1:8011` | Intent内部服务 |

容器将Whisper、Qwen和YOLO权重复制进镜像，默认目录如下：

```text
/home/aicore/pruned_sandbox/models/
├── whisper-models/whisper-large-v3/model.bin
├── semantic-models/Qwen/Qwen2.5-0.5B-Instruct/model.safetensors
└── yolo-models/yolov8s-worldv2.pt
```

Sandbox作为当前GPU机器上的独立容器运行，使用宿主机网络并通过Tailscale与上下游
通信，不依赖上游Docker网络。容器模式需要Docker Compose、NVIDIA Container
Toolkit和可用NVIDIA GPU；`8011`只监听容器内部。

一个镜像、一个容器、四个职责目录和三个独立进程：

| 目录 | 进程 | 端口 | 职责 |
| --- | --- | ---: | --- |
| `services/asr` | `python -m services.asr.main` | 0.0.0.0:9004（对外） | 眼镜直接调用的faster-whisper语音转文字 |
| `services/intent` | `python -m services.intent.main` | 127.0.0.1:8011（内部） | 文字语义与意图分类 |
| `services/sandbox` | `python -m services.sandbox.main` | 0.0.0.0:28501（CMF管理面）、0.0.0.0:28502（N6用户面） | Sandbox标准接口与运行时装配 |
| `services/video` | 由Sandbox进程加载 | - | WebRTC、YOLO与处理视频输出 |

Supervisor 同时启动并分别守护三个Python进程。Sandbox进程提供两个隔离的
HTTP入口，但共享`binding_ref`、幂等记录、WebRTC PeerConnection、识别目标
和动作状态，避免接口层与媒体层产生两套状态。ASR是眼镜直接调用的独立辅助服务，
Intent是Sandbox内部推理能力；
算网会话申请、算力分配和PDU策略仍由Sandbox外部组件管理。
容器发布CMF管理接口`28501`、N6用户面接口`28502`和独立ASR接口`9004`；
`8011`仅供容器内部调用。管理路径不会注册到用户面端口，用户路径也不会注册到
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
- `POST /v1/control-actions`
- `POST /v1/audio-control-actions`（运行期语音动作扩展）
- `GET /v1/control-actions/{action_id}`

管理接口使用`FREE6GC_COMPUTING_SANDBOX_MANAGEMENT_TOKEN`配置Bearer Token。
用户面请求必须携带完整`computing_context`，服务按`binding_ref`校验会话、
实例、角色及Agent。producer和consumer均由终端提交Offer，Sandbox返回
非Trickle ICE Answer。

`search_object`直接在当前绑定的视频管线上执行一次性搜索，不替换
`U-RECOGNITION`维护的持续目标。`movement`和`grab`通过
`SANDBOX_PRODUCER_CONTROL_URL`转发给当前绑定的机器狗；接口文档未规定机器狗
业务端口，因此该URL由部署方配置，例如
`http://{ue_ipv4_address}:8080/v1/control-actions`。未配置时动作会进入`FAILED`，
cause为`producer-control-endpoint-unconfigured`，不会伪装成已执行。

园区巡逻业务建立后的自然语言方向指令由Sandbox归一化为接口文档规定的
`movement`结构化参数：`向前/前进`对应`direction=forward`，`退后/后退`
对应`direction=backward`，`向左/左转`对应`direction=left`，`向右/右转`
对应`direction=right`。`派机器狗巡逻园区内A区域`属于上游业务会话创建意图，
不作为Sandbox运行期移动动作执行。

## 模型

构建镜像时直接复制当前仓库`models`目录中的权重，不在线下载：

- `whisper-large-v3`
- `Qwen2.5-0.5B-Instruct`
- `yolov8s-worldv2.pt`

默认模型源目录由`sandbox.env.example`中的`ASR_MODEL_SOURCE`、`INTENT_MODEL_SOURCE`
和`YOLO_MODEL_SOURCE`配置。运行时无需模型挂载。

## 运行配置说明

容器使用宿主机网络，上下游通过`VIDEO_PUBLIC_IP`访问`28501`、`28502`和`9004`。
容器需要Docker Compose、NVIDIA Container Toolkit及可用GPU。默认基础镜像为
`nvidia/cuda:13.0.1-devel-ubuntu24.04`，可通过`BASE_IMAGE`覆盖为兼容的本地镜像。

日志同时输出到终端并滚动保存到`logs/asr.log`、`logs/intent.log`和
`logs/sandbox.log`。默认单文件上限10 MiB、保留5份历史记录，可通过
`LOG_DIR`、`LOG_MAX_BYTES`和`LOG_BACKUP_COUNT`调整。

本地轻量模式关闭ASR和YOLO推理，并使用规则意图分类，适合接口Mock；真实模型模式
会从当前仓库`models/`目录加载Whisper和Qwen。具体命令见顶部“快速开始”。

## ASR

ASR是独立辅助服务，不属于十个Sandbox标准接口。Sandbox尚未拉起时，眼镜录制首条
业务创建语音后直接向`9004`上传音频，取得`TEXT`后调用核心网拉起业务。业务绑定建立后，眼镜可经
N6向`28502`的`POST /v1/audio-control-actions`上传运行期动作音频，Sandbox校验
`computing_context`后通过`SANDBOX_ASR_URL`调用内部ASR，并把转写文本接入现有
意图分类、`ControlAction`状态和机器狗转发链路。该扩展不改变十个标准接口。

接口：

- `GET /health`
- `POST /api/v1/transcribe`

转写使用 `multipart/form-data` 上传音频文件：

```bash
cd /home/aicore/pruned_sandbox
curl http://127.0.0.1:9004/api/v1/transcribe \
  -F file=@speech.wav \
  -F session_id=demo-room \
  -F task_id=task-001 \
  -F source=glasses \
  -F language=zh
```

Sandbox已绑定后的运行期语音动作示例：

```bash
cd /home/aicore/pruned_sandbox
curl -X POST http://127.0.0.1:28502/v1/audio-control-actions \
  -F request_id=voice-action-001 \
  -F 'computing_context={"compute_service_session_id":"css-001","compute_instance_id":"ci-001","binding_ref":"binding-css-001","role":"consumer","agent_id":"glasses"}' \
  -F language=zh \
  -F file=@speech.wav
```

`9004`直连接口会在转写后完成意图识别，一并返回文本和`intent`，但始终不触发控制。
候选意图由服务端`INTENT_CANDIDATES`预设，眼镜请求无需携带候选列表。
`intent`使用园区业务槽位：`executor`表示执行主体，`intent`表示业务意图；
安防巡逻使用`area`参数，移动控制使用`direction`参数，目标相关意图使用`object`参数。
`matched`和`backend`分别表示是否命中候选意图及实际识别后端。例如
“派机器狗巡逻园区内A区域”返回`executor=robot dog`、
`intent=security patrol`和`area=A`。

`28502`会等待转写和意图识别完成后返回`transcription`、`intent`、`action_id`、
`normalized_action`及`normalized_parameters`。命中运行期控制意图时设置
`control_triggered=true`并异步下发，初始动作状态为`ACCEPTED`；未命中时设置
`control_triggered=false`、`status=COMPLETED`且不下发控制，不再将普通语音作为422错误处理。
动作执行状态继续通过原`GET /v1/control-actions/{action_id}`查询。默认内部ASR地址为
`http://127.0.0.1:9004/api/v1/transcribe`，可通过`SANDBOX_ASR_URL`覆盖。

支持 `wav/mp3/m4a/flac/ogg/webm`，默认最大 50 MiB。默认加载原
`whisper-large-v3`，使用 CUDA `float16`。默认`initial_prompt`和热词面向机器狗
园区巡逻场景，覆盖园区、区域、巡逻及四个方向；部署方仍可通过
`ASR_INITIAL_PROMPT`和`ASR_HOTWORDS`覆盖。

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

默认候选意图为`patrol`、`movement`、`find_object`、`grab`和`other`，可通过
`INTENT_CANDIDATES`配置。默认
`INTENT_BACKEND=hybrid`：使用镜像内原 Qwen 模型，模型失败时回退规则。
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
