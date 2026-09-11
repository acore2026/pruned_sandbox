# Sandbox Lite

## 快速开始

首次创建本地Python虚拟环境并安装依赖：

```bash
cd /home/aicore/pruned_sandbox
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

使用已经下载到`models/`目录的真实Whisper和Qwen模型启动全部服务：

```bash
cd /home/aicore/pruned_sandbox
SANDBOX_REAL_MODELS=true ./scripts/run_local.sh
```

脚本会自动为本地虚拟环境补充CTranslate2需要的CUDA动态库路径。启动端口为：

| 地址 | 用途 |
| --- | --- |
| `http://127.0.0.1:28501` | CMF管理面 |
| `http://127.0.0.1:28502` | N6用户面 |
| `http://127.0.0.1:9004` | ASR语音转文字 |
| `http://127.0.0.1:8011` | Intent内部服务 |

检查服务状态：

```bash
curl --noproxy '*' http://127.0.0.1:28501/healthz
curl --noproxy '*' http://127.0.0.1:28502/healthz
curl --noproxy '*' http://127.0.0.1:9004/health
curl --noproxy '*' http://127.0.0.1:8011/health
```

Whisper采用首次请求时加载。发送真实音频即可同时完成模型加载和转写验证：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:9004/api/v1/transcribe \
  -F file=@test_audio/patrol-area-a.mp3 \
  -F session_id=patrol-test \
  -F task_id=asr-001 \
  -F source=glasses \
  -F language=zh
```

按`Ctrl-C`会统一停止三个本地进程。没有GPU或只想验证接口时使用轻量模式：

```bash
./scripts/run_local.sh
```

轻量模式关闭ASR和YOLO推理，并使用规则意图分类；HTTP接口仍可用于Mock联调。

### 使用Docker容器启动

容器启动会把Whisper、Qwen和YOLO权重直接复制进镜像。默认从当前仓库读取
Whisper和Qwen，从旧`compute`工程读取唯一使用的通用YOLO权重：

```text
/home/aicore/
├── pruned_sandbox/models/
│   ├── whisper-models/whisper-large-v3/model.bin
│   └── semantic-models/Qwen/Qwen2.5-0.5B-Instruct/model.safetensors
└── compute/yolo/assets/models/yolov8s-worldv2.pt
```

复制环境变量模板。使用上述默认目录时无需修改模型配置：

```bash
cd /home/aicore/pruned_sandbox
cp .env.example .env
```

模型位于其他目录时，在`.env`中分别配置其来源，例如：

```dotenv
ASR_MODEL_SOURCE=/data/models/whisper-large-v3
INTENT_MODEL_SOURCE=/data/models/Qwen2.5-0.5B-Instruct
YOLO_MODEL_SOURCE=/data/models/yolo
FREE6GC_COMPUTING_SANDBOX_MANAGEMENT_TOKEN=replace-with-management-token
```

Sandbox默认加入Orange Core创建的外部N6网络`compose_n6`。先确认该网络存在，
再检查权重、构建并启动容器：

```bash
docker network inspect compose_n6
bash scripts/check_model_sources.sh
docker compose build
docker compose up -d
```

如果`.env`修改了三个模型源变量，执行检查脚本时需将它们导出到当前Shell；
`docker compose`会自动读取`.env`。可以执行：

```bash
set -a
. ./.env
set +a
bash scripts/check_model_sources.sh
docker compose build
docker compose up -d
```

检查容器、三个进程和对外端口：

```bash
docker compose ps
docker compose logs --tail=100 sandbox
docker exec sandbox-lite supervisorctl -c /etc/supervisor/conf.d/sandbox.conf status
curl --noproxy '*' http://127.0.0.1:28501/healthz
curl --noproxy '*' http://127.0.0.1:28502/healthz
curl --noproxy '*' http://127.0.0.1:9004/health
```

容器内的Intent只监听`127.0.0.1:8011`，不会发布到宿主机。需要检查时使用：

```bash
docker exec sandbox-lite curl --noproxy '*' -fsS http://127.0.0.1:8011/health
```

停止并删除容器（模型仍保留在镜像和宿主机模型源目录中）：

```bash
docker compose down
```

容器模式需要Docker Compose、NVIDIA Container Toolkit、可用NVIDIA GPU，以及
预先创建的`compose_n6`网络。`box0612.pt`、`toy.pt`和`bottles.pt`不再使用，
不需要复制或下载。

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

构建镜像时直接复制原工程权重，不在线下载：

- `whisper-large-v3`
- `Qwen2.5-0.5B-Instruct`
- `yolov8s-worldv2.pt`

默认模型源目录由`.env.example`中的`ASR_MODEL_SOURCE`、`INTENT_MODEL_SOURCE`
和`YOLO_MODEL_SOURCE`配置。运行时无需模型挂载。

## 启动

Sandbox服务默认接入 Orange Core 创建的外部 Docker 网络 `compose_n6`，
用户面使用 `172.30.0.10:28502`，并添加经
`172.30.0.2` 到 `10.60.0.0/16`、`10.61.0.0/16` 的路由。
先启动 Orange Core 并确认网络存在：

```bash
docker network inspect compose_n6
cp .env.example .env
bash scripts/check_model_sources.sh
docker compose build
docker compose up -d
```

旧的 `mock-video-server` 需要停止，避免占用同一 N6 地址。

检查Sandbox、ASR和三个进程：

```bash
curl http://127.0.0.1:28501/healthz
curl http://127.0.0.1:28502/healthz
curl http://127.0.0.1:9004/health
docker exec sandbox-lite supervisorctl -c /etc/supervisor/conf.d/sandbox.conf status
docker exec sandbox-lite curl -fsS http://127.0.0.1:8011/health
```

容器需要 NVIDIA Container Toolkit。默认基础镜像为
`nvidia/cuda:13.0.1-devel-ubuntu24.04`；可通过 `BASE_IMAGE` 使用已缓存
的兼容 CUDA 镜像。

### 本地Python环境

首次安装和后续进入环境：

```bash
cd /home/aicore/pruned_sandbox
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

本地无模型或GPU时可使用规则意图并关闭ASR、YOLO，分别在三个终端启动：

```bash
ASR_ENABLED=false .venv/bin/python -m services.asr.main
INTENT_BACKEND=rules .venv/bin/python -m services.intent.main
YOLO_ENABLED=false .venv/bin/python -m services.sandbox.main
```

也可以使用一个命令启动并统一停止这三个本地进程（默认即采用上述轻量配置）：

```bash
./scripts/run_local.sh
```

轻量模式会启动`28501`、`28502`、`9004`和内部`8011`，但ASR推理默认关闭。使用真实
Whisper时设置`ASR_ENABLED=true`及本地模型路径，例如：

```bash
ASR_ENABLED=true ASR_MODEL=/path/to/whisper-large-v3 ./scripts/run_local.sh
```

运行Mock E2E与全部单元测试：

```bash
python -m unittest discover -s tests -v
```

## ASR

ASR是独立辅助服务，不属于十个Sandbox标准接口。眼镜录音后直接向`9004`
上传音频，取得`TEXT`后再根据业务状态调用核心网或Sandbox；Sandbox本身不调用
ASR。该流程只参考原`compute`项目，实现在当前仓库中保持独立。

接口：

- `GET /health`
- `POST /api/v1/transcribe`

转写使用 `multipart/form-data` 上传音频文件：

```bash
curl http://127.0.0.1:9004/api/v1/transcribe \
  -F file=@speech.wav \
  -F session_id=demo-room \
  -F task_id=task-001 \
  -F source=glasses \
  -F language=zh
```

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
curl http://127.0.0.1:8011/api/v1/intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"帮我找黄色的狗"}'
```

意图固定为 `find_object`、`movement`、`grab`、`other`。默认
`INTENT_BACKEND=hybrid`：使用镜像内原 Qwen 模型，模型失败时回退规则。

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
    "video_server_ip": "172.30.0.10",
    "source_start_url": "http://172.30.0.10:28502/video/v1/sessions/{session_id}/source",
    "source_stop_url": "http://172.30.0.10:28502/video/v1/sessions/{session_id}/source/stop"
  },
  "processed_stream": {
    "video_server_ip": "172.30.0.10",
    "offer_url": "http://172.30.0.10:28502/video/v1/sessions/{session_id}/processed",
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
curl -X POST http://127.0.0.1:28502/api/v1/detection/classes \
  -H 'Content-Type: application/json' \
  -d '{"classes":["person","dog"]}'
```

默认且仅使用原通用 `yolov8s-worldv2.pt`。YOLO 模型进程内只加载一次，
每个核心网 session ID 使用独立帧管线，多个观看端不会重复推理。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖三个独立服务、原模型默认值、无会话管理接口，以及真实 aiortc
端到端链路（消费端先建链、占位帧、H.264 High 源 Answer、YOLO 帧无重协商
切换、RTP payload 限制和停止媒体）。
