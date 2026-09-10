# Sandbox Lite

一个镜像、一个容器、三个独立目录和三个独立进程：

| 目录 | 进程 | 端口 | 职责 |
| --- | --- | ---: | --- |
| `services/asr` | `python -m services.asr.main` | 9004 | faster-whisper 语音转文字 |
| `services/intent` | `python -m services.intent.main` | 8011 | 文字语义与意图分类 |
| `services/video` | `python -m services.video.main` | 28500 | Orange WebRTC 信令、YOLO 检测、处理视频输出 |

Supervisor 同时启动并分别守护三个进程。视频服务的 HTTP 路径、鉴权、非
trickle SDP 格式和信令方向兼容
`orange_sdk/mock-video-server`，可以替换原 mock 服务；视频处理由 mock
标记替换为真实 YOLO。

## 模型

构建镜像时直接复制原工程权重，不在线下载：

- `whisper-large-v3`
- `Qwen2.5-0.5B-Instruct`
- `yolov8s-worldv2.pt`
- `box0612.pt`、`toy.pt`、`bottles.pt`

默认从 `../sandbox-demo` 读取；原工程位于其他目录时设置
`ORIGINAL_MODEL_ROOT`。运行时无需模型挂载。

## 启动

视频服务默认接入 Orange Core 创建的外部 Docker 网络 `compose_n6`，
使用原 mock 的 `172.30.0.10:28500`，并添加经
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

检查三个进程：

```bash
curl http://127.0.0.1:9004/health
curl http://127.0.0.1:8011/health
curl http://127.0.0.1:28500/healthz
docker exec sandbox-lite supervisorctl -c /etc/supervisor/conf.d/sandbox.conf status
```

容器需要 NVIDIA Container Toolkit。默认基础镜像为
`nvidia/cuda:13.0.1-devel-ubuntu24.04`；可通过 `BASE_IMAGE` 使用已缓存
的兼容 CUDA 镜像。

## ASR

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
`whisper-large-v3`，使用 CUDA `float16`。

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

## Orange WebRTC + YOLO

完整流程：

```text
Orange SDK 分配 session
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
为目标 Agent 创建 consumer ticket
        │
        ▼
消费端创建 recvonly Offer → Server 返回 Answer → 接收 YOLO 视频
```

Offer 是 Video Server 对手机发起的视频协商，但通过手机主动发出的 HTTP
请求响应返回；手机端不需要 HTTP 服务端点。

### Orange 兼容接口

| 方法 | 路径 | 鉴权 | 功能 |
| --- | --- | --- | --- |
| GET | `/healthz` | 无 | Orange 健康检查 |
| GET | `/debug/v1/sessions` | 无 | 查看会话状态 |
| POST | `/compute/v1/offloading-sessions` | 无 | 分配视频会话及 producer token |
| POST | `/compute/v1/offloading-sessions/{session_id}/consumers` | 无 | 创建 consumer tickets |
| POST | `/video/v1/sessions/{session_id}/source` | Bearer producer token | 创建源 Offer或提交源 Answer |
| POST | `/video/v1/sessions/{session_id}/source/stop` | Bearer producer token | 停止源和该会话所有 PeerConnection |
| POST | `/video/v1/sessions/{session_id}/processed` | Bearer consumer ticket | 提交消费端 Offer并获得 YOLO 视频 Answer |

分配会话：

```http
POST /compute/v1/offloading-sessions
Content-Type: application/json

{
  "request_id": "create-1",
  "agent_id": "agent-b",
  "group_id": "group-ab",
  "workload_type": "video"
}
```

源端第一次请求 Offer：

```http
POST /video/v1/sessions/{session_id}/source
Authorization: Bearer {producer.access_token}
Content-Type: application/json

{"action":"create_offer"}
```

返回的 SDP 使用嵌套格式：

```json
{
  "session_id": "mock-...",
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

只有服务端收到第一帧后才返回 `SOURCE_CONNECTED`；默认等待 12 秒。
随后创建消费者：

```http
POST /compute/v1/offloading-sessions/{session_id}/consumers
Content-Type: application/json

{
  "agent_id": "agent-b",
  "group_id": "group-ab",
  "target_agent_ids": ["agent-a", "agent-c"]
}
```

每个目标都会获得互不相同的 `access_ticket` 和同一个 session 的
`offer_url`。消费端创建 `recvonly` Offer 并请求：

```http
POST /video/v1/sessions/{session_id}/processed
Authorization: Bearer {access_ticket}
Content-Type: application/json

{
  "sdp_offer": {
    "type": "offer",
    "sdp": "v=0\r\n..."
  }
}
```

响应中的 `sdp_answer` 用于消费端的 Remote Description。

### YOLO 扩展接口

- `GET /api/v1/detection/classes`：查询当前检测目标。
- `POST /api/v1/detection/classes`：动态设置检测目标。
- `GET /health`、`GET /api/health`：YOLO 和帧处理详细状态。
- `DELETE /api/v1/webrtc/sessions/{session_id}`：运维用强制关闭会话。

```bash
curl -X POST http://127.0.0.1:28500/api/v1/detection/classes \
  -H 'Content-Type: application/json' \
  -d '{"classes":["person","dog"]}'
```

默认使用原 `yolov8s-worldv2.pt`。可设置
`YOLO_MODEL=box0612`、`YOLO_MODEL=toy` 或
`YOLO_MODEL=bottles` 切换镜像内原定制权重。YOLO 模型进程内只加载一次，
每个 Orange session 使用独立帧管线，多个观看端不会重复推理。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖三个独立服务、原模型默认值、Orange 接口错误码和真实 aiortc
端到端链路（服务端 Offer、源 Answer、首帧、consumer ticket、处理视频、
停止会话）。
