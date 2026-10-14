# Sandbox Lite

## 1. 快速开始

### 1.1 准备配置

```bash
cd /home/aicor/pruned_sandbox
test -f sandbox.env || cp sandbox.env.example sandbox.env
tailscale ip -4
```

将`sandbox.env`中的`VIDEO_PUBLIC_IP`改为本机联调地址；使用Tailscale时可通过
`tailscale ip -4`查询。

### 1.2 使用容器启动（推荐）

如果运行环境是 WSL2，在 `sandbox.env` 中加入以下配置；原生 Linux 不需要：

```dotenv
COMPOSE_FILE=docker-compose.yml:docker-compose.wsl.yml
```

该配置显式声明 WSL2 的 GPU 设备 `/dev/dxg`，让 Docker 保留设备访问权限，
避免 `systemctl daemon-reload` 后容器无法访问 GPU。

```bash
cd /home/aicor/pruned_sandbox
set -a
. ./sandbox.env
set +a
bash scripts/check_model_sources.sh
docker compose --env-file sandbox.env up -d --build
```

首次构建完成后，再次启动可省略`--build`：

```bash
cd /home/aicor/pruned_sandbox
docker compose --env-file sandbox.env up -d
```

### 1.3 查看状态和日志

```bash
cd /home/aicor/pruned_sandbox
docker compose --env-file sandbox.env ps
docker compose --env-file sandbox.env logs -f sandbox
```

宿主机滚动日志：

```bash
cd /home/aicor/pruned_sandbox
tail -f logs/asr.log logs/intent.log logs/sandbox.log
```

### 1.4 检查接口

```bash
cd /home/aicor/pruned_sandbox
curl --noproxy '*' http://127.0.0.1:28501/healthz
curl --noproxy '*' http://127.0.0.1:28502/healthz
curl --noproxy '*' http://127.0.0.1:9004/health
docker exec sandbox-lite curl --noproxy '*' -fsS http://127.0.0.1:9005/health
docker exec sandbox-lite curl --noproxy '*' -fsS http://127.0.0.1:8011/health
```

### 1.5 测试语音识别

```bash
cd /home/aicor/pruned_sandbox
curl --noproxy '*' -X POST http://127.0.0.1:9004/api/v1/transcribe \
  -F file=@test_audio/generated-asr-en.mp3 \
  -F request_id=asr-001
```

```bash
cd /home/aicor/pruned_sandbox
curl --noproxy '*' -X POST http://127.0.0.1:9004/api/v1/transcribe \
  -F file=@test_audio/turn-left.mp3 \
  -F request_id=asr-001
```

生成英文测试音频（通过本机 7899 代理）：

```bash
curl --proxy http://127.0.0.1:7899 --fail --silent --show-error \
  --output test_audio/generated-asr-en.mp3 \
  'https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=en&q=Please%20patrol%20area%20B%20and%20locate%20the%20red%20doll.'
```

预期转写文本为 `Please patrol area B and locate the red doll.`。

实际返回：

```json
{"request_id":"asr-generated-en-002","text":"Please patrol area B and locate the red doll.","intent":{"type":"UNKNOWN","parameters":{}},"required_skills":[]}
```

### 1.6 Enter 全流程测试

```bash
cd /home/aicor/pruned_sandbox
.venv/bin/python scripts/e2e_upstream_demo.py \
  --video robotdog2-640x480.mp4 \
  --direction-audio left=test_audio/turn-left.mp3
```

按 Enter 依次执行巡逻、绑定、设置 `sword` 检测目标、双向 WebRTC、驱逐和左移，
最后解除绑定。第 5 步启动机器狗视频后等待 10～15 秒，让 YOLO 产生检测框；输出保存到
`artifacts/robotdog-annotated.mp4`（640×480）。

测试光剑视频时，先准备 640×480 的输入，再执行自动全流程测试：

```bash
ffmpeg -i sword.MOV -vf 'scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,fps=15' \
  -an -c:v libx264 -pix_fmt yuv420p artifacts/sword-640x480.mp4
.venv/bin/python scripts/e2e_upstream_demo.py \
  --video artifacts/sword-640x480.mp4 --recognition-target '寻找红色光剑' \
  --auto --observe-seconds 45 --output-fps 30 --direction-audio left=test_audio/turn-left.mp3 \
  --output-video artifacts/sword-webrtc-annotated.mp4
```

`--auto` 模式会记录失败的运行期语音步骤并继续后续测试，结束时返回失败状态，
方便保留视频测试结果和定位语音接口问题。测试只解除本次创建的独立绑定。

### 1.7 停止容器

```bash
cd /home/aicor/pruned_sandbox
docker compose --env-file sandbox.env down
```

### 1.8 本地运行（不使用容器）

首次准备虚拟环境：

```bash
cd /home/aicor/pruned_sandbox
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

使用真实模型启动：

```bash
cd /home/aicor/pruned_sandbox
SANDBOX_REAL_MODELS=true ./scripts/run_local.sh
```

按`Ctrl-C`统一停止服务。无GPU的Mock模式执行：

```bash
cd /home/aicor/pruned_sandbox
./scripts/run_local.sh
```

## 2. 运行架构

服务端口：

| 地址 | 用途 |
| --- | --- |
| `http://127.0.0.1:28501` | CMF管理面 |
| `http://127.0.0.1:28502` | N6用户面 |
| `http://{host}:9004` | 对外独立任务 ASR（眼镜首句语音） |
| `http://127.0.0.1:9005` | 内部运行期 ASR（仅Sandbox调用） |
| `http://127.0.0.1:8011` | Intent内部服务 |

容器在运行期以只读方式挂载宿主机的Whisper、Qwen和YOLO权重；模型不复制进镜像。默认目录如下：
容器内 ASR 保持 `/models/asr/whisper-large-v3` 这一稳定挂载点，实际内容来自宿主机的多语言 `whisper-large-v3` 目录。

```text
/home/aicor/pruned_sandbox/models/
├── whisper-models/whisper-large-v3/model.bin
├── semantic-models/Qwen/Qwen2.5-0.5B-Instruct/model.safetensors
└── yolo-models/sword.pt
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

## 3. 算网标准接口

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

## 4. 模型

容器启动时从当前仓库`models`目录只读挂载权重，不在线下载，也不将权重写入镜像层：

- `whisper-large-v3`（支持中文和英文）
- `Qwen2.5-0.5B-Instruct`
- 自训练的 `sword.pt`，类别为 `sword`；使用权重中已有的类别和文本特征，无需额外 CLIP 权重。

默认模型源目录由`sandbox.env.example`中的`ASR_MODEL_SOURCE`、`INTENT_MODEL_SOURCE`、
`YOLO_MODEL_SOURCE`配置。这三个模型目录是容器启动前的必要条件；
`bash scripts/check_model_sources.sh`
可检查它们是否完整。

## 5. 运行配置说明

容器使用宿主机网络，上下游通过`VIDEO_PUBLIC_IP`访问`28501`、`28502`和`9004`。
容器需要Docker Compose、NVIDIA Container Toolkit及可用GPU。默认基础镜像为
`nvidia/cuda:13.0.1-runtime-ubuntu24.04`；它不包含CUDA编译工具，可显著减小镜像。
可通过`BASE_IMAGE`覆盖为兼容的本地镜像。

WSL2 下，如果语音接口返回 `CUDA driver version is insufficient for CUDA runtime version`，
先对比宿主机和容器内的 `nvidia-smi`。若宿主机正常、容器报
`GPU access blocked by the operating system`，应检查容器 GPU 设备权限。
NVIDIA runtime hook 注入的设备权限可能在 systemd 重载后丢失，参见
[NVIDIA 故障排查文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#containers-losing-access-to-gpus-with-error-failed-to-initialize-nvml-unknown-error)。
启用上面的 WSL2 配置后，重建容器并重新验证真实转写（会中断当前会话）：

```bash
docker compose --env-file sandbox.env up -d --force-recreate --no-build sandbox
docker exec sandbox-lite nvidia-smi
curl --noproxy '*' -X POST http://127.0.0.1:9004/api/v1/transcribe \
  -F file=@test_audio/generated-asr-en.mp3 \
  -F request_id=asr-gpu-check
```

ASR 模型在首次转写时才加载；启动后的 `/health` 成功不能代替真实转写验证。

日志同时输出到终端并滚动保存到`logs/asr.log`、`logs/intent.log`和
`logs/sandbox.log`。默认单文件上限10 MiB、保留5份历史记录，可通过
`LOG_DIR`、`LOG_MAX_BYTES`和`LOG_BACKUP_COUNT`调整。

本地轻量模式关闭ASR和YOLO推理，并使用规则意图分类，适合接口Mock；真实模型模式
会从当前仓库`models/`目录加载Whisper和Qwen。具体命令见顶部“快速开始”。

## 6. ASR

ASR是独立辅助服务，不属于十个Sandbox标准接口。对外接口为`POST /api/v1/transcribe`，
使用`multipart/form-data`提交`file`、`request_id`和可选`language`：

```bash
cd /home/aicor/pruned_sandbox
curl --noproxy '*' -X POST http://127.0.0.1:9004/api/v1/transcribe \
  -F file=@test_audio/generated-asr-en.mp3 \
  -F request_id=asr-001 \
  -F language=en
```

响应只包含`request_id`、`text`和`intent`；任务发现实例额外返回`required_skills`。
同一个ASR进程通过监听端口区分两类请求：`9004`的`discovery`用于Sandbox拉起前的首句任务，
严格按场景文档映射“巡逻、巡检”为`TASK`和`["patrol", "camera"]`，“实时画面、
查看现场”为`VIDEO_TASK`和相同技能，“可疑物识别”为`OBJECT_RECOGNITION`和
`["camera"]`。仅本机可访问的`9005`为`runtime`，用于已建立算力会话后的Sandbox；“威吓歹徒”和“驱逐歹徒”
均返回`executor=robot dog`、`type=movement`，威吓的`direction=deter`，驱逐的`direction=expel`。任意音频都返回
转写文本；任务发现未命中时为`intent.type=UNKNOWN`，运行期未命中时为
`intent.matched=false`。

意图识别优先使用规则：扫描整句话，任一关键词命中即可触发，无需完整句型。
英文按完整单词匹配，中文按包含关系匹配；响应中的`text`保留实际转写。
例如`detered`、`deterred the suspect`返回`direction=deter`；`expelled the suspect`
返回`expel`；`move forwards/backwards`和`turn write/lift`分别匹配对应移动方向。
`go straight`、`get forward`返回`forward`；`deters the suspect`和`deters a suspect`返回`deter`。
明确别名支持句中任意位置；相似拼写仍仅在原有命令模板中进行唯一的一次编辑匹配，
不对目标物名称或区域参数进行模糊纠错。明确否定和多个不同方向/防御动作返回`matched=false`；
转述不再拦截，跨意图按既有优先级选择，例如`patrol area A and deter the suspect`返回巡逻。
接口映射保持原样：巡逻仅在9004映射为TASK，运行期音频仅映射移动和威吓/驱逐。
未命中的移动、威吓和驱逐文本不会接受Qwen猜测的动作结果。
全部现行规则、命令别名和拒绝条件见[意图识别规则清单](意图识别规则清单.md)。

`POST /v1/audio-control-actions`仅为已绑定会话保留兼容入口；它校验
`computing_context`后调用本机`9005`，返回文本和运行期动作意图，不创建动作任务，也不控制机器狗。
省略`language`时自动检测中文或英文；传入`language=en`或`language=zh`时分别强制使用英文或中文。

支持 `wav/mp3/m4a/flac/ogg/webm`，默认最大 50 MiB。默认加载多语言
`whisper-large-v3`，使用 CUDA `float16`。默认自动检测中文或英文，也可以通过请求中的
`language=zh`或`language=en`指定语言。可通过`ASR_INITIAL_PROMPT`、`ASR_HOTWORDS`、
`ASR_DISCOVERY_HOST`、`ASR_DISCOVERY_PORT`、`ASR_RUNTIME_HOST`和`ASR_RUNTIME_PORT`配置监听地址。详细交接契约见
`AR眼镜语音识别与意图接口定义.md`。

ASR 启动时自动加载模型，用内存中的模拟音频完成一次 Whisper 编码解码和 VAD 初始化，
预热完成后才开放 `9004/9005`。两个端口共享模型，每次进程启动只预热一次。
`/health` 中的 `ready=true`、`warmedUp=true` 表示已完成预热，`warmupMs` 包含加载和预热耗时。
预热不会生成业务转写或覆盖 `latestTranscript`；关闭 ASR 时跳过。

Sandbox 启动时用与输入视频相同尺寸的空白图执行一次 YOLO 推理，提前触发 Ultralytics
已有的内置预热，完成后开放用户接口。`28502/health` 的 `yolo.warmedUp` 和
`yolo.warmupMs` 可用于检查；模拟检测结果不计入业务检测。关闭 YOLO 时跳过。
模型加载或预热失败时启动失败，由 Supervisor 按现有配置重试。
Intent 服务使用 Qwen/hybrid 时，也会在启动时加载 Qwen 并完成一次模拟生成；
`8011/health` 的 `warmedUp` 标记完成状态。规则模式跳过；hybrid 的 Qwen 预热失败时
保留规则回退，并在健康接口的 `lastError` 中报告错误。

## 7. 意图分类

接口：

- `GET /health`
- `POST /api/v1/intent`
- `POST /api/v1/semantic/route`（兼容原入口）

```bash
cd /home/aicor/pruned_sandbox
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

## 8. Orange兼容WebRTC接口

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

### 8.1 Orange 媒体接口

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

### 8.2 H.264 与 MTU

- B→Server 只协商 H.264 High/Constrained High（`64001f`/`640c1f`，
  `packetization-mode=1`），并校验 Answer，禁止静默回退。
- Server→消费端优先 H.264 Baseline，与 aiortc 实际编码能力一致。
- 与当前 mock 一样，H.264 FU-A 的 RTP payload 上限默认是 **1150 bytes**，
  可通过 `VIDEO_H264_RTP_PAYLOAD_BYTES` 在 `1100..1150` 内调整。这不是网卡 MTU；
  它是为 **1280-byte Android TUN MTU** 的 RTP 扩展、SRTP、UDP 和 IP 头预留
  空间后的媒体 payload 上限。
- 上下行连接都会在必要时主动请求关键帧，减少启动关键帧丢失导致的首帧等待。

处理流默认使用横屏 `640x480@30fps`、VP8 250 kbps 编码上限。上游必须发送这个分辨率，服务不会自动缩放或
letterbox；YOLO 也以 `640x480`（宽×高）进行推理，输出视频保持相同分辨率。

### 8.3 YOLO 扩展接口

- `GET /api/v1/detection/classes`：查询当前检测目标。
- `POST /api/v1/detection/classes`：动态设置检测目标。
- `GET /health`、`GET /api/health`：YOLO 和帧处理详细状态。

```bash
cd /home/aicor/pruned_sandbox
curl -X POST http://127.0.0.1:28502/api/v1/detection/classes \
  -H 'Content-Type: application/json' \
  -d '{"classes":["sword"]}'
```

默认使用自训练的 `sword.pt`（容器路径 `/models/yolo/sword.pt`）。YOLO 模型进程内只加载一次，
每个核心网 session ID 使用独立帧管线，多个观看端不会重复推理。未设置会话识别目标时，
默认类别为 `sword`；上游设置 `PUT /v1/recognition-targets/{compute_service_session_id}`
后，以会话目标覆盖默认类别。

光剑、剑、宝剑、长剑、玩具剑、激光剑及 `sword`、`swords`、`lightsaber`、
`light saber`、`light-saber`、`laser sword`、`toy sword` 等名称统一映射为 `sword`。
支持大小写、空格和常见颜色修饰词，例如“寻找红色光剑”对应模型类别 `sword`；
识别目标接口保留用户目标名称，检测框及检测结果使用模型标签 `sword`。
该模型只有 `sword` 类别，设置其他目标不会新增模型可识别类别。

### 高时延视频链路

Python WebRTC 的 consent 响应窗口为 15 秒，探测独立按 4–6 秒间隔发送，保留
30 秒无有效响应时失联判定。此策略只作用于本服务的 PeerConnection，不修改
全局 STUN 重传常量；实现与 Orange Python SDK 的 `rtc_policy.py` 同步维护。
VP8 回传编码器上限为 250 kbps（H264 受库最低 500 kbps 限制），REMB 仍可下调码率，无法提高到上限以上。
H.264/VP8 RTP payload 均限制为 1150 字节以适配 1280 字节 TUN。
当 `VIDEO_PUBLIC_IP` 是已采集到的本机 ICE 地址时，SDP 仅公布该地址的 host 候选；
保留 relay/srflx 候选。避免公布 Docker/代理虚拟网卡，导致 Android 在 ICE 协商中
追加无用 VPN 路由并重建 TUN。STUN 认证直接校验原始数据，兼容 libwebrtc 扩展字段。

## 9. 测试

```bash
cd /home/aicor/pruned_sandbox
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖三个独立服务、原模型默认值、无会话管理接口，以及真实 aiortc
端到端链路（消费端先建链、占位帧、H.264 High 源 Answer、YOLO 帧无重协商
切换、RTP payload 限制和停止媒体）。

回传 RTP 按 350 kbps 的含封装预算逐包发送；RTCP 和 STUN 不经过视频限速，
避免一个关键帧的全部 RTP 包同时进入核心网队列。检测分辨率仍为 640×480。

回传视频的 RTP 时间戳按单调时钟实际经过时间生成。发送阻塞后取最新处理图像，
时间戳跳到当前时间，避免用固定帧步长伪装成较慢的媒体时间轴。
VP8 内部码率控制将单帧时间间隔上限设为 100 ms，避免停顿后生成过大单帧；
RTP 仍使用实际时间戳。`rtc_video_encoded` 日志记录编码大小及耗时。
连续 3 秒没有更新处理图像时，回传改为 `Video source stalled` 提示，
`source_frame_available` 返回 false；恢复新帧后自动恢复图像，避免摄像头断开时
继续把最后一帧作为实时画面发送。ICE 连接保持，摄像头 USB 重连本身仍需 producer 重新打开设备。

## 视频码率对照与处理指标

`sandbox.env` 新增 `VIDEO_BITRATE_BPS`（默认 0，不施加应用上限；显式正数范围 250000～1500000），`VIDEO_WIRE_BITRATE_BPS` 默认 0。编码质量与发包预算独立，修改后需重启对应服务生效。每 5 秒的 `pipeline_metrics` 记录帧计数、丢帧及最近 200 帧队列等待/处理 P95；DEBUG 的 `video_output` 记录结果版本与输出帧间隔。仍为容量 1 队列、只发新帧、上限 20 fps。

当前正式默认：Python 上传 bitrate_kbps=0，不安装应用编码限幅、不写 SDP b=AS；两段 pacing 默认 0，不安装自定义发包限速。保留 WebRTC 拥塞控制和编码器固有边界。无需演示时调参；部署时若已有正数环境覆盖，须清理或设为 0。
