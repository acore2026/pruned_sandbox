# AR眼镜语音识别与意图接口定义

## 1. 文档范围

本文定义AR眼镜在园区巡逻场景中使用的两类语音识别接口：

1. 巡逻任务发起前，调用独立ASR服务识别任务，并返回用于智能体发现的技能条件；
2. 算力会话运行期间，调用Sandbox内置ASR服务识别动作意图。

两个接口都只返回语音文本和结构化意图，不创建群组、不发送A2A消息，也不向机器狗下发控制指令。眼镜端负责根据返回结果调用H-DISCOVERY或构造后续A2A `TASK`消息。

当前联调阶段允许使用HTTP。服务地址由眼镜端配置，不应在应用代码中写死。

## 2. 接口总览

| 场景 | 服务地址示例 | 接口 | 返回重点 |
| --- | --- | --- | --- |
| 巡逻任务发现 | `http://{sandbox_host}:9004` | `POST /api/v1/transcribe` | `text`、`intent.type=TASK`、`required_skills` |
| 运行期动作识别 | `http://{sandbox_host}:28502` | `POST /v1/audio-control-actions` | `text`、动作`intent` |

两类服务可以使用相同的HTTP路径，但属于不同部署实例，候选意图配置和业务职责相互独立。

## 3. 通用请求格式

### 3.1 HTTP请求

```http
POST /api/v1/transcribe
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | binary | 是 | 音频文件；建议使用`wav`、`mp3`或`m4a` |
| `request_id` | string | 是 | 眼镜端生成的本次识别请求ID；用于响应关联、日志定位和安全重试 |
| `language` | string | 否 | 语言代码，中文使用`zh` |

### 3.2 通用响应字段

| 字段 | 类型 | 必有 | 说明 |
| --- | --- | --- | --- |
| `request_id` | string | 是 | 原样回显请求中的`request_id` |
| `text` | string | 是 | 语音识别得到的原始文本；不因意图未命中而省略 |
| `intent` | object | 是 | 结构化业务意图；字段形式以具体场景为准，见第4、5节 |
| `required_skills` | array[string] | 仅任务发现接口 | H-DISCOVERY 的技能筛选条件；运行期动作接口不返回该字段 |

服务端的转写记录ID、内部会话ID、音频时长和处理耗时只用于服务日志和运维观测，不进入眼镜端业务响应。

## 4. 巡逻任务发现接口

### 4.1 业务语义

该接口用于Sandbox尚未拉起时的首条巡逻任务语音，例如：

> 派机器狗巡逻园区内A区域

服务返回识别文本、巡逻任务意图和发现机器狗所需的技能。眼镜端随后将`text`作为任务描述、将`required_skills`原样传给H-DISCOVERY。

独立ASR服务必须采用场景文档第7.2节规定的最小映射规则：

| 用户语音业务意图 | `intent.type` | `required_skills` | 眼镜端后续处理 |
| --- | --- | --- | --- |
| 巡逻、巡检 | `TASK` | `patrol`、`camera` | 发起H-DISCOVERY；确认候选机器狗后建组并发送巡逻任务 |
| 实时画面、查看现场 | `VIDEO_TASK` | `patrol`、`camera` | 发起H-DISCOVERY；完成建组后申请或使用视频业务 |
| 可疑物识别 | `OBJECT_RECOGNITION` | `camera` | 发起H-DISCOVERY；完成组网后再显式申请算力识别服务 |

该映射在独立ASR服务中完成。ACN、ACF、System Agent和Sandbox运行期ASR均不重新解析首句文本，也不自行从文本推导技能。

### 4.2 成功响应示例

```json
{
  "request_id": "asr-001",
  "text": "派机器狗巡逻园区内A区域",
  "intent": {
    "type": "TASK",
    "parameters": {
      "area": "A"
    }
  },
  "required_skills": [
    "patrol",
    "camera"
  ]
}
```

字段约定：

- `intent.type=TASK`：用户的业务意图是巡逻任务，不表示Sandbox或ASR服务直接执行巡逻；
- `intent.parameters.area`：巡逻区域，从语音中提取；无法确定时省略该字段；
- `required_skills`：直接传给H-DISCOVERY的规范化技能ID。本场景为`patrol`（巡逻）和`camera`（视频能力）。

`intent`描述“用户希望做什么”，`required_skills`描述“发现时必须具备什么能力”，两者不能混用。`area`是巡逻参数，不是技能；`executor`也不属于`required_skills`。

`required_skills`必须与机器狗Agent Profile中发布的技能ID完全一致。若Profile尚未发布`patrol`和`camera`，H-DISCOVERY不会返回该机器狗；此时应统一调整Profile和本接口配置，不能只在眼镜端做中文名称匹配。

### 4.3 其他发现任务响应示例

实时画面或查看现场：

```json
{
  "request_id": "asr-video-001",
  "text": "查看机器狗实时画面。",
  "intent": {
    "type": "VIDEO_TASK",
    "parameters": {}
  },
  "required_skills": [
    "patrol",
    "camera"
  ]
}
```

可疑物识别：

```json
{
  "request_id": "asr-object-001",
  "text": "识别园区内可疑物。",
  "intent": {
    "type": "OBJECT_RECOGNITION",
    "parameters": {}
  },
  "required_skills": [
    "camera"
  ]
}
```

### 4.4 未匹配发现任务

任意语音都应正常完成转写。若文本不是可识别的智能体发现任务，返回文本并标记未命中：

```json
{
  "request_id": "asr-002",
  "text": "今天是周五。",
  "intent": {
    "type": "UNKNOWN",
    "parameters": {}
  },
  "required_skills": []
}
```

此时仍返回HTTP `200`，眼镜端展示`text`，但不调用H-DISCOVERY。

## 5. Sandbox运行期动作意图接口

### 5.1 业务语义

算力会话建立后，眼镜端把“威吓歹徒”“驱逐歹徒”等音频提交给Sandbox。Sandbox同步完成语音转写和意图解析并返回结果。

Sandbox到此结束处理，不调用机器狗接口。眼镜端根据意图自行决定是否构造并发送A2A `TASK`。

### 5.2 动作意图响应示例

“威吓歹徒”和“驱逐歹徒”在当前场景中均归一化为向前移动意图。按场景文档第10.2节，
`executor`、`intent`、`direction`、`matched`和`backend`均需保留。具体的`Scrape`或
`FrontPounce`由AR应用根据业务确认结果填入后续群组A2A `TASK`，不由Sandbox决定。

```json
{
  "request_id": "asr-action-001",
  "text": "威吓歹徒。",
  "intent": {
    "executor": "robot dog",
    "intent": "movement",
    "direction": "forward",
    "matched": true,
    "backend": "qwen"
  }
}
```

“驱逐歹徒”使用相同的结构：

```json
{
  "request_id": "asr-action-002",
  "text": "驱逐歹徒。",
  "intent": {
    "executor": "robot dog",
    "intent": "movement",
    "direction": "forward",
    "matched": true,
    "backend": "qwen"
  }
}
```

动作意图与机器狗设备命令的映射由眼镜端或机器狗业务适配器维护。Sandbox只返回
`movement/direction=forward`，不区分、映射或下发任何具体设备动作。

### 5.4 未匹配动作意图

```json
{
  "request_id": "asr-action-003",
  "text": "今天是周五。",
  "intent": {
    "executor": null,
    "intent": "other",
    "matched": false,
    "backend": "qwen"
  }
}
```

眼镜端应始终显示`text`；只有`intent.matched=true`且意图在本地允许列表内时，才进入后续A2A业务流程。

## 6. 意图字段约定

场景文档对两类ASR使用不同的`intent`结构，眼镜端应按调用的服务地址解析，不应混用。

| 场景 | 字段 | 说明 |
| --- | --- | --- |
| 独立ASR任务发现 | `type` | `TASK`、`VIDEO_TASK`或`OBJECT_RECOGNITION`；未命中时为`UNKNOWN` |
| 独立ASR任务发现 | `parameters.area` | `TASK`时的巡逻区域；未识别时为空对象或省略`area` |
| 运行期动作 | `executor` | 命中时固定为`robot dog`；未命中时为`null` |
| 运行期动作 | `intent` | 命中时为`movement`；未命中时为`other` |
| 运行期动作 | `direction` | 命中时为`forward` |
| 运行期动作 | `matched` | 是否命中动作候选意图 |
| 运行期动作 | `backend` | 实际意图识别后端，例如`qwen` |

接口不返回`control_triggered`、`normalized_action`或机器狗设备动作名。`Scrape`和
`FrontPounce`由眼镜端构造后续A2A `TASK`时确定，不是Sandbox ASR接口返回值。

## 7. 错误响应

只有请求或服务本身失败时才返回非`2xx`；“转写成功但意图未命中”不是错误。

```json
{
  "error": {
    "code": "unsupported-audio-format",
    "message": "unsupported audio format",
    "request_id": "asr-004"
  }
}
```

| HTTP状态码 | 场景 |
| --- | --- |
| `400` | 缺少必填字段或音频内容无效 |
| `413` | 音频文件超过大小限制 |
| `415` | 不支持的音频格式 |
| `500` | 服务内部处理失败 |
| `503` | ASR或意图模型未就绪 |

## 8. 调用示例

### 8.1 独立ASR任务发现

```bash
curl --noproxy '*' -X POST \
  http://{sandbox_host}:9004/api/v1/transcribe \
  -F file=@patrol-area-a.mp3 \
  -F request_id=asr-001 \
  -F language=zh
```

### 8.2 Sandbox运行期动作识别

```bash
curl --noproxy '*' -X POST \
  http://{sandbox_host}:28502/v1/audio-control-actions \
  -F file=@deter-suspect.mp3 \
  -F request_id=asr-action-001 \
  -F language=zh \
  -F 'computing_context={"compute_service_session_id":"css-001","compute_instance_id":"ci-001","binding_ref":"binding-css-001","role":"consumer","agent_id":"glasses"}'
```

## 9. 调用方处理规则

1. 眼镜端收到响应后立即显示`text`；
2. 独立ASR任务仅在`intent.type`为`TASK`、`VIDEO_TASK`或`OBJECT_RECOGNITION`且`required_skills`非空时发起H-DISCOVERY；
3. `TASK`和`VIDEO_TASK`使用`["patrol", "camera"]`发现机器狗；`OBJECT_RECOGNITION`使用`["camera"]`，并在组网后再申请算力识别服务；
4. 运行期动作只有在`intent.matched=true`且`intent.intent=movement`时才构造A2A `TASK`；
5. 重试时沿用同一个`request_id`，新的一次用户录音使用新的`request_id`；
6. Sandbox不会因为任何识别结果直接控制机器狗。

## 10. 当前实现调整项

本文描述的是目标接口契约。现有`pruned_sandbox`代码还需要完成以下调整后才能按本文联调：

- 删除运行期音频接口触发机器狗控制适配器的逻辑；
- 在独立ASR模式实现第4.1节的三类映射：`TASK`/`VIDEO_TASK`返回`["patrol", "camera"]`，`OBJECT_RECOGNITION`返回`["camera"]`；
- 将“威吓歹徒”和“驱逐歹徒”返回为`executor=robot dog`、`intent=movement`、`direction=forward`、`matched=true`和`backend=qwen`；
- 按第4、5节分别返回任务和动作意图未命中结构，同时始终返回转写文本；
- 停止向眼镜端返回`control_triggered`等控制执行状态字段。
- 将当前`session_id`、`task_id`等内部关联字段收敛为对外唯一的`request_id`。
