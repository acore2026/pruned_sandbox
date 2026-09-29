#!/usr/bin/env python3
"""Enter-driven end-to-end demo for the patrol, video and voice-control flow.

The script emulates two terminals: a robot-dog video producer and glasses as a
video/audio consumer. It intentionally prints the command that an upstream
controller would send to the dog; the sandbox audio endpoint itself has no
machine-dog control side effect.
"""

from __future__ import annotations

import argparse
import asyncio
from fractions import Fraction
import json
import os
from pathlib import Path
import time
from typing import Any

import aiohttp
import av
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack


class VideoFileTrack(MediaStreamTrack):
    """Read a local video file and publish it as a WebRTC video track."""

    kind = "video"

    def __init__(self, path: Path, fps: float, loop: bool) -> None:
        super().__init__()
        self.path = path
        self.fps = max(fps, 1.0)
        self.loop = loop
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.frame_number = 0

    async def recv(self) -> av.VideoFrame:
        ok, image = self.capture.read()
        if not ok and self.loop:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self.capture.read()
        if not ok:
            self.stop()
            raise MediaStreamError
        if self.frame_number:
            await asyncio.sleep(1.0 / self.fps)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = self.frame_number
        frame.time_base = Fraction(1, int(round(self.fps)))
        self.frame_number += 1
        return frame

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        super().stop()


class Demo:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.management = args.management_url.rstrip("/")
        self.sandbox = args.sandbox_url.rstrip("/")
        self.asr = args.asr_url.rstrip("/")
        self.run_id = str(time.time_ns())
        self.binding_ref = f"{args.binding_ref}-{self.run_id}"
        self.session_id = f"{args.session_id}-{self.run_id}"
        self.instance_id = f"{args.instance_id}-{self.run_id}"
        self.owner_ref = args.owner_ref
        self.http: aiohttp.ClientSession | None = None
        self.peer_connections: list[RTCPeerConnection] = []
        self.source_track: VideoFileTrack | None = None
        self.output_task: asyncio.Task[None] | None = None
        self.output_writer: cv2.VideoWriter | None = None
        self.output_frames = 0
        self.bound = False

    @property
    def producer_context(self) -> dict[str, str]:
        return {
            "compute_service_session_id": self.session_id,
            "compute_instance_id": self.instance_id,
            "binding_ref": self.binding_ref,
            "role": "producer",
            "agent_id": "robot-dog-demo",
        }

    @property
    def consumer_context(self) -> dict[str, str]:
        return {
            "compute_service_session_id": self.session_id,
            "compute_instance_id": self.instance_id,
            "binding_ref": self.binding_ref,
            "role": "consumer",
            "agent_id": "glasses-demo",
        }

    def management_headers(self) -> dict[str, str]:
        if self.args.management_token:
            return {"Authorization": f"Bearer {self.args.management_token}"}
        return {}

    @staticmethod
    async def wait_for_enter(prompt: str) -> None:
        """Wait for Enter without blocking WebRTC and YOLO tasks."""
        await asyncio.to_thread(input, prompt)

    async def request_json(self, method: str, url: str, **kwargs: Any) -> Any:
        assert self.http is not None
        async with self.http.request(method, url, **kwargs) as response:
            body = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError(f"{method} {url} -> {response.status}: {body}")
            return body

    async def bind(self) -> None:
        configuration = {
            "compute_service_session_id": self.session_id,
            "connection_parameters": {
                "transport": "WEBRTC",
                "media_connections_path": "/v1/media-connections",
                "recognition_target_path_template": "/v1/recognition-targets/{compute_service_session_id}",
            },
            "participants_network_facts": [
                {
                    "role": "consumer",
                    "agent_id": "glasses-demo",
                    "up_session_id": "up-glasses-demo",
                    "pdu_session_id": 1,
                    "ue_ipv4_address": "10.60.0.11",
                    "dnn": "internet",
                    "snssai": "1-010203",
                    "generation": "1",
                },
                {
                    "role": "producer",
                    "agent_id": "robot-dog-demo",
                    "up_session_id": "up-dog-demo",
                    "pdu_session_id": 1,
                    "ue_ipv4_address": "10.60.0.12",
                    "dnn": "internet",
                    "snssai": "1-010203",
                    "generation": "1",
                },
            ],
        }
        payload = {
            "activation_idempotency_key": f"activate-{self.binding_ref}-{self.run_id}",
            "owner_ref": self.owner_ref,
            "binding_ref": self.binding_ref,
            "compute_service_session_id": self.session_id,
            "compute_instance_id": self.instance_id,
            "configuration": configuration,
            "configuration_digest": self.canonical_digest(configuration),
            "expected_binding_absent": True,
        }
        result = await self.request_json(
            "POST",
            f"{self.management}/management/v1/compute-session-bindings:bind",
            headers=self.management_headers(),
            json=payload,
        )
        self.bound = True
        print("绑定结果:", json.dumps(result, ensure_ascii=False))

    async def unbind(self) -> None:
        if not self.bound:
            return
        try:
            result = await self.request_json(
                "POST",
                f"{self.management}/management/v1/compute-session-bindings/{self.binding_ref}:unbind",
                headers=self.management_headers(),
                json={
                    "owner_ref": self.owner_ref,
                    "unbind_idempotency_key": f"unbind-{self.binding_ref}-{self.run_id}",
                    "cause": "e2e-demo-finished",
                },
            )
            print("解除绑定:", json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            print(f"解除绑定失败: {exc}")

    async def send_patrol(self) -> None:
        path = Path(self.args.patrol_audio)
        data = aiohttp.FormData()
        data.add_field("request_id", f"e2e-patrol-{self.run_id}")
        data.add_field("language", self.args.patrol_language)
        data.add_field("file", path.open("rb"), filename=path.name)
        result = await self.request_json("POST", f"{self.asr}/api/v1/transcribe", data=data)
        print("9004 巡逻 ASR:", json.dumps(result, ensure_ascii=False, indent=2))

    async def set_recognition_target(self) -> None:
        result = await self.request_json(
            "PUT",
            f"{self.sandbox}/v1/recognition-targets/{self.session_id}",
            json={
                "request_id": f"e2e-target-robotdog-{self.run_id}",
                "computing_context": self.consumer_context,
                "input": {"type": "TEXT", "text": "Find the robot dog."},
            },
        )
        print("识别目标:", json.dumps(result, ensure_ascii=False))

    async def negotiate(self, role: str) -> RTCPeerConnection:
        pc = RTCPeerConnection()
        self.peer_connections.append(pc)
        if role == "consumer":
            pc.addTransceiver("video", direction="recvonly")
            offer = await pc.createOffer()
        else:
            self.source_track = VideoFileTrack(
                Path(self.args.video), self.args.video_fps, self.args.loop_video
            )
            pc.addTrack(self.source_track)
            offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await self.wait_ice(pc)
        context = self.consumer_context if role == "consumer" else self.producer_context
        result = await self.request_json(
            "POST",
            f"{self.sandbox}/v1/media-connections",
            json={
                "request_id": f"e2e-media-{role}-{self.run_id}",
                "computing_context": context,
                "offer": {"type": "offer", "sdp": pc.localDescription.sdp},
            },
        )
        answer = result["answer"]
        await pc.setRemoteDescription(RTCSessionDescription(**answer))
        print(f"{role} 视频连接已建立")
        return pc

    async def start_consumer(self) -> None:
        consumer_pc = await self.negotiate("consumer")
        ready = asyncio.Event()
        track = next(
            (
                receiver.track
                for receiver in consumer_pc.getReceivers()
                if receiver.track is not None and receiver.track.kind == "video"
            ),
            None,
        )
        if track is None:
            raise RuntimeError("sandbox did not return a processed video track")
        self.output_task = asyncio.create_task(self.consume_video(track, ready))
        await asyncio.wait_for(ready.wait(), timeout=self.args.video_wait_seconds)

    async def consume_video(self, track: MediaStreamTrack, ready: asyncio.Event) -> None:
        output = Path(self.args.output_video)
        while True:
            try:
                frame = await track.recv()
            except (MediaStreamError, asyncio.CancelledError):
                return
            image = frame.to_ndarray(format="bgr24")
            if self.output_writer is None:
                height, width = image.shape[:2]
                if (width, height) != (640, 480):
                    raise RuntimeError(
                        "sandbox output must be fixed at 640x480; "
                        f"received {width}x{height}"
                    )
                output.parent.mkdir(parents=True, exist_ok=True)
                self.output_writer = cv2.VideoWriter(
                    str(output), cv2.VideoWriter_fourcc(*"mp4v"), self.args.output_fps, (width, height)
                )
                print(f"眼镜端视频输出: {output.resolve()}")
            self.output_writer.write(image)
            self.output_frames += 1
            ready.set()

    async def start_source(self) -> None:
        await self.negotiate("producer")
        await self.wait_for_source()
        print(f"机器狗视频源已接入: {self.args.video}")

    async def wait_for_source(self) -> None:
        deadline = time.monotonic() + self.args.video_wait_seconds
        url = f"{self.management}/management/v1/compute-session-bindings/{self.binding_ref}/media-state"
        while time.monotonic() < deadline:
            state = await self.request_json(
                "GET", url, headers=self.management_headers()
            )
            if state.get("producer_connected"):
                return
            await asyncio.sleep(0.25)
        raise RuntimeError(
            "WebRTC producer did not connect or deliver a source frame within "
            f"{self.args.video_wait_seconds:g}s; check VIDEO_PUBLIC_IP, firewall, "
            "and the Tailscale/host network path"
        )

    async def send_action(self, name: str, path: Path) -> None:
        if not path.exists():
            print(f"跳过 {name}: 音频文件不存在 {path}")
            return
        data = aiohttp.FormData()
        data.add_field("request_id", f"e2e-action-{name}-{self.run_id}")
        if self.args.runtime_language:
            data.add_field("language", self.args.runtime_language)
        data.add_field("computing_context", json.dumps(self.consumer_context))
        data.add_field("source", "glasses")
        data.add_field("file", path.open("rb"), filename=path.name)
        result = await self.request_json("POST", f"{self.sandbox}/v1/audio-control-actions", data=data)
        print(f"28502 {name}:", json.dumps(result, ensure_ascii=False, indent=2))
        intent = result.get("intent") or {}
        print("模拟下发机器狗指令:", json.dumps(self.command_from_intent(intent), ensure_ascii=False))

    @staticmethod
    def command_from_intent(intent: dict[str, Any]) -> dict[str, Any]:
        intent_type = intent.get("type") or intent.get("intent") or "UNKNOWN"
        direction = intent.get("direction")
        if direction == "expel":
            return {"command": "EXPEL", "action": "expel"}
        if intent_type in {"movement", "MOVEMENT"} and direction:
            return {"command": "MOVE", "direction": direction}
        return {"command": "NOOP", "reason": "intent not mapped"}

    async def run(self) -> None:
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.args.http_timeout))
        try:
            await self.wait_for_enter("[1/8] 按 Enter：发送巡逻语音到 9004... ")
            await self.send_patrol()
            await self.wait_for_enter("[2/8] 按 Enter：创建 sandbox 测试绑定... ")
            await self.bind()
            await self.wait_for_enter("[3/8] 按 Enter：设置 YOLO 目标 robot dog... ")
            await self.set_recognition_target()
            await self.wait_for_enter("[4/8] 按 Enter：启动眼镜端视频接收... ")
            await self.start_consumer()
            await self.wait_for_enter("[5/8] 按 Enter：启动机器狗模拟视频... ")
            await self.start_source()
            await self.wait_for_enter("[6/8] 按 Enter：发送驱逐语音... ")
            await self.send_action("expel", Path(self.args.expel_audio))
            for index, (name, path) in enumerate(self.args.direction_audio, start=1):
                await self.wait_for_enter(f"[7.{index}/8] 按 Enter：发送 {name} 语音... ")
                await self.send_action(name, Path(path))
            await self.wait_for_enter("[8/8] 按 Enter：结束测试并解除绑定... ")
        finally:
            if self.output_task is not None:
                self.output_task.cancel()
                await asyncio.gather(self.output_task, return_exceptions=True)
            if self.output_writer is not None:
                self.output_writer.release()
            for pc in self.peer_connections:
                await pc.close()
            if self.http is not None:
                await self.unbind()
                await self.http.close()
            print(f"测试结束，共接收处理后视频帧: {self.output_frames}")

    @staticmethod
    async def wait_ice(pc: RTCPeerConnection) -> None:
        if pc.iceGatheringState == "complete":
            return
        event = asyncio.Event()

        @pc.on("icegatheringstatechange")
        def on_state() -> None:
            if pc.iceGatheringState == "complete":
                event.set()

        await asyncio.wait_for(event.wait(), 10)

    @staticmethod
    def canonical_digest(value: Any) -> str:
        import hashlib

        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    audio = root / "test_audio"
    parser.add_argument("--video", default=str(root / "robotdog2-640x480.mp4"))
    parser.add_argument("--patrol-audio", default=str(audio / "generated-asr-en.mp3"))
    parser.add_argument("--expel-audio", default=str(audio / "deter-suspect.mp3"))
    parser.add_argument("--direction-audio", action="append", default=[], metavar="NAME=FILE")
    parser.add_argument("--management-url", default="http://127.0.0.1:28501")
    parser.add_argument("--sandbox-url", default="http://127.0.0.1:28502")
    parser.add_argument("--asr-url", default="http://127.0.0.1:9004")
    parser.add_argument(
        "--management-token",
        default=os.getenv("FREE6GC_COMPUTING_SANDBOX_MANAGEMENT_TOKEN", ""),
    )
    parser.add_argument("--patrol-language", default="en")
    parser.add_argument("--runtime-language", default="")
    parser.add_argument("--binding-ref", default="e2e-robotdog-binding")
    parser.add_argument("--session-id", default="e2e-robotdog-session")
    parser.add_argument("--instance-id", default="e2e-robotdog-instance")
    parser.add_argument("--owner-ref", default="e2e-demo")
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--output-fps", type=float, default=15.0)
    parser.add_argument("--output-video", default="artifacts/robotdog-annotated.mp4")
    parser.add_argument("--video-wait-seconds", type=float, default=30.0)
    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument("--loop-video", action="store_true", default=True)
    args = parser.parse_args()
    args.direction_audio = [parse_audio_item(item) for item in args.direction_audio]
    if not args.direction_audio:
        args.direction_audio = [
            ("forward", str(audio / "move-forward.mp3")),
            ("backward", str(audio / "move-backward.mp3")),
            ("left", str(audio / "turn-left.mp3")),
            ("right", str(audio / "turn-right.mp3")),
        ]
    return args


def parse_audio_item(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("audio mapping must be NAME=FILE")
    return name.strip(), path.strip()


if __name__ == "__main__":
    try:
        asyncio.run(Demo(parse_args()).run())
    except KeyboardInterrupt:
        print("测试已中断")
