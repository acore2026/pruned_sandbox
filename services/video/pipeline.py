from __future__ import annotations

import asyncio
import fractions
import logging
import time
from typing import Any

import av
import cv2
import numpy as np
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from .config import VideoSettings
from .detector import YoloDetector


LOGGER = logging.getLogger("sandbox.video.pipeline")


class FramePipeline:
    """上游持续收帧，容量为 1 的队列保证 YOLO 不积压旧帧。"""

    def __init__(self, settings: VideoSettings, detector: YoloDetector) -> None:
        self.settings = settings
        self.detector = detector
        self._latest_image = self._placeholder("Waiting for WebRTC source")
        self._source_task: asyncio.Task[None] | None = None
        self._source_track: MediaStreamTrack | None = None
        self._source_id: str | None = None
        self.first_frame = asyncio.Event()
        self.processed_frame = asyncio.Event()
        self.received_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.last_frame_at_ms = 0
        self.last_error: str | None = None

    async def attach_source(self, track: MediaStreamTrack, source_id: str) -> None:
        await self._cancel_source_task()
        self._source_track = track
        self._source_id = source_id
        self.first_frame.clear()
        self.processed_frame.clear()
        self.last_error = None
        self._source_task = asyncio.create_task(
            self._run_source(track, source_id),
            name=f"video-source:{source_id}",
        )
        LOGGER.info("已接入视频源 source=%s", source_id)

    async def clear_source(self, source_id: str | None = None) -> bool:
        if source_id is not None and self._source_id != source_id:
            return False
        await self._cancel_source_task()
        self._source_track = None
        self._source_id = None
        self.first_frame.clear()
        self.processed_frame.clear()
        self._latest_image = self._placeholder("Waiting for WebRTC source")
        return True

    async def close(self) -> None:
        await self.clear_source()

    async def _cancel_source_task(self) -> None:
        task = self._source_task
        self._source_task = None
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run_source(self, track: MediaStreamTrack, source_id: str) -> None:
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=1)
        reader = asyncio.create_task(self._read_frames(track, queue), name=f"video-reader:{source_id}")
        processor = asyncio.create_task(self._process_frames(queue, source_id), name=f"video-yolo:{source_id}")
        try:
            await reader
        except asyncio.CancelledError:
            raise
        except MediaStreamError:
            LOGGER.info("视频源结束 source=%s", source_id)
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.exception("读取 WebRTC 视频源失败 source=%s", source_id)
        finally:
            for task in (reader, processor):
                if not task.done():
                    task.cancel()
            await asyncio.gather(reader, processor, return_exceptions=True)
            if self._source_id == source_id:
                self._source_track = None
                self._source_id = None
                self._latest_image = self._placeholder("WebRTC source disconnected")

    async def _read_frames(
        self,
        track: MediaStreamTrack,
        queue: asyncio.Queue[np.ndarray],
    ) -> None:
        while True:
            frame = await track.recv()
            self.received_frames += 1
            self.first_frame.set()
            image = frame.to_ndarray(format="bgr24")
            if queue.full():
                queue.get_nowait()
                self.dropped_frames += 1
            queue.put_nowait(image)

    async def _process_frames(
        self,
        queue: asyncio.Queue[np.ndarray],
        source_id: str,
    ) -> None:
        while True:
            image = await queue.get()
            try:
                annotated, _detections = await asyncio.to_thread(self.detector.process, image)
            except Exception as exc:
                self.last_error = str(exc)
                annotated = image
                LOGGER.exception("YOLO 推理失败，当前帧按原图输出")
            if self._source_id != source_id:
                return
            self._latest_image = annotated
            self.processed_frames += 1
            self.last_frame_at_ms = int(time.time() * 1000)
            self.processed_frame.set()

    def latest_image(self) -> np.ndarray:
        return self._latest_image

    def new_output_track(self) -> "OutputVideoTrack":
        return OutputVideoTrack(self, self.settings.video_fps)

    def health(self) -> dict[str, Any]:
        return {
            "sourceConnected": self._source_track is not None,
            "sourceId": self._source_id,
            "receivedFrames": self.received_frames,
            "processedFrames": self.processed_frames,
            "droppedFrames": self.dropped_frames,
            "lastFrameAtMs": self.last_frame_at_ms,
            "lastError": self.last_error,
        }

    def _placeholder(self, text: str) -> np.ndarray:
        canvas = np.zeros((self.settings.video_height, self.settings.video_width, 3), dtype=np.uint8)
        canvas[:, :, :] = (18, 18, 18)
        cv2.putText(
            canvas,
            text,
            (max(24, self.settings.video_width // 12), self.settings.video_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.7, self.settings.video_width / 1400),
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        return canvas


class OutputVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, pipeline: FramePipeline, fps: float) -> None:
        super().__init__()
        self.pipeline = pipeline
        self._interval = 1.0 / max(fps, 1.0)
        self._timestamp_step = max(1, round(90000 / max(fps, 1.0)))
        self._timestamp = 0
        self._next_frame_at = 0.0

    async def recv(self) -> av.VideoFrame:
        now = time.perf_counter()
        if self._next_frame_at == 0.0:
            self._next_frame_at = now
        else:
            self._next_frame_at += self._interval
            delay = self._next_frame_at - now
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -(self._interval * 2):
                self._next_frame_at = time.perf_counter()
        self._timestamp += self._timestamp_step
        frame = av.VideoFrame.from_ndarray(self.pipeline.latest_image(), format="bgr24")
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, 90000)
        return frame
