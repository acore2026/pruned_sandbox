from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch

from services.asr.config import AsrSettings
from services.intent.config import IntentSettings
from services.video.config import VideoSettings


class ServiceSettingsTest(TestCase):
    def test_three_services_have_independent_default_ports(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            asr = AsrSettings.from_env()
            intent = IntentSettings.from_env()
            video = VideoSettings.from_env()

        self.assertEqual(9004, asr.port)
        self.assertEqual(8011, intent.port)
        self.assertEqual(28500, video.port)
        self.assertEqual("172.30.0.10", video.public_ip)
        self.assertEqual("/models/asr/whisper-large-v3", asr.model)
        self.assertEqual("hybrid", intent.backend)
        self.assertTrue(video.yolo_enabled)

    def test_video_host_only_ice_and_class_filters(self) -> None:
        with patch.dict(
            os.environ,
            {"WEBRTC_ICE_SERVERS": "", "YOLO_CLASSES": "person,dog"},
            clear=True,
        ):
            settings = VideoSettings.from_env()

        self.assertEqual((), settings.rtc_ice_servers)
        self.assertEqual(("person", "dog"), settings.yolo_classes)
