from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]


class ProcessLayoutTest(TestCase):
    def test_three_processes_have_independent_entrypoints(self) -> None:
        for service in ("asr", "intent", "sandbox"):
            service_dir = ROOT / "services" / service
            self.assertTrue((service_dir / "main.py").is_file())
        self.assertTrue((ROOT / "services" / "video" / "rtc_server.py").is_file())

    def test_supervisor_starts_three_processes_and_asr_owns_dual_listeners(self) -> None:
        parser = ConfigParser()
        parser.read(ROOT / "deploy" / "supervisord.conf", encoding="utf-8")
        programs = {section for section in parser.sections() if section.startswith("program:")}

        self.assertEqual({"program:asr", "program:intent", "program:sandbox"}, programs)
        self.assertIn("services.asr.main", parser["program:asr"]["command"])
        self.assertIn("services.intent.main", parser["program:intent"]["command"])
        self.assertIn(
            "services.sandbox.main",
            parser["program:sandbox"]["command"],
        )

    def test_container_mounts_models_instead_of_copying_them(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("ASR_MODEL_SOURCE", compose)
        self.assertIn("INTENT_MODEL_SOURCE", compose)
        self.assertIn("YOLO_MODEL_SOURCE", compose)
        self.assertIn(":/models/asr/whisper-large-v3:ro", compose)
        self.assertIn(":/models/intent/Qwen2.5-0.5B-Instruct:ro", compose)
        self.assertIn(":/models/yolo:ro", compose)
        self.assertNotIn("additional_contexts", compose)
        self.assertNotIn("COPY --from=asr_model", dockerfile)
        self.assertNotIn("COPY --from=intent_model", dockerfile)
        self.assertNotIn("COPY --from=yolo_models", dockerfile)
        self.assertNotIn("yolo11n", dockerfile)
        self.assertNotIn("ASR_MODEL_ID=small", dockerfile)

    def test_container_uses_host_network_for_standalone_deployment(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("network_mode: host", compose)
        self.assertNotIn("compose_n6", compose)
        self.assertNotIn("UPF_N6_IP", compose)
        self.assertIn("EXPOSE 9004 28501 28502", dockerfile)
        self.assertNotIn("EXPOSE 9004 9005", dockerfile)
        self.assertIn("127.0.0.1:9005/health", dockerfile)
