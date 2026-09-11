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

    def test_supervisor_starts_exactly_three_service_programs(self) -> None:
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

    def test_image_copies_original_model_contexts(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        for context in ("asr_model", "intent_model", "yolo_models"):
            self.assertIn(f"{context}:", compose)
            self.assertIn(f"COPY --from={context}", dockerfile)
        self.assertNotIn("yolo11n", dockerfile)
        self.assertNotIn("ASR_MODEL_ID=small", dockerfile)

    def test_container_publishes_asr_and_sandbox_ports(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn('"${SANDBOX_MANAGEMENT_PORT:-28501}:28501"', compose)
        self.assertIn('"${SANDBOX_USER_PORT:-28502}:28502"', compose)
        self.assertIn('"${ASR_PORT:-9004}:9004"', compose)
        self.assertNotIn(":8011\"", compose)
        self.assertIn("EXPOSE 9004 28501 28502", dockerfile)
