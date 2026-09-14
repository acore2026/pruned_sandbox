from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from services.logging_config import configure_logging


class LoggingConfigTest(TestCase):
    def test_writes_service_log_and_rotates(self) -> None:
        self.addCleanup(_reset_logging)
        with TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "LOG_DIR": directory,
                "LOG_MAX_BYTES": "256",
                "LOG_BACKUP_COUNT": "2",
                "LOG_LEVEL": "INFO",
            },
            clear=True,
        ):
            configure_logging("sandbox-test")
            logger = logging.getLogger("sandbox.test")
            for index in range(20):
                logger.info("integration-log-%d %s", index, "x" * 80)
            for handler in logging.getLogger().handlers:
                handler.flush()

            log_dir = Path(directory)
            self.assertTrue((log_dir / "sandbox-test.log").is_file())
            self.assertTrue((log_dir / "sandbox-test.log.1").is_file())
            self.assertLessEqual(
                len(list(log_dir.glob("sandbox-test.log*"))),
                3,
            )


def _reset_logging() -> None:
    with patch.dict(os.environ, {"LOG_DIR": "", "LOG_LEVEL": "WARNING"}, clear=True):
        configure_logging("test-cleanup")
