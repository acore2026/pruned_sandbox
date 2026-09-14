from __future__ import annotations

import unittest

from services.sandbox.main import _sanitize_log_value


class TrafficLoggingTest(unittest.TestCase):
    def test_redacts_credentials_and_summarizes_sdp(self) -> None:
        value = _sanitize_log_value(
            {
                "access_token": "secret-value",
                "offer": {"type": "offer", "sdp": "v=0\r\nexample"},
                "command": "向左",
            }
        )

        self.assertEqual("<redacted>", value["access_token"])
        self.assertEqual("sdp", value["offer"]["sdp"]["omitted"])
        self.assertEqual("向左", value["command"])

    def test_summarizes_long_strings(self) -> None:
        value = _sanitize_log_value({"value": "x" * 1001})
        self.assertEqual(
            {"omitted": "long string", "characters": 1001}, value["value"]
        )


if __name__ == "__main__":
    unittest.main()
