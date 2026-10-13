from unittest import TestCase

from services.video.ice import public_host_candidates


class PublicIceCandidateTest(TestCase):
    def test_filters_virtual_interfaces_but_preserves_relay(self):
        sdp = ("v=0\r\n"
               "a=candidate:1 1 udp 1 10.255.255.254 10000 typ host\r\n"
               "a=candidate:2 1 udp 1 100.123.44.87 20000 typ host\r\n"
               "a=candidate:3 1 udp 1 198.18.0.1 30000 typ host\r\n"
               "a=candidate:4 1 udp 1 203.0.113.1 40000 typ relay\r\n"
               "a=end-of-candidates\r\n")
        result = public_host_candidates(sdp, "100.123.44.87")
        self.assertNotIn("10.255.255.254", result)
        self.assertNotIn("198.18.0.1", result)
        self.assertIn("100.123.44.87 20000", result)
        self.assertIn("203.0.113.1 40000", result)

    def test_nat_configuration_retains_original_candidates(self):
        sdp = "a=candidate:1 1 udp 1 192.168.1.2 10000 typ host\r\n"
        self.assertEqual(sdp, public_host_candidates(sdp, "203.0.113.1"))
