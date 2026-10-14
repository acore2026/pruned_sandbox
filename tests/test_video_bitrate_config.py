import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('encoded,wire', [(0,0),(250000,1000000),(500000,2000000),(750000,3000000)])
def test_video_profile_environment(encoded, wire):
    env = dict(os.environ, VIDEO_BITRATE_BPS=str(encoded), VIDEO_WIRE_BITRATE_BPS=str(wire),
               PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, '-c',
        'from services.video.rtc_policy import VIDEO_BITRATE_BPS, VIDEO_WIRE_BITRATE_BPS; '
        'print(VIDEO_BITRATE_BPS, VIDEO_WIRE_BITRATE_BPS)'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f'{encoded} {wire}'


@pytest.mark.parametrize('key,value', [('VIDEO_BITRATE_BPS','249999'),('VIDEO_BITRATE_BPS','nan'),
                                      ('VIDEO_WIRE_BITRATE_BPS','-1')])
def test_invalid_profile_fails_fast(key, value):
    env = dict(os.environ, VIDEO_BITRATE_BPS='250000', VIDEO_WIRE_BITRATE_BPS='1000000',
               PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    env[key] = value
    result = subprocess.run([sys.executable, '-c', 'import services.video.rtc_policy'], env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'ValueError' in result.stderr


def test_native_mode_keeps_sender_unmodified():
    from types import SimpleNamespace
    from services.video.rtc_policy import limit_sender_bitrate
    sender = SimpleNamespace(transport=SimpleNamespace())
    limit_sender_bitrate(sender, 0, wire_bitrate=0)
    assert not hasattr(sender, '_video_bitrate_limit')
    assert not hasattr(sender.transport, '_video_packet_pacer')


def test_unset_environment_defaults_to_native_sender():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    env.pop('VIDEO_BITRATE_BPS', None)
    env.pop('VIDEO_WIRE_BITRATE_BPS', None)
    result = subprocess.run([sys.executable, '-c',
        'from services.video.rtc_policy import limit_sender_bitrate; '
        'from types import SimpleNamespace; s=SimpleNamespace(transport=SimpleNamespace()); '
        'limit_sender_bitrate(s); assert not hasattr(s,"_video_bitrate_limit"); '
        'assert not hasattr(s.transport,"_video_packet_pacer")'],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
