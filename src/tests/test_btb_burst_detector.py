from __future__ import annotations

import time
import pytest
from research.btb.burst_detector import BurstConfig, BurstDetector


def test_burst_detector_cold_route_short_prefix():
    detector = BurstDetector(BurstConfig(min_prefix_tokens=128, min_arrivals=2))
    # Tokens menores que 128
    prompt = list(range(50))
    res = detector.process_request(prompt)
    assert res["is_burst"] is False
    assert res["action"] == "route_cold"


def test_burst_detector_triggers_burst_on_x_arrivals():
    config = BurstConfig(min_prefix_tokens=128, min_arrivals=2, window_seconds=1.0, warmup_replicas=4)
    detector = BurstDetector(config)

    prompt = list(range(200))
    now = time.time()

    # Primeira chegada -> route_cold
    res1 = detector.process_request(prompt, timestamp=now)
    assert res1["is_burst"] is False
    assert res1["action"] == "route_cold"

    # Segunda chegada dentro de Z -> trigger_warmup
    res2 = detector.process_request(prompt, timestamp=now + 0.1)
    assert res2["is_burst"] is True
    assert res2["action"] == "trigger_warmup"
    assert res2["target_replicas"] == 4

    # Terceira chegada -> route_warm
    res3 = detector.process_request(prompt, timestamp=now + 0.2)
    assert res3["is_burst"] is True
    assert res3["action"] == "route_warm"


def test_burst_detector_expires_outside_window():
    config = BurstConfig(min_prefix_tokens=128, min_arrivals=2, window_seconds=1.0)
    detector = BurstDetector(config)

    prompt = list(range(200))
    now = time.time()

    # Chegada 1 em t=0
    detector.process_request(prompt, timestamp=now)

    # Chegada 2 em t=2.0 (fora da janela Z=1.0s)
    res2 = detector.process_request(prompt, timestamp=now + 2.0)
    assert res2["is_burst"] is False
    assert res2["action"] == "route_cold"
