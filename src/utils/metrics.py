"""
Metrics Collector
Tracks pipeline performance metrics: latency, throughput, fraud rates.
Exposes Prometheus metrics endpoint.
"""

import time
import logging
from collections import deque
from typing import Dict, Deque
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Lightweight in-process metrics collector.
    Tracks latency percentiles, throughput, and fraud detection rates.
    """

    def __init__(self, window_size: int = 10000):
        self._latencies: Deque[float] = deque(maxlen=window_size)
        self._scores: Deque[float] = deque(maxlen=window_size)
        self._processed = 0
        self._alerted = 0
        self._start_time = time.time()

    def record_latency(self, latency_ms: float):
        self._latencies.append(latency_ms)
        self._processed += 1

    def record_score(self, score: float):
        self._scores.append(score)

    def record_alert(self):
        self._alerted += 1

    def summary(self) -> Dict:
        latencies = list(self._latencies)
        scores = list(self._scores)
        elapsed = time.time() - self._start_time

        return {
            "processed":       self._processed,
            "alerted":         self._alerted,
            "alert_rate_pct":  round(self._alerted / max(self._processed, 1) * 100, 3),
            "throughput_tps":  round(self._processed / max(elapsed, 1), 1),
            "p50_ms":          round(float(np.percentile(latencies, 50)), 2) if latencies else 0,
            "p95_ms":          round(float(np.percentile(latencies, 95)), 2) if latencies else 0,
            "p99_ms":          round(float(np.percentile(latencies, 99)), 2) if latencies else 0,
            "avg_fraud_score": round(float(np.mean(scores)), 4) if scores else 0,
            "uptime_seconds":  round(elapsed, 1),
        }

    def log_summary(self):
        s = self.summary()
        logger.info(
            f"METRICS | processed={s['processed']:,} | "
            f"alerts={s['alerted']} ({s['alert_rate_pct']}%) | "
            f"tps={s['throughput_tps']} | "
            f"p50={s['p50_ms']}ms p99={s['p99_ms']}ms"
        )
