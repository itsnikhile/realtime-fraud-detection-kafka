"""
Feature Store
Redis-backed rolling feature store for real-time fraud detection.
Maintains sliding window aggregations per user.
"""

import json
import time
import logging
import math
from typing import Dict, Optional

import redis

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Stores and retrieves per-user features using Redis sorted sets.
    Features are computed over configurable rolling windows.
    """

    WINDOW_1H  = 3600
    WINDOW_24H = 86400
    WINDOW_7D  = 604800
    TTL_BUFFER = 2  # keep data 2x the window for safety

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.redis.ping()
            logger.info("Redis connection established")
        except redis.ConnectionError:
            logger.warning("Redis not available — feature store running in mock mode")
            self.redis = None

    def update_user_features(self, txn: Dict):
        """Update rolling features after processing a transaction."""
        if not self.redis:
            return

        uid = txn["user_id"]
        now = int(time.time())
        txn_id = txn["transaction_id"]
        amount = float(txn.get("amount", 0))

        pipe = self.redis.pipeline()

        # Rolling txn count (sorted set: member=txn_id, score=timestamp)
        pipe.zadd(f"u:{uid}:txns", {txn_id: now})
        pipe.zremrangebyscore(f"u:{uid}:txns", 0, now - self.WINDOW_24H)
        pipe.expire(f"u:{uid}:txns", self.WINDOW_24H * self.TTL_BUFFER)

        # Rolling amount log (member=txn_id:amount, score=timestamp)
        pipe.zadd(f"u:{uid}:amounts", {f"{txn_id}:{amount:.4f}": now})
        pipe.zremrangebyscore(f"u:{uid}:amounts", 0, now - self.WINDOW_7D)
        pipe.expire(f"u:{uid}:amounts", self.WINDOW_7D * self.TTL_BUFFER)

        # Last known location
        pipe.hset(f"u:{uid}:location", mapping={
            "lat": txn.get("lat", 0),
            "lon": txn.get("lon", 0),
            "ts": now,
            "country": txn.get("country", ""),
        })
        pipe.expire(f"u:{uid}:location", self.WINDOW_7D)

        pipe.execute()

    def get_user_features(self, uid: str, current_txn: Dict) -> Dict:
        """Retrieve aggregated features for a user."""
        if not self.redis:
            return self._empty_features()

        now = int(time.time())
        pipe = self.redis.pipeline()

        pipe.zcount(f"u:{uid}:txns", now - self.WINDOW_1H, now)         # txn count 1h
        pipe.zcount(f"u:{uid}:txns", now - self.WINDOW_24H, now)         # txn count 24h
        pipe.zrangebyscore(f"u:{uid}:amounts", now - self.WINDOW_1H, now)  # amounts 1h
        pipe.zrangebyscore(f"u:{uid}:amounts", now - self.WINDOW_7D * 4, now)  # amounts 30d
        pipe.hgetall(f"u:{uid}:location")

        results = pipe.execute()

        txn_count_1h  = int(results[0] or 0)
        txn_count_24h = int(results[1] or 0)

        amounts_1h = [float(m.split(":")[-1]) for m in (results[2] or [])]
        amounts_30d = [float(m.split(":")[-1]) for m in (results[3] or [])]

        total_1h  = sum(amounts_1h)
        avg_1h    = total_1h / len(amounts_1h) if amounts_1h else 0
        avg_30d   = sum(amounts_30d) / len(amounts_30d) if amounts_30d else 0

        current_amount = float(current_txn.get("amount", 0))
        ratio = current_amount / avg_30d if avg_30d > 0 else 1.0

        last_loc = results[4] or {}
        geo_km = self._haversine(
            float(last_loc.get("lat", current_txn.get("lat", 0))),
            float(last_loc.get("lon", current_txn.get("lon", 0))),
            float(current_txn.get("lat", 0)),
            float(current_txn.get("lon", 0)),
        ) if last_loc else 0.0

        return {
            "txn_count_1h":          txn_count_1h,
            "txn_count_24h":         txn_count_24h,
            "total_amount_1h":       round(total_1h, 2),
            "avg_amount_1h":         round(avg_1h, 2),
            "avg_amount_30d":        round(avg_30d, 2),
            "amount_vs_avg_ratio":   round(ratio, 4),
            "geo_distance_km":       round(geo_km, 2),
            "last_country":          last_loc.get("country", ""),
        }

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _empty_features() -> Dict:
        return {
            "txn_count_1h": 0, "txn_count_24h": 0,
            "total_amount_1h": 0.0, "avg_amount_1h": 0.0,
            "avg_amount_30d": 0.0, "amount_vs_avg_ratio": 1.0,
            "geo_distance_km": 0.0, "last_country": "",
        }

    def close(self):
        if self.redis:
            self.redis.close()
