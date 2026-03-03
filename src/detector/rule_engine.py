"""
Rule Engine
Deterministic fraud rules applied before ML scoring.
Rules are fast, interpretable, and catch known fraud patterns.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Applies a set of deterministic fraud detection rules.
    Returns triggered rules and an alert type classification.
    """

    # ── Thresholds ──────────────────────────────────────────────────────────
    VELOCITY_TXN_LIMIT      = 8       # max transactions per hour
    VELOCITY_AMOUNT_LIMIT   = 5000    # max total spend per hour ($)
    AMOUNT_SPIKE_RATIO      = 6.0     # current amount / 30d avg
    GEO_DISTANCE_KM         = 400     # impossible travel threshold
    HIGH_AMOUNT_ONLINE      = 2500    # single online txn threshold
    INTERNATIONAL_THRESHOLD = 1500    # flag high international amounts
    CARD_SHARING_LIMIT      = 3       # max unique cards per user per day

    def evaluate(self, txn: Dict, features: Dict) -> Dict:
        triggered: List[str] = []
        alert_type = "ML_FLAG"

        # Rule 1: Transaction velocity
        if features.get("txn_count_1h", 0) > self.VELOCITY_TXN_LIMIT:
            triggered.append(
                f"HIGH_VELOCITY: {features['txn_count_1h']} txns in last hour "
                f"(limit={self.VELOCITY_TXN_LIMIT})"
            )
            alert_type = "HIGH_VELOCITY"

        # Rule 2: Hourly spend velocity
        if features.get("total_amount_1h", 0) > self.VELOCITY_AMOUNT_LIMIT:
            triggered.append(
                f"SPEND_VELOCITY: ${features['total_amount_1h']:.2f} spent in last hour "
                f"(limit=${self.VELOCITY_AMOUNT_LIMIT})"
            )

        # Rule 3: Amount spike vs historical average
        ratio = features.get("amount_vs_avg_ratio", 1.0)
        if ratio > self.AMOUNT_SPIKE_RATIO:
            triggered.append(
                f"AMOUNT_SPIKE: {ratio:.1f}x above 30d average "
                f"(threshold={self.AMOUNT_SPIKE_RATIO}x)"
            )
            alert_type = "AMOUNT_SPIKE"

        # Rule 4: Impossible geo travel
        geo_km = features.get("geo_distance_km", 0)
        if geo_km > self.GEO_DISTANCE_KM:
            triggered.append(
                f"GEO_ANOMALY: {geo_km:.0f}km from last known location "
                f"(limit={self.GEO_DISTANCE_KM}km)"
            )
            alert_type = "GEO_ANOMALY"

        # Rule 5: High-value online transaction
        if txn.get("channel") == "online" and txn.get("amount", 0) > self.HIGH_AMOUNT_ONLINE:
            triggered.append(
                f"HIGH_ONLINE_AMOUNT: ${txn['amount']:.2f} online "
                f"(threshold=${self.HIGH_AMOUNT_ONLINE})"
            )

        # Rule 6: International transaction flag
        if txn.get("is_international") and txn.get("amount", 0) > self.INTERNATIONAL_THRESHOLD:
            triggered.append(
                f"INTERNATIONAL_HIGH_AMOUNT: ${txn['amount']:.2f} in {txn.get('country', 'UNKNOWN')}"
            )

        # Rule 7: Night-time ATM high withdrawal
        from datetime import datetime
        hour = datetime.fromisoformat(txn.get("timestamp", "2000-01-01T12:00:00").rstrip("Z")).hour
        if txn.get("channel") == "atm" and txn.get("amount", 0) > 500 and (hour >= 23 or hour <= 4):
            triggered.append(
                f"LATE_NIGHT_ATM: ${txn['amount']:.2f} at {hour:02d}:00"
            )

        return {
            "triggered_rules": triggered,
            "alert_type": alert_type,
            "rule_count": len(triggered),
        }
