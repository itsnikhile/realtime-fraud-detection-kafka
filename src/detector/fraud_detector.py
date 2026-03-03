"""
Fraud Detector
Consumes transactions from Kafka, applies rule engine + ML scoring,
publishes fraud alerts to alert topic.
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from kafka import KafkaConsumer, KafkaProducer

from src.models.ml_scorer import MLScorer
from src.detector.rule_engine import RuleEngine
from src.utils.feature_store import FeatureStore
from src.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class FraudAlert:
    alert_id: str
    transaction_id: str
    user_id: str
    amount: float
    fraud_score: float
    fraud_reasons: List[str]
    alert_type: str
    risk_level: str        # LOW | MEDIUM | HIGH | CRITICAL
    recommended_action: str  # ALLOW | REVIEW | BLOCK | CHALLENGE
    timestamp: str
    processing_latency_ms: float


class FraudDetector:
    """
    Main fraud detection engine.
    Orchestrates: feature retrieval → rule checks → ML scoring → alert publishing.
    """

    SCORE_THRESHOLDS = {
        "LOW":      (0.0,  0.3),
        "MEDIUM":   (0.3,  0.6),
        "HIGH":     (0.6,  0.8),
        "CRITICAL": (0.8,  1.0),
    }

    ACTION_MAP = {
        "LOW":      "ALLOW",
        "MEDIUM":   "REVIEW",
        "HIGH":     "CHALLENGE",
        "CRITICAL": "BLOCK",
    }

    def __init__(
        self,
        bootstrap_servers: str,
        input_topic: str,
        alert_topic: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="fraud-detector-v2",
            auto_offset_reset="latest",
            enable_auto_commit=False,
            max_poll_records=500,
            fetch_max_wait_ms=100,
        )
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
            acks=1,
        )
        self.alert_topic = alert_topic
        self.feature_store = FeatureStore(redis_host, redis_port)
        self.rule_engine = RuleEngine()
        self.ml_scorer = MLScorer()
        self.metrics = MetricsCollector()

    def _classify_risk(self, score: float) -> str:
        for level, (lo, hi) in self.SCORE_THRESHOLDS.items():
            if lo <= score < hi:
                return level
        return "CRITICAL"

    def _combine_scores(self, rule_score: float, ml_score: float) -> float:
        """Weighted ensemble: 35% rules + 65% ML."""
        return 0.35 * rule_score + 0.65 * ml_score

    def _build_alert(
        self,
        txn: Dict,
        fraud_score: float,
        rule_flags: Dict,
        latency_ms: float,
    ) -> FraudAlert:
        import uuid
        risk = self._classify_risk(fraud_score)
        return FraudAlert(
            alert_id=str(uuid.uuid4()).replace("-", "")[:16],
            transaction_id=txn["transaction_id"],
            user_id=txn["user_id"],
            amount=txn["amount"],
            fraud_score=round(fraud_score, 4),
            fraud_reasons=rule_flags.get("triggered_rules", []),
            alert_type=rule_flags.get("alert_type", "ML_FLAG"),
            risk_level=risk,
            recommended_action=self.ACTION_MAP[risk],
            timestamp=datetime.utcnow().isoformat() + "Z",
            processing_latency_ms=round(latency_ms, 2),
        )

    def process(self):
        logger.info("Fraud detector started — listening for transactions...")
        processed = 0
        alerted = 0

        for message in self.consumer:
            t_start = time.perf_counter()
            txn = message.value

            try:
                # 1. Get + update features
                features = self.feature_store.get_user_features(txn["user_id"], txn)
                self.feature_store.update_user_features(txn)

                # 2. Rule-based checks
                rule_flags = self.rule_engine.evaluate(txn, features)
                rule_score = min(len(rule_flags["triggered_rules"]) * 0.3, 1.0)

                # 3. ML score
                ml_score = self.ml_scorer.score(txn, features)

                # 4. Combine
                fraud_score = self._combine_scores(rule_score, ml_score)
                latency_ms = (time.perf_counter() - t_start) * 1000

                self.metrics.record_latency(latency_ms)
                self.metrics.record_score(fraud_score)

                # 5. Alert if threshold exceeded
                if fraud_score >= 0.3:
                    alert = self._build_alert(txn, fraud_score, rule_flags, latency_ms)
                    self.producer.send(
                        self.alert_topic,
                        key=txn["user_id"],
                        value=asdict(alert),
                    )
                    alerted += 1
                    if fraud_score >= 0.6:
                        logger.warning(
                            f"FRAUD [{alert.risk_level}] | {txn['user_id']} | "
                            f"${txn['amount']:.2f} | score={fraud_score:.3f} | "
                            f"action={alert.recommended_action} | {latency_ms:.1f}ms"
                        )

                processed += 1
                if processed % 5000 == 0:
                    stats = self.metrics.summary()
                    logger.info(
                        f"Processed {processed:,} | Alerts {alerted} | "
                        f"p50={stats['p50_ms']:.1f}ms p99={stats['p99_ms']:.1f}ms"
                    )

                self.consumer.commit()

            except Exception as e:
                logger.error(f"Error processing transaction {txn.get('transaction_id')}: {e}", exc_info=True)

    def close(self):
        self.consumer.close()
        self.producer.close()
        self.feature_store.close()
