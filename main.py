"""
Real-Time Fraud Detection Pipeline
Entry point — starts producer or detector based on CLI args.

Usage:
  python main.py producer    # Start transaction producer
  python main.py detector    # Start fraud detector
  python main.py setup       # Create Kafka topics
  python main.py demo        # Run 60-second demo
"""

import sys
import logging
import yaml
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_setup(config: dict):
    from src.utils.kafka_setup import setup_topics
    logger.info("Setting up Kafka topics...")
    setup_topics(config["kafka"]["bootstrap_servers"])
    logger.info("Setup complete.")


def run_producer(config: dict):
    from src.producer.transaction_producer import TransactionProducer
    p = config["producer"]
    k = config["kafka"]
    producer = TransactionProducer(
        bootstrap_servers=k["bootstrap_servers"],
        topic=k["topics"]["transactions_raw"],
    )
    try:
        producer.run(tps=p["tps"], fraud_rate=p["fraud_rate"])
    finally:
        producer.close()


def run_detector(config: dict):
    from src.detector.fraud_detector import FraudDetector
    k = config["kafka"]
    r = config["redis"]
    detector = FraudDetector(
        bootstrap_servers=k["bootstrap_servers"],
        input_topic=k["topics"]["transactions_raw"],
        alert_topic=k["topics"]["fraud_alerts"],
        redis_host=r["host"],
        redis_port=r["port"],
    )
    try:
        detector.process()
    finally:
        detector.close()


def run_demo(config: dict):
    """Runs a self-contained demo using mock data (no Kafka/Redis required)."""
    import random
    from src.producer.transaction_producer import TransactionGenerator
    from src.detector.rule_engine import RuleEngine
    from src.models.ml_scorer import MLScorer

    logger.info("=== FRAUD DETECTION DEMO (no Kafka/Redis) ===")
    gen = TransactionGenerator(n_users=1000, n_merchants=200)
    rules = RuleEngine()
    scorer = MLScorer()

    alerts = 0
    for i in range(500):
        is_fraud = random.random() < 0.05
        txn = gen.generate(fraudulent=is_fraud).__dict__
        mock_features = {
            "txn_count_1h":        random.randint(1, 15) if is_fraud else random.randint(1, 4),
            "txn_count_24h":       random.randint(5, 40) if is_fraud else random.randint(1, 8),
            "total_amount_1h":     random.uniform(1000, 9000) if is_fraud else random.uniform(50, 400),
            "avg_amount_1h":       random.uniform(500, 3000) if is_fraud else random.uniform(50, 200),
            "avg_amount_30d":      150.0,
            "amount_vs_avg_ratio": random.uniform(5, 12) if is_fraud else random.uniform(0.5, 2),
            "geo_distance_km":     random.uniform(300, 1500) if is_fraud else random.uniform(0, 50),
            "last_country":        "RU" if is_fraud else "US",
        }
        rule_flags = rules.evaluate(txn, mock_features)
        ml_score = scorer.score(txn, mock_features)
        rule_score = min(len(rule_flags["triggered_rules"]) * 0.3, 1.0)
        final_score = 0.35 * rule_score + 0.65 * ml_score

        if final_score >= 0.3:
            alerts += 1
            if final_score >= 0.6:
                logger.warning(
                    f"FRAUD DETECTED | {txn['user_id']} | ${txn['amount']:.2f} | "
                    f"score={final_score:.3f} | rules={rule_flags['rule_count']}"
                )

    logger.info(f"Demo complete. Processed 500 txns | Alerts raised: {alerts}")


if __name__ == "__main__":
    config = load_config()
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if mode == "setup":
        run_setup(config)
    elif mode == "producer":
        run_producer(config)
    elif mode == "detector":
        run_detector(config)
    elif mode == "demo":
        run_demo(config)
    else:
        print(f"Unknown mode: {mode}. Use: setup | producer | detector | demo")
        sys.exit(1)
