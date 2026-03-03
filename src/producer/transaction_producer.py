"""
Transaction Producer
Generates synthetic transaction events and publishes to Kafka at configurable TPS.
"""

import json
import time
import uuid
import random
import logging
import hashlib
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

from kafka import KafkaProducer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


MERCHANT_CATEGORIES = ["grocery", "gas", "restaurant", "online", "atm", "travel", "pharmacy", "electronics"]
CHANNELS = ["online", "atm", "pos", "mobile"]
COUNTRIES = ["US", "UK", "DE", "FR", "IN", "JP", "BR", "CA", "AU", "SG"]


@dataclass
class Transaction:
    transaction_id: str
    user_id: str
    amount: float
    currency: str
    merchant_id: str
    merchant_category: str
    merchant_name: str
    lat: float
    lon: float
    timestamp: str
    card_last4: str
    channel: str
    country: str
    device_fingerprint: str
    ip_address: str
    is_international: bool

    def to_dict(self) -> dict:
        return asdict(self)


class TransactionGenerator:
    """Generates realistic synthetic transaction data."""

    def __init__(self, n_users: int = 50000, n_merchants: int = 5000):
        self.users = [f"USER_{i:06d}" for i in range(n_users)]
        self.merchants = [f"MERCH_{i:05d}" for i in range(n_merchants)]
        self.merchant_names = [
            "Amazon", "Walmart", "Target", "Costco", "Shell", "BP",
            "McDonald's", "Starbucks", "Apple Store", "Best Buy",
            "Whole Foods", "CVS Pharmacy", "7-Eleven", "Home Depot"
        ]

    def generate(self, user_id: Optional[str] = None, fraudulent: bool = False) -> Transaction:
        uid = user_id or random.choice(self.users)

        if fraudulent:
            amount = random.uniform(800, 9999)
            lat = random.uniform(-90, 90)
            lon = random.uniform(-180, 180)
            country = random.choice(COUNTRIES)
        else:
            amount = round(random.lognormvariate(3.5, 1.2), 2)
            lat = random.uniform(25, 48)
            lon = random.uniform(-125, -65)
            country = "US"

        return Transaction(
            transaction_id=str(uuid.uuid4()).replace("-", "")[:20],
            user_id=uid,
            amount=round(amount, 2),
            currency="USD",
            merchant_id=random.choice(self.merchants),
            merchant_category=random.choice(MERCHANT_CATEGORIES),
            merchant_name=random.choice(self.merchant_names),
            lat=round(lat, 6),
            lon=round(lon, 6),
            timestamp=datetime.utcnow().isoformat() + "Z",
            card_last4=str(random.randint(1000, 9999)),
            channel=random.choice(CHANNELS),
            country=country,
            device_fingerprint=hashlib.md5(f"{uid}{random.randint(1,1000)}".encode()).hexdigest()[:16],
            ip_address=f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
            is_international=(country != "US"),
        )


class TransactionProducer:
    """Kafka producer for transaction events."""

    def __init__(self, bootstrap_servers: str, topic: str):
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
            acks="all",
            retries=5,
            max_in_flight_requests_per_connection=1,
            compression_type="snappy",
            batch_size=65536,
            linger_ms=5,
        )
        self.generator = TransactionGenerator()
        self._sent = 0
        self._errors = 0

    def _on_success(self, metadata):
        self._sent += 1

    def _on_error(self, e: KafkaError):
        self._errors += 1
        logger.error(f"Failed to send message: {e}")

    def send_transaction(self, txn: Transaction):
        self.producer.send(
            self.topic,
            key=txn.user_id,
            value=txn.to_dict(),
        ).add_callback(self._on_success).add_errback(self._on_error)

    def run(self, tps: int = 1000, fraud_rate: float = 0.015, duration_seconds: Optional[int] = None):
        """
        Produce transactions at target TPS.
        Args:
            tps: Target transactions per second
            fraud_rate: Fraction of transactions that are fraudulent
            duration_seconds: Run for N seconds (None = run forever)
        """
        logger.info(f"Starting producer | TPS={tps} | fraud_rate={fraud_rate*100:.1f}%")
        interval = 1.0 / tps
        start = time.time()
        last_report = start

        while True:
            is_fraud = random.random() < fraud_rate
            txn = self.generator.generate(fraudulent=is_fraud)
            self.send_transaction(txn)

            now = time.time()
            if now - last_report >= 10:
                elapsed = now - start
                logger.info(
                    f"Sent {self._sent:,} txns | "
                    f"Actual TPS: {self._sent/elapsed:.0f} | "
                    f"Errors: {self._errors}"
                )
                last_report = now

            if duration_seconds and (now - start) >= duration_seconds:
                break

            time.sleep(interval)

        self.producer.flush()
        logger.info(f"Producer done. Total sent: {self._sent:,} | Errors: {self._errors}")

    def close(self):
        self.producer.close()
