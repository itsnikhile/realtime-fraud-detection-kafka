"""
Tests for Fraud Detection Pipeline
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.detector.rule_engine import RuleEngine
from src.models.ml_scorer import MLScorer, FeatureExtractor, HeuristicModel
from src.producer.transaction_producer import TransactionGenerator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rule_engine():
    return RuleEngine()

@pytest.fixture
def ml_scorer():
    return MLScorer(model_path="nonexistent.onnx")  # forces heuristic fallback

@pytest.fixture
def sample_txn():
    return {
        "transaction_id": "TXN_001",
        "user_id": "USER_000001",
        "amount": 150.00,
        "currency": "USD",
        "merchant_category": "grocery",
        "channel": "pos",
        "country": "US",
        "is_international": False,
        "lat": 37.7749,
        "lon": -122.4194,
        "timestamp": "2024-01-15T14:30:00Z",
    }

@pytest.fixture
def clean_features():
    return {
        "txn_count_1h": 2,
        "txn_count_24h": 5,
        "total_amount_1h": 200.0,
        "avg_amount_1h": 100.0,
        "avg_amount_30d": 125.0,
        "amount_vs_avg_ratio": 1.2,
        "geo_distance_km": 5.0,
        "last_country": "US",
    }

@pytest.fixture
def fraud_features():
    return {
        "txn_count_1h": 15,           # HIGH_VELOCITY
        "txn_count_24h": 40,
        "total_amount_1h": 8500.0,     # SPEND_VELOCITY
        "avg_amount_1h": 566.0,
        "avg_amount_30d": 125.0,
        "amount_vs_avg_ratio": 7.5,    # AMOUNT_SPIKE
        "geo_distance_km": 800.0,      # GEO_ANOMALY
        "last_country": "US",
    }


# ── Rule Engine Tests ─────────────────────────────────────────────────────────

class TestRuleEngine:

    def test_clean_transaction_no_flags(self, rule_engine, sample_txn, clean_features):
        result = rule_engine.evaluate(sample_txn, clean_features)
        assert result["rule_count"] == 0
        assert len(result["triggered_rules"]) == 0

    def test_high_velocity_flagged(self, rule_engine, sample_txn, fraud_features):
        result = rule_engine.evaluate(sample_txn, fraud_features)
        assert any("HIGH_VELOCITY" in r for r in result["triggered_rules"])

    def test_geo_anomaly_flagged(self, rule_engine, sample_txn, fraud_features):
        result = rule_engine.evaluate(sample_txn, fraud_features)
        assert any("GEO_ANOMALY" in r for r in result["triggered_rules"])
        assert result["alert_type"] in ("GEO_ANOMALY", "HIGH_VELOCITY")

    def test_amount_spike_flagged(self, rule_engine, sample_txn, fraud_features):
        result = rule_engine.evaluate(sample_txn, fraud_features)
        assert any("AMOUNT_SPIKE" in r for r in result["triggered_rules"])

    def test_high_online_amount(self, rule_engine, clean_features):
        txn = {
            "transaction_id": "TXN_002",
            "user_id": "USER_000002",
            "amount": 3500.00,
            "channel": "online",
            "merchant_category": "electronics",
            "country": "US",
            "is_international": False,
            "lat": 37.7749, "lon": -122.4194,
            "timestamp": "2024-01-15T14:30:00Z",
        }
        result = rule_engine.evaluate(txn, clean_features)
        assert any("HIGH_ONLINE_AMOUNT" in r for r in result["triggered_rules"])

    def test_international_high_amount(self, rule_engine, clean_features):
        txn = {
            "transaction_id": "TXN_003",
            "user_id": "USER_000003",
            "amount": 2000.00,
            "channel": "pos",
            "merchant_category": "travel",
            "country": "RU",
            "is_international": True,
            "lat": 55.7558, "lon": 37.6173,
            "timestamp": "2024-01-15T14:30:00Z",
        }
        result = rule_engine.evaluate(txn, clean_features)
        assert any("INTERNATIONAL" in r for r in result["triggered_rules"])


# ── ML Scorer Tests ───────────────────────────────────────────────────────────

class TestMLScorer:

    def test_score_range(self, ml_scorer, sample_txn, clean_features):
        score = ml_scorer.score(sample_txn, clean_features)
        assert 0.0 <= score <= 1.0

    def test_fraud_score_higher_than_clean(self, ml_scorer, sample_txn, clean_features, fraud_features):
        clean_score = ml_scorer.score(sample_txn, clean_features)
        fraud_txn = dict(sample_txn, amount=5000.0, is_international=True, channel="online")
        fraud_score = ml_scorer.score(fraud_txn, fraud_features)
        assert fraud_score > clean_score

    def test_feature_extraction_shape(self):
        extractor = FeatureExtractor()
        txn = {"amount": 100, "channel": "pos", "is_international": False,
               "merchant_category": "grocery", "timestamp": "2024-01-15T14:30:00Z"}
        features = {"txn_count_1h": 2, "total_amount_1h": 200, "avg_amount_30d": 150,
                    "amount_vs_avg_ratio": 0.67, "geo_distance_km": 5.0}
        vec = extractor.extract(txn, features)
        assert vec.shape == (len(FeatureExtractor.FEATURE_NAMES),)

    def test_heuristic_model_deterministic(self):
        model = HeuristicModel()
        features = np.array([0.1, 0.2, 0.1, 0.05, 0.1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.8, 0.0])
        score1 = model.predict(features)
        score2 = model.predict(features)
        assert score1 == score2


# ── Transaction Generator Tests ───────────────────────────────────────────────

class TestTransactionGenerator:

    def test_generates_valid_transaction(self):
        gen = TransactionGenerator()
        txn = gen.generate()
        assert txn.transaction_id
        assert txn.user_id.startswith("USER_")
        assert txn.amount > 0
        assert txn.currency == "USD"
        assert txn.channel in ["online", "atm", "pos", "mobile"]

    def test_fraudulent_transaction_higher_amount(self):
        gen = TransactionGenerator()
        normal_amounts = [gen.generate(fraudulent=False).amount for _ in range(100)]
        fraud_amounts = [gen.generate(fraudulent=True).amount for _ in range(100)]
        assert np.mean(fraud_amounts) > np.mean(normal_amounts)

    def test_fraudulent_transaction_broader_geo(self):
        gen = TransactionGenerator()
        normal = [gen.generate(fraudulent=False) for _ in range(50)]
        fraud = [gen.generate(fraudulent=True) for _ in range(50)]
        normal_lat_range = max(t.lat for t in normal) - min(t.lat for t in normal)
        fraud_lat_range = max(t.lat for t in fraud) - min(t.lat for t in fraud)
        assert fraud_lat_range > normal_lat_range
