"""
Kafka Setup
Creates and configures required Kafka topics with production settings.
"""

import logging
from typing import List
from kafka.admin import KafkaAdminClient, NewTopic, ConfigResource, ConfigResourceType
from kafka.errors import TopicAlreadyExistsError

logger = logging.getLogger(__name__)


TOPICS_CONFIG = [
    {
        "name": "transactions.raw",
        "partitions": 12,
        "replication_factor": 1,
        "config": {
            "retention.ms": str(7 * 24 * 3600 * 1000),   # 7 days
            "compression.type": "snappy",
            "min.insync.replicas": "1",
        },
    },
    {
        "name": "transactions.scored",
        "partitions": 6,
        "replication_factor": 1,
        "config": {
            "retention.ms": str(3 * 24 * 3600 * 1000),   # 3 days
            "compression.type": "snappy",
        },
    },
    {
        "name": "fraud.alerts",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": str(30 * 24 * 3600 * 1000),  # 30 days
            "compression.type": "gzip",
        },
    },
    {
        "name": "fraud.alerts.dead-letter",
        "partitions": 1,
        "replication_factor": 1,
        "config": {
            "retention.ms": str(90 * 24 * 3600 * 1000),  # 90 days
        },
    },
]


def setup_topics(bootstrap_servers: str):
    admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers, client_id="fraud-admin")

    new_topics = [
        NewTopic(
            name=t["name"],
            num_partitions=t["partitions"],
            replication_factor=t["replication_factor"],
            topic_configs=t["config"],
        )
        for t in TOPICS_CONFIG
    ]

    try:
        admin.create_topics(new_topics, validate_only=False)
        logger.info(f"Created {len(new_topics)} Kafka topics")
    except TopicAlreadyExistsError:
        logger.info("Topics already exist — skipping creation")
    except Exception as e:
        logger.error(f"Failed to create topics: {e}")
    finally:
        admin.close()


def list_topics(bootstrap_servers: str) -> List[str]:
    admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    topics = list(admin.list_topics())
    admin.close()
    return sorted(topics)
