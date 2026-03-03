.PHONY: setup install test run-producer run-detector demo docker-up docker-down

install:
	pip install -r requirements.txt

docker-up:
	docker-compose up -d
	@echo "Waiting for Kafka to be ready..."
	@sleep 15
	python main.py setup

docker-down:
	docker-compose down -v

run-producer:
	python main.py producer

run-detector:
	python main.py detector

demo:
	python main.py demo

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=120
