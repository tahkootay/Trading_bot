# Trading Bot Makefile

.PHONY: help install dev test lint format typecheck clean run docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  typecheck   - Run type checking"
	@echo "  clean       - Clean up generated files"
	@echo "  run         - Run the trading bot"
	@echo "  run-paper   - Run in paper trading mode"
	@echo "  docker-build- Build Docker image"
	@echo "  docker-run  - Run in Docker container"

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
dev: install
	pip install -e ".[dev]"

# Run tests
test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run linting
lint:
	ruff check src/ tests/
	mypy src/

# Format code
format:
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

# Type checking
typecheck:
	mypy src/

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Run the bot
run:
	python -m src.main --config config/default.yaml

# Run in paper trading mode
run-paper:
	python -m src.main --config config/default.yaml --paper-trading

# Run with debug
run-debug:
	python -m src.main --config config/default.yaml --paper-trading --debug

# Docker commands
docker-build:
	docker build -t trading-bot .

docker-run:
	docker run --env-file .env -v $(PWD)/config:/app/config -v $(PWD)/models:/app/models trading-bot

# Development environment
setup-dev: dev
	cp .env.example .env
	@echo "Please edit .env with your API credentials"

# Install and setup everything
setup: setup-dev
	@echo "Development environment setup complete!"
	@echo "Next steps:"
	@echo "1. Edit .env with your API credentials"
	@echo "2. Run 'make run-paper' to start paper trading"