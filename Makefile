# Modular Trading Bot Makefile

.PHONY: help install dev test lint format typecheck clean modules docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  dev          - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  typecheck    - Run type checking"
	@echo "  clean        - Clean up generated files"
	@echo "  collect-data - Collect historical data"
	@echo "  backtest     - Run backtest"
	@echo "  report       - Generate HTML report"
	@echo "  trading-bot  - Run trading bot (placeholder)"
	@echo "  modules      - Show available modules"

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
dev: install
	@echo "Development dependencies installed"

# Run tests
test:
	python -m pytest tests/ -v --cov=modules --cov-report=term-missing

# Run linting
lint:
	ruff check modules/ tests/
	mypy modules/

# Format code
format:
	black modules/ tests/
	isort modules/ tests/
	ruff check --fix modules/ tests/

# Type checking
typecheck:
	mypy modules/

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

# Module commands
modules:
	@echo "Available modules:"
	@echo "  Module 1: Data Collector  - python -m modules.data_collector --help"
	@echo "  Module 2: Backtester     - python -m modules.backtester --help"
	@echo "  Module 3: Reporter       - python -m modules.reporter --help"
	@echo "  Module 4: Trading Bot    - python -m modules.trading_bot --help"

# Collect historical data
collect-data:
	python -m modules.data_collector --symbol SOLUSDT --timeframe 5m --period week

# Run backtest
backtest:
	@echo "Usage: python -m modules.backtester --strategy ./examples/strategy.py --data ./data.csv"

# Generate HTML report
report:
	@echo "Usage: python -m modules.reporter --results ./backtest_results.json"

# Trading bot (placeholder)
trading-bot:
	python -m modules.trading_bot --help

# Docker commands (if needed)
docker-build:
	docker build -t modular-trading-bot .

# Example workflow
example:
	@echo "Example workflow:"
	@echo "1. Collect data:  make collect-data"
	@echo "2. Run backtest:  python -m modules.backtester --strategy ./strategy.py --data ./data.csv"
	@echo "3. Generate report: python -m modules.reporter --results ./results.json"