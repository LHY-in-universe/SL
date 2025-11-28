.PHONY: help install install-dev test test-cov test-quick lint format clean proto docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"

proto:  ## Compile Protocol Buffer files
	bash scripts/compile_proto.sh

test:  ## Run all tests with coverage
	pytest

test-cov:  ## Run tests and generate coverage report
	pytest --cov=splitlearn_comm --cov-report=html --cov-report=term

test-quick:  ## Run tests without coverage (faster)
	pytest --no-cov -x

test-parallel:  ## Run tests in parallel
	pytest -n auto

lint:  ## Run linting checks
	@echo "Running flake8..."
	flake8 src/ tests/
	@echo "Running mypy..."
	mypy src/
	@echo "Checking import order..."
	isort --check-only src/ tests/
	@echo "Checking code format..."
	black --check src/ tests/

format:  ## Format code with black and isort
	@echo "Sorting imports..."
	isort src/ tests/
	@echo "Formatting code..."
	black src/ tests/

clean:  ## Clean up generated files
	@echo "Cleaning up..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*~' -delete

docs:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs && python -m http.server 8000

build:  ## Build package distribution
	python -m build

publish-test:  ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	twine upload dist/*

check:  ## Run all checks (lint, test)
	$(MAKE) lint
	$(MAKE) test

# Development workflow shortcuts
dev-setup:  ## Complete development setup
	$(MAKE) install-dev
	$(MAKE) proto

dev-check:  ## Quick development check
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test-quick
