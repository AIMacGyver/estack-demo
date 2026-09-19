.PHONY: sync lint format test hooks analysis-sync analysis-static benchmark

sync:
	uv sync --group dev

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

hooks:
	uv run pre-commit install

analysis-sync:
	uv sync --group analysis

analysis-static:
	uv run --group analysis vulture src tests --min-confidence 100
	uv run --group analysis radon cc src -s -a
	uv run --group analysis radon mi src -s
	uv run --group analysis complexipy src --max-complexity-allowed 60

benchmark:
	uv run --group analysis pytest benchmarks --benchmark-only --benchmark-sort=mean
