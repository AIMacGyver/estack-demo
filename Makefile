.PHONY: sync lint format test hooks

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
