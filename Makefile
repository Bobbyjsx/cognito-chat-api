.PHONY: install run lint format test clean

VENV = .venv/bin

install:
	uv sync


run:
	docker compose up --build --watch cognito-chat-api
	docker image prune -f

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

test:
	docker compose up --build -d firestore
	sleep 5
	PYTHONPATH=. uv run pytest tests/ || (docker compose down && exit 1)
	docker compose down

latency-regression:
	PYTHONPATH=. uv run python scripts/latency_regression.py

test-latency-regression: latency-regression

# If the first argument is "migrate", treat remaining words as feature names
ifeq (migrate,$(firstword $(MAKECMDGOALS)))
  MIGRATE_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(MIGRATE_ARGS):;@:)
endif

migrate:
	PYTHONPATH=. uv run python scripts/migrate.py $(MIGRATE_ARGS)

migrate-list:
	PYTHONPATH=. uv run python scripts/migrate.py --list

clean:
	rm -rf __pycache__
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	docker compose down --rmi local
	docker image prune -f
