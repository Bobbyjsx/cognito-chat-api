.PHONY: install run lint format test clean

VENV = .venv/bin

install:
	$(VENV)/python -m pip install -r requirements.txt


run:
	docker compose up --build --watch cognito-chat-api
	docker image prune -f

lint:
	$(VENV)/ruff check .

lint-fix:
	$(VENV)/ruff check . --fix

format:
	$(VENV)/ruff format .

test:
	docker compose up --build -d firestore
	sleep 5
	PYTHONPATH=. $(VENV)/pytest tests/ || (docker compose down && exit 1)
	docker compose down

latency-regression:
	PYTHONPATH=. $(VENV)/python scripts/latency_regression.py

test-latency-regression: latency-regression

# If the first argument is "migrate", treat remaining words as feature names
ifeq (migrate,$(firstword $(MAKECMDGOALS)))
  MIGRATE_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(MIGRATE_ARGS):;@:)
endif

migrate:
	PYTHONPATH=. $(VENV)/python scripts/migrate.py $(MIGRATE_ARGS)

migrate-list:
	PYTHONPATH=. $(VENV)/python scripts/migrate.py --list

clean:
	rm -rf __pycache__
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	docker compose down --rmi local
	docker image prune -f
