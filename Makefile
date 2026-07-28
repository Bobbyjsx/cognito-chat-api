.PHONY: install run lint format test clean

VENV = .venv/bin

install:
	$(VENV)/pip install -r requirements.txt

run:
	docker compose up --build --watch cognito-chat-api -d
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

clean:
	rm -rf __pycache__
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	docker compose down --rmi local
	docker image prune -f
