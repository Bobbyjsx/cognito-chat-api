.PHONY: install run lint format test clean

VENV = .venv/bin

install:
	$(VENV)/pip install -r requirements.txt

run:
	$(VENV)/uvicorn app.main:app --reload

lint:
	$(VENV)/ruff check .

lint-fix:
	$(VENV)/ruff check . --fix

format:
	$(VENV)/ruff format .

test:
	PYTHONPATH=. $(VENV)/pytest tests/

clean:
	rm -rf __pycache__
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
