.PHONY: install dev test lint run clean docker

install:
	pip install -e .

dev:
	pip install -e ".[all]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

run:
	streamlit run src/viz/app.py

demo:
	python examples/demo.py

cli:
	python -m src.cli --game entry_deterrence --solver backward

docker:
	docker build -t extensive-form-solver .
	docker run -p 8501:8501 extensive-form-solver

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
