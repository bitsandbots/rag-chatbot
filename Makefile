.PHONY: install run test lint format clean build release

install:
	python3 -m venv venv
	venv/bin/pip install -e ".[dev]"

run:
	venv/bin/python -m rag_chatbot

test:
	venv/bin/pytest -q

lint:
	venv/bin/ruff check src/ tests/

format:
	venv/bin/ruff format src/ tests/

clean:
	rm -rf venv/ chroma_db/ *.egg-info src/*.egg-info dist/ build/

build:
	venv/bin/python -m build

release:
	@if [ -z "$(VERSION)" ]; then echo "Usage: make release VERSION=x.y.z"; exit 1; fi
	bash scripts/release.sh $(VERSION)
