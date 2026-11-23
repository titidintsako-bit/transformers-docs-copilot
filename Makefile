PY := .venv/bin/python
UV := uv

.PHONY: setup fetch check test clean

setup: .venv/.installed

.venv/.installed: pyproject.toml
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PY) -e .
	@touch $@

fetch: setup
	$(PY) -m src.fetch_docs

check: setup
	$(PY) -m src.fetch_docs --check

test: setup
	$(PY) -m unittest discover -s tests -v

clean:
	rm -rf .venv artifacts/index artifacts/adapter __pycache__ src/__pycache__ tests/__pycache__
