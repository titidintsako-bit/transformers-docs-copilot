PY := .venv/bin/python
UV := uv

.PHONY: setup fetch check chunk index eval-retrieval test clean

setup: .venv/.installed

.venv/.installed: pyproject.toml
	@test -x .venv/bin/python || $(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PY) -e .
	@touch $@

fetch: setup
	$(PY) -m src.fetch_docs

check: setup
	$(PY) -m src.fetch_docs --check

chunk: setup
	$(PY) -m src.chunk

index: chunk
	$(PY) -m src.index

eval-retrieval: index
	$(PY) -m src.eval_retrieval

test: setup
	$(PY) -m unittest discover -s tests -v

clean:
	rm -rf .venv artifacts __pycache__ src/__pycache__ tests/__pycache__
