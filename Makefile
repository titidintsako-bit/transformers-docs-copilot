PY := .venv/bin/python
UV := uv
MODEL ?= base
SPLIT ?= test

.PHONY: setup ml fetch check chunk index eval-retrieval validate-eval label-sheet answer eval-answers test clean

setup: .venv/.installed

.venv/.installed: pyproject.toml
	@test -x .venv/bin/python || $(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PY) -e .
	@touch $@

ml: setup
	$(UV) pip install --python $(PY) -e ".[ml]"

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

validate-eval: setup
	$(PY) -m src.eval_questions --write

label-sheet: validate-eval
	$(PY) -m src.label_sheet

answer: index ml
	$(PY) -m src.generate "$(Q)" --model $(MODEL)

eval-answers: index ml
	$(PY) -m src.eval_answers --split $(SPLIT) --model $(MODEL)

test: setup
	$(PY) -m unittest discover -s tests -v

clean:
	rm -rf .venv artifacts __pycache__ src/__pycache__ tests/__pycache__
