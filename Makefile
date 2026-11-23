PY := .venv/bin/python
UV := uv
MODEL ?= base
SPLIT ?= test

.PHONY: setup ml train fetch check chunk index eval-retrieval validate-eval label-sheet answer eval-answers sft train-qlora eval-tuned test clean

setup: .venv/.installed

.venv/.installed: pyproject.toml
	@test -x .venv/bin/python || $(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PY) -e .
	@touch $@

ml: setup
	$(UV) pip install --python $(PY) -e ".[ml]"

train: setup
	$(UV) pip install --python $(PY) -e ".[train]"

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

sft: index setup
	$(PY) -m src.build_sft

# Requires a CUDA GPU. On the local laptop use the Colab training notebook instead.
train-qlora: sft train ml
	$(PY) -m src.train_qlora --train data/sft/train.jsonl --val data/sft/val.jsonl

eval-tuned: index ml train
	$(PY) -m src.eval_answers --split $(SPLIT) --model $(MODEL) \
		--adapter artifacts/adapter --out-prefix results/answers_tuned \
		--labels results/label_sheet_tuned.csv

test: setup
	$(PY) -m unittest discover -s tests -v

clean:
	rm -rf .venv artifacts __pycache__ src/__pycache__ tests/__pycache__
