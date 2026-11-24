# Notebooks

- `colab.ipynb` — baseline notebook. Open it directly in Colab (public repo, no token).
- `colab_train.ipynb` — maintained QLoRA training and tuned evaluation notebook.
- `colab_train_run.ipynb` — executed copy of the successful T4 run: 1.0 format,
  0.821 grounded answer, 0.93 behaviour accuracy on the 43-question test split.
- `colab_quick_run.ipynb` — executed copy of the first T4 quick run, kept as an artifact.
  It predates the inline-citation prompt revision, so both sampled answers are forced
  abstentions: the 1.5B model answered but emitted no inline `[n]` citations.
