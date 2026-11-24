# Tuned adapter — human review packet

Generated from `results/predictions_tuned.jsonl` (n=43). 28 answerable, 15 negatives; 27 answered, 16 abstained.

Automatic flags: false abstentions **lm-05, lm-08**; over-answer **un-10**; answered without a gold-section citation **gs-01, lm-07, peft-03**.

Score by comparing each answer to **the passage it cites**, not to the reference. Fill `supported_0_1_2`, `citation_precision_0_1` and `citation_recall_0_1` in `results/label_sheet_tuned.csv`. Abstentions: score `abstained` only and leave the three support/citation columns blank. Open the reference only after scoring.

---

## gs-01 — answerable · generation · answered

**cited passage is not the gold section**

**Question:** Which decoding strategy does generate use by default, and how many new tokens can it produce at most when no GenerationConfig is specified?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✗ · gold-page citation ✗ · auto precision 0.00 · gold `generation_strategies#greedy-search`

**Model answer:**

> The default decoding strategy in `~GenerationMixin.generate` is greedy search [1]. It selects the next most likely token at each step [1]. Unless specified in `GenerationConfig`, this strategy generates a maximum of 20 new tokens [1]. Greedy search works well for tasks with relatively short outputs where creativity is not a priority [1].

**Cited passage(s):**

### [1] Pitfalls > Decoding strategy

`llm_tutorial:decoding-strategy:0` · page `llm_tutorial` · relevant to gold ✗

<details><summary>Passage text</summary>

Transformers v5.24.0 > Text generation > Pitfalls > Decoding strategy

The default decoding strategy in [`~GenerationMixin.generate`] is *greedy search*, which selects the next most likely token, unless otherwise specified in a models [`GenerationConfig`]. While this decoding strategy works well for input-grounded tasks (transcription, translation), it is not optimal for more creative use cases (story writing, chat applications).

For example, enable a [multinomial sampling](./generation_strategies#sampling) strategy to generate more diverse outputs. Refer to the [Generation strategy](./generation_strategies) guide for more decoding strategies.

```py
model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to(model.device)
```

<hfoptions id="decoding">
<hfoption id="greedy search">

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

</hfoption>
<hfoption id="multinomial sampling">

```py
generated_ids = model.generate(**model_inputs, do_sample=True)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

</hfoption>
</hfoptions>

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Greedy search is the default. Without a GenerationConfig it generates at most 20 new tokens.

</details>

---

## gs-02 — answerable · generation · answered

**Question:** Why does greedy search break down when generating long sequences?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `generation_strategies#greedy-search`

**Model answer:**

> Greedy search works well for tasks with relatively short outputs where creativity is not a priority [1]. However, it breaks down when generating longer sequences because it begins to repeat itself [1].

**Cited passage(s):**

### [1] Basic decoding methods > Greedy search

`generation_strategies:greedy-search:0` · page `generation_strategies` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Generation strategies > Basic decoding methods > Greedy search

Greedy search is the default decoding strategy. It selects the next most likely token at each step. Unless specified in [`GenerationConfig`], this strategy generates a maximum of 20 new tokens.

Greedy search works well for tasks with relatively short outputs where creativity is not a priority. However, it breaks down when generating longer sequences because it begins to repeat itself.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.float16).to(device)
# explicitly set to default length because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a suite of tools and services for building, deploying, and maintaining natural language processing'
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

It begins to repeat itself because it always takes the locally most likely next token.

</details>

---

## gs-03 — answerable · generation · answered

**Question:** How do I enable multinomial sampling instead of greedy decoding?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `generation_strategies#sampling`

**Model answer:**

> Enable multinomial sampling with `do_sample=True` and `num_beams=1` [2].

**Cited passage(s):**

### [2] Basic decoding methods > Sampling

`generation_strategies:sampling:0` · page `generation_strategies` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Generation strategies > Basic decoding methods > Sampling

Sampling, or multinomial sampling, randomly selects a token based on the probability distribution over the entire model's vocabulary (as opposed to the most likely token, as in greedy search). This means every token with a non-zero probability has a chance to be selected. Sampling strategies reduce repetition and can generate more creative and diverse outputs.

Enable multinomial sampling with `do_sample=True` and `num_beams=1`.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.float16).to(device)
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, num_beams=1)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company 🤗\nWe are open-source and believe that open-source is the best way to build technology. Our mission is to make AI accessible to everyone, and we believe that open-source is the best way to achieve that.'
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Set do_sample=True and num_beams=1.

</details>

---

## gs-04 — answerable · generation · answered

**Question:** What are the benefits of sampling compared with greedy search?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `generation_strategies#sampling`

**Model answer:**

> Sampling, or multinomial sampling, randomly selects a token based on the probability distribution over the entire model's vocabulary (as opposed to the most likely token, as in greedy search) [2]. This means every token with a non-zero probability has a chance to be selected [2]. Enable multinomial sampling with `do_sample=True` and `num_beams=1` [2].

**Cited passage(s):**

### [2] Basic decoding methods > Sampling

`generation_strategies:sampling:0` · page `generation_strategies` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Generation strategies > Basic decoding methods > Sampling

Sampling, or multinomial sampling, randomly selects a token based on the probability distribution over the entire model's vocabulary (as opposed to the most likely token, as in greedy search). This means every token with a non-zero probability has a chance to be selected. Sampling strategies reduce repetition and can generate more creative and diverse outputs.

Enable multinomial sampling with `do_sample=True` and `num_beams=1`.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.float16).to(device)
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, num_beams=1)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company 🤗\nWe are open-source and believe that open-source is the best way to build technology. Our mission is to make AI accessible to everyone, and we believe that open-source is the best way to achieve that.'
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Sampling reduces repetition and can generate more creative and diverse outputs.

</details>

---

## gs-05 — answerable · generation · answered

**Question:** How does beam search differ from greedy search?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `generation_strategies#beam-search`

**Model answer:**

> Beam search keeps track of several generated sequences (beams) at each time step [1]. After a certain number of steps, it selects the sequence with the highest overall probability [1]. Unlike greedy search, this strategy can "look ahead" and pick a sequence with a higher probability overall even if the initial tokens have a lower probability [1]. Beam search is best suited for input-grounded tasks, like describing an image or speech recognition [1]. You can also use `do_sample=True` with beam search to sample at each step, but beam search will still greedily prune out low probability sequences between steps [1]. Enable beam search with the `num_beams` parameter (should be greater than 1 otherwise it's equivalent to greedy search) [1].

**Cited passage(s):**

### [1] Basic decoding methods > Beam search

`generation_strategies:beam-search:0` · page `generation_strategies` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Generation strategies > Basic decoding methods > Beam search

Beam search keeps track of several generated sequences (beams) at each time step. After a certain number of steps, it selects the sequence with the highest *overall* probability. Unlike greedy search, this strategy can "look ahead" and pick a sequence with a higher probability overall even if the initial tokens have a lower probability. It is best suited for input-grounded tasks, like describing an image or speech recognition. You can also use `do_sample=True` with beam search to sample at each step, but beam search will still greedily prune out low probability sequences between steps.

> [!TIP]
> Check out the [beam search visualizer](https://huggingface.co/spaces/m-ric/beam_search_visualizer) to see how beam search works.

Enable beam search with the `num_beams` parameter (should be greater than 1 otherwise it's equivalent to greedy search).

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.float16).to(device)
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, num_beams=2)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
"['Hugging Face is an open-source company that develops and maintains the Hugging Face platform, which is a collection of tools and libraries for building and deploying natural language processing (NLP) models. Hugging Face was founded in 2018 by Thomas Wolf']"
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Beam search tracks several sequences and selects the one with the highest overall probability, so it can look ahead; greedy search follows only the most likely next token.

</details>

---

## gs-06 — answerable · generation · answered

**Question:** For which kinds of tasks is beam search best suited?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `generation_strategies#beam-search`

**Model answer:**

> Beam search keeps track of several generated sequences (beams) at each time step [1]. After a certain number of steps, it selects the sequence with the highest overall probability [1]. Unlike greedy search, this strategy can "look ahead" and pick a sequence with a higher probability overall even if the initial tokens have a lower probability [1]. It is best suited for input-grounded tasks, like describing an image or speech recognition [1]. You can also use `do_sample=True` with beam search to sample at each step, but beam search will still greedily prune out low probability sequences between steps [1].

**Cited passage(s):**

### [1] Basic decoding methods > Beam search

`generation_strategies:beam-search:0` · page `generation_strategies` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Generation strategies > Basic decoding methods > Beam search

Beam search keeps track of several generated sequences (beams) at each time step. After a certain number of steps, it selects the sequence with the highest *overall* probability. Unlike greedy search, this strategy can "look ahead" and pick a sequence with a higher probability overall even if the initial tokens have a lower probability. It is best suited for input-grounded tasks, like describing an image or speech recognition. You can also use `do_sample=True` with beam search to sample at each step, but beam search will still greedily prune out low probability sequences between steps.

> [!TIP]
> Check out the [beam search visualizer](https://huggingface.co/spaces/m-ric/beam_search_visualizer) to see how beam search works.

Enable beam search with the `num_beams` parameter (should be greater than 1 otherwise it's equivalent to greedy search).

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.float16).to(device)
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, num_beams=2)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
"['Hugging Face is an open-source company that develops and maintains the Hugging Face platform, which is a collection of tools and libraries for building and deploying natural language processing (NLP) models. Hugging Face was founded in 2018 by Thomas Wolf']"
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Input-grounded tasks such as describing an image or speech recognition.

</details>

---

## gs-07 — answerable · generation · answered

**Question:** Which files must a model repository contain to publish a custom generation method?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `generation_strategies#creating-a-custom-generation-method`

**Model answer:**

> Your custom `generate` method can relative import code from the `custom_generate` folder [3]. Only relative imports from the same-level `custom_generate` folder are supported [3]. Parent/sibling folder imports are not valid [3].

**Cited passage(s):**

### [3] Custom generation methods > Creating a custom generation method

`generation_strategies:creating-a-custom-generation-method:3` · page `generation_strategies` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Generation strategies > Custom generation methods > Creating a custom generation method

Your custom `generate` method can relative import code from the `custom_generate` folder. For example, if you have a `utils.py` file, you can import it like this:

```py
from .utils import some_function
```

Only relative imports from the same-level `custom_generate` folder are supported. Parent/sibling folder imports are not valid. The `custom_generate` argument also works locally with any directory that contains a `custom_generate` structure, which is the recommended workflow for developing your custom generation method.

```py
gen_out = model.generate(**inputs, custom_generate="path/to/local/dir", trust_remote_code=True)
```

> [!WARNING]
> Loading a local directory still executes its `custom_generate/generate.py`, so it requires `trust_remote_code=True` just like a Hub repository. Only pass `trust_remote_code=True` for code you've written or reviewed.

#### requirements.txt

You can optionally specify additional Python requirements in a `requirements.txt` file inside the `custom_generate` folder. These are checked at runtime and an exception will be thrown if they're missing, nudging users to update their environment accordingly.

#### README.md

The root level `README.md` in the model repository usually describes the model therein. However, since the focus of the repository is the custom generation method, we highly recommend to shift its focus towards describing the custom generation method. In addition to a description of the method, we recommend documenting any input and/or output differences to the original [`~GenerationMixin.generate`]. This way, users can focus on what's new, and rely on Transformers docs for generic implementation details.

For discoverability, we highly recommend you to add the `custom_generate` tag to your repository. To do so, the top of your `README.md` file should look like the example below. After you push the file, you should see the tag in your repository!

```text
---
library_name: transformers
tags:
  - custom_generate
---

(your markdown content here)
```

Recommended practices:

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

The model, custom_generate/generate.py, an optional custom_generate/requirements.txt, and a README tagged custom_generate.

</details>

---

## gs-08 — answerable · generation · answered

**Question:** In a custom generation method, what must the generate function's first argument be?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `generation_strategies#creating-a-custom-generation-method`

**Model answer:**

> `model` is the model instance, which means you have access to all attributes and methods in the model, including the ones defined in `GenerationMixin` (like the base `generate` method) [1].

**Cited passage(s):**

### [1] Custom generation methods > Creating a custom generation method

`generation_strategies:creating-a-custom-generation-method:1` · page `generation_strategies` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Generation strategies > Custom generation methods > Creating a custom generation method

#### generate.py

This is the core of your generation method. It *must* contain a method named `generate`, and this method *must* contain a `model` argument as its first argument. `model` is the model instance, which means you have access to all attributes and methods in the model, including the ones defined in [`GenerationMixin`] (like the base `generate` method).

> [!WARNING]
> `generate.py` must be placed in a folder named `custom_generate`, and not at the root level of the repository. The file paths for this feature are hardcoded.

Under the hood, when the base [`~GenerationMixin.generate`] method is called with a `custom_generate` argument, it first checks its Python requirements (if any), then locates the custom `generate` method in `generate.py`, and finally calls the custom `generate`. All received arguments and `model` are forwarded to your custom `generate` method, with the exception of the arguments used to trigger the custom generation (`trust_remote_code` and `custom_generate`).

This means your `generate` can have a mix of original and custom arguments (as well as a different output type) as shown below.

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

A model argument.

</details>

---

## gs-09 — answerable · generation · answered

**Question:** How can I find custom generation methods on the Hub?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `generation_strategies#finding-custom-generation-methods`

**Model answer:**

> You can find all custom generation methods by searching for their custom tag `custom_generate` [1]. In addition to the tag, we curate two collections of `custom_generate` methods: a collection of powerful methods contributed by the community; a collection of reference implementations for methods that previously were part of `transformers`, as well as tutorials for `custom_generate` [1]. Consider the Hub repository `transformers-community/custom_generate_example` as an example [1].

**Cited passage(s):**

### [1] Custom generation methods > Finding custom generation methods

`generation_strategies:finding-custom-generation-methods:0` · page `generation_strategies` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Generation strategies > Custom generation methods > Finding custom generation methods

You can find all custom generation methods by [searching for their custom tag.](https://huggingface.co/models?other=custom_generate), `custom_generate`. In addition to the tag, we curate two collections of `custom_generate` methods:

- [Custom generation methods - Community](https://huggingface.co/collections/transformers-community/custom-generation-methods-community-6888fb1da0efbc592d3a8ab6) -- a collection of powerful methods contributed by the community;
- [Custom generation methods - Tutorials](https://huggingface.co/collections/transformers-community/custom-generation-methods-tutorials-6823589657a94940ea02cfec) -- a collection of reference implementations for methods that previously were part of `transformers`, as well as tutorials for `custom_generate`.

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Search models for the custom_generate tag or browse the curated community and tutorial collections.

</details>

---

## gs-10 — answerable · generation · answered

**Question:** Can a custom decoding loop reuse generate's input preparation, and what does that preserve?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `generation_strategies#reusing-generates-input-preparation`

**Model answer:**

> If you're adding a new decoding loop, you might want to preserve the input preparation present in `generate` (batch expansion, attention masks, logits processors, stopping criteria, etc.) [1]. You can also pass a **callable** to `custom_generate` to reuse `~GenerationMixin.generate`'s full preparation pipeline while overriding only the decoding loop [1].

**Cited passage(s):**

### [1] Custom generation methods > Reusing `generate`'s input preparation

`generation_strategies:reusing-generates-input-preparation:0` · page `generation_strategies` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Generation strategies > Custom generation methods > Reusing `generate`'s input preparation

If you're adding a new decoding loop, you might want to preserve the input preparation present in `generate` (batch expansion, attention masks, logits processors, stopping criteria, etc.). You can also pass a **callable** to `custom_generate` to reuse [`~GenerationMixin.generate`]'s full preparation pipeline while overriding only the decoding loop.

```py
def custom_loop(model, input_ids, attention_mask, logits_processor, stopping_criteria, generation_config, **model_kwargs):
    next_tokens = input_ids
    while input_ids.shape[1] < stopping_criteria[0].max_length:
        logits = model(next_tokens, attention_mask=attention_mask, **model_kwargs).logits
        next_token_logits = logits_processor(input_ids, logits[:, -1, :])
        next_tokens = torch.argmax(next_token_logits, dim=-1)[:, None]
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_tokens)), dim=-1)
    return input_ids

output = model.generate(
    **inputs,
    custom_generate=custom_loop,
    max_new_tokens=10,
)
```

> [!TIP]
> If you publish a `custom_generate` repository, your `generate` implementation can itself define a callable and pass it to `model.generate()`. This lets you customize the decoding loop while still benefiting from Transformers' built-in input preparation logic.

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Yes; pass a callable to custom_generate to reuse batch expansion, attention masks, logits processors and stopping criteria while replacing the decoding loop.

</details>

---

## lm-01 — answerable · fine-tuning · answered

**Question:** What is causal language modeling?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `tasks_language_modeling#`

**Model answer:**

> There are two types of language modeling, causal and masked [1]. This guide illustrates causal language modeling [1]. Causal language models are frequently used for text generation [1].

**Cited passage(s):**

### [1] Causal language modeling

`tasks_language_modeling:_intro:0` · page `tasks_language_modeling` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Causal language modeling

[[open-in-colab]]

There are two types of language modeling, causal and masked. This guide illustrates causal language modeling.
Causal language models are frequently used for text generation. You can use these models for creative applications like
choosing your own text adventure or an intelligent coding assistant like Copilot or CodeParrot.

<Youtube id="Vpjb1lu0MDk"/>

Causal language modeling predicts the next token in a sequence of tokens, and the model can only attend to tokens on
the left. This means the model cannot see future tokens. GPT-2 is an example of a causal language model.

This guide will show you how to:

1. Finetune [DistilGPT2](https://huggingface.co/distilbert/distilgpt2) on the [r/askscience](https://www.reddit.com/r/askscience/) subset of the [ELI5](https://huggingface.co/datasets/dany0407/eli5_category) dataset.
2. Use your finetuned model for inference.

<Tip>

To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/text-generation)

</Tip>

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install transformers datasets evaluate
```

We encourage you to log in to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to log in:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Predicting the next token from the tokens to its left; the model cannot attend to future tokens.

</details>

---

## lm-02 — answerable · fine-tuning · answered

**Question:** Which pretrained model does the causal language modeling guide fine-tune?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `tasks_language_modeling#`

**Model answer:**

> There are two types of language modeling, causal and masked [2]. This guide illustrates causal language modeling [2].

**Cited passage(s):**

### [2] Causal language modeling

`tasks_language_modeling:_intro:0` · page `tasks_language_modeling` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Causal language modeling

[[open-in-colab]]

There are two types of language modeling, causal and masked. This guide illustrates causal language modeling.
Causal language models are frequently used for text generation. You can use these models for creative applications like
choosing your own text adventure or an intelligent coding assistant like Copilot or CodeParrot.

<Youtube id="Vpjb1lu0MDk"/>

Causal language modeling predicts the next token in a sequence of tokens, and the model can only attend to tokens on
the left. This means the model cannot see future tokens. GPT-2 is an example of a causal language model.

This guide will show you how to:

1. Finetune [DistilGPT2](https://huggingface.co/distilbert/distilgpt2) on the [r/askscience](https://www.reddit.com/r/askscience/) subset of the [ELI5](https://huggingface.co/datasets/dany0407/eli5_category) dataset.
2. Use your finetuned model for inference.

<Tip>

To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/text-generation)

</Tip>

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install transformers datasets evaluate
```

We encourage you to log in to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to log in:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

DistilGPT2.

</details>

---

## lm-03 — answerable · fine-tuning · answered

**Question:** Which dataset and how many examples does the guide load for fine-tuning?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `tasks_language_modeling#load-eli5-dataset`

**Model answer:**

> Start by loading the first 5000 examples from the ELI5-Category dataset with the 🤗 Datasets library [3]. Split the dataset's train split into a train and test set with the ~datasets.Dataset.train_test_split method [3]. Then take a look at an example [3].

**Cited passage(s):**

### [3] Load ELI5 dataset

`tasks_language_modeling:load-eli5-dataset:0` · page `tasks_language_modeling` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Causal language modeling > Load ELI5 dataset

Start by loading the first 5000 examples from the [ELI5-Category](https://huggingface.co/datasets/dany0407/eli5_category) dataset with the 🤗 Datasets library. This'll give you a chance to experiment and make sure everything works before spending more time training on the full dataset.

```py
>>> from datasets import load_dataset

>>> eli5 = load_dataset("dany0407/eli5_category", split="train[:5000]")
```

Split the dataset's `train` split into a train and test set with the [`~datasets.Dataset.train_test_split`] method:

```py
>>> eli5 = eli5.train_test_split(test_size=0.2)
```

Then take a look at an example:

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

The ELI5-Category dataset, first 5,000 examples.

</details>

---

## lm-04 — answerable · fine-tuning · answered

**Question:** How does the guide split the loaded dataset into train and test sets?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `tasks_language_modeling#load-eli5-dataset`

**Model answer:**

> Start by loading the first 5000 examples from the ELI5-Category dataset with the 🤗 Datasets library [1]. Split the dataset's train split into a train and test set with the ~datasets.Dataset.train_test_split method [1]. Then take a look at an example [1].

**Cited passage(s):**

### [1] Load ELI5 dataset

`tasks_language_modeling:load-eli5-dataset:0` · page `tasks_language_modeling` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Causal language modeling > Load ELI5 dataset

Start by loading the first 5000 examples from the [ELI5-Category](https://huggingface.co/datasets/dany0407/eli5_category) dataset with the 🤗 Datasets library. This'll give you a chance to experiment and make sure everything works before spending more time training on the full dataset.

```py
>>> from datasets import load_dataset

>>> eli5 = load_dataset("dany0407/eli5_category", split="train[:5000]")
```

Split the dataset's `train` split into a train and test set with the [`~datasets.Dataset.train_test_split`] method:

```py
>>> eli5 = eli5.train_test_split(test_size=0.2)
```

Then take a look at an example:

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

train_test_split with test_size=0.2.

</details>

---

## lm-05 — answerable · fine-tuning · abstained

**FALSE ABSTAIN — expected an answer**

**Question:** Why does the guide flatten the ELI5 dataset before tokenizing?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold `tasks_language_modeling#preprocess`

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Because the text field is nested inside answers.

</details>

---

## lm-06 — answerable · fine-tuning · answered

**Question:** What does the group_texts function do?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `tasks_language_modeling#preprocess`

**Model answer:**

> Apply the `group_texts` function over the entire dataset [1]. Now create a batch of examples using `DataCollatorForLanguageModeling` [1]. It's more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length [1]. Use the end-of-sequence token as the padding token and set `mlm=False` [1]. This will use the inputs as labels shifted to the right by one element [1].

**Cited passage(s):**

### [1] Preprocess

`tasks_language_modeling:preprocess:2` · page `tasks_language_modeling` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Causal language modeling > Preprocess

Apply the `group_texts` function over the entire dataset:

```py
>>> lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
```

Now create a batch of examples using [`DataCollatorForLanguageModeling`]. It's more efficient to *dynamically pad* the
sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.

Use the end-of-sequence token as the padding token and set `mlm=False`. This will use the inputs as labels shifted to the right by one element:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Concatenates token sequences, splits them into block_size chunks (128), and copies input_ids into labels.

</details>

---

## lm-07 — answerable · fine-tuning · answered

**cited passage is not the gold section**

**Question:** Which data collator does the guide use, and with what mlm setting?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✗ · gold-page citation ✗ · auto precision 0.00 · gold `tasks_language_modeling#preprocess`

**Model answer:**

> Use the `~datasets.train_test_split` method to create a test split for evaluating the model [2]. Set `mlm=False` to avoid randomly masking tokens [2].

**Cited passage(s):**

### [2] Tokenization

`training:tokenization:0` · page `training` · relevant to gold ✗

<details><summary>Passage text</summary>

Transformers v5.24.0 > Fine-tuning > Tokenization

Load a dataset and [tokenize](./fast_tokenizers) the text column the model trains on (`horoscope` in the dataset below).

<iframe
  src="https://huggingface.co/datasets/karthiksagarn/astro_horoscope/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

The tokenizer creates the model inputs, `input_ids` and `attention_mask`. The model's forward method only accepts `input_ids` and `attention_mask`, so set `remove_columns` to drop columns like `horoscope` after tokenization.

- Set `truncation=True` and a `max_length` to truncate longer sequences to a specified maximum length.
- Use the [`~datasets.train_test_split`] method to create a test split for evaluating the model.

```py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")

def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)
```

A data collator assembles dataset samples into batches for the model to process. [`DataCollatorForLanguageModeling`] *dynamically* pads each batch to the longest sequence in that batch rather than padding every sequence in the dataset to the same length. This saves compute and memory by avoiding computing unnecessary padding tokens.

- Set `mlm=False` to avoid randomly masking tokens.

```py
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

DataCollatorForLanguageModeling with mlm=False; the EOS token is used for padding.

</details>

---

## lm-08 — answerable · fine-tuning · abstained

**FALSE ABSTAIN — expected an answer**

**Question:** How does the guide run inference after fine-tuning the causal language model?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold `tasks_language_modeling#inference`

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Use pipeline('text-generation', model=...) or load with AutoModelForCausalLM and call generate with max_new_tokens.

</details>

---

## peft-01 — answerable · fine-tuning · answered

**Question:** Why do PEFT methods use less memory than full fine-tuning?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `peft#`

**Model answer:**

> Because only adapter parameters are updated, the optimizer tracks far fewer gradients and states, reducing memory usage significantly [3]. Adapter parameters are lightweight, making them convenient to share, store, and load [3].

**Cited passage(s):**

### [3] Parameter-efficient fine-tuning

`peft:_intro:0` · page `peft` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Parameter-efficient fine-tuning

[Parameter-efficient fine-tuning (PEFT)](https://huggingface.co/docs/peft/index) methods only fine-tune a small number of extra model parameters (adapters) on top of a pretrained model. Because only adapter parameters are updated, the optimizer tracks far fewer gradients and states, reducing memory usage significantly. Adapters are lightweight, making them convenient to share, store, and load.

Transformers integrates directly with the PEFT library through [`~integrations.PeftAdapterMixin`], added to all [`PreTrainedModel`] classes. You can load, add, train, switch, and delete adapters without wrapping your model in a separate [`~peft.PeftModel`]. All non-prompt-learning PEFT methods are supported (LoRA, IA3, AdaLoRA). Prompt-based methods like prompt tuning and prefix tuning require using the [PEFT library](https://huggingface.co/docs/peft/index) directly.

Install PEFT to get started. The integration requires `peft >= 0.20.0`.

```shell
pip install -U peft
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Only adapter parameters are updated, so the optimizer tracks far fewer gradients and states.

</details>

---

## peft-02 — answerable · fine-tuning · answered

**Question:** Which mixin class gives all PreTrainedModel classes adapter support?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `peft#`

**Model answer:**

> Install PEFT to get started [2]. The integration requires `peft >= 0.20.0` [2].

**Cited passage(s):**

### [2] Parameter-efficient fine-tuning

`peft:_intro:0` · page `peft` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Parameter-efficient fine-tuning

[Parameter-efficient fine-tuning (PEFT)](https://huggingface.co/docs/peft/index) methods only fine-tune a small number of extra model parameters (adapters) on top of a pretrained model. Because only adapter parameters are updated, the optimizer tracks far fewer gradients and states, reducing memory usage significantly. Adapters are lightweight, making them convenient to share, store, and load.

Transformers integrates directly with the PEFT library through [`~integrations.PeftAdapterMixin`], added to all [`PreTrainedModel`] classes. You can load, add, train, switch, and delete adapters without wrapping your model in a separate [`~peft.PeftModel`]. All non-prompt-learning PEFT methods are supported (LoRA, IA3, AdaLoRA). Prompt-based methods like prompt tuning and prefix tuning require using the [PEFT library](https://huggingface.co/docs/peft/index) directly.

Install PEFT to get started. The integration requires `peft >= 0.20.0`.

```shell
pip install -U peft
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

PeftAdapterMixin.

</details>

---

## peft-03 — answerable · fine-tuning · answered

**cited passage is not the gold section**

**Question:** How do I attach a LoRA adapter to a model?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✗ · gold-page citation ✓ · auto precision 0.00 · gold `peft#add-an-adapter`

**Model answer:**

> Create a PEFT config, like `~peft.LoraConfig` for example, and attach it to a model with `~integrations.PeftAdapterMixin.add_adapter` [4].

**Cited passage(s):**

### [4] Hotswapping adapters > torch.compile

`peft:torchcompile:0` · page `peft` · relevant to gold ✗

<details><summary>Passage text</summary>

Transformers v5.24.0 > Parameter-efficient fine-tuning > Hotswapping adapters > torch.compile

For compiled models, call [`~integrations.peft.PeftAdapterMixin.enable_peft_hotswap`] *before* loading the first adapter and before compiling.

```py
model = AutoModel.from_pretrained(...)
max_rank = ...  # highest rank among all LoRAs you'll load
model.enable_peft_hotswap(target_rank=max_rank)
model.load_adapter(adapter_path_1, adapter_name="default")
model = torch.compile(model, ...)
output_1 = model(...)

# Hotswap without recompilation
model.load_adapter(adapter_path_2, adapter_name="default")
output_2 = model(...)
```

The `target_rank` argument sets the maximum rank among all LoRA adapters you'll load. If you have adapters with rank 8 and rank 16, pass `target_rank=16`. The default is 128.

After calling `enable_peft_hotswap`, all subsequent `load_adapter` calls hotswap by default. Pass `hotswap=False` explicitly to disable hotswapping.

Recompilation may still occur if the hotswapped adapter targets more layers than the initial adapter. Load the adapter that targets the most layers first to avoid recompilation.

> [!TIP]
> Wrap your code in `with torch._dynamo.config.patch(error_on_recompile=True)` to detect unexpected recompilation. If you detect recompilation despite following the steps above, open an issue with [PEFT](https://github.com/huggingface/peft/issues) with a reproducible example.

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Create a LoraConfig and call model.add_adapter(config, adapter_name=...).

</details>

---

## peft-04 — answerable · fine-tuning · answered

**Question:** How can I fully fine-tune the language model head alongside a LoRA adapter?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `peft#fully-fine-tuning-specific-layers`

**Model answer:**

> To train additional modules alongside an adapter (for example, the language model head), specify them in `modules_to_save` [1]. `modules_to_save` specifies layers that are fully fine-tuned alongside the adapter, so *all* of their parameters are updated [1]. This is useful when certain layers need updates, for example the language model head (`lm_head`) [1].

**Cited passage(s):**

### [1] Add an adapter > Fully fine-tuning specific layers

`peft:fully-fine-tuning-specific-layers:0` · page `peft` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Parameter-efficient fine-tuning > Add an adapter > Fully fine-tuning specific layers

To train additional modules alongside an adapter (for example, the language model head), specify them in `modules_to_save`. `modules_to_save` specifies layers that are fully fine-tuned alongside the adapter, so *all* of their parameters are updated. This is useful when certain layers need updates, for example the language model head (`lm_head`), when adapting a causal LM for sequence classification.

```py
lora_config = LoraConfig(
    modules_to_save=["lm_head"],
    ...
)
model.add_adapter(lora_config)
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Set modules_to_save=['lm_head'] in the LoraConfig.

</details>

---

## peft-05 — answerable · fine-tuning · answered

**Question:** Do I need to specify target_modules for Llama or Gemma models?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `peft#choosing-which-layers-to-adapt`

**Model answer:**

> For common architectures (Llama, Gemma, Qwen, etc.), PEFT has predefined default targets (like `q_proj` and `v_proj`) [1]. So you don't need to specify `target_modules` explicitly as a list of module names or a regex pattern [1].

**Cited passage(s):**

### [1] Add an adapter > Choosing which layers to adapt

`peft:choosing-which-layers-to-adapt:0` · page `peft` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Parameter-efficient fine-tuning > Add an adapter > Choosing which layers to adapt

For common architectures (Llama, Gemma, Qwen, etc.), PEFT has predefined default targets (like `q_proj` and `v_proj`), so you don't need to specify `target_modules`. If you want to target different layers, or the model doesn't have predefined targets, pass `target_modules` explicitly as a list of module names or a regex pattern.

```py
lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    ...
)
model.add_adapter(lora_config)
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

No; PEFT provides defaults such as q_proj and v_proj for common architectures. Specify target_modules for other layers or architectures without defaults.

</details>

---

## peft-06 — answerable · fine-tuning · answered

**Question:** What do Trainer checkpoints contain when training a PEFT model?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `peft#training`

**Model answer:**

> During training, `Trainer` checkpoints contain only the adapter weights (`adapter_model.safetensors`) and configuration (`adapter_config.json`), keeping checkpoints small [1]. The base model isn't included [1].

**Cited passage(s):**

### [1] Training

`peft:training:0` · page `peft` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Parameter-efficient fine-tuning > Training

Pass the model with an attached adapter to [`Trainer`] and call [`~Trainer.train`]. [`Trainer`] only updates the adapter parameters (those with `requires_grad=True`) because the base model is frozen.

```py
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

During training, [`Trainer`] checkpoints contain only the adapter weights (`adapter_model.safetensors`) and configuration (`adapter_config.json`), keeping checkpoints small. The base model isn't included.

After training, save the final adapter with [`~PreTrainedModel.save_pretrained`].

```py
model.save_pretrained("./my_adapter")
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Only adapter_model.safetensors and adapter_config.json; the base model is not included.

</details>

---

## peft-07 — answerable · fine-tuning · answered

**Question:** Does Trainer detect adapter checkpoints automatically when resuming?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `peft#resuming-from-a-checkpoint`

**Model answer:**

> `Trainer` automatically detects adapter checkpoints when resuming [1]. `Trainer` scans the checkpoint directory for subdirectories containing adapter weights and reloads each adapter with the correct trainable state [1].

**Cited passage(s):**

### [1] Training > Resuming from a checkpoint

`peft:resuming-from-a-checkpoint:0` · page `peft` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Parameter-efficient fine-tuning > Training > Resuming from a checkpoint

[`Trainer`] automatically detects adapter checkpoints when resuming. [`Trainer`] scans the checkpoint directory for subdirectories containing adapter weights and reloads each adapter with the correct trainable state.

```py
trainer.train(resume_from_checkpoint="./output/checkpoint-1000")
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Yes; Trainer scans for adapter checkpoints and reloads each adapter with its trainable state.

</details>

---

## peft-08 — answerable · fine-tuning · answered

**Question:** What must a repository or directory contain for from_pretrained to load an adapter automatically?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `peft#loading-an-adapter`

**Model answer:**

> To load an adapter, the Hub repository or local directory must contain an `adapter_config.json` file and the adapter weights [1].

**Cited passage(s):**

### [1] Loading an adapter

`peft:loading-an-adapter:0` · page `peft` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Parameter-efficient fine-tuning > Loading an adapter

To load an adapter, the Hub repository or local directory must contain an `adapter_config.json` file and the adapter weights.

<hfoptions id="load">
<hfoption id="from_pretrained">

[`~PreTrainedModel.from_pretrained`] automatically detects adapters. When it finds an `adapter_config.json`, it reads the `base_model_name_or_path` field to load the correct base model, then loads the adapter on top.

```py
from transformers import AutoModelForCausalLM

# Automatically loads the base model and attaches the adapter
model = AutoModelForCausalLM.from_pretrained("klcsp/gemma7b-lora-alpaca-11-v1")
```

</hfoption>
<hfoption id="load_adapter">

To load an adapter onto an existing model, use [`~integrations.PeftAdapterMixin.load_adapter`].

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
model.load_adapter("klcsp/gemma7b-lora-alpaca-11-v1")
```

</hfoption>
</hfoptions>

For large models, load a quantized version in 8-bit or 4-bit precision with [bitsandbytes](./quantization/bitsandbytes) to save memory. Add `device_map="auto"` to distribute the model across available hardware.

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "klcsp/gemma7b-lora-alpaca-11-v1",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

adapter_config.json and the adapter weights; base_model_name_or_path points to the correct base model.

</details>

---

## peft-09 — answerable · fine-tuning · answered

**Question:** How do I switch to a different adapter when a model has several attached?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `peft#managing-multiple-adapters`

**Model answer:**

> A model can hold multiple adapters at once [1]. Add adapters with unique names, and switch between them as needed [1]. Use `~integrations.PeftAdapterMixin.set_adapter` to activate a specific adapter [1]. `~integrations.PeftAdapterMixin.enable_adapters` enables all attached adapters, and `~integrations.PeftAdapterMixin.disable_adapters` disables all of them [1]. Use `~integrations.PeftAdapterMixin.active_adapters` to see which adapters are currently active [1]. Remove adapters you no longer need with `~integrations.PeftAdapterMixin.delete_adapter` to free memory [1].

**Cited passage(s):**

### [1] Managing multiple adapters

`peft:managing-multiple-adapters:0` · page `peft` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Parameter-efficient fine-tuning > Managing multiple adapters

A model can hold multiple adapters at once. Add adapters with unique names, and switch between them as needed.

```py
from peft import LoraConfig

model.add_adapter(LoraConfig(r=8, lora_alpha=32), adapter_name="adapter_1")
model.add_adapter(LoraConfig(r=16, lora_alpha=64), adapter_name="adapter_2")
```

Use [`~integrations.PeftAdapterMixin.set_adapter`] to activate a specific adapter. The other adapters are disabled but remain in memory.

```py
model.set_adapter("adapter_2")
```

[`~integrations.PeftAdapterMixin.enable_adapters`] enables all attached adapters, and [`~integrations.PeftAdapterMixin.disable_adapters`] disables all of them.

```py
# Disable all adapters for base model inference
model.disable_adapters()

# Re-enable all adapters
model.enable_adapters()
```

Use [`~integrations.PeftAdapterMixin.active_adapters`] to see which adapters are currently active.

```py
model.active_adapters()
# ["adapter_1"]
```

Remove adapters you no longer need with [`~integrations.PeftAdapterMixin.delete_adapter`] to free memory.

```py
model.delete_adapter("adapter_1")
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Call model.set_adapter(name); the other adapters are disabled but remain in memory.

</details>

---

## peft-10 — answerable · fine-tuning · answered

**Question:** What is adapter hotswapping and which adapter type does it support?

**Auto:** format ✓ · retrieval hit ✓ · gold-section citation ✓ · gold-page citation ✓ · auto precision 1.00 · gold `peft#hotswapping-adapters`

**Model answer:**

> Loading a new adapter each time you serve a request allocates new memory [1]. Hotswapping replaces adapter weights in-place, avoiding both issues [1]. Only LoRA adapters are supported [1]. Pass `hotswap=True` when loading a LoRA adapter to swap its weights into an existing adapter slot [1]. Set `adapter_name` to the name of the adapter to replace (`"default"` is the default adapter name) [1].

**Cited passage(s):**

### [1] Hotswapping adapters

`peft:hotswapping-adapters:0` · page `peft` · relevant to gold ✓

<details><summary>Passage text</summary>

Transformers v5.24.0 > Parameter-efficient fine-tuning > Hotswapping adapters

Loading a new adapter each time you serve a request allocates new memory. If the model is compiled with `torch.compile`, each new adapter triggers recompilation. Hotswapping replaces adapter weights in-place, avoiding both issues. Only LoRA adapters are supported.

Pass `hotswap=True` when loading a LoRA adapter to swap its weights into an existing adapter slot. Set `adapter_name` to the name of the adapter to replace (`"default"` is the default adapter name).

```py
model = AutoModel.from_pretrained(...)
# Load the first adapter normally
model.load_adapter(adapter_path_1)
# Generate outputs with adapter 1
...
# Hotswap the second adapter in-place
model.load_adapter(adapter_path_2, hotswap=True, adapter_name="default")
# Generate outputs with adapter 2
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Replacing adapter weights in place to avoid new memory allocation and torch.compile recompilation; only LoRA is supported.

</details>

---

## pt-01 — parametric · out-of-scope · abstained

**Question:** Who authored the paper 'Attention Is All You Need'?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Vaswani et al. World knowledge, absent from the corpus.

</details>

---

## pt-02 — parametric · out-of-scope · abstained

**Question:** In what year was the original Transformer paper published?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

2017. World knowledge, absent from the corpus.

</details>

---

## pt-03 — parametric · out-of-scope · abstained

**Question:** What does BERT stand for?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Bidirectional Encoder Representations from Transformers. World knowledge, absent from the corpus.

</details>

---

## pt-04 — parametric · out-of-scope · abstained

**Question:** How many parameters does GPT-2 XL have?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

About 1.5 billion. World knowledge, absent from the corpus.

</details>

---

## pt-05 — parametric · out-of-scope · abstained

**Question:** What does LoRA stand for?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Low-Rank Adaptation. World knowledge, absent from the corpus.

</details>

---

## un-01 — unanswerable · out-of-scope · abstained

**Question:** How do I configure DeepSpeed ZeRO-3 with Trainer?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Not covered by the selected pages: DeepSpeed configuration is absent.

</details>

---

## un-02 — unanswerable · out-of-scope · abstained

**Question:** How do I serve a model with vLLM?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Not covered by the selected pages: vLLM is absent.

</details>

---

## un-03 — unanswerable · out-of-scope · abstained

**Question:** How do I train a new tokenizer from scratch?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Not covered by the selected pages: tokenizer training is absent.

</details>

---

## un-04 — unanswerable · out-of-scope · abstained

**Question:** How do I quantize a model with GPTQ?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Not covered by the selected pages: GPTQ is absent.

</details>

---

## un-05 — unanswerable · out-of-scope · abstained

**Question:** Which license does the PEFT library use?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Not covered by the selected pages: the PEFT library license is not stated.

</details>

---

## un-06 — unanswerable · out-of-scope · abstained

**Question:** How do I write a custom Trainer callback?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Not covered by the selected pages: callbacks are only linked, not explained.

</details>

---

## un-07 — unanswerable · out-of-scope · abstained

**Question:** How do I stream a dataset while fine-tuning?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Not covered by the selected pages: dataset streaming is absent.

</details>

---

## un-08 — unanswerable · out-of-scope · abstained

**Question:** How do I compute BLEU for generated text?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Not covered by the selected pages: BLEU evaluation is absent.

</details>

---

## un-09 — unanswerable · out-of-scope · abstained

**Question:** How do I launch a multi-GPU training run with accelerate config?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision n/a · gold ``

**Model answer:**

> Insufficient evidence in the retrieved documentation.

No citations (abstained).

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Not covered by the selected pages: accelerate launch configuration is absent.

</details>

---

## un-10 — unanswerable · out-of-scope · answered

**OVER-ANSWER — expected abstention**

**Question:** What is the default value of max_shard_size in save_pretrained?

**Auto:** format ✓ · retrieval hit ✗ · gold-section citation ✗ · gold-page citation ✗ · auto precision 0.00 · gold ``

**Model answer:**

> Parameters load in parallel and peak memory only depends on model size [1]. Use `max_shard_size` in `~PreTrainedModel.save_pretrained` to set the maximum checkpoint size before sharding [1].

**Cited passage(s):**

### [1] Large models > Sharded checkpoints

`models:sharded-checkpoints:0` · page `models` · relevant to gold ✗

<details><summary>Passage text</summary>

Transformers v5.24.0 > Loading models > Large models > Sharded checkpoints

[`~PreTrainedModel.save_pretrained`] automatically shards checkpoints larger than 50GB. This keeps shard counts low for large models and simplifies file management.

Parameters load in parallel and peak memory only depends on model size. Use `max_shard_size` in [`~PreTrainedModel.save_pretrained`] to set the maximum checkpoint size before sharding.

> [!NOTE]
> Memory usage for models requiring dynamic weight conversion depends on the model size and the size of the largest parameters in a single conversion. This typically applies to mixture-of-experts (MoE) models where the memory usage is the model size plus the number of experts on one layer. Refer to the [dynamic weight loader](./weightconverter#fast-and-efficient-model-loading) guide to learn more about how models are loaded.

[`~PreTrainedModel.save_pretrained`] also creates an index file mapping parameter names to their shard files. The index contains two keys, `metadata` and `weight_map`.

```py
import json
import os
import tempfile

with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir, max_shard_size="50GB")
    with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "r") as f:
        index = json.load(f)

print(index.keys())
```

`metadata` stores the total model size.

```py
index["metadata"]
{'total_size': 28966928384}
```

`weight_map` maps each parameter to its shard file.

```py
index["weight_map"]
{'lm_head.weight': 'model-00006-of-00006.safetensors',
 'model.embed_tokens.weight': 'model-00001-of-00006.safetensors',
 'model.layers.0.input_layernorm.weight': 'model-00001-of-00006.safetensors',
 'model.layers.0.mlp.down_proj.weight': 'model-00001-of-00006.safetensors',
 ...
}
```

</details>

**Scores:** supported ___/2 · citation precision __/1 · citation recall __/1

<details><summary>Reference (read after scoring)</summary>

Not stated: the docs give the 50GB automatic sharding threshold but not the max_shard_size default.

</details>

---
