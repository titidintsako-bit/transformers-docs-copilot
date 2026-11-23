"""Train a QLoRA adapter on the synthetic SFT split.

Only the assistant answer tokens are supervised; prompt tokens are masked out. The
adapter is saved as a PEFT directory and can be loaded by `src.eval_answers --adapter`.
Training is intended for a CUDA GPU (Colab T4); the local machine has no NVIDIA GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.common import ROOT, load_config, read_jsonl

DEFAULT_OUTPUT = ROOT / "artifacts" / "adapter"


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class TokenizedRows(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def tokenize_example(example: dict, tokenizer, max_length: int) -> dict | None:
    messages = example["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"example {example.get('id')} has no assistant target")
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    answer = messages[-1]["content"].strip() + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_length:
        return None
    input_ids = prompt_ids + answer_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def build_tokenized(rows: list[dict], tokenizer, max_length: int) -> tuple[list[dict], int]:
    tokenized = []
    skipped = 0
    for row in rows:
        item = tokenize_example(row, tokenizer, max_length)
        if item is None:
            skipped += 1
        else:
            tokenized.append(item)
    return tokenized, skipped



def training_arguments(*, output_dir: Path, epochs: int, config: dict, bf16: bool):
    from transformers import TrainingArguments

    qlora = config["models"]["qlora"]
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        activation_checkpointing=True,
        activation_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=qlora["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=2,
        max_grad_norm=0.3,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not bf16,
        bf16=bf16,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        seed=config["project"]["seed"],
        data_seed=config["project"]["seed"],
    )

def train(
    *,
    train_path: Path,
    val_path: Path,
    output_dir: Path = DEFAULT_OUTPUT,
    model_id: str | None = None,
    epochs: int | None = None,
    max_seq_length: int | None = None,
    limit: int = 0,
    allow_cpu: bool = False,
) -> dict:
    import torch as _torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    config = load_config()
    model_id = model_id or config["models"]["base"]
    qlora = config["models"]["qlora"]
    epochs = epochs or qlora["epochs"]
    max_seq_length = max_seq_length or qlora["max_seq_length"]

    if not _torch.cuda.is_available() and not allow_cpu:
        raise SystemExit("no CUDA GPU; QLoRA training is meant for the Colab T4")
    if not train_path.exists():
        raise SystemExit(f"missing {train_path}; run `python -m src.build_sft` first")

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    if limit:
        train_rows = train_rows[:limit]
        val_rows = val_rows[: min(limit, len(val_rows))]
    if not train_rows or not val_rows:
        raise SystemExit("train and validation splits must both be non-empty")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_data, train_skipped = build_tokenized(train_rows, tokenizer, max_seq_length)
    val_data, val_skipped = build_tokenized(val_rows, tokenizer, max_seq_length)
    if not train_data or not val_data:
        raise SystemExit("all examples were longer than max_seq_length")
    print(
        f"tokenized train={len(train_data)} val={len(val_data)} "
        f"skipped_long={train_skipped + val_skipped}",
        file=sys.stderr,
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qlora["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=_torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        dtype=_torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    lora = LoraConfig(
        r=qlora["lora_r"],
        lora_alpha=qlora["lora_alpha"],
        lora_dropout=qlora["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    bf16 = _torch.cuda.is_bf16_supported()
    args = training_arguments(
        output_dir=output_dir, epochs=epochs, config=config, bf16=bf16
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=TokenizedRows(train_data),
        eval_dataset=TokenizedRows(val_data),
        data_collator=CausalCollator(pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    try:
        adapter_display = str(output_dir.resolve().relative_to(ROOT))
    except ValueError:
        adapter_display = str(output_dir)
    meta = {
        "model_id": model_id,
        "adapter": adapter_display,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "skipped_long": train_skipped + val_skipped,
        "epochs": epochs,
        "max_seq_length": max_seq_length,
        "lora": {key: qlora[key] for key in qlora if key != "learning_rate"},
        "learning_rate": qlora["learning_rate"],
        "train_loss": trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None,
    }
    (output_dir / "training_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"adapter saved to {output_dir}", file=sys.stderr)
    return meta


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "sft" / "train.jsonl")
    parser.add_argument("--val", type=Path, default=ROOT / "data" / "sft" / "val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=config["models"]["base"])
    parser.add_argument("--epochs", type=int, default=config["models"]["qlora"]["epochs"])
    parser.add_argument(
        "--max-seq-length", type=int, default=config["models"]["qlora"]["max_seq_length"]
    )
    parser.add_argument("--limit", type=int, default=0, help="debug: cap train/val examples")
    parser.add_argument("--allow-cpu", action="store_true", help="debug only; extremely slow")
    args = parser.parse_args(argv)
    train(
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output_dir,
        model_id=args.model,
        epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        limit=args.limit,
        allow_cpu=args.allow_cpu,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
