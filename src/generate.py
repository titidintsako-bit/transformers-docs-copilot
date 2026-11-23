"""Turn retrieved chunks into a grounded answer with validated citations.

The model never sees the gold labels. It receives numbered passages and must either
answer with ``[n]`` citations or emit the exact abstention sentence. Everything it
returns is parsed and checked here before it can count as a grounded answer.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

from src.common import load_config
from src.retrieve import Retriever

ABSTAIN_TEXT = "Insufficient evidence in the retrieved documentation."
MAX_PASSAGE_CHARS = 4000

CITATION_LIST = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
SOURCE_LINE = re.compile(r"^\s*\[(\d{1,3})\]\s+(.*?)\s+(?:-|\u2013|\u2014)\s+(https?://\S+)\s*$", re.M)
SOURCES_HEADER = re.compile(r"^\s*sources\s*:\s*$", re.I | re.M)
ANSWER_PREFIX = re.compile(r"^\s*answer\s*:\s*", re.I)
URL_IN_TEXT = re.compile(r"https?://[^\s)\]>\"']+")

SYSTEM_PROMPT = f"""You answer questions about the Hugging Face Transformers documentation using ONLY the numbered passages supplied by the user.

Rules:
- Use only facts stated in the passages. Do not use outside knowledge.
- Cite the passages you rely on with bracketed numbers like [1]. Put a citation after every sentence that makes a factual claim.
- Do not cite a passage you did not use. Do not list sources you did not cite.
- Copy the URL exactly as it appears in the passage. Never invent a URL, API name, argument or default value.
- If the passages do not contain the answer, reply with exactly: {ABSTAIN_TEXT}

When the passages do contain the answer, use exactly this format:
Answer: <2-6 sentences>
Sources:
[1] <passage title> - <passage URL>

List only the cited passages, one per line."""


def build_messages(question: str, passages: list[dict]) -> list[dict]:
    if passages:
        blocks = []
        for number, passage in enumerate(passages, start=1):
            text = passage["text"][:MAX_PASSAGE_CHARS]
            blocks.append(
                f"[{number}] {passage['page_title']} - {passage['section']}\n"
                f"URL: {passage['url']}\n"
                f"{text}"
            )
        context = "\n\n".join(blocks)
    else:
        context = "No passages were retrieved."
    user = f"Passages:\n\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _normalize_url(url: str) -> str:
    return url.strip().rstrip(".,;:)]}\"'")


def _split_sources(text: str) -> tuple[str, str]:
    matches = list(SOURCES_HEADER.finditer(text))
    if not matches:
        return text, ""
    last = matches[-1]
    return text[: last.start()], text[last.end() :]


def parse_response(raw: str, passages: list[dict]) -> dict:
    text = raw.strip()
    body, source_block = _split_sources(text)
    stripped_body = ANSWER_PREFIX.sub("", body.strip()).strip()
    abstain_detected = stripped_body.lower().startswith(ABSTAIN_TEXT.lower())

    invalid_entries: list[dict] = []
    entries: dict[int, dict] = {}
    repairs_for_anchor: list[int] = []
    bad_url_numbers: set[int] = set()
    for match in SOURCE_LINE.finditer(source_block):
        number = int(match.group(1))
        title = match.group(2).strip()
        url = _normalize_url(match.group(3))
        if not 1 <= number <= len(passages):
            invalid_entries.append({"n": number, "title": title, "url": url, "reason": "out_of_range"})
            continue
        passage = passages[number - 1]
        expected = _normalize_url(passage["url"])
        if url == expected:
            anchor_match = True
        elif url == expected.split("#", 1)[0]:
            anchor_match = False
        else:
            invalid_entries.append({"n": number, "title": title, "url": url, "reason": "url_mismatch"})
            bad_url_numbers.add(number)
            continue
        if not anchor_match:
            repairs_for_anchor.append(number)
        entries.setdefault(
            number,
            {
                "n": number,
                "title": title,
                "url": passage["url"],
                "anchor_match": anchor_match,
                "chunk_id": passage["chunk_id"],
                "page_id": passage["page_id"],
                "anchor": passage["anchor"],
                "section": passage["section"],
            },
        )

    if abstain_detected:
        return {
            "status": "abstained",
            "answer": ABSTAIN_TEXT,
            "citations": [],
            "sources": [],
            "invalid_source_entries": invalid_entries,
            "repairs": [{"action": "ignored_sources_with_abstention"}] if entries else [],
            "unverified_urls": [],
            "format_ok": not entries and not invalid_entries,
            "raw": raw,
        }

    repairs: list[dict] = [
        {"action": "url_missing_anchor", "n": number} for number in repairs_for_anchor
    ]
    known_urls = {_normalize_url(passage["url"]) for passage in passages}

    def expand(match: re.Match) -> str:
        numbers = [int(part) for part in match.group(1).split(",")]
        kept = []
        for number in numbers:
            if 1 <= number <= len(passages):
                if number not in entries:
                    passage = passages[number - 1]
                    entries[number] = {
                        "n": number,
                        "title": passage["page_title"],
                        "url": passage["url"],
                        "anchor_match": True,
                        "chunk_id": passage["chunk_id"],
                        "page_id": passage["page_id"],
                        "anchor": passage["anchor"],
                        "section": passage["section"],
                    }
                    action = (
                        "replaced_bad_source_url"
                        if number in bad_url_numbers
                        else "added_missing_source"
                    )
                    repairs.append({"action": action, "n": number})
                kept.append(number)
            else:
                repairs.append({"action": "removed_out_of_range_citation", "n": number})
        return " ".join(f"[{number}]" for number in kept)

    cleaned_body = CITATION_LIST.sub(expand, stripped_body).strip()
    cited = sorted(entries)
    unverified_urls = [
        url
        for url in (_normalize_url(found) for found in URL_IN_TEXT.findall(cleaned_body))
        if url not in known_urls
    ]

    sources = [entries[number] for number in cited]
    status = "answered"
    reason = None
    if unverified_urls:
        status, reason = "forced_abstain", "unverified_url_in_answer"
    elif not cited:
        status, reason = "forced_abstain", "no_valid_citations"

    inline_present = bool(CITATION_LIST.search(stripped_body))
    prefix_ok = bool(ANSWER_PREFIX.match(text))
    format_ok = (
        status == "answered"
        and prefix_ok
        and bool(sources)
        and inline_present
        and not invalid_entries
        and not repairs
    )

    if status == "forced_abstain":
        answer = ABSTAIN_TEXT
    else:
        answer = cleaned_body

    return {
        "status": status,
        "reason": reason,
        "answer": answer,
        "extracted_answer": cleaned_body,
        "citations": cited,
        "sources": sources,
        "invalid_source_entries": invalid_entries,
        "repairs": repairs,
        "unverified_urls": unverified_urls,
        "format_ok": format_ok,
        "raw": raw,
    }


def resolve_model(name: str, config: dict | None = None) -> str:
    config = config or load_config()
    models = config["models"]
    alias = {"base": models["base"], "smoke": models["smoke"], "fallback": models["fallback"]}
    return alias.get(name, name)


def load_hf(model_id: str, *, quantize_4bit: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict = {"device_map": "auto"}
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["dtype"] = "auto"
    elif torch.cuda.is_available():
        kwargs["dtype"] = "auto"
    else:
        kwargs["dtype"] = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def make_generate_fn(model, tokenizer, *, max_new_tokens: int = 320):
    import torch

    def generate(question: str, passages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(
            build_messages(question, passages), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="Answer one question from the pinned docs.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=config["generation"]["top_k_passages"])
    parser.add_argument("--model", default="smoke", help="base | smoke | fallback | HF model id")
    parser.add_argument("--max-new-tokens", type=int, default=config["generation"]["max_new_tokens"])
    parser.add_argument("--quantize", action="store_true", help="4-bit (GPU only)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    retriever = Retriever.load()
    passages = retriever.search(args.question, k=args.k)
    if not passages:
        parsed = parse_response("", [])
        parsed["reason"] = "no_retrieved_passages"
        parsed["latency_s"] = 0.0
        parsed["model_id"] = "(not loaded)"
        if args.json:
            import json

            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        else:
            print("status: forced_abstain  (no passages retrieved)")
            print(parsed["answer"])
        return 0

    model_id = resolve_model(args.model, config)
    print(f"model: {model_id} | passages: {len(passages)}", file=sys.stderr)
    model, tokenizer = load_hf(model_id, quantize_4bit=args.quantize)
    started = time.time()
    raw = make_generate_fn(model, tokenizer, max_new_tokens=args.max_new_tokens)(args.question, passages)
    elapsed = time.time() - started
    parsed = parse_response(raw, passages)
    parsed["latency_s"] = round(elapsed, 2)
    parsed["model_id"] = model_id

    if args.json:
        import json

        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    else:
        print(f"status: {parsed['status']}  ({elapsed:.1f}s)")
        print(parsed["answer"])
        for source in parsed["sources"]:
            print(f"  [{source['n']}] {source['title']} - {source['url']}")
        if parsed["repairs"]:
            print(f"repairs: {parsed['repairs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
