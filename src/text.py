"""Text helpers shared by ingestion, chunking and citation code."""

from __future__ import annotations

import hashlib
import re

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def strip_html_comments(text: str) -> str:
    """Replace HTML comments with blank lines so line numbers stay stable."""
    return _HTML_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(text: str) -> str:
    """Approximate GitHub's heading-anchor slugger.

    Known limitation: emoji and unusual unicode are dropped rather than
    transliterated, so a handful of anchors may differ from GitHub's. We
    validate anchors against the heading list we generate, never by guessing.
    """
    s = strip_html_comments(text).strip().lower()
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars per token).

    Good enough for chunk sizing and metrics; we never feed these estimates to a
    model. The fine-tuning stage uses the real tokenizer.
    """
    return max(1, round(len(text) / 4))


def iter_paragraphs(text: str):
    """Yield (paragraph, start_line) splitting on blank lines outside code fences."""
    lines = text.splitlines()
    buffer: list[str] = []
    start: int | None = None
    in_fence = False
    for lineno, line in enumerate(lines, start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            if start is None:
                start = lineno
            buffer.append(line)
            continue
        if not in_fence and not line.strip():
            if buffer:
                yield "\n".join(buffer).strip(), start
                buffer, start = [], None
            continue
        if start is None:
            start = lineno
        buffer.append(line)
    if buffer:
        yield "\n".join(buffer).strip(), start


def split_sections(text: str) -> tuple[str | None, list[dict]]:
    """Split a page into H2/H3 sections, tracking the heading path.

    Returns (title, sections) where each section has:
      level, anchor (None for the intro), heading, path (list of labels),
      anchors (list of ancestors incl. itself), body, line.
    """
    clean = strip_html_comments(text)
    title, headings = parse_markdown(clean)
    lines = clean.splitlines()

    sections: list[dict] = []
    first_heading_line = headings[0]["line"] if headings else len(lines) + 1
    intro_lines = [
        line
        for line in lines[: first_heading_line - 1]
        if not _HEADING.match(line)  # drop the H1; keep any H4+ prose
    ]
    intro = "\n".join(intro_lines).strip()
    if intro:
        sections.append(
            {
                "level": 0,
                "anchor": None,
                "heading": title,
                "path": [title] if title else [],
                "anchors": [],
                "body": intro,
                "line": 1,
            }
        )

    last_h2: dict | None = None
    for index, heading in enumerate(headings):
        body_start = heading["line"]  # 0-indexed slice starts on the line after the heading
        body_end = headings[index + 1]["line"] - 1 if index + 1 < len(headings) else len(lines)
        body = "\n".join(lines[body_start:body_end]).strip()
        if heading["level"] == 2:
            last_h2 = heading
            path = [heading["text"]]
            anchors = [heading["anchor"]]
        else:
            path = ([last_h2["text"]] if last_h2 else []) + [heading["text"]]
            anchors = ([last_h2["anchor"]] if last_h2 else []) + [heading["anchor"]]
        sections.append(
            {
                "level": heading["level"],
                "anchor": heading["anchor"],
                "heading": heading["text"],
                "path": path,
                "anchors": anchors,
                "body": body,
                "line": heading["line"],
            }
        )
    return title, sections


def parse_markdown(text: str) -> tuple[str | None, list[dict]]:
    """Return the H1 title and every H2/H3 heading with its GitHub-style anchor.

    Headings inside fenced code blocks and HTML comments are ignored.
    """
    clean = strip_html_comments(text)
    title: str | None = None
    headings: list[dict] = []
    seen: dict[str, int] = {}
    in_fence = False

    for lineno, line in enumerate(clean.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING.match(line)
        if not match:
            continue
        level = len(match.group(1))
        label = match.group(2).strip()
        if level == 1:
            if title is None:
                title = label
            continue
        if level not in (2, 3):
            continue
        base = slugify(label) or f"section-{lineno}"
        if base in seen:
            seen[base] += 1
            anchor = f"{base}-{seen[base]}"
        else:
            seen[base] = 0
            anchor = base
        headings.append({"level": level, "text": label, "anchor": anchor, "line": lineno})

    return title, headings
