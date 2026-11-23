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
