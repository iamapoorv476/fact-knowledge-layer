"""
parser.py — Deterministic PDF → canonical text, with verbatim quote anchoring.

Why this module is the foundation of the whole system
-----------------------------------------------------
Every fact in the knowledge layer must point at a character span. That is only
meaningful if the text those offsets index into is *reproducible*: parse the
same PDF twice, on any machine, and you must get byte-identical page text.
So this parser is fully deterministic — no OCR heuristics, no model calls, no
dict-ordering surprises — and the text it produces (``ParsedPage.text``) is the
canonical coordinate system for provenance across the entire pipeline.

What it does
------------
* Extracts text block-by-block via PyMuPDF and re-orders blocks into human
  reading order (including a two-column detector, because prospectus front
  matter and risk-factor sections are frequently two-column).
* Detects tables with ``page.find_tables()`` and renders them as Markdown
  pipe tables *in place*, so a financial figure keeps its row and column
  labels next to it. Prose blocks that live inside a table's bounding box are
  dropped to avoid duplicating the same numbers twice on a page.
* Anchors quotes back into that text through an escalating match ladder:
  exact → whitespace/unicode-normalized → case-insensitive → fuzzy alignment.
  Every anchor records which rung it landed on, so downstream code can trust
  EXACT matches and flag FUZZY ones for review.

Anything the extractor claims but this module cannot ground is *not stored*.
That is the single rule that keeps hallucinated facts out of the graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

try:  
    import pymupdf as fitz
except ImportError:  
    import fitz  

try:  
    from .schemas import (
        BlockKind,
        MatchStrategy,
        PageBlock,
        ParsedDocument,
        ParsedPage,
        SourceAnchor,
        TextChunk,
        collapse_whitespace,
    )
except ImportError:  
    from schemas import (  # type: ignore[no-redef]
        BlockKind,
        MatchStrategy,
        PageBlock,
        ParsedDocument,
        ParsedPage,
        SourceAnchor,
        TextChunk,
        collapse_whitespace,
    )

PARSER_VERSION: str = "1.0.0"

DEFAULT_FUZZY_THRESHOLD: float = 0.82

BLOCK_SEPARATOR: str = "\n\n"

_CHAR_FOLD: Dict[str, str] = {
    "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ",
    "\u2002": " ", "\u2003": " ", "\u2005": " ", "\ufeff": "", "\u200b": "",
    "\u00ad": "",  # soft hyphen
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2026": "...",
}


def normalize_for_match(text: str) -> Tuple[str, List[int]]:
    """Fold a string for matching and return a map from folded → original index.

    Whitespace runs collapse to one space, unicode punctuation folds to ASCII,
    and zero-width characters vanish. ``index_map[i]`` is the index in ``text``
    of the character that produced folded character ``i``, which is what lets a
    normalized match be reported as exact offsets in the *original* page text.

    >>> folded, imap = normalize_for_match("Revenue  of \\u20b9 1,240")
    >>> folded
    'Revenue of ₹ 1,240'
    >>> imap[folded.index('1')]
    14
    """
    folded_chars: List[str] = []
    index_map: List[int] = []
    pending_space = False

    for original_index, char in enumerate(text):
        replacement = _CHAR_FOLD.get(char, char)
        if not replacement:
            continue
        if replacement.isspace() or char.isspace():
            pending_space = True
            continue
        if pending_space and folded_chars:
            folded_chars.append(" ")
            index_map.append(original_index)
        pending_space = False
        for sub_char in replacement:
            folded_chars.append(sub_char)
            index_map.append(original_index)

    return "".join(folded_chars), index_map


def _map_span(index_map: Sequence[int], start: int, end: int) -> Tuple[int, int]:
    """Translate a folded-text span back to original-text offsets."""
    if not index_map or start >= len(index_map) or end <= start:
        return (-1, -1)
    end = min(end, len(index_map))
    return (index_map[start], index_map[end - 1] + 1)


@dataclass(frozen=True)
class QuoteMatch:
    """Result of locating a quote inside a page's canonical text."""

    start: int
    end: int
    strategy: MatchStrategy
    score: float
    matched_text: str
    occurrences: int = 0

    @property
    def found(self) -> bool:
        return self.start >= 0 and self.end > self.start

    @property
    def is_ambiguous(self) -> bool:
        """The quote appears more than once — the anchor points at the first hit."""
        return self.occurrences > 1

    def as_tuple(self) -> Tuple[int, int]:
        return (self.start, self.end)


_MISS = QuoteMatch(-1, -1, MatchStrategy.UNVERIFIED, 0.0, "", 0)


def locate_quote(
    page_text: str,
    quote: str,
    *,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    allow_fuzzy: bool = True,
) -> QuoteMatch:
    """Find ``quote`` inside ``page_text`` using an escalating match ladder.

    Rungs, in order of decreasing trust:

    1. ``EXACT`` — byte-identical substring.
    2. ``NORMALIZED`` — matches after whitespace/unicode folding (line wraps,
       non-breaking spaces, en-dashes retyped as hyphens).
    3. ``CASE_INSENSITIVE`` — same, ignoring case (headings re-typed in title case).
    4. ``FUZZY`` — best contiguous alignment above ``fuzzy_threshold``, for quotes
       that dropped or added a word.

    Returns a miss (``start == -1``) rather than guessing when nothing clears the
    threshold.
    """
    if not page_text or not quote or not quote.strip():
        return _MISS

    exact_index = page_text.find(quote)
    if exact_index != -1:
        return QuoteMatch(
            start=exact_index,
            end=exact_index + len(quote),
            strategy=MatchStrategy.EXACT,
            score=1.0,
            matched_text=quote,
            occurrences=page_text.count(quote),
        )

    folded_page, index_map = normalize_for_match(page_text)
    folded_quote, _ = normalize_for_match(quote)
    if not folded_quote:
        return _MISS
    hit = folded_page.find(folded_quote)
    if hit != -1:
        start, end = _map_span(index_map, hit, hit + len(folded_quote))
        if start >= 0:
            return QuoteMatch(
                start=start,
                end=end,
                strategy=MatchStrategy.NORMALIZED,
                score=0.99,
                matched_text=page_text[start:end],
                occurrences=folded_page.count(folded_quote),
            )

    lowered_page = folded_page.casefold()
    lowered_quote = folded_quote.casefold()
    if len(lowered_page) == len(folded_page) and len(lowered_quote) == len(folded_quote):
        hit = lowered_page.find(lowered_quote)
        if hit != -1:
            start, end = _map_span(index_map, hit, hit + len(lowered_quote))
            if start >= 0:
                return QuoteMatch(
                    start=start,
                    end=end,
                    strategy=MatchStrategy.CASE_INSENSITIVE,
                    score=0.95,
                    matched_text=page_text[start:end],
                    occurrences=lowered_page.count(lowered_quote),
                )
        page_for_fuzzy, quote_for_fuzzy = lowered_page, lowered_quote
    else:
        page_for_fuzzy, quote_for_fuzzy = folded_page, folded_quote
    if not allow_fuzzy:
        return _MISS
    span = _best_fuzzy_span(page_for_fuzzy, quote_for_fuzzy, fuzzy_threshold)
    if span is None:
        return _MISS
    folded_start, folded_end, score = span
    start, end = _map_span(index_map, folded_start, folded_end)
    if start < 0:
        return _MISS
    return QuoteMatch(
        start=start,
        end=end,
        strategy=MatchStrategy.FUZZY,
        score=round(score, 4),
        matched_text=page_text[start:end],
        occurrences=1,
    )


def _best_fuzzy_span(
    haystack: str, needle: str, threshold: float
) -> Optional[Tuple[int, int, float]]:
    """Find the densest contiguous region of ``haystack`` aligning with ``needle``.

    ``SequenceMatcher`` gives matching blocks scattered over the whole page; a
    naive first-to-last window would stretch across unrelated paragraphs. So the
    blocks are greedily grouped, allowing gaps of at most the needle's length,
    and the group with the most matched characters wins. The returned score is
    the standard 2*M/T ratio computed over that window only.
    """
    if not haystack or not needle:
        return None

    matcher = SequenceMatcher(None, haystack, needle, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= 3]
    if not blocks:
        return None

    max_gap = max(len(needle), 40)
    groups: List[List[Any]] = [[blocks[0]]]
    for block in blocks[1:]:
        previous = groups[-1][-1]
        if block.a - (previous.a + previous.size) <= max_gap:
            groups[-1].append(block)
        else:
            groups.append([block])

    best: Optional[Tuple[int, int, float]] = None
    for group in groups:
        matched = sum(b.size for b in group)
        start = group[0].a
        end = group[-1].a + group[-1].size
        window = end - start
        if window <= 0:
            continue
        score = (2.0 * matched) / (window + len(needle))
        if score >= threshold and (best is None or score > best[2]):
            best = (start, end, score)
    return best


def locate_quote_offsets(page_text: str, quote: str) -> Tuple[int, int]:
    """Verify a quote and return its ``(char_start, char_end)`` in ``page_text``.

    Returns ``(-1, -1)`` when the quote cannot be grounded. This is the
    deterministic contract the extraction layer depends on: a quote it cannot
    place is a quote the model invented or garbled, and the fact is discarded.
    """
    return locate_quote(page_text, quote).as_tuple()


def build_anchor(
    *,
    page: ParsedPage,
    file_id: str,
    quote: str,
    document_name: Optional[str] = None,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    allow_fuzzy: bool = True,
) -> Optional[SourceAnchor]:
    """Ground a quote against a parsed page and return a validated anchor.

    Returns ``None`` if the quote does not resolve. The anchor stores the text
    *as it appears in the page* (not as the model typed it), so the UI highlight
    and the stored quote can never drift apart.
    """
    match = locate_quote(
        page.text, quote, fuzzy_threshold=fuzzy_threshold, allow_fuzzy=allow_fuzzy
    )
    if not match.found:
        return None

    block = page.block_at(match.start)
    end_block = page.block_at(max(match.start, match.end - 1))
    if block is None:
        block_kind = "text"
    elif end_block is not None and end_block.kind is not block.kind:
        block_kind = "mixed"
    else:
        block_kind = block.kind.value

    return SourceAnchor(
        file_id=file_id,
        page_number=page.page_number,
        verbatim_quote=match.matched_text,
        char_start=match.start,
        char_end=match.end,
        match_strategy=match.strategy,
        match_score=match.score,
        document_name=document_name,
        block_kind=block_kind,  
    )


def verify_anchor(document: ParsedDocument, anchor: SourceAnchor, *, strict: bool = False) -> bool:
    """Re-resolve an anchor against a freshly parsed document (integrity check)."""
    if anchor.file_id != document.file_id:
        return False
    page = document.page(anchor.page_number)
    if page is None:
        return False
    return anchor.verify(page.text, strict=strict)


Bbox = Tuple[float, float, float, float]


def _area(box: Bbox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(a: Bbox, b: Bbox) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _overlap_ratio(inner: Bbox, outer: Bbox) -> float:
    inner_area = _area(inner)
    if inner_area <= 0:
        return 0.0
    return _intersection_area(inner, outer) / inner_area

@dataclass
class _LayoutItem:
    kind: BlockKind
    text: str
    bbox: Bbox
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None


def _is_full_width(box: Bbox, page_width: float) -> bool:
    if page_width <= 0:
        return True
    return (box[2] - box[0]) >= 0.62 * page_width


def _reading_order(items: List[_LayoutItem], page_width: float) -> List[_LayoutItem]:
    """Sort layout items the way a person reads them.

    Single-column pages sort top-to-bottom, left-to-right with a small vertical
    tolerance so items on the same visual line stay together. Two-column pages
    are split into vertical bands delimited by full-width items (headers,
    section titles, wide tables); inside each band the left column is emitted
    fully before the right one.
    """
    if len(items) <= 1:
        return list(items)

    def simple_key(item: _LayoutItem) -> Tuple[float, float]:
        return (round(item.bbox[1] / 4.0), item.bbox[0])

    narrow = [it for it in items if not _is_full_width(it.bbox, page_width)]
    if len(narrow) < 4 or page_width <= 0:
        return sorted(items, key=simple_key)

    midpoint = page_width / 2.0
    left = [it for it in narrow if (it.bbox[0] + it.bbox[2]) / 2.0 < midpoint]
    right = [it for it in narrow if (it.bbox[0] + it.bbox[2]) / 2.0 >= midpoint]
    crossing = [it for it in narrow if it.bbox[0] < 0.45 * page_width < 0.55 * page_width < it.bbox[2]]

    two_column = len(left) >= 2 and len(right) >= 2 and len(crossing) <= max(1, len(narrow) // 10)
    if not two_column:
        return sorted(items, key=simple_key)

    delimiters = sorted(
        (it for it in items if _is_full_width(it.bbox, page_width)), key=lambda it: it.bbox[1]
    )
    boundaries = [it.bbox[1] for it in delimiters] + [float("inf")]

    ordered: List[_LayoutItem] = []
    band_start = float("-inf")
    for band_index, band_end in enumerate(boundaries):
        band_left = sorted(
            (it for it in left if band_start <= it.bbox[1] < band_end), key=simple_key
        )
        band_right = sorted(
            (it for it in right if band_start <= it.bbox[1] < band_end), key=simple_key
        )
        ordered.extend(band_left)
        ordered.extend(band_right)
        if band_index < len(delimiters):
            ordered.append(delimiters[band_index])
        band_start = band_end
    return ordered

def _lines_from_block(block: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for line in block.get("lines", []):
        text = "".join(span.get("text", "") for span in line.get("spans", []))
        text = text.rstrip()
        if text:
            lines.append(text)
    return lines


def _join_lines(lines: Sequence[str]) -> str:
    """Join a block's lines, repairing words split by a hyphen at the line break."""
    if not lines:
        return ""
    parts: List[str] = [lines[0]]
    for line in lines[1:]:
        previous = parts[-1]
        if previous.endswith("-") and len(previous) > 1 and previous[-2].isalpha() and line[:1].islower():
            parts[-1] = previous[:-1] + line
        else:
            parts.append(line)
    return "\n".join(parts)


def _table_to_markdown(rows: Sequence[Sequence[Optional[str]]]) -> str:
    """Render extracted table cells as a Markdown pipe table.

    Markdown is used rather than TSV because it keeps every number visually
    attached to its row label and column header on a single line — which is
    exactly the context an extractor needs to turn ``1,240`` into
    ``(Acme Ltd, total_revenue, 1240, INR crore, FY23)``.
    """
    cleaned: List[List[str]] = []
    for row in rows:
        cleaned.append(
            [collapse_whitespace(str(cell)).replace("|", r"\|") if cell is not None else "" for cell in row]
        )
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    header = cleaned[0]
    if not any(cell for cell in header):
        header = [f"col_{i + 1}" for i in range(width)]
        body = cleaned
    else:
        body = cleaned[1:]

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _extract_tables(page: "fitz.Page", warnings: List[str]) -> List[_LayoutItem]:
    items: List[_LayoutItem] = []
    if not hasattr(page, "find_tables"):
        return items
    try:
        finder = page.find_tables()
        tables = list(getattr(finder, "tables", finder) or [])
    except Exception as exc:  # pragma: no cover - depends on PDF internals
        warnings.append(f"page {page.number + 1}: table detection failed ({exc.__class__.__name__})")
        return items

    for table in tables:
        try:
            rows = table.extract()
            markdown = _table_to_markdown(rows)
            if not markdown:
                continue
            items.append(
                _LayoutItem(
                    kind=BlockKind.TABLE,
                    text=markdown,
                    bbox=tuple(float(v) for v in table.bbox),  
                    n_rows=getattr(table, "row_count", len(rows)),
                    n_cols=getattr(table, "col_count", len(rows[0]) if rows else 0),
                )
            )
        except Exception as exc:  
            warnings.append(
                f"page {page.number + 1}: table extraction failed ({exc.__class__.__name__})"
            )
    return items


def parse_page(page: "fitz.Page", *, extract_tables: bool = True, warnings: Optional[List[str]] = None) -> ParsedPage:
    """Build the canonical :class:`ParsedPage` for one PDF page."""
    warnings = warnings if warnings is not None else []
    page_number = page.number + 1
    rect = page.rect
    page_width = float(rect.width)

    table_items = _extract_tables(page, warnings) if extract_tables else []
    table_boxes = [item.bbox for item in table_items]

    text_items: List[_LayoutItem] = []
    try:
        raw = page.get_text("dict")
    except Exception as exc:  
        warnings.append(f"page {page_number}: text extraction failed ({exc.__class__.__name__})")
        raw = {"blocks": []}

    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        bbox: Bbox = tuple(float(v) for v in block.get("bbox", (0, 0, 0, 0)))  # type: ignore[assignment]
        if any(_overlap_ratio(bbox, tb) > 0.5 for tb in table_boxes):
            continue
        text = _join_lines(_lines_from_block(block))
        if not text.strip():
            continue
        text_items.append(_LayoutItem(kind=BlockKind.TEXT, text=text, bbox=bbox))

    ordered = _reading_order(text_items + table_items, page_width)

    blocks: List[PageBlock] = []
    buffer: List[str] = []
    cursor = 0
    for order_index, item in enumerate(ordered):
        if buffer:
            cursor += len(BLOCK_SEPARATOR)
            buffer.append(BLOCK_SEPARATOR)
        start = cursor
        buffer.append(item.text)
        cursor += len(item.text)
        blocks.append(
            PageBlock(
                kind=item.kind,
                order_index=order_index,
                text=item.text,
                char_start=start,
                char_end=cursor,
                bbox=item.bbox,
                n_rows=item.n_rows,
                n_cols=item.n_cols,
            )
        )

    try:
        image_count = len(page.get_images(full=True))
    except Exception:  # pragma: no cover
        image_count = 0

    return ParsedPage(
        page_number=page_number,
        text="".join(buffer),
        blocks=blocks,
        width=page_width,
        height=float(rect.height),
        image_count=image_count,
    )

def compute_file_id(sha256_hex: str, length: int = 16) -> str:
    """Short, stable document identifier derived from the file's content hash."""
    return sha256_hex[:length]


def _hash_path(path: Path, chunk_size: int = 1 << 20) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


@contextmanager
def open_document(source: Union[str, Path, bytes, bytearray]) -> Iterator[Tuple["fitz.Document", str, int, str]]:
    """Open a PDF from a path or raw bytes.

    Yields ``(document, sha256_hex, byte_size, default_name)``. Paths are hashed
    by streaming so a 300 MB filing is never held in memory twice.
    """
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        sha256_hex = hashlib.sha256(data).hexdigest()
        document = fitz.open(stream=data, filetype="pdf")
        default_name = f"upload-{compute_file_id(sha256_hex)}.pdf"
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"no such PDF: {path}")
        sha256_hex, size = _hash_path(path)
        document = fitz.open(path)
        default_name = path.name

    try:
        if getattr(document, "needs_pass", False) and not document.authenticate(""):
            raise ValueError("PDF is password-protected and cannot be parsed")
        size = len(data) if isinstance(source, (bytes, bytearray)) else size
        yield document, sha256_hex, size, default_name
    finally:
        document.close()


def iter_parsed_pages(
    source: Union[str, Path, bytes, bytearray],
    *,
    page_numbers: Optional[Iterable[int]] = None,
    extract_tables: bool = True,
) -> Iterator[ParsedPage]:
    """Stream pages one at a time — the memory-safe path for very large PDFs.

    Callers that only need to extract facts page-by-page (the ingestion worker)
    should use this instead of :func:`parse_pdf`, which materializes everything.
    """
    with open_document(source) as (document, _sha, _size, _name):
        targets = (
            sorted({n for n in page_numbers if 1 <= n <= document.page_count})
            if page_numbers is not None
            else range(1, document.page_count + 1)
        )
        for page_number in targets:
            yield parse_page(document[page_number - 1], extract_tables=extract_tables)


def parse_pdf(
    source: Union[str, Path, bytes, bytearray],
    *,
    filename: Optional[str] = None,
    page_numbers: Optional[Iterable[int]] = None,
    max_pages: Optional[int] = None,
    extract_tables: bool = True,
) -> ParsedDocument:
    """Parse a PDF into its canonical, provenance-ready representation.

    Args:
        source: filesystem path or raw PDF bytes (Streamlit uploads give bytes).
        filename: display name; defaults to the path name or a content-derived name.
        page_numbers: optional 1-based subset to parse.
        max_pages: hard cap, applied after ``page_numbers``, to bound cost on
            very large filings during interactive use.
        extract_tables: set ``False`` to skip table detection (roughly 3–5x
            faster on dense financial pages, at the cost of row/column context).

    The returned ``file_id`` is content-derived, so re-uploading the same PDF
    yields the same id and the same fact ids — incremental ingestion is a
    natural consequence rather than a special case.
    """
    warnings: List[str] = []
    pages: List[ParsedPage] = []

    with open_document(source) as (document, sha256_hex, byte_size, default_name):
        page_count = document.page_count
        targets: List[int] = (
            sorted({n for n in page_numbers if 1 <= n <= page_count})
            if page_numbers is not None
            else list(range(1, page_count + 1))
        )
        if max_pages is not None and len(targets) > max_pages:
            warnings.append(f"truncated to first {max_pages} of {len(targets)} pages")
            targets = targets[:max_pages]

        for page_number in targets:
            pages.append(
                parse_page(
                    document[page_number - 1],
                    extract_tables=extract_tables,
                    warnings=warnings,
                )
            )

        metadata = {
            key: str(value)
            for key, value in (document.metadata or {}).items()
            if value not in (None, "")
        }

    parsed = ParsedDocument(
        file_id=compute_file_id(sha256_hex),
        filename=filename or default_name,
        sha256=sha256_hex,
        byte_size=byte_size,
        page_count=page_count,
        pages=pages,
        metadata=metadata,
        parser_version=PARSER_VERSION,
        warnings=warnings,
    )
    if parsed.needs_ocr:
        object.__setattr__(
            parsed,
            "warnings",
            parsed.warnings
            + ["most pages have no text layer; document likely needs OCR before extraction"],
        )
    return parsed


def chunk_page(
    page: ParsedPage,
    *,
    file_id: str,
    max_chars: int = 3500,
    overlap: int = 250,
    document_name: Optional[str] = None,
) -> List[TextChunk]:
    """Split a page into extraction windows that keep page-absolute offsets.

    Boundaries prefer paragraph breaks, then line breaks, then sentence ends, so
    a table row or a sentence is rarely cut in half. The overlap exists so a
    fact whose statement straddles a boundary is still fully visible in one
    window; duplicate facts produced by the overlap collapse automatically
    because ``AtomicFact.id`` is a content-and-position hash.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap = max(0, min(overlap, max_chars // 2))

    text = page.text
    chunks: List[TextChunk] = []
    if not text.strip():
        return chunks

    position = 0
    index = 0
    length = len(text)
    while position < length:
        end = min(length, position + max_chars)
        if end < length:
            window = text[position:end]
            floor = int(max_chars * 0.5)
            for separator in (BLOCK_SEPARATOR, "\n", ". ", " "):
                cut = window.rfind(separator)
                if cut > floor:
                    end = position + cut + len(separator)
                    break
        segment = text[position:end]
        if segment.strip():
            chunks.append(
                TextChunk(
                    file_id=file_id,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=segment,
                    char_start=position,
                    char_end=end,
                    document_name=document_name,
                )
            )
            index += 1
        if end >= length:
            break
        next_position = max(end - overlap, position + 1)
        # Snap the overlap back to a line start so a window never opens in the
        # middle of a table row; the extractor needs the row label to read a cell.
        line_start = text.rfind("\n", position, next_position) + 1
        if line_start > position and (end - line_start) <= max_chars:
            next_position = line_start
        position = next_position
    return chunks


def chunk_document(
    document: ParsedDocument, *, max_chars: int = 3500, overlap: int = 250
) -> List[TextChunk]:
    """Chunk every page of a parsed document, preserving page-absolute offsets."""
    chunks: List[TextChunk] = []
    for page in document.iter_pages():
        chunks.extend(
            chunk_page(
                page,
                file_id=document.file_id,
                max_chars=max_chars,
                overlap=overlap,
                document_name=document.filename,
            )
        )
    return chunks

def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the canonical text this parser produces.")
    parser.add_argument("pdf", type=Path, help="path to a PDF")
    parser.add_argument("--page", type=int, default=None, help="1-based page to dump")
    parser.add_argument("--quote", type=str, default=None, help="quote to locate on --page")
    parser.add_argument("--no-tables", action="store_true", help="disable table detection")
    parser.add_argument("--json", action="store_true", help="emit a structural summary as JSON")
    args = parser.parse_args(argv)

    document = parse_pdf(
        args.pdf,
        page_numbers=[args.page] if args.page else None,
        extract_tables=not args.no_tables,
    )

    if args.json:
        summary = {
            "file_id": document.file_id,
            "filename": document.filename,
            "page_count": document.page_count,
            "parsed_pages": len(document.pages),
            "total_chars": document.total_chars,
            "needs_ocr": document.needs_ocr,
            "warnings": document.warnings,
            "pages": [
                {
                    "page_number": p.page_number,
                    "chars": p.char_count,
                    "blocks": len(p.blocks),
                    "tables": p.table_count,
                }
                for p in document.pages
            ],
        }
        print(json.dumps(summary, indent=2))
        return 0

    for page in document.pages:
        print(f"===== page {page.page_number} ({page.char_count} chars, {page.table_count} tables) =====")
        print(page.text)
        if args.quote:
            match = locate_quote(page.text, args.quote)
            print("\n----- quote lookup -----")
            print(f"strategy={match.strategy.value} score={match.score} span={match.as_tuple()}")
            print(f"matched={match.matched_text!r}")
    for warning in document.warnings:
        print(f"[warn] {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":  
    raise SystemExit(_cli())


__all__ = [
    "PARSER_VERSION",
    "DEFAULT_FUZZY_THRESHOLD",
    "QuoteMatch",
    "normalize_for_match",
    "locate_quote",
    "locate_quote_offsets",
    "build_anchor",
    "verify_anchor",
    "parse_page",
    "parse_pdf",
    "iter_parsed_pages",
    "open_document",
    "chunk_page",
    "chunk_document",
    "compute_file_id",
]