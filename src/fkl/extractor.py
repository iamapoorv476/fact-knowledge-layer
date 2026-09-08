"""
extractor.py — Turning parsed pages into grounded atomic facts.

The pipeline
------------
    parse_pdf  →  classify blocks  →  extract  →  ground  →  dedupe  →  facts

**Classification comes first and it matters more than the extractors.** In the
Economic Survey, 46 of the 70 blocks that PyMuPDF calls "tables" are actually
chart furniture — axis ticks and legends harvested into rows like
``Real GDP 100 Real GDP Growth (RHS) 20 90 15 80 10 70 crore 5 60``. Any
extractor pointed at that will mint facts out of gridlines. So every block is
routed to a role first, and only DATA_TABLE and PROSE blocks are ever read.

**Three deterministic extractors, one optional model.**

* ``TableFactExtractor`` reads Markdown tables whose header row contains
  periods. The corner cell usually carries the unit (``₹ Cr``), which is how a
  bare ``8,142`` becomes ₹81.42 billion.
* ``OrphanRowExtractor`` handles the borderless statement layout that defeats
  table detection: a label line followed by bare numeric lines, with the column
  periods inherited from the ruled header band above. Confidence is lower and
  the binding is recorded as inferred, because it is.
* ``ProseFactExtractor`` catches sentence-shaped claims using generic English
  patterns ("X was N unit in PERIOD"), not domain vocabulary.
* ``LLMFactExtractor`` handles semantic and awkwardly-phrased facts. It is
  optional: with no API key the pipeline still produces the deterministic facts,
  so the project runs and can be evaluated without credentials.

**The grounding gate is the same for all four.** A fact is kept only if its
quote resolves to a character span in the parsed page. Deterministic extractors
know their offsets exactly; the model's quotes go through ``build_anchor`` and
are dropped when they do not resolve. Nothing ungrounded is ever stored.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

try:
    from .normalize import (
        Dimension,
        Quantity,
        UnitContext,
        UnitSpec,
        build_unit_context,
        parse_number,
        parse_quantity,
        parse_unit,
    )
    from .parser import build_anchor
    from .schemas import (
        AtomicFact,
        BlockKind,
        MatchStrategy,
        PageBlock,
        ParsedDocument,
        ParsedPage,
        PeriodType,
        SourceAnchor,
        TemporalScope,
        TextChunk,
        ValueKind,
        collapse_whitespace,
        normalize_key,
    )
except ImportError:  # pragma: no cover - flat script layout
    from normalize import (  # type: ignore[no-redef]
        Dimension,
        Quantity,
        UnitContext,
        UnitSpec,
        build_unit_context,
        parse_number,
        parse_quantity,
        parse_unit,
    )
    from parser import build_anchor  # type: ignore[no-redef]
    from schemas import (  # type: ignore[no-redef]
        AtomicFact,
        BlockKind,
        MatchStrategy,
        PageBlock,
        ParsedDocument,
        ParsedPage,
        PeriodType,
        SourceAnchor,
        TemporalScope,
        TextChunk,
        ValueKind,
        collapse_whitespace,
        normalize_key,
    )

EXTRACTOR_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Block classification
# --------------------------------------------------------------------------- #
class BlockRole(str, Enum):
    """What a block is, which decides whether and how it is read."""

    DATA_TABLE = "DATA_TABLE"      # real tabular data with period columns
    CHART = "CHART"                # axis labels and legends masquerading as a table
    PROSE = "PROSE"                # sentences
    HEADING = "HEADING"            # short title line
    RUNNING = "RUNNING"            # header/footer repeated across pages
    BOILERPLATE = "BOILERPLATE"    # disclaimers, safe-harbour text
    EMPTY = "EMPTY"


_CHART_CAPTION_RE = re.compile(r"^\s*\|?\s*(chart|figure|fig\.|graph|exhibit)\s*[\w.]*\s*[:.]", re.I)
_CHART_MARKER_RE = re.compile(r"\(RHS\)|\(LHS\)|right axis|left axis|y-axis|x-axis", re.I)
_BOILERPLATE_RE = re.compile(
    r"forward[- ]looking statements|safe harbou?r|this presentation is prepared|"
    r"no part of it shall|disclaimer|all rights reserved|private and confidential",
    re.I,
)


@dataclass(frozen=True)
class ClassifiedBlock:
    """A page block with its role and the reason the classifier chose it."""

    page_number: int
    block: PageBlock
    role: BlockRole
    reason: str


def _table_rows(text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= {"-", ":"} and c for c in cells):
            continue  # separator row
        rows.append(cells)
    return rows


def _empty_ratio(rows: Sequence[Sequence[str]]) -> float:
    total = sum(len(r) for r in rows)
    if not total:
        return 1.0
    empty = sum(1 for r in rows for c in r if not c)
    return empty / total


def _has_period_header(rows: Sequence[Sequence[str]]) -> bool:
    for row in rows[:3]:
        known = sum(1 for cell in row if cell and TemporalScope.parse(cell).is_known)
        if known >= 2:
            return True
    return False


def _numeric_cell_ratio(rows: Sequence[Sequence[str]]) -> float:
    cells = [c for r in rows for c in r if c]
    if not cells:
        return 0.0
    numeric = sum(1 for c in cells if parse_number(c) is not None)
    return numeric / len(cells)


_NUMERIC_TOKEN_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w.])")


def _has_axis_tick_run(rows: Sequence[Sequence[str]]) -> bool:
    """True when a single cell holds a run of loose numbers — a chart's axis ticks.

    When PyMuPDF reads a chart as a table, the tick labels of one axis land in
    one cell as ``20 90 15 80 10 70 crore 5 60``. Real table cells hold one
    value. This is the most reliable chart signal in the macro documents, where
    captions are sometimes absent.
    """
    for row in rows:
        for cell in row:
            if len(_NUMERIC_TOKEN_RE.findall(cell)) >= 5:
                return True
    return False


def classify_block(block: PageBlock, page_number: int, *, running_texts: Optional[set] = None) -> ClassifiedBlock:
    """Assign a role to one block."""
    text = block.text
    stripped = text.strip()

    def out(role: BlockRole, reason: str) -> ClassifiedBlock:
        return ClassifiedBlock(page_number=page_number, block=block, role=role, reason=reason)

    if not stripped:
        return out(BlockRole.EMPTY, "no content")
    if running_texts and collapse_whitespace(stripped)[:80] in running_texts:
        return out(BlockRole.RUNNING, "text repeats across many pages")
    if _BOILERPLATE_RE.search(stripped):
        return out(BlockRole.BOILERPLATE, "matches disclaimer language")

    if block.kind is BlockKind.TABLE:
        rows = _table_rows(text)
        if not rows:
            return out(BlockRole.EMPTY, "table rendered with no rows")

        # Positive chart evidence first — these are precise and rarely wrong.
        if _CHART_CAPTION_RE.search(rows[0][0] if rows[0] else ""):
            return out(BlockRole.CHART, "first cell is a chart caption")
        if _CHART_MARKER_RE.search(text[:600]):
            return out(BlockRole.CHART, "contains axis/legend markers")
        if _has_axis_tick_run(rows):
            return out(BlockRole.CHART, "a cell holds a run of loose numbers — axis ticks")

        # Then positive table evidence. This must be checked BEFORE the sparsity
        # heuristics: real statistical annexes (IMF Table 1, the prospectus
        # financial summaries) are wide and mostly empty because the column
        # detector splits finely, and rejecting them on emptiness alone throws
        # away the densest fact sources in the corpus.
        if _has_period_header(rows):
            return out(BlockRole.DATA_TABLE, "header row carries period labels")

        if len(rows) == 1:
            return out(BlockRole.CHART, "single row — no tabular structure")
        if _numeric_cell_ratio(rows) >= 0.25 and len(rows[0]) >= 2:
            return out(BlockRole.DATA_TABLE, "dense numeric grid with row labels")
        if _empty_ratio(rows) > 0.6:
            return out(BlockRole.CHART, f"{_empty_ratio(rows):.0%} of cells empty and no periods")
        return out(BlockRole.CHART, "no period header and sparse numerics")

    if (
        len(stripped) < 60
        and "\n" not in stripped
        and not stripped.endswith((".", ":", ";"))
        and not any(ch.isdigit() for ch in stripped)
    ):
        # A short line carrying digits is a fact-bearing fragment, not a title —
        # PDFs of dense reports split such lines into their own blocks constantly.
        return out(BlockRole.HEADING, "short single line, no digits")
    return out(BlockRole.PROSE, "default prose block")


def find_running_texts(document: ParsedDocument, *, min_fraction: float = 0.25) -> set:
    """Find block texts that repeat across pages — running headers and footers.

    Generic and document-agnostic: whatever appears on a quarter of the pages is
    furniture, whether it says "Annual Report 2023-24" or "IMF Country Report".
    """
    counts: Dict[str, int] = {}
    for page in document.pages:
        for block in page.blocks:
            if block.kind is not BlockKind.TEXT:
                continue
            key = collapse_whitespace(block.text)[:80]
            if len(key) < 8:
                continue
            counts[key] = counts.get(key, 0) + 1
    threshold = max(3, int(len(document.pages) * min_fraction))
    return {key for key, count in counts.items() if count >= threshold}


def classify_document(document: ParsedDocument) -> List[ClassifiedBlock]:
    """Classify every block in a document."""
    running = find_running_texts(document)
    classified: List[ClassifiedBlock] = []
    for page in document.pages:
        for block in page.blocks:
            classified.append(classify_block(block, page.page_number, running_texts=running))
    return classified


# --------------------------------------------------------------------------- #
# Periods
# --------------------------------------------------------------------------- #
_YEAR_ENDED_RE = re.compile(r"\b(?:for the )?(year|quarter|half[- ]year|period) ended\b", re.I)
_FY_END_RE = re.compile(
    r"\b(?:financial|fiscal) year end(?:ed|ing)\s+(\d{1,2})?\s*([A-Za-z]{3,9})\s*(\d{1,2})?", re.I
)
_MONTH_INDEX = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def infer_fiscal_year_end_month(document: ParsedDocument, default: int = 3) -> int:
    """Infer the month a fiscal year ends in from the document's own language.

    Read from the text rather than assumed, so a March-ending Indian filing and a
    December-ending one are both handled without a code change. Falls back to
    ``default`` when the document never says.
    """
    votes: Dict[int, int] = {}
    for page in document.pages[:20]:
        for match in _FY_END_RE.finditer(page.text):
            month = _MONTH_INDEX.get(match.group(2)[:3].casefold())
            if month:
                votes[month] = votes.get(month, 0) + 1
    if votes:
        return max(votes.items(), key=lambda kv: kv[1])[0]
    return default


@dataclass(frozen=True)
class ResolvedPeriod:
    """A column header turned into a period label the reconciler can compare."""

    label: str
    source_text: str
    refined: bool = False


class PeriodResolver:
    """Turns header cells like ``FY24`` or ``March 31, 2024`` into period labels.

    A bare date in a column header is ambiguous: on a balance sheet it is a point
    in time, on a P&L it is the year that ended on that date. The surrounding
    "for the year ended" language disambiguates it, so that context is passed in
    rather than guessed.
    """

    def __init__(self, fiscal_year_end_month: int = 3) -> None:
        self.fiscal_year_end_month = fiscal_year_end_month

    def resolve(self, cell: str, *, context: str = "") -> Optional[ResolvedPeriod]:
        text = collapse_whitespace(cell)
        if not text:
            return None
        if len(text) > 28:
            # Wide header cells swallow a whole statement title. Keep only the
            # period expression, or the fact ends up scoped to a sentence.
            shortened = extract_period_label(text)
            if shortened is None:
                return None
            text = shortened
        scope = TemporalScope.parse(text)
        if not scope.is_known:
            return None
        if scope.period_type.value != "POINT_IN_TIME" or scope.as_of is None:
            return ResolvedPeriod(label=text, source_text=text)

        flow = _YEAR_ENDED_RE.search(context or "")
        if not flow:
            return ResolvedPeriod(label=text, source_text=text)

        as_of = scope.as_of
        fiscal_year = as_of.year if as_of.month <= self.fiscal_year_end_month else as_of.year + 1
        kind = flow.group(1).casefold()
        if kind == "year":
            return ResolvedPeriod(label=f"FY{fiscal_year}", source_text=text, refined=True)
        if kind == "quarter":
            offset = (as_of.month - self.fiscal_year_end_month - 1) % 12
            quarter = offset // 3 + 1
            return ResolvedPeriod(label=f"Q{quarter} FY{fiscal_year}", source_text=text, refined=True)
        if kind.startswith("half"):
            half = 1 if (as_of.month - self.fiscal_year_end_month - 1) % 12 < 6 else 2
            return ResolvedPeriod(label=f"H{half} FY{fiscal_year}", source_text=text, refined=True)
        return ResolvedPeriod(label=text, source_text=text)


# --------------------------------------------------------------------------- #
# Grounding helpers
# --------------------------------------------------------------------------- #
def anchor_at(
    page: ParsedPage,
    *,
    file_id: str,
    char_start: int,
    char_end: int,
    document_name: Optional[str] = None,
) -> Optional[SourceAnchor]:
    """Build an anchor from offsets the extractor already knows exactly.

    Deterministic extractors do not need to search for their quote — they read
    it out of a known span. The span is still validated against the page text so
    an off-by-one in a slicing loop cannot silently produce a bad citation.
    """
    if char_start < 0 or char_end <= char_start or char_end > len(page.text):
        return None
    quote = page.text[char_start:char_end]
    if not quote.strip():
        return None
    block = page.block_at(char_start)
    return SourceAnchor(
        file_id=file_id,
        page_number=page.page_number,
        verbatim_quote=quote,
        char_start=char_start,
        char_end=char_end,
        match_strategy=MatchStrategy.EXACT,
        match_score=1.0,
        document_name=document_name,
        block_kind=(block.kind.value if block else "text"),  # type: ignore[arg-type]
    )


def _value_kind(unit: Optional[UnitSpec]) -> ValueKind:
    if unit is None:
        return ValueKind.NUMBER
    return {
        Dimension.CURRENCY: ValueKind.MONEY,
        Dimension.PERCENT: ValueKind.PERCENT,
    }.get(unit.dimension, ValueKind.NUMBER)


#: A leading list-numbering marker on a table row label: "I. Total ...",
#: "(ii) Private", "A. Government", "1. Others", or a compound hierarchical
#: form like "II.1 Manufacturing", "II.a. Infrastructure", "II.5.8 New
#: issuances...". Common in national-accounts and budget-style tables that
#: number their row categories through several levels. The marker carries no
#: measurement meaning, so it is dropped rather than treated as part of the
#: attribute's name.
_LIST_MARKER_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"\((?:[ivxlcdm]{1,4}|[a-z]|\d{1,3})\)"
    r"|"
    r"(?:[IVXLCDM]{1,4}|[A-Za-z]|\d{1,3})(?:\.(?:\d{1,3}|[a-z]))*\.?"
    r")"
    r"\s+(?=[A-Za-z])"
)


def _clean_label(text: str) -> str:
    """Strip list markers, footnote markers and trailing punctuation from a label."""
    cleaned = collapse_whitespace(text)
    cleaned = _LIST_MARKER_RE.sub("", cleaned)
    cleaned = re.sub(r"\s*\(\d{1,2}(?:,\d{1,2})*\)\s*$", "", cleaned)  # "(1)", "(1,2)"
    cleaned = re.sub(r"[*†‡#]+\s*$", "", cleaned)
    cleaned = cleaned.rstrip(" :;.")
    return collapse_whitespace(cleaned)


_CONTINUATION_LABEL_RE = re.compile(
    r"^\s*(?:%|per\s?cent|margin|of\s+revenue|of\s+total|yoy|y-o-y|qoq|q-o-q|growth|change)\b"
    r"|^\s*%\s",
    re.I,
)


def _is_continuation_label(label: str) -> bool:
    """True for row labels that qualify the row above rather than name a measure."""
    return bool(_CONTINUATION_LABEL_RE.match(label))


def _looks_like_label(text: str) -> bool:
    letters = sum(1 for c in text if c.isalpha())
    return letters >= 3 and letters / max(1, len(text)) > 0.4


# --------------------------------------------------------------------------- #
# Extraction context
# --------------------------------------------------------------------------- #
@dataclass
class ExtractionContext:
    """Everything an extractor needs about the document it is reading."""

    document: ParsedDocument
    subject: str
    unit_context: UnitContext
    periods: PeriodResolver
    classified: Dict[int, List[ClassifiedBlock]] = field(default_factory=dict)

    @property
    def file_id(self) -> str:
        return self.document.file_id

    @property
    def name(self) -> str:
        return self.document.filename

    def blocks_on(self, page_number: int) -> List[ClassifiedBlock]:
        return self.classified.get(page_number, [])

    def heading_before(self, page_number: int, position: int) -> Optional[str]:
        """The nearest heading above a position on the page.

        A page of an annual report can carry a standalone balance sheet and a
        consolidated one, or a company's statement next to an acquiree's. The
        rows are identical; only the heading distinguishes them. Without it the
        reconciler sees two different values for the same line item in the same
        period and has no choice but to call it a contradiction.
        """
        best: Optional[str] = None
        for item in self.classified.get(page_number, []):
            if item.role is not BlockRole.HEADING:
                continue
            if item.block.char_start <= position:
                best = collapse_whitespace(item.block.text)[:120]
            else:
                break
        return best


_SUBJECT_PATTERNS = [
    re.compile(
        r"\b([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,4}\s+"
        r"(?:Limited|LIMITED|Ltd\.?|LTD\.?|Incorporated|PLC|Corporation|CORPORATION))\b"
    ),
    re.compile(
        r"\b((?:Reserve\s+Bank|RESERVE\s+BANK|Republic|Ministry|International\s+Monetary\s+Fund)"
        r"(?:\s+of\s+[A-Z][A-Za-z]+)?)\b"
    ),
]

#: Words that are capitalised everywhere in filings and name nothing.
_SUBJECT_STOPWORDS = {
    "the", "our", "this", "that", "these", "such", "its", "their", "his", "her",
    "annual", "report", "company", "companies", "limited", "ltd", "board",
    "directors", "director", "group", "note", "notes", "total", "table", "chart",
    "figure", "source", "particulars", "financial", "statements", "statement",
    "year", "quarter", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "january", "february",
    "fiscal", "act", "section", "schedule", "annexure", "page", "rs", "inr",
    "crore", "million", "billion", "lakh", "per", "cent", "and", "for", "of",
    "india's", "chapter", "part", "sub", "no", "yes", "government", "ministry",
    "bank", "offer", "equity", "shares", "share", "prospectus", "audit",
}

_CAPITALISED_TOKEN_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,})\b")

_GENERIC_ORG_PREFIXES = ("our ", "the ", "this ", "such ", "its ", "a ", "an ")

#: Signals used by institutional "country report" style documents, where the
#: publisher (IMF, World Bank, OECD, ...) is explicitly not the subject — the
#: country is. None of these patterns name any specific country; they match
#: the structural convention such reports use to state their subject.
_COUNTRY_REPORT_LABEL_RE = re.compile(r"Country\s+Report\b", re.I)
_STANDALONE_CAPS_LINE_RE = re.compile(r"^[ \t]*([A-Z][A-Z\s]{2,40})[ \t]*$", re.M)
_TITLE_FOR_COUNTRY_RE = re.compile(
    r"\bFOR\s+(?:THE\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b\.?\s*$", re.M
)


def _country_report_subject(document: ParsedDocument) -> Optional[str]:
    """Detect the "this is a report about X, published by Y" pattern.

    IMF/World Bank/OECD-style country reports state their subject two ways
    near the top, independent of each other: a short, standalone all-caps line
    right after the report's own label ("IMF Country Report No. 25/314" /
    "INDIA"), and a title ending "...FOR <COUNTRY>" or "...FOR THE <COUNTRY>".
    When either fires, it overrides the publisher-name heuristic below, which
    would otherwise report the institution that wrote the document rather than
    the country the document is about.
    """
    front = "\n".join(page.text for page in document.pages[:3])
    label_match = _COUNTRY_REPORT_LABEL_RE.search(front)
    if label_match is None:
        return None

    window = front[label_match.end() : label_match.end() + 200]
    caps_match = _STANDALONE_CAPS_LINE_RE.search(window)
    if caps_match:
        candidate = collapse_whitespace(caps_match.group(1))
        if 2 <= len(candidate) <= 40 and candidate.casefold() not in _SUBJECT_STOPWORDS:
            return candidate.title()

    for match in _TITLE_FOR_COUNTRY_RE.finditer(front):
        candidate = collapse_whitespace(match.group(1))
        if candidate.casefold() not in _SUBJECT_STOPWORDS:
            return candidate

    return None


def infer_subject(document: ParsedDocument) -> str:
    """Guess the entity a document is primarily about.

    Three passes, in priority order. First, a structural "country report"
    signal that catches the case where the publisher is explicitly not the
    subject. Second, the most frequent distinctive proper noun across the
    opening pages — "Delhivery" in the company filings, "India" in the
    institutional reports that aren't country-report-labelled. Third, an
    organisation-shaped name containing that noun, so "Delhivery" is promoted
    to "Delhivery Limited".

    Getting this right matters more than it looks: the entity is half of the
    blocking key, so a document whose subject resolves differently from its peers
    can never corroborate or contradict them.
    """
    country_report_subject = _country_report_subject(document)
    if country_report_subject is not None:
        return country_report_subject

    # Organisation names on the front pages are the strongest available signal:
    # a filing repeats its own legal name in the cover, the letterhead and the
    # signature block. Matched case-insensitively because covers are set in caps.
    # Organisation names on the cover and letterhead are the strongest signal a
    # filing gives about itself. Restricting candidates to the first two pages
    # keeps out institutions the body merely cites — which is what stops an
    # economic survey from being attributed to a fund it quotes four times.
    cover_counts: Dict[str, int] = {}
    display: Dict[str, str] = {}
    for page in document.pages[:2]:
        for name in _organisation_names(page.text):
            key = name.casefold()
            cover_counts[key] = cover_counts.get(key, 0) + 1
            display.setdefault(key, name)

    if cover_counts:
        # A cover can name several parties (an earnings letter is addressed to
        # the exchanges; a prospectus lists selling shareholders). Break ties on
        # how often each name recurs in the body: the filer's own name is
        # everywhere, a counterparty's is not.
        document_counts: Dict[str, int] = {key: 0 for key in cover_counts}
        for page in document.pages:
            lowered = page.text.casefold()
            for key in document_counts:
                document_counts[key] += lowered.count(key)
        best = max(cover_counts, key=lambda k: (cover_counts[k], document_counts[k]))
        return _titlecase_name(display.get(best, best))

    # No organisation named up front — an institutional report about a place or
    # an economy. Fall back to the most frequent *standalone* proper noun: a
    # token with no capitalised neighbour, so "Selling" in "Selling Shareholders"
    # and "Reserve" in "Reserve Bank" are excluded as parts of noun phrases,
    # while "India" in "India's exports" counts.
    counts: Dict[str, int] = {}
    display: Dict[str, str] = {}
    for page in document.pages:
        tokens = list(_CAPITALISED_TOKEN_RE.finditer(page.text))
        for index, match in enumerate(tokens):
            token = match.group(1)
            key = token.casefold()
            if len(key) < 4 or key in _SUBJECT_STOPWORDS:
                continue
            if not (token[0].isupper() and token[1:].islower()):
                continue
            previous = tokens[index - 1] if index else None
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if previous is not None and previous.end() + 1 >= match.start():
                continue
            if following is not None and match.end() + 1 >= following.start():
                continue
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, token)
    if counts:
        head_key = max(counts.items(), key=lambda kv: kv[1])[0]
        return display.get(head_key, head_key.title())

    head_key = None
    org_counts = {}
    title = document.metadata.get("title", "").strip()
    if title and len(title) <= 80:
        return collapse_whitespace(title)
    return "Unknown subject"


def _organisation_names(text: str) -> List[str]:
    """All organisation-shaped names in a block of text, cleaned of stray lead-ins."""
    names: List[str] = []
    for pattern in _SUBJECT_PATTERNS:
        for match in pattern.finditer(text):
            name = _strip_leading_stopwords(collapse_whitespace(match.group(1)))
            if 4 <= len(name) <= 60 and " " in name:
                names.append(name)
    return names


def _strip_leading_stopwords(name: str) -> str:
    """Drop sentence words the regex swept up before the real name.

    "For Delhivery Limited" and "by Delhivery Limited" must collapse to the same
    candidate, or the counts that decide the subject get split three ways.
    """
    words = name.split()
    while words and words[0].casefold() in _SUBJECT_STOPWORDS:
        words.pop(0)
    return " ".join(words)


def _titlecase_name(name: str) -> str:
    """Render a matched organisation name in title case, keeping small words down."""
    small = {"of", "and", "the", "for"}
    parts = []
    for index, word in enumerate(name.split()):
        parts.append(word if word.isupper() and len(word) <= 3 else
                     (word.lower() if index and word.lower() in small else word.capitalize()))
    return " ".join(parts)


_POSSESSIVE_RE = re.compile(r"^([A-Z][\w.&\-]*(?:\s+[A-Z][\w.&\-]*){0,3})['’]s\s+(.{3,})$")


def split_possessive_entity(attribute: str, default_entity: str) -> Tuple[str, str]:
    """Turn "India's current account deficit" into ("India", "current account deficit").

    Prose names its subject inline far more often than tables do, and honouring
    that is what lets a fact from a report *about* India carry the entity India
    rather than the entity of whoever published the report.
    """
    match = _POSSESSIVE_RE.match(attribute.strip())
    if not match:
        return default_entity, attribute
    entity = collapse_whitespace(match.group(1))
    if entity.casefold() in _SUBJECT_STOPWORDS:
        return default_entity, attribute
    return entity, collapse_whitespace(match.group(2))


def build_context(
    document: ParsedDocument, *, subject_override: Optional[str] = None
) -> ExtractionContext:
    """Assemble the per-document context: subject, units, periods, block roles.

    ``subject_override`` exists because subject inference is a heuristic and the
    entity is half of every blocking key. When a reviewer can see it guessed
    wrong, correcting it should not require a code change.
    """
    classified = classify_document(document)
    by_page: Dict[int, List[ClassifiedBlock]] = {}
    for item in classified:
        by_page.setdefault(item.page_number, []).append(item)
    return ExtractionContext(
        document=document,
        subject=subject_override or infer_subject(document),
        unit_context=build_unit_context(document),
        periods=PeriodResolver(infer_fiscal_year_end_month(document)),
        classified=by_page,
    )


# --------------------------------------------------------------------------- #
# Table extraction
# --------------------------------------------------------------------------- #
_CAPTION_ENTITY_RE = re.compile(
    r"\b(?:Table|Statement|Appendix|Annex(?:ure)?|Chart|Exhibit)\s*[\dIVXivx.]*\s*[.:]?\s*"
    r"([A-Z][A-Za-z .&\-]{2,40}?)\s*:",
)


def caption_entity(lead_in: str, rows: Sequence["TableRow"]) -> Optional[str]:
    """Pull the entity out of a caption like "Table 1. India: Selected Indicators".

    Institutional reports name the subject of a statistical table in its caption,
    and that subject is frequently *not* the publisher — an IMF table about India
    should produce facts about India, not about the Fund.
    """
    candidates = [lead_in[-300:]]
    if rows and rows[0].cells:
        candidates.append(" ".join(cell.text for cell in rows[0].cells[:3]))
    for text in candidates:
        match = _CAPTION_ENTITY_RE.search(text)
        if match:
            name = collapse_whitespace(match.group(1))
            if 2 < len(name) <= 40 and name.casefold() not in _SUBJECT_STOPWORDS:
                return name
    return None


@dataclass(frozen=True)
class TableCell:
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class TableRow:
    cells: List[TableCell]
    line_start: int
    line_end: int
    text: str


def parse_markdown_table(block: PageBlock) -> List[TableRow]:
    """Re-read a rendered Markdown table, recovering each cell's page offsets.

    The parser wrote these tables, so this is not fragile text scraping — the
    offsets it recovers are the same ones the page text was assembled from, which
    is what lets a cell citation point at the exact figure rather than the row.
    """
    rows: List[TableRow] = []
    cursor = block.char_start
    for line in block.text.split("\n"):
        line_start = cursor
        cursor += len(line) + 1  # +1 for the newline separator
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells_text = stripped.strip("|").split("|")
        if all(set(c.strip()) <= {"-", ":"} and c.strip() for c in cells_text):
            continue

        offset = line_start + line.index("|") + 1
        cells: List[TableCell] = []
        for raw_cell in cells_text:
            text = raw_cell.strip()
            if text:
                lead = len(raw_cell) - len(raw_cell.lstrip())
                cells.append(
                    TableCell(text=text, char_start=offset + lead, char_end=offset + lead + len(text))
                )
            else:
                cells.append(TableCell(text="", char_start=offset, char_end=offset))
            offset += len(raw_cell) + 1
        rows.append(
            TableRow(cells=cells, line_start=line_start, line_end=line_start + len(line), text=line)
        )
    return rows


@dataclass(frozen=True)
class TableHeader:
    """Which columns of a table are periods, plus the table's unit.

    ``value_columns`` is the column layout actually used for binding: every
    column with a resolvable unit, mapped to the period it belongs to. For a
    plain table this is identical to ``periods`` (one value column per period).
    For a table that interleaves an absolute figure with a percentage under
    one period — "(₹ in million) | % of revenue" repeated per year — it also
    includes the percent columns, each carrying the period of the money column
    to its left. Without that, a row scanner sees only as many "slots" as
    ``periods`` has entries and binds a percentage into the next year's money
    figure by mistake.
    """

    row_index: int
    periods: Dict[int, ResolvedPeriod]
    unit: Optional[UnitSpec]
    skipped_columns: List[str]
    value_columns: Dict[int, Tuple[ResolvedPeriod, Optional[UnitSpec]]] = field(default_factory=dict)


def detect_table_header(
    rows: Sequence[TableRow], resolver: PeriodResolver, *, context: str = ""
) -> Optional[TableHeader]:
    """Find the header row and map column index → period.

    Handles the two-row headers common in Indian annual reports, where the first
    row says ``Particulars | Current`` and the second carries the dates. Columns
    whose header is not a period (``QoQ%``, ``YoY%``, ``Remarks``) are recorded
    as skipped rather than silently misread as data.
    """
    best: Optional[TableHeader] = None
    for index, row in enumerate(rows[:3]):
        periods: Dict[int, ResolvedPeriod] = {}
        skipped: List[str] = []
        for column, cell in enumerate(row.cells):
            if not cell.text:
                continue
            resolved = resolver.resolve(cell.text, context=context)
            if resolved is not None:
                periods[column] = resolved
            elif column != 0:
                # Column 0 is normally the row-label corner and its failure to
                # parse as a period is expected, not a skipped column. But some
                # tables (IMF-style annexes) have no label column at all — every
                # header cell, including the first, is a period. Only count a
                # genuine skip when column 0 *isn't* one of those all-period
                # headers, which the loop already handles by simply including
                # it in ``periods`` whenever it does resolve.
                skipped.append(cell.text)
        if len(periods) >= 2:
            corner = row.cells[0].text if row.cells else ""
            unit = parse_unit(corner)
            if unit is None and index > 0 and rows[0].cells:
                unit = parse_unit(rows[0].cells[0].text)
            if index > 0:
                periods = _qualify_with_year_row(periods, rows[index - 1])
            candidate = TableHeader(row_index=index, periods=periods, unit=unit, skipped_columns=skipped)
            if best is None or len(candidate.periods) > len(best.periods):
                best = candidate

    if best is not None:
        labels = [resolved.label for resolved in best.periods.values()]
        if len(set(labels)) < len(labels):
            # "Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4" across two years with no year row to
            # disambiguate them. Binding a value to "Q1" would silently pick one
            # of two different quarters, so the whole table is refused.
            return None
        value_columns = _expand_value_columns(rows, best)
        best = TableHeader(
            row_index=best.row_index,
            periods=best.periods,
            unit=best.unit,
            skipped_columns=best.skipped_columns,
            value_columns=value_columns,
        )
    return best


def _expand_value_columns(
    rows: Sequence[TableRow], header: TableHeader
) -> Dict[int, Tuple[ResolvedPeriod, Optional[UnitSpec]]]:
    """Map every value-bearing column to its period and its own unit.

    Looks a few rows below the recognised period row for a per-column unit
    declaration ("(₹ in million) | % of revenue ..."). When one exists, every
    unit-bearing column is kept — not just the ones that carried period text —
    and each is assigned the period of the nearest money-bearing column to its
    left. When no such row exists (the ordinary case), this returns exactly
    one slot per recognised period with no per-column unit override, which is
    the same layout callers relied on before this function existed.
    """
    best_row: Optional[TableRow] = None
    best_hits = 0
    for row in rows[header.row_index + 1 : header.row_index + 5]:
        if not row.cells:
            continue
        first = _clean_label(row.cells[0].text) if row.cells[0].text else ""
        if first and _looks_like_label(first):
            break  # real data has started; no further header rows to check
        hits = sum(1 for cell in row.cells[1:] if cell.text and parse_unit(cell.text) is not None)
        if hits > best_hits:
            best_hits, best_row = hits, row

    if best_row is None or best_hits < 2:
        return {column: (period, None) for column, period in header.periods.items()}

    # Assign each unit-bearing non-period column to the nearest period column
    # to its left — but only within a short span, and never past the next
    # period column. Without that bound, columns belonging to a period this
    # table never resolved (a third header row this parser didn't capture)
    # would silently inherit whatever period happened to be last recognised,
    # producing several different values for one period instead of leaving
    # that period unrecognised.
    period_columns = sorted(header.periods)
    max_gap = 2
    value_columns: Dict[int, Tuple[ResolvedPeriod, Optional[UnitSpec]]] = {}
    for column, cell in enumerate(best_row.cells):
        if column in header.periods:
            unit = parse_unit(cell.text) if cell.text else header.unit
            value_columns[column] = (header.periods[column], unit)
            continue
        if not cell.text:
            continue
        unit = parse_unit(cell.text)
        if unit is None:
            continue
        preceding = [p for p in period_columns if p < column]
        if not preceding:
            continue
        nearest = max(preceding)
        if column - nearest > max_gap:
            continue
        later = min((p for p in period_columns if p > nearest), default=None)
        if later is not None and column >= later:
            continue
        value_columns[column] = (header.periods[nearest], unit)

    # A period column whose own unit-row cell was blank (the currency is
    # declared once and only the percent column repeats a label) still needs a
    # slot, falling back to the table-level unit.
    for column, period in header.periods.items():
        value_columns.setdefault(column, (period, header.unit))

    return value_columns


def _qualify_with_year_row(
    periods: Dict[int, ResolvedPeriod], year_row: "TableRow"
) -> Dict[int, ResolvedPeriod]:
    """Attach years from a spanning header row to bare quarter/half labels.

    Financial tables stack headers: a year row above ("FY24 ... FY25") and a
    period row below ("Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4"). Spanning labels are
    left-aligned over the columns they cover, so each period takes the nearest
    year at or to its left.
    """
    years: Dict[int, str] = {}
    for column, cell in enumerate(year_row.cells):
        if not cell.text:
            continue
        scope = TemporalScope.parse(cell.text)
        if scope.is_known and scope.year and scope.period_type in (
            PeriodType.FISCAL_YEAR,
            PeriodType.CALENDAR_YEAR,
        ):
            years[column] = f"FY{scope.year}"
    if not years:
        return periods

    qualified: Dict[int, ResolvedPeriod] = {}
    for column, resolved in periods.items():
        scope = TemporalScope.parse(resolved.label)
        if scope.year is not None or scope.period_type not in (
            PeriodType.QUARTER,
            PeriodType.HALF,
            PeriodType.MONTHS,
        ):
            qualified[column] = resolved
            continue
        candidates = [c for c in years if c <= column]
        if not candidates:
            qualified[column] = resolved
            continue
        year_label = years[max(candidates)]
        qualified[column] = ResolvedPeriod(
            label=f"{resolved.label} {year_label}",
            source_text=resolved.source_text,
            refined=True,
        )
    return qualified


class TableFactExtractor:
    """Reads facts out of DATA_TABLE blocks."""

    name = "table-v1"

    def extract(self, context: ExtractionContext, page: ParsedPage) -> Tuple[List[AtomicFact], Dict[str, int]]:
        facts: List[AtomicFact] = []
        stats: Dict[str, int] = {"tables_read": 0, "tables_no_header": 0, "cells_unparsed": 0, "columns_skipped": 0}

        for classified in context.blocks_on(page.page_number):
            if classified.role is not BlockRole.DATA_TABLE:
                continue
            block = classified.block
            rows = parse_markdown_table(block)
            if len(rows) < 2:
                continue
            lead_in = page.text[max(0, block.char_start - 400) : block.char_start]
            header = detect_table_header(rows, context.periods, context=lead_in + page.text[:200])
            if header is None:
                stats["tables_no_header"] += 1
                continue
            stats["tables_read"] += 1
            entity = caption_entity(lead_in, rows) or context.subject
            stats["columns_skipped"] += len(header.skipped_columns)

            table_unit = header.unit
            if table_unit is None:
                resolved, source = context.unit_context.resolve(page.page_number, block.char_start)
                # Only inherit a unit declared on this page. Falling back to the
                # document default across an entire report attaches "₹ million"
                # to production indices and percentages several chapters away.
                table_unit = resolved if source.startswith("page-declaration") else None

            section: Optional[str] = None
            last_label: Optional[str] = None
            for row in rows[header.row_index + 1 :]:
                if not row.cells:
                    continue
                label = _clean_label(row.cells[0].text)
                if not label or not _looks_like_label(label):
                    continue
                if _is_continuation_label(label):
                    # "% margin" and "% of revenue" rows describe the line item
                    # printed above them. Taken literally, one table yields four
                    # different values for one attribute in one period, and the
                    # reconciler has no honest choice but to call that a
                    # contradiction. The label belongs to its parent row.
                    if last_label:
                        label = f"{last_label} — {label}"
                else:
                    last_label = label
                populated = [c for i, c in enumerate(row.cells) if i > 0 and c.text]
                if not populated:
                    collapsed = self._collapsed_row(row, header, table_unit, context, page, section)
                    if collapsed:
                        facts.extend(collapsed)
                        stats["collapsed_rows_bound"] = stats.get("collapsed_rows_bound", 0) + 1
                    elif _NUMERIC_TOKEN_RE.search(row.cells[0].text):
                        stats["collapsed_rows_unbound"] = stats.get("collapsed_rows_unbound", 0) + 1
                    else:
                        section = label  # a lone label row is a section heading
                    continue

                for column, (period, column_unit) in header.value_columns.items():
                    if column >= len(row.cells):
                        continue
                    cell = row.cells[column]
                    if not cell.text:
                        continue
                    cell_unit = column_unit if column_unit is not None else table_unit
                    quantity = parse_quantity(cell.text, fallback_unit=cell_unit)
                    if quantity is None:
                        if cell.text.strip() not in {"-", "–", "—", ""}:
                            stats["cells_unparsed"] += 1
                        continue
                    anchor = anchor_at(
                        page,
                        file_id=context.file_id,
                        char_start=row.line_start,
                        char_end=row.line_end,
                        document_name=context.name,
                    )
                    if anchor is None:
                        continue
                    qualifiers = {"column": period.source_text, "cell": cell.text}
                    if section:
                        qualifiers["section"] = section
                    heading = context.heading_before(page.page_number, block.char_start)
                    if heading:
                        qualifiers["context"] = heading
                    if period.refined:
                        qualifiers["period_inferred_from"] = "column date + 'year ended' context"
                    if column_unit is not None and column_unit.dimension is Dimension.PERCENT:
                        qualifiers["value_type"] = "share of total"
                    facts.append(
                        self._make_fact(
                            context, label, quantity, period, anchor, qualifiers, row.text, entity
                        )
                    )
        return facts, stats

    def _collapsed_row(
        self,
        row: TableRow,
        header: TableHeader,
        unit: Optional[UnitSpec],
        context: ExtractionContext,
        page: ParsedPage,
        section: Optional[str],
    ) -> List[AtomicFact]:
        """Recover a row whose columns collapsed into a single cell.

        The IMF statistical annexes arrive as one cell per row::

            Real GDP growth (percent) 6.5 6.6 6.2 6.4 6.5 6.5

        The label and the values are all there; only the column boundaries are
        gone. They can be rebound positionally, but *only* when the number of
        values matches the number of period columns exactly. Any other count
        means the row also carries footnote digits or merged columns, and a
        guess there would produce confidently-wrong facts — so those rows are
        counted as unbound and dropped instead.
        """
        cell = row.cells[0]
        text = cell.text
        matches = list(_NUMERIC_TOKEN_RE.finditer(text))
        periods = [header.periods[key] for key in sorted(header.periods)]
        if len(matches) != len(periods) or not matches:
            return []
        label = _clean_label(text[: matches[0].start()])
        if not label or not _looks_like_label(label):
            return []

        inline_unit = parse_unit(label) or unit
        facts: List[AtomicFact] = []
        for period, match in zip(periods, matches):
            quantity = parse_quantity(match.group(0), fallback_unit=inline_unit)
            if quantity is None:
                continue
            anchor = anchor_at(
                page,
                file_id=context.file_id,
                char_start=row.line_start,
                char_end=row.line_end,
                document_name=context.name,
            )
            if anchor is None:
                continue
            qualifiers = {
                "column": period.source_text,
                "cell": match.group(0),
                "binding": "positional — column boundaries lost in extraction",
            }
            if section:
                qualifiers["section"] = section
            facts.append(
                AtomicFact(
                    entity=context.subject,
                    attribute=_clean_label(label),
                    value=quantity.value,
                    unit=inline_unit.describe() if inline_unit else None,
                    temporal_scope=period.label,
                    raw_statement=collapse_whitespace(row.text)[:400],
                    provenance=anchor,
                    confidence=0.6,
                    value_kind=_value_kind(inline_unit),
                    normalized_value=quantity.canonical_value,
                    normalized_unit=(inline_unit.currency or inline_unit.canonical) if inline_unit else None,
                    qualifiers=qualifiers,
                    extractor=self.name,
                )
            )
        return facts

    def _make_fact(
        self,
        context: ExtractionContext,
        label: str,
        quantity: Quantity,
        period: ResolvedPeriod,
        anchor: SourceAnchor,
        qualifiers: Dict[str, str],
        raw_statement: str,
        entity: Optional[str] = None,
    ) -> AtomicFact:
        unit = quantity.unit
        return AtomicFact(
            entity=entity or context.subject,
            attribute=label,
            value=quantity.value,
            unit=unit.describe() if unit else None,
            temporal_scope=period.label,
            raw_statement=collapse_whitespace(raw_statement)[:400],
            provenance=anchor,
            confidence=0.9 if unit is not None else 0.78,
            value_kind=_value_kind(unit),
            normalized_value=quantity.canonical_value,
            normalized_unit=(unit.currency or unit.canonical) if unit else None,
            qualifiers=qualifiers,
            extractor=self.name,
        )


# --------------------------------------------------------------------------- #
# Orphan-row extraction (borderless statements)
# --------------------------------------------------------------------------- #
_BULLET_ARTIFACT_RE = re.compile(r"^[\x00-\x1f\x7f]?[A-Za-z\u2022\u25cf\u25aa\u00b7]?$")


def _is_bullet_artifact(line: str) -> bool:
    """A physical line that is just a stray bullet-glyph, not real text.

    Some PDFs set bullet points in a symbol font whose glyph table maps the
    bullet shape to an ordinary character — a lowercase "y", a bell control
    character, a bare bullet dot. PyMuPDF extracts exactly what the font's
    cmap says, so this is a real character in the text layer, not a parsing
    error; it just is not part of any label and must not become one.
    """
    stripped = line.strip()
    return len(stripped) <= 1 and bool(_BULLET_ARTIFACT_RE.match(stripped))


def _trailing_scale_word(parts: Sequence[str]) -> Optional[UnitSpec]:
    """True when the last folded label line is a bare scale word, not a name.

    ``parse_unit("million")`` already resolves to a valid ``UnitSpec`` with
    ``Dimension.UNKNOWN`` — a pure multiplier with no attached measure. That is
    the signature of a stray unit declaration that line-wrapped onto its own
    line ("Express parcel" / "million" / numbers...). A word carrying a real
    dimension, like "days" in "Net Working Capital Days", must not be stripped
    this way since there it is genuinely part of the measure's name.
    """
    if len(parts) < 2:
        return None
    candidate = parse_unit(parts[-1])
    if candidate is not None and candidate.dimension is Dimension.UNKNOWN:
        return candidate
    return None


def _looks_like_note_reference(quantity: Quantity) -> bool:
    raw = quantity.raw.strip()
    return "." not in raw and "," not in raw and abs(quantity.value) < 100


class OrphanRowExtractor:
    """Recovers facts from statement rows that lost their table structure.

    Indian financial statements are typeset without cell borders, so
    ``find_tables()`` catches only the ruled header band and the body arrives as
    plain lines::

        Advance from Customers
        397.51
        408.28

    The label and its values are adjacent and the column periods are available
    from the header table immediately above, so the pairing is recoverable — but
    it is an *inference about column order*, not something read off the page.
    Facts from here are marked with lower confidence and a ``binding`` qualifier
    saying so, which is the honest way to ship a partial fix.
    """

    name = "orphan-row-v1"

    def extract(self, context: ExtractionContext, page: ParsedPage) -> Tuple[List[AtomicFact], Dict[str, int]]:
        facts: List[AtomicFact] = []
        stats: Dict[str, int] = {"orphan_rows": 0, "orphan_blocks_scanned": 0}

        blocks = context.blocks_on(page.page_number)
        for index, classified in enumerate(blocks):
            if classified.role is not BlockRole.PROSE:
                continue
            header = self._preceding_header(context, blocks, index, page)
            if header is None:
                continue
            stats["orphan_blocks_scanned"] += 1
            columns = sorted(header.value_columns) if header.value_columns else sorted(header.periods)
            periods = [
                (header.value_columns[key][0] if header.value_columns else header.periods[key])
                for key in columns
            ]
            column_units = [
                (header.value_columns[key][1] if header.value_columns else None) for key in columns
            ]
            default_unit = header.unit
            if default_unit is None:
                resolved, source = context.unit_context.resolve(
                    page.page_number, classified.block.char_start
                )
                # Same rule as the table and prose paths: a unit declared on
                # this page can be inherited, a document-wide default cannot.
                # Without this guard, a deck whose financial slides dominate
                # the unit count (e.g. mostly "₹ Cr") leaks that currency onto
                # an unrelated page of pure counts, like an operating-metrics
                # table of gateways and sort centers.
                default_unit = resolved if source.startswith("page-declaration") else None
            facts.extend(
                self._scan_block(
                    context, page, classified.block, periods, default_unit, stats, column_units=column_units
                )
            )
        return self._drop_page_conflicts(facts, stats), stats

    @staticmethod
    def _drop_page_conflicts(
        facts: List[AtomicFact], stats: Dict[str, int]
    ) -> List[AtomicFact]:
        """Discard rows that produced two different values for one measure and period.

        A page often holds several statements with identical row labels. When
        positional binding yields "Services = 16,538.97" and "Services =
        1,320.09" for the same fiscal year on the same page, at least one is
        bound to the wrong table — and nothing here can tell which. Emitting
        both would manufacture a contradiction out of an extraction failure, so
        both are dropped and counted.
        """
        grouped: Dict[Tuple[str, Optional[str], str], List[AtomicFact]] = {}
        for fact in facts:
            # value_kind is part of the key: a row legitimately printing both an
            # absolute figure and its "% of revenue" under one label is not two
            # conflicting claims about one number, it is two different measures
            # that happen to share a label. Only same-kind values competing for
            # one label and period are a real binding conflict.
            grouped.setdefault(
                (fact.attribute_key, fact.temporal_scope, fact.value_kind.value), []
            ).append(fact)
        kept: List[AtomicFact] = []
        for members in grouped.values():
            values = {round(float(m.value), 6) for m in members if m.is_numeric}
            if len(values) > 1:
                stats["orphan_page_conflicts_dropped"] = (
                    stats.get("orphan_page_conflicts_dropped", 0) + len(members)
                )
                continue
            kept.extend(members)
        return kept

    def _preceding_header(
        self,
        context: ExtractionContext,
        blocks: Sequence[ClassifiedBlock],
        index: int,
        page: ParsedPage,
    ) -> Optional[TableHeader]:
        """The nearest DATA_TABLE header above this block, within two blocks."""
        for previous in reversed(blocks[max(0, index - 2) : index]):
            if previous.role is not BlockRole.DATA_TABLE:
                continue
            rows = parse_markdown_table(previous.block)
            lead_in = page.text[max(0, previous.block.char_start - 400) : previous.block.char_start]
            return detect_table_header(rows, context.periods, context=lead_in)
        return None

    def _scan_block(
        self,
        context: ExtractionContext,
        page: ParsedPage,
        block: PageBlock,
        periods: Sequence[ResolvedPeriod],
        unit: Optional[UnitSpec],
        stats: Dict[str, int],
        column_units: Optional[Sequence[Optional[UnitSpec]]] = None,
    ) -> List[AtomicFact]:
        facts: List[AtomicFact] = []
        lines: List[Tuple[str, int, int]] = []
        cursor = block.char_start
        for line in block.text.split("\n"):
            lines.append((line, cursor, cursor + len(line)))
            cursor += len(line) + 1

        position = 0
        while position < len(lines):
            label_line, label_start, label_end = lines[position]
            if _is_bullet_artifact(label_line):
                # Some PDFs render a bullet point through a symbol font whose
                # glyph table maps the bullet to an ordinary letter or control
                # character — this document's bullets decode as "y" or "\x07".
                # PyMuPDF is extracting exactly what the font declares; the fix
                # belongs here, not in the parser.
                position += 1
                continue
            label = _clean_label(label_line)
            if not label or not _looks_like_label(label) or parse_number(label_line) is not None:
                position += 1
                continue

            # A row label can span several physical lines ("Revenue from" /
            # "Part Truck" / "Load Services"). Fold forward while the next line
            # still reads as more label text, so the fact is attributed to the
            # whole phrase rather than only whichever fragment sat directly
            # above the first number.
            label_parts = [label]
            merge_probe = position + 1
            while merge_probe < len(lines) and len(label_parts) < 6:
                next_line, _next_start, _next_end = lines[merge_probe]
                next_text = _clean_label(next_line)
                if not next_text or not _looks_like_label(next_text) or parse_number(next_line) is not None:
                    break
                label_parts.append(next_text)
                merge_probe += 1
            label = " ".join(label_parts)

            # A last part that is nothing but a bare scale word ("million",
            # "crore") is a stray unit declaration that wrapped onto its own
            # line, not part of the measure's name — pull it out as a unit
            # hint instead of leaving "million" as the attribute.
            row_unit = unit
            scale_hint = _trailing_scale_word(label_parts)
            if scale_hint is not None:
                label = " ".join(label_parts[:-1]) or label
                if row_unit is None:
                    row_unit = scale_hint

            values: List[Tuple[Quantity, int, int]] = []
            probe = merge_probe
            while probe < len(lines) and len(values) < len(periods):
                candidate, start, end = lines[probe]
                text = candidate.strip()
                if not text:
                    break
                offset = len(values)
                slot_unit = column_units[offset] if column_units and offset < len(column_units) else None
                quantity = parse_quantity(text, fallback_unit=slot_unit if slot_unit is not None else row_unit)
                if quantity is None or _looks_like_label(text):
                    break
                values.append((quantity, start, end))
                probe += 1

            if (
                len(values) > 1
                and _looks_like_note_reference(values[0][0])
                and any(("." in v[0].raw or "," in v[0].raw) for v in values[1:])
            ):
                # Indian statements print a "Notes" column before the figures:
                # "Other income | 22 | 4,530.15 | 3,050.00". The 22 is a
                # cross-reference, and binding it positionally shifts every
                # value in the row into the wrong year.
                values = values[1:]
                stats["orphan_note_columns_dropped"] = (
                    stats.get("orphan_note_columns_dropped", 0) + 1
                )

            if len(values) == 1 and _looks_like_note_reference(values[0][0]):
                # "Property, plant and equipment\n3" — in a notes section a lone
                # small integer with no decimal or thousands separator is almost
                # always a cross-reference to another note, not a figure.
                stats["orphan_note_refs_skipped"] = stats.get("orphan_note_refs_skipped", 0) + 1
                position += 1
                continue

            if len(values) >= 1 and len(values) <= len(periods):
                stats["orphan_rows"] += 1
                for offset, (quantity, start, end) in enumerate(values):
                    period = periods[offset]
                    slot_unit = column_units[offset] if column_units and offset < len(column_units) else None
                    value_unit = quantity.unit or slot_unit or row_unit
                    anchor = anchor_at(
                        page,
                        file_id=context.file_id,
                        char_start=label_start,
                        char_end=end,
                        document_name=context.name,
                    )
                    if anchor is None:
                        continue
                    qualifiers = {
                        key: value
                        for key, value in {
                            "binding": "inferred from column order — table borders absent",
                            "column": period.source_text,
                            "context": context.heading_before(page.page_number, block.char_start),
                        }.items()
                        if value
                    }
                    if slot_unit is not None and slot_unit.dimension is Dimension.PERCENT:
                        qualifiers["value_type"] = "share of total"
                    facts.append(
                        AtomicFact(
                            entity=context.subject,
                            attribute=label,
                            value=quantity.value,
                            unit=value_unit.describe() if value_unit else None,
                            temporal_scope=period.label,
                            raw_statement=collapse_whitespace(
                                f"{label}: {quantity.raw} ({period.source_text})"
                            ),
                            provenance=anchor,
                            confidence=0.55,
                            value_kind=_value_kind(value_unit),
                            normalized_value=quantity.canonical_value,
                            normalized_unit=(value_unit.currency or value_unit.canonical) if value_unit else None,
                            qualifiers=qualifiers,
                            extractor=self.name,
                        )
                    )
                position = probe
            else:
                position += 1
        return facts


# --------------------------------------------------------------------------- #
# Prose extraction
# --------------------------------------------------------------------------- #
_VERB_PATTERN = (
    r"(?:was|were|is|are|stood at|stands at|amounted to|amounts to|totalled|totaled|"
    r"reached|rose to|grew to|increased to|declined to|decreased to|fell to|moderated to|"
    r"remained at|came in at|is projected at|projected at|estimated at|recorded at|of)"
)
_VALUE_PATTERN = (
    r"(?P<value>(?:₹|Rs\.?|INR|USD|US\$|\$)?\s?\(?-?\d[\d,]*(?:\.\d+)?\)?\s*"
    r"(?:per cent|percent|%|bps|basis points|crore|crores|cr|lakh|lakhs|million|mn|billion|bn|"
    r"trillion|days|tonnes|tons|mn tons|sq\.? ?ft\.?)?)"
)
_PROSE_RE = re.compile(
    rf"(?P<attr>[A-Za-z][A-Za-z0-9 ,'’\-\(\)/&\.]{{4,90}}?)\s+{_VERB_PATTERN}\s+{_VALUE_PATTERN}(?![\w.\-])",
    re.I,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9₹])")

_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
#: A compact period expression. Used to pull the *label* out of a sentence, so a
#: fact's temporal scope is "Q2 FY25" and not the 200-word paragraph it sat in.
_PERIOD_TOKEN_RE = re.compile(
    r"(?:Q[1-4]\s*(?:FY|CY)?\s*['’]?\d{2,4}"
    r"|H[12]\s*(?:FY|CY)?\s*['’]?\d{2,4}"
    r"|\d{1,2}\s?M\s*FY\s*['’]?\d{2,4}"
    r"|(?:FY|CY)\s*['’]?\d{2,4}(?:\s*[-/]\s*\d{2,4})?"
    r"|(?:fiscal|financial)\s+(?:year\s+)?['’]?\d{2,4}"
    rf"|(?:{_MONTH_NAMES})\.?\s+\d{{1,2}},?\s+\d{{4}}"
    rf"|\d{{1,2}}\s+(?:{_MONTH_NAMES})\.?,?\s+\d{{4}}"
    r"|\d{4}\s?[-/]\s?\d{2,4}"
    r"|\bTTM\b"
    r"|\b(?:19|20)\d{2}\b)",
    re.I,
)


def extract_period_label(text: str) -> Optional[str]:
    """Return the first period expression in ``text``, or None."""
    match = _PERIOD_TOKEN_RE.search(text)
    return collapse_whitespace(match.group(0)) if match else None


def nearest_period_label(sentence: str, start: int, end: int, *, window: int = 120) -> Optional[str]:
    """The period expression closest to a specific match, not just the first in the sentence.

    A long sentence can legitimately name more than one year — a statute's
    enactment year, a regulation, and the year the fact actually happened
    ("...allotted equity shares after expiry of 60 days ... and violated the
    Foreign Exchange Management Act, 1999"). The value here is "60 days"; 1999
    is a citation over a hundred characters away. Taking whichever period
    appears first in the sentence binds facts to statute years, filing years,
    and other incidental dates that have nothing to do with the claim. This
    looks in a window immediately around the match and only widens to the
    whole sentence — preserving the old behaviour — when nothing is close by.
    """
    lo, hi = max(0, start - window), min(len(sentence), end + window)
    before = list(_PERIOD_TOKEN_RE.finditer(sentence[lo:start]))
    after = _PERIOD_TOKEN_RE.search(sentence[end:hi])

    candidates: List[Tuple[int, str]] = []
    if before:
        closest_before = before[-1]
        distance = start - (lo + closest_before.end())
        candidates.append((distance, collapse_whitespace(closest_before.group(0))))
    if after:
        candidates.append((after.start(), collapse_whitespace(after.group(0))))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    # Deliberately no whole-sentence fallback here. A period far outside the
    # window is exactly the failure this function exists to avoid — a statute
    # year or an unrelated citation elsewhere in a long sentence is not this
    # claim's period, and a fact with no defensible period should be dropped,
    # not guessed.
    return None


#: Function words a regex sweeps up in front of the real attribute.
_ATTRIBUTE_LEAD_WORDS = {
    "in", "as", "of", "and", "or", "but", "each", "from", "with", "to", "for",
    "by", "on", "at", "that", "which", "while", "where", "the", "a", "an",
    "this", "these", "those", "its", "their", "our", "we", "it", "there",
    "however", "although", "though", "since", "when", "than", "then", "also",
    "against", "about", "into", "over", "under", "per", "was", "were", "is",
    "are", "been", "being", "had", "has", "have",
    "still", "meanwhile", "moreover", "furthermore", "notably", "importantly",
    "similarly", "conversely", "additionally", "accordingly", "thus",
    "therefore", "nevertheless", "nonetheless", "following", "consequently",
    "given", "amid", "amidst", "despite", "besides", "hence", "so",
}

#: Auxiliary/helper verbs left dangling at the end of a captured attribute when
#: the matched reporting verb is a two-word phrase ("has declined to" — the
#: regex matches only "declined to", leaving "has" stuck to the attribute).
_TRAILING_AUXILIARY_WORDS = {
    "has", "have", "had", "is", "are", "was", "were", "be", "been", "being",
    "will", "would", "can", "could", "should", "shall", "may", "might",
    "must", "did", "do", "does",
}

_ATTRIBUTE_NUMBER_RE = re.compile(r"\d[\d,]{2,}|\b(?:19|20)\d{2}\b|\b(?:FY|CY|Q[1-4])\s?\d{2,4}\b", )

#: Pronouns that refer to a specific, unnamed third party — never to the
#: document's own subject. "its total revenue" almost always means the
#: document's subject's revenue, so stripping "its" and defaulting to that
#: subject is correct; "he received a compensation of ₹X" refers to whichever
#: named director's biography the sentence sits in, which this system cannot
#: resolve. Attributing that fact to the document subject would be wrong, not
#: just untidy, so these are rejected outright rather than cleaned up.
_UNRESOLVED_PRONOUN_SUBJECTS = {"he", "she", "him", "her", "himself", "herself"}


def clean_attribute_phrase(text: str) -> Optional[str]:
    """Trim a regex-captured phrase down to a usable attribute, or reject it.

    The prose pattern captures whatever precedes the verb, which routinely picks
    up sentence connectives ("in FY24 as against a compound annual growth rate")
    and fragments of adjacent clauses ("each, 300,000 preference shares"). An
    attribute that still contains an embedded figure or year is not an attribute
    at all — it is a slice of a sentence, and a fact built on it would be
    grounded in real text yet describe nothing. Those are rejected outright.
    """
    words = collapse_whitespace(text).split()
    words = _LIST_MARKER_RE.sub("", " ".join(words)).split()
    while words and (
        words[0].strip(",;:").casefold() in _ATTRIBUTE_LEAD_WORDS
        or _is_bullet_artifact(words[0])
    ):
        words.pop(0)
    phrase = " ".join(words).strip(" ,;:.-")

    # A prefatory clause ("Under staff's baseline scenario, real GDP growth")
    # leaves its trailing half as the only part that is actually a measure
    # name. A comma inside a genuine attribute is otherwise vanishingly rare
    # in financial and macro text, so preferring the segment after the last
    # comma is safe and fixes this whole class of capture in one place.
    if "," in phrase:
        tail = phrase.rsplit(",", 1)[1].strip(" ,;:.-")
        tail_tokens = tail.split()
        while tail_tokens and tail_tokens[0].strip(",;:").casefold() in _ATTRIBUTE_LEAD_WORDS:
            tail_tokens.pop(0)
        tail = " ".join(tail_tokens)
        if len(tail.split()) >= 2:
            phrase = tail

    # A trailing auxiliary verb ("Headline inflation has" — the regex matched
    # only "declined to", leaving "has" stuck to the end of the attribute) is
    # dropped the same way a leading one already is.
    tail_words = phrase.split()
    while tail_words and tail_words[-1].strip(",;:.").casefold() in _TRAILING_AUXILIARY_WORDS:
        tail_words.pop()
    phrase = " ".join(tail_words)

    # Checked last, on the fully-cleaned phrase, because the pronoun is often
    # only exposed after the comma-split above runs — "In Fiscal 2021, he
    # received..." only becomes "he received..." once the prefatory clause is
    # already gone.
    leading_word = phrase.split()[0].strip(",;:").casefold() if phrase else ""
    if leading_word in _UNRESOLVED_PRONOUN_SUBJECTS:
        return None

    if len(phrase) < 6 or len(phrase) > 60:
        return None
    if len(phrase.split()) < 2:
        return None
    if _ATTRIBUTE_NUMBER_RE.search(phrase):
        return None
    if not any(c.isalpha() for c in phrase):
        return None
    return phrase


_BARE_YEAR_VALUE_RE = re.compile(r"^\s*\(?(19|20)\d{2}\)?\s*$")


def _is_bare_year(raw_value: str) -> bool:
    return bool(_BARE_YEAR_VALUE_RE.match(raw_value))


class ProseFactExtractor:
    """Extracts sentence-shaped numeric claims using generic English patterns.

    Deliberately conservative — it requires an explicit copula or reporting verb,
    a parseable quantity, and a resolvable unit. It will miss plenty of prose
    facts; the LLM extractor exists for those. What it must not do is invent
    facts, because a wrong fact with a real citation is worse than no fact.
    """

    name = "prose-v1"

    def __init__(self, *, min_confidence: float = 0.55) -> None:
        self.min_confidence = min_confidence

    def extract(self, context: ExtractionContext, page: ParsedPage) -> Tuple[List[AtomicFact], Dict[str, int]]:
        facts: List[AtomicFact] = []
        stats: Dict[str, int] = {"prose_sentences": 0, "prose_rejected_no_unit": 0, "prose_rejected_no_period": 0}

        for classified in context.blocks_on(page.page_number):
            if classified.role is not BlockRole.PROSE:
                continue
            block = classified.block
            for sentence, sentence_start in self._sentences(block):
                stats["prose_sentences"] += 1
                for match in _PROSE_RE.finditer(sentence):
                    period_label = nearest_period_label(sentence, match.start(), match.end())
                    if period_label is None:
                        stats["prose_rejected_no_period"] += 1
                        continue
                    attribute = clean_attribute_phrase(match.group("attr"))
                    if attribute is None:
                        stats["prose_rejected_bad_attribute"] = (
                            stats.get("prose_rejected_bad_attribute", 0) + 1
                        )
                        continue
                    entity, attribute = split_possessive_entity(attribute, context.subject)
                    raw_value = match.group("value")
                    inline_unit = parse_quantity(raw_value)
                    if inline_unit is None:
                        continue
                    if inline_unit.unit is None and _is_bare_year(raw_value):
                        # "the Act of 1999" is not ₹1,999 — a bare year with no
                        # unit of its own is a date, and inheriting the page's
                        # currency would turn every statutory reference into money.
                        stats["prose_rejected_bare_year"] = (
                            stats.get("prose_rejected_bare_year", 0) + 1
                        )
                        continue
                    unit = inline_unit.unit
                    if unit is None:
                        resolved, source = context.unit_context.resolve(
                            page.page_number, sentence_start + match.start()
                        )
                        # Same rule as the table path: a unit declared on this
                        # page can be inherited, a document-wide default cannot.
                        # Otherwise "the monthly injury rate stood at 0.20"
                        # becomes ₹0.2 million because a note page said so.
                        unit = resolved if source.startswith("page-declaration") else None
                    if unit is None:
                        stats["prose_rejected_no_unit"] += 1
                        continue
                    quantity = parse_quantity(raw_value, fallback_unit=unit)
                    if quantity is None:
                        continue

                    start = sentence_start + match.start()
                    end = sentence_start + match.end()
                    anchor = anchor_at(
                        page,
                        file_id=context.file_id,
                        char_start=start,
                        char_end=end,
                        document_name=context.name,
                    )
                    if anchor is None:
                        continue
                    facts.append(
                        AtomicFact(
                            entity=entity,
                            attribute=attribute,
                            value=quantity.value,
                            unit=unit.describe(),
                            temporal_scope=period_label,
                            raw_statement=collapse_whitespace(sentence)[:400],
                            provenance=anchor,
                            confidence=self.min_confidence + (0.1 if inline_unit.unit else 0.0),
                            value_kind=_value_kind(unit),
                            normalized_value=quantity.canonical_value,
                            normalized_unit=(unit.currency or unit.canonical),
                            qualifiers={"source": "prose"},
                            extractor=self.name,
                        )
                    )
        return facts, stats

    @staticmethod
    def _sentences(block: PageBlock) -> Iterable[Tuple[str, int]]:
        text = block.text.replace("\n", " ")
        position = 0
        for piece in _SENTENCE_SPLIT_RE.split(text):
            index = text.find(piece, position)
            if index == -1:
                index = position
            position = index + len(piece)
            if 20 < len(piece) <= 600:
                yield piece, block.char_start + index
            elif len(piece) > 600:
                # A 2,000-character run with no sentence break is a table that
                # lost its structure, not prose. Reading it as a sentence
                # produces attributes stitched from unrelated clauses.
                continue


# --------------------------------------------------------------------------- #
# LLM extraction (optional)
# --------------------------------------------------------------------------- #
LLM_SYSTEM_PROMPT = """You extract atomic facts from financial and institutional documents.

Return ONLY a JSON array. No prose, no markdown fences. Each element:
{
  "entity":         string  - who/what the fact is about; use the document subject unless the text names another party
  "attribute":      string  - the measured property, in the document's own words (e.g. "revenue from services")
  "value":          number or string
  "unit":           string or null - e.g. "INR crore", "%", "days", "shipments"
  "temporal_scope": string or null - EXACTLY as written: "FY24", "Q4 FY24", "9M FY23", "as at March 31, 2024"
  "quote":          string  - a VERBATIM span copied character-for-character from the text below
  "confidence":     number between 0 and 1
}

Hard rules:
- The quote MUST appear verbatim in the supplied text. Do not paraphrase, reflow, fix typos, or join lines.
  Any fact whose quote cannot be found is discarded, so copying exactly is the whole job.
- Extract only what is stated. Never compute, infer, annualise, or convert units.
- Prefer facts that carry a period and a unit. Skip decorative or narrative statements.
- If the text is a chart's axis labels or a table of contents, return [].
"""


@dataclass
class LLMConfig:
    """Configuration for the optional model-based extractor."""

    model: str = os.environ.get("FKL_MODEL", "claude-sonnet-5")
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 4096
    timeout: float = 120.0
    max_retries: int = 3
    max_chunks: Optional[int] = None
    base_url: str = "https://api.anthropic.com/v1/messages"


Transport = Callable[[str, str], str]
"""A callable ``(system_prompt, user_prompt) -> raw_model_text``.

Injectable so the pipeline can be tested, replayed from fixtures, or pointed at
a different provider without touching the extraction logic.
"""


class AnthropicTransport:
    """Minimal Anthropic Messages API client over the standard library.

    No SDK dependency and no key in the repository: the key is read from the
    environment at call time, and its absence raises before any network call.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @property
    def available(self) -> bool:
        return bool(os.environ.get(self.config.api_key_env))

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.config.api_key_env} is not set; LLM extraction is unavailable"
            )
        payload = json.dumps(
            {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.config.base_url,
            data=payload,
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return "".join(
                    part.get("text", "") for part in body.get("content", []) if part.get("type") == "text"
                )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in (429, 500, 502, 503, 529):
                    raise
                time.sleep(min(2**attempt, 8))
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Anthropic request failed after {self.config.max_retries} attempts: {last_error}")


def _parse_llm_json(text: str) -> List[dict]:
    """Extract a JSON array from a model response, tolerating stray fencing."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.M).strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)]


class LLMFactExtractor:
    """Model-based extraction for prose and semantic facts, behind a grounding gate.

    Every returned quote is re-located in the page with the parser's match ladder.
    Quotes that do not resolve are counted and dropped — that count is the single
    most useful number for judging whether the prompt is working.
    """

    name = "llm-v1"

    def __init__(
        self,
        transport: Optional[Transport] = None,
        *,
        config: Optional[LLMConfig] = None,
        chunk_chars: int = 3500,
    ) -> None:
        self.config = config or LLMConfig()
        self.transport = transport or AnthropicTransport(self.config)
        self.chunk_chars = chunk_chars

    @property
    def available(self) -> bool:
        transport = self.transport
        return getattr(transport, "available", True)

    def extract_chunk(
        self, context: ExtractionContext, page: ParsedPage, chunk: TextChunk
    ) -> Tuple[List[AtomicFact], Dict[str, int]]:
        stats = {"llm_calls": 0, "llm_returned": 0, "llm_ungrounded": 0, "llm_invalid": 0}
        prompt = (
            f"Document: {context.name}\n"
            f"Document subject (use unless the text names another party): {context.subject}\n"
            f"Page: {page.page_number}\n\n"
            f"TEXT:\n{chunk.text}"
        )
        stats["llm_calls"] += 1
        raw = self.transport(LLM_SYSTEM_PROMPT, prompt)
        items = _parse_llm_json(raw)
        stats["llm_returned"] += len(items)

        facts: List[AtomicFact] = []
        for item in items:
            fact = self._build(context, page, item, stats)
            if fact is not None:
                facts.append(fact)
        return facts, stats

    def _build(
        self, context: ExtractionContext, page: ParsedPage, item: dict, stats: Dict[str, int]
    ) -> Optional[AtomicFact]:
        quote = item.get("quote")
        attribute = item.get("attribute")
        value = item.get("value")
        if not isinstance(quote, str) or not isinstance(attribute, str) or value is None:
            stats["llm_invalid"] += 1
            return None

        anchor = build_anchor(
            page=page,
            file_id=context.file_id,
            quote=quote,
            document_name=context.name,
        )
        if anchor is None:
            stats["llm_ungrounded"] += 1
            return None

        unit_text = item.get("unit")
        unit = parse_unit(unit_text) if isinstance(unit_text, str) else None
        if unit is None:
            unit, _ = context.unit_context.resolve(page.page_number, anchor.char_start)

        if isinstance(value, (int, float)):
            quantity: Optional[Quantity] = parse_quantity(str(value), fallback_unit=unit)
        else:
            quantity = parse_quantity(str(value), fallback_unit=unit)

        confidence = item.get("confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.6
        # A fuzzy-matched quote is weaker evidence than an exact one; the anchor
        # already knows which rung it landed on, so fold that into the score.
        confidence = max(0.1, min(0.9, confidence * anchor.match_score))

        entity = item.get("entity")
        temporal = item.get("temporal_scope")
        return AtomicFact(
            entity=collapse_whitespace(entity) if isinstance(entity, str) and entity.strip() else context.subject,
            attribute=_clean_label(attribute),
            value=quantity.value if quantity is not None else str(value),
            unit=(unit.describe() if unit else (unit_text if isinstance(unit_text, str) else None)),
            temporal_scope=temporal if isinstance(temporal, str) and temporal.strip() else None,
            raw_statement=collapse_whitespace(anchor.verbatim_quote)[:400],
            provenance=anchor,
            confidence=confidence,
            value_kind=_value_kind(unit) if quantity is not None else ValueKind.TEXT,
            normalized_value=quantity.canonical_value if quantity is not None else None,
            normalized_unit=((unit.currency or unit.canonical) if unit else None) if quantity else None,
            qualifiers={"source": "llm"},
            extractor=self.name,
        )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class ExtractionReport(BaseModel):
    """What happened during extraction — the number that matter for debugging."""

    model_config = ConfigDict(extra="forbid")

    file_id: str
    filename: str
    subject: str
    pages_processed: int = 0
    fiscal_year_end_month: int = 3
    unit_declarations: int = 0
    document_default_unit: Optional[str] = None
    blocks_by_role: Dict[str, int] = Field(default_factory=dict)
    facts_by_extractor: Dict[str, int] = Field(default_factory=dict)
    duplicates_collapsed: int = 0
    counters: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    seconds: float = 0.0

    @property
    def total_facts(self) -> int:
        return sum(self.facts_by_extractor.values())

    def summary(self) -> str:
        roles = ", ".join(f"{k}={v}" for k, v in sorted(self.blocks_by_role.items()))
        extractors = ", ".join(f"{k}={v}" for k, v in sorted(self.facts_by_extractor.items()))
        return (
            f"{self.filename}: {self.total_facts} facts from {self.pages_processed} pages "
            f"in {self.seconds:.1f}s\n  subject: {self.subject}\n  blocks: {roles}\n"
            f"  facts: {extractors}\n  units: {self.unit_declarations} declarations, "
            f"default={self.document_default_unit}"
        )


@dataclass
class ExtractionResult:
    facts: List[AtomicFact]
    report: ExtractionReport
    context: ExtractionContext


class ExtractionPipeline:
    """Runs the extractors over a parsed document and returns grounded facts.

    The LLM extractor is opt-in and degrades to a no-op when no key is present,
    so the deterministic path always produces output. Facts are de-duplicated by
    their content-and-position hash, which means overlapping chunks and a second
    extractor finding the same cell collapse to one fact with the best
    confidence, rather than inflating the graph.
    """

    def __init__(
        self,
        *,
        use_llm: bool = False,
        llm: Optional[LLMFactExtractor] = None,
        use_prose: bool = True,
        use_orphan_rows: bool = True,
        chunk_chars: int = 3500,
    ) -> None:
        self.table_extractor = TableFactExtractor()
        self.orphan_extractor = OrphanRowExtractor() if use_orphan_rows else None
        self.prose_extractor = ProseFactExtractor() if use_prose else None
        self.llm_extractor = llm if llm is not None else (LLMFactExtractor() if use_llm else None)
        self.chunk_chars = chunk_chars

    def run(
        self,
        document: ParsedDocument,
        *,
        page_numbers: Optional[Iterable[int]] = None,
        subject_override: Optional[str] = None,
    ) -> ExtractionResult:
        started = time.time()
        context = build_context(document, subject_override=subject_override)
        report = ExtractionReport(
            file_id=document.file_id,
            filename=document.filename,
            subject=context.subject,
            fiscal_year_end_month=context.periods.fiscal_year_end_month,
            unit_declarations=len(context.unit_context),
            document_default_unit=(
                context.unit_context.document_default.describe()
                if context.unit_context.document_default
                else None
            ),
            warnings=list(document.warnings),
        )
        for items in context.classified.values():
            for item in items:
                report.blocks_by_role[item.role.value] = report.blocks_by_role.get(item.role.value, 0) + 1

        targets = set(page_numbers) if page_numbers is not None else None
        collected: Dict[str, AtomicFact] = {}
        duplicates = 0

        for page in document.pages:
            if targets is not None and page.page_number not in targets:
                continue
            report.pages_processed += 1

            for extractor in self._deterministic_extractors():
                facts, stats = extractor.extract(context, page)
                self._merge_counters(report, stats)
                duplicates += self._collect(collected, facts, report, extractor.name)

            if self.llm_extractor is not None and self.llm_extractor.available:
                duplicates += self._run_llm(context, page, collected, report)

        report.duplicates_collapsed = duplicates
        report.seconds = round(time.time() - started, 2)
        facts = sorted(collected.values(), key=lambda f: (f.provenance.page_number, f.provenance.char_start))
        return ExtractionResult(facts=facts, report=report, context=context)

    # -- internals ---------------------------------------------------------- #
    def _deterministic_extractors(self) -> List[object]:
        extractors: List[object] = [self.table_extractor]
        if self.orphan_extractor is not None:
            extractors.append(self.orphan_extractor)
        if self.prose_extractor is not None:
            extractors.append(self.prose_extractor)
        return extractors

    def _run_llm(
        self,
        context: ExtractionContext,
        page: ParsedPage,
        collected: Dict[str, AtomicFact],
        report: ExtractionReport,
    ) -> int:
        try:
            from .parser import chunk_page  # type: ignore[attr-defined]
        except ImportError:  # pragma: no cover
            from parser import chunk_page  # type: ignore[no-redef]

        duplicates = 0
        chunks = chunk_page(
            page,
            file_id=context.file_id,
            max_chars=self.chunk_chars,
            document_name=context.name,
        )
        assert self.llm_extractor is not None
        for chunk in chunks:
            try:
                facts, stats = self.llm_extractor.extract_chunk(context, page, chunk)
            except Exception as exc:  # network, quota, malformed response
                report.warnings.append(f"llm extraction failed on page {page.page_number}: {exc}")
                break
            self._merge_counters(report, stats)
            duplicates += self._collect(collected, facts, report, self.llm_extractor.name)
        return duplicates

    @staticmethod
    def _merge_counters(report: ExtractionReport, stats: Dict[str, int]) -> None:
        for key, value in stats.items():
            report.counters[key] = report.counters.get(key, 0) + value

    @staticmethod
    def _collect(
        collected: Dict[str, AtomicFact],
        facts: Sequence[AtomicFact],
        report: ExtractionReport,
        extractor_name: str,
    ) -> int:
        duplicates = 0
        for fact in facts:
            existing = collected.get(fact.id)
            if existing is None:
                collected[fact.id] = fact
                report.facts_by_extractor[extractor_name] = (
                    report.facts_by_extractor.get(extractor_name, 0) + 1
                )
            else:
                duplicates += 1
                if fact.confidence > existing.confidence:
                    collected[fact.id] = fact
        return duplicates


def extract_facts(
    document: ParsedDocument, **kwargs: object
) -> ExtractionResult:
    """Convenience wrapper: build a default pipeline and run it."""
    return ExtractionPipeline(**kwargs).run(document)  # type: ignore[arg-type]


__all__ = [
    "EXTRACTOR_VERSION",
    "BlockRole",
    "ClassifiedBlock",
    "classify_block",
    "classify_document",
    "find_running_texts",
    "PeriodResolver",
    "ResolvedPeriod",
    "infer_fiscal_year_end_month",
    "infer_subject",
    "split_possessive_entity",
    "extract_period_label",
    "clean_attribute_phrase",
    "ExtractionContext",
    "build_context",
    "anchor_at",
    "parse_markdown_table",
    "detect_table_header",
    "caption_entity",
    "TableFactExtractor",
    "OrphanRowExtractor",
    "ProseFactExtractor",
    "LLMFactExtractor",
    "LLMConfig",
    "AnthropicTransport",
    "LLM_SYSTEM_PROMPT",
    "ExtractionPipeline",
    "ExtractionReport",
    "ExtractionResult",
    "extract_facts",
]