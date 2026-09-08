"""
schemas.py — Core data contracts for the Fact Knowledge Layer.

Design principles
-----------------
1. **Provenance is structural, not decorative.** A fact cannot exist without a
   `SourceAnchor` that resolves to a deterministic character span inside the
   parsed text of a specific page of a specific file. Anything the extractor
   cannot ground is dropped upstream, not stored with a shrug.

2. **Deterministic identity.** `AtomicFact.id` and `FactRelationship.id` are
   content hashes. Re-ingesting the same document produces the same ids, which
   is what makes incremental ingestion (add a 4th PDF without rebuilding the
   graph) safe and idempotent.

3. **Reconciliation lives in the schema, not in prompts.** Entity/attribute
   blocking keys and a parsed `TemporalScope` are computed here so the
   relationship engine can decide "same claim, different period" with plain
   Python comparisons instead of asking a model twice.

4. **No document-specific rules.** Nothing in this file knows about revenue,
   directors, or any particular filing. `entity`/`attribute` are free-form
   strings; the predicate vocabulary is whatever the documents produce.

All models are frozen: once a fact is written it is immutable, and derived
variants are produced with `model_copy(update=...)`.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

SCHEMA_VERSION: str = "1.0.0"

FactValue = Union[float, int, str]

_WS_RE = re.compile(r"\s+")
_NON_KEY_RE = re.compile(r"[^a-z0-9]+")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize_key(text: str) -> str:
    """Lowercase, unaccent and underscore-join a label to build a blocking key.

    ``"Total Revenue (₹)"`` and ``"total   revenue"`` both collapse to
    ``"total_revenue"``, which lets the relationship engine cheaply group
    candidate facts before doing any expensive comparison.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_KEY_RE.sub("_", ascii_text.casefold()).strip("_")


def collapse_whitespace(text: str) -> str:
    """Collapse all whitespace runs to a single space and strip the ends."""
    return _WS_RE.sub(" ", text).strip()


def _stable_hash(*parts: Any, length: int = 16) -> str:
    """Deterministic short hash over the string form of ``parts``."""
    payload = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Temporal scope
# --------------------------------------------------------------------------- #
class PeriodType(str, Enum):
    """The shape of the period a fact is scoped to."""

    FISCAL_YEAR = "FISCAL_YEAR"      # FY22, FY2022-23
    CALENDAR_YEAR = "CALENDAR_YEAR"  # CY2022, 2022
    QUARTER = "QUARTER"              # Q3 FY23
    HALF = "HALF"                    # H1 FY24
    MONTHS = "MONTHS"                # 9M FY23, six months ended ...
    TTM = "TTM"                      # trailing twelve months
    POINT_IN_TIME = "POINT_IN_TIME"  # as on 31 March 2023
    UNSPECIFIED = "UNSPECIFIED"      # no temporal marker found


_MONTHS: Dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Order matters: the most specific pattern wins.
_QUARTER_RE = re.compile(r"\bq([1-4])\b[\s\-_/]*(?:fy|cy)?[\s\-_/]*'?(\d{2,4})?", re.I)
_HALF_RE = re.compile(r"\bh([12])\b[\s\-_/]*(?:fy|cy)?[\s\-_/]*'?(\d{2,4})?", re.I)
_MONTHS_RE = re.compile(r"\b(\d{1,2})\s*m(?:onths?)?\b[\s\-_/]*(?:fy|cy)?[\s\-_/]*'?(\d{2,4})?", re.I)
_TTM_RE = re.compile(r"\b(ttm|trailing twelve months|last twelve months|ltm)\b", re.I)
_FY_RE = re.compile(r"\bfy\s*'?(\d{4}|\d{2})(?:\s*[-/]\s*(\d{2,4}))?\b", re.I)
_CY_RE = re.compile(r"\bcy\s*'?(\d{4}|\d{2})\b", re.I)
_FY_WORDS_RE = re.compile(r"\bfiscal(?:\s+year)?\s*'?(\d{4}|\d{2})\b", re.I)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
#: "End-March 2024", "end December 2024", "as at end-Sep 2023" — a month-end
#: snapshot. Without this they collapse to a bare year, and two snapshots taken
#: nine months apart look like two claims about the same period.
_MONTH_END_RE = re.compile(
    r"\b(?:as\s+(?:at|on)\s+)?end[\s\-]+([a-z]{3,9})\.?,?\s*(\d{4})\b", re.I
)
_DMY_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})\.?,?\s+(\d{4})\b", re.I
)
_MDY_RE = re.compile(r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I)
_BARE_YEAR_RE = re.compile(r"\b(19|20)(\d{2})\b")
#: "2024-25", "2023/24" — the fiscal-year-range notation Indian budget and
#: national-accounts tables use constantly, almost never with an explicit "FY"
#: prefix. Without this, "2024-25" falls all the way through to the bare-year
#: fallback, loses the "-25" entirely, and is typed CALENDAR_YEAR — so it can
#: never compare equal to "FY2024/25" from a document that did write "FY",
#: even though both almost certainly mean the same reporting year.
_FISCAL_YEAR_RANGE_RE = re.compile(r"\b(19|20)(\d{2})[-/](\d{2})\b")


def _expand_year(raw: Optional[str]) -> Optional[int]:
    """``"22"`` -> 2022, ``"2022"`` -> 2022, ``None`` -> ``None``."""
    if not raw:
        return None
    value = int(raw)
    if value < 100:
        return 2000 + value if value < 80 else 1900 + value
    return value


def find_all_dates(text: str) -> List[date]:
    """Every explicit day-month-year date in a text, in the order found.

    Reuses the same patterns ``TemporalScope._parse_date`` matches against a
    single string, but scans the whole text for every occurrence rather than
    stopping at the first. Used to estimate a document's real publication
    date from its own front matter and press-release language — a signal
    that generalizes across any report stating its own release date, not
    something specific to one document or corpus.
    """
    found: List[date] = []
    for match in _ISO_DATE_RE.finditer(text):
        try:
            found.append(date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
        except ValueError:
            continue
    for match in _DMY_RE.finditer(text):
        month = _MONTHS.get(match.group(2)[:4].casefold().rstrip(".")) or _MONTHS.get(
            match.group(2)[:3].casefold()
        )
        if month:
            try:
                found.append(date(int(match.group(3)), month, int(match.group(1))))
            except ValueError:
                continue
    for match in _MDY_RE.finditer(text):
        month = _MONTHS.get(match.group(1)[:3].casefold())
        if month:
            try:
                found.append(date(int(match.group(3)), month, int(match.group(2))))
            except ValueError:
                continue
    return found


class TemporalScope(BaseModel):
    """A parsed, comparable view of a free-text period label.

    The raw label stays on ``AtomicFact.temporal_scope`` exactly as the document
    expressed it; this object is the machine-comparable projection of it. Two
    facts are only a *genuine* contradiction if their scopes are equal; if one
    scope contains the other (Q3 FY23 inside FY23, 9M FY23 inside FY23), the
    relationship engine has a ready-made reconciliation story.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: str = ""
    period_type: PeriodType = PeriodType.UNSPECIFIED
    year: Optional[int] = None
    index: Optional[int] = Field(
        default=None,
        description="Quarter number, half number, or month count depending on period_type.",
    )
    as_of: Optional[date] = None

    # -- construction ------------------------------------------------------- #
    @classmethod
    def parse(cls, raw: Optional[str]) -> "TemporalScope":
        """Best-effort parse of any period label the documents use.

        Handles ``FY22``, ``FY2022-23``, ``Q3 FY23``, ``H1 FY24``, ``9M FY23``,
        ``TTM``, ``CY2022``, ``2022``, ``31 March 2023``, ``March 31, 2023`` and
        ISO dates. Unknown shapes degrade to ``UNSPECIFIED`` rather than raising,
        because an unparsed scope must still be storable and visible in the UI.
        """
        if raw is None:
            return cls()
        text = collapse_whitespace(str(raw))
        if not text:
            return cls()

        fy_year: Optional[int] = None
        for pattern in (_FY_RE, _FY_WORDS_RE):
            m = pattern.search(text)
            if m:
                fy_year = _expand_year(m.group(1))
                break
        cy = _CY_RE.search(text)
        cy_year = _expand_year(cy.group(1)) if cy else None

        m = _QUARTER_RE.search(text)
        if m:
            year = _expand_year(m.group(2)) or fy_year or cy_year
            return cls(raw=text, period_type=PeriodType.QUARTER, year=year, index=int(m.group(1)))

        m = _HALF_RE.search(text)
        if m:
            year = _expand_year(m.group(2)) or fy_year or cy_year
            return cls(raw=text, period_type=PeriodType.HALF, year=year, index=int(m.group(1)))

        m = _MONTHS_RE.search(text)
        if m and 1 <= int(m.group(1)) <= 12:
            year = _expand_year(m.group(2)) or fy_year or cy_year
            months = int(m.group(1))
            if months == 12:
                return cls(raw=text, period_type=PeriodType.FISCAL_YEAR, year=year)
            return cls(raw=text, period_type=PeriodType.MONTHS, year=year, index=months)

        if _TTM_RE.search(text):
            return cls(raw=text, period_type=PeriodType.TTM, year=fy_year or cy_year)

        parsed_date = cls._parse_date(text)
        if parsed_date is not None:
            return cls(
                raw=text,
                period_type=PeriodType.POINT_IN_TIME,
                year=parsed_date.year,
                as_of=parsed_date,
            )

        if fy_year is not None:
            return cls(raw=text, period_type=PeriodType.FISCAL_YEAR, year=fy_year)
        if cy_year is not None:
            return cls(raw=text, period_type=PeriodType.CALENDAR_YEAR, year=cy_year)

        m = _FISCAL_YEAR_RANGE_RE.search(text)
        if m:
            start_year = int(m.group(1) + m.group(2))
            end_suffix = int(m.group(3))
            if end_suffix == (start_year + 1) % 100:
                # Anchored at the starting year, matching how "FY2024/25"
                # already resolves via _FY_RE above — so the two notations
                # land on the identical canonical period and can be compared.
                return cls(raw=text, period_type=PeriodType.FISCAL_YEAR, year=start_year)

        m = _BARE_YEAR_RE.search(text)
        if m:
            return cls(raw=text, period_type=PeriodType.CALENDAR_YEAR, year=int(m.group(0)))

        return cls(raw=text, period_type=PeriodType.UNSPECIFIED)

    @staticmethod
    def _parse_date(text: str) -> Optional[date]:
        m = _MONTH_END_RE.search(text)
        if m:
            month = _MONTHS.get(m.group(1)[:4].casefold().rstrip(".")) or _MONTHS.get(
                m.group(1)[:3].casefold()
            )
            if month:
                year = int(m.group(2))
                last_day = [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 28,
                            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
                try:
                    return date(year, month, last_day)
                except ValueError:
                    return None
        m = _ISO_DATE_RE.search(text)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
        m = _DMY_RE.search(text)
        if m:
            month = _MONTHS.get(m.group(2)[:4].casefold().rstrip(".")) or _MONTHS.get(
                m.group(2)[:3].casefold()
            )
            if month:
                try:
                    return date(int(m.group(3)), month, int(m.group(1)))
                except ValueError:
                    return None
        m = _MDY_RE.search(text)
        if m:
            month = _MONTHS.get(m.group(1)[:3].casefold())
            if month:
                try:
                    return date(int(m.group(3)), month, int(m.group(2)))
                except ValueError:
                    return None
        return None

    # -- comparison --------------------------------------------------------- #
    @property
    def canonical(self) -> str:
        """Stable string key used for grouping and display."""
        if self.period_type is PeriodType.UNSPECIFIED:
            return "UNSPECIFIED"
        if self.period_type is PeriodType.POINT_IN_TIME and self.as_of:
            return f"ASOF:{self.as_of.isoformat()}"
        year = str(self.year) if self.year is not None else "????"
        if self.period_type is PeriodType.QUARTER:
            return f"FY{year}-Q{self.index}"
        if self.period_type is PeriodType.HALF:
            return f"FY{year}-H{self.index}"
        if self.period_type is PeriodType.MONTHS:
            return f"FY{year}-{self.index}M"
        if self.period_type is PeriodType.TTM:
            return f"TTM@{year}" if self.year is not None else "TTM"
        if self.period_type is PeriodType.CALENDAR_YEAR:
            return f"CY{year}"
        return f"FY{year}"

    @property
    def is_known(self) -> bool:
        return self.period_type is not PeriodType.UNSPECIFIED

    @property
    def month_span(self) -> Optional[int]:
        """Duration in months, when it is knowable. Used to explain deltas."""
        return {
            PeriodType.FISCAL_YEAR: 12,
            PeriodType.CALENDAR_YEAR: 12,
            PeriodType.TTM: 12,
            PeriodType.QUARTER: 3,
            PeriodType.HALF: 6,
            PeriodType.MONTHS: self.index,
            PeriodType.POINT_IN_TIME: 0,
        }.get(self.period_type)

    def same_period(self, other: "TemporalScope") -> bool:
        """True only when both scopes are known and identical."""
        if not (self.is_known and other.is_known):
            return False
        return self.canonical == other.canonical

    def contains(self, other: "TemporalScope") -> bool:
        """True when ``other`` is a sub-period of ``self`` in the same year.

        This is the workhorse behind APPARENT_CONTRADICTION: a 9-month figure
        and a full-year figure for the same entity/attribute disagree only
        because one is a slice of the other.
        """
        if not (self.is_known and other.is_known):
            return False
        annual = {PeriodType.FISCAL_YEAR, PeriodType.CALENDAR_YEAR, PeriodType.TTM}
        partial = {PeriodType.QUARTER, PeriodType.HALF, PeriodType.MONTHS}
        if self.period_type not in annual or other.period_type not in partial:
            return False
        if self.year is None or other.year is None:
            return False
        return self.year == other.year

    def overlaps(self, other: "TemporalScope") -> bool:
        return self.same_period(other) or self.contains(other) or other.contains(self)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
class MatchStrategy(str, Enum):
    """How a verbatim quote was anchored back into the parsed page text."""

    EXACT = "EXACT"                    # byte-identical substring
    NORMALIZED = "NORMALIZED"          # matched after whitespace/unicode folding
    CASE_INSENSITIVE = "CASE_INSENSITIVE"
    FUZZY = "FUZZY"                    # aligned above a similarity threshold
    UNVERIFIED = "UNVERIFIED"          # could not be grounded (must not be stored)


class SourceAnchor(BaseModel):
    """A verified pointer into one page of one parsed document.

    ``char_start``/``char_end`` index into ``ParsedPage.text`` — the *parser's*
    canonical text for that page, not the raw PDF stream. That indirection is
    deliberate: the parser is deterministic, so the same PDF always yields the
    same offsets, and the UI can highlight the span by simple slicing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_id: str = Field(min_length=1, description="Content hash of the source PDF.")
    page_number: PositiveInt = Field(description="1-based page number.")
    verbatim_quote: str = Field(min_length=1, description="Text as it appears in the page.")
    char_start: NonNegativeInt
    char_end: NonNegativeInt

    # Auditing metadata: how confident we are that this span is the quote.
    match_strategy: MatchStrategy = MatchStrategy.EXACT
    match_score: float = Field(default=1.0, ge=0.0, le=1.0)
    document_name: Optional[str] = None
    block_kind: Literal["text", "table", "mixed"] = "text"

    @field_validator("verbatim_quote")
    @classmethod
    def _quote_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("verbatim_quote cannot be whitespace-only")
        return value

    @model_validator(mode="after")
    def _validate_span(self) -> "SourceAnchor":
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be greater than char_start ({self.char_start})"
            )
        span = self.char_end - self.char_start
        quote_len = len(self.verbatim_quote)
        # Whitespace folding can shrink or stretch a span, but a 3x divergence
        # means the offsets belong to different text than the quote.
        if span > max(64, quote_len * 3) or quote_len > max(64, span * 3):
            raise ValueError(
                f"span length {span} is implausible for a {quote_len}-char quote"
            )
        if self.match_strategy is MatchStrategy.UNVERIFIED:
            raise ValueError("refusing to persist an UNVERIFIED anchor")
        return self

    @property
    def span(self) -> Tuple[int, int]:
        return (self.char_start, self.char_end)

    @property
    def locator(self) -> str:
        """Human-readable citation, e.g. ``prospectus.pdf p.14 [1820:1904]``."""
        name = self.document_name or self.file_id
        return f"{name} p.{self.page_number} [{self.char_start}:{self.char_end}]"

    def slice_from(self, page_text: str) -> str:
        """Return the page substring this anchor points at."""
        return page_text[self.char_start : self.char_end]

    def verify(self, page_text: str, *, strict: bool = False) -> bool:
        """Re-check the anchor against page text (used by tests and the UI).

        ``strict`` requires byte equality; otherwise whitespace and unicode
        punctuation differences are tolerated, which is what the NORMALIZED and
        FUZZY strategies produced in the first place.
        """
        actual = self.slice_from(page_text)
        if strict:
            return actual == self.verbatim_quote
        return collapse_whitespace(actual).casefold() == collapse_whitespace(
            self.verbatim_quote
        ).casefold()


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #
class ValueKind(str, Enum):
    """Coarse type of the fact value; drives how the reconciler compares two facts."""

    NUMBER = "NUMBER"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    DATE = "DATE"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"


class AtomicFact(BaseModel):
    """One grounded (entity, attribute, value) claim with a period and a source.

    The predicate is intentionally open: the documents decide what an entity and
    an attribute are. What the schema enforces is that every claim carries a
    scope and a verified anchor, so any two facts can be compared and any single
    fact can be defended with a quote.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    id: str = Field(default="", description="Deterministic content hash; auto-filled.")
    entity: str = Field(min_length=1, description='Subject, e.g. "Acme Ltd" or "Jane Doe".')
    attribute: str = Field(min_length=1, description='Predicate, e.g. "total_revenue".')
    value: FactValue
    unit: Optional[str] = Field(default=None, description='e.g. "INR crore", "%", "employees".')
    temporal_scope: Optional[str] = Field(
        default=None, description="Period label exactly as the document stated it."
    )
    raw_statement: str = Field(
        min_length=1, description="The sentence/row the fact was read from."
    )
    provenance: SourceAnchor
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # -- normalization slots, filled by the extraction/normalization layer ---- #
    value_kind: ValueKind = ValueKind.TEXT
    normalized_value: Optional[float] = Field(
        default=None,
        description="Value converted to the canonical unit (e.g. absolute INR), when numeric.",
    )
    normalized_unit: Optional[str] = Field(
        default=None, description="Canonical unit matching normalized_value."
    )
    qualifiers: Dict[str, str] = Field(
        default_factory=dict,
        description='Free-form scope modifiers, e.g. {"basis": "consolidated"}. '
        "Differences here are the second main source of apparent contradictions.",
    )

    # -- lineage ------------------------------------------------------------- #
    extractor: str = Field(default="unknown", description="Which extractor produced this.")
    schema_version: str = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("entity", "attribute", "raw_statement")
    @classmethod
    def _tidy(cls, value: str) -> str:
        cleaned = collapse_whitespace(value)
        if not cleaned:
            raise ValueError("field cannot be empty after whitespace collapse")
        return cleaned

    @field_validator("unit", "temporal_scope", "normalized_unit")
    @classmethod
    def _tidy_optional(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = collapse_whitespace(value)
        return cleaned or None

    @model_validator(mode="after")
    def _assign_id(self) -> "AtomicFact":
        if not self.id:
            object.__setattr__(self, "id", self.compute_id())
        return self

    # -- identity ------------------------------------------------------------ #
    def compute_id(self) -> str:
        """Content hash over the claim *and* its location.

        Location is included on purpose: the same claim repeated on two pages is
        two facts with two pieces of evidence (and will be linked as
        CORROBORATED), while re-ingesting the same PDF is a no-op.
        """
        return _stable_hash(
            self.provenance.file_id,
            self.provenance.page_number,
            self.provenance.char_start,
            normalize_key(self.entity),
            normalize_key(self.attribute),
            self.value,
            self.unit,
            self.temporal_scope,
        )

    # -- derived views ------------------------------------------------------- #
    @property
    def entity_key(self) -> str:
        return normalize_key(self.entity)

    @property
    def attribute_key(self) -> str:
        return normalize_key(self.attribute)

    @property
    def predicate_key(self) -> str:
        """Blocking key: only facts sharing this key are worth comparing."""
        return f"{self.entity_key}::{self.attribute_key}"

    @property
    def scope(self) -> TemporalScope:
        return TemporalScope.parse(self.temporal_scope)

    @property
    def file_id(self) -> str:
        return self.provenance.file_id

    @property
    def is_numeric(self) -> bool:
        return isinstance(self.value, (int, float)) and not isinstance(self.value, bool)

    @property
    def comparable_value(self) -> Optional[float]:
        """Numeric value to compare against another fact, preferring the normalized one."""
        if self.normalized_value is not None:
            return float(self.normalized_value)
        if self.is_numeric:
            return float(self.value)  # type: ignore[arg-type]
        return None

    @property
    def qualifier_key(self) -> str:
        """Stable rendering of qualifiers, so scope differences are diffable."""
        if not self.qualifiers:
            return ""
        return "|".join(f"{normalize_key(k)}={normalize_key(v)}" for k, v in sorted(self.qualifiers.items()))

    def summary(self) -> str:
        """One-line rendering used in explanations and the UI."""
        unit = f" {self.unit}" if self.unit else ""
        scope = f" ({self.temporal_scope})" if self.temporal_scope else ""
        return f"{self.entity} · {self.attribute} = {self.value}{unit}{scope}"


# --------------------------------------------------------------------------- #
# Relationships
# --------------------------------------------------------------------------- #
class RelationshipType(str, Enum):
    """How two facts stand relative to each other."""

    CORROBORATED = "CORROBORATED"
    GENUINE_CONTRADICTION = "GENUINE_CONTRADICTION"
    APPARENT_CONTRADICTION = "APPARENT_CONTRADICTION"
    ORTHOGONAL = "ORTHOGONAL"


class Detector(str, Enum):
    """Which layer asserted a relationship — kept for error analysis."""

    DETERMINISTIC = "DETERMINISTIC"
    LLM_ADJUDICATED = "LLM_ADJUDICATED"
    HYBRID = "HYBRID"
    HUMAN = "HUMAN"


class FactRelationship(BaseModel):
    """An explained edge between two facts.

    ``explanation`` is written for a human reviewer (a merchant banker checking
    an IPO data room), and ``reconciliation_context`` carries the machine-readable
    reason — which dimension differed and by how much — so the UI can render the
    evidence without re-parsing prose.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default="", description="Deterministic hash of the unordered pair + type.")
    fact_a_id: str = Field(min_length=1)
    fact_b_id: str = Field(min_length=1)
    relationship_type: RelationshipType
    explanation: str = Field(min_length=1)
    reconciliation_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description='e.g. {"dimension": "temporal_scope", "a": "9M FY23", "b": "FY23", '
        '"relation": "sub_period", "delta_pct": 27.4}',
    )

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    detector: Detector = Detector.DETERMINISTIC
    created_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _validate(self) -> "FactRelationship":
        if self.fact_a_id == self.fact_b_id:
            raise ValueError("a fact cannot be related to itself")
        if not self.id:
            object.__setattr__(self, "id", self.compute_id())
        return self

    def compute_id(self) -> str:
        return _stable_hash(*self.unordered_key, self.relationship_type.value)

    @property
    def unordered_key(self) -> Tuple[str, str]:
        """Pair key that is stable regardless of which fact was seen first."""
        return tuple(sorted((self.fact_a_id, self.fact_b_id)))  # type: ignore[return-value]

    @property
    def is_conflict(self) -> bool:
        return self.relationship_type in (
            RelationshipType.GENUINE_CONTRADICTION,
            RelationshipType.APPARENT_CONTRADICTION,
        )


# --------------------------------------------------------------------------- #
# Parsed document structures (produced by parser.py)
# --------------------------------------------------------------------------- #
class BlockKind(str, Enum):
    TEXT = "text"
    TABLE = "table"


class PageBlock(BaseModel):
    """A layout unit within a page, already placed in reading order.

    Blocks own their character range inside ``ParsedPage.text``, which is how an
    anchor can report whether it landed in prose or inside a table.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: BlockKind
    order_index: NonNegativeInt
    text: str
    char_start: NonNegativeInt
    char_end: NonNegativeInt
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None

    def contains(self, position: int) -> bool:
        return self.char_start <= position < self.char_end


class ParsedPage(BaseModel):
    """Canonical text of a single page plus its layout blocks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: PositiveInt
    text: str
    blocks: List[PageBlock] = Field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    image_count: int = 0

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def has_text_layer(self) -> bool:
        return bool(self.text.strip())

    @property
    def needs_ocr(self) -> bool:
        """Little/no text but images present — almost certainly a scan."""
        return not self.has_text_layer and self.image_count > 0

    @property
    def table_count(self) -> int:
        return sum(1 for b in self.blocks if b.kind is BlockKind.TABLE)

    def block_at(self, position: int) -> Optional[PageBlock]:
        for block in self.blocks:
            if block.contains(position):
                return block
        return None


class ParsedDocument(BaseModel):
    """A whole PDF after deterministic parsing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_id: str = Field(min_length=1, description="Truncated SHA-256 of the file bytes.")
    filename: str
    sha256: str
    byte_size: NonNegativeInt
    page_count: NonNegativeInt
    pages: List[ParsedPage] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)
    parser_version: str = SCHEMA_VERSION
    parsed_at: datetime = Field(default_factory=_utcnow)
    warnings: List[str] = Field(default_factory=list)

    def page(self, page_number: int) -> Optional[ParsedPage]:
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def page_text(self, page_number: int) -> str:
        page = self.page(page_number)
        return page.text if page else ""

    def resolve(self, anchor: SourceAnchor) -> str:
        """Return the exact text an anchor points to, or '' if it does not resolve."""
        if anchor.file_id != self.file_id:
            return ""
        return anchor.slice_from(self.page_text(anchor.page_number))

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def needs_ocr(self) -> bool:
        """True when most pages carry no text layer."""
        if not self.pages:
            return True
        blank = sum(1 for p in self.pages if not p.has_text_layer)
        # 60% keeps a cover image or a blank divider from triggering a false alarm.
        return blank / len(self.pages) >= 0.6

    def iter_pages(self) -> Iterator[ParsedPage]:
        return iter(self.pages)


class TextChunk(BaseModel):
    """An extraction window carrying absolute page offsets.

    The extractor sees chunk-local text but every quote it returns is anchored
    with ``char_start + local_offset``, so provenance stays page-absolute no
    matter how the document was windowed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_id: str
    page_number: PositiveInt
    chunk_index: NonNegativeInt
    text: str
    char_start: NonNegativeInt
    char_end: NonNegativeInt
    document_name: Optional[str] = None

    @property
    def chunk_id(self) -> str:
        return f"{self.file_id}:p{self.page_number}:c{self.chunk_index}"


__all__ = [
    "SCHEMA_VERSION",
    "FactValue",
    "normalize_key",
    "collapse_whitespace",
    "find_all_dates",
    "PeriodType",
    "TemporalScope",
    "MatchStrategy",
    "SourceAnchor",
    "ValueKind",
    "AtomicFact",
    "RelationshipType",
    "Detector",
    "FactRelationship",
    "BlockKind",
    "PageBlock",
    "ParsedPage",
    "ParsedDocument",
    "TextChunk",
]