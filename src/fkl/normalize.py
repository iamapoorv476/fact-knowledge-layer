"""
normalize.py — Units, scales, and precision-aware numeric agreement.

Why this module exists
----------------------
Two things in the starter corpus make naive numeric comparison useless:

1. **Units are declared once, far from the number.** A note page in the FY24
   annual report opens with "(All amounts in Indian Rupees in million, unless
   otherwise stated)" and then prints bare figures like ``81,415.38`` for the
   next thousand characters. A per-number regex sees "81415.38" and has no idea
   it means ₹81.4 billion. So unit resolution is *scoped*: inline beats the
   nearest preceding declaration, which beats the page declaration, which beats
   the document default.

2. **The same fact is printed at different precisions.** The earnings deck says
   FY24 revenue from services is ₹8,142 Cr; the annual report says ₹81,415.38
   million. Those are the same fact, and any fixed tolerance is wrong: 1% is
   loose enough to fuse genuinely different figures, 0.001% rejects this pair.
   The right tolerance is derived from how precisely each figure was *written* —
   "8,142" carries an implied ±0.5 Cr, i.e. ±₹5 million. Under that rule the two
   figures corroborate, and a real ₹50 million discrepancy still does not.

Nothing here knows about logistics or Indian macroeconomics. It knows about
currencies, SI-and-Indian scale words, percentages, and significant figures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .schemas import ParsedDocument, ParsedPage, collapse_whitespace
except ImportError:  # pragma: no cover - flat script layout
    from schemas import ParsedDocument, ParsedPage, collapse_whitespace  # type: ignore[no-redef]


# --------------------------------------------------------------------------- #
# Dimensions and units
# --------------------------------------------------------------------------- #
class Dimension(str, Enum):
    """What kind of quantity a unit measures. Only same-dimension facts compare."""

    CURRENCY = "CURRENCY"
    PERCENT = "PERCENT"
    COUNT = "COUNT"
    DAYS = "DAYS"
    MASS = "MASS"
    AREA = "AREA"
    RATIO = "RATIO"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class UnitSpec:
    """A parsed unit: what it measures, its multiplier, and its canonical form."""

    raw: str
    dimension: Dimension
    scale: float = 1.0
    currency: Optional[str] = None
    canonical: str = ""

    @property
    def key(self) -> str:
        """Comparison key — dimension plus currency, ignoring scale."""
        if self.dimension is Dimension.CURRENCY:
            return f"CURRENCY:{self.currency or '?'}"
        return self.dimension.value

    def describe(self) -> str:
        return self.canonical or self.raw


#: Scale words. Indian (lakh, crore) and SI (million, billion) both appear in the
#: same document family, and "lakh crore" compounds to 1e12.
_SCALES: Dict[str, float] = {
    "hundred": 1e2,
    "thousand": 1e3,
    "k": 1e3,
    "'000": 1e3,
    "000s": 1e3,
    "lakh": 1e5,
    "lakhs": 1e5,
    "lac": 1e5,
    "lacs": 1e5,
    "mn": 1e6,
    "million": 1e6,
    "millions": 1e6,
    "mio": 1e6,
    "cr": 1e7,
    "crore": 1e7,
    "crores": 1e7,
    "bn": 1e9,
    "billion": 1e9,
    "billions": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
    "lakh crore": 1e12,
}

_CURRENCIES: Dict[str, str] = {
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "rupee": "INR",
    "rupees": "INR",
    "$": "USD",
    "us$": "USD",
    "usd": "USD",
    "us dollar": "USD",
    "us dollars": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
}

_MASS: Dict[str, float] = {
    "ton": 1.0, "tons": 1.0, "tonne": 1.0, "tonnes": 1.0, "mt": 1.0,
    "kg": 0.001, "kgs": 0.001,
}

_AREA = {"sq ft": 1.0, "sq. ft": 1.0, "sq ft.": 1.0, "square feet": 1.0, "sqft": 1.0,
         "sq m": 10.7639, "square metres": 10.7639, "square meters": 10.7639}

_COUNT_NOUNS = {
    "shipments", "parcels", "customers", "employees", "people", "pin codes",
    "pincodes", "centres", "centers", "vehicles", "units", "options", "shares",
    "gateways", "no.", "number", "count", "days",
}

_PERCENT_WORDS = ("%", "per cent", "percent", "percentage", "pct")
_BPS_WORDS = ("bps", "basis points", "bp")

_SCALE_ALTERNATION = "|".join(
    sorted((re.escape(k) for k in _SCALES), key=len, reverse=True)
)
_CURRENCY_ALTERNATION = "|".join(
    sorted((re.escape(k) for k in _CURRENCIES), key=len, reverse=True)
)


def parse_unit(text: Optional[str]) -> Optional[UnitSpec]:
    """Parse a unit label such as ``₹ Cr``, ``INR million``, ``%``, ``'000 Tons``.

    Returns ``None`` when the text carries no recognizable unit, which callers
    treat as "inherit from context" rather than "dimensionless".
    """
    if not text:
        return None
    raw = collapse_whitespace(text)
    if not raw:
        return None
    lowered = raw.casefold().strip("()[] \t")
    lowered = re.sub(r"\b(in|of|amount|amounts|all amounts|figures)\b", " ", lowered)
    lowered = re.sub(r"\bunless otherwise stated\b", " ", lowered)
    lowered = collapse_whitespace(lowered.replace("indian rupees", "inr"))
    if not lowered:
        return None

    if any(word in lowered for word in _BPS_WORDS) and not any(
        c.isalpha() for c in lowered.replace("bps", "").replace("bp", "").replace("basis points", "")
    ):
        return UnitSpec(raw=raw, dimension=Dimension.PERCENT, scale=0.01, canonical="%")
    if any(word in lowered for word in _PERCENT_WORDS):
        return UnitSpec(raw=raw, dimension=Dimension.PERCENT, scale=1.0, canonical="%")

    scale = 1.0
    scale_match = re.search(rf"(?<![a-z])({_SCALE_ALTERNATION})(?![a-z])", lowered)
    if scale_match:
        scale = _SCALES[scale_match.group(1)]

    currency_match = re.search(rf"(?<![a-z])({_CURRENCY_ALTERNATION})(?![a-z])", lowered)
    if currency_match:
        code = _CURRENCIES[currency_match.group(1)]
        canonical = code if scale == 1.0 else f"{code} ({_format_scale(scale)})"
        return UnitSpec(raw=raw, dimension=Dimension.CURRENCY, scale=scale,
                        currency=code, canonical=canonical)

    for token, factor in _MASS.items():
        if re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", lowered):
            return UnitSpec(raw=raw, dimension=Dimension.MASS, scale=scale * factor,
                            canonical="tonnes")
    for token, factor in _AREA.items():
        if token in lowered:
            return UnitSpec(raw=raw, dimension=Dimension.AREA, scale=scale * factor,
                            canonical="sq ft")
    if re.search(r"(?<![a-z])days?(?![a-z])", lowered):
        return UnitSpec(raw=raw, dimension=Dimension.DAYS, scale=scale, canonical="days")
    if re.fullmatch(r"\d*\.?\d*x", lowered):
        return UnitSpec(raw=raw, dimension=Dimension.RATIO, scale=1.0, canonical="x")

    for noun in _COUNT_NOUNS:
        if re.search(rf"(?<![a-z]){re.escape(noun)}(?![a-z])", lowered):
            return UnitSpec(raw=raw, dimension=Dimension.COUNT, scale=scale, canonical="count")

    if scale_match:
        # A bare scale word ("in million") with no noun: dimension is unknown but
        # the multiplier is real and must not be lost.
        return UnitSpec(raw=raw, dimension=Dimension.UNKNOWN, scale=scale,
                        canonical=_format_scale(scale))
    return None


def _format_scale(scale: float) -> str:
    return {
        1e2: "hundred", 1e3: "thousand", 1e5: "lakh", 1e6: "million",
        1e7: "crore", 1e9: "billion", 1e12: "trillion",
    }.get(scale, f"x{scale:g}")


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #
_NUMBER_CORE = r"\d[\d,\u00a0\s]*(?:\.\d+)?"
_NUMBER_RE = re.compile(rf"(?<![\w.])(-|\u2212)?\s*({_NUMBER_CORE})")
_NIL_TOKENS = {"-", "–", "—", "nil", "na", "n.a.", "n/a", "", "*"}


@dataclass(frozen=True)
class Quantity:
    """A number plus the precision it was written with, plus any inline unit."""

    value: float
    decimals: int
    raw: str
    unit: Optional[UnitSpec] = None
    negated_by_parens: bool = False

    @property
    def canonical_value(self) -> float:
        return self.value * (self.unit.scale if self.unit else 1.0)

    @property
    def half_ulp(self) -> float:
        """Half of the last written digit, in canonical units.

        ``8,142`` (0 decimals, crore scale) → ±5,000,000. This is the honest
        uncertainty introduced by however the author chose to round.
        """
        scale = self.unit.scale if self.unit else 1.0
        return 0.5 * (10.0 ** -self.decimals) * scale


def parse_number(text: str) -> Optional[Quantity]:
    """Parse a numeric cell or token.

    Handles Indian and Western digit grouping, parenthesised negatives (which is
    how every Indian financial statement writes a loss), unicode minus signs, and
    trailing percent/scale markers. Returns ``None`` for nil markers (``-``,
    ``N.A.``) so a missing value never becomes a zero.
    """
    if text is None:
        return None
    raw = collapse_whitespace(str(text))
    if raw.casefold().strip() in _NIL_TOKENS:
        return None

    working = raw
    # Strip footnote superscripts that survived extraction, e.g. "1,860(1)".
    working = re.sub(r"\((\d{1,2})\)\s*$", lambda m: "" if len(m.group(1)) <= 2 and "." not in raw else m.group(0), working) if re.search(r"[a-zA-Z]", working) else working

    negated = False
    stripped = working.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        inner = stripped[1:-1].strip()
        if re.fullmatch(rf"[₹$€£\s]*{_NUMBER_CORE}\s*(%|per cent|percent|bps)?", inner, re.I):
            negated = True
            working = inner
    else:
        # Statements also write "₹(452) Cr": the parentheses wrap only the digits
        # while the currency and scale sit outside them.
        inline_paren = re.search(rf"\(\s*({_NUMBER_CORE})\s*\)", working)
        if inline_paren:
            negated = True
            working = (
                working[: inline_paren.start()]
                + " "
                + inline_paren.group(1)
                + " "
                + working[inline_paren.end() :]
            )

    match = _NUMBER_RE.search(working)
    if not match:
        return None
    sign = -1.0 if match.group(1) else 1.0
    digits = re.sub(r"[,\s\u00a0]", "", match.group(2))
    if not digits or digits == ".":
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    if negated:
        sign = -abs(sign)

    unit = parse_unit(_strip_number(working, match))
    return Quantity(
        value=sign * value,
        decimals=decimals,
        raw=raw,
        unit=unit,
        negated_by_parens=negated,
    )


def _strip_number(text: str, match: "re.Match[str]") -> str:
    """Everything in the token except the digits — that's the inline unit."""
    return (text[: match.start()] + " " + text[match.end():]).strip()


def parse_quantity(text: str, *, fallback_unit: Optional[UnitSpec] = None) -> Optional[Quantity]:
    """Parse a free-form quantity like ``₹8,142 Cr`` or ``6.5 per cent``.

    ``fallback_unit`` is the unit resolved from page context; it is applied only
    when the text itself carries no unit, which is exactly the precedence the
    documents assume.
    """
    quantity = parse_number(text)
    if quantity is None:
        return None
    if quantity.unit is None and fallback_unit is not None:
        return Quantity(
            value=quantity.value,
            decimals=quantity.decimals,
            raw=quantity.raw,
            unit=fallback_unit,
            negated_by_parens=quantity.negated_by_parens,
        )
    return quantity


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #
class AgreementBand(str, Enum):
    """Three-state verdict, because a boolean here would be a lie.

    Two figures rounded to the same precision can differ by up to one full unit
    and still describe the same underlying number (₹7,224 Cr vs ₹7,225 Cr), but
    they can equally be genuinely different (6.5% vs 6.6% GDP growth from two
    institutions). ROUNDING_BOUNDARY names that zone so the relationship engine
    can report it honestly instead of picking a side.
    """

    MATCH = "MATCH"
    ROUNDING_BOUNDARY = "ROUNDING_BOUNDARY"
    MISMATCH = "MISMATCH"
    INCOMPARABLE = "INCOMPARABLE"


@dataclass(frozen=True)
class Agreement:
    """Result of comparing two quantities that claim to be the same fact."""

    comparable: bool
    agree: bool
    band: AgreementBand = AgreementBand.INCOMPARABLE
    difference: float = 0.0
    relative_difference: float = 0.0
    tolerance: float = 0.0
    boundary_tolerance: float = 0.0
    reason: str = ""

    def explain(self) -> str:
        if not self.comparable:
            return self.reason
        if self.band is AgreementBand.MATCH:
            return (
                f"values match within the precision they were written to "
                f"(difference {self.difference:,.2f} vs tolerance ±{self.tolerance:,.2f})"
            )
        if self.band is AgreementBand.ROUNDING_BOUNDARY:
            return (
                f"values differ by {self.difference:,.2f} — exactly the amount two figures "
                f"rounded this way can differ by, so this is consistent with rounding "
                f"but is not proof of it"
            )
        return (
            f"values differ by {self.difference:,.2f} "
            f"({self.relative_difference * 100:.2f}%), beyond the ±{self.tolerance:,.2f} "
            f"implied by their stated precision"
        )


def compare_quantities(
    a: Quantity, b: Quantity, *, extra_relative_tolerance: float = 0.0
) -> Agreement:
    """Compare two quantities in canonical units using precision-derived tolerance.

    ``extra_relative_tolerance`` lets a caller loosen the bar (e.g. 0.005 when
    one source is a rounded chart label rather than a statement figure).
    """
    unit_a = a.unit
    unit_b = b.unit
    if unit_a is not None and unit_b is not None:
        if unit_a.dimension is not Dimension.UNKNOWN and unit_b.dimension is not Dimension.UNKNOWN:
            if unit_a.key != unit_b.key:
                return Agreement(
                    comparable=False,
                    agree=False,
                    band=AgreementBand.INCOMPARABLE,
                    reason=f"different dimensions: {unit_a.describe()} vs {unit_b.describe()}",
                )

    left = a.canonical_value
    right = b.canonical_value
    difference = abs(left - right)
    magnitude = max(abs(left), abs(right), 1e-12)
    tolerance = max(a.half_ulp, b.half_ulp) + extra_relative_tolerance * magnitude
    boundary = a.half_ulp + b.half_ulp + extra_relative_tolerance * magnitude
    if difference <= tolerance:
        band = AgreementBand.MATCH
    elif difference <= boundary:
        band = AgreementBand.ROUNDING_BOUNDARY
    else:
        band = AgreementBand.MISMATCH
    return Agreement(
        comparable=True,
        agree=band is AgreementBand.MATCH,
        band=band,
        difference=difference,
        relative_difference=difference / magnitude,
        tolerance=tolerance,
        boundary_tolerance=boundary,
    )


def values_agree(a: Quantity, b: Quantity, **kwargs: float) -> bool:
    """Convenience boolean wrapper over :func:`compare_quantities`."""
    return compare_quantities(a, b, **kwargs).agree


# --------------------------------------------------------------------------- #
# Page-scoped unit context
# --------------------------------------------------------------------------- #
#: Parenthesised declarations, e.g. "(₹ Cr)", "(All amounts in Indian Rupees in
#: million, unless otherwise stated)", "('000 Tons)", "(Days)".
_DECLARATION_RE = re.compile(r"\(([^()\n]{2,90})\)")

#: Unparenthesised declarations that sit on their own line above a table.
_BARE_DECLARATION_RE = re.compile(
    r"^\s*(?:\u20b9|Rs\.?|INR|USD|\$)\s*(?:in\s+)?"
    rf"(?:{_SCALE_ALTERNATION})\s*$",
    re.I | re.M,
)


@dataclass(frozen=True)
class UnitDeclaration:
    """A unit statement found in the text, with the position it takes effect from."""

    unit: UnitSpec
    page_number: int
    char_start: int
    char_end: int
    text: str
    is_global: bool = False


class UnitContext:
    """Resolves the unit that applies to a number at a given page position.

    Precedence, highest first:

    1. a unit written on the number itself (handled by the caller);
    2. the nearest declaration *earlier on the same page*;
    3. the first declaration on that page;
    4. the document-wide default (a declaration repeated across many pages, or
       one that says "unless otherwise stated").
    """

    def __init__(
        self,
        declarations: Sequence[UnitDeclaration],
        document_default: Optional[UnitSpec] = None,
    ) -> None:
        self._by_page: Dict[int, List[UnitDeclaration]] = {}
        for declaration in declarations:
            self._by_page.setdefault(declaration.page_number, []).append(declaration)
        for entries in self._by_page.values():
            entries.sort(key=lambda d: d.char_start)
        self.document_default = document_default
        self.declarations = list(declarations)

    def resolve(
        self, page_number: int, char_position: int, *, inline: Optional[UnitSpec] = None
    ) -> Tuple[Optional[UnitSpec], str]:
        """Return ``(unit, source)`` where source explains which rule applied."""
        if inline is not None:
            return inline, "inline"
        entries = self._by_page.get(page_number, [])
        preceding = [d for d in entries if d.char_start <= char_position]
        if preceding:
            return preceding[-1].unit, "page-declaration"
        if entries:
            return entries[0].unit, "page-declaration-after"
        if self.document_default is not None:
            return self.document_default, "document-default"
        return None, "none"

    def declarations_on(self, page_number: int) -> List[UnitDeclaration]:
        return list(self._by_page.get(page_number, []))

    def __len__(self) -> int:
        return len(self.declarations)


def build_unit_context(document: ParsedDocument) -> UnitContext:
    """Scan a parsed document for unit declarations and build the resolver.

    A declaration that appears on at least a quarter of the pages, or that says
    "unless otherwise stated", becomes the document default — that is how the
    annual report's "(All amounts in Indian Rupees in million…)" header carries
    to every note page that reprints it.
    """
    declarations: List[UnitDeclaration] = []
    counts: Dict[str, int] = {}
    global_candidates: Dict[str, UnitSpec] = {}

    for page in document.pages:
        seen_on_page: set = set()
        for match in _DECLARATION_RE.finditer(page.text):
            inner = match.group(1)
            unit = parse_unit(inner)
            if unit is None or unit.dimension is Dimension.RATIO:
                continue
            # A parenthesised negative number is not a unit declaration.
            if re.fullmatch(r"[\d,.\s%]+", inner):
                continue
            is_global = "unless otherwise stated" in inner.casefold()
            declarations.append(
                UnitDeclaration(
                    unit=unit,
                    page_number=page.page_number,
                    char_start=match.start(),
                    char_end=match.end(),
                    text=collapse_whitespace(match.group(0)),
                    is_global=is_global,
                )
            )
            key = unit.key + "|" + str(unit.scale)
            if key not in seen_on_page:
                counts[key] = counts.get(key, 0) + 1
                seen_on_page.add(key)
            if is_global or key not in global_candidates:
                global_candidates[key] = unit

        for match in _BARE_DECLARATION_RE.finditer(page.text):
            unit = parse_unit(match.group(0))
            if unit is None:
                continue
            declarations.append(
                UnitDeclaration(
                    unit=unit,
                    page_number=page.page_number,
                    char_start=match.start(),
                    char_end=match.end(),
                    text=collapse_whitespace(match.group(0)),
                )
            )

    document_default: Optional[UnitSpec] = None
    explicit_global = [d for d in declarations if d.is_global]
    if explicit_global:
        document_default = explicit_global[0].unit
    elif counts and document.pages:
        key, count = max(counts.items(), key=lambda kv: kv[1])
        if count >= max(2, len(document.pages) // 4):
            document_default = global_candidates[key]

    return UnitContext(declarations, document_default)


__all__ = [
    "Dimension",
    "UnitSpec",
    "parse_unit",
    "Quantity",
    "parse_number",
    "parse_quantity",
    "Agreement",
    "AgreementBand",
    "compare_quantities",
    "values_agree",
    "UnitDeclaration",
    "UnitContext",
    "build_unit_context",
]