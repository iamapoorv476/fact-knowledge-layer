"""
reconcile.py — Deciding how any two grounded facts stand to each other.

The problem this solves
-----------------------
Extraction produces thousands of facts that each look sensible alone. The value
is in the pairs: ₹8,142 Cr in a deck and ₹81,415.38 million in an annual report
are the same fact; ₹127 Cr EBITDA and ₹76 Cr Adjusted EBITDA are not a
discrepancy; 6.5% and 6.6% GDP growth from two institutions might be either.

Three resolution layers, then one decision
------------------------------------------
1. **Entities** are clustered by alias — "Delhivery Limited", "Delhivery" and
   "the Company" must land in one bucket or nothing will ever compare.
2. **Attributes** are split into a *core* ("ebitda") and *modifiers*
   ("adjusted", "consolidated", "total"). Two facts whose cores agree but whose
   modifiers differ are measuring related but distinct things — that distinction
   is what separates an apparent contradiction from a genuine one.
3. **Periods** come parsed from the schema, so sub-period containment
   (9M FY23 inside FY23) is a plain comparison.

Only then does the classifier run, and it always states which signal decided.
An unexplained verdict is useless to a banker who has to defend the number.

Design stance: a contradiction claim is expensive to be wrong about. Where the
evidence is thin — unknown period, one-unit rounding gap, missing unit — the
engine says so in the explanation and lowers confidence rather than picking the
dramatic label.
"""

from __future__ import annotations

import itertools
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

try:
    from .normalize import (
        Agreement,
        AgreementBand,
        Dimension,
        Quantity,
        UnitSpec,
        compare_quantities,
        parse_unit,
    )
    from .schemas import (
        AtomicFact,
        Detector,
        FactRelationship,
        PeriodType,
        RelationshipType,
        TemporalScope,
        collapse_whitespace,
        normalize_key,
    )
except ImportError:  # pragma: no cover - flat script layout
    from normalize import (  # type: ignore[no-redef]
        Agreement,
        AgreementBand,
        Dimension,
        Quantity,
        UnitSpec,
        compare_quantities,
        parse_unit,
    )
    from schemas import (  # type: ignore[no-redef]
        AtomicFact,
        Detector,
        FactRelationship,
        PeriodType,
        RelationshipType,
        TemporalScope,
        collapse_whitespace,
        normalize_key,
    )

RECONCILER_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Entity resolution
# --------------------------------------------------------------------------- #
#: Legal-form words that carry no identity. "Delhivery Limited" and "Delhivery"
#: are the same company; "Reserve Bank" and "Bank" are not the same thing, which
#: is why only trailing legal forms are stripped, never head nouns.
_LEGAL_SUFFIXES = {
    "limited", "ltd", "private", "pvt", "plc", "inc", "incorporated",
    "corporation", "corp", "company", "co", "llp", "llc", "group", "holdings",
}
_ENTITY_LEAD_WORDS = {"the", "our", "its", "their", "this"}
_GENERIC_ENTITIES = {"", "company", "group", "issuer", "bank", "fund", "board"}


def normalize_entity(name: str) -> str:
    """Reduce an entity name to its identifying core."""
    tokens = re.split(r"[^a-z0-9]+", collapse_whitespace(name).casefold())
    tokens = [t for t in tokens if t]
    while tokens and tokens[0] in _ENTITY_LEAD_WORDS:
        tokens.pop(0)
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _initials(normalized: str) -> str:
    parts = normalized.split()
    return "".join(p[0] for p in parts) if len(parts) > 1 else ""


class EntityResolver:
    """Clusters entity strings that name the same thing.

    Merges on three relations: identical cores, token-subset containment
    ("delhivery" inside "delhivery logistics"), and initialisms ("rbi" for
    "reserve bank of india"). Generic self-references ("the Company") carry no
    identity of their own and are mapped to the subject of the document they
    came from, which is the only place that information exists.
    """

    def __init__(self, alias_map: Optional[Dict[str, str]] = None) -> None:
        self.alias_map = {normalize_entity(k): v for k, v in (alias_map or {}).items()}
        self._parent: Dict[str, str] = {}
        self._counts: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._canonical: Dict[str, str] = {}

    # -- union-find --------------------------------------------------------- #
    def _find(self, key: str) -> str:
        self._parent.setdefault(key, key)
        while self._parent[key] != key:
            self._parent[key] = self._parent[self._parent[key]]
            key = self._parent[key]
        return key

    def _union(self, a: str, b: str) -> None:
        root_a, root_b = self._find(a), self._find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a

    # -- fitting ------------------------------------------------------------ #
    def fit(self, facts: Sequence[AtomicFact], document_subjects: Optional[Dict[str, str]] = None) -> None:
        """Learn the clusters present in a fact set."""
        subjects = document_subjects or {}
        surface: Dict[str, Dict[str, int]] = defaultdict(dict)

        for fact in facts:
            raw = fact.entity
            key = normalize_entity(raw)
            if key in _GENERIC_ENTITIES:
                subject = subjects.get(fact.provenance.file_id)
                if subject:
                    raw = subject
                    key = normalize_entity(subject)
            if not key:
                continue
            key = self.alias_map.get(key, key)
            key = normalize_entity(key) if key not in self._parent else key
            self._find(key)
            surface[key][raw] = surface[key].get(raw, 0) + 1

        keys = sorted(surface)
        for left, right in itertools.combinations(keys, 2):
            if self._are_aliases(left, right):
                self._union(left, right)

        merged: Dict[str, Dict[str, int]] = defaultdict(dict)
        for key, forms in surface.items():
            root = self._find(key)
            for form, count in forms.items():
                merged[root][form] = merged[root].get(form, 0) + count
        self._counts = merged

        for root, forms in merged.items():
            # Prefer the form the corpus uses most; break ties toward the longer
            # name, which is normally the full legal one.
            self._canonical[root] = max(forms.items(), key=lambda kv: (kv[1], len(kv[0])))[0]

    @staticmethod
    def _are_aliases(left: str, right: str) -> bool:
        if left == right:
            return True
        left_tokens, right_tokens = set(left.split()), set(right.split())
        if not left_tokens or not right_tokens:
            return False
        if left_tokens < right_tokens or right_tokens < left_tokens:
            # Require the shorter name to be distinctive, not a bare common word.
            shorter = left_tokens if len(left_tokens) < len(right_tokens) else right_tokens
            if all(len(token) >= 4 for token in shorter):
                return True
        if _initials(left) == right.replace(" ", "") or _initials(right) == left.replace(" ", ""):
            return True
        return False

    # -- lookup ------------------------------------------------------------- #
    def canonical(self, name: str, file_id: Optional[str] = None,
                  document_subjects: Optional[Dict[str, str]] = None) -> str:
        key = normalize_entity(name)
        if key in _GENERIC_ENTITIES and file_id and document_subjects:
            subject = document_subjects.get(file_id)
            if subject:
                key = normalize_entity(subject)
        key = self.alias_map.get(key, key)
        root = self._find(normalize_entity(key))
        return self._canonical.get(root, collapse_whitespace(name))

    def cluster_key(self, name: str, file_id: Optional[str] = None,
                    document_subjects: Optional[Dict[str, str]] = None) -> str:
        return normalize_entity(
            self.canonical(name, file_id=file_id, document_subjects=document_subjects)
        )

    @property
    def clusters(self) -> Dict[str, List[str]]:
        return {self._canonical[root]: sorted(forms) for root, forms in self._counts.items()}


# --------------------------------------------------------------------------- #
# Attribute resolution
# --------------------------------------------------------------------------- #
#: Words that qualify *how* something is measured. Two facts whose cores agree
#: but whose modifiers differ are the classic apparent contradiction: same
#: subject, same period, different definition.
#: Words denoting an alternate computation or presentation basis of the SAME
#: underlying measure — "Adjusted EBITDA" and "EBITDA" are two ways of
#: computing one thing, so stripping the qualifier and comparing what's left
#: is exactly right; a modifier difference is the reconciler's signal for
#: APPARENT_CONTRADICTION-by-definition.
_MODIFIER_TOKENS = {
    "adjusted", "adj", "reported", "restated", "normalised", "normalized",
    "consolidated", "standalone", "proforma", "pro", "forma", "underlying",
    "excluding", "including", "incl", "excl", "annualised", "annualized",
    "average", "estimated", "projected", "provisional", "revised", "budgeted",
    "actual",
}
#: Words that instead denote a STRUCTURAL, part-whole relationship — "Total X"
#: is the sum X is a component of; "Other X" is the residual after named
#: components are removed; "Current X" and "Noncurrent X" partition X into
#: disjoint pieces. Stripping these as if they were mere presentation
#: variants collapses genuinely distinct line items into one core: "Other
#: income" and "Total income" both reduce to bare "income" and get compared
#: as if they were the same figure at different precision, when one is a
#: component and the other is the sum that component feeds into. These stay
#: in the core, so "other income" and "total income" get different cores and
#: are never mistaken for each other.
_STRUCTURAL_WORDS = {
    "total", "totals", "net", "gross", "other", "others", "current",
    "noncurrent", "service", "segment",
}
_ATTRIBUTE_STOPWORDS = {
    "the", "a", "an", "of", "from", "for", "in", "on", "at", "to", "and", "or",
    "by", "with", "as", "per", "less", "add", "is", "was", "были",
}


@dataclass(frozen=True)
class AttributeSignature:
    """An attribute split into what is measured and how it is qualified."""

    core: Tuple[str, ...]
    modifiers: Tuple[str, ...]
    raw: str

    @property
    def core_key(self) -> str:
        return " ".join(sorted(self.core))

    def similarity(self, other: "AttributeSignature") -> float:
        left, right = set(self.core), set(other.core)
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def modifiers_differ(self, other: "AttributeSignature") -> bool:
        return set(self.modifiers) != set(other.modifiers)

    def modifier_delta(self, other: "AttributeSignature") -> List[str]:
        return sorted(set(self.modifiers) ^ set(other.modifiers))


def attribute_signature(attribute: str) -> AttributeSignature:
    """Split an attribute label into core tokens and modifier tokens."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", attribute.casefold()) if t]
    core = tuple(t for t in tokens if t not in _MODIFIER_TOKENS and t not in _ATTRIBUTE_STOPWORDS)
    modifiers = tuple(t for t in tokens if t in _MODIFIER_TOKENS)
    if not core:  # an attribute made only of modifiers keeps them as its core
        core = modifiers or tuple(tokens)
    return AttributeSignature(core=core, modifiers=modifiers, raw=collapse_whitespace(attribute))


# --------------------------------------------------------------------------- #
# Document metadata used in explanations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DocumentProfile:
    """What the reconciler needs to know about a source document."""

    file_id: str
    filename: str
    subject: str
    vintage_year: Optional[int] = None

    @property
    def label(self) -> str:
        return self.filename


def profile_documents(
    facts: Sequence[AtomicFact], names: Optional[Dict[str, str]] = None,
    subjects: Optional[Dict[str, str]] = None,
) -> Dict[str, DocumentProfile]:
    """Derive a per-document profile, including its data vintage.

    Vintage is the median period year any fact in the document refers to —
    not the max. A report that mostly discusses the last two years but
    includes one long-range forecast table (a decade of projections) would
    have its estimated vintage dragged years into the future by that single
    table if the max were used; the median stays anchored to what the
    document is actually mostly about. Two reports that disagree on the same
    measure often differ simply because one was written later and had the
    revised number — that is a reconciliation, not a contradiction, and this
    is the cheapest available proxy for it, since neither of these PDFs'
    metadata carries a usable publication date.
    """
    years_by_file: Dict[str, List[int]] = {}
    filenames: Dict[str, str] = {}
    for fact in facts:
        file_id = fact.provenance.file_id
        filenames.setdefault(file_id, fact.provenance.document_name or file_id)
        year = fact.scope.year
        if year and 1900 < year < 2200:
            years_by_file.setdefault(file_id, []).append(year)
    vintage: Dict[str, int] = {
        file_id: round(statistics.median(years)) for file_id, years in years_by_file.items()
    }
    profiles: Dict[str, DocumentProfile] = {}
    for file_id, name in filenames.items():
        profiles[file_id] = DocumentProfile(
            file_id=file_id,
            filename=(names or {}).get(file_id, name),
            subject=(subjects or {}).get(file_id, ""),
            vintage_year=vintage.get(file_id),
        )
    return profiles


# --------------------------------------------------------------------------- #
# Pair classification
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    """The classifier's decision plus everything needed to justify it."""

    relationship: RelationshipType
    explanation: str
    context: Dict[str, object] = field(default_factory=dict)
    confidence: float = 0.5


def _quantity_of(fact: AtomicFact) -> Optional[Quantity]:
    """Rebuild a comparable quantity from a stored fact."""
    if not fact.is_numeric:
        return None
    unit = parse_unit(fact.unit) if fact.unit else None
    decimals = 0
    text = str(fact.value)
    if "." in text:
        decimals = len(text.split(".")[1].rstrip("0")) or 0
    return Quantity(value=float(fact.value), decimals=decimals, raw=text, unit=unit)


def _period_relation(a: TemporalScope, b: TemporalScope) -> str:
    if a.same_period(b):
        return "same"
    if a.contains(b):
        return "a_contains_b"
    if b.contains(a):
        return "b_contains_a"
    if a.is_known and b.is_known:
        return "disjoint"
    return "unknown"


def _qualifier_delta(a: AtomicFact, b: AtomicFact) -> Dict[str, List[Optional[str]]]:
    """Qualifier keys whose values differ between two facts."""
    interesting = {"basis", "section", "binding", "source", "context"}
    delta: Dict[str, List[Optional[str]]] = {}
    for key in interesting & (set(a.qualifiers) | set(b.qualifiers)):
        left, right = a.qualifiers.get(key), b.qualifiers.get(key)
        if left != right:
            delta[key] = [left, right]
    return delta


#: Line-item labels so generic that the word alone identifies nothing. Every
#: financial statement has an "Others" bucket, and different statements' "Others"
#: buckets are essentially never the same thing — one document's residual
#: revenue category and another's residual segment profit share nothing but
#: the label. This list is domain-agnostic: it names common statement-line
#: vocabulary, not anything specific to logistics, India, or this corpus.
_GENERIC_LABEL_CORES = {
    "others", "other", "total", "totals", "net", "gross", "balance",
    "miscellaneous", "sundry", "various", "change", "movement", "movements",
    "adjustment", "adjustments", "remaining", "reconciliation", "difference",
    "subtotal", "provision", "provisions", "reserve", "reserves", "addition",
    "additions", "deduction", "deductions", "general", "misc",
}


class PairClassifier:
    """Decides the relationship between two facts that share entity and attribute."""

    def __init__(
        self,
        *,
        profiles: Optional[Dict[str, DocumentProfile]] = None,
        core_similarity_threshold: float = 0.6,
    ) -> None:
        self.profiles = profiles or {}
        self.core_similarity_threshold = core_similarity_threshold

    def _generic_label_mismatch(
        self,
        signature_a: "AttributeSignature",
        signature_b: "AttributeSignature",
        a: AtomicFact,
        b: AtomicFact,
    ) -> Optional[Verdict]:
        """Refuse to compare a bare generic label unless context confirms it.

        "Others = 101.9" and "Others = 1" pass every other check — same entity,
        same core, same period — yet come from a revenue note and a segment
        profit table that have nothing to do with each other. The label alone
        cannot tell them apart, so this requires the disambiguating context
        qualifier to be present on *both* sides and to actually correspond
        before treating them as the same measure. Missing or differing context
        is grounds to decline the comparison, not to assume a match.
        """
        core_a = normalize_key(a.attribute)
        core_b = normalize_key(b.attribute)
        if core_a not in _GENERIC_LABEL_CORES and core_b not in _GENERIC_LABEL_CORES:
            return None

        context_a = a.qualifiers.get("context") or a.qualifiers.get("section")
        context_b = b.qualifiers.get("context") or b.qualifiers.get("section")
        if context_a and context_b:
            key_a, key_b = normalize_key(context_a), normalize_key(context_b)
            if key_a == key_b or key_a in key_b or key_b in key_a:
                return None  # contexts agree — safe to let normal comparison proceed
            return Verdict(
                RelationshipType.ORTHOGONAL,
                f"Both use the generic label '{a.attribute}', but come from different "
                f"contexts ({context_a!r} vs {context_b!r}) — almost certainly unrelated "
                f"line items that happen to share a name, not the same measure.",
                {"dimension": "attribute", "reason": "generic label, mismatched context"},
                confidence=0.7,
            )

        return Verdict(
            RelationshipType.ORTHOGONAL,
            f"'{a.attribute}' is too generic a label to compare without knowing which "
            f"line item it refers to on both sides, and at least one side's surrounding "
            f"context wasn't captured — treating as unrelated rather than guessing.",
            {"dimension": "attribute", "reason": "generic label, context unavailable"},
            confidence=0.4,
        )

    def classify(self, a: AtomicFact, b: AtomicFact) -> Verdict:
        signature_a = attribute_signature(a.attribute)
        signature_b = attribute_signature(b.attribute)
        similarity = signature_a.similarity(signature_b)
        same_document = a.provenance.file_id == b.provenance.file_id

        if similarity < self.core_similarity_threshold:
            return Verdict(
                RelationshipType.ORTHOGONAL,
                f"'{a.attribute}' and '{b.attribute}' measure different things "
                f"(core overlap {similarity:.0%}).",
                {"dimension": "attribute", "core_similarity": round(similarity, 2)},
                confidence=0.5,
            )

        generic_block = self._generic_label_mismatch(signature_a, signature_b, a, b)
        if generic_block is not None:
            return generic_block

        quantity_a, quantity_b = _quantity_of(a), _quantity_of(b)
        if quantity_a is None or quantity_b is None:
            return self._non_numeric(a, b, similarity)

        agreement = compare_quantities(quantity_a, quantity_b)
        if not agreement.comparable:
            return Verdict(
                RelationshipType.ORTHOGONAL,
                f"Not comparable: {agreement.reason}.",
                {"dimension": "unit", "reason": agreement.reason},
                confidence=0.6,
            )

        scope_a, scope_b = a.scope, b.scope
        relation = _period_relation(scope_a, scope_b)
        base = {
            "period_a": a.temporal_scope,
            "period_b": b.temporal_scope,
            "period_relation": relation,
            "value_a": f"{a.value} {a.unit or ''}".strip(),
            "value_b": f"{b.value} {b.unit or ''}".strip(),
            "normalized_a": quantity_a.canonical_value,
            "normalized_b": quantity_b.canonical_value,
            "difference": round(agreement.difference, 4),
            "relative_difference": round(agreement.relative_difference, 6),
            "tolerance": round(agreement.tolerance, 4),
            "agreement_band": agreement.band.value,
            "same_document": same_document,
        }

        if relation == "same":
            return self._same_period(a, b, signature_a, signature_b, agreement, base, same_document)
        if relation in ("a_contains_b", "b_contains_a"):
            return self._nested_periods(a, b, agreement, base, relation)
        if relation == "disjoint":
            return Verdict(
                RelationshipType.ORTHOGONAL,
                f"Same measure at different times ({a.temporal_scope} vs {b.temporal_scope}) — "
                f"a series, not a disagreement.",
                base,
                confidence=0.8,
            )
        return self._unknown_period(a, b, agreement, base)

    # -- branches ----------------------------------------------------------- #
    def _same_period(
        self,
        a: AtomicFact,
        b: AtomicFact,
        signature_a: AttributeSignature,
        signature_b: AttributeSignature,
        agreement: Agreement,
        base: Dict[str, object],
        same_document: bool,
    ) -> Verdict:
        scale_a = parse_unit(a.unit).scale if a.unit and parse_unit(a.unit) else 1.0
        scale_b = parse_unit(b.unit).scale if b.unit and parse_unit(b.unit) else 1.0
        converted = scale_a != scale_b

        if agreement.band is AgreementBand.MATCH:
            detail = agreement.explain()
            if converted:
                detail += f"; stated as {a.unit} in one source and {b.unit} in the other"
            return Verdict(
                RelationshipType.CORROBORATED,
                f"Both report {a.attribute} for {a.temporal_scope}: {detail}.",
                {**base, "dimension": "unit conversion" if converted else "direct"},
                confidence=0.92 if not same_document else 0.8,
            )

        if agreement.band is AgreementBand.ROUNDING_BOUNDARY:
            # A one-unit gap is usually rounding — unless there is a better
            # explanation on the table. Two institutions publishing a year apart
            # are reporting different vintages of the same series, and calling
            # that "corroboration" would hide exactly the thing a reviewer needs
            # to see.
            vintage = self._vintage_delta(a, b)
            if vintage is not None:
                older, newer, gap = vintage
                return Verdict(
                    RelationshipType.APPARENT_CONTRADICTION,
                    f"{a.attribute} for {a.temporal_scope} differs by one rounding unit between "
                    f"two publications {gap} year(s) apart ({older} then {newer}). More likely a "
                    f"revised vintage of the same series than either a disagreement or a "
                    f"coincidence of rounding.",
                    {**base, "dimension": "data vintage", "older": older, "newer": newer},
                    confidence=0.55,
                )
            if signature_a.modifiers_differ(signature_b):
                delta = signature_a.modifier_delta(signature_b)
                return Verdict(
                    RelationshipType.APPARENT_CONTRADICTION,
                    f"'{a.attribute}' and '{b.attribute}' differ by {', '.join(delta)} and their "
                    f"values differ by one rounding unit — a definitional gap, not a conflict.",
                    {**base, "dimension": "definition", "modifier_delta": delta},
                    confidence=0.5,
                )
            return Verdict(
                RelationshipType.CORROBORATED,
                f"Both report {a.attribute} for {a.temporal_scope} and {agreement.explain()}. "
                f"Treated as corroboration at reduced confidence — the gap is within what "
                f"rounding alone can produce, but a real difference of this size would look "
                f"identical.",
                {**base, "dimension": "rounding"},
                confidence=0.55,
            )

        # Values genuinely differ. Work through the reconcilable explanations
        # before reaching for a contradiction.
        if signature_a.modifiers_differ(signature_b):
            delta = signature_a.modifier_delta(signature_b)
            return Verdict(
                RelationshipType.APPARENT_CONTRADICTION,
                f"'{a.attribute}' and '{b.attribute}' cover the same period but are different "
                f"measures — they differ by {', '.join(delta)}. {agreement.explain().capitalize()}.",
                {**base, "dimension": "definition", "modifier_delta": delta},
                confidence=0.85,
            )

        qualifiers = _qualifier_delta(a, b)
        if qualifiers:
            context_only = set(qualifiers) == {"context"}
            return Verdict(
                RelationshipType.APPARENT_CONTRADICTION,
                f"Same measure and period, but the two figures sit under different headings "
                f"({', '.join(f'{k}: {v[0]!r} vs {v[1]!r}' for k, v in qualifiers.items())}) — "
                f"typically standalone against consolidated, or one entity's statement beside "
                f"another's. {agreement.explain().capitalize()}. "
                + ("Worth confirming which basis each figure is on." if context_only else ""),
                {**base, "dimension": "scope qualifier", "qualifier_delta": qualifiers},
                confidence=0.55 if context_only else 0.7,
            )

        vintage = self._vintage_delta(a, b)
        if vintage is not None:
            older, newer, gap = vintage
            return Verdict(
                RelationshipType.APPARENT_CONTRADICTION,
                f"Two publications report {a.attribute} for {a.temporal_scope} differently. "
                f"{newer} is {gap} year(s) later than {older}, so the figures are most likely "
                f"different vintages of the same series — an estimate against a revision. "
                f"{agreement.explain().capitalize()}.",
                {**base, "dimension": "data vintage", "older": older, "newer": newer},
                confidence=0.6,
            )

        weak = self._weak_evidence(a, b)
        if weak:
            # Do not accuse two documents of contradicting each other on the
            # strength of a binding this system guessed. Where the column
            # association was inferred rather than read, the difference is at
            # least as likely to be our error as theirs, and saying that is more
            # useful to a reviewer than a confident wrong verdict.
            return Verdict(
                RelationshipType.APPARENT_CONTRADICTION,
                f"The figures for {a.attribute} in {a.temporal_scope} differ, but "
                f"{weak} — so this may be an extraction artefact rather than a real "
                f"disagreement. {agreement.explain().capitalize()}. Check the two quotes "
                f"before treating it as a discrepancy.",
                {**base, "dimension": "extraction uncertainty", "weakness": weak},
                confidence=0.35,
            )

        return Verdict(
            RelationshipType.GENUINE_CONTRADICTION,
            f"Same entity, same measure ({a.attribute}), same period ({a.temporal_scope}), "
            f"same units, no qualifier or vintage difference to explain it — "
            f"{agreement.explain()}.",
            {**base, "dimension": "none found"},
            confidence=0.8 if not base.get("same_document") else 0.7,
        )

    @staticmethod
    def _weak_evidence(a: AtomicFact, b: AtomicFact) -> Optional[str]:
        """Describe why a pair is too weakly grounded to call a contradiction."""
        reasons = []
        if a.qualifiers.get("binding") or b.qualifiers.get("binding"):
            reasons.append("at least one value's column was inferred, not read")
        if min(a.confidence, b.confidence) < 0.6:
            reasons.append(
                f"extraction confidence is low ({min(a.confidence, b.confidence):.2f})"
            )
        if a.provenance.match_strategy.value == "FUZZY" or b.provenance.match_strategy.value == "FUZZY":
            reasons.append("a quote matched only approximately")
        return " and ".join(reasons) if reasons else None

    def _nested_periods(
        self,
        a: AtomicFact,
        b: AtomicFact,
        agreement: Agreement,
        base: Dict[str, object],
        relation: str,
    ) -> Verdict:
        whole, part = (a, b) if relation == "a_contains_b" else (b, a)
        whole_months = whole.scope.month_span or 12
        part_months = part.scope.month_span or 0
        if agreement.band is AgreementBand.MATCH:
            return Verdict(
                RelationshipType.ORTHOGONAL,
                f"{part.temporal_scope} is inside {whole.temporal_scope}; the values coincide, "
                f"which for a sub-period is unusual enough to be worth an eye rather than a claim.",
                {**base, "dimension": "temporal scope", "relation": "sub-period"},
                confidence=0.4,
            )
        share = None
        if agreement.comparable and whole.comparable_value:
            try:
                share = round(abs(part.comparable_value or 0) / abs(whole.comparable_value), 3)
            except ZeroDivisionError:
                share = None
        if share is not None and share > 1.05:
            # A quarter cannot exceed its own year. When it appears to, the
            # period story is not the explanation — one of the two facts is
            # mis-extracted, and saying so is more useful than a tidy verdict.
            return Verdict(
                RelationshipType.APPARENT_CONTRADICTION,
                f"{part.temporal_scope} sits inside {whole.temporal_scope}, yet its value is "
                f"{share:.0%} of the longer period's. A sub-period cannot exceed its whole, so "
                f"one of these two figures is very likely mis-extracted rather than genuinely "
                f"in conflict. {agreement.explain().capitalize()}.",
                {
                    **base,
                    "dimension": "temporal scope",
                    "relation": "sub-period",
                    "part_share_of_whole": share,
                    "flag": "implausible — review extraction",
                },
                confidence=0.35,
            )
        return Verdict(
            RelationshipType.APPARENT_CONTRADICTION,
            f"These look contradictory but cover different windows: {whole.temporal_scope} "
            f"({whole_months} months) against {part.temporal_scope} ({part_months} months). "
            f"The shorter period accounts for "
            f"{f'{share:.0%}' if share is not None else 'part'} of the longer one, which is "
            f"consistent with a slice of the same series rather than a disagreement.",
            {
                **base,
                "dimension": "temporal scope",
                "relation": "sub-period",
                "whole": whole.temporal_scope,
                "part": part.temporal_scope,
                "part_share_of_whole": share,
            },
            confidence=0.85,
        )

    def _unknown_period(
        self, a: AtomicFact, b: AtomicFact, agreement: Agreement, base: Dict[str, object]
    ) -> Verdict:
        if agreement.band is AgreementBand.MATCH:
            return Verdict(
                RelationshipType.CORROBORATED,
                f"Values agree for {a.attribute}, but at least one source did not state a "
                f"period, so this is corroboration on the number only.",
                {**base, "dimension": "period unknown"},
                confidence=0.45,
            )
        return Verdict(
            RelationshipType.ORTHOGONAL,
            f"Values differ but at least one period is unstated — not enough to call either "
            f"way.",
            {**base, "dimension": "period unknown"},
            confidence=0.3,
        )

    def _non_numeric(
        self, a: AtomicFact, b: AtomicFact, similarity: float
    ) -> Verdict:
        left = collapse_whitespace(str(a.value)).casefold()
        right = collapse_whitespace(str(b.value)).casefold()
        relation = _period_relation(a.scope, b.scope)
        if left == right:
            return Verdict(
                RelationshipType.CORROBORATED,
                f"Both state {a.attribute} as '{a.value}'.",
                {"dimension": "text", "period_relation": relation},
                confidence=0.7,
            )
        if relation == "same":
            return Verdict(
                RelationshipType.GENUINE_CONTRADICTION,
                f"Same period, same measure, incompatible statements: '{a.value}' vs '{b.value}'.",
                {"dimension": "text", "value_a": a.value, "value_b": b.value},
                confidence=0.6,
            )
        return Verdict(
            RelationshipType.APPARENT_CONTRADICTION,
            f"'{a.value}' and '{b.value}' differ but describe different points in time "
            f"({a.temporal_scope} vs {b.temporal_scope}) — a change of state, not a conflict.",
            {"dimension": "temporal scope", "value_a": a.value, "value_b": b.value},
            confidence=0.6,
        )

    def _vintage_delta(self, a: AtomicFact, b: AtomicFact) -> Optional[Tuple[str, str, int]]:
        left = self.profiles.get(a.provenance.file_id)
        right = self.profiles.get(b.provenance.file_id)
        if not left or not right or left.file_id == right.file_id:
            return None
        if not left.vintage_year or not right.vintage_year:
            return None
        gap = abs(left.vintage_year - right.vintage_year)
        if gap == 0:
            return None
        older, newer = (left, right) if left.vintage_year < right.vintage_year else (right, left)
        return older.label, newer.label, gap


# --------------------------------------------------------------------------- #
# The reconciler
# --------------------------------------------------------------------------- #
class ReconciliationReport(BaseModel):
    """Counters for what the reconciler did and how much it skipped."""

    model_config = ConfigDict(extra="forbid")

    facts: int = 0
    entity_clusters: int = 0
    blocks: int = 0
    pairs_evaluated: int = 0
    pairs_skipped_budget: int = 0
    relationships_by_type: Dict[str, int] = Field(default_factory=dict)
    largest_block: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.relationships_by_type.items()))
        return (
            f"{self.facts} facts → {self.blocks} comparison blocks, "
            f"{self.pairs_evaluated} pairs in {self.seconds:.1f}s\n"
            f"  entities: {self.entity_clusters} clusters, largest block {self.largest_block} facts\n"
            f"  relationships: {counts or 'none'}"
        )


@dataclass
class ReconciliationResult:
    relationships: List[FactRelationship]
    report: ReconciliationReport
    resolver: EntityResolver
    profiles: Dict[str, DocumentProfile]


class Reconciler:
    """Blocks facts into comparable groups and classifies every pair inside them.

    Blocking is what makes this tractable: comparing 2,400 facts pairwise is 2.9
    million comparisons, but facts can only relate if they share an entity
    cluster and an attribute core, which cuts it to a few thousand. Groups larger
    than ``max_block_size`` are skipped and counted rather than silently
    truncated — an unnoticed omission in a reconciliation tool is worse than a
    reported one.
    """

    def __init__(
        self,
        *,
        alias_map: Optional[Dict[str, str]] = None,
        emit_orthogonal: bool = False,
        max_block_size: int = 250,
        min_confidence: float = 0.0,
        cross_document_only: bool = False,
        core_similarity_threshold: float = 0.6,
    ) -> None:
        self.alias_map = alias_map
        self.emit_orthogonal = emit_orthogonal
        self.max_block_size = max_block_size
        self.min_confidence = min_confidence
        self.cross_document_only = cross_document_only
        self.core_similarity_threshold = core_similarity_threshold

    def run(
        self,
        facts: Sequence[AtomicFact],
        *,
        document_subjects: Optional[Dict[str, str]] = None,
        document_names: Optional[Dict[str, str]] = None,
    ) -> ReconciliationResult:
        started = time.time()
        report = ReconciliationReport(facts=len(facts))

        resolver = EntityResolver(self.alias_map)
        resolver.fit(facts, document_subjects)
        report.entity_clusters = len(resolver.clusters)

        profiles = profile_documents(facts, names=document_names, subjects=document_subjects)
        classifier = PairClassifier(
            profiles=profiles, core_similarity_threshold=self.core_similarity_threshold
        )

        blocks: Dict[Tuple[str, str], List[AtomicFact]] = defaultdict(list)
        for fact in facts:
            entity_key = resolver.cluster_key(
                fact.entity, file_id=fact.provenance.file_id, document_subjects=document_subjects
            )
            signature = attribute_signature(fact.attribute)
            blocks[(entity_key, signature.core_key)].append(fact)
        report.blocks = len(blocks)
        report.largest_block = max((len(v) for v in blocks.values()), default=0)

        relationships: List[FactRelationship] = []
        seen: Set[Tuple[str, str]] = set()

        for members in blocks.values():
            if len(members) < 2:
                continue
            if len(members) > self.max_block_size:
                report.pairs_skipped_budget += len(members) * (len(members) - 1) // 2
                continue
            for left, right in itertools.combinations(members, 2):
                if self.cross_document_only and left.provenance.file_id == right.provenance.file_id:
                    continue
                pair = tuple(sorted((left.id, right.id)))
                if pair in seen:
                    continue
                seen.add(pair)  # type: ignore[arg-type]
                report.pairs_evaluated += 1

                verdict = classifier.classify(left, right)
                if verdict.relationship is RelationshipType.ORTHOGONAL and not self.emit_orthogonal:
                    self._count(report, verdict.relationship)
                    continue
                if verdict.confidence < self.min_confidence:
                    continue
                relationships.append(
                    FactRelationship(
                        fact_a_id=left.id,
                        fact_b_id=right.id,
                        relationship_type=verdict.relationship,
                        explanation=verdict.explanation,
                        reconciliation_context=verdict.context,
                        confidence=verdict.confidence,
                        detector=Detector.DETERMINISTIC,
                    )
                )
                self._count(report, verdict.relationship)

        report.seconds = round(time.time() - started, 2)
        relationships.sort(key=lambda r: (-r.confidence, r.relationship_type.value))
        return ReconciliationResult(
            relationships=relationships, report=report, resolver=resolver, profiles=profiles
        )

    @staticmethod
    def _count(report: ReconciliationReport, relationship: RelationshipType) -> None:
        key = relationship.value
        report.relationships_by_type[key] = report.relationships_by_type.get(key, 0) + 1


# --------------------------------------------------------------------------- #
# Knowledge layer
# --------------------------------------------------------------------------- #
class KnowledgeLayer:
    """Facts plus their relationships, with the lookups a UI or API needs."""

    def __init__(
        self,
        facts: Sequence[AtomicFact],
        relationships: Sequence[FactRelationship],
        *,
        profiles: Optional[Dict[str, DocumentProfile]] = None,
        resolver: Optional[EntityResolver] = None,
    ) -> None:
        self.facts: Dict[str, AtomicFact] = {fact.id: fact for fact in facts}
        self.relationships: List[FactRelationship] = list(relationships)
        self.profiles = profiles or {}
        self.resolver = resolver
        self._by_fact: Dict[str, List[FactRelationship]] = defaultdict(list)
        for relationship in self.relationships:
            self._by_fact[relationship.fact_a_id].append(relationship)
            self._by_fact[relationship.fact_b_id].append(relationship)

    def fact(self, fact_id: str) -> Optional[AtomicFact]:
        return self.facts.get(fact_id)

    def relationships_for(self, fact_id: str) -> List[FactRelationship]:
        return list(self._by_fact.get(fact_id, []))

    def of_type(self, relationship_type: RelationshipType) -> List[FactRelationship]:
        return [r for r in self.relationships if r.relationship_type is relationship_type]

    def cross_document(self) -> List[FactRelationship]:
        result = []
        for relationship in self.relationships:
            left, right = self.facts.get(relationship.fact_a_id), self.facts.get(relationship.fact_b_id)
            if left and right and left.provenance.file_id != right.provenance.file_id:
                result.append(relationship)
        return result

    def render(self, relationship: FactRelationship) -> str:
        """A reviewer-facing rendering of one edge, with both quotes."""
        left = self.facts.get(relationship.fact_a_id)
        right = self.facts.get(relationship.fact_b_id)
        if not left or not right:
            return f"{relationship.relationship_type.value}: (missing fact)"
        lines = [
            f"[{relationship.relationship_type.value}] confidence {relationship.confidence:.2f}",
            f"  A: {left.summary()}",
            f"     {left.provenance.locator}",
            f"     “{collapse_whitespace(left.provenance.verbatim_quote)[:160]}”",
            f"  B: {right.summary()}",
            f"     {right.provenance.locator}",
            f"     “{collapse_whitespace(right.provenance.verbatim_quote)[:160]}”",
            f"  Why: {relationship.explanation}",
        ]
        return "\n".join(lines)


def reconcile(
    facts: Sequence[AtomicFact], **kwargs: object
) -> ReconciliationResult:
    """Convenience wrapper around :class:`Reconciler`."""
    subjects = kwargs.pop("document_subjects", None)
    names = kwargs.pop("document_names", None)
    return Reconciler(**kwargs).run(  # type: ignore[arg-type]
        facts, document_subjects=subjects, document_names=names  # type: ignore[arg-type]
    )


__all__ = [
    "RECONCILER_VERSION",
    "normalize_entity",
    "EntityResolver",
    "AttributeSignature",
    "attribute_signature",
    "DocumentProfile",
    "profile_documents",
    "Verdict",
    "PairClassifier",
    "Reconciler",
    "ReconciliationReport",
    "ReconciliationResult",
    "KnowledgeLayer",
    "reconcile",
]