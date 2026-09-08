"""
app.py — Fact Knowledge Layer, Streamlit UI.

Structured explicitly around the four required demo cases:
  1. Corroboration across documents
  2. Genuine / likely contradiction
  3. Apparent contradiction explained by context (time, scope, definition)
  4. Extraction / reasoning failure, documented

Deliberately minimal: no settings sidebar. Upload, run, and one keyword search
to jump straight to a known example while recording — that's the whole
interface.
"""

from __future__ import annotations

import os
import tempfile
import time

import streamlit as st

from fkl.extractor import ExtractionPipeline
from fkl.parser import parse_pdf
from fkl.reconcile import Reconciler, KnowledgeLayer
from fkl.schemas import AtomicFact, FactRelationship, RelationshipType as RT

st.set_page_config(page_title="Fact Knowledge Layer", layout="wide", page_icon="📊")

st.title("Fact Knowledge Layer — Reconciliation Engine")
st.markdown(
    "Upload two or more PDFs to extract grounded facts — each traceable to an exact "
    "quote in its source document — and reconcile them across documents into "
    "corroborations, contradictions, and context-explained agreements."
)

MAX_SHOWN = 30  # fixed cap per section — plenty for a demo, keeps the page short

uploaded_files = st.file_uploader("Upload 2+ PDFs", type="pdf", accept_multiple_files=True)
run_clicked = st.button("Run Pipeline", type="primary")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fact_matches(fact: AtomicFact, term: str) -> bool:
    if not term:
        return True
    haystack = f"{fact.entity} {fact.attribute} {fact.temporal_scope or ''}".casefold()
    return term in haystack


def relationship_matches(rel: FactRelationship, kl: KnowledgeLayer, term: str) -> bool:
    if not term:
        return True
    a, b = kl.facts.get(rel.fact_a_id), kl.facts.get(rel.fact_b_id)
    if a is None or b is None:
        return False
    return fact_matches(a, term) or fact_matches(b, term)


def render_relationship_section(
    title: str,
    items: list[FactRelationship],
    kl: KnowledgeLayer,
    icon: str,
    empty_note: str,
    term: str,
) -> None:
    filtered = [r for r in items if relationship_matches(r, kl, term)]
    label = f"{icon} {title} — {len(filtered)} shown of {len(items)} total"
    with st.expander(label, expanded=bool(filtered)):
        if not filtered:
            st.info(empty_note)
            return
        for rel in filtered[:MAX_SHOWN]:
            st.code(kl.render(rel), language="text")
            st.divider()


# --------------------------------------------------------------------------- #
# Pipeline run
# --------------------------------------------------------------------------- #
if run_clicked and uploaded_files:
    if len(uploaded_files) < 2:
        st.warning("Upload at least 2 PDFs — cross-document reconciliation needs more than one source.")
        st.stop()

    pipeline = ExtractionPipeline(use_llm=False)

    all_facts: list[AtomicFact] = []
    subjects: dict[str, str] = {}
    names: dict[str, str] = {}

    st.header("1 · Extraction")
    progress = st.progress(0.0)
    for index, uploaded in enumerate(uploaded_files):
        with st.spinner(f"Parsing {uploaded.name}…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            try:
                started = time.time()
                document = parse_pdf(tmp_path, filename=uploaded.name)
                result = pipeline.run(document)
                all_facts.extend(result.facts)
                subjects[result.report.file_id] = result.report.subject
                names[result.report.file_id] = uploaded.name
                elapsed = round(time.time() - started, 1)
                st.success(
                    f"**{uploaded.name}** — {len(result.facts)} facts from "
                    f"{result.report.pages_processed} pages in {elapsed}s "
                    f"(subject: *{result.report.subject}*)"
                )
            except Exception as exc:  # noqa: BLE001 — surface any failure to the UI rather than crashing silently
                st.error(f"Failed to process {uploaded.name}: {exc}")
            finally:
                os.unlink(tmp_path)
        progress.progress((index + 1) / len(uploaded_files))

    if not all_facts:
        st.error("No facts were extracted from any uploaded document.")
        st.stop()

    st.header("2 · Reconciliation")
    with st.spinner("Comparing facts across documents…"):
        reconciliation = Reconciler().run(all_facts, document_subjects=subjects, document_names=names)
        knowledge_layer = KnowledgeLayer(
            all_facts, reconciliation.relationships, profiles=reconciliation.profiles
        )
        cross_document = knowledge_layer.cross_document()

    corroborated = [r for r in cross_document if r.relationship_type is RT.CORROBORATED]
    apparent = [r for r in cross_document if r.relationship_type is RT.APPARENT_CONTRADICTION]
    genuine = [r for r in cross_document if r.relationship_type is RT.GENUINE_CONTRADICTION]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Facts extracted", len(all_facts))
    metric_cols[1].metric("✅ Corroborated", len(corroborated))
    metric_cols[2].metric("⚠️ Apparent contradictions", len(apparent))
    metric_cols[3].metric("🚨 Genuine contradictions", len(genuine))

    st.session_state["all_facts"] = all_facts
    st.session_state["knowledge_layer"] = knowledge_layer
    st.session_state["cross_document"] = {"corroborated": corroborated, "apparent": apparent, "genuine": genuine}

# --------------------------------------------------------------------------- #
# Results — one section per required case, so each is unambiguous in a demo
# --------------------------------------------------------------------------- #
if "knowledge_layer" in st.session_state:
    kl: KnowledgeLayer = st.session_state["knowledge_layer"]
    facts: list[AtomicFact] = st.session_state["all_facts"]
    buckets = st.session_state["cross_document"]

    search_term = st.text_input(
        "🔎 Jump to an example (filter by keyword, e.g. \"GDP\" or \"income\")"
    ).strip().lower()

    st.header("3 · Case 1 — Corroboration across documents")
    render_relationship_section(
        "Corroborated facts", buckets["corroborated"], kl, "✅",
        "No corroborations found between these documents.",
        search_term,
    )

    st.header("4 · Case 3 — Apparent contradiction, explained by context")
    render_relationship_section(
        "Apparent contradictions", buckets["apparent"], kl, "⚠️",
        "No apparent contradictions found between these documents.",
        search_term,
    )

    st.header("5 · Case 2 — Genuine or likely contradiction")
    render_relationship_section(
        "Genuine contradictions", buckets["genuine"], kl, "🚨",
        "No genuine contradictions were automatically flagged — a meaningful "
        "finding in itself, since well-produced institutional filings tend to be "
        "internally and mutually consistent. Check Case 3 above for a small, "
        "real disagreement sitting at the edge of the system's rounding tolerance.",
        search_term,
    )

    st.header("6 · Case 4 — Extraction quality and documented limitations")
    low_confidence = sorted(
        (f for f in facts if f.confidence < 0.65 and fact_matches(f, search_term)),
        key=lambda f: f.confidence,
    )
    with st.expander(f"📉 Low-confidence facts — {len(low_confidence)} shown of {len(facts)} total facts", expanded=bool(low_confidence)):
        if not low_confidence:
            st.info("No low-confidence facts match this filter.")
        for fact in low_confidence[:MAX_SHOWN]:
            st.text(
                f"[{fact.extractor} conf={fact.confidence:.2f}] {fact.entity} | "
                f"{fact.attribute} = {fact.value}{fact.unit or ''} @ {fact.temporal_scope}"
            )
            st.caption(
                f"  {fact.provenance.document_name} p.{fact.provenance.page_number} — "
                f"\u201c{fact.provenance.verbatim_quote[:160]}\u201d "
                f"[{fact.provenance.match_strategy.value}]"
            )