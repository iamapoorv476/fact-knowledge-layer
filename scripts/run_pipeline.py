"""
run_pipeline.py — End-to-end smoke test: parse -> extract -> reconcile -> report.

Usage:
    python run_pipeline.py file1.pdf [file2.pdf ...]

With one file: prints extraction stats and a sample of facts.
With two or more files: also runs reconciliation and prints cross-document
relationships (corroborations and contradictions).
"""
from __future__ import annotations

import sys
import time

from fkl.parser import parse_pdf, verify_anchor
from fkl.extractor import ExtractionPipeline
from fkl.reconcile import Reconciler, KnowledgeLayer
from fkl.schemas import RelationshipType as RT


def main(paths):
    if not paths:
        print("usage: python run_pipeline.py file1.pdf [file2.pdf ...]")
        sys.exit(1)

    pipeline = ExtractionPipeline(use_llm=False)  # deterministic only, no API key needed
    all_facts = []
    subjects = {}
    names = {}

    for path in paths:
        print(f"\n{'='*70}\nPARSING: {path}")
        t0 = time.time()
        doc = parse_pdf(path)
        print(f"  {doc.page_count} pages, {doc.total_chars:,} chars, "
              f"{sum(1 for p in doc.pages if not p.has_text_layer)} blank, "
              f"parsed in {time.time()-t0:.1f}s")
        if doc.warnings:
            print(f"  warnings: {doc.warnings}")

        t0 = time.time()
        result = pipeline.run(doc)
        print(f"EXTRACTED: {result.report.total_facts} facts in {time.time()-t0:.1f}s")
        print(f"  subject: {result.report.subject}")
        print(f"  blocks by role: {result.report.blocks_by_role}")
        print(f"  facts by extractor: {result.report.facts_by_extractor}")

        bad = sum(1 for f in result.facts if not verify_anchor(doc, f.provenance, strict=True))
        print(f"  anchor integrity: {len(result.facts)-bad}/{len(result.facts)} byte-exact")

        print(f"\n  Sample facts:")
        for f in result.facts[:8]:
            print(f"    [{f.extractor:14s} conf={f.confidence:.2f}] {f.entity} | "
                  f"{f.attribute} = {f.value} {f.unit or ''} @ {f.temporal_scope}")
            print(f"        p{f.provenance.page_number} \"{f.provenance.verbatim_quote[:80]}\"")

        all_facts.extend(result.facts)
        subjects[result.report.file_id] = result.report.subject
        names[result.report.file_id] = result.report.filename

    if len(paths) < 2:
        print(f"\n{'='*70}\nProvide 2+ PDFs to also see cross-document reconciliation.")
        return

    print(f"\n{'='*70}\nRECONCILING {len(all_facts)} facts across {len(paths)} documents...")
    result = Reconciler().run(all_facts, document_subjects=subjects, document_names=names)
    print(result.report.summary())
    kl = KnowledgeLayer(all_facts, result.relationships, profiles=result.profiles)

    cross_doc = kl.cross_document()
    print(f"\ncross-document relationships: {len(cross_doc)}")
    for rel_type in (RT.CORROBORATED, RT.GENUINE_CONTRADICTION, RT.APPARENT_CONTRADICTION):
        matches = [r for r in cross_doc if r.relationship_type is rel_type]
        print(f"\n{'-'*70}\n{rel_type.value} ({len(matches)} found, showing up to 3)")
        for r in matches[:3]:
            print(kl.render(r))
            print()


if __name__ == "__main__":
    main(sys.argv[1:])