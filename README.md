# Fact Knowledge Layer

This is a system that lets you extract valuable facts and points from your documents and analyze them completely,
surfacing all the important facts. It works across documents, showing where they corroborate each other,where they
contradict each other, and where an apparent contradiction is really just explained by context.

I built and tested this against six real documents - a Delhivery IPO prospectus, its FY24 annual report, its Q4FY24 earnings deck, India's Economic Survey 2024-25, the RBI's Annual Report 2024-25, and the IMF's 2025 India Article IV report - none of which the code has any hardcoded knowledge of. I ran multiple rounds of testing, first on single documents and then across documents, specifically to make the architecture strong rather than something that only works on the examples I happened to build it against.

---

## Setup and Run Instructions

### 1. Requirements
 
- Python 3.9+
- `pymupdf`, `pydantic` (core pipeline)
- `streamlit` (UI)
### 2. Clone the repository and enter the project folder
 
```bash
git clone https://github.com/iamapoorv476/fact-knowledge-layer.git
cd fact-knowledge-layer
```
 
### 3. (Recommended) Create and activate a virtual environment
 
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```
 
### 4. Install dependencies
 
```bash
pip install pymupdf pydantic streamlit
```

### 5. Project structure
 
```
fact-knowledge-layer/
├── app.py                      # Streamlit UI
├── src/
│   └── fkl/
│       ├── schemas.py          # Pydantic data contracts
│       ├── parser.py           # PyMuPDF parsing + quote anchoring
│       ├── normalize.py        # units, scales, precision-aware agreement
│       ├── extractor.py        # block classification + fact extractors
│       └── reconcile.py        # entity/attribute resolution + classifier
├── scripts/
│   └── run_pipeline.py         # CLI entry point
└── README.md
```
 
The pipeline code lives under `src/fkl/`, so it needs to be on the Python path — every command below sets `PYTHONPATH=src` for exactly this reason.
 
### 6. Run the UI
 
```bash
PYTHONPATH=src python -m streamlit run app.py
```

This opens the app in your browser (usually `localhost:8501`). From there:
 
1. Upload **two or more** PDFs using the file uploader.
2. Click **Run Pipeline** and wait for extraction to finish for each file (a progress bar and a green summary line appear per document — table-heavy PDFs can take longer, see the performance note below).
3. Once reconciliation finishes, four numbered result sections appear, matching the assignment's four required cases:
   - **Case 1 — Corroboration** (§3)
   - **Case 3 — Apparent contradiction** (§4)
   - **Case 2 — Genuine/likely contradiction** (§5)
   - **Case 4 — Extraction limitations** (§6, low-confidence facts)
4. Use the **keyword search box** above the results to jump straight to a specific fact or relationship (e.g. type "GDP" or "income") instead of scrolling.

**Tip:** not every pair of PDFs will show something in every section — that depends on whether the documents actually cover overlapping periods and comparable measures. For a pair that reliably produces both Case 1 and Case 3, use the Q4 FY24 earnings deck together with the FY24 annual report (both report the same fiscal year, so their figures overlap).

### 7. Run via CLI instead
 
```bash
PYTHONPATH=src python scripts/run_pipeline.py path/to/file1.pdf path/to/file2.pdf
```
 
Prints extraction stats per document (fact count, anchor integrity, sample facts) and, with two or more files, runs reconciliation and prints corroborations/contradictions with full source quotes directly in the terminal.

### A note on performance
 
Table-detection speed depends heavily on the installed PyMuPDF version — confirmed directly during development: the same 100-page Delhivery annual report took **~500 seconds on PyMuPDF 1.26.7** and dropped to **~49 seconds after upgrading** to a current release, on the same machine, with no code changes. If a document is taking multiple minutes to parse, `pip install --upgrade pymupdf` is the first thing to try, and it is the single highest-leverage fix available. Lighter documents — slide decks, prose-heavy reports — parse in 1–30 seconds regardless of version, since table detection is the dominant cost and they have little to no tabular content.
 
**Practical implication:** a 100-page statutory filing will still take tens of seconds even on a fast setup, simply because of how much character parsing and block traversal a dense table-heavy document requires. For live demonstration or rapid iteration, a lightweight excerpt or an earnings presentation stays the more convenient choice; the full statutory filing remains the right input when thoroughness matters more than turnaround time. The Streamlit UI also caches parsed results by file content, so re-processing an already-seen PDF (common while testing different document pairings) is instant on the second run.
 
---

## Video Demo
 https://youtu.be/BMbglXiaeiY
---

## Approach
 
### Architecture
 
Five modules, each with a single responsibility:
 
| Module | Responsibility |
|---|---|
| `schemas.py` | Pydantic v2 data contracts: `AtomicFact`, `SourceAnchor`, `FactRelationship`, `TemporalScope` (with period parsing/comparison logic) |
| `parser.py` | Deterministic PyMuPDF text/table extraction, two-column reading-order detection, and the quote-anchoring match ladder (exact → normalized → case-insensitive → fuzzy) |
| `normalize.py` | Unit parsing (currency, Indian/SI scale words, percentages), and precision-aware numeric agreement (a tolerance derived from how precisely each number was *written*, not a fixed threshold) |
| `extractor.py` | Block classification (table / chart / prose / heading / boilerplate) and four extractors: table rows, borderless-statement rows, prose sentences, and an optional LLM pass |
| `reconcile.py` | Entity alias resolution, attribute-core blocking, and the four-way relationship classifier (`CORROBORATED` / `GENUINE_CONTRADICTION` / `APPARENT_CONTRADICTION` / `ORTHOGONAL`) |

### Core design decisions

**1. I made verbatim provenance a hard constraint, not a best-effort feature.** A `SourceAnchor` can't be constructed from an unverified quote, I built the validation to actively reject an `UNVERIFIED` match strategy. Every fact's quote is checked against the actual parsed page text before I ever let it get stored. I verified this across the full six-document corpus:**100% of extracted facts (2,000+) resolve byte-exactly to their claimed source span.** This held true throughout development, not just at the end, it was the first thing I tested and the first thing I re-tested after every single change.

**2. I went with a hybrid architecture: a deterministic core, with LLM extraction as an isolated, optional adapter.** Three regex/layout-based extractors (`table-v1`, `orphan-row-v1`, `prose-v1`) do all extraction by default, no API key needed, fully reproducible, and this is what gives me the 100% byte-exact anchor provenance above with zero hallucination risk on numeric facts. I built `LLMFactExtractor` as a separate, opt-in adapter, designed to reach semantic/categorical facts the deterministic path structurally can't, a board or director status change, for instance, has no number for a regex pattern to anchor to. Its output goes through the exact same grounding gate as everything else: a model-proposed quote that doesn't resolve exactly in the source page gets dropped, never stored. I verified this with a mock transport that included a deliberately hallucinated fact, and it was correctly rejected while three genuine facts passed through, but I want to be precise about what that actually proves: it confirms the grounding gate holds under an adversarial LLM response, not that I demonstrated categorical extraction end-to-end against a live model. I've called that distinction out explicitly in Limitations below rather than glossing over it.

**3. I chose an open predicate schema over a fixed fact taxonomy.** `AtomicFact` has no enumerated "fact type": `entity`, `attribute`, `value`, `unit`, and `temporal_scope` are all free-form, populated according to what the document actually says. This lets me run the same code path against a logistics company's revenue line and a central bank's GDP forecast without writing per-document configuration.

**4. I deliberately kept blocking and classification as separate stages.** Facts get grouped into comparison blocks by (entity, attribute-core) using exact string matching, which is cheap and bounded even on an 800+ fact document. Only within a block does the classifier apply fuzzy reasoning: period nesting, unit conversion, and modifier differences. This separation is also where the most instructive trade-off of the whole project came from. I'll walk through it below.

**5. I used content-hash fact IDs so incremental ingestion comes for free.** A fact's ID is a hash of its source file, page, character offset, and content. Re-ingesting the same PDF produces identical IDs, so re-running the pipeline on a document already in a knowledge base is a no-op instead of generating duplicates.

**6. I built numeric reconciliation around a half_ulp tolerance instead of naive equality.** I never compare two figures using a fixed threshold. Each Quantity carries a half_ulp, meaning half the value of its own last written digit in canonical units. This means "8,142" (written to zero decimals, in crore) and "81,415.38" (written to two decimals, in million) are compared using a tolerance derived from the precision of each side after unit conversion. This lets the engine correctly match ₹453 crore against ₹4,526.96 million without either side being explicitly given the conversion factor in advance. A real example from the corpus: Delhivery's Q4 FY24 earnings deck states FY24 "Other income" as ₹453 crore (p. 17), while the annual report states the identical fact as ₹4,526.96 million (p. 68). ₹453 crore converts to ₹4,530 million, a ₹3.04 million difference that falls within the ±₹5 million tolerance derived from the deck's own precision, so the engine correctly flags it as CORROBORATED at confidence 0.92.

**7. I made the temporal comparison context-aware rather than relying on period equality alone.** Two periods are not simply checked for equality. TemporalScope.contains() evaluates genuine sub-period containment, such as a quarter within the fiscal year it belongs to or a nine-month figure within a full fiscal year. This lets the reconciler distinguish "these disagree" from "one of these is a slice of the other." This mechanism is demonstrated by a real example from the corpus: the same deck states Q3 FY24 "Other income" as ₹131 crore, while the annual report states FY2024 "Other income" as ₹4,526.96 million. Taken at face value, these appear contradictory (₹131 Cr vs ₹452.696 Cr). However, the engine recognizes Q3 FY24 as a three-month sub-period nested within the twelve-month FY2024, computes that the quarter accounts for 29% of the annual total, and correctly classifies it as APPARENT_CONTRADICTION at confidence 0.85. The explanation explicitly states the sub-period relationship rather than returning only a bare label.

### Trade-offs, with evidence
 
The most valuable parts of this build came from decisions I tested, measured, and in one case reversed after the evidence came in.

**Blocking strictness vs. fuzzy attribute matching.** Exact-match blocking is precise, but it misses genuine matches that are worded slightly differently. For example, `"foreign exchange reserves"` (Economic Survey) and `"foreign exchange (FX) reserves"` (IMF) never even got compared, despite referring to the same measure. I built a fix that merges blocks whose attribute cores are near-duplicates, using the same Jaccard-similarity measure the classifier already relies on internally. It worked for that case. I then stress-tested it against a larger corpus and found that it created **three new categories of false contradictions**: `"Gross Capital Formation"` vs. `"Gross Fixed Capital Formation"`, `"Commercial Vehicle Sales"` vs. `"...Retail Sales"`, and `"Non-debt Receipts"` vs. `"...Capital Receipts"`. A single added word can either clarify meaning (`fx`) or narrow the scope (`fixed`, `retail`, `capital`), and I couldn't find a purely structural rule that reliably distinguished the two. The case it fixed did not even change the final answer because the values turned out to represent different points in a time series anyway. The net effect was real harm with no meaningful gain, so I reverted it. I've kept this as a documented design boundary rather than smoothing it over. The full reasoning is in `reconcile.py`'s docstrings.

**Modifier vocabulary: presentation variants vs. structural relationships.** Words like `adjusted`, `consolidated`, and `restated` genuinely mean "the same measure, computed differently." Stripping them out and comparing what remains is therefore the right approach. This is exactly how `"EBITDA"` vs. `"Adjusted EBITDA"` correctly becomes an apparent contradiction by definition. However, I originally put `total`, `other`, `net`, `gross`, and `current` in the same bucket, and that was wrong. `"Other income"` and `"Total income"` both reduced to the bare core `"income"` and were compared as if they were interchangeable, even though one is a component and the other is the aggregate it contributes to. I found this collision in the Delhivery corpus and split the vocabulary into two categories once I identified it.

**Numeric-only deterministic extraction, LLM extraction for everything else.** My regex extractors will never invent a fact, but they also can't capture a categorical claim like "Director X resigned" — there's no number for a verb-phrase pattern to anchor to. I treated this as a deliberate scope boundary: I'm giving up recall on categorical facts in exchange for zero hallucination risk on numeric ones.

**Document "vintage" from data content, not metadata.** Neither PDF here carries a usable creation date in its metadata, both just showed the re-export timestamp of whatever tool built this assignment's curated excerpts, mere seconds apart, useless as a proxy. So I estimate vintage from the **median** year mentioned across a document's own facts instead of the max, because a single long-range forecast table (IMF's report tabulates projections out to 2030/31) would otherwise drag the estimate years into the future. This is a real, general-purpose fix, it took an estimated 5-year publication gap between RBI and IMF down to an honest ~1-year estimate (the true gap is closer to 7 months, I couldn't find a fully general date-detector reliable enough to nail that exact number without introducing new false positives, so I documented it as a known imprecision instead of hiding it).

**Reporting genuine ambiguity honestly instead of forcing a confident label.** RBI's Annual Report and the IMF's Article IV report both forecast India's real GDP growth for the identical fiscal year: RBI (p.17) says 6.5%, IMF (p.13) says 6.6% — same entity, same attribute wording, same period. I could have called this `GENUINE_CONTRADICTION` outright, but a 0.1-percentage-point gap between two figures each written to one decimal place sits exactly at the mathematical boundary of what plain rounding could produce. Rather than overclaim at a boundary that narrow, I had the classifier report it as `APPARENT_CONTRADICTION` at 0.55 confidence and look for an explanation (a real, if small, ~1-year data-vintage difference) before settling on a verdict. I'd still call this a *likely* contradiction in plain language — two independent institutions rarely converge on identical decimal forecasts — but I'd rather the system say "this is ambiguous, and here's why" than force false confidence either way. It's worth noting how rare a finding like this is in this corpus at all: I found zero genuine numeric contradictions across all six documents and 2,000+ facts, and I don't think that's the system failing to look hard enough — it checked every attribute-matched pair across the whole corpus. Audited statutory filings are legally required to be internally consistent, and Delhivery's own annual report is a small proof of that: **S.R. Batliboi & Associates LLP** served as statutory auditor through Q1 FY24 before **Deloitte Haskins & Sells LLP** took over for the rest of the year (p.48, signed report on p.66-70), and the resulting consolidated figures still reconcile cleanly everywhere I checked. When a real numeric contradiction does show up, it's far more likely to be a restatement, a consolidation-basis difference, or a reporting-vintage gap — which is exactly why I built the classifier to reach for those explanations before it reaches for "contradiction."

### AI tools used

I used Claude (Anthropic) as my coding assistant throughout this project, giving it detailed prompts to implement the key architecture decisions I'd already landed on, then using it to write the actual code and to test various parts of the pipeline. The most substantive use of it was adversarial testing: I had it run the pipeline against the actual PDFs repeatedly, find real bugs from the real output (not synthetic test cases), trace each one to its exact root cause in the source text, fix it, and re-verify against the full corpus before moving on. Several fixes took two or three rounds after the first attempt turned out to be incomplete or introduced a new, different problem, documented inline in the code comments and summarized in the sections above.

---

## Brownie Points Addressed
 
- **Large PDFs:** tested on 100-page financial statements (Delhivery annual report, RBI annual report) without restructuring.
- **Many PDFs in one knowledge layer:** verified with 3-document (Delhivery prospectus + deck + annual report) and 6-document combined reconciliation runs.
- **Schema evolves dynamically:** `AtomicFact.attribute` is free text; no enumerated fact-type list exists anywhere in the schema. The same pipeline handles logistics revenue lines and central-bank GDP forecasts with zero configuration difference.
- **Incremental ingestion:** fact IDs are content hashes of `(file, page, offset, entity, attribute, value, unit, period)`. Re-ingesting an already-processed document produces identical IDs — a natural no-op — without any explicit "already seen this" bookkeeping.

---

## Limitations and Next Steps

Ranked roughly by how much they'd change the system if fixed next:
 
1. **Attribute-wording generalization needs semantic understanding, not more regex.** The blocking-vs-fuzzy-matching trade-off above is the clearest evidence: some attribute-name variations are safe to treat as identical, some are not, and no purely structural (token-overlap) rule reliably tells them apart. This is exactly the class of problem an LLM-based similarity check would solve well — recommended as the highest-value next addition, specifically scoped to *attribute matching*, not full extraction.
2. **Table/statement hierarchy is not tracked.** A fact only carries its row label, not the section or statement it came from — RBI's Centre/State/Combined fiscal tables and Delhivery's standalone/consolidated statements both hit this, producing same-labeled facts at very different scales (Delhivery's "Total Income (I)" standalone vs consolidated: ₹2,927.52M vs ₹85,942.34M) that look like a ~97% contradiction but aren't. This was the single largest source of false-looking large gaps in the whole corpus. Fix: an indentation- or heading-based hierarchy detector at extraction time, feeding a qualifier the reconciler can require before comparing two same-labeled facts.
3. **Categorical/state facts are out of scope for the deterministic path.** The assignment's own example — "a director may appear active in one document and resigned in a later one" — requires extracting a state change from prose with no accompanying number, which the numeric-pattern extractors cannot do by design. The `LLMFactExtractor` is built and tested (with a mock transport confirming its grounding gate correctly rejects hallucinations) but was not run against a live API for this submission.
4. **A subject/entity is one label per document, but some documents mix subjects.** RBI's own Annual Report genuinely discusses both the institution itself (balance sheet, policy actions) and the broader Indian economy (GDP, inflation) in the same document. A targeted fix re-attributes known macroeconomic-indicator facts to a separately-inferred "country context" rather than the document's primary institutional subject — this closed the gap for GDP/inflation/CAD-style facts specifically, but is not a general solution to mixed-subject documents.
5. **The unit-recognition vocabulary is incomplete.** It has no notion of physical/energy units or ratio phrases — in the RBI Annual Report, "1.6 times the norm" got extracted as `1.6%`, and a capacity stated in gigawatts got extracted as `(thousand)`, because a value with no recognizable inline unit falls back to whatever unit is dominant elsewhere on the page. Fix: extend the vocabulary with an energy/power dimension and ratio phrases, and stop the page-context fallback from applying when the surrounding unit's dimension is implausible for the attribute.
6. **Non-time-series tables are skipped by design.** `detect_table_header` requires recognizable period labels (`FY21`, `FY22`) before trusting a row enough to bind a value to it — a deliberate precision choice, but it means a genuinely one-off disclosure (the prospectus's capital-structure/offer-size table, for one) gets skipped entirely, since it has nothing for the header detector to anchor on. Fix: extend the schema to support a genuine "point fact" with no temporal scope, which is a schema change, not a parsing fix.
7. **No persistent storage layer.** Each run is stateless — re-uploading the same PDFs re-parses from scratch beyond the Streamlit session cache. The content-hash fact IDs make this safe to layer a cache or database on top of without redesign; it just hasn't been built yet.

---

## Additional Notes
 
- **Testing philosophy:** every fix in this codebase was verified against the *actual* corpus, not synthetic unit tests alone (though those exist too — smoke tests, edge-case tests, 11 classifier boundary tests, and an LLM-mock grounding-gate test). Several bugs were only found because a fix for one problem was stress-tested against a larger, different combination of real documents than the one it was built for.
- **Anchor integrity was checked after every single change** to any of the five core modules, across all six documents, without exception — 100% byte-exact resolution held throughout development, not just at submission time.
- Full credentials/API keys are not required to run or evaluate this submission — the deterministic pipeline (which produces all four demonstrated cases) needs no external service.
 