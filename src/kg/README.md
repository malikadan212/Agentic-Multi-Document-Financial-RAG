# Knowledge Graph Module

A co-occurrence based **financial entity knowledge graph** that augments
vector retrieval to improve answer quality and confidence.

## What it is (and what it isn't)

This is a *pragmatic* KG, designed to give measurable retrieval improvements
within a course-project timeline:

- **Co-occurrence based** — nodes are entities; an edge between two entities
  means they appeared together in at least one chunk. Edge weight = number of
  co-occurrences. The graph is undirected.
- **Not** a typed-relation ontology. We do **not** invent labels like
  `revenue_of` or `applies_to` because the source documents do not mark them
  and any inferred labels would be noisy. Co-occurrence is a defensible
  primitive that genuinely helps recall.
- **In-memory NetworkX graph**, persisted to `chunk_metadata/kg.pkl`. No
  Neo4j, no triplestore — the corpus (~27k chunks, ~tens of thousands of
  unique entities) fits comfortably in RAM.

## Entity types

| Type | Source | Examples |
|---|---|---|
| `BANK` | chunk metadata `bank_name` | HBL, MCB, UBL |
| `DOC_TYPE` | metadata `document_type` + body regex | KFS, SOBC, T&Cs, MITC |
| `PRODUCT` | metadata `product_type` + body regex | credit card, car loan, savings account |
| `SECTION` | metadata `section` | "Schedule of Charges", "Eligibility" |
| `MONEY` | regex on body text | Rs. 5,000, PKR 50,000, $100, 1,500 rupees |
| `PERCENT` | regex on body text | 25.99%, 15.5% APR |
| `FEE` | regex (curated phrase list) | annual fee, late payment fee, processing fee |
| `ORG` | spaCy NER (`en_core_web_sm`) | Visa, Mastercard, State Bank |

Three independent extraction layers — metadata (free, instant), regex (fast,
domain-tuned), and spaCy (slower, optional). Each layer is gracefully optional:
the graph still builds without spaCy, just with lower `ORG` coverage.

## How it improves retrieval

`KGAwareRetriever` wraps any base retriever (vector or temporal-aware) and
applies three mechanics in order:

1. **Expansion** — extract entities from the query, look up their strongest
   KG neighbors, fetch chunks where the (query_entity, neighbor) pair
   co-occurred. These are *related* chunks the vector retriever often misses.
2. **Candidate gather** — combine over-fetched vector results with expanded
   chunks into a single candidate set.
3. **Rerank** — score each candidate as
   `final = base_score + α·entity_overlap + β·pair_overlap`
   where `entity_overlap` is the fraction of query entities present in the
   chunk and `pair_overlap` is the fraction of query-entity pairs that
   co-occur in the chunk. Defaults: α = 0.30, β = 0.20.

## How it boosts confidence

The Streamlit app's `calculate_confidence_score()` accepts a `kg_coverage`
argument — the fraction of query entities present in the retrieved chunks.
When KG retrieval is enabled, this contributes up to **+15 points** on the
0–100 confidence scale (capped at 100). When KG is disabled, the score is
unchanged.

## Build the graph

From the repo root:

```bash
# Inside the container (recommended)
docker compose run --rm rag-app python scripts/build_kg.py

# Or on the host (requires `pip install networkx spacy && python -m spacy download en_core_web_sm`)
python scripts/build_kg.py
```

The build is idempotent — re-running overwrites `chunk_metadata/kg.pkl`.

Useful flags:

```bash
python scripts/build_kg.py --no-spacy       # skip spaCy NER (faster, lower ORG coverage)
python scripts/build_kg.py --limit 1000     # smoke test on first 1000 chunks
python scripts/build_kg.py --output some/path.pkl
```

Approximate timing on the 27,283-chunk corpus:
- Metadata + regex layers only: ~30 seconds
- All layers including spaCy: ~12-15 minutes (single-process CPU)

## Use it from code

```python
from kg import (
    FinancialEntityExtractor,
    FinancialKnowledgeGraph,
    KGAwareRetriever,
)

# Load the persisted graph
kg = FinancialKnowledgeGraph.load("chunk_metadata/kg.pkl")

# Lightweight extractor for query-time use (no spaCy needed for short queries)
extractor = FinancialEntityExtractor(use_spacy=False)

# Wrap any retriever
kg_retriever = KGAwareRetriever(
    base_retriever=my_vector_retriever,
    kg=kg,
    extractor=extractor,
)

results = kg_retriever.retrieve("What is HBL credit card late fee?", top_k=5)
print(kg_retriever.last_signals)  # diagnostics: query entities, expansion hits, etc.
print(kg_retriever.query_entity_coverage(results))  # 0..1, drives the confidence bonus
```

## Use it from the UI

1. Build the KG once (see above).
2. Start the app: `docker compose up`.
3. In the sidebar, toggle **Enable Knowledge Graph Retrieval** (only shown when
   `kg.pkl` exists).
4. Ask multi-entity questions — the confidence card will show a higher score
   when KG entities are matched in retrieved chunks.

## Limitations (be honest in the paper)

- **Co-occurrence ≠ semantic relation.** Two entities co-occurring frequently
  often *implies* a relationship, but the relationship type is not asserted.
  Statements like "X causes Y" or "X is a subtype of Y" cannot be answered by
  this graph alone — they require the LLM to infer from retrieved text.
- **Domain regex is curated**, not learned. A new fee phrase in unseen documents
  will not be captured until the regex list is updated.
- **No cross-doc entity resolution.** "HBL" and "Habib Bank Limited" are
  treated as separate `BANK` nodes unless the metadata normalizes them. A
  future enhancement would add an alias table or fuzzy-match resolver.
- **In-memory only.** Suitable for ~100k entities, ~1M edges. For a larger
  corpus, swap the NetworkX backend for a graph database without changing
  callers.

## Testing it improves retrieval

`KGAwareRetriever.retrieve()` populates `last_signals` with diagnostics:
the extracted query entities, expanded neighbor entities, per-chunk entity
overlap, and per-chunk query-pair co-occurrence count. You can use these in
ablation experiments to compare:

- Vector-only retrieval vs vector + KG (expansion only)
- Vector + KG (expansion + rerank) — the default
- Vector + temporal + KG (full stack)

The `scripts/run_eval.py` runner (in progress) will use the existing
`ComprehensiveEvaluator` to compute per-config Recall@K, MRR, citation
precision/recall, and BERTScore deltas so the paper's experiments section
has hard numbers, not assertions.
