"""
KG-aware retriever — wraps a base (vector) retriever and uses the knowledge
graph to improve recall and rerank.

How it improves answers
-----------------------
Three mechanics, applied in order:

1. EXPANSION
   Extract entities from the query. For each query entity, pull the strongest
   KG neighbors (entities that frequently co-occur). The neighbor entities tell
   us which *related* chunks the vector retriever might be missing.

2. CANDIDATE GATHER
   Combine:
   - Top-K from the base vector retriever (semantic match), and
   - Chunks containing any (query_entity, neighbor_entity) edge — these are
     chunks that contain *relationships* between the query topic and related
     concepts.

3. RERANK
   Score each candidate by:
       final = base_score + α * entity_overlap + β * pair_overlap
   where:
     entity_overlap = (# query entities present in chunk) / (# query entities)
     pair_overlap   = (# query-entity pairs that co-occur in chunk) / (# pairs)

   Then return top-K by final score.

The base retriever's API is preserved — same `retrieve(query, top_k)` signature,
same return shape — so this slots in transparently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

from .entity_extractor import FinancialEntityExtractor
from .financial_entity import EntityType, FinancialEntity
from .graph_builder import FinancialKnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class KGSignals:
    """Diagnostic info attached to retrieval results when KG is active."""

    query_entities: list[FinancialEntity]
    expanded_neighbors: list[tuple[str, str]]  # (canonical, type)
    entities_per_chunk: dict[str, set[tuple[str, str]]]  # chunk_id -> entity keys
    pair_hits_per_chunk: dict[str, int]  # chunk_id -> # query pairs co-occurring


class KGAwareRetriever:
    """
    Retriever that wraps a base retriever and uses a FinancialKnowledgeGraph
    to expand candidates and rerank by entity overlap.
    """

    def __init__(
        self,
        base_retriever: Any,
        kg: FinancialKnowledgeGraph,
        extractor: FinancialEntityExtractor,
        *,
        chunk_loader: Any = None,
        expansion_neighbors: int = 5,
        expansion_min_weight: int = 2,
        max_expanded_chunks: int = 50,
        alpha_entity_overlap: float = 0.30,
        beta_pair_overlap: float = 0.20,
    ):
        """
        Args:
            base_retriever: Object exposing `retrieve(query, top_k)` returning a list
                of dicts (PreloadedRetriever) or RetrievalResult-like objects. May
                itself be a wrapper (e.g. TemporalAwareRetriever).
            kg: Built FinancialKnowledgeGraph.
            extractor: FinancialEntityExtractor for query-time entity extraction.
            chunk_loader: Optional separate object used to resolve chunk_id → full
                chunk during expansion. Should expose either `get_chunk_by_id(cid)`
                or a `.loader.chunks` list (e.g. PreloadedRetriever). Defaults to
                `base_retriever` when not provided.
            expansion_neighbors: per query entity, how many KG neighbors to pull.
            expansion_min_weight: skip KG edges weaker than this (low-signal pairs).
            max_expanded_chunks: cap on extra chunks added via expansion (avoid
                blow-up on common entities like "credit card").
            alpha_entity_overlap: rerank weight for fraction of query entities in chunk.
            beta_pair_overlap: rerank weight for fraction of query-entity pairs that
                co-occur in chunk.
        """
        self.base_retriever = base_retriever
        self.chunk_loader = chunk_loader if chunk_loader is not None else base_retriever
        self.kg = kg
        self.extractor = extractor
        self.expansion_neighbors = expansion_neighbors
        self.expansion_min_weight = expansion_min_weight
        self.max_expanded_chunks = max_expanded_chunks
        self.alpha_entity_overlap = alpha_entity_overlap
        self.beta_pair_overlap = beta_pair_overlap
        self.last_signals: KGSignals | None = None

    # ------------------------------------------------------------------
    # Public retrieve API — mirrors PreloadedRetriever / HybridRetriever shape
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> list[dict]:
        # Step 1 — base vector retrieval (always run; semantic recall floor)
        # We over-fetch because we'll merge & rerank with KG-expanded candidates.
        over_fetch = max(top_k * 4, 20)
        base_results = self.base_retriever.retrieve(query, top_k=over_fetch, **kwargs)
        base_results = list(self._normalize_results(base_results))

        # Step 2 — extract entities from the query
        query_entities = self.extractor.extract_from_query(query)

        # Bail out fast: no entities → behave exactly like the base retriever
        if not query_entities:
            self.last_signals = KGSignals(
                query_entities=[],
                expanded_neighbors=[],
                entities_per_chunk={},
                pair_hits_per_chunk={},
            )
            return base_results[:top_k]

        # Step 3 — expansion: gather extra candidate chunks via the graph
        expanded_neighbors = self._collect_neighbors(query_entities)
        expanded_chunk_ids = self._collect_expanded_chunks(
            query_entities, expanded_neighbors
        )

        # Step 4 — fetch the expanded chunks from the underlying loader
        expanded_results = self._materialize_chunks(
            expanded_chunk_ids - {r["chunk_id"] for r in base_results}
        )

        # Step 5 — rerank
        candidates = base_results + expanded_results
        entities_per_chunk = self._compute_chunk_entities(candidates, query_entities)
        pair_hits_per_chunk = self._compute_pair_hits(
            candidates, query_entities, entities_per_chunk
        )

        for cand in candidates:
            cid = cand["chunk_id"]
            qe_present = entities_per_chunk.get(cid, set())
            entity_overlap = len(qe_present) / max(1, len(query_entities))
            n_pairs = max(1, len(query_entities) * (len(query_entities) - 1) // 2)
            pair_overlap = pair_hits_per_chunk.get(cid, 0) / n_pairs
            base_score = float(cand.get("score", 0.0) or 0.0)
            cand["score_base"] = base_score
            cand["score_kg_entity_overlap"] = entity_overlap
            cand["score_kg_pair_overlap"] = pair_overlap
            cand["score"] = (
                base_score
                + self.alpha_entity_overlap * entity_overlap
                + self.beta_pair_overlap * pair_overlap
            )

        candidates.sort(key=lambda c: c["score"], reverse=True)
        # Renumber ranks after rerank
        for rank, cand in enumerate(candidates[:top_k], start=1):
            cand["rank"] = rank

        self.last_signals = KGSignals(
            query_entities=query_entities,
            expanded_neighbors=expanded_neighbors,
            entities_per_chunk=entities_per_chunk,
            pair_hits_per_chunk=pair_hits_per_chunk,
        )

        logger.info(
            "🧭 KG retrieval: %d query entities → %d neighbors → +%d expanded chunks → "
            "reranked %d candidates → top %d",
            len(query_entities),
            len(expanded_neighbors),
            len(expanded_results),
            len(candidates),
            min(top_k, len(candidates)),
        )

        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Confidence helper — used by the UI to compute the KG bonus.
    # ------------------------------------------------------------------
    def query_entity_coverage(self, retrieved_results: Iterable[dict]) -> float:
        """
        Fraction of the last query's entities that appear in at least one of the
        provided retrieved results. Returns 0.0 if no last query was processed
        or the query had no entities.
        """
        if not self.last_signals or not self.last_signals.query_entities:
            return 0.0
        present_entities: set[tuple[str, str]] = set()
        for r in retrieved_results:
            cid = r.get("chunk_id") if isinstance(r, dict) else getattr(r, "chunk_id", None)
            if not cid:
                continue
            present_entities.update(self.kg.entities_in_chunk(cid))
        query_keys = {e.key for e in self.last_signals.query_entities}
        if not query_keys:
            return 0.0
        return len(query_keys & present_entities) / len(query_keys)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_neighbors(
        self, query_entities: list[FinancialEntity]
    ) -> list[tuple[str, str]]:
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for ent in query_entities:
            for (nbr_text, nbr_type), _w in self.kg.neighbors(
                ent.canonical,
                ent.entity_type,
                top_k=self.expansion_neighbors,
                min_weight=self.expansion_min_weight,
            ):
                key = (nbr_text, nbr_type)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
        return out

    def _collect_expanded_chunks(
        self,
        query_entities: list[FinancialEntity],
        neighbors: list[tuple[str, str]],
    ) -> set[str]:
        chunk_ids: set[str] = set()
        for q in query_entities:
            for nbr_text, nbr_type in neighbors:
                # The (query_entity, neighbor) edge is what identifies *related*
                # chunks. We use chunks_for_pair to fetch only the strong-signal
                # chunks (where both co-occur), not the full neighbor's chunk set.
                try:
                    nbr_etype = EntityType(nbr_type)
                except ValueError:
                    continue
                pair_chunks = self.kg.chunks_for_pair(
                    (q.canonical, q.entity_type),
                    (nbr_text, nbr_etype),
                )
                chunk_ids.update(pair_chunks)
                if len(chunk_ids) >= self.max_expanded_chunks:
                    return set(list(chunk_ids)[: self.max_expanded_chunks])
        return chunk_ids

    def _materialize_chunks(self, chunk_ids: set[str]) -> list[dict]:
        """
        Resolve chunk_ids to full result dicts via the base retriever's loader.

        We call `get_chunk_by_id` if the base retriever exposes it (PreloadedRetriever
        does); otherwise we scan its loader.chunks. If neither exists, expansion is
        a no-op (we still rerank base results by entity overlap).
        """
        if not chunk_ids:
            return []
        out: list[dict] = []
        get_chunk = getattr(self.chunk_loader, "get_chunk_by_id", None)
        if callable(get_chunk):
            for cid in chunk_ids:
                chunk = get_chunk(cid)
                if not chunk:
                    continue
                out.append(self._chunk_to_result_dict(chunk))
        else:
            loader = getattr(self.chunk_loader, "loader", None)
            chunks = getattr(loader, "chunks", []) if loader else []
            id_set = chunk_ids
            for chunk in chunks:
                cid = chunk.get("chunk_id") if isinstance(chunk, dict) else getattr(
                    chunk, "chunk_id", None
                )
                if cid in id_set:
                    out.append(self._chunk_to_result_dict(chunk))
        return out

    @staticmethod
    def _chunk_to_result_dict(chunk: Any) -> dict:
        if isinstance(chunk, dict):
            return {
                "chunk_id": chunk.get("chunk_id"),
                "content": chunk.get("content", ""),
                "score": 0.0,  # KG-only candidate, no vector score yet
                "metadata": chunk.get("metadata", {}),
                "rank": 0,
            }
        return {
            "chunk_id": getattr(chunk, "chunk_id", None),
            "content": getattr(chunk, "content", ""),
            "score": 0.0,
            "metadata": getattr(chunk, "metadata", {}),
            "rank": 0,
        }

    @staticmethod
    def _normalize_results(results: Iterable[Any]) -> Iterable[dict]:
        """Convert any RetrievalResult-likes into plain dicts so the rerank loop
        can mutate them safely."""
        for r in results:
            if isinstance(r, dict):
                yield dict(r)  # shallow copy
            else:
                yield {
                    "chunk_id": getattr(r, "chunk_id", None),
                    "content": getattr(r, "content", ""),
                    "score": getattr(r, "score", 0.0),
                    "metadata": getattr(r, "metadata", {}),
                    "rank": getattr(r, "rank", 0),
                }

    def _compute_chunk_entities(
        self,
        candidates: list[dict],
        query_entities: list[FinancialEntity],
    ) -> dict[str, set[tuple[str, str]]]:
        query_keys = {e.key for e in query_entities}
        out: dict[str, set[tuple[str, str]]] = {}
        for cand in candidates:
            cid = cand["chunk_id"]
            chunk_keys = self.kg.entities_in_chunk(cid)
            out[cid] = chunk_keys & query_keys
        return out

    def _compute_pair_hits(
        self,
        candidates: list[dict],
        query_entities: list[FinancialEntity],
        entities_per_chunk: dict[str, set[tuple[str, str]]],
    ) -> dict[str, int]:
        if len(query_entities) < 2:
            return {c["chunk_id"]: 0 for c in candidates}
        out: dict[str, int] = {}
        for cand in candidates:
            cid = cand["chunk_id"]
            present = entities_per_chunk.get(cid, set())
            hits = sum(1 for a, b in combinations(present, 2) if a != b)
            out[cid] = hits
        return out
