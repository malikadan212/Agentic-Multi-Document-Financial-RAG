"""
Financial knowledge graph backed by NetworkX.

Schema
------
- Nodes are (canonical_text, entity_type) tuples. Storing the type in the key
  prevents collisions like "credit card" the PRODUCT vs. "credit card" the
  literal string in some other slot.
- Node attributes:
    text:  canonical surface form
    type:  EntityType.value
    count: number of distinct chunks the entity appears in
- Edges are undirected and represent co-occurrence within the same chunk.
- Edge attributes:
    weight:  number of chunks where both entities co-occur
    chunks:  set of chunk_ids where they co-occur

Why undirected? Co-occurrence is symmetric. We're not asserting causal or
typed relations ("revenue_of", "applies_to") because we'd be inventing them.
A co-occurrence graph is a defensible primitive that genuinely helps retrieval.
"""

from __future__ import annotations

import logging
import pickle
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

try:
    import networkx as nx

    _NX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NX_AVAILABLE = False

from .financial_entity import EntityType, FinancialEntity

logger = logging.getLogger(__name__)


# Edges from cliques bigger than this are skipped at build time. A handful of
# very-large chunks would otherwise contribute O(n^2) low-signal edges.
_MAX_CLIQUE_SIZE = 25


class FinancialKnowledgeGraph:
    """In-memory co-occurrence graph over financial entities."""

    SCHEMA_VERSION = 1

    def __init__(self):
        if not _NX_AVAILABLE:
            raise ImportError(
                "networkx is required for the KG. Install with: pip install networkx"
            )
        self.graph = nx.Graph()
        # Reverse index: chunk_id -> set of node keys, for fast retrieval-time
        # entity lookups ("which entities are in chunk X?")
        self._chunk_index: dict[str, set[tuple[str, str]]] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def add_chunk(self, chunk_id: str, entities: Iterable[FinancialEntity]) -> None:
        """Register a chunk's entities into the graph."""
        ents = list({e.key: e for e in entities}.values())  # dedupe by key
        if not ents:
            return

        node_keys: set[tuple[str, str]] = set()
        for ent in ents:
            key = ent.key
            node_keys.add(key)
            if self.graph.has_node(key):
                self.graph.nodes[key]["count"] += 1
            else:
                self.graph.add_node(
                    key,
                    text=ent.canonical,
                    type=ent.entity_type.value,
                    count=1,
                )

        self._chunk_index[chunk_id] = node_keys

        # Add co-occurrence edges. Skip absurdly large cliques.
        if len(ents) > _MAX_CLIQUE_SIZE:
            logger.debug(
                "Chunk %s has %d entities; skipping edge expansion (>= %d).",
                chunk_id,
                len(ents),
                _MAX_CLIQUE_SIZE,
            )
            return

        for a, b in combinations(sorted(node_keys), 2):
            if self.graph.has_edge(a, b):
                edata = self.graph[a][b]
                edata["weight"] += 1
                edata["chunks"].add(chunk_id)
            else:
                self.graph.add_edge(a, b, weight=1, chunks={chunk_id})

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def has_node(self, canonical: str, entity_type: EntityType) -> bool:
        return self.graph.has_node((canonical, entity_type.value))

    def neighbors(
        self,
        canonical: str,
        entity_type: EntityType,
        top_k: int = 10,
        min_weight: int = 1,
    ) -> list[tuple[tuple[str, str], int]]:
        """
        Return up to top_k strongest neighbors of (canonical, type), sorted by
        co-occurrence weight descending. Each result is ((text, type), weight).
        """
        node = (canonical, entity_type.value)
        if not self.graph.has_node(node):
            return []
        scored = [
            ((nbr_text, nbr_type), self.graph[node][(nbr_text, nbr_type)]["weight"])
            for (nbr_text, nbr_type) in self.graph.neighbors(node)
            if self.graph[node][(nbr_text, nbr_type)]["weight"] >= min_weight
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def chunks_for_entity(
        self,
        canonical: str,
        entity_type: EntityType,
    ) -> set[str]:
        """All chunk_ids that mentioned this entity."""
        node = (canonical, entity_type.value)
        if not self.graph.has_node(node):
            return set()
        out: set[str] = set()
        for nbr in self.graph.neighbors(node):
            out.update(self.graph[node][nbr]["chunks"])
        # Also include chunks where the entity appears alone (no neighbors).
        # We have to scan _chunk_index for those.
        for cid, keys in self._chunk_index.items():
            if node in keys:
                out.add(cid)
        return out

    def chunks_for_pair(
        self,
        e1: tuple[str, EntityType],
        e2: tuple[str, EntityType],
    ) -> set[str]:
        """Chunk_ids where both entities co-occurred."""
        n1 = (e1[0], e1[1].value)
        n2 = (e2[0], e2[1].value)
        if not self.graph.has_edge(n1, n2):
            return set()
        return set(self.graph[n1][n2]["chunks"])

    def entities_in_chunk(self, chunk_id: str) -> set[tuple[str, str]]:
        """Entity node-keys present in a given chunk."""
        return self._chunk_index.get(chunk_id, set())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "graph": self.graph,
            "chunk_index": self._chunk_index,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "💾 KG saved to %s (%d nodes, %d edges, %d chunks indexed)",
            path,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            len(self._chunk_index),
        )

    @classmethod
    def load(cls, path: str | Path) -> Optional["FinancialKnowledgeGraph"]:
        """Load a previously persisted graph. Returns None if the file is missing."""
        path = Path(path)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            logger.warning(
                "KG schema version mismatch (%s vs %s). Rebuild recommended.",
                payload.get("schema_version"),
                cls.SCHEMA_VERSION,
            )
        kg = cls()
        kg.graph = payload["graph"]
        kg._chunk_index = payload["chunk_index"]
        logger.info(
            "📂 KG loaded from %s (%d nodes, %d edges, %d chunks indexed)",
            path,
            kg.graph.number_of_nodes(),
            kg.graph.number_of_edges(),
            len(kg._chunk_index),
        )
        return kg

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        n_by_type: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            n_by_type[data["type"]] = n_by_type.get(data["type"], 0) + 1
        edge_weights = [d["weight"] for _, _, d in self.graph.edges(data=True)]
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "chunks_indexed": len(self._chunk_index),
            "nodes_by_type": n_by_type,
            "max_edge_weight": max(edge_weights) if edge_weights else 0,
            "avg_edge_weight": (
                sum(edge_weights) / len(edge_weights) if edge_weights else 0.0
            ),
        }
