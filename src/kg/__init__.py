"""
Knowledge Graph module for the Agentic Multi-Document Financial RAG system.

Provides a co-occurrence based financial-entity knowledge graph that augments
vector retrieval with:

- Entity expansion (query → related entities → additional candidate chunks)
- Co-occurrence reranking (chunks where multiple query entities co-occur are boosted)
- Confidence signal (% of query entities found in retrieved chunks)

Public API:

    from kg import (
        FinancialEntity,
        EntityType,
        FinancialEntityExtractor,
        FinancialKnowledgeGraph,
        KGAwareRetriever,
    )

The graph is intentionally simple — undirected NetworkX with weighted edges and
chunk-id sets per edge — so it's fast to build, persist, and reason over for the
27,283-chunk banking corpus.
"""

from .financial_entity import EntityType, FinancialEntity
from .entity_extractor import FinancialEntityExtractor
from .graph_builder import FinancialKnowledgeGraph
from .kg_retriever import KGAwareRetriever

__all__ = [
    "EntityType",
    "FinancialEntity",
    "FinancialEntityExtractor",
    "FinancialKnowledgeGraph",
    "KGAwareRetriever",
]
