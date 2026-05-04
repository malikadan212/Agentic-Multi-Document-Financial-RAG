# src/retrieval/temporal_retriever.py
"""
Temporal-Aware Retriever
Wraps existing retrievers with temporal intelligence
"""

from typing import List, Optional, Dict, Any
import logging

from .retriever import HybridRetriever, RetrievalResult
from .preloaded_retriever import PreloadedRetriever

# Import temporal components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from temporal.temporal_query_parser import TemporalQueryParser, TemporalIntent
from temporal.temporal_scorer import TemporalScorer

logger = logging.getLogger(__name__)


class TemporalAwareRetriever:
    """
    Wraps any retriever with temporal awareness
    Adds temporal query parsing, scoring, and filtering
    """
    
    def __init__(
        self,
        base_retriever,
        enable_temporal_scoring: bool = True,
        enable_query_expansion: bool = True,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.7
    ):
        """
        Initialize temporal-aware retriever
        
        Args:
            base_retriever: Underlying retriever (HybridRetriever or PreloadedRetriever)
            enable_temporal_scoring: Whether to apply temporal scoring
            enable_query_expansion: Whether to expand queries with temporal context
            recency_weight: Weight for recency in scoring
            relevance_weight: Weight for semantic relevance in scoring
        """
        self.base_retriever = base_retriever
        self.enable_temporal_scoring = enable_temporal_scoring
        self.enable_query_expansion = enable_query_expansion
        
        # Initialize temporal components
        self.query_parser = TemporalQueryParser()
        self.temporal_scorer = TemporalScorer(
            recency_weight=recency_weight,
            relevance_weight=relevance_weight
        )
        
        logger.info("✅ Temporal-aware retriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        temporal_filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve with temporal awareness
        
        Args:
            query: User query
            top_k: Number of results to return
            temporal_filter: Optional temporal filter config
                {
                    'enabled': bool,
                    'start_date': str,
                    'end_date': str,
                    'strict': bool
                }
        
        Returns:
            List of temporally-aware retrieval results
        """
        # Parse temporal intent from query
        temporal_intent = self.query_parser.parse(query)
        
        logger.info(f"Temporal intent: {temporal_intent}")
        
        # Expand query if temporal intent detected
        expanded_query = query
        if self.enable_query_expansion and temporal_intent.has_temporal_intent:
            expanded_query = self.query_parser.expand_query_with_temporal_context(
                query, temporal_intent
            )
            if expanded_query != query:
                logger.info(f"Expanded query: {query} -> {expanded_query}")
        
        # Retrieve using base retriever (get more results for re-ranking)
        retrieve_k = top_k * 2 if self.enable_temporal_scoring else top_k
        results = self.base_retriever.retrieve(expanded_query, top_k=retrieve_k)
        
        # Convert to standard format if needed
        if isinstance(self.base_retriever, PreloadedRetriever):
            results = self._convert_preloaded_results(results)
        
        # Apply temporal filtering if specified
        if temporal_filter and temporal_filter.get('enabled'):
            start_date = temporal_filter.get('start_date')
            end_date = temporal_filter.get('end_date')
            strict = temporal_filter.get('strict', False)
            
            if start_date and end_date:
                logger.info(f"Applying temporal filter: {start_date} to {end_date}")
                results = self.temporal_scorer.filter_by_date_range(
                    results, start_date, end_date, strict=strict
                )
        
        # Apply temporal scoring if enabled
        if self.enable_temporal_scoring and temporal_intent.has_temporal_intent:
            logger.info("Applying temporal scoring")
            results = self.temporal_scorer.score_with_temporal_relevance(
                results, temporal_intent
            )
        
        # Return top-k results
        return results[:top_k]
    
    def retrieve_with_trend_analysis(
        self,
        query: str,
        time_periods: List[tuple],
        top_k: int = 5
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve results for multiple time periods (for trend analysis)
        
        Args:
            query: User query
            time_periods: List of (start_date, end_date) tuples
            top_k: Results per period
            
        Returns:
            Dictionary mapping period to results
        """
        results_by_period = {}
        
        for start_date, end_date in time_periods:
            period_key = f"{start_date}_to_{end_date}"
            
            # Retrieve with temporal filter for this period
            temporal_filter = {
                'enabled': True,
                'start_date': start_date,
                'end_date': end_date,
                'strict': False
            }
            
            period_results = self.retrieve(
                query, top_k=top_k, temporal_filter=temporal_filter
            )
            
            results_by_period[period_key] = period_results
        
        return results_by_period
    
    def _convert_preloaded_results(self, results: List) -> List[RetrievalResult]:
        """Convert PreloadedRetriever results to RetrievalResult format"""
        converted = []
        for rank, result in enumerate(results, start=1):
            # Check if already a RetrievalResult
            if isinstance(result, RetrievalResult):
                converted.append(result)
            # Otherwise convert from dict
            elif isinstance(result, dict):
                converted.append(RetrievalResult(
                    chunk_id=result.get('chunk_id', ''),
                    content=result.get('content', ''),
                    score=result.get('score', 0.0),
                    metadata=result.get('metadata', {}),
                    rank=rank
                ))
            else:
                # Unknown format, skip
                logger.warning(f"Unknown result format: {type(result)}")
        return converted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics including temporal info"""
        stats = {}
        
        # Get base retriever stats
        if hasattr(self.base_retriever, 'get_stats'):
            stats = self.base_retriever.get_stats()
        
        # Add temporal stats
        stats['temporal_enabled'] = True
        stats['temporal_scoring'] = self.enable_temporal_scoring
        stats['query_expansion'] = self.enable_query_expansion
        
        # Count temporal chunks
        temporal_count = 0
        try:
            if hasattr(self.base_retriever, 'vector_store'):
                if hasattr(self.base_retriever.vector_store, 'chunks'):
                    chunks = self.base_retriever.vector_store.chunks
                    temporal_count = sum(
                        1 for c in chunks 
                        if hasattr(c, 'temporal_entities') and c.temporal_entities
                    )
            elif hasattr(self.base_retriever, 'loader'):
                if hasattr(self.base_retriever.loader, 'chunks'):
                    chunks = self.base_retriever.loader.chunks
                    temporal_count = sum(
                        1 for c in chunks 
                        if hasattr(c, 'temporal_entities') and c.temporal_entities
                    )
        except:
            pass
        
        stats['temporal_chunks'] = temporal_count
        
        return stats


def create_temporal_retriever(
    base_retriever,
    **kwargs
) -> TemporalAwareRetriever:
    """
    Factory function to create temporal-aware retriever
    
    Args:
        base_retriever: Base retriever to wrap
        **kwargs: Additional arguments for TemporalAwareRetriever
        
    Returns:
        TemporalAwareRetriever instance
    """
    return TemporalAwareRetriever(base_retriever, **kwargs)
