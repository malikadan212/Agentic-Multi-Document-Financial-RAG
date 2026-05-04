# src/temporal/temporal_scorer.py
"""
Temporal Scoring for Retrieval
Adjusts relevance scores based on temporal factors
"""

from typing import List, Optional, Tuple
from datetime import datetime
import math
import logging

logger = logging.getLogger(__name__)


class TemporalScorer:
    """
    Scores and re-ranks retrieved chunks based on temporal relevance
    """
    
    def __init__(
        self,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.7,
        decay_rate: float = 0.1
    ):
        """
        Initialize temporal scorer
        
        Args:
            recency_weight: Weight for recency score (0-1)
            relevance_weight: Weight for semantic relevance (0-1)
            decay_rate: Exponential decay rate for older documents
        """
        self.recency_weight = recency_weight
        self.relevance_weight = relevance_weight
        self.decay_rate = decay_rate
        
        # Ensure weights sum to 1
        total = recency_weight + relevance_weight
        if total != 1.0:
            self.recency_weight = recency_weight / total
            self.relevance_weight = relevance_weight / total
    
    def score_with_temporal_relevance(
        self,
        chunks: List,
        temporal_intent,
        reference_date: Optional[str] = None
    ) -> List:
        """
        Re-score chunks based on temporal relevance
        
        Args:
            chunks: List of retrieved chunks (with scores)
            temporal_intent: TemporalIntent from query parser
            reference_date: Reference date for recency calculation (ISO format)
            
        Returns:
            Re-scored and re-ranked chunks
        """
        if not temporal_intent.has_temporal_intent:
            return chunks
        
        reference_date = reference_date or datetime.now().strftime('%Y-%m-%d')
        
        scored_chunks = []
        for chunk in chunks:
            # Get original semantic score
            semantic_score = getattr(chunk, 'score', 0.5)
            
            # Calculate temporal score
            temporal_score = self._calculate_temporal_score(
                chunk, temporal_intent, reference_date
            )
            
            # Combine scores
            final_score = (
                self.relevance_weight * semantic_score +
                self.recency_weight * temporal_score
            )
            
            # Update chunk score
            chunk.score = final_score
            chunk.metadata['temporal_score'] = temporal_score
            chunk.metadata['original_score'] = semantic_score
            
            scored_chunks.append(chunk)
        
        # Re-rank by final score
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for rank, chunk in enumerate(scored_chunks, start=1):
            chunk.rank = rank
        
        return scored_chunks
    
    def _calculate_temporal_score(
        self,
        chunk,
        temporal_intent,
        reference_date: str
    ) -> float:
        """
        Calculate temporal relevance score for a chunk
        
        Returns score between 0 and 1
        """
        # If chunk has no temporal info, return neutral score
        chunk_start = None
        chunk_end = None
        
        # Try to get temporal validity from chunk
        if hasattr(chunk, 'valid_from'):
            chunk_start = chunk.valid_from
            chunk_end = chunk.valid_to or chunk.valid_from
        elif hasattr(chunk, 'metadata'):
            # Try to get from metadata
            metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
            chunk_start = metadata.get('valid_from')
            chunk_end = metadata.get('valid_to', chunk_start)
        
        if not chunk_start:
            return 0.5  # Neutral score for chunks without temporal info
        
        # Check if chunk's validity period matches query intent
        if temporal_intent.time_range:
            query_start, query_end = temporal_intent.time_range
            
            # Check for overlap
            overlap = self._calculate_overlap(
                (chunk_start, chunk_end),
                (query_start, query_end)
            )
            
            if overlap > 0:
                # Perfect match or overlap
                return 1.0
            else:
                # No overlap, calculate proximity
                proximity = self._calculate_proximity(
                    (chunk_start, chunk_end),
                    (query_start, query_end)
                )
                return max(0.3, 1.0 - proximity)  # Min score 0.3
        
        # Recency-based scoring
        if temporal_intent.recency_preference == 'recent':
            return self._calculate_recency_score(chunk_start, reference_date)
        elif temporal_intent.recency_preference == 'historical':
            # Prefer older documents
            recency = self._calculate_recency_score(chunk_start, reference_date)
            return 1.0 - recency
        
        # Default: neutral score
        return 0.5
    
    def _calculate_overlap(
        self,
        range1: Tuple[str, str],
        range2: Tuple[str, str]
    ) -> float:
        """
        Calculate overlap between two date ranges
        Returns overlap ratio (0-1)
        """
        start1, end1 = range1
        start2, end2 = range2
        
        # Convert to datetime for comparison
        try:
            s1 = datetime.fromisoformat(start1)
            e1 = datetime.fromisoformat(end1)
            s2 = datetime.fromisoformat(start2)
            e2 = datetime.fromisoformat(end2)
        except:
            return 0.0
        
        # Calculate overlap
        overlap_start = max(s1, s2)
        overlap_end = min(e1, e2)
        
        if overlap_start > overlap_end:
            return 0.0  # No overlap
        
        overlap_days = (overlap_end - overlap_start).days
        range2_days = (e2 - s2).days
        
        if range2_days == 0:
            return 1.0 if overlap_days >= 0 else 0.0
        
        return min(1.0, overlap_days / range2_days)
    
    def _calculate_proximity(
        self,
        range1: Tuple[str, str],
        range2: Tuple[str, str]
    ) -> float:
        """
        Calculate temporal proximity between two ranges
        Returns normalized distance (0-1)
        """
        start1, end1 = range1
        start2, end2 = range2
        
        try:
            s1 = datetime.fromisoformat(start1)
            e1 = datetime.fromisoformat(end1)
            s2 = datetime.fromisoformat(start2)
            e2 = datetime.fromisoformat(end2)
        except:
            return 1.0  # Max distance
        
        # Calculate gap between ranges
        if e1 < s2:
            gap_days = (s2 - e1).days
        elif e2 < s1:
            gap_days = (s1 - e2).days
        else:
            return 0.0  # Ranges overlap
        
        # Normalize gap (1 year = 365 days)
        normalized_gap = min(1.0, gap_days / 365.0)
        
        return normalized_gap
    
    def _calculate_recency_score(
        self,
        date: str,
        reference_date: str
    ) -> float:
        """
        Calculate recency score using exponential decay
        More recent = higher score
        """
        try:
            doc_date = datetime.fromisoformat(date)
            ref_date = datetime.fromisoformat(reference_date)
        except:
            return 0.5
        
        # Calculate age in days
        age_days = (ref_date - doc_date).days
        
        if age_days < 0:
            # Future date, give high score
            return 1.0
        
        # Exponential decay: score = e^(-decay_rate * age_years)
        age_years = age_days / 365.0
        score = math.exp(-self.decay_rate * age_years)
        
        return max(0.1, min(1.0, score))  # Clamp between 0.1 and 1.0
    
    def filter_by_date_range(
        self,
        chunks: List,
        start_date: str,
        end_date: str,
        strict: bool = False
    ) -> List:
        """
        Filter chunks by date range
        
        Args:
            chunks: List of chunks
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            strict: If True, only include chunks fully within range
            
        Returns:
            Filtered chunks
        """
        filtered = []
        
        for chunk in chunks:
            if not hasattr(chunk, 'valid_from') or not chunk.valid_from:
                # No temporal info, include if not strict
                if not strict:
                    filtered.append(chunk)
                continue
            
            chunk_start = chunk.valid_from
            chunk_end = chunk.valid_to or chunk.valid_from
            
            if strict:
                # Chunk must be fully within range
                if chunk_start >= start_date and chunk_end <= end_date:
                    filtered.append(chunk)
            else:
                # Any overlap is acceptable
                if chunk_start <= end_date and chunk_end >= start_date:
                    filtered.append(chunk)
        
        return filtered
