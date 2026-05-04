# src/temporal/temporal_query_parser.py
"""
Temporal Query Parser
Extracts temporal intent from user queries
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging

from .temporal_entity import TemporalType
from .temporal_extractor import TemporalEntityExtractor
from .temporal_normalizer import TemporalNormalizer

logger = logging.getLogger(__name__)


class TemporalIntent:
    """Represents temporal intent extracted from a query"""
    
    def __init__(
        self,
        has_temporal_intent: bool = False,
        temporal_entities: List = None,
        comparison_type: Optional[str] = None,  # 'trend', 'comparison', 'point_in_time'
        time_range: Optional[Tuple[str, str]] = None,  # (start_date, end_date)
        recency_preference: Optional[str] = None,  # 'recent', 'historical', 'specific'
        keywords: List[str] = None
    ):
        self.has_temporal_intent = has_temporal_intent
        self.temporal_entities = temporal_entities or []
        self.comparison_type = comparison_type
        self.time_range = time_range
        self.recency_preference = recency_preference
        self.keywords = keywords or []
    
    def __repr__(self):
        return f"TemporalIntent(has_intent={self.has_temporal_intent}, type={self.comparison_type}, range={self.time_range})"


class TemporalQueryParser:
    """
    Parses user queries to extract temporal intent
    Identifies:
    - Explicit temporal mentions (Q2 2024, last year)
    - Temporal keywords (trend, change, compare)
    - Recency preferences (latest, current, recent)
    """
    
    # Temporal keywords by category
    TREND_KEYWORDS = [
        'trend', 'change', 'evolution', 'growth', 'decline', 'increase', 'decrease',
        'over time', 'progression', 'development', 'trajectory'
    ]
    
    COMPARISON_KEYWORDS = [
        'compare', 'comparison', 'versus', 'vs', 'difference', 'contrast',
        'year-over-year', 'yoy', 'quarter-over-quarter', 'qoq'
    ]
    
    RECENCY_KEYWORDS = [
        'latest', 'recent', 'current', 'now', 'today', 'this year', 'this quarter',
        'most recent', 'up-to-date', 'newest'
    ]
    
    HISTORICAL_KEYWORDS = [
        'historical', 'past', 'previous', 'old', 'former', 'earlier', 'before'
    ]
    
    def __init__(self):
        self.extractor = TemporalEntityExtractor(use_spacy=False)
        self.normalizer = TemporalNormalizer()
    
    def parse(self, query: str) -> TemporalIntent:
        """
        Parse query to extract temporal intent
        
        Args:
            query: User query string
            
        Returns:
            TemporalIntent object
        """
        query_lower = query.lower()
        
        # Extract temporal entities from query
        temporal_entities = self.extractor.extract_from_text(query)
        
        # Detect temporal keywords
        has_trend = any(kw in query_lower for kw in self.TREND_KEYWORDS)
        has_comparison = any(kw in query_lower for kw in self.COMPARISON_KEYWORDS)
        has_recency = any(kw in query_lower for kw in self.RECENCY_KEYWORDS)
        has_historical = any(kw in query_lower for kw in self.HISTORICAL_KEYWORDS)
        
        # Determine if query has temporal intent
        has_temporal_intent = (
            len(temporal_entities) > 0 or
            has_trend or
            has_comparison or
            has_recency or
            has_historical
        )
        
        if not has_temporal_intent:
            return TemporalIntent(has_temporal_intent=False)
        
        # Determine comparison type
        comparison_type = None
        if has_trend:
            comparison_type = 'trend'
        elif has_comparison:
            comparison_type = 'comparison'
        elif len(temporal_entities) > 0:
            comparison_type = 'point_in_time'
        
        # Determine recency preference
        recency_preference = None
        if has_recency:
            recency_preference = 'recent'
        elif has_historical:
            recency_preference = 'historical'
        elif len(temporal_entities) > 0:
            recency_preference = 'specific'
        
        # Extract time range from entities
        time_range = self._extract_time_range(temporal_entities, query_lower)
        
        # Extract relevant keywords
        keywords = []
        if has_trend:
            keywords.extend([kw for kw in self.TREND_KEYWORDS if kw in query_lower])
        if has_comparison:
            keywords.extend([kw for kw in self.COMPARISON_KEYWORDS if kw in query_lower])
        if has_recency:
            keywords.extend([kw for kw in self.RECENCY_KEYWORDS if kw in query_lower])
        
        return TemporalIntent(
            has_temporal_intent=True,
            temporal_entities=temporal_entities,
            comparison_type=comparison_type,
            time_range=time_range,
            recency_preference=recency_preference,
            keywords=keywords
        )
    
    def _extract_time_range(
        self, 
        entities: List, 
        query_lower: str
    ) -> Optional[Tuple[str, str]]:
        """Extract time range from temporal entities"""
        if not entities:
            # Check for relative time expressions
            if 'last year' in query_lower or 'previous year' in query_lower:
                last_year = datetime.now().year - 1
                return (f"{last_year}-01-01", f"{last_year}-12-31")
            elif 'this year' in query_lower or 'current year' in query_lower:
                this_year = datetime.now().year
                return (f"{this_year}-01-01", f"{this_year}-12-31")
            elif 'last quarter' in query_lower:
                # Calculate last quarter
                now = datetime.now()
                current_quarter = (now.month - 1) // 3 + 1
                if current_quarter == 1:
                    last_q_start = datetime(now.year - 1, 10, 1)
                    last_q_end = datetime(now.year - 1, 12, 31)
                else:
                    last_q_month = (current_quarter - 2) * 3 + 1
                    last_q_start = datetime(now.year, last_q_month, 1)
                    last_q_end = datetime(now.year, last_q_month + 2, 28)  # Approximate
                return (last_q_start.strftime('%Y-%m-%d'), last_q_end.strftime('%Y-%m-%d'))
            return None
        
        # If single entity with range
        if len(entities) == 1:
            entity = entities[0]
            if entity.start_date and entity.end_date:
                return (entity.start_date, entity.end_date)
        
        # If multiple entities, find min and max dates
        if len(entities) > 1:
            dates = []
            for entity in entities:
                if entity.start_date:
                    dates.append(entity.start_date)
                if entity.end_date:
                    dates.append(entity.end_date)
            
            if dates:
                return (min(dates), max(dates))
        
        return None
    
    def expand_query_with_temporal_context(self, query: str, intent: TemporalIntent) -> str:
        """
        Expand query with temporal context for better retrieval
        
        Args:
            query: Original query
            intent: Parsed temporal intent
            
        Returns:
            Expanded query string
        """
        if not intent.has_temporal_intent:
            return query
        
        expanded_parts = [query]
        
        # Add temporal entity text
        for entity in intent.temporal_entities:
            if entity.text not in query:
                expanded_parts.append(entity.text)
        
        # Add temporal keywords
        if intent.comparison_type == 'trend':
            expanded_parts.append("change over time")
        elif intent.comparison_type == 'comparison':
            expanded_parts.append("comparison")
        
        # Add time range context
        if intent.time_range:
            start, end = intent.time_range
            # Extract year from dates
            start_year = start[:4]
            end_year = end[:4]
            if start_year not in query:
                expanded_parts.append(start_year)
            if end_year not in query and end_year != start_year:
                expanded_parts.append(end_year)
        
        return " ".join(expanded_parts)
