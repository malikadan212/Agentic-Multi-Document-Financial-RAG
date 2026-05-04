# src/temporal/temporal_entity.py
"""
Data classes for temporal entities
Based on ISO-TimeML and TIMEX3 standards
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime


class TemporalType(Enum):
    """Types of temporal expressions based on TimeML framework"""
    DATE = "DATE"  # Specific date: "2025-07-01"
    TIME = "TIME"  # Specific time: "14:30:00"
    DURATION = "DURATION"  # Duration: "3 months"
    SET = "SET"  # Recurring time: "every Monday"
    DATE_RANGE = "DATE_RANGE"  # Range: "Jul-Dec 2025"
    QUARTER = "QUARTER"  # Quarter: "Q2 2024"
    YEAR = "YEAR"  # Year only: "2025"
    MONTH = "MONTH"  # Month only: "July 2025"
    RELATIVE = "RELATIVE"  # Relative: "last month", "next year"


class Granularity(Enum):
    """Temporal granularity levels"""
    YEAR = "YEAR"
    QUARTER = "QUARTER"
    MONTH = "MONTH"
    WEEK = "WEEK"
    DAY = "DAY"
    HOUR = "HOUR"
    MINUTE = "MINUTE"
    SECOND = "SECOND"


@dataclass
class TemporalEntity:
    """
    Represents a temporal entity extracted from text
    Follows ISO-TimeML TIMEX3 annotation standard
    
    Attributes:
        text: Original text of the temporal expression
        start_char: Start character position in document
        end_char: End character position in document
        temporal_type: Type of temporal expression
        value: Normalized ISO format value
        start_date: Start date for ranges (ISO format: YYYY-MM-DD)
        end_date: End date for ranges (ISO format: YYYY-MM-DD)
        granularity: Level of temporal granularity
        confidence: Confidence score (0.0 to 1.0)
        metadata: Additional metadata
    """
    text: str
    start_char: int
    end_char: int
    temporal_type: TemporalType
    value: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    granularity: Optional[Granularity] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'text': self.text,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'temporal_type': self.temporal_type.value,
            'value': self.value,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'granularity': self.granularity.value if self.granularity else None,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        if self.start_date and self.end_date:
            return f"TemporalEntity('{self.text}', {self.temporal_type.value}, {self.start_date} to {self.end_date})"
        elif self.value:
            return f"TemporalEntity('{self.text}', {self.temporal_type.value}, {self.value})"
        else:
            return f"TemporalEntity('{self.text}', {self.temporal_type.value})"
    
    def overlaps_with(self, other: 'TemporalEntity') -> bool:
        """Check if this entity overlaps with another in the text"""
        return not (self.end_char <= other.start_char or self.start_char >= other.end_char)
    
    def contains_date(self, date_str: str) -> bool:
        """
        Check if a given date falls within this temporal entity's range
        
        Args:
            date_str: Date in ISO format (YYYY-MM-DD)
            
        Returns:
            True if date is within range, False otherwise
        """
        if not self.start_date or not self.end_date:
            return False
        
        try:
            check_date = datetime.fromisoformat(date_str)
            start = datetime.fromisoformat(self.start_date)
            end = datetime.fromisoformat(self.end_date)
            return start <= check_date <= end
        except (ValueError, TypeError):
            return False


@dataclass
class DocumentTemporalMetadata:
    """
    Temporal metadata for an entire document
    
    Attributes:
        doc_name: Document name
        creation_time: Document creation time (DCT)
        temporal_entities: List of extracted temporal entities
        valid_from: Document validity start date
        valid_to: Document validity end date
        primary_time_period: Main time period the document refers to
    """
    doc_name: str
    creation_time: Optional[str] = None  # Document Creation Time (DCT)
    temporal_entities: List[TemporalEntity] = field(default_factory=list)
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    primary_time_period: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'doc_name': self.doc_name,
            'creation_time': self.creation_time,
            'temporal_entities': [entity.to_dict() for entity in self.temporal_entities],
            'valid_from': self.valid_from,
            'valid_to': self.valid_to,
            'primary_time_period': self.primary_time_period
        }
    
    def get_entities_by_type(self, temporal_type: TemporalType) -> List[TemporalEntity]:
        """Get all entities of a specific type"""
        return [e for e in self.temporal_entities if e.temporal_type == temporal_type]
    
    def get_date_ranges(self) -> List[TemporalEntity]:
        """Get all date range entities"""
        return [e for e in self.temporal_entities 
                if e.temporal_type in [TemporalType.DATE_RANGE, TemporalType.QUARTER]]
