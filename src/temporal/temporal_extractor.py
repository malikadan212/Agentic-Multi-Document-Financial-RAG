# src/temporal/temporal_extractor.py
"""
Temporal Entity Extraction
Extracts temporal expressions from text using pattern matching and NLP
Based on approaches from the Transformer-Based Temporal IE paper
"""

import re
import logging
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from datetime import datetime

try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Install with: pip install spacy")

from .temporal_entity import TemporalEntity, TemporalType, DocumentTemporalMetadata
from .temporal_normalizer import TemporalNormalizer

logger = logging.getLogger(__name__)


class TemporalEntityExtractor:
    """
    Extracts temporal entities from text
    Uses combination of:
    1. Regex patterns for explicit temporal expressions
    2. spaCy NER for date entities
    3. Custom pattern matching for financial document formats
    """
    
    def __init__(self, use_spacy: bool = True, spacy_model: str = 'en_core_web_sm'):
        """
        Initialize temporal entity extractor
        
        Args:
            use_spacy: Whether to use spaCy for entity recognition
            spacy_model: spaCy model to use
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.nlp = None
        self.matcher = None
        
        if self.use_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
                self.matcher = Matcher(self.nlp.vocab)
                self._setup_patterns()
                logger.info(f"✅ Loaded spaCy model: {spacy_model}")
            except OSError:
                logger.warning(f"spaCy model '{spacy_model}' not found. Run: python -m spacy download {spacy_model}")
                self.use_spacy = False
        
        # Compile regex patterns for temporal expressions
        self._compile_patterns()
    
    def _setup_patterns(self):
        """Setup spaCy matcher patterns for temporal expressions"""
        if not self.matcher:
            return
        
        # Pattern for quarters: Q1, Q2, Q3, Q4
        quarter_pattern = [
            {"TEXT": {"REGEX": r"[Qq][1-4]"}},
            {"IS_SPACE": True, "OP": "?"},
            {"SHAPE": "dddd"}  # Year
        ]
        self.matcher.add("QUARTER", [quarter_pattern])
        
        # Pattern for month ranges: Jul-Dec, July-December
        month_range_pattern = [
            {"TEXT": {"REGEX": r"[A-Za-z]{3,9}"}},  # Month name
            {"TEXT": {"IN": ["-", "–", "to"]}},
            {"TEXT": {"REGEX": r"[A-Za-z]{3,9}"}},  # Month name
            {"IS_SPACE": True, "OP": "?"},
            {"SHAPE": "dddd"}  # Year
        ]
        self.matcher.add("MONTH_RANGE", [month_range_pattern])
    
    def _compile_patterns(self):
        """Compile regex patterns for temporal expression extraction"""
        
        # Pattern for quarters: Q1 2024, Q2-2024, 2024 Q3
        self.quarter_pattern = re.compile(
            r'\b([Qq][1-4])[-\s]+(\d{4})\b|\b(\d{4})[-\s]+([Qq][1-4])\b',
            re.IGNORECASE
        )
        
        # Pattern for month ranges: Jul-Dec 2025, July-December 2025, 2025 Jul-Dec, Jul_to_Dec_2025, July-2025-to-Dec-2025
        self.month_range_pattern = re.compile(
            r'\b([A-Za-z]{3,9})[-–\s]+([A-Za-z]{3,9})\s+(\d{4})\b|'
            r'\b(\d{4})[-_\s]+([A-Za-z]{3,9})[-–\s]+([A-Za-z]{3,9})\b|'
            r'\b([A-Za-z]{3,9})[\s_-]+to[\s_-]+([A-Za-z]{3,9})[\s_-]+(\d{4})\b|'
            r'\b([A-Za-z]{3,9})[\s_-]+(\d{4})[\s_-]+to[\s_-]+([A-Za-z]{3,9})[\s_-]+(\d{4})\b',
            re.IGNORECASE
        )
        
        # Pattern for specific dates: 16OCT2025, 16-OCT-2025, July 16 2025
        self.date_pattern = re.compile(
            r'\b(\d{1,2})[-\s]?([A-Za-z]{3,9})[-\s]?(\d{4})\b|'  # 16OCT2025
            r'\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b|'  # July 16, 2025
            r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b|'  # 2025-07-16
            r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',  # 16/07/2025
            re.IGNORECASE
        )
        
        # Pattern for year only: 2024, 2025
        self.year_pattern = re.compile(
            r'\b(20\d{2})\b'
        )
        
        # Pattern for day-month: 22_July, 22-July, 22 July
        self.day_month_pattern = re.compile(
            r'\b(\d{1,2})[\s_-]+([A-Za-z]{3,9})\b',
            re.IGNORECASE
        )
        
        # Pattern for month-year: July 2025, Jul 2025
        self.month_year_pattern = re.compile(
            r'\b([A-Za-z]{3,9})\s+(\d{4})\b',
            re.IGNORECASE
        )
        
        # Pattern for relative expressions
        self.relative_pattern = re.compile(
            r'\b(last|previous|next|current|this)\s+(month|year|quarter|week)\b',
            re.IGNORECASE
        )
    
    def extract_from_text(
        self, 
        text: str, 
        document_creation_time: Optional[str] = None
    ) -> List[TemporalEntity]:
        """
        Extract temporal entities from text
        
        Args:
            text: Input text
            document_creation_time: Document Creation Time (DCT) for relative expressions
            
        Returns:
            List of TemporalEntity objects
        """
        entities = []
        
        # Initialize normalizer with DCT
        normalizer = TemporalNormalizer(document_creation_time)
        
        # Extract using regex patterns (most reliable for financial documents)
        entities.extend(self._extract_quarters(text, normalizer))
        entities.extend(self._extract_month_ranges(text, normalizer))
        entities.extend(self._extract_dates(text, normalizer))
        entities.extend(self._extract_day_months(text, normalizer))  # NEW
        entities.extend(self._extract_month_years(text, normalizer))
        entities.extend(self._extract_years(text, normalizer))
        entities.extend(self._extract_relative(text, normalizer))
        
        # Extract using spaCy if available
        if self.use_spacy and self.nlp:
            entities.extend(self._extract_with_spacy(text, normalizer))
        
        # Remove overlapping entities (keep higher confidence ones)
        entities = self._remove_overlaps(entities)
        
        # Sort by position in text
        entities.sort(key=lambda e: e.start_char)
        
        logger.info(f"Extracted {len(entities)} temporal entities from text")
        return entities
    
    def _extract_quarters(self, text: str, normalizer: TemporalNormalizer) -> List[TemporalEntity]:
        """Extract quarter expressions"""
        entities = []
        
        for match in self.quarter_pattern.finditer(text):
            matched_text = match.group(0)
            start_char = match.start()
            end_char = match.end()
            
            # Normalize the quarter
            normalized = normalizer.normalize(matched_text, TemporalType.QUARTER)
            
            if normalized['start_date']:
                entity = TemporalEntity(
                    text=matched_text,
                    start_char=start_char,
                    end_char=end_char,
                    temporal_type=TemporalType.QUARTER,
                    value=normalized['value'],
                    start_date=normalized['start_date'],
                    end_date=normalized['end_date'],
                    granularity=normalized['granularity'],
                    confidence=0.95
                )
                entities.append(entity)
        
        return entities
    
    def _extract_month_ranges(self, text: str, normalizer: TemporalNormalizer) -> List[TemporalEntity]:
        """Extract month range expressions"""
        entities = []
        
        for match in self.month_range_pattern.finditer(text):
            matched_text = match.group(0)
            start_char = match.start()
            end_char = match.end()
            
            # Normalize the month range
            normalized = normalizer.normalize(matched_text, TemporalType.DATE_RANGE)
            
            if normalized['start_date']:
                entity = TemporalEntity(
                    text=matched_text,
                    start_char=start_char,
                    end_char=end_char,
                    temporal_type=TemporalType.DATE_RANGE,
                    value=normalized['value'],
                    start_date=normalized['start_date'],
                    end_date=normalized['end_date'],
                    granularity=normalized['granularity'],
                    confidence=0.95
                )
                entities.append(entity)
        
        return entities
    
    def _extract_day_months(self, text: str, normalizer: TemporalNormalizer) -> List[TemporalEntity]:
        """Extract day-month expressions (e.g., 22_July, 22-July)"""
        entities = []
        
        for match in self.day_month_pattern.finditer(text):
            matched_text = match.group(0)
            start_char = match.start()
            end_char = match.end()
            
            # Skip if this is part of a full date (already extracted)
            if start_char > 0 and text[start_char - 1].isdigit():
                continue
            if end_char < len(text) and text[end_char:end_char+5].strip().startswith(('20', '19')):
                continue
            
            # Try to infer year from context or use current year
            # For now, we'll skip these as they're ambiguous without year
            # They're mainly useful for filename context
            continue
        
        return entities
    
    def _extract_dates(self, text: str, normalizer: TemporalNormalizer) -> List[TemporalEntity]:
        """Extract specific date expressions"""
        entities = []
        
        for match in self.date_pattern.finditer(text):
            matched_text = match.group(0)
            start_char = match.start()
            end_char = match.end()
            
            # Normalize the date
            normalized = normalizer.normalize(matched_text, TemporalType.DATE)
            
            if normalized['value']:
                entity = TemporalEntity(
                    text=matched_text,
                    start_char=start_char,
                    end_char=end_char,
                    temporal_type=TemporalType.DATE,
                    value=normalized['value'],
                    start_date=normalized['start_date'],
                    end_date=normalized['end_date'],
                    granularity=normalized['granularity'],
                    confidence=0.90
                )
                entities.append(entity)
        
        return entities
    
    def _extract_month_years(self, text: str, normalizer: TemporalNormalizer) -> List[TemporalEntity]:
        """Extract month-year expressions"""
        entities = []
        
        for match in self.month_year_pattern.finditer(text):
            matched_text = match.group(0)
            start_char = match.start()
            end_char = match.end()
            
            # Skip if this is part of a date range (already extracted)
            # Check if there's a hyphen before or after
            if start_char > 0 and text[start_char - 1] in ['-', '–']:
                continue
            if end_char < len(text) and text[end_char] in ['-', '–']:
                continue
            
            # Normalize the month-year
            normalized = normalizer.normalize(matched_text, TemporalType.MONTH)
            
            if normalized['value']:
                entity = TemporalEntity(
                    text=matched_text,
                    start_char=start_char,
                    end_char=end_char,
                    temporal_type=TemporalType.MONTH,
                    value=normalized['value'],
                    start_date=normalized['start_date'],
                    end_date=normalized['end_date'],
                    granularity=normalized['granularity'],
                    confidence=0.85
                )
                entities.append(entity)
        
        return entities
    
    def _extract_years(self, text: str, normalizer: TemporalNormalizer) -> List[TemporalEntity]:
        """Extract year-only expressions"""
        entities = []
        
        for match in self.year_pattern.finditer(text):
            matched_text = match.group(0)
            start_char = match.start()
            end_char = match.end()
            
            # Skip if this year is part of a larger temporal expression
            # Check context before and after
            if start_char > 0:
                prev_char = text[start_char - 1]
                if prev_char.isalpha() or prev_char in ['-', '/', 'Q', 'q']:
                    continue
            
            if end_char < len(text):
                next_char = text[end_char]
                if next_char in ['-', '/']:
                    continue
            
            # Normalize the year
            normalized = normalizer.normalize(matched_text, TemporalType.YEAR)
            
            if normalized['value']:
                entity = TemporalEntity(
                    text=matched_text,
                    start_char=start_char,
                    end_char=end_char,
                    temporal_type=TemporalType.YEAR,
                    value=normalized['value'],
                    start_date=normalized['start_date'],
                    end_date=normalized['end_date'],
                    granularity=normalized['granularity'],
                    confidence=0.75  # Increased from 0.70 to reduce edge cases
                )
                entities.append(entity)
        
        return entities
    
    def _extract_relative(self, text: str, normalizer: TemporalNormalizer) -> List[TemporalEntity]:
        """Extract relative temporal expressions"""
        entities = []
        
        for match in self.relative_pattern.finditer(text):
            matched_text = match.group(0)
            start_char = match.start()
            end_char = match.end()
            
            # Normalize the relative expression
            normalized = normalizer.normalize(matched_text, TemporalType.RELATIVE)
            
            if normalized['value']:
                entity = TemporalEntity(
                    text=matched_text,
                    start_char=start_char,
                    end_char=end_char,
                    temporal_type=TemporalType.RELATIVE,
                    value=normalized['value'],
                    start_date=normalized['start_date'],
                    end_date=normalized['end_date'],
                    granularity=normalized['granularity'],
                    confidence=0.80
                )
                entities.append(entity)
        
        return entities
    
    def _extract_with_spacy(self, text: str, normalizer: TemporalNormalizer) -> List[TemporalEntity]:
        """Extract temporal entities using spaCy NER"""
        entities = []
        
        if not self.nlp:
            return entities
        
        doc = self.nlp(text)
        
        # Extract DATE entities from spaCy
        for ent in doc.ents:
            if ent.label_ == "DATE":
                matched_text = ent.text
                start_char = ent.start_char
                end_char = ent.end_char
                
                # Try to determine type and normalize
                # This is a fallback for expressions not caught by regex
                temporal_type = self._infer_temporal_type(matched_text)
                normalized = normalizer.normalize(matched_text, temporal_type)
                
                if normalized.get('value') or normalized.get('start_date'):
                    entity = TemporalEntity(
                        text=matched_text,
                        start_char=start_char,
                        end_char=end_char,
                        temporal_type=temporal_type,
                        value=normalized.get('value'),
                        start_date=normalized.get('start_date'),
                        end_date=normalized.get('end_date'),
                        granularity=normalized.get('granularity'),
                        confidence=0.75,  # Lower confidence for spaCy fallback
                        metadata={'source': 'spacy'}
                    )
                    entities.append(entity)
        
        return entities
    
    def _infer_temporal_type(self, text: str) -> TemporalType:
        """Infer temporal type from text"""
        text_lower = text.lower()
        
        if re.search(r'[Qq][1-4]', text):
            return TemporalType.QUARTER
        elif '-' in text and any(month in text_lower for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
            return TemporalType.DATE_RANGE
        elif re.search(r'\d{4}', text) and len(text) == 4:
            return TemporalType.YEAR
        elif any(word in text_lower for word in ['last', 'next', 'current', 'previous', 'this']):
            return TemporalType.RELATIVE
        else:
            return TemporalType.DATE
    
    def _remove_overlaps(self, entities: List[TemporalEntity]) -> List[TemporalEntity]:
        """
        Remove overlapping entities, keeping higher confidence ones
        If confidence is equal, keep more specific temporal types
        """
        if not entities:
            return entities
        
        # Sort by start position, then by confidence (descending)
        entities.sort(key=lambda e: (e.start_char, -e.confidence))
        
        # Type specificity ranking (higher = more specific)
        type_specificity = {
            TemporalType.DATE: 5,
            TemporalType.QUARTER: 4,
            TemporalType.DATE_RANGE: 4,
            TemporalType.MONTH: 3,
            TemporalType.YEAR: 2,
            TemporalType.RELATIVE: 1
        }
        
        filtered = []
        for entity in entities:
            # Check if this entity overlaps with any already filtered entity
            overlaps = False
            for existing in filtered:
                if entity.overlaps_with(existing):
                    # Keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        filtered.remove(existing)
                        filtered.append(entity)
                    # If confidence is equal, keep more specific type
                    elif entity.confidence == existing.confidence:
                        if type_specificity.get(entity.temporal_type, 0) > type_specificity.get(existing.temporal_type, 0):
                            filtered.remove(existing)
                            filtered.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered
    
    def extract_from_filename(self, filename: str) -> List[TemporalEntity]:
        """
        Extract temporal entities from filename
        Banking documents often have temporal info in filenames
        
        Args:
            filename: Document filename
            
        Returns:
            List of TemporalEntity objects
        """
        # Remove file extension
        name = Path(filename).stem
        
        # Better normalization: replace underscores with spaces but keep hyphens in date ranges
        # First, protect date range patterns
        name_normalized = name.replace('_to_', ' to ').replace('-to-', ' to ')
        name_normalized = name_normalized.replace('_', ' ')
        
        # Extract temporal entities from filename
        entities = self.extract_from_text(name_normalized)
        
        # Special handling for day-month patterns without year (e.g., "22 July")
        # Try to infer year from nearby year mentions or use current year
        day_month_pattern = re.compile(r'\b(\d{1,2})\s+([A-Za-z]{3,9})\b', re.IGNORECASE)
        for match in day_month_pattern.finditer(name_normalized):
            day_str, month_str = match.groups()
            month = self.MONTH_NAMES.get(month_str.lower()) if hasattr(self, 'MONTH_NAMES') else None
            
            if not month:
                from .temporal_normalizer import TemporalNormalizer
                normalizer = TemporalNormalizer()
                month = normalizer.MONTH_NAMES.get(month_str.lower())
            
            if month:
                # Look for year in the filename
                year_match = re.search(r'\b(20\d{2})\b', name_normalized)
                year = year_match.group(1) if year_match else str(datetime.now().year)
                
                # Create date string and normalize
                date_str = f"{month_str} {day_str}, {year}"
                from .temporal_normalizer import TemporalNormalizer
                normalizer = TemporalNormalizer()
                normalized = normalizer.normalize(date_str, TemporalType.DATE)
                
                if normalized['value']:
                    # Check if we already have this entity
                    already_exists = any(
                        e.start_date == normalized['start_date'] for e in entities
                    )
                    
                    if not already_exists:
                        entity = TemporalEntity(
                            text=f"{day_str} {month_str}",
                            start_char=match.start(),
                            end_char=match.end(),
                            temporal_type=TemporalType.DATE,
                            value=normalized['value'],
                            start_date=normalized['start_date'],
                            end_date=normalized['end_date'],
                            granularity=normalized['granularity'],
                            confidence=0.85,
                            metadata={'source': 'filename', 'inferred_year': year}
                        )
                        entities.append(entity)
        
        # Adjust metadata and boost confidence for filename extractions
        for entity in entities:
            if 'source' not in entity.metadata:
                entity.metadata['source'] = 'filename'
            entity.confidence = min(entity.confidence + 0.05, 1.0)
        
        return entities
    
    def extract_document_metadata(
        self, 
        text: str, 
        filename: str,
        document_creation_time: Optional[str] = None
    ) -> DocumentTemporalMetadata:
        """
        Extract complete temporal metadata for a document
        
        Args:
            text: Document text content
            filename: Document filename
            document_creation_time: Document Creation Time (DCT)
            
        Returns:
            DocumentTemporalMetadata object
        """
        # Extract from filename
        filename_entities = self.extract_from_filename(filename)
        
        # Extract from text content
        text_entities = self.extract_from_text(text, document_creation_time)
        
        # Combine all entities
        all_entities = filename_entities + text_entities
        
        # Determine document validity period
        # Use the most prominent date range or quarter from filename
        valid_from = None
        valid_to = None
        primary_period = None
        
        for entity in filename_entities:
            if entity.temporal_type in [TemporalType.DATE_RANGE, TemporalType.QUARTER]:
                valid_from = entity.start_date
                valid_to = entity.end_date
                primary_period = {
                    'type': entity.temporal_type.value,
                    'text': entity.text,
                    'start': entity.start_date,
                    'end': entity.end_date
                }
                break
        
        # Create metadata object
        metadata = DocumentTemporalMetadata(
            doc_name=filename,
            creation_time=document_creation_time,
            temporal_entities=all_entities,
            valid_from=valid_from,
            valid_to=valid_to,
            primary_time_period=primary_period
        )
        
        return metadata
