# src/temporal/temporal_normalizer.py
"""
Temporal Expression Normalization
Converts temporal expressions to ISO-8601 standard format
Based on ISO-TimeML TIMEX3 normalization standards
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging
from calendar import monthrange

from .temporal_entity import TemporalType, Granularity

logger = logging.getLogger(__name__)


class TemporalNormalizer:
    """
    Normalizes temporal expressions to ISO-8601 format
    Handles various formats common in financial documents
    """
    
    # Month name mappings (full and abbreviated)
    MONTH_NAMES = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12
    }
    
    # Quarter to month mappings
    QUARTER_MONTHS = {
        'Q1': (1, 3), 'q1': (1, 3),
        'Q2': (4, 6), 'q2': (4, 6),
        'Q3': (7, 9), 'q3': (7, 9),
        'Q4': (10, 12), 'q4': (10, 12)
    }
    
    def __init__(self, document_creation_time: Optional[str] = None):
        """
        Initialize normalizer
        
        Args:
            document_creation_time: Document Creation Time (DCT) in ISO format
                                   Used for resolving relative temporal expressions
        """
        self.dct = document_creation_time or datetime.now().strftime('%Y-%m-%d')
        self.dct_datetime = datetime.fromisoformat(self.dct)
    
    def normalize(self, text: str, temporal_type: TemporalType) -> Dict[str, Optional[str]]:
        """
        Normalize temporal expression to ISO format
        
        Args:
            text: Temporal expression text
            temporal_type: Type of temporal expression
            
        Returns:
            Dictionary with normalized values:
            - 'value': Single normalized value (for dates, times)
            - 'start_date': Start date for ranges
            - 'end_date': End date for ranges
            - 'granularity': Temporal granularity
        """
        text = text.strip()
        
        try:
            if temporal_type == TemporalType.DATE:
                return self._normalize_date(text)
            elif temporal_type == TemporalType.DATE_RANGE:
                return self._normalize_date_range(text)
            elif temporal_type == TemporalType.QUARTER:
                return self._normalize_quarter(text)
            elif temporal_type == TemporalType.YEAR:
                return self._normalize_year(text)
            elif temporal_type == TemporalType.MONTH:
                return self._normalize_month(text)
            elif temporal_type == TemporalType.RELATIVE:
                return self._normalize_relative(text)
            else:
                logger.warning(f"Unsupported temporal type for normalization: {temporal_type}")
                return {'value': None, 'start_date': None, 'end_date': None, 'granularity': None}
        except Exception as e:
            logger.error(f"Error normalizing '{text}': {e}")
            return {'value': None, 'start_date': None, 'end_date': None, 'granularity': None}
    
    def _normalize_date(self, text: str) -> Dict[str, Optional[str]]:
        """
        Normalize specific date expressions
        Handles formats like:
        - "16OCT2025", "16-OCT-2025"
        - "2025-07-01"
        - "July 16, 2025"
        - "16/07/2025"
        """
        # Pattern 1: 16OCT2025, 16-OCT-2025
        pattern1 = r'(\d{1,2})[-\s]?([A-Za-z]{3,})[-\s]?(\d{4})'
        match = re.match(pattern1, text, re.IGNORECASE)
        if match:
            day, month_str, year = match.groups()
            month = self.MONTH_NAMES.get(month_str.lower())
            if month:
                date_str = f"{year}-{month:02d}-{int(day):02d}"
                return {
                    'value': date_str,
                    'start_date': date_str,
                    'end_date': date_str,
                    'granularity': Granularity.DAY.value
                }
        
        # Pattern 2: ISO format YYYY-MM-DD
        pattern2 = r'(\d{4})-(\d{1,2})-(\d{1,2})'
        match = re.match(pattern2, text)
        if match:
            year, month, day = match.groups()
            date_str = f"{year}-{int(month):02d}-{int(day):02d}"
            return {
                'value': date_str,
                'start_date': date_str,
                'end_date': date_str,
                'granularity': Granularity.DAY.value
            }
        
        # Pattern 3: Month Day, Year (July 16, 2025)
        pattern3 = r'([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})'
        match = re.match(pattern3, text, re.IGNORECASE)
        if match:
            month_str, day, year = match.groups()
            month = self.MONTH_NAMES.get(month_str.lower())
            if month:
                date_str = f"{year}-{month:02d}-{int(day):02d}"
                return {
                    'value': date_str,
                    'start_date': date_str,
                    'end_date': date_str,
                    'granularity': Granularity.DAY.value
                }
        
        # Pattern 4: DD/MM/YYYY or MM/DD/YYYY (ambiguous - assume DD/MM/YYYY for international)
        pattern4 = r'(\d{1,2})/(\d{1,2})/(\d{4})'
        match = re.match(pattern4, text)
        if match:
            day, month, year = match.groups()
            # Assume DD/MM/YYYY format (international standard)
            date_str = f"{year}-{int(month):02d}-{int(day):02d}"
            return {
                'value': date_str,
                'start_date': date_str,
                'end_date': date_str,
                'granularity': Granularity.DAY.value
            }
        
        return {'value': None, 'start_date': None, 'end_date': None, 'granularity': None}
    
    def _normalize_date_range(self, text: str) -> Dict[str, Optional[str]]:
        """
        Normalize date range expressions
        Handles formats like:
        - "Jul-Dec 2025"
        - "July-December 2025"
        - "2025 Jul-Dec"
        - "Jan 2024 - Mar 2024"
        - "Jul to Dec 2025"
        """
        # Pattern 1: Jul-Dec 2025 or July-December 2025
        pattern1 = r'([A-Za-z]+)[-\s]+([A-Za-z]+)\s+(\d{4})'
        match = re.match(pattern1, text, re.IGNORECASE)
        if match:
            start_month_str, end_month_str, year = match.groups()
            start_month = self.MONTH_NAMES.get(start_month_str.lower())
            end_month = self.MONTH_NAMES.get(end_month_str.lower())
            
            if start_month and end_month:
                start_date = f"{year}-{start_month:02d}-01"
                # Get last day of end month
                last_day = monthrange(int(year), end_month)[1]
                end_date = f"{year}-{end_month:02d}-{last_day:02d}"
                
                return {
                    'value': f"{start_date}/{end_date}",
                    'start_date': start_date,
                    'end_date': end_date,
                    'granularity': Granularity.MONTH.value
                }
        
        # Pattern 2: 2025 Jul-Dec or 2025_Jul-Dec
        pattern2 = r'(\d{4})[-_\s]+([A-Za-z]+)[-\s]+([A-Za-z]+)'
        match = re.match(pattern2, text, re.IGNORECASE)
        if match:
            year, start_month_str, end_month_str = match.groups()
            start_month = self.MONTH_NAMES.get(start_month_str.lower())
            end_month = self.MONTH_NAMES.get(end_month_str.lower())
            
            if start_month and end_month:
                start_date = f"{year}-{start_month:02d}-01"
                last_day = monthrange(int(year), end_month)[1]
                end_date = f"{year}-{end_month:02d}-{last_day:02d}"
                
                return {
                    'value': f"{start_date}/{end_date}",
                    'start_date': start_date,
                    'end_date': end_date,
                    'granularity': Granularity.MONTH.value
                }
        
        # Pattern 3: Jan 2024 - Mar 2024
        pattern3 = r'([A-Za-z]+)\s+(\d{4})\s*[-–]\s*([A-Za-z]+)\s+(\d{4})'
        match = re.match(pattern3, text, re.IGNORECASE)
        if match:
            start_month_str, start_year, end_month_str, end_year = match.groups()
            start_month = self.MONTH_NAMES.get(start_month_str.lower())
            end_month = self.MONTH_NAMES.get(end_month_str.lower())
            
            if start_month and end_month:
                start_date = f"{start_year}-{start_month:02d}-01"
                last_day = monthrange(int(end_year), end_month)[1]
                end_date = f"{end_year}-{end_month:02d}-{last_day:02d}"
                
                return {
                    'value': f"{start_date}/{end_date}",
                    'start_date': start_date,
                    'end_date': end_date,
                    'granularity': Granularity.MONTH.value
                }
        
        # Pattern 4: Jul to Dec 2025 or July to December 2025
        pattern4 = r'([A-Za-z]+)\s+to\s+([A-Za-z]+)\s+(\d{4})'
        match = re.match(pattern4, text, re.IGNORECASE)
        if match:
            start_month_str, end_month_str, year = match.groups()
            start_month = self.MONTH_NAMES.get(start_month_str.lower())
            end_month = self.MONTH_NAMES.get(end_month_str.lower())
            
            if start_month and end_month:
                start_date = f"{year}-{start_month:02d}-01"
                last_day = monthrange(int(year), end_month)[1]
                end_date = f"{year}-{end_month:02d}-{last_day:02d}"
                
                return {
                    'value': f"{start_date}/{end_date}",
                    'start_date': start_date,
                    'end_date': end_date,
                    'granularity': Granularity.MONTH.value
                }
        
        # Pattern 5: July 2025 to Dec 2025 or July-2025-to-Dec-2025
        pattern5 = r'([A-Za-z]+)[\s-]+(\d{4})[\s-]+to[\s-]+([A-Za-z]+)[\s-]+(\d{4})'
        match = re.match(pattern5, text, re.IGNORECASE)
        if match:
            start_month_str, start_year, end_month_str, end_year = match.groups()
            start_month = self.MONTH_NAMES.get(start_month_str.lower())
            end_month = self.MONTH_NAMES.get(end_month_str.lower())
            
            if start_month and end_month:
                start_date = f"{start_year}-{start_month:02d}-01"
                last_day = monthrange(int(end_year), end_month)[1]
                end_date = f"{end_year}-{end_month:02d}-{last_day:02d}"
                
                return {
                    'value': f"{start_date}/{end_date}",
                    'start_date': start_date,
                    'end_date': end_date,
                    'granularity': Granularity.MONTH.value
                }
        
        return {'value': None, 'start_date': None, 'end_date': None, 'granularity': None}
    
    def _normalize_quarter(self, text: str) -> Dict[str, Optional[str]]:
        """
        Normalize quarter expressions
        Handles formats like:
        - "Q2 2024"
        - "Q1-2024"
        - "2024 Q3"
        """
        # Pattern 1: Q2 2024 or Q2-2024
        pattern1 = r'([Qq][1-4])[-\s]+(\d{4})'
        match = re.match(pattern1, text)
        if match:
            quarter, year = match.groups()
            start_month, end_month = self.QUARTER_MONTHS[quarter]
            
            start_date = f"{year}-{start_month:02d}-01"
            last_day = monthrange(int(year), end_month)[1]
            end_date = f"{year}-{end_month:02d}-{last_day:02d}"
            
            return {
                'value': f"{year}-{quarter.upper()}",
                'start_date': start_date,
                'end_date': end_date,
                'granularity': Granularity.QUARTER.value
            }
        
        # Pattern 2: 2024 Q3
        pattern2 = r'(\d{4})[-\s]+([Qq][1-4])'
        match = re.match(pattern2, text)
        if match:
            year, quarter = match.groups()
            start_month, end_month = self.QUARTER_MONTHS[quarter]
            
            start_date = f"{year}-{start_month:02d}-01"
            last_day = monthrange(int(year), end_month)[1]
            end_date = f"{year}-{end_month:02d}-{last_day:02d}"
            
            return {
                'value': f"{year}-{quarter.upper()}",
                'start_date': start_date,
                'end_date': end_date,
                'granularity': Granularity.QUARTER.value
            }
        
        return {'value': None, 'start_date': None, 'end_date': None, 'granularity': None}
    
    def _normalize_year(self, text: str) -> Dict[str, Optional[str]]:
        """
        Normalize year expressions
        Handles formats like: "2025", "year 2025"
        """
        # Extract 4-digit year
        pattern = r'(\d{4})'
        match = re.search(pattern, text)
        if match:
            year = match.group(1)
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            return {
                'value': year,
                'start_date': start_date,
                'end_date': end_date,
                'granularity': Granularity.YEAR.value
            }
        
        return {'value': None, 'start_date': None, 'end_date': None, 'granularity': None}
    
    def _normalize_month(self, text: str) -> Dict[str, Optional[str]]:
        """
        Normalize month expressions
        Handles formats like: "July 2025", "2025-07"
        """
        # Pattern 1: July 2025 or Jul 2025
        pattern1 = r'([A-Za-z]+)\s+(\d{4})'
        match = re.match(pattern1, text, re.IGNORECASE)
        if match:
            month_str, year = match.groups()
            month = self.MONTH_NAMES.get(month_str.lower())
            
            if month:
                start_date = f"{year}-{month:02d}-01"
                last_day = monthrange(int(year), month)[1]
                end_date = f"{year}-{month:02d}-{last_day:02d}"
                
                return {
                    'value': f"{year}-{month:02d}",
                    'start_date': start_date,
                    'end_date': end_date,
                    'granularity': Granularity.MONTH.value
                }
        
        # Pattern 2: 2025-07
        pattern2 = r'(\d{4})-(\d{1,2})'
        match = re.match(pattern2, text)
        if match:
            year, month = match.groups()
            month = int(month)
            
            start_date = f"{year}-{month:02d}-01"
            last_day = monthrange(int(year), month)[1]
            end_date = f"{year}-{month:02d}-{last_day:02d}"
            
            return {
                'value': f"{year}-{month:02d}",
                'start_date': start_date,
                'end_date': end_date,
                'granularity': Granularity.MONTH.value
            }
        
        return {'value': None, 'start_date': None, 'end_date': None, 'granularity': None}
    
    def _normalize_relative(self, text: str) -> Dict[str, Optional[str]]:
        """
        Normalize relative temporal expressions
        Handles formats like: "last month", "next year", "current quarter"
        Uses Document Creation Time (DCT) as anchor
        """
        text_lower = text.lower()
        
        # Last month
        if 'last month' in text_lower or 'previous month' in text_lower:
            last_month = self.dct_datetime.replace(day=1) - timedelta(days=1)
            start_date = last_month.replace(day=1).strftime('%Y-%m-%d')
            last_day = monthrange(last_month.year, last_month.month)[1]
            end_date = last_month.replace(day=last_day).strftime('%Y-%m-%d')
            
            return {
                'value': last_month.strftime('%Y-%m'),
                'start_date': start_date,
                'end_date': end_date,
                'granularity': Granularity.MONTH.value
            }
        
        # Next month
        if 'next month' in text_lower:
            # Get first day of next month
            if self.dct_datetime.month == 12:
                next_month = self.dct_datetime.replace(year=self.dct_datetime.year + 1, month=1, day=1)
            else:
                next_month = self.dct_datetime.replace(month=self.dct_datetime.month + 1, day=1)
            
            start_date = next_month.strftime('%Y-%m-%d')
            last_day = monthrange(next_month.year, next_month.month)[1]
            end_date = next_month.replace(day=last_day).strftime('%Y-%m-%d')
            
            return {
                'value': next_month.strftime('%Y-%m'),
                'start_date': start_date,
                'end_date': end_date,
                'granularity': Granularity.MONTH.value
            }
        
        # Current/this year
        if 'current year' in text_lower or 'this year' in text_lower:
            year = self.dct_datetime.year
            return {
                'value': str(year),
                'start_date': f"{year}-01-01",
                'end_date': f"{year}-12-31",
                'granularity': Granularity.YEAR.value
            }
        
        # Last year
        if 'last year' in text_lower or 'previous year' in text_lower:
            year = self.dct_datetime.year - 1
            return {
                'value': str(year),
                'start_date': f"{year}-01-01",
                'end_date': f"{year}-12-31",
                'granularity': Granularity.YEAR.value
            }
        
        # Next year
        if 'next year' in text_lower:
            year = self.dct_datetime.year + 1
            return {
                'value': str(year),
                'start_date': f"{year}-01-01",
                'end_date': f"{year}-12-31",
                'granularity': Granularity.YEAR.value
            }
        
        return {'value': None, 'start_date': None, 'end_date': None, 'granularity': None}
