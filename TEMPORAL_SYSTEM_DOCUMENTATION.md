# Temporal Information Extraction System - Complete Documentation

## Overview

This document provides complete documentation for the temporal information extraction system implemented for the Multi-Document Multimodal Financial Analysis System.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Processing Pipeline              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      PDF/Excel Processor                     │
│  • Extracts text from documents                              │
│  • Extracts tables                                           │
│  • OCR for scanned documents                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Document Chunker                        │
│  • Semantic chunking with overlap                            │
│  • Temporal entity extraction (NEW)                          │
│  • Temporal metadata assignment (NEW)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Temporal Entity Extractor                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  1. Pattern Matching (Regex)                          │  │
│  │     • Quarters: Q1 2024, Q2-2024                      │  │
│  │     • Date Ranges: Jul-Dec 2025                       │  │
│  │     • Specific Dates: 16OCT2025                       │  │
│  │     • Years: 2025                                     │  │
│  │     • Months: July 2025                               │  │
│  │     • Relative: last month, next year                 │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  2. Temporal Normalizer                               │  │
│  │     • Converts to ISO-8601 format                     │  │
│  │     • Handles date ranges                             │  │
│  │     • Quarter to month mapping                        │  │
│  │     • Relative date resolution (DCT-based)            │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  3. Entity Post-Processing                            │  │
│  │     • Overlap removal                                 │  │
│  │     • Confidence scoring                              │  │
│  │     • Type-based prioritization                       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Document Chunks                         │
│  • content: str                                              │
│  • metadata: Dict                                            │
│  • chunk_id: str                                             │
│  • temporal_entities: List[TemporalEntity] (NEW)             │
│  • valid_from: str (NEW)                                     │
│  • valid_to: str (NEW)                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Models

### TemporalEntity

```python
@dataclass
class TemporalEntity:
    text: str                    # Original text: "Jul-Dec 2025"
    start_char: int              # Position in document
    end_char: int                # End position
    temporal_type: TemporalType  # DATE_RANGE, QUARTER, etc.
    value: Optional[str]         # Normalized value
    start_date: Optional[str]    # ISO format: "2025-07-01"
    end_date: Optional[str]      # ISO format: "2025-12-31"
    granularity: Optional[Granularity]  # MONTH, QUARTER, etc.
    confidence: float            # 0.0 to 1.0
    metadata: Dict[str, Any]     # Additional info
```

### TemporalType Enum

```python
class TemporalType(Enum):
    DATE = "DATE"              # Specific date
    TIME = "TIME"              # Specific time
    DURATION = "DURATION"      # Duration
    SET = "SET"                # Recurring time
    DATE_RANGE = "DATE_RANGE"  # Range of dates
    QUARTER = "QUARTER"        # Financial quarter
    YEAR = "YEAR"              # Year only
    MONTH = "MONTH"            # Month only
    RELATIVE = "RELATIVE"      # Relative expression
```

### Granularity Enum

```python
class Granularity(Enum):
    YEAR = "YEAR"
    QUARTER = "QUARTER"
    MONTH = "MONTH"
    WEEK = "WEEK"
    DAY = "DAY"
    HOUR = "HOUR"
    MINUTE = "MINUTE"
    SECOND = "SECOND"
```

---

## API Reference

### TemporalEntityExtractor

#### Initialization

```python
from temporal import TemporalEntityExtractor

extractor = TemporalEntityExtractor(
    use_spacy=True,              # Use spaCy for additional coverage
    spacy_model='en_core_web_sm' # spaCy model to use
)
```

#### Methods

##### `extract_from_text(text, document_creation_time=None)`

Extract temporal entities from text.

**Parameters:**
- `text` (str): Input text
- `document_creation_time` (str, optional): DCT in ISO format for relative expressions

**Returns:**
- `List[TemporalEntity]`: Extracted temporal entities

**Example:**
```python
entities = extractor.extract_from_text("Q2 2024 and Jul-Dec 2025")

for entity in entities:
    print(f"{entity.text}: {entity.start_date} to {entity.end_date}")
# Output:
# Q2 2024: 2024-04-01 to 2024-06-30
# Jul-Dec 2025: 2025-07-01 to 2025-12-31
```

##### `extract_from_filename(filename)`

Extract temporal entities from filename.

**Parameters:**
- `filename` (str): Document filename

**Returns:**
- `List[TemporalEntity]`: Extracted temporal entities

**Example:**
```python
entities = extractor.extract_from_filename(
    "HBL_CarLoan_SOBC_2025_Jul-Dec_-_English_.pdf"
)
```

##### `extract_document_metadata(text, filename, document_creation_time=None)`

Extract complete temporal metadata for a document.

**Parameters:**
- `text` (str): Document text content
- `filename` (str): Document filename
- `document_creation_time` (str, optional): DCT in ISO format

**Returns:**
- `DocumentTemporalMetadata`: Complete temporal metadata

**Example:**
```python
metadata = extractor.extract_document_metadata(
    text=document_text,
    filename="HBL_CreditCard_KFS_Jul-Dec_2025.pdf"
)

print(f"Valid from: {metadata.valid_from}")
print(f"Valid to: {metadata.valid_to}")
print(f"Total entities: {len(metadata.temporal_entities)}")
```

---

### TemporalNormalizer

#### Initialization

```python
from temporal import TemporalNormalizer

normalizer = TemporalNormalizer(
    document_creation_time="2024-01-15"  # DCT for relative expressions
)
```

#### Methods

##### `normalize(text, temporal_type)`

Normalize temporal expression to ISO format.

**Parameters:**
- `text` (str): Temporal expression text
- `temporal_type` (TemporalType): Type of temporal expression

**Returns:**
- `Dict`: Normalized values with keys:
  - `value`: Single normalized value
  - `start_date`: Start date (ISO format)
  - `end_date`: End date (ISO format)
  - `granularity`: Temporal granularity

**Example:**
```python
from temporal import TemporalType

result = normalizer.normalize("Q2 2024", TemporalType.QUARTER)

print(result)
# Output:
# {
#     'value': '2024-Q2',
#     'start_date': '2024-04-01',
#     'end_date': '2024-06-30',
#     'granularity': 'QUARTER'
# }
```

---

### DocumentChunk (Updated)

#### Attributes

```python
@dataclass
class DocumentChunk:
    content: str                           # Text content
    metadata: Dict[str, Any]               # Metadata
    chunk_id: str                          # Unique ID
    temporal_entities: Optional[List]      # Temporal entities (NEW)
    valid_from: Optional[str]              # Validity start (NEW)
    valid_to: Optional[str]                # Validity end (NEW)
```

#### Example Usage

```python
# Access temporal information from chunks
for chunk in chunks:
    if chunk.temporal_entities:
        print(f"Chunk {chunk.chunk_id}:")
        print(f"  Valid: {chunk.valid_from} to {chunk.valid_to}")
        
        for entity in chunk.temporal_entities:
            print(f"  - {entity.text} ({entity.temporal_type.value})")
```

---

### DocumentPipeline (Updated)

#### Initialization

```python
from document_processing.processor import DocumentPipeline

pipeline = DocumentPipeline(
    chunk_size=512,      # Maximum tokens per chunk
    overlap=50,          # Overlap between chunks
    use_ocr=True         # Use OCR for scanned documents
)
```

#### Methods

##### `process_directory(directory)`

Process all documents in a directory with temporal extraction.

**Parameters:**
- `directory` (str): Path to directory containing documents

**Returns:**
- `List[DocumentChunk]`: Processed chunks with temporal metadata

**Example:**
```python
chunks = pipeline.process_directory("data/datasetsforchatbot")

# Statistics
total_chunks = len(chunks)
temporal_chunks = sum(1 for c in chunks if c.temporal_entities)
validity_chunks = sum(1 for c in chunks if c.valid_from and c.valid_to)

print(f"Total chunks: {total_chunks}")
print(f"With temporal entities: {temporal_chunks}")
print(f"With validity periods: {validity_chunks}")
```

---

## Temporal Format Support

### Date Formats

| Format | Example | Normalized |
|--------|---------|------------|
| DDMMMYYYY | 16OCT2025 | 2025-10-16 |
| DD-MMM-YYYY | 16-OCT-2025 | 2025-10-16 |
| Month DD, YYYY | July 16, 2025 | 2025-07-16 |
| YYYY-MM-DD | 2025-07-16 | 2025-07-16 |
| DD/MM/YYYY | 16/07/2025 | 2025-07-16 |

### Date Range Formats

| Format | Example | Start | End |
|--------|---------|-------|-----|
| MMM-MMM YYYY | Jul-Dec 2025 | 2025-07-01 | 2025-12-31 |
| Month-Month YYYY | July-December 2025 | 2025-07-01 | 2025-12-31 |
| YYYY MMM-MMM | 2025 Jul-Dec | 2025-07-01 | 2025-12-31 |
| MMM YYYY - MMM YYYY | Jan 2024 - Mar 2024 | 2024-01-01 | 2024-03-31 |

### Quarter Formats

| Format | Example | Start | End |
|--------|---------|-------|-----|
| QN YYYY | Q2 2024 | 2024-04-01 | 2024-06-30 |
| QN-YYYY | Q1-2024 | 2024-01-01 | 2024-03-31 |
| YYYY QN | 2024 Q3 | 2024-07-01 | 2024-09-30 |

### Other Formats

| Type | Example | Normalized |
|------|---------|------------|
| Year | 2025 | 2025-01-01 to 2025-12-31 |
| Month | July 2025 | 2025-07-01 to 2025-07-31 |
| Relative | last month | (calculated from DCT) |
| Relative | next year | (calculated from DCT) |

---

## Configuration

### Enable/Disable Temporal Extraction

```python
# Enable (default)
chunker = DocumentChunker(
    chunk_size=512,
    overlap=50,
    extract_temporal=True  # Default
)

# Disable
chunker = DocumentChunker(
    chunk_size=512,
    overlap=50,
    extract_temporal=False
)
```

### Configure spaCy Usage

```python
# With spaCy (recommended)
extractor = TemporalEntityExtractor(
    use_spacy=True,
    spacy_model='en_core_web_sm'
)

# Without spaCy (regex only)
extractor = TemporalEntityExtractor(
    use_spacy=False
)
```

---

## Performance Considerations

### Processing Time

- **Overhead**: ~10% increase in processing time
- **Regex patterns**: Very fast (<1ms per chunk)
- **spaCy NER**: Slower but more comprehensive (~5-10ms per chunk)

### Memory Usage

- **Minimal impact**: ~50KB per 1000 chunks
- **Temporal entities**: Small objects (~200 bytes each)

### Optimization Tips

1. **Disable spaCy** for faster processing if regex patterns are sufficient
2. **Batch processing**: Process multiple files in parallel
3. **Chunk size**: Larger chunks = fewer extractions = faster processing

---

## Error Handling

### Common Issues

#### 1. spaCy Model Not Found

**Error:**
```
WARNING: spaCy model 'en_core_web_sm' not found
```

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

#### 2. Import Error

**Error:**
```
ImportError: cannot import name 'TemporalEntityExtractor'
```

**Solution:**
Ensure `src` is in Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
```

#### 3. No Temporal Entities Extracted

**Possible Causes:**
- Text doesn't contain temporal expressions
- Unsupported format
- Extraction disabled

**Debug:**
```python
# Test extraction directly
from temporal import TemporalEntityExtractor

extractor = TemporalEntityExtractor(use_spacy=False)
entities = extractor.extract_from_text("Your text here")
print(f"Found {len(entities)} entities")
```

---

## Testing

### Unit Tests

```python
# Test normalization
from temporal import TemporalNormalizer, TemporalType

normalizer = TemporalNormalizer()
result = normalizer.normalize("Q2 2024", TemporalType.QUARTER)
assert result['start_date'] == "2024-04-01"
assert result['end_date'] == "2024-06-30"
```

### Integration Tests

```python
# Test full pipeline
from document_processing.processor import DocumentPipeline

pipeline = DocumentPipeline()
chunks = pipeline.process_directory("test_data")

# Verify temporal extraction
temporal_chunks = [c for c in chunks if c.temporal_entities]
assert len(temporal_chunks) > 0
```

### Test Scripts

- `test_temporal_extraction.py`: Tests extraction and normalization
- `test_single_file_temporal.py`: Tests integration on single file
- `test_temporal_integration.py`: Full integration test

---

## Future Enhancements

### Phase 2: Temporal-Aware Retrieval
- Filter chunks by date ranges
- Temporal similarity scoring
- Recency-based ranking

### Phase 3: Advanced Features
- Temporal relation extraction (before, after, simultaneous)
- Temporal conflict resolution
- Trend analysis across time periods
- Temporal question answering

---

## References

### Research Papers
- "Transformer-Based Temporal Information Extraction and Application" (arXiv:2504.07470)
- "Time-Sensitive Modeling and Retrieval for Evolving Knowledge" (arXiv:2510.13590)

### Standards
- ISO-TimeML
- TIMEX3 annotation standard
- ISO-8601 date format

---

## Support

For issues or questions:
1. Check this documentation
2. Review test scripts for examples
3. Check implementation summary: `TEMPORAL_EXTRACTION_IMPLEMENTATION_SUMMARY.md`

---

## Changelog

### Version 1.0.0 (Phase 1 Complete)
- ✅ Temporal entity extraction
- ✅ Temporal normalization
- ✅ Integration with document processing
- ✅ Support for 9+ temporal formats
- ✅ Confidence scoring
- ✅ Document-level metadata extraction
