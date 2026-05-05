#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test for temporal extraction on a single file
"""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from document_processing.processor import PDFProcessor, DocumentChunker


def test_single_file():
    """Test temporal extraction on a single PDF"""
    
    print("Testing temporal extraction on single file...")
    
    # Initialize processors
    pdf_processor = PDFProcessor(use_ocr=False)
    chunker = DocumentChunker(chunk_size=512, overlap=50, extract_temporal=True)
    
    # Process one PDF
    pdf_path = "data/datasetsforchatbot/HBL_CarLoan_SOBC_2025_Jul-Dec_-_English_.pdf"
    
    print(f"\nProcessing: {Path(pdf_path).name}")
    
    # Extract text
    pages = pdf_processor.extract_text(pdf_path)
    print(f"Extracted {len(pages)} pages")
    
    # Chunk with temporal extraction
    chunks = chunker.chunk_documents(pages, doc_filename=Path(pdf_path).name)
    
    print(f"\nCreated {len(chunks)} chunks")
    
    # Show temporal statistics
    chunks_with_temporal = [c for c in chunks if c.temporal_entities and len(c.temporal_entities) > 0]
    chunks_with_validity = [c for c in chunks if c.valid_from and c.valid_to]
    
    print(f"Chunks with temporal entities: {len(chunks_with_temporal)}")
    print(f"Chunks with validity periods: {len(chunks_with_validity)}")
    
    # Show first chunk with temporal data
    if chunks_with_validity:
        chunk = chunks_with_validity[0]
        print(f"\nSample Chunk:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Validity: {chunk.valid_from} to {chunk.valid_to}")
        print(f"  Temporal Entities: {len(chunk.temporal_entities)}")
        for entity in chunk.temporal_entities[:3]:
            print(f"    - {entity.text} ({entity.temporal_type.value}): {entity.start_date} to {entity.end_date}")
        print(f"  Content: {chunk.content[:200]}...")
    
    print("\nTest complete!")
    return chunks


if __name__ == "__main__":
    chunks = test_single_file()
