#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for temporal entity extraction
Tests on actual banking PDFs to validate extraction quality
"""

import sys
import io
from pathlib import Path
import fitz  # PyMuPDF

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from temporal import TemporalEntityExtractor, TemporalType


def extract_pdf_text(pdf_path: str, max_pages: int = 3) -> str:
    """Extract text from first few pages of PDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(min(max_pages, len(doc))):
            text += doc[page_num].get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""


def test_temporal_extraction():
    """Test temporal extraction on banking PDFs"""
    
    print("=" * 80)
    print("TEMPORAL ENTITY EXTRACTION TEST")
    print("=" * 80)
    
    # Initialize extractor
    print("\n📦 Initializing Temporal Entity Extractor...")
    extractor = TemporalEntityExtractor(use_spacy=True)
    
    # Test PDFs
    pdf_dir = Path("data/datasetsforchatbot")
    test_pdfs = [
        "HBL_Car_Loan_KFS_Jul-Dec_2025_(English).pdf",
        "HBL_CreditCard-KFS_(Eng)_16OCT2025.pdf",
        "HBL_CarLoan_SOBC_2025_Jul-Dec_-_English_.pdf"
    ]
    
    for pdf_name in test_pdfs:
        pdf_path = pdf_dir / pdf_name
        
        if not pdf_path.exists():
            print(f"\n⚠️  PDF not found: {pdf_name}")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"📄 Testing: {pdf_name}")
        print(f"{'=' * 80}")
        
        # Extract text from PDF
        print("\n📖 Extracting text from PDF...")
        text = extract_pdf_text(str(pdf_path), max_pages=2)
        
        if not text:
            print("❌ No text extracted")
            continue
        
        print(f"✅ Extracted {len(text)} characters")
        
        # Extract temporal entities from filename
        print(f"\n🔍 Extracting temporal entities from FILENAME...")
        filename_entities = extractor.extract_from_filename(pdf_name)
        
        if filename_entities:
            print(f"✅ Found {len(filename_entities)} temporal entities in filename:")
            for entity in filename_entities:
                print(f"   • {entity}")
                print(f"     Type: {entity.temporal_type.value}")
                print(f"     Normalized: {entity.start_date} to {entity.end_date}")
                print(f"     Confidence: {entity.confidence:.2f}")
        else:
            print("❌ No temporal entities found in filename")
        
        # Extract temporal entities from content
        print(f"\n🔍 Extracting temporal entities from CONTENT...")
        content_entities = extractor.extract_from_text(text[:2000])  # First 2000 chars
        
        if content_entities:
            print(f"✅ Found {len(content_entities)} temporal entities in content:")
            for entity in content_entities[:10]:  # Show first 10
                print(f"   • {entity}")
                print(f"     Type: {entity.temporal_type.value}")
                if entity.start_date and entity.end_date:
                    print(f"     Normalized: {entity.start_date} to {entity.end_date}")
                elif entity.value:
                    print(f"     Normalized: {entity.value}")
                print(f"     Confidence: {entity.confidence:.2f}")
        else:
            print("❌ No temporal entities found in content")
        
        # Extract complete document metadata
        print(f"\n📊 Extracting COMPLETE DOCUMENT METADATA...")
        metadata = extractor.extract_document_metadata(
            text=text[:2000],
            filename=pdf_name
        )
        
        print(f"\n📋 Document Temporal Metadata:")
        print(f"   Document: {metadata.doc_name}")
        print(f"   Total Entities: {len(metadata.temporal_entities)}")
        print(f"   Valid From: {metadata.valid_from}")
        print(f"   Valid To: {metadata.valid_to}")
        
        if metadata.primary_time_period:
            print(f"   Primary Period: {metadata.primary_time_period}")
        
        # Show entity type distribution
        type_counts = {}
        for entity in metadata.temporal_entities:
            type_name = entity.temporal_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        print(f"\n   Entity Type Distribution:")
        for type_name, count in sorted(type_counts.items()):
            print(f"     {type_name}: {count}")
    
    print(f"\n{'=' * 80}")
    print("✅ TEST COMPLETE")
    print(f"{'=' * 80}\n")


def test_normalization_examples():
    """Test normalization with specific examples"""
    
    print("\n" + "=" * 80)
    print("NORMALIZATION TEST - Specific Examples")
    print("=" * 80)
    
    extractor = TemporalEntityExtractor(use_spacy=False)
    
    test_cases = [
        "Jul-Dec 2025",
        "Q2 2024",
        "16OCT2025",
        "July 16, 2025",
        "2025-07-01",
        "2025 Jul-Dec",
        "Q1-2024",
        "January 2025",
        "2025"
    ]
    
    for test_text in test_cases:
        print(f"\n📝 Input: '{test_text}'")
        entities = extractor.extract_from_text(test_text)
        
        if entities:
            entity = entities[0]
            print(f"   ✅ Type: {entity.temporal_type.value}")
            print(f"   ✅ Start: {entity.start_date}")
            print(f"   ✅ End: {entity.end_date}")
            print(f"   ✅ Value: {entity.value}")
            print(f"   ✅ Granularity: {entity.granularity}")
        else:
            print(f"   ❌ No entities extracted")
    
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    # Run normalization tests first
    test_normalization_examples()
    
    # Then run full PDF tests
    test_temporal_extraction()
