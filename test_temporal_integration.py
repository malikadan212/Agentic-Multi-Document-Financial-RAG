#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test for temporal extraction in document processing pipeline
Tests the complete flow from PDF to chunks with temporal metadata
"""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from document_processing.processor import DocumentPipeline


def test_temporal_integration():
    """Test complete temporal extraction integration"""
    
    print("=" * 80)
    print("TEMPORAL EXTRACTION INTEGRATION TEST")
    print("=" * 80)
    
    # Initialize pipeline
    print("\n[1/3] Initializing Document Processing Pipeline...")
    pipeline = DocumentPipeline(chunk_size=512, overlap=50, use_ocr=False)
    
    # Process documents
    print("\n[2/3] Processing documents from data/datasetsforchatbot...")
    print("      (This will extract text, chunk, and extract temporal entities)")
    
    chunks = pipeline.process_directory("data/datasetsforchatbot")
    
    print(f"\n[3/3] Analyzing Results...")
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")
    
    # Overall statistics
    print(f"\nTotal Chunks: {len(chunks)}")
    
    # Temporal statistics
    chunks_with_temporal = [c for c in chunks if c.temporal_entities and len(c.temporal_entities) > 0]
    chunks_with_validity = [c for c in chunks if c.valid_from and c.valid_to]
    
    print(f"Chunks with Temporal Entities: {len(chunks_with_temporal)} ({len(chunks_with_temporal)/len(chunks)*100:.1f}%)")
    print(f"Chunks with Validity Periods: {len(chunks_with_validity)} ({len(chunks_with_validity)/len(chunks)*100:.1f}%)")
    
    # Temporal entity type distribution
    print(f"\n{'=' * 80}")
    print("TEMPORAL ENTITY DISTRIBUTION")
    print(f"{'=' * 80}")
    
    entity_types = {}
    total_entities = 0
    for chunk in chunks_with_temporal:
        for entity in chunk.temporal_entities:
            entity_type = entity.temporal_type.value
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            total_entities += 1
    
    print(f"\nTotal Temporal Entities Extracted: {total_entities}")
    print(f"\nBy Type:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type:15s}: {count:4d} ({count/total_entities*100:5.1f}%)")
    
    # Sample chunks with temporal information
    print(f"\n{'=' * 80}")
    print("SAMPLE CHUNKS WITH TEMPORAL INFORMATION")
    print(f"{'=' * 80}")
    
    sample_chunks = [c for c in chunks_with_validity[:5]]  # First 5 with validity
    
    for i, chunk in enumerate(sample_chunks, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Document: {chunk.metadata.get('doc_name')}")
        print(f"Page: {chunk.metadata.get('page')}")
        print(f"Validity: {chunk.valid_from} to {chunk.valid_to}")
        print(f"Temporal Entities: {len(chunk.temporal_entities) if chunk.temporal_entities else 0}")
        
        if chunk.temporal_entities:
            print(f"Entities:")
            for entity in chunk.temporal_entities[:3]:  # Show first 3
                print(f"  - {entity.text} ({entity.temporal_type.value})")
                if entity.start_date and entity.end_date:
                    print(f"    Normalized: {entity.start_date} to {entity.end_date}")
        
        print(f"Content Preview: {chunk.content[:150]}...")
    
    # Document-level temporal coverage
    print(f"\n{'=' * 80}")
    print("DOCUMENT-LEVEL TEMPORAL COVERAGE")
    print(f"{'=' * 80}")
    
    doc_stats = {}
    for chunk in chunks:
        doc_name = chunk.metadata.get('doc_name', 'unknown')
        if doc_name not in doc_stats:
            doc_stats[doc_name] = {
                'total_chunks': 0,
                'temporal_chunks': 0,
                'validity_chunks': 0,
                'entities': []
            }
        
        doc_stats[doc_name]['total_chunks'] += 1
        if chunk.temporal_entities and len(chunk.temporal_entities) > 0:
            doc_stats[doc_name]['temporal_chunks'] += 1
            doc_stats[doc_name]['entities'].extend(chunk.temporal_entities)
        if chunk.valid_from and chunk.valid_to:
            doc_stats[doc_name]['validity_chunks'] += 1
    
    print(f"\nPer-Document Statistics:")
    for doc_name, stats in sorted(doc_stats.items()):
        print(f"\n{doc_name}:")
        print(f"  Total Chunks: {stats['total_chunks']}")
        print(f"  Chunks with Temporal Entities: {stats['temporal_chunks']}")
        print(f"  Chunks with Validity: {stats['validity_chunks']}")
        print(f"  Total Entities Extracted: {len(stats['entities'])}")
    
    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print(f"{'=' * 80}\n")
    
    return chunks


if __name__ == "__main__":
    chunks = test_temporal_integration()
    
    # Save a sample chunk for inspection
    if chunks:
        print("\nSaving sample chunk with temporal data to 'sample_chunk.txt'...")
        with open('sample_chunk.txt', 'w', encoding='utf-8') as f:
            for chunk in chunks:
                if chunk.temporal_entities and len(chunk.temporal_entities) > 0:
                    f.write(f"Chunk ID: {chunk.chunk_id}\n")
                    f.write(f"Document: {chunk.metadata.get('doc_name')}\n")
                    f.write(f"Validity: {chunk.valid_from} to {chunk.valid_to}\n")
                    f.write(f"\nTemporal Entities:\n")
                    for entity in chunk.temporal_entities:
                        f.write(f"  - {entity}\n")
                    f.write(f"\nContent:\n{chunk.content}\n")
                    f.write("\n" + "=" * 80 + "\n\n")
                    break
        print("Sample saved!")
