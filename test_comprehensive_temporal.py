"""
Comprehensive Temporal Extraction Testing
Tests the temporal extraction system on the full dataset
Identifies edge cases and provides detailed statistics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from document_processing.processor import DocumentPipeline
from temporal import TemporalEntityExtractor, TemporalType
from collections import defaultdict, Counter
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TemporalTestSuite:
    """Comprehensive test suite for temporal extraction"""
    
    def __init__(self, data_directory: str, max_pages: int = 50):
        self.data_directory = data_directory
        self.max_pages = max_pages  # Skip files with too many pages
        self.pipeline = DocumentPipeline(chunk_size=512, overlap=50, use_ocr=False)
        self.extractor = TemporalEntityExtractor(use_spacy=False)
        self.results = {
            'total_files': 0,
            'total_chunks': 0,
            'chunks_with_temporal': 0,
            'chunks_with_validity': 0,
            'temporal_types': Counter(),
            'files_analysis': [],
            'edge_cases': [],
            'filename_extraction': [],
            'extraction_patterns': defaultdict(list),
            'skipped_files': []
        }
    
    def run_full_test(self):
        """Run comprehensive tests on full dataset"""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE TEMPORAL EXTRACTION TEST SUITE")
        logger.info("=" * 80)
        
        # Test 1: Process full dataset
        logger.info("\n📊 TEST 1: Processing Full Dataset")
        chunks = self.test_full_dataset()
        
        # Test 2: Analyze filename extraction
        logger.info("\n📝 TEST 2: Filename Temporal Extraction Analysis")
        self.test_filename_extraction()
        
        # Test 3: Analyze temporal patterns
        logger.info("\n🔍 TEST 3: Temporal Pattern Analysis")
        self.analyze_temporal_patterns(chunks)
        
        # Test 4: Edge case detection
        logger.info("\n⚠️  TEST 4: Edge Case Detection")
        self.detect_edge_cases(chunks)
        
        # Test 5: Validation accuracy
        logger.info("\n✅ TEST 5: Validation Accuracy Check")
        self.validate_extractions(chunks)
        
        # Generate report
        logger.info("\n📋 Generating Comprehensive Report...")
        self.generate_report()
        
        return self.results
    
    def test_full_dataset(self):
        """Test temporal extraction on full dataset (with smart filtering)"""
        logger.info(f"Processing directory: {self.data_directory}")
        
        try:
            # Process files individually with page limit check
            data_path = Path(self.data_directory)
            all_chunks = []
            
            for pdf_file in sorted(data_path.glob("**/*.pdf")):
                try:
                    # Quick check: count pages before processing
                    import fitz
                    with fitz.open(str(pdf_file)) as doc:
                        page_count = len(doc)
                    
                    if page_count > self.max_pages:
                        logger.warning(f"⏭️  Skipping {pdf_file.name} ({page_count} pages > {self.max_pages} limit)")
                        self.results['skipped_files'].append({
                            'filename': pdf_file.name,
                            'reason': f'Too many pages ({page_count})',
                            'page_count': page_count
                        })
                        continue
                    
                    logger.info(f"📄 Processing {pdf_file.name} ({page_count} pages)...")
                    
                    # Extract text only (skip tables for speed)
                    pages = self.pipeline.pdf_processor.extract_text(str(pdf_file))
                    
                    # Chunk with temporal extraction
                    file_chunks = self.pipeline.chunker.chunk_documents(pages, doc_filename=pdf_file.name)
                    all_chunks.extend(file_chunks)
                    
                    logger.info(f"   ✅ Created {len(file_chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"   ❌ Error processing {pdf_file.name}: {e}")
                    self.results['skipped_files'].append({
                        'filename': pdf_file.name,
                        'reason': f'Error: {str(e)}',
                        'page_count': 0
                    })
            
            chunks = all_chunks
            
            if not chunks:
                logger.error("❌ No chunks created!")
                return []
            
            self.results['total_chunks'] = len(chunks)
            self.results['chunks_with_temporal'] = sum(
                1 for c in chunks if c.temporal_entities and len(c.temporal_entities) > 0
            )
            self.results['chunks_with_validity'] = sum(
                1 for c in chunks if c.valid_from and c.valid_to
            )
            
            # Count temporal types
            for chunk in chunks:
                if chunk.temporal_entities:
                    for entity in chunk.temporal_entities:
                        self.results['temporal_types'][entity.temporal_type.value] += 1
            
            # Analyze by file
            files_data = defaultdict(lambda: {
                'chunks': 0,
                'temporal_chunks': 0,
                'validity_chunks': 0,
                'entities': []
            })
            
            for chunk in chunks:
                doc_name = chunk.metadata.get('doc_name', 'unknown')
                files_data[doc_name]['chunks'] += 1
                
                if chunk.temporal_entities:
                    files_data[doc_name]['temporal_chunks'] += 1
                    files_data[doc_name]['entities'].extend([
                        {
                            'text': e.text,
                            'type': e.temporal_type.value,
                            'start': e.start_date,
                            'end': e.end_date,
                            'confidence': e.confidence
                        }
                        for e in chunk.temporal_entities
                    ])
                
                if chunk.valid_from and chunk.valid_to:
                    files_data[doc_name]['validity_chunks'] += 1
            
            self.results['files_analysis'] = [
                {'file': k, **v} for k, v in files_data.items()
            ]
            self.results['total_files'] = len(files_data)
            
            logger.info(f"\n✅ Processed {len(chunks)} chunks from {len(files_data)} files")
            logger.info(f"   ⏭️  Skipped {len(self.results['skipped_files'])} files")
            logger.info(f"   📅 {self.results['chunks_with_temporal']} chunks with temporal entities ({self.results['chunks_with_temporal']/len(chunks)*100:.1f}%)")
            logger.info(f"   📅 {self.results['chunks_with_validity']} chunks with validity periods ({self.results['chunks_with_validity']/len(chunks)*100:.1f}%)")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error processing dataset: {e}")
            raise
    
    def test_filename_extraction(self):
        """Test temporal extraction from filenames"""
        data_path = Path(self.data_directory)
        pdf_files = list(data_path.glob("**/*.pdf"))
        
        logger.info(f"Testing filename extraction on {len(pdf_files)} files")
        
        successful = 0
        failed = 0
        
        for pdf_file in pdf_files:
            filename = pdf_file.name
            entities = self.extractor.extract_from_filename(filename)
            
            result = {
                'filename': filename,
                'entities_found': len(entities),
                'entities': [
                    {
                        'text': e.text,
                        'type': e.temporal_type.value,
                        'start': e.start_date,
                        'end': e.end_date,
                        'confidence': e.confidence
                    }
                    for e in entities
                ]
            }
            
            self.results['filename_extraction'].append(result)
            
            if entities:
                successful += 1
            else:
                failed += 1
        
        logger.info(f"   ✅ Successful: {successful}/{len(pdf_files)} ({successful/len(pdf_files)*100:.1f}%)")
        logger.info(f"   ❌ Failed: {failed}/{len(pdf_files)} ({failed/len(pdf_files)*100:.1f}%)")
        
        # Show examples
        logger.info("\n   Examples of successful filename extraction:")
        for result in self.results['filename_extraction'][:5]:
            if result['entities_found'] > 0:
                logger.info(f"      {result['filename']}")
                for entity in result['entities']:
                    logger.info(f"         → {entity['text']} ({entity['type']}): {entity['start']} to {entity['end']}")
    
    def analyze_temporal_patterns(self, chunks):
        """Analyze temporal patterns in the dataset"""
        logger.info("Analyzing temporal patterns...")
        
        # Pattern: Which formats are most common?
        format_patterns = defaultdict(int)
        
        for chunk in chunks:
            if chunk.temporal_entities:
                for entity in chunk.temporal_entities:
                    # Categorize by pattern
                    text = entity.text
                    if 'Q' in text.upper():
                        format_patterns['Quarter (Q1-Q4)'] += 1
                    elif '-' in text and any(month in text.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                        format_patterns['Month Range (Jul-Dec)'] += 1
                    elif any(month in text.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']):
                        if ',' in text:
                            format_patterns['Full Date (Month DD, YYYY)'] += 1
                        else:
                            format_patterns['Month-Year'] += 1
                    elif text.isdigit() and len(text) == 4:
                        format_patterns['Year Only'] += 1
                    else:
                        format_patterns['Other Format'] += 1
        
        self.results['extraction_patterns'] = dict(format_patterns)
        
        logger.info("   Temporal format distribution:")
        for pattern, count in sorted(format_patterns.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"      {pattern}: {count}")
    
    def detect_edge_cases(self, chunks):
        """Detect edge cases and potential issues"""
        logger.info("Detecting edge cases...")
        
        edge_cases = []
        
        for chunk in chunks:
            if chunk.temporal_entities:
                for entity in chunk.temporal_entities:
                    # Edge case 1: Very low confidence
                    if entity.confidence < 0.75:
                        edge_cases.append({
                            'type': 'Low Confidence',
                            'chunk_id': chunk.chunk_id,
                            'entity': entity.text,
                            'confidence': entity.confidence,
                            'context': chunk.content[max(0, entity.start_char-50):entity.end_char+50]
                        })
                    
                    # Edge case 2: Missing normalization
                    if not entity.start_date or not entity.end_date:
                        edge_cases.append({
                            'type': 'Missing Normalization',
                            'chunk_id': chunk.chunk_id,
                            'entity': entity.text,
                            'temporal_type': entity.temporal_type.value,
                            'context': chunk.content[max(0, entity.start_char-50):entity.end_char+50]
                        })
                    
                    # Edge case 3: Suspicious date ranges (end before start)
                    if entity.start_date and entity.end_date:
                        if entity.start_date > entity.end_date:
                            edge_cases.append({
                                'type': 'Invalid Date Range',
                                'chunk_id': chunk.chunk_id,
                                'entity': entity.text,
                                'start': entity.start_date,
                                'end': entity.end_date,
                                'context': chunk.content[max(0, entity.start_char-50):entity.end_char+50]
                            })
        
        self.results['edge_cases'] = edge_cases
        
        if edge_cases:
            logger.info(f"   ⚠️  Found {len(edge_cases)} edge cases:")
            
            # Group by type
            by_type = defaultdict(list)
            for case in edge_cases:
                by_type[case['type']].append(case)
            
            for case_type, cases in by_type.items():
                logger.info(f"      {case_type}: {len(cases)} cases")
                # Show first example
                if cases:
                    example = cases[0]
                    logger.info(f"         Example: '{example['entity']}' in {example['chunk_id']}")
        else:
            logger.info("   ✅ No edge cases detected!")
    
    def validate_extractions(self, chunks):
        """Validate extraction accuracy on known patterns"""
        logger.info("Validating extraction accuracy...")
        
        # Test known patterns
        test_cases = [
            ("Q1 2024", TemporalType.QUARTER, "2024-01-01", "2024-03-31"),
            ("Q2 2024", TemporalType.QUARTER, "2024-04-01", "2024-06-30"),
            ("Jul-Dec 2025", TemporalType.DATE_RANGE, "2025-07-01", "2025-12-31"),
            ("16OCT2025", TemporalType.DATE, "2025-10-16", "2025-10-16"),
            ("July 16, 2025", TemporalType.DATE, "2025-07-16", "2025-07-16"),
            ("2025", TemporalType.YEAR, "2025-01-01", "2025-12-31"),
        ]
        
        passed = 0
        failed = 0
        
        for text, expected_type, expected_start, expected_end in test_cases:
            entities = self.extractor.extract_from_text(text)
            
            if entities:
                entity = entities[0]
                if (entity.temporal_type == expected_type and 
                    entity.start_date == expected_start and 
                    entity.end_date == expected_end):
                    passed += 1
                    logger.info(f"   ✅ PASS: '{text}' → {entity.start_date} to {entity.end_date}")
                else:
                    failed += 1
                    logger.info(f"   ❌ FAIL: '{text}' → Expected {expected_start} to {expected_end}, got {entity.start_date} to {entity.end_date}")
            else:
                failed += 1
                logger.info(f"   ❌ FAIL: '{text}' → No entities extracted")
        
        accuracy = passed / len(test_cases) * 100
        logger.info(f"\n   Validation Accuracy: {accuracy:.1f}% ({passed}/{len(test_cases)} passed)")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        report_path = Path("temporal_test_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"✅ Report saved to: {report_path}")
        
        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Files Processed: {self.results['total_files']}")
        logger.info(f"Total Chunks Created: {self.results['total_chunks']}")
        logger.info(f"Chunks with Temporal Entities: {self.results['chunks_with_temporal']} ({self.results['chunks_with_temporal']/self.results['total_chunks']*100:.1f}%)")
        logger.info(f"Chunks with Validity Periods: {self.results['chunks_with_validity']} ({self.results['chunks_with_validity']/self.results['total_chunks']*100:.1f}%)")
        
        logger.info("\nTemporal Type Distribution:")
        for temp_type, count in self.results['temporal_types'].most_common():
            logger.info(f"   {temp_type}: {count}")
        
        logger.info("\nFilename Extraction Success Rate:")
        successful = sum(1 for r in self.results['filename_extraction'] if r['entities_found'] > 0)
        total = len(self.results['filename_extraction'])
        if total > 0:
            logger.info(f"   {successful}/{total} ({successful/total*100:.1f}%)")
        
        logger.info("\nEdge Cases Found:")
        logger.info(f"   {len(self.results['edge_cases'])} cases")
        
        logger.info("\nSkipped Files:")
        if self.results['skipped_files']:
            for skipped in self.results['skipped_files']:
                logger.info(f"   {skipped['filename']}: {skipped['reason']}")
        else:
            logger.info("   None")
        
        logger.info("\n" + "=" * 80)


def main():
    """Run comprehensive test suite"""
    # Path to your dataset
    data_dir = "data/datasetsforchatbot"
    
    # Create test suite
    test_suite = TemporalTestSuite(data_dir)
    
    # Run all tests
    results = test_suite.run_full_test()
    
    print("\n✅ Comprehensive testing complete!")
    print(f"📋 Detailed report saved to: temporal_test_report.json")


if __name__ == "__main__":
    main()
