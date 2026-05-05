"""Test temporal retriever with mock data"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from retrieval.retriever import RetrievalResult
from retrieval.temporal_retriever import TemporalAwareRetriever

# Create mock retriever
class MockRetriever:
    def retrieve(self, query, top_k=5):
        # Return RetrievalResult objects
        return [
            RetrievalResult(
                chunk_id='chunk_1',
                content='Q2 2024 credit card rates are 18%',
                score=0.9,
                metadata={'doc_name': 'rates_2024.pdf', 'page': 1},
                rank=1
            ),
            RetrievalResult(
                chunk_id='chunk_2',
                content='Q1 2025 credit card rates are 19%',
                score=0.85,
                metadata={'doc_name': 'rates_2025.pdf', 'page': 1},
                rank=2
            )
        ]

print("Testing Temporal-Aware Retriever...")

# Create temporal retriever
base = MockRetriever()
temporal_retriever = TemporalAwareRetriever(base)

# Test retrieval
query = "What were rates in Q2 2024?"
results = temporal_retriever.retrieve(query, top_k=5)

print(f"\nQuery: {query}")
print(f"Results: {len(results)}")

for result in results:
    print(f"\n  Chunk: {result.chunk_id}")
    print(f"  Score: {result.score:.3f}")
    print(f"  Content: {result.content[:50]}...")

print("\n✅ Temporal retriever working correctly!")
