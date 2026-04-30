# src/retrieval/preloaded_retriever.py
"""
Retriever that works with pre-loaded FAISS index and chunk metadata
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import ChunkMetadataLoader from utils module
from utils.chunk_loader import ChunkMetadataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreloadedRetriever:
    """
    Retriever that uses pre-existing FAISS index and chunk metadata
    """
    
    def __init__(self, 
                 metadata_path: str = "chunk_metadata/chunk_metadata.json",
                 faiss_path: str = "chunk_metadata/rag_index.faiss",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize retriever with pre-loaded data
        
        Args:
            metadata_path: Path to chunk metadata JSON
            faiss_path: Path to FAISS index
            embedding_model: Name of sentence-transformer model (should match the one used to create embeddings)
        """
        self.metadata_path = metadata_path
        self.faiss_path = faiss_path
        self.embedding_model_name = embedding_model
        
        # Load chunks and index first to get metadata
        logger.info("Loading pre-existing chunks and FAISS index...")
        self.loader = ChunkMetadataLoader(metadata_path, faiss_path)
        self.loader.load()
        
        # Validate embedding model matches the one used to create the index
        stored_model = self.loader.metadata.get('embedding_model', '')
        if stored_model and embedding_model not in stored_model and stored_model not in embedding_model:
            logger.warning(
                f"Embedding model mismatch: requested '{embedding_model}' but index was created with '{stored_model}'. "
                f"This may cause poor retrieval results."
            )
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"✅ PreloadedRetriever initialized with {len(self.loader.chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of chunk dictionaries with scores and metadata
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        # Search
        results = self.loader.search(query_embedding, top_k)
        
        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Get a specific chunk by ID"""
        for chunk in self.loader.chunks:
            if chunk['chunk_id'] == chunk_id:
                return chunk
        return None
    
    def get_stats(self) -> Dict:
        """Get statistics about the loaded data"""
        return {
            'total_chunks': len(self.loader.chunks),
            'embedding_model': self.loader.metadata['embedding_model'],
            'embedding_dimension': self.loader.metadata['embedding_dimension'],
            'chunk_size': self.loader.metadata['chunk_size'],
            'chunk_overlap': self.loader.metadata['chunk_overlap'],
            'faiss_vectors': self.loader.index.ntotal
        }


# Example usage
if __name__ == "__main__":
    # Initialize retriever
    retriever = PreloadedRetriever(
        metadata_path="chunk_metadata/chunk_metadata.json",
        faiss_path="chunk_metadata/rag_index.faiss"
    )
    
    # Print stats
    stats = retriever.get_stats()
    print("\nRetriever Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test retrieval
    test_queries = [
        "What are the credit card fees?",
        "How to apply for a credit card?",
        "What is the interest rate?"
    ]
    
    print("\n" + "="*70)
    print("TEST RETRIEVAL")
    print("="*70)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=3)
        
        for result in results:
            print(f"\n  Rank {result['rank']}: Score={result['score']:.3f}")
            print(f"  Doc: {result['metadata']['doc_name']} | Type: {result['metadata']['document_type']}")
            print(f"  Content: {result['content'][:150]}...")
