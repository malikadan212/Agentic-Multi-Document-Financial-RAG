# src/utils/chunk_loader.py
"""
Utility to load pre-existing chunk metadata and FAISS index
Adapts the existing chunk_metadata format to work with the RAG system
"""

# Import necessary libraries
import json  # For reading JSON files
import faiss  # For working with FAISS vector index
import numpy as np  # For numerical operations and array handling
from typing import List, Dict  # For type hints
from pathlib import Path  # For path manipulation
import logging  # For logging functionality

# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO)
# Create logger instance for this module
logger = logging.getLogger(__name__)


# Define class to handle loading of pre-processed chunk metadata and FAISS index
class ChunkMetadataLoader:
    """
    Loads pre-processed chunk metadata and FAISS index
    """
    
    # Initialize the loader with file paths
    def __init__(self, metadata_path: str, faiss_path: str):
        """
        Initialize loader with paths to metadata and FAISS index
        
        Args:
            metadata_path: Path to chunk_metadata.json
            faiss_path: Path to rag_index.faiss
        """
        # Convert string paths to Path objects for better path handling
        self.metadata_path = Path(metadata_path)
        self.faiss_path = Path(faiss_path)
        # Initialize empty list to store chunk data
        self.chunks = []
        # Initialize index as None (will hold FAISS index if loaded)
        self.index = None
        # Initialize empty dictionary to store global metadata
        self.metadata = {}
        
    # Main loading method
    def load(self, load_faiss: bool = True):
        """
        Load both metadata and FAISS index
        
        Args:
            load_faiss: If False, skip loading FAISS index (useful for metadata-only analysis)
        """
        # Log start of metadata loading
        logger.info("Loading chunk metadata...")
        # Call internal method to load metadata
        self._load_metadata()
        
        # Conditionally load FAISS index
        if load_faiss:
            # Log start of FAISS loading
            logger.info("Loading FAISS index...")
            try:
                # Attempt to load FAISS index
                self._load_faiss_index()
                # Log success with counts
                logger.info(f"✅ Loaded {len(self.chunks)} chunks and FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                # Log warning if FAISS loading fails
                logger.warning(f"Could not load FAISS index: {e}")
                # Still log success for metadata loading
                logger.info(f"✅ Loaded {len(self.chunks)} chunks (FAISS index not available)")
        else:
            # Log metadata-only loading
            logger.info(f"✅ Loaded {len(self.chunks)} chunks (FAISS index skipped)")
        
    # Internal method to load metadata from JSON file
    def _load_metadata(self):
        """Load chunk metadata from JSON file"""
        # Open and read JSON file with UTF-8 encoding
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            # Parse JSON content into Python data structure
            data = json.load(f)
        
        # Extract global metadata into structured dictionary
        self.metadata = {
            'created_at': data.get('created_at'),  # Timestamp of creation
            'embedding_model': data.get('embedding_model'),  # Model used for embeddings
            'embedding_dimension': data.get('embedding_dimension'),  # Dimension of embeddings
            'chunk_size': data.get('chunk_size'),  # Size of text chunks
            'chunk_overlap': data.get('chunk_overlap'),  # Overlap between chunks
            'total_vectors': data.get('total_vectors')  # Total number of vectors
        }
        
        # Process each record in the JSON data
        for record in data['records']:
            # Transform raw record into standardized chunk format
            chunk = {
                'chunk_id': record['chunk_id'],  # Unique identifier for chunk
                'content': record['text'],  # The actual text content
                'metadata': {  # Structured metadata dictionary
                    'doc_name': record['metadata'].get('bank_name', 'Unknown'),  # Document name or default
                    'page': record['metadata'].get('page_number'),  # Page number if available
                    'document_type': record['metadata'].get('document_type', 'other'),  # Type of document
                    'product_type': record['metadata'].get('product_type', 'other'),  # Product category
                    'section': record['metadata'].get('section'),  # Section within document
                    'vector_id': record['vector_id']  # Corresponding vector ID in FAISS index
                }
            }
            # Add chunk to the list
            self.chunks.append(chunk)
    
    # Internal method to load FAISS index from file
    def _load_faiss_index(self):
        """Load FAISS index from file"""
        # Read FAISS index file (convert Path to string for FAISS API)
        self.index = faiss.read_index(str(self.faiss_path))
        
        # Check for consistency between metadata and index
        if self.index.ntotal != len(self.chunks):
            # Log warning if counts don't match
            logger.warning(
                f"Mismatch: FAISS has {self.index.ntotal} vectors but metadata has {len(self.chunks)} chunks"
            )
    
    # Getter method to retrieve all chunks
    def get_chunks(self) -> List[Dict]:
        """Get all chunks"""
        return self.chunks
    
    # Getter method to retrieve FAISS index
    def get_index(self):
        """Get FAISS index"""
        return self.index
    
    # Getter method to retrieve global metadata
    def get_metadata(self) -> Dict:
        """Get global metadata"""
        return self.metadata
    
    # Search method to find similar chunks using FAISS
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query vector (1D or 2D array)
            top_k: Number of results to return
            
        Returns:
            List of chunk dictionaries with scores
        """
        # Reshape if query is 1D array (single query)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure query is float32 (FAISS requirement)
        query_embedding = query_embedding.astype('float32')
        
        # Search FAISS index for top_k nearest neighbors
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Build results list
        results = []
        # Iterate through search results
        for idx, (vector_idx, distance) in enumerate(zip(indices[0], distances[0])):
            # Ensure index is within bounds
            if vector_idx < len(self.chunks):
                # Create copy of chunk to avoid modifying original
                chunk = self.chunks[vector_idx].copy()
                # Add score (distance metric - higher is better for IndexFlatIP)
                chunk['score'] = float(distance)
                # Add rank (1-based)
                chunk['rank'] = idx + 1
                # Add to results
                results.append(chunk)
        
        return results


# Example usage when script is run directly
if __name__ == "__main__":
    # Create loader instance with example paths
    loader = ChunkMetadataLoader(
        metadata_path="chunk_metadata/chunk_metadata.json",
        faiss_path="chunk_metadata/rag_index.faiss"
    )
    
    # Load data and index
    loader.load()
    
    # Print basic information
    print(f"Loaded {len(loader.get_chunks())} chunks")
    print(f"Metadata: {loader.get_metadata()}")
    
    # Test search with random vector (384-dimensional, matching typical embedding size)
    test_query = np.random.randn(384).astype('float32')
    # Perform search for top 3 results
    results = loader.search(test_query, top_k=3)
    
    # Display search results
    print("\nSample search results:")
    for result in results:
        # Print rank and score
        print(f"  Rank {result['rank']}: Score={result['score']:.3f}")
        # Print first 100 characters of content
        print(f"  Content: {result['content'][:100]}...")