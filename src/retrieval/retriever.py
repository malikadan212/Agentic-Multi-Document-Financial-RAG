# src/retrieval/retriever.py
"""
Advanced Retrieval System with Multiple Embedding Models
Implements semantic search with FAISS and ChromaDB
"""

# Import necessary libraries
import numpy as np  # For numerical operations and array handling
from sentence_transformers import SentenceTransformer  # For text embeddings
from typing import List, Dict, Tuple, Optional  # For type hints
import faiss  # For efficient similarity search
import chromadb  # For vector database with persistence
import pickle  # For serializing Python objects
from pathlib import Path  # For path manipulation
import logging  # For logging functionality
from dataclasses import dataclass  # For creating data classes

# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO)
# Create logger instance for this module
logger = logging.getLogger(__name__)


# Define a data class for retrieval results
@dataclass
class RetrievalResult:
    """Represents a retrieval result with score and metadata"""
    chunk_id: str  # Unique identifier for the chunk
    content: str  # The text content of the chunk
    score: float  # Similarity score (higher is better)
    metadata: Dict  # Additional metadata about the chunk
    rank: int  # Rank position in search results
    
    # String representation method for easy debugging
    def __repr__(self):
        return f"Result(rank={self.rank}, score={self.score:.3f}, doc={self.metadata.get('doc_name')})"


# Class for handling different embedding models
class EmbeddingModel:
    """
    Wrapper for different embedding models
    Supports multiple sentence-transformer models for comparison
    """
    
    # Dictionary mapping model keys to actual model names
    SUPPORTED_MODELS = {
        'minilm': 'all-MiniLM-L6-v2',          # Fast model with 384 dimensions
        'mpnet': 'all-mpnet-base-v2',           # Balanced model with 768 dimensions
        'distilbert': 'msmarco-distilbert-base-v4',  # Search-optimized model with 768 dimensions
        'roberta': 'all-roberta-large-v1'       # High-quality model with 1024 dimensions
    }
    
    # Initialize the embedding model
    def __init__(self, model_name: str = 'minilm'):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the model (from SUPPORTED_MODELS)
        """
        # Validate that the requested model is supported
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(self.SUPPORTED_MODELS.keys())}")
        
        # Store model configuration
        self.model_key = model_name  # User-friendly key (e.g., 'minilm')
        self.model_path = self.SUPPORTED_MODELS[model_name]  # Actual model name for SentenceTransformer
        # Log model loading
        logger.info(f"Loading embedding model: {self.model_path}")
        # Load the sentence transformer model
        self.model = SentenceTransformer(self.model_path)
        # Get the dimension of embeddings from the model
        self.dimension = self.model.get_sentence_embedding_dimension()
        # Log successful loading
        logger.info(f"✅ Model loaded: {self.dimension} dimensions")
    
    # Method to encode texts into embeddings
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: List of text strings to encode
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        # Generate embeddings using the loaded model
        embeddings = self.model.encode(
            texts,  # Input texts
            show_progress_bar=show_progress,  # Show/hide progress bar
            convert_to_numpy=True,  # Return numpy array
            normalize_embeddings=True  # Apply L2 normalization for cosine similarity
        )
        return embeddings


# Class for FAISS-based vector storage and retrieval
class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search
    Implements both flat and IVF indices
    """
    
    # Initialize FAISS index
    def __init__(self, dimension: int, use_ivf: bool = False, n_clusters: int = 100):
        """
        Initialize FAISS index
        
        Args:
            dimension: Embedding dimension
            use_ivf: Whether to use IVF (Inverted File) index for large datasets
            n_clusters: Number of clusters for IVF
        """
        # Store configuration
        self.dimension = dimension  # Dimension of embeddings
        self.use_ivf = use_ivf  # Whether to use IVF indexing
        
        # Create appropriate FAISS index based on configuration
        if use_ivf:
            # IVF index for large-scale retrieval (faster but approximate)
            quantizer = faiss.IndexFlatL2(dimension)  # L2 distance quantizer
            self.index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)  # IVF index with clusters
            self.is_trained = False  # IVF index needs training
        else:
            # Flat index for smaller datasets (exact but slower)
            self.index = faiss.IndexFlatL2(dimension)  # Simple L2 distance index
            self.is_trained = True  # Flat index doesn't need training
        
        # Initialize storage for chunks and embeddings
        self.chunks = []  # Store original DocumentChunk objects
        self.embeddings = None  # Store embeddings as numpy array
    
    # Method to add documents to the index
    def add_documents(self, chunks: List, embeddings: np.ndarray):
        """
        Add document chunks and their embeddings to the index
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: numpy array of embeddings
        """
        # Validate that chunks and embeddings have same length
        assert len(chunks) == len(embeddings), "Chunks and embeddings must have same length"
        
        # Train IVF index if necessary (first time adding documents)
        if self.use_ivf and not self.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)  # Train on the embeddings
            self.is_trained = True  # Mark as trained
        
        # Add embeddings to the index (converted to float32 for FAISS)
        self.index.add(embeddings.astype('float32'))
        # Store chunks and embeddings
        self.chunks.extend(chunks)
        self.embeddings = embeddings
        
        # Log successful addition
        logger.info(f"✅ Added {len(chunks)} documents to FAISS index")
    
    # Method to search for similar documents
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (index, distance) tuples
        """
        # Reshape query to 2D array (1 x dimension) and ensure float32 type
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        # Search the index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return list of (index, distance) pairs
        return list(zip(indices[0], distances[0]))
    
    # Method to save the index to disk
    def save(self, path: str):
        """Save index and chunks to disk"""
        # Create parent directories if they don't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index to file
        faiss.write_index(self.index, f"{path}.index")
        
        # Save chunks and metadata using pickle
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,  # Document chunks
                'embeddings': self.embeddings,  # Embedding vectors
                'dimension': self.dimension,  # Embedding dimension
                'use_ivf': self.use_ivf  # Whether IVF was used
            }, f)
        
        # Log successful save
        logger.info(f"✅ Saved FAISS index to {path}")
    
    # Class method to load index from disk
    @classmethod
    def load(cls, path: str):
        """Load index and chunks from disk"""
        # Load FAISS index from file
        index = faiss.read_index(f"{path}.index")
        
        # Load chunks and metadata from pickle file
        with open(f"{path}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct the FAISSVectorStore object
        store = cls(data['dimension'], data['use_ivf'])  # Create new instance
        store.index = index  # Restore FAISS index
        store.chunks = data['chunks']  # Restore chunks
        store.embeddings = data['embeddings']  # Restore embeddings
        store.is_trained = True  # Mark as trained (already trained if saved)
        
        # Log successful load
        logger.info(f"✅ Loaded FAISS index from {path}")
        return store


# Class for ChromaDB-based vector storage and retrieval
class ChromaVectorStore:
    """
    ChromaDB-based vector store
    Alternative to FAISS with built-in persistence
    """
    
    # Initialize ChromaDB client and collection
    def __init__(self, collection_name: str = "financial_docs", persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client and collection
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        # Try to create persistent client (saves to disk)
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
        except (OSError, PermissionError) as e:
            # Fallback to in-memory client if persistence fails
            logger.warning(f"Could not create persistent client ({e}), using in-memory client")
            self.client = chromadb.Client()  # In-memory client
        except Exception as e:
            # Catch any ChromaDB-specific errors
            logger.warning(f"ChromaDB error ({type(e).__name__}: {e}), using in-memory client")
            self.client = chromadb.Client()  # In-memory client
        
        # Get existing collection or create new one
        self.collection = self.client.get_or_create_collection(
            name=collection_name,  # Collection name
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for search
        )
        
        # Log successful initialization
        logger.info(f"✅ ChromaDB initialized: {collection_name}")
    
    # Method to add documents to ChromaDB
    def add_documents(self, chunks: List, embeddings: np.ndarray):
        """
        Add documents to ChromaDB
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: numpy array of embeddings
        """
        # Prepare data in format required by ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]  # Unique IDs
        documents = [chunk.content for chunk in chunks]  # Text content
        metadatas = [chunk.metadata for chunk in chunks]  # Metadata dictionaries
        embeddings_list = embeddings.tolist()  # Convert numpy array to list of lists
        
        # Add to ChromaDB collection
        self.collection.add(
            ids=ids,  # Document IDs
            documents=documents,  # Document texts
            embeddings=embeddings_list,  # Embedding vectors
            metadatas=metadatas  # Metadata
        )
        
        # Log successful addition
        logger.info(f"✅ Added {len(chunks)} documents to ChromaDB")
    
    # Method to search ChromaDB
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Search ChromaDB with optional metadata filtering
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            metadata_filter: Optional dictionary for metadata filtering
            
        Returns:
            List of result dictionaries
        """
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],  # Query embedding (as list)
            n_results=top_k,  # Number of results to return
            where=metadata_filter  # Optional metadata filter
        )
        
        return results


# Main retriever class that combines embedding model and vector store
class HybridRetriever:
    """
    Advanced retrieval system supporting multiple strategies
    Combines dense retrieval with optional filtering
    """
    
    # Initialize the hybrid retriever
    def __init__(self, 
                 embedding_model: str = 'minilm',
                 vector_store_type: str = 'faiss',
                 top_k: int = 5):
        """
        Initialize hybrid retriever
        
        Args:
            embedding_model: Name of embedding model
            vector_store_type: 'faiss' or 'chroma'
            top_k: Default number of results to retrieve
        """
        # Initialize embedding model
        self.embedding_model = EmbeddingModel(embedding_model)
        # Store configuration
        self.vector_store_type = vector_store_type
        self.top_k = top_k
        # Vector store will be initialized later when documents are indexed
        self.vector_store = None
        
        # Log successful initialization
        logger.info(f"✅ HybridRetriever initialized: {embedding_model}, {vector_store_type}")
    
    # Method to index document chunks
    def index_documents(self, chunks: List):
        """
        Index document chunks
        
        Args:
            chunks: List of DocumentChunk objects
        """
        # Log start of indexing
        logger.info(f"Indexing {len(chunks)} document chunks...")
        
        # Generate embeddings for all chunk texts
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # Create appropriate vector store based on configuration
        if self.vector_store_type == 'faiss':
            self.vector_store = FAISSVectorStore(
                dimension=self.embedding_model.dimension,  # Embedding dimension
                use_ivf=(len(chunks) > 10000)  # Use IVF only for large datasets (>10k chunks)
            )
        elif self.vector_store_type == 'chroma':
            self.vector_store = ChromaVectorStore()  # ChromaDB with default settings
        else:
            raise ValueError(f"Unknown vector store type: {self.vector_store_type}")
        
        # Add documents to the vector store
        self.vector_store.add_documents(chunks, embeddings)
        # Log successful indexing
        logger.info(f"✅ Indexed {len(chunks)} chunks")
    
    # Main retrieval method
    def retrieve(self, 
                 query: str, 
                 top_k: Optional[int] = None,
                 metadata_filter: Optional[Dict] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query string
            top_k: Number of results (uses default if None)
            metadata_filter: Optional metadata filtering
            
        Returns:
            List of RetrievalResult objects
        """
        # Check if documents have been indexed
        if self.vector_store is None:
            raise ValueError("No documents indexed. Call index_documents() first.")
        
        # Use provided top_k or default value
        k = top_k or self.top_k
        
        # Encode the query into an embedding vector
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Route to appropriate retrieval method based on vector store type
        if isinstance(self.vector_store, FAISSVectorStore):
            results = self._retrieve_faiss(query_embedding, k)
        else:
            results = self._retrieve_chroma(query_embedding, k, metadata_filter)
        
        return results
    
    # Private method for FAISS retrieval
    def _retrieve_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievalResult]:
        """Retrieve from FAISS index"""
        # Search FAISS index
        search_results = self.vector_store.search(query_embedding, top_k)
        
        # Process results into RetrievalResult objects
        results = []
        for rank, (idx, distance) in enumerate(search_results, start=1):
            # Get the corresponding chunk
            chunk = self.vector_store.chunks[idx]
            
            # Convert L2 distance to similarity score (0-1 range)
            # Formula: similarity = 1 / (1 + distance)
            similarity = 1 / (1 + distance)
            
            # Create RetrievalResult object
            result = RetrievalResult(
                chunk_id=chunk.chunk_id,  # Chunk identifier
                content=chunk.content,  # Text content
                score=similarity,  # Similarity score
                metadata=chunk.metadata,  # Metadata
                rank=rank  # Rank position
            )
            results.append(result)
        
        return results
    
    # Private method for ChromaDB retrieval
    def _retrieve_chroma(self, query_embedding: np.ndarray, top_k: int, 
                        metadata_filter: Optional[Dict]) -> List[RetrievalResult]:
        """Retrieve from ChromaDB"""
        # Search ChromaDB with optional metadata filter
        search_results = self.vector_store.search(query_embedding, top_k, metadata_filter)
        
        # Process results into RetrievalResult objects
        results = []
        for rank in range(len(search_results['ids'][0])):  # Iterate through results
            result = RetrievalResult(
                chunk_id=search_results['ids'][0][rank],  # Chunk ID from ChromaDB
                content=search_results['documents'][0][rank],  # Text content
                # Convert cosine distance to similarity (distance = 1 - similarity)
                score=1 - search_results['distances'][0][rank],
                metadata=search_results['metadatas'][0][rank],  # Metadata
                rank=rank + 1  # Rank (1-indexed)
            )
            results.append(result)
        
        return results
    
    # Method to save retriever state (FAISS only)
    def save(self, path: str):
        """Save retriever state"""
        # Only FAISSVectorStore supports saving
        if isinstance(self.vector_store, FAISSVectorStore):
            self.vector_store.save(path)
    
    # Class method to load retriever from saved state
    @classmethod
    def load(cls, path: str, embedding_model: str = 'minilm'):
        """Load retriever from saved state"""
        # Create new retriever instance
        retriever = cls(embedding_model=embedding_model, vector_store_type='faiss')
        # Load FAISS vector store from disk
        retriever.vector_store = FAISSVectorStore.load(path)
        return retriever


# Example Usage
if __name__ == "__main__":
    # Import DocumentPipeline for processing documents
    from document_processing.processor import DocumentPipeline
    
    # Process documents from directory
    pipeline = DocumentPipeline()
    chunks = pipeline.process_directory("data/raw/pdfs")
    
    # Create retriever with MiniLM embeddings and FAISS storage
    retriever = HybridRetriever(embedding_model='minilm', vector_store_type='faiss')
    # Index the processed chunks
    retriever.index_documents(chunks)
    
    # Test retrieval with a sample query
    query = "What was the total revenue in Q3 2023?"
    results = retriever.retrieve(query, top_k=5)
    
    # Display results
    print(f"\nQuery: {query}\n")
    for result in results:
        print(f"{result}")  # Use __repr__ method
        print(f"Content: {result.content[:200]}...\n")  # Show first 200 characters