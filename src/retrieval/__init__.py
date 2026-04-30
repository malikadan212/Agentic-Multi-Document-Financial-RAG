"""
Retrieval Module for Financial RAG System
Advanced Retrieval System with Multiple Embedding Models
"""

from .retriever import (
    RetrievalResult,
    EmbeddingModel,
    FAISSVectorStore,
    ChromaVectorStore,
    HybridRetriever
)

from .preloaded_retriever import PreloadedRetriever

# Multimodal components (optional, may not be available)
try:
    from .multimodal_retriever import (
        MultimodalResult,
        CLIPEmbedding,
        MultimodalRetriever,
        image_to_base64
    )
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

__all__ = [
    'RetrievalResult',
    'EmbeddingModel',
    'FAISSVectorStore',
    'ChromaVectorStore',
    'HybridRetriever',
    'PreloadedRetriever',
    'MULTIMODAL_AVAILABLE'
]

# Add multimodal exports if available
if MULTIMODAL_AVAILABLE:
    __all__.extend([
        'MultimodalResult',
        'CLIPEmbedding',
        'MultimodalRetriever',
        'image_to_base64'
    ])
