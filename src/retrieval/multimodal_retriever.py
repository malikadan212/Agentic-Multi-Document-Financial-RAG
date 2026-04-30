# src/retrieval/multimodal_retriever.py
"""
Multimodal Retrieval System with CLIP Embeddings
Enables text-to-image and image-to-image similarity search for Financial RAG
Uses OpenCLIP for free, local image/text embeddings
"""

# Import necessary libraries
import numpy as np  # For numerical operations
from typing import List, Dict, Optional, Union, Tuple  # For type hints
from dataclasses import dataclass  # For data classes
import logging  # For logging functionality
from pathlib import Path  # For path handling
import io  # For byte stream handling
import base64  # For encoding images

# Try to import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try to import OpenCLIP for multimodal embeddings
try:
    import open_clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Data class for multimodal retrieval results
@dataclass
class MultimodalResult:
    """
    Represents a multimodal retrieval result (text or image)
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        content: Text content (if text) or description (if image)
        score: Similarity score
        metadata: Source information
        rank: Position in retrieval results
        is_image: Whether this result is an image
        image_bytes: Raw image bytes (if image result)
    """
    chunk_id: str
    content: str  # Text content or image description
    score: float
    metadata: Dict
    rank: int
    is_image: bool = False
    image_bytes: Optional[bytes] = None
    
    def __repr__(self):
        type_str = "Image" if self.is_image else "Text"
        return f"MultimodalResult({type_str}, {self.chunk_id}, score={self.score:.3f})"
    
    def get_base64_image(self) -> Optional[str]:
        """Get base64 encoded image for API calls"""
        if self.image_bytes:
            return base64.b64encode(self.image_bytes).decode('utf-8')
        return None


# CLIP Embedding Model for multimodal embeddings
class CLIPEmbedding:
    """
    CLIP-based embedding model for text and images
    Uses OpenCLIP for free, local multimodal embeddings
    """
    
    # Available CLIP models (name -> (model_name, pretrained_weights))
    MODELS = {
        'vit-b-32': ('ViT-B-32', 'laion2b_s34b_b79k'),  # Balanced model
        'vit-b-16': ('ViT-B-16', 'laion2b_s34b_b88k'),  # Higher quality
        'vit-l-14': ('ViT-L-14', 'laion2b_s32b_b82k'),  # Best quality, slower
    }
    
    def __init__(self, model_name: str = 'vit-b-32', device: str = None):
        """
        Initialize CLIP embedding model
        
        Args:
            model_name: Name of CLIP model variant
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        if not CLIP_AVAILABLE:
            raise ImportError(
                "OpenCLIP not available. Install with: pip install open-clip-torch torch"
            )
        
        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL not available. Install with: pip install Pillow"
            )
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Get model configuration
        if model_name not in self.MODELS:
            logger.warning(f"Model {model_name} not found, using vit-b-32")
            model_name = 'vit-b-32'
        
        model_arch, pretrained = self.MODELS[model_name]
        
        logger.info(f"Loading CLIP model: {model_arch} on {self.device}")
        
        try:
            # Load CLIP model, preprocess, and tokenizer
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_arch, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_arch)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Get embedding dimension
            self.dimension = self.model.visual.output_dim
            
            logger.info(f"✅ CLIP model loaded. Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def encode_text(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode text strings into embeddings
        
        Args:
            texts: List of text strings to encode
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        with torch.no_grad():
            # Tokenize texts
            tokens = self.tokenizer(texts).to(self.device)
            
            # Get embeddings
            text_features = self.model.encode_text(tokens)
            
            # Normalize if requested
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy()
    
    def encode_images(self, images: List[Union[Image.Image, bytes, str]], 
                      normalize: bool = True) -> np.ndarray:
        """
        Encode images into embeddings
        
        Args:
            images: List of PIL Images, image bytes, or file paths
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of shape (num_images, embedding_dim)
        """
        if not images:
            return np.array([])
        
        processed_images = []
        
        for img in images:
            # Handle different input types
            if isinstance(img, bytes):
                pil_img = Image.open(io.BytesIO(img))
            elif isinstance(img, str):
                pil_img = Image.open(img)
            elif isinstance(img, Image.Image):
                pil_img = img
            else:
                logger.warning(f"Unknown image type: {type(img)}, skipping")
                continue
            
            # Ensure RGB
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Apply preprocessing
            processed = self.preprocess(pil_img).unsqueeze(0)
            processed_images.append(processed)
        
        if not processed_images:
            return np.array([])
        
        with torch.no_grad():
            # Stack all images
            image_batch = torch.cat(processed_images, dim=0).to(self.device)
            
            # Get embeddings
            image_features = self.model.encode_image(image_batch)
            
            # Normalize if requested
            if normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                           candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and candidates
        
        Args:
            query_embedding: Query embedding (1, dim) or (dim,)
            candidate_embeddings: Candidate embeddings (n, dim)
            
        Returns:
            Similarity scores (n,)
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute dot product (embeddings are L2-normalized)
        similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()
        
        return similarities


# Multimodal Retriever combining text and image search
class MultimodalRetriever:
    """
    Multimodal retriever that can search both text and image chunks
    Combines traditional text embeddings with CLIP for cross-modal retrieval
    """
    
    def __init__(self, 
                 clip_model: str = 'vit-b-32',
                 text_weight: float = 0.6,
                 image_weight: float = 0.4):
        """
        Initialize multimodal retriever
        
        Args:
            clip_model: CLIP model variant to use
            text_weight: Weight for text results in hybrid search
            image_weight: Weight for image results in hybrid search
        """
        self.text_weight = text_weight
        self.image_weight = image_weight
        
        # Initialize CLIP for image/cross-modal embeddings
        try:
            self.clip = CLIPEmbedding(model_name=clip_model)
            self.clip_available = True
            logger.info("✅ CLIP model initialized for multimodal retrieval")
        except Exception as e:
            logger.warning(f"CLIP not available: {e}. Image search disabled.")
            self.clip = None
            self.clip_available = False
        
        # Storage for indexed content
        self.text_chunks = []  # List of text DocumentChunks
        self.image_chunks = []  # List of ImageChunks
        self.text_embeddings = None  # Text CLIP embeddings (for cross-modal)
        self.image_embeddings = None  # Image CLIP embeddings
    
    def index_text_chunks(self, chunks: List) -> None:
        """
        Index text chunks with CLIP embeddings for cross-modal search
        
        Args:
            chunks: List of DocumentChunk objects
        """
        if not self.clip_available:
            logger.warning("CLIP not available, text indexing skipped")
            return
        
        self.text_chunks = chunks
        
        # Extract text content
        texts = [chunk.content for chunk in chunks]
        
        if texts:
            logger.info(f"Generating CLIP embeddings for {len(texts)} text chunks...")
            self.text_embeddings = self.clip.encode_text(texts)
            logger.info(f"✅ Indexed {len(texts)} text chunks with CLIP")
    
    def index_image_chunks(self, image_chunks: List) -> None:
        """
        Index image chunks with CLIP embeddings
        
        Args:
            image_chunks: List of ImageChunk objects
        """
        if not self.clip_available:
            logger.warning("CLIP not available, image indexing skipped")
            return
        
        self.image_chunks = image_chunks
        
        # Extract images
        images = [chunk.image for chunk in image_chunks if chunk.image is not None]
        
        if images:
            logger.info(f"Generating CLIP embeddings for {len(images)} images...")
            self.image_embeddings = self.clip.encode_images(images)
            logger.info(f"✅ Indexed {len(images)} image chunks with CLIP")
    
    def retrieve_by_text(self, query: str, top_k: int = 5, 
                         include_images: bool = True) -> List[MultimodalResult]:
        """
        Retrieve relevant content using text query
        Can retrieve both text and image results
        
        Args:
            query: Text query string
            top_k: Number of results to return
            include_images: Whether to include image results
            
        Returns:
            List of MultimodalResult objects
        """
        if not self.clip_available:
            logger.warning("CLIP not available for retrieval")
            return []
        
        results = []
        
        # Encode query
        query_embedding = self.clip.encode_text([query])
        
        # Search text chunks
        if self.text_embeddings is not None and len(self.text_embeddings) > 0:
            text_scores = self.clip.compute_similarity(query_embedding, self.text_embeddings)
            
            # Get top text results
            top_text_idx = np.argsort(text_scores)[::-1][:top_k]
            
            for rank, idx in enumerate(top_text_idx, start=1):
                chunk = self.text_chunks[idx]
                results.append(MultimodalResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=float(text_scores[idx]) * self.text_weight,
                    metadata=chunk.metadata,
                    rank=rank,
                    is_image=False
                ))
        
        # Search image chunks
        if include_images and self.image_embeddings is not None and len(self.image_embeddings) > 0:
            image_scores = self.clip.compute_similarity(query_embedding, self.image_embeddings)
            
            # Get top image results
            top_img_idx = np.argsort(image_scores)[::-1][:top_k]
            
            for rank, idx in enumerate(top_img_idx, start=1):
                chunk = self.image_chunks[idx]
                results.append(MultimodalResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.description or f"Image from {chunk.metadata.get('doc_name', 'unknown')}",
                    score=float(image_scores[idx]) * self.image_weight,
                    metadata=chunk.metadata,
                    rank=rank,
                    is_image=True,
                    image_bytes=chunk.image_bytes
                ))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Re-rank
        for rank, result in enumerate(results[:top_k], start=1):
            result.rank = rank
        
        return results[:top_k]
    
    def retrieve_by_image(self, image: Union[Image.Image, bytes, str], 
                          top_k: int = 5) -> List[MultimodalResult]:
        """
        Retrieve relevant content using image query
        
        Args:
            image: Query image (PIL Image, bytes, or path)
            top_k: Number of results to return
            
        Returns:
            List of MultimodalResult objects
        """
        if not self.clip_available:
            logger.warning("CLIP not available for image retrieval")
            return []
        
        results = []
        
        # Encode query image
        query_embedding = self.clip.encode_images([image])
        
        # Search text chunks (find text describing similar content)
        if self.text_embeddings is not None and len(self.text_embeddings) > 0:
            text_scores = self.clip.compute_similarity(query_embedding, self.text_embeddings)
            
            top_text_idx = np.argsort(text_scores)[::-1][:top_k]
            
            for rank, idx in enumerate(top_text_idx, start=1):
                chunk = self.text_chunks[idx]
                results.append(MultimodalResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=float(text_scores[idx]) * self.text_weight,
                    metadata=chunk.metadata,
                    rank=rank,
                    is_image=False
                ))
        
        # Search image chunks
        if self.image_embeddings is not None and len(self.image_embeddings) > 0:
            image_scores = self.clip.compute_similarity(query_embedding, self.image_embeddings)
            
            top_img_idx = np.argsort(image_scores)[::-1][:top_k]
            
            for rank, idx in enumerate(top_img_idx, start=1):
                chunk = self.image_chunks[idx]
                results.append(MultimodalResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.description or f"Image from {chunk.metadata.get('doc_name', 'unknown')}",
                    score=float(image_scores[idx]) * self.image_weight,
                    metadata=chunk.metadata,
                    rank=rank,
                    is_image=True,
                    image_bytes=chunk.image_bytes
                ))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        
        for rank, result in enumerate(results[:top_k], start=1):
            result.rank = rank
        
        return results[:top_k]
    
    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        return {
            'clip_available': self.clip_available,
            'text_chunks': len(self.text_chunks),
            'image_chunks': len(self.image_chunks),
            'text_embeddings_shape': self.text_embeddings.shape if self.text_embeddings is not None else None,
            'image_embeddings_shape': self.image_embeddings.shape if self.image_embeddings is not None else None,
            'clip_dimension': self.clip.dimension if self.clip else None
        }


# Helper function to convert image to base64 for API calls
def image_to_base64(image: Union[Image.Image, bytes, str]) -> str:
    """
    Convert image to base64 string for API calls
    
    Args:
        image: PIL Image, bytes, or file path
        
    Returns:
        Base64 encoded string
    """
    if isinstance(image, bytes):
        return base64.b64encode(image).decode('utf-8')
    elif isinstance(image, str):
        with open(image, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


# Example usage
if __name__ == "__main__":
    # Test CLIP embeddings
    print("Testing CLIP Embedding...")
    
    try:
        clip = CLIPEmbedding()
        
        # Test text encoding
        texts = ["A bar chart showing revenue growth", "Financial statement table"]
        text_embeddings = clip.encode_text(texts)
        print(f"Text embeddings shape: {text_embeddings.shape}")
        
        # Test similarity
        query = clip.encode_text(["revenue chart"])
        similarity = clip.compute_similarity(query, text_embeddings)
        print(f"Similarities: {similarity}")
        
        print("✅ CLIP embedding test passed!")
        
    except Exception as e:
        print(f"❌ CLIP test failed: {e}")
