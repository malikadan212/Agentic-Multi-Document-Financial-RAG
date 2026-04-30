# src/generation/simple_generator.py
"""
Simplified Generation System using only Groq (FREE API)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os
import logging
import re
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for generation parameters"""
    temperature: float = 0.1
    max_tokens: int = 1000
    top_p: float = 0.9


@dataclass
class GeneratedResponse:
    """Represents a generated response with citations"""
    answer: str
    citations: List[Dict]
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_cost: float = 0.0


class SimpleRAGGenerator:
    """
    Simplified RAG Generator using only Groq API (FREE)
    """
    
    def __init__(self, 
                 model_name: str = 'llama-3.1-8b-instant',
                 config: GenerationConfig = None):
        """
        Initialize generator with Groq
        
        Args:
            model_name: Groq model name
            config: Generation configuration
        """
        self.model_name = model_name
        self.config = config or GenerationConfig()
        
        # Initialize Groq client
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment. Please set it in .env file")
        
        self.client = Groq(api_key=api_key)
        logger.info(f"✅ SimpleRAGGenerator initialized with Groq: {model_name}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict]:
        """Generate response using Groq API"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )
            
            answer = response.choices[0].message.content
            
            # Get usage info
            usage_raw = getattr(response, "usage", None)
            if usage_raw:
                prompt_tokens = getattr(usage_raw, "prompt_tokens", 0)
                completion_tokens = getattr(usage_raw, "completion_tokens", 0)
            else:
                prompt_tokens = 0
                completion_tokens = 0
            
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": 0.0  # Groq is free!
            }
            
            return answer, usage
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise
    
    def extract_citations(self, response: str, retrieved_chunks: List) -> List[Dict]:
        """Extract and validate citations from response"""
        citations = []
        
        # Multiple patterns to catch different citation formats
        patterns = [
            r'\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]',  # [Source: DocName, Page X]
            r'\[Source:\s*([^,\]]+)(?:,\s*Page\s*(\d+))?\]',  # [Source: DocName] or [Source: DocName, Page X]
            r'\(Source:\s*([^,\)]+)(?:,\s*Page\s*(\d+))?\)',  # (Source: DocName) or (Source: DocName, Page X)
            r'\[([^,\]]+),\s*Page\s*(\d+)\]',  # [DocName, Page X]
        ]
        
        seen_citations = set()  # Avoid duplicates

        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            
            for match in matches:
                doc_name = match.group(1).strip()
                page_num = int(match.group(2)) if match.group(2) else None
                
                # Create unique key for deduplication
                citation_key = (doc_name.lower(), page_num)
                if citation_key in seen_citations:
                    continue
                seen_citations.add(citation_key)
                
                # Validate citation against retrieved chunks
                is_valid = False
                for chunk in retrieved_chunks:
                    chunk_doc = chunk.metadata.get('doc_name', '')
                    if doc_name.lower() in chunk_doc.lower():
                        is_valid = True
                        break
                
                citations.append({
                    'doc_name': doc_name,
                    'page': page_num,
                    'valid': is_valid,
                    'text': match.group(0)
                })
        
        return citations
    
    def generate_with_citations(self, 
                                query: str,
                                retrieved_chunks: List,
                                include_metadata: bool = True) -> GeneratedResponse:
        """
        Generate response with citations based on retrieved chunks
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved chunk objects
            include_metadata: Whether to include metadata
            
        Returns:
            GeneratedResponse object
        """
        # Build context from retrieved chunks
        context = self._build_context(retrieved_chunks)
        
        # Create system prompt
        system_prompt = self._get_system_prompt()
        
        # Create user prompt with context
        user_prompt = self._format_prompt(query, context, retrieved_chunks)
        
        # Generate response
        answer, usage = self.generate(user_prompt, system_prompt)
        
        # Extract and validate citations
        citations = self.extract_citations(answer, retrieved_chunks)
        
        response = GeneratedResponse(
            answer=answer,
            citations=citations,
            model_used=self.model_name,
            prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens'],
            total_cost=0.0  # Groq is free!
        )
        
        logger.info(f"✅ Generated response with {len(citations)} citations")
        return response
    
    def _build_context(self, retrieved_chunks: List) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            # Handle both dict and object formats
            if hasattr(chunk, 'metadata'):
                doc_name = chunk.metadata.get('doc_name', 'Unknown')
                content = chunk.content
            else:
                doc_name = chunk.get('metadata', {}).get('doc_name', 'Unknown')
                content = chunk.get('content', '')
            
            context_parts.append(
                f"[Document {idx}: {doc_name}]\n"
                f"{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for banking document analysis"""
        return """You are a helpful banking assistant that provides accurate, well-cited answers based on provided documents.

IMPORTANT INSTRUCTIONS:
1. Answer questions ONLY based on the provided context documents
2. For every factual claim, provide a citation in the format: [Source: DocumentName]
3. If information is not in the context, clearly state "I cannot find this information in the provided documents"
4. Be concise but thorough
5. Use clear and simple language

CITATION RULES:
- Cite the specific document name for each claim
- Multiple claims from the same source still need individual citations
- Use exact document names from the context provided

DO NOT:
- Make up information not in the documents
- Speculate or infer information not explicitly stated
- Provide answers without citations"""
    
    def _format_prompt(self, query: str, context: str, retrieved_chunks: List) -> str:
        """Format the complete prompt with query and context"""
        prompt = f"""CONTEXT DOCUMENTS:
{context}

USER QUESTION:
{query}

Please provide a detailed answer based ONLY on the information in the context documents above. Remember to cite your sources using [Source: DocumentName] format for every factual claim."""
        
        return prompt


# Example usage
if __name__ == "__main__":
    print("Testing SimpleRAGGenerator with Groq...")
    
    try:
        generator = SimpleRAGGenerator(model_name='llama-3.1-8b-instant')
        print("✅ Generator initialized successfully!")
        
        # Test with a simple prompt
        response, usage = generator.generate("What is 2+2?")
        print(f"\nTest response: {response}")
        print(f"Tokens: {usage['total_tokens']}")
        print(f"Cost: ${usage['cost']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
