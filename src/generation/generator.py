# src/generation/generator.py
"""
Multi-LLM Generation System with Citation Support
Supports OpenAI GPT, Anthropic Claude, Google Gemini, and Cohere
"""

# Import necessary libraries and modules
from typing import List, Dict, Optional, Tuple, Union  # For type hints
from dataclasses import dataclass  # For creating data classes
import os  # For accessing environment variables
from abc import ABC, abstractmethod  # For creating abstract base classes
import logging  # For logging functionality
import re  # For regular expressions (citation extraction)
from pathlib import Path  # For path manipulation

# Configure logging FIRST (before any imports that use logger)
logging.basicConfig(level=logging.INFO)  # Set default logging level to INFO
logger = logging.getLogger(__name__)  # Create logger instance for this module

# Import third-party LLM API libraries (optional - only needed if using that provider)
try:
    import openai  # OpenAI API client
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Install with: pip install openai")

try:
    from anthropic import Anthropic  # Anthropic Claude API client
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not installed. Install with: pip install anthropic")

try:
    # Note: google.generativeai is deprecated, but google.genai requires significant code changes
    # TODO: Migrate to google.genai when ready - see https://github.com/google-gemini/deprecated-generative-ai-python
    import google.generativeai as genai  # Google Gemini API client (deprecated, migrate to google.genai)
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logger.warning("Google Generative AI not installed. Install with: pip install google-generativeai")

try:
    import cohere  # Cohere API client
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    logger.warning("Cohere not installed. Install with: pip install cohere")

try:
    from groq import Groq  # Groq API client
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not installed. Install with: pip install groq")

from dotenv import load_dotenv  # For loading environment variables

# Load environment variables from .env file
_project_root = Path(__file__).resolve().parents[2]  # Get project root (two levels up)
_env_path = _project_root / ".env"  # Path to .env file
# Check if .env file exists, if not try .env.example
if not _env_path.exists():
    fallback = _project_root / ".env.example"  # Fallback path
    if fallback.exists():  # If .env.example exists
        _env_path = fallback  # Use .env.example instead
# Load environment variables from the determined path
load_dotenv(dotenv_path=_env_path, override=False)


# Data class for generation configuration
@dataclass
class GenerationConfig:
    """Configuration for generation parameters"""
    temperature: float = 0.1  # Low temperature for factual, deterministic responses
    max_tokens: int = 1000  # Maximum tokens in the generated response
    top_p: float = 0.9  # Nucleus sampling parameter
    frequency_penalty: float = 0.0  # Penalty for frequent token usage
    presence_penalty: float = 0.0  # Penalty for new token presence


# Data class for generated responses
@dataclass
class GeneratedResponse:
    """Represents a generated response with citations"""
    answer: str  # The generated answer text
    citations: List[Dict]  # List of citation dictionaries
    model_used: str  # Name of the model used for generation
    prompt_tokens: int  # Number of tokens in the prompt
    completion_tokens: int  # Number of tokens in the completion
    total_cost: float = 0.0  # Total cost of the API call
    
    # String representation for debugging
    def __repr__(self):
        return f"Response(model={self.model_used}, citations={len(self.citations)}, tokens={self.completion_tokens})"


# Abstract base class for all LLM providers
class BaseLLM(ABC):
    """Abstract base class for LLM providers"""
    
    # Initialize with model name and configuration
    def __init__(self, model_name: str, config: GenerationConfig):
        self.model_name = model_name  # Name of the model
        self.config = config  # Generation configuration
    
    # Abstract method that must be implemented by subclasses
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict]:
        """Generate response from LLM"""
        pass
    
    # Method to extract citations from generated text
    def extract_citations(self, response: str, retrieved_chunks: List) -> List[Dict]:
        """
        Extract and validate citations from response
        
        Args:
            response: Generated response text
            retrieved_chunks: List of retrieved chunks used in context
            
        Returns:
            List of citation dictionaries
        """
        citations = []  # Initialize empty list for citations
        
        # Define multiple regex patterns to catch different citation formats
        patterns = [
            r'\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]',  # [Source: DocName, Page X]
            r'\[Source:\s*([^,\]]+)(?:,\s*Page\s*(\d+))?\]',  # [Source: DocName] or [Source: DocName, Page X]
            r'\(Source:\s*([^,\)]+)(?:,\s*Page\s*(\d+))?\)',  # (Source: DocName) or (Source: DocName, Page X)
            r'\[([^,\]]+),\s*Page\s*(\d+)\]',  # [DocName, Page X]
        ]
        
        seen_citations = set()  # Set to track unique citations and avoid duplicates
        
        # Try each pattern to find citations
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)  # Find all matches (case-insensitive)
            
            # Process each match found
            for match in matches:
                doc_name = match.group(1).strip()  # Extract document name (first capture group)
                page_num = int(match.group(2)) if match.group(2) else None  # Extract page number if present
                
                # Create unique key for deduplication (lowercase name and page number)
                citation_key = (doc_name.lower(), page_num)
                if citation_key in seen_citations:
                    continue  # Skip if already seen
                seen_citations.add(citation_key)  # Add to seen citations
                
                # Validate citation against retrieved chunks
                is_valid = any(  # Check if any retrieved chunk matches this document
                    doc_name.lower() in chunk.metadata.get('doc_name', '').lower()  # Case-insensitive comparison
                    for chunk in retrieved_chunks
                )
                
                # Add citation to list
                citations.append({
                    'doc_name': doc_name,  # Document name
                    'page': page_num,  # Page number (or None)
                    'valid': is_valid,  # Whether citation matches retrieved documents
                    'text': match.group(0)  # The full citation text as found in response
                })
        
        return citations  # Return list of citations


# Implementation for OpenAI GPT models
class OpenAILLM(BaseLLM):
    """OpenAI GPT models"""
    
    # Model pricing information (cost per 1000 tokens)
    MODELS = {
        'gpt-3.5-turbo': {'cost_per_1k_input': 0.0015, 'cost_per_1k_output': 0.002},
        'gpt-4': {'cost_per_1k_input': 0.03, 'cost_per_1k_output': 0.06},
        'gpt-4-turbo': {'cost_per_1k_input': 0.01, 'cost_per_1k_output': 0.03}
    }
    
    # Initialize OpenAI client
    def __init__(self, model_name: str = 'gpt-3.5-turbo', config: GenerationConfig = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        super().__init__(model_name, config or GenerationConfig())  # Call parent constructor
        openai.api_key = os.getenv('OPENAI_API_KEY')  # Get API key from environment
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")  # Validate API key
    
    # Generate method for OpenAI
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict]:
        """Generate using OpenAI API"""
        messages = []  # Initialize messages list
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Make API call to OpenAI
            response = openai.chat.completions.create(
                model=self.model_name,  # Model to use
                messages=messages,  # Conversation messages
                temperature=self.config.temperature,  # Temperature parameter
                max_tokens=self.config.max_tokens,  # Max tokens in response
                top_p=self.config.top_p  # Top-p sampling parameter
            )
            
            answer = response.choices[0].message.content  # Extract answer text
            
            # Extract usage information
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,  # Tokens in prompt
                'completion_tokens': response.usage.completion_tokens,  # Tokens in completion
                'total_tokens': response.usage.total_tokens,  # Total tokens
                'cost': self._calculate_cost(response.usage)  # Calculate cost
            }
            
            return answer, usage  # Return answer and usage info
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")  # Log error
            raise  # Re-raise exception
    
    # Calculate cost based on token usage
    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on token usage"""
        # Get pricing for current model, default to gpt-3.5-turbo if not found
        pricing = self.MODELS.get(self.model_name, self.MODELS['gpt-3.5-turbo'])
        # Calculate input cost
        input_cost = (usage.prompt_tokens / 1000) * pricing['cost_per_1k_input']
        # Calculate output cost
        output_cost = (usage.completion_tokens / 1000) * pricing['cost_per_1k_output']
        return input_cost + output_cost  # Return total cost


# Implementation for Anthropic Claude models
class AnthropicLLM(BaseLLM):
    """Anthropic Claude models"""
    
    # Model pricing information
    MODELS = {
        'claude-3-sonnet-20240229': {'cost_per_1k_input': 0.003, 'cost_per_1k_output': 0.015},
        'claude-3-opus-20240229': {'cost_per_1k_input': 0.015, 'cost_per_1k_output': 0.075}
    }
    
    # Initialize Anthropic client
    def __init__(self, model_name: str = 'claude-3-sonnet-20240229', config: GenerationConfig = None):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
        super().__init__(model_name, config or GenerationConfig())  # Call parent constructor
        api_key = os.getenv('ANTHROPIC_API_KEY')  # Get API key from environment
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")  # Validate API key
        self.client = Anthropic(api_key=api_key)  # Create Anthropic client
    
    # Generate method for Anthropic
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict]:
        """Generate using Anthropic API"""
        try:
            # Make API call to Anthropic
            message = self.client.messages.create(
                model=self.model_name,  # Model to use
                max_tokens=self.config.max_tokens,  # Max tokens in response
                temperature=self.config.temperature,  # Temperature parameter
                system=system_prompt or "",  # System prompt (empty string if None)
                messages=[{"role": "user", "content": prompt}]  # User message
            )
            
            answer = message.content[0].text  # Extract answer text (first content block)
            
            # Extract usage information
            usage = {
                'prompt_tokens': message.usage.input_tokens,  # Input tokens
                'completion_tokens': message.usage.output_tokens,  # Output tokens
                'total_tokens': message.usage.input_tokens + message.usage.output_tokens,  # Total tokens
                'cost': self._calculate_cost(message.usage)  # Calculate cost
            }
            
            return answer, usage  # Return answer and usage info
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")  # Log error
            raise  # Re-raise exception
    
    # Calculate cost based on token usage
    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on token usage"""
        # Get pricing for current model, default to sonnet if not found
        pricing = self.MODELS.get(self.model_name, self.MODELS['claude-3-sonnet-20240229'])
        # Calculate input cost
        input_cost = (usage.input_tokens / 1000) * pricing['cost_per_1k_input']
        # Calculate output cost
        output_cost = (usage.output_tokens / 1000) * pricing['cost_per_1k_output']
        return input_cost + output_cost  # Return total cost


# Implementation for Google Gemini models
class GoogleLLM(BaseLLM):
    """Google Gemini models"""
    
    # Model name mapping (user-friendly names to actual model names)
    MODEL_MAPPING = {
        'gemini-1.5-flash': 'gemini-1.5-flash',  # Fast model
        'gemini-1.5-pro': 'gemini-1.5-pro',  # High quality model
        'gemini-2.0-flash': 'gemini-2.0-flash-exp',  # Experimental 2.0 model
        'gemini-pro': 'gemini-1.5-flash',  # Legacy name mapping
        # Fallback aliases for non-existent models
        'gemini-2.5-flash': 'gemini-1.5-flash',  # Map to 1.5 if 2.5 doesn't exist
        'gemini-2.5-pro': 'gemini-1.5-pro',  # Map to 1.5 if 2.5 doesn't exist
    }
    
    # Initialize Google Gemini client
    def __init__(self, model_name: str = 'gemini-1.5-flash', config: GenerationConfig = None):
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")
        super().__init__(model_name, config or GenerationConfig())  # Call parent constructor
        api_key = os.getenv('GOOGLE_API_KEY')  # Get API key from environment
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")  # Validate API key
        genai.configure(api_key=api_key)  # Configure Google AI
        
        # Use mapped model name (default to gemini-2.5-flash if not in mapping)
        actual_model_name = self.MODEL_MAPPING.get(model_name, 'gemini-2.5-flash')
        try:
            self.model = genai.GenerativeModel(actual_model_name)  # Create model instance
            self.model_name = actual_model_name  # Store actual model name
            logger.info(f"Initialized Google Gemini with model: {actual_model_name}")  # Log success
        except Exception as e:
            # Handle model initialization failure
            logger.error(f"Failed to initialize model {actual_model_name}: {e}")
            raise ValueError(  # Raise helpful error message
                f"Could not initialize Gemini model '{actual_model_name}'. "
                f"Error: {str(e)}. "
                f"Try using 'gemini-2.5-flash' (latest) or 'gemini-1.5-pro' (previous gen), "
                f"or check your API key permissions in Google Cloud Console."
            )
    
    # Static method to list available models
    @staticmethod
    def list_available_models(api_key: str = None) -> List[str]:
        """List available Gemini models"""
        if not api_key:
            api_key = os.getenv('GOOGLE_API_KEY')  # Get API key from environment if not provided
        if not api_key:
            raise ValueError("GOOGLE_API_KEY required to list models")  # Validate API key
        
        try:
            genai.configure(api_key=api_key)  # Configure with API key
            models = genai.list_models()  # Get list of available models
            # Filter models that support content generation
            available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            return available  # Return list of available model names
        except Exception as e:
            logger.error(f"Error listing models: {e}")  # Log error
            return []  # Return empty list on error
    
    # Generate method for Google Gemini
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict]:
        """Generate using Google Gemini API"""
        try:
            # Combine system prompt and user prompt
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            # Make API call to Google Gemini
            response = self.model.generate_content(
                full_prompt,  # Combined prompt
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,  # Temperature parameter
                    max_output_tokens=self.config.max_tokens,  # Max tokens in response
                )
            )
            
            answer = response.text  # Extract answer text
            
            # Gemini doesn't provide token counts, so use zeros
            usage = {
                'prompt_tokens': 0,  # Placeholder (not provided by API)
                'completion_tokens': 0,  # Placeholder (not provided by API)
                'total_tokens': 0,  # Placeholder (not provided by API)
                'cost': 0.0  # Gemini pricing varies, hard to calculate here
            }
            
            return answer, usage  # Return answer and usage info
            
        except Exception as e:
            error_msg = str(e)  # Convert exception to string
            # Check for model not found errors
            if "404" in error_msg or "not found" in error_msg.lower() or "not supported" in error_msg.lower():
                logger.error(f"Google Gemini model error: {error_msg}")  # Log error
                
                # Create helpful error message with troubleshooting tips
                suggestion = (
                    f"\n\n**Troubleshooting:**\n"
                    f"1. The model '{self.model_name}' may not be available for your API version or region.\n"
                    f"2. Try using 'gemini-2.5-flash' (latest fast model) or 'gemini-2.5-pro' (latest quality model).\n"
                    f"3. Check your API key has access to Gemini models in Google Cloud Console.\n"
                    f"4. Verify you're using the correct API version (v1 vs v1beta).\n"
                    f"5. Visit https://ai.google.dev/models/gemini for current model availability.\n"
                    f"6. Note: Gemini 2.5 may require a newer API key or specific permissions."
                )
                
                # Raise informative error
                raise ValueError(
                    f"Model '{self.model_name}' is not available. "
                    f"Error: {error_msg}{suggestion}"
                )
            logger.error(f"Google API error: {error_msg}")  # Log other errors
            raise  # Re-raise exception


# Implementation for Cohere models
class CohereLLM(BaseLLM):
    """Cohere models"""
    
    # Initialize Cohere client
    def __init__(self, model_name: str = 'command', config: GenerationConfig = None):
        if not COHERE_AVAILABLE:
            raise ImportError("Cohere library not installed. Install with: pip install cohere")
        super().__init__(model_name, config or GenerationConfig())  # Call parent constructor
        api_key = os.getenv('COHERE_API_KEY')  # Get API key from environment
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment")  # Validate API key
        self.client = cohere.Client(api_key)  # Create Cohere client
    
    # Generate method for Cohere
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict]:
        """Generate using Cohere API"""
        try:
            # Combine system prompt and user prompt
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            # Make API call to Cohere
            response = self.client.generate(
                prompt=full_prompt,  # Combined prompt
                model=self.model_name,  # Model to use
                temperature=self.config.temperature,  # Temperature parameter
                max_tokens=self.config.max_tokens,  # Max tokens in response
            )
            
            answer = response.generations[0].text  # Extract answer text (first generation)
            
            # Cohere doesn't provide token counts in response, so use zeros
            usage = {
                'prompt_tokens': 0,  # Placeholder (not provided by API)
                'completion_tokens': 0,  # Placeholder (not provided by API)
                'total_tokens': 0,  # Placeholder (not provided by API)
                'cost': 0.0  # Cohere pricing varies, hard to calculate here
            }
            
            return answer, usage  # Return answer and usage info
            
        except Exception as e:
            logger.error(f"Cohere API error: {str(e)}")  # Log error
            raise  # Re-raise exception


# Implementation for Groq models
class GroqLLM(BaseLLM):
    """Groq-hosted open models (e.g., Llama 3, Mixtral)."""

    # Model pricing information (currently free for listed models)
    MODELS = {
        "llama-3.1-8b-instant": {"cost_per_1k_input": 0.0, "cost_per_1k_output": 0.0},
        "llama-3.3-70b-versatile": {"cost_per_1k_input": 0.0, "cost_per_1k_output": 0.0},
        "mixtral-8x7b-32768": {"cost_per_1k_input": 0.0, "cost_per_1k_output": 0.0},
    }

    # Initialize Groq client
    def __init__(self, model_name: str = "llama-3.1-8b-instant", config: GenerationConfig = None):
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library not installed. Install with: pip install groq")
        super().__init__(model_name, config or GenerationConfig())  # Call parent constructor
        api_key = os.getenv("GROQ_API_KEY")  # Get API key from environment
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")  # Validate API key
        self.client = Groq(api_key=api_key)  # Create Groq client

    # Generate method for Groq
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict]:
        """Generate using Groq Chat Completions API."""
        messages: List[Dict[str, str]] = []  # Initialize messages list

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user prompt
        messages.append({"role": "user", "content": prompt})

        try:
            # Make API call to Groq
            response = self.client.chat.completions.create(
                model=self.model_name,  # Model to use
                messages=messages,  # Conversation messages
                temperature=self.config.temperature,  # Temperature parameter
                max_tokens=self.config.max_tokens,  # Max tokens in response
                top_p=self.config.top_p,  # Top-p sampling parameter
            )

            answer = response.choices[0].message.content  # Extract answer text

            # Safely extract usage information (Groq API may not have usage field)
            usage_raw = getattr(response, "usage", None)  # Get usage attribute if exists
            if usage_raw:
                # Extract token counts with safe defaults
                prompt_tokens = getattr(usage_raw, "prompt_tokens", 0)
                completion_tokens = getattr(usage_raw, "completion_tokens", 0)
            else:
                # Default to zeros if no usage info
                prompt_tokens = 0
                completion_tokens = 0

            # Create usage dictionary
            usage = {
                "prompt_tokens": prompt_tokens,  # Tokens in prompt
                "completion_tokens": completion_tokens,  # Tokens in completion
                "total_tokens": prompt_tokens + completion_tokens,  # Total tokens
                "cost": self._calculate_cost(prompt_tokens, completion_tokens),  # Calculate cost
            }

            return answer, usage  # Return answer and usage info

        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")  # Log error
            raise  # Re-raise exception

    # Calculate cost based on token usage
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for Groq usage. Defaults to 0.0 for free-tier usage."""
        # Get pricing for current model, default to free if not found
        pricing = self.MODELS.get(self.model_name, {"cost_per_1k_input": 0.0, "cost_per_1k_output": 0.0})
        # Calculate input cost
        input_cost = (prompt_tokens / 1000) * pricing["cost_per_1k_input"]
        # Calculate output cost
        output_cost = (completion_tokens / 1000) * pricing["cost_per_1k_output"]
        return input_cost + output_cost  # Return total cost
    
    # Streaming generation method
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        """
        Generate response using streaming for real-time token display.
        Yields tokens as they arrive from the API.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            
        Yields:
            str: Individual tokens/chunks as they arrive
        """
        messages: List[Dict[str, str]] = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Create streaming response
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=True,  # Enable streaming
            )
            
            # Yield tokens as they arrive
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Groq streaming error: {str(e)}")
            raise


# Implementation for Groq Vision models (Llama 4 Vision)
class GroqVisionLLM(BaseLLM):
    """
    Groq-hosted vision models for multimodal understanding.
    Supports analyzing images along with text prompts.
    FREE to use with Groq API key.
    """
    
    # Available vision models (updated December 2024)
    MODELS = {
        "meta-llama/llama-4-scout-17b-16e-instruct": {"cost_per_1k_input": 0.0, "cost_per_1k_output": 0.0},
        "meta-llama/llama-4-maverick-17b-128e-instruct": {"cost_per_1k_input": 0.0, "cost_per_1k_output": 0.0},
    }
    
    def __init__(self, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct", config: GenerationConfig = None):
        """
        Initialize Groq Vision model
        
        Args:
            model_name: Vision model to use (default: llama-3.2-11b-vision-preview)
            config: Generation configuration
        """
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library not installed. Install with: pip install groq")
        super().__init__(model_name, config or GenerationConfig())
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        self.client = Groq(api_key=api_key)
        logger.info(f"✅ Groq Vision model initialized: {model_name}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Generate text-only response (for compatibility with base class).
        For image analysis, use generate_with_images() method.
        """
        return self.generate_with_images(prompt, images=[], system_prompt=system_prompt)
    
    def generate_with_images(self, prompt: str, images: List[Union[str, bytes]] = None,
                             system_prompt: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Generate response with image understanding.
        
        Args:
            prompt: Text prompt/question
            images: List of images (base64 strings or raw bytes)
            system_prompt: Optional system instruction
            
        Returns:
            Tuple of (answer, usage_dict)
        """
        import base64
        
        messages: List[Dict] = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Build content array with text and images
        content = []
        
        # Add images first
        if images:
            for img in images:
                # Handle different image input types
                if isinstance(img, bytes):
                    # Raw bytes - encode to base64
                    img_base64 = base64.b64encode(img).decode('utf-8')
                elif isinstance(img, str) and len(img) > 1000:
                    # Already base64 encoded
                    img_base64 = img
                elif isinstance(img, str):
                    # File path - read and encode
                    with open(img, 'rb') as f:
                        img_base64 = base64.b64encode(f.read()).decode('utf-8')
                else:
                    continue
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Add user message with content array
        messages.append({"role": "user", "content": content})
        
        try:
            # Make API call to Groq Vision
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            answer = response.choices[0].message.content
            
            # Extract usage info
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
                "cost": 0.0  # Free tier
            }
            
            return answer, usage
            
        except Exception as e:
            logger.error(f"Groq Vision API error: {str(e)}")
            raise
    
    def describe_image(self, image: Union[str, bytes], 
                       context: str = "financial document") -> str:
        """
        Generate a description of an image for indexing/retrieval.
        
        Args:
            image: Image as base64 string or bytes
            context: Context hint for better descriptions
            
        Returns:
            Text description of the image
        """
        prompt = f"""Analyze this image from a {context}. 
Describe what you see in detail, including:
- Type of visualization (chart, table, graph, diagram)
- Key data points or values shown
- Trends or patterns visible
- Labels and titles

Provide a concise but informative description."""
        
        answer, _ = self.generate_with_images(prompt, images=[image])
        return answer


# Main RAG Generator class that orchestrates everything
class RAGGenerator:
    """
    RAG Generator that combines retrieval results with LLM generation
    Supports multiple LLM providers for comparison
    """
    
    # Mapping of provider names to their implementation classes (only include available ones)
    LLM_PROVIDERS = {}
    if OPENAI_AVAILABLE:
        LLM_PROVIDERS['openai'] = OpenAILLM
    if ANTHROPIC_AVAILABLE:
        LLM_PROVIDERS['anthropic'] = AnthropicLLM
    if GOOGLE_AVAILABLE:
        LLM_PROVIDERS['google'] = GoogleLLM
    if COHERE_AVAILABLE:
        LLM_PROVIDERS['cohere'] = CohereLLM
    if GROQ_AVAILABLE:
        LLM_PROVIDERS['groq'] = GroqLLM
        LLM_PROVIDERS['groq_vision'] = GroqVisionLLM
    
    # Initialize RAG Generator
    def __init__(self, 
                 provider: str = 'openai',
                 model_name: Optional[str] = None,
                 config: GenerationConfig = None):
        """
        Initialize RAG Generator
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'google', 'cohere')
            model_name: Specific model name (uses default if None)
            config: Generation configuration
        """
        # Validate provider
        if provider not in self.LLM_PROVIDERS:
            raise ValueError(f"Provider {provider} not supported. Choose from {list(self.LLM_PROVIDERS.keys())}")
        
        # Get the appropriate LLM class
        llm_class = self.LLM_PROVIDERS[provider]
        
        # Create LLM instance with or without specific model name
        if model_name:
            self.llm = llm_class(model_name, config)  # Use specified model name
        else:
            self.llm = llm_class(config=config)  # Use default model name
        
        self.provider = provider  # Store provider name
        logger.info(f"✅ RAG Generator initialized: {provider}, {self.llm.model_name}")  # Log success
    
    # Main method to generate responses with citations
    def generate_with_citations(self, 
                                query: str,
                                retrieved_chunks: List,
                                include_metadata: bool = True) -> GeneratedResponse:
        """
        Generate response with citations based on retrieved chunks
        
        Args:
            query: User query
            retrieved_chunks: List of RetrievalResult objects
            include_metadata: Whether to include metadata in citations
            
        Returns:
            GeneratedResponse object
        """
        # Build context string from retrieved chunks
        context = self._build_context(retrieved_chunks)
        
        # Create system prompt
        system_prompt = self._get_system_prompt()
        
        # Create user prompt with context
        user_prompt = self._format_prompt(query, context, retrieved_chunks)
        
        # Generate response using LLM
        answer, usage = self.llm.generate(user_prompt, system_prompt)
        
        # Extract and validate citations from the generated answer
        citations = self.llm.extract_citations(answer, retrieved_chunks)
        
        # Create GeneratedResponse object
        response = GeneratedResponse(
            answer=answer,  # Generated answer text
            citations=citations,  # Extracted citations
            model_used=self.llm.model_name,  # Model used
            prompt_tokens=usage['prompt_tokens'],  # Prompt tokens
            completion_tokens=usage['completion_tokens'],  # Completion tokens
            total_cost=usage['cost']  # Total cost
        )
        
        logger.info(f"✅ Generated response with {len(citations)} citations")  # Log success
        return response
    
    # Build context string from retrieved chunks
    def _build_context(self, retrieved_chunks: List) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []  # Initialize list for context parts
        
        # Process each retrieved chunk
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            doc_name = chunk.metadata.get('doc_name', 'Unknown')  # Get document name
            page = chunk.metadata.get('page', 0)  # Get page number (default to 0)
            
            # Format chunk information
            context_parts.append(
                f"[Document {idx}: {doc_name}, Page {page}]\n"  # Document header
                f"{chunk.content}\n"  # Document content
            )
        
        return "\n".join(context_parts)  # Join all parts with newlines
    
    # Create system prompt for financial analysis
    def _get_system_prompt(self) -> str:
        """Get system prompt for financial analysis"""
        return """You are a financial analysis assistant that provides accurate, well-cited answers based on provided documents.

IMPORTANT INSTRUCTIONS:
1. Answer questions ONLY based on the provided context
2. For every factual claim, provide a citation in the format: [Source: DocumentName, Page X]
3. If information is not in the context, clearly state "I cannot find this information in the provided documents"
4. Maintain numerical precision - do not round or approximate financial figures
5. For calculations, show your work step by step
6. Be concise but thorough

CITATION RULES:
- Cite the specific document and page number for each claim
- Multiple claims from the same source still need individual citations
- Use exact document names and page numbers from the context provided"""
    
    # Format the complete prompt with query and context
    def _format_prompt(self, query: str, context: str, retrieved_chunks: List) -> str:
        """Format the complete prompt with query and context"""
        prompt = f"""CONTEXT DOCUMENTS:
{context}

USER QUESTION:
{query}

Please provide a detailed answer based ONLY on the information in the context documents above. Remember to cite your sources using [Source: DocumentName, Page X] format for every factual claim."""
        
        return prompt


# Example Usage
if __name__ == "__main__":
    # Import required modules
    from retrieval.retriever import HybridRetriever
    from document_processing.processor import DocumentPipeline
    
    # Setup document processing pipeline
    pipeline = DocumentPipeline()
    chunks = pipeline.process_directory("data/raw/pdfs")  # Process PDF documents
    
    # Setup retriever
    retriever = HybridRetriever(embedding_model='minilm')  # Create retriever with MiniLM
    retriever.index_documents(chunks)  # Index the processed chunks
    
    # Test different LLM providers
    providers = ['openai', 'anthropic', 'google']  # List of providers to test
    
    query = "What was Apple's revenue in Q3 2023?"  # Test query
    retrieved_results = retriever.retrieve(query, top_k=5)  # Retrieve relevant chunks
    
    # Test each provider
    for provider in providers:
        try:
            print(f"\n{'='*60}")  # Separator line
            print(f"Testing {provider.upper()}")  # Provider name
            print(f"{'='*60}")  # Separator line
            
            generator = RAGGenerator(provider=provider)  # Create generator for provider
            response = generator.generate_with_citations(query, retrieved_results)  # Generate response
            
            print(f"\n{response}")  # Print response summary
            print(f"\nAnswer:\n{response.answer}")  # Print answer text
            print(f"\nCitations: {len(response.citations)}")  # Print citation count
            print(f"Cost: ${response.total_cost:.4f}")  # Print cost
            
        except Exception as e:
            print(f"❌ Error with {provider}: {str(e)}")  # Print error if provider fails