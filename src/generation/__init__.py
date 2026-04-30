"""
Generation Module for Financial RAG System
Multi-LLM Generation System with Citation Support
"""

from .generator import (
    GenerationConfig,
    GeneratedResponse,
    BaseLLM,
    OpenAILLM,
    AnthropicLLM,
    GoogleLLM,
    CohereLLM,
    GroqLLM,
    GroqVisionLLM,
    RAGGenerator
)

from .simple_generator import SimpleRAGGenerator

__all__ = [
    'GenerationConfig',
    'GeneratedResponse',
    'BaseLLM',
    'OpenAILLM',
    'AnthropicLLM',
    'GoogleLLM',
    'CohereLLM',
    'GroqLLM',
    'GroqVisionLLM',
    'RAGGenerator',
    'SimpleRAGGenerator'
]
