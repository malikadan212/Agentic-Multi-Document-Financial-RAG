# src/agentic/self_reflection.py
"""
Self-Reflection Module
Evaluates generated answers and suggests improvements
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Result of self-reflection on an answer"""
    needs_refinement: bool
    confidence: float
    suggested_confidence: float
    issues: List[str]
    suggestions: List[str]
    missing_information: List[str]


class SelfReflector:
    """
    Evaluates generated answers for quality and completeness
    Provides feedback for iterative improvement
    """
    
    def __init__(self, generator):
        """
        Initialize reflector
        
        Args:
            generator: Answer generator (LLM) for reflection
        """
        self.generator = generator
    
    def reflect(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Any],
        confidence: float
    ) -> ReflectionResult:
        """
        Reflect on answer quality and suggest improvements
        
        Args:
            query: Original query
            answer: Generated answer
            retrieved_chunks: Retrieved document chunks
            confidence: Current confidence score
            
        Returns:
            ReflectionResult with evaluation and suggestions
        """
        
        issues = []
        suggestions = []
        missing_info = []
        
        # Check answer completeness
        if len(answer.split()) < 20:
            issues.append("Answer is too brief")
            suggestions.append("Provide more detailed explanation")
        
        # Check for citations
        citation_count = answer.count('[Source')
        if citation_count == 0:
            issues.append("No source citations found")
            suggestions.append("Add citations to support claims")
        elif citation_count < len(retrieved_chunks) / 2:
            issues.append("Insufficient citations")
            suggestions.append("Cite more sources to increase credibility")
        
        # Check for vague language
        vague_phrases = [
            'might be', 'could be', 'possibly', 'perhaps',
            'it seems', 'appears to', 'may indicate'
        ]
        vague_count = sum(1 for phrase in vague_phrases if phrase in answer.lower())
        if vague_count > 2:
            issues.append("Answer contains vague or uncertain language")
            suggestions.append("Be more definitive where evidence supports it")
        
        # Check for numerical accuracy
        numbers_in_answer = re.findall(r'\$?[\d,]+\.?\d*', answer)
        if numbers_in_answer:
            # Verify numbers appear in source chunks
            chunk_texts = ' '.join(
                chunk.content if hasattr(chunk, 'content') else str(chunk)
                for chunk in retrieved_chunks
            )
            
            unverified_numbers = []
            for num in numbers_in_answer:
                if num not in chunk_texts:
                    unverified_numbers.append(num)
            
            if unverified_numbers:
                issues.append(f"Potentially hallucinated numbers: {', '.join(unverified_numbers[:3])}")
                suggestions.append("Verify all numerical claims against source documents")
        
        # Check if query is fully addressed
        query_keywords = self._extract_keywords(query)
        answer_lower = answer.lower()
        
        unaddressed_keywords = [
            kw for kw in query_keywords 
            if kw not in answer_lower and len(kw) > 3
        ]
        
        if len(unaddressed_keywords) > len(query_keywords) / 2:
            issues.append("Query not fully addressed")
            suggestions.append(f"Address these aspects: {', '.join(unaddressed_keywords[:3])}")
            missing_info.extend(unaddressed_keywords)
        
        # Use LLM for deeper reflection if issues found
        if issues and self.generator:
            llm_reflection = self._llm_reflect(query, answer, retrieved_chunks)
            if llm_reflection:
                issues.extend(llm_reflection.get('issues', []))
                suggestions.extend(llm_reflection.get('suggestions', []))
                missing_info.extend(llm_reflection.get('missing', []))
        
        # Determine if refinement is needed
        needs_refinement = (
            len(issues) > 2 or
            confidence < 0.6 or
            citation_count == 0
        )
        
        # Suggest adjusted confidence
        suggested_confidence = confidence
        if issues:
            # Reduce confidence based on issue severity
            penalty = min(0.3, len(issues) * 0.05)
            suggested_confidence = max(0.1, confidence - penalty)
        
        result = ReflectionResult(
            needs_refinement=needs_refinement,
            confidence=confidence,
            suggested_confidence=suggested_confidence,
            issues=list(set(issues)),  # Remove duplicates
            suggestions=list(set(suggestions)),
            missing_information=list(set(missing_info))
        )
        
        logger.info(f"Reflection: {len(issues)} issues found, refinement={'needed' if needs_refinement else 'not needed'}")
        
        return result
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'can', 'what', 'how',
            'when', 'where', 'why', 'which', 'who', 'whom'
        }
        
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        return keywords
    
    def _llm_reflect(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Any]
    ) -> Optional[Dict[str, List[str]]]:
        """
        Use LLM to perform deeper reflection on answer quality
        
        Returns:
            Dict with 'issues', 'suggestions', 'missing' lists
        """
        
        try:
            # Build context from chunks
            chunk_summaries = []
            for i, chunk in enumerate(retrieved_chunks[:3], 1):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                chunk_summaries.append(f"Source {i}: {content[:200]}...")
            
            context = "\n".join(chunk_summaries)
            
            reflection_prompt = f"""Evaluate the quality of this answer and identify any issues.

Question: {query}

Answer: {answer}

Available Context:
{context}

Evaluate the answer for:
1. Completeness - Does it fully answer the question?
2. Accuracy - Are all claims supported by the context?
3. Clarity - Is it clear and well-structured?
4. Citations - Are sources properly cited?

Provide your evaluation in this format:
ISSUES:
- [list any problems]

SUGGESTIONS:
- [list improvements]

MISSING:
- [list missing information]

Evaluation:"""

            system_prompt = "You are a critical evaluator of answer quality. Be thorough but fair."
            
            reflection_text, _ = self.generator.llm.generate(reflection_prompt, system_prompt)
            
            # Parse reflection
            result = {
                'issues': [],
                'suggestions': [],
                'missing': []
            }
            
            current_section = None
            for line in reflection_text.split('\n'):
                line = line.strip()
                
                if line.startswith('ISSUES:'):
                    current_section = 'issues'
                elif line.startswith('SUGGESTIONS:'):
                    current_section = 'suggestions'
                elif line.startswith('MISSING:'):
                    current_section = 'missing'
                elif line.startswith('-') and current_section:
                    item = line.lstrip('- ').strip()
                    if item and len(item) > 5:
                        result[current_section].append(item)
            
            return result if any(result.values()) else None
            
        except Exception as e:
            logger.warning(f"LLM reflection failed: {e}")
            return None
    
    def validate_citations(
        self,
        answer: str,
        retrieved_chunks: List[Any]
    ) -> Dict[str, Any]:
        """
        Validate that all citations in answer correspond to actual sources
        
        Returns:
            Dict with validation results
        """
        
        # Extract citations from answer
        citation_pattern = r'\[Source\s+(\d+)(?::\s*([^,\]]+)(?:,\s*Page\s+(\d+))?)?\]'
        citations = re.findall(citation_pattern, answer)
        
        valid_citations = []
        invalid_citations = []
        
        for citation in citations:
            source_num = int(citation[0])
            
            # Check if source number is valid
            if source_num <= len(retrieved_chunks):
                chunk = retrieved_chunks[source_num - 1]
                
                # Verify document name if provided
                if citation[1]:
                    doc_name = citation[1].strip()
                    actual_doc = chunk.metadata.get('doc_name', '') if hasattr(chunk, 'metadata') else ''
                    
                    if doc_name.lower() in actual_doc.lower() or actual_doc.lower() in doc_name.lower():
                        valid_citations.append(citation)
                    else:
                        invalid_citations.append(f"Source {source_num}: document name mismatch")
                else:
                    valid_citations.append(citation)
            else:
                invalid_citations.append(f"Source {source_num}: does not exist")
        
        return {
            'total_citations': len(citations),
            'valid_citations': len(valid_citations),
            'invalid_citations': len(invalid_citations),
            'invalid_details': invalid_citations,
            'citation_accuracy': len(valid_citations) / max(1, len(citations))
        }
