# src/agentic/agent_executor.py
"""
Agent Executor
Executes query plans with adaptive retrieval and self-reflection
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time

from .agent_planner import ExecutionPlan, SubQuery, QueryComplexity
from .self_reflection import SelfReflector, ReflectionResult

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStep:
    """Result of executing a single sub-query"""
    sub_query: SubQuery
    retrieved_chunks: List[Any]
    answer: str
    confidence: float
    execution_time: float
    reflection: Optional[ReflectionResult] = None


@dataclass
class ExecutionResult:
    """Complete result of executing a query plan"""
    original_query: str
    plan: ExecutionPlan
    steps: List[ExecutionStep]
    final_answer: str
    final_confidence: float
    total_time: float
    iterations: int  # Number of refinement iterations
    success: bool


class AgentExecutor:
    """
    Executes query plans with adaptive retrieval
    Implements self-reflection and iterative refinement
    """
    
    def __init__(
        self,
        retriever,
        ,
        reflector: Optional[SelfReflector] = None,
        max_iterations: int = 3,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize executor
        
        Args:
            retriever: Document retriever
            : Answer  (LLM)
            reflector: Self-reflection module
            max_iterations: Maximum refinement iterations
            confidence_threshold: Minimum confidence to accept answer
        """
        self.retriever = retriever
        self. = 
        self.reflector = reflector or SelfReflector()
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
    
    def execute(
        self, 
        plan: ExecutionPlan,
        top_k: int = 5,
        enable_reflection: bool = True
    ) -> ExecutionResult:
        """
        Execute a query plan
        
        Args:
            plan: Execution plan from AgentPlanner
            top_k: Number of chunks to retrieve per step
            enable_reflection: Whether to use self-reflection
            
        Returns:
            ExecutionResult with final answer and metadata
        """
        start_time = time.time()
        steps = []
        completed_indices = []
        iteration = 0
        
        logger.info(f"Executing plan with {len(plan.sub_queries)} steps")
        
        # Execute sub-queries in order
        while len(completed_indices) < len(plan.sub_queries):
            # Get next sub-query
            next_idx = len(completed_indices)
            if next_idx >= len(plan.sub_queries):
                break
            
            sub_query = plan.sub_queries[next_idx]
            
            # Execute sub-query
            step = self._execute_sub_query(
                sub_query=sub_query,
                previous_steps=steps,
                top_k=top_k,
                enable_reflection=enable_reflection
            )
            
            steps.append(step)
            completed_indices.append(next_idx)
            
            logger.info(f"Completed step {next_idx + 1}/{len(plan.sub_queries)}: confidence={step.confidence:.2f}")
        
        # Synthesize final answer
        final_answer, final_confidence = self._synthesize_answer(
            plan=plan,
            steps=steps
        )
        
        # Reflect on final answer if enabled
        if enable_reflection and final_confidence < self.confidence_threshold and iteration < self.max_iterations:
            logger.info(f"Final confidence {final_confidence:.2f} below threshold, reflecting...")
            
            reflection = self.reflector.reflect(
                query=plan.original_query,
                answer=final_answer,
                retrieved_chunks=[chunk for step in steps for chunk in step.retrieved_chunks],
                confidence=final_confidence
            )
            
            if reflection.needs_refinement:
                # Refine answer based on reflection
                final_answer = self._refine_answer(
                    original_answer=final_answer,
                    reflection=reflection,
                    steps=steps
                )
                final_confidence = reflection.suggested_confidence
                iteration += 1
        
        total_time = time.time() - start_time
        
        result = ExecutionResult(
            original_query=plan.original_query,
            plan=plan,
            steps=steps,
            final_answer=final_answer,
            final_confidence=final_confidence,
            total_time=total_time,
            iterations=iteration,
            success=final_confidence >= self.confidence_threshold
        )
        
        logger.info(f"Execution complete: {total_time:.2f}s, confidence={final_confidence:.2f}")
        
        return result
    
    def _execute_sub_query(
        self,
        sub_query: SubQuery,
        previous_steps: List[ExecutionStep],
        top_k: int,
        enable_reflection: bool
    ) -> ExecutionStep:
        """Execute a single sub-query"""
        
        start_time = time.time()
        
        # Build context from previous steps if needed
        context = ""
        if sub_query.context_needed and previous_steps:
            context = self._build_context(previous_steps)
        
        # Enhance query with context
        enhanced_query = sub_query.question
        if context:
            enhanced_query = f"{context}\n\nBased on the above information: {sub_query.question}"
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(enhanced_query, top_k=top_k)
        
        # Generate answer
        answer = self._generate_answer(
            query=enhanced_query,
            chunks=retrieved_chunks,
            context=context
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(retrieved_chunks, answer)
        
        # Reflect if enabled and confidence is low
        reflection = None
        if enable_reflection and confidence < self.confidence_threshold:
            reflection = self.reflector.reflect(
                query=sub_query.question,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                confidence=confidence
            )
            
            if reflection.needs_refinement:
                # Adaptive retrieval: get more chunks
                logger.info(f"Low confidence, retrieving more chunks...")
                additional_chunks = self.retriever.retrieve(
                    enhanced_query, 
                    top_k=top_k * 2
                )
                
                # Regenerate with more context
                answer = self._generate_answer(
                    query=enhanced_query,
                    chunks=additional_chunks,
                    context=context
                )
                confidence = self._calculate_confidence(additional_chunks, answer)
                retrieved_chunks = additional_chunks
        
        execution_time = time.time() - start_time
        
        return ExecutionStep(
            sub_query=sub_query,
            retrieved_chunks=retrieved_chunks,
            answer=answer,
            confidence=confidence,
            execution_time=execution_time,
            reflection=reflection
        )
    
    def _build_context(self, previous_steps: List[ExecutionStep]) -> str:
        """Build context from previous execution steps"""
        context_parts = []
        
        for i, step in enumerate(previous_steps, 1):
            context_parts.append(f"Step {i}: {step.sub_query.question}")
            context_parts.append(f"Answer: {step.answer}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(
        self,
        query: str,
        chunks: List[Any],
        context: str = ""
    ) -> str:
        """Generate answer using LLM"""
        
        # Build prompt with retrieved context
        chunk_texts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            doc_name = chunk.metadata.get('doc_name', 'Unknown') if hasattr(chunk, 'metadata') else 'Unknown'
            page = chunk.metadata.get('page', '?') if hasattr(chunk, 'metadata') else '?'
            
            chunk_texts.append(f"[Source {i}: {doc_name}, Page {page}]\n{content}")
        
        context_text = "\n\n".join(chunk_texts)
        
        prompt = f"""Answer the following question based on the provided context.

Context:
{context_text}

{f"Previous Information:\n{context}\n" if context else ""}

Question: {query}

Instructions:
- Provide a clear, concise answer
- Cite sources using [Source X] format
- If information is insufficient, state what's missing
- Be factual and avoid speculation

Answer:"""

        system_prompt = "You are a financial document analysis expert. Provide accurate, well-cited answers."
        
        answer, _ = self..llm.generate(prompt, system_prompt)
        
        return answer.strip()
    
    def _calculate_confidence(self, chunks: List[Any], answer: str) -> float:
        """
        Calculate confidence score for an answer
        
        Based on:
        - Retrieval scores
        - Answer length and completeness
        - Citation density
        """
        if not chunks:
            return 0.0
        
        # Retrieval score component (0-0.4)
        avg_retrieval_score = sum(
            getattr(chunk, 'score', 0.5) for chunk in chunks
        ) / len(chunks)
        retrieval_component = min(0.4, avg_retrieval_score * 0.5)
        
        # Answer completeness component (0-0.3)
        answer_length = len(answer.split())
        completeness_component = min(0.3, answer_length / 100 * 0.3)
        
        # Citation component (0-0.3)
        citation_count = answer.count('[Source')
        citation_component = min(0.3, citation_count / len(chunks) * 0.3)
        
        confidence = retrieval_component + completeness_component + citation_component
        
        return min(1.0, confidence)
    
    def _synthesize_answer(
        self,
        plan: ExecutionPlan,
        steps: List[ExecutionStep]
    ) -> Tuple[str, float]:
        """
        Synthesize final answer from all execution steps
        
        Returns:
            (final_answer, confidence)
        """
        
        # For simple queries, return the single step answer
        if plan.complexity == QueryComplexity.SIMPLE:
            if steps:
                return steps[0].answer, steps[0].confidence
            return "No answer generated.", 0.0
        
        # For complex queries, synthesize from all steps
        synthesis_prompt = f"""Synthesize a comprehensive answer to the original question based on the following sub-answers.

Original Question: {plan.original_query}

Sub-answers:
"""
        
        for i, step in enumerate(steps, 1):
            synthesis_prompt += f"\n{i}. {step.sub_query.question}\n   Answer: {step.answer}\n"
        
        synthesis_prompt += f"""

Instructions:
- Provide a complete, coherent answer to the original question
- Integrate information from all sub-answers
- Maintain all source citations
- Ensure logical flow and clarity

Final Answer:"""

        system_prompt = "You are an expert at synthesizing information from multiple sources into coherent answers."
        
        final_answer, _ = self..llm.generate(synthesis_prompt, system_prompt)
        
        # Calculate overall confidence (weighted average)
        total_confidence = sum(step.confidence for step in steps) / len(steps)
        
        return final_answer.strip(), total_confidence
    
    def _refine_answer(
        self,
        original_answer: str,
        reflection: ReflectionResult,
        steps: List[ExecutionStep]
    ) -> str:
        """Refine answer based on reflection feedback"""
        
        refinement_prompt = f"""Refine the following answer based on the identified issues.

Original Answer:
{original_answer}

Issues Identified:
{chr(10).join(f"- {issue}" for issue in reflection.issues)}

Suggestions:
{chr(10).join(f"- {suggestion}" for suggestion in reflection.suggestions)}

Available Context:
"""
        
        for i, step in enumerate(steps, 1):
            refinement_prompt += f"\n{i}. {step.answer[:200]}..."
        
        refinement_prompt += """

Instructions:
- Address all identified issues
- Incorporate suggestions
- Maintain factual accuracy
- Keep all valid citations

Refined Answer:"""

        system_prompt = "You are an expert at refining and improving answers based on feedback."
        
        refined_answer, _ = self..llm.generate(refinement_prompt, system_prompt)
        
        return refined_answer.strip()
