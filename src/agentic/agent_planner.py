# src/agentic/agent_planner.py
"""
Agent Planner
Analyzes queries and creates execution plans for complex questions
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"  # Single-step retrieval
    MODERATE = "moderate"  # 2-3 steps
    COMPLEX = "complex"  # 4+ steps, requires decomposition


class QueryType(Enum):
    """Types of queries the system can handle"""
    FACTUAL = "factual"  # "What is X?"
    COMPARISON = "comparison"  # "Compare X and Y"
    AGGREGATION = "aggregation"  # "What is the total/average?"
    TREND = "trend"  # "How has X changed?"
    MULTI_HOP = "multi_hop"  # Requires multiple retrieval steps
    CALCULATION = "calculation"  # Requires computation
    TEMPORAL = "temporal"  # Time-based query


@dataclass
class SubQuery:
    """Represents a sub-question in a decomposed query"""
    question: str
    query_type: QueryType
    dependencies: List[int]  # Indices of sub-queries this depends on
    priority: int  # Execution order
    context_needed: bool = False  # Whether it needs results from previous steps


@dataclass
class ExecutionPlan:
    """Plan for executing a complex query"""
    original_query: str
    complexity: QueryComplexity
    query_type: QueryType
    sub_queries: List[SubQuery]
    requires_calculation: bool = False
    requires_aggregation: bool = False
    requires_temporal_reasoning: bool = False
    estimated_steps: int = 1


class AgentPlanner:
    """
    Plans query execution strategies
    Decomposes complex queries into manageable sub-questions
    """
    
    # Keywords for query type detection
    COMPARISON_KEYWORDS = [
        'compare', 'comparison', 'versus', 'vs', 'difference', 'contrast',
        'better', 'worse', 'higher', 'lower', 'between'
    ]
    
    AGGREGATION_KEYWORDS = [
        'total', 'sum', 'average', 'mean', 'count', 'how many',
        'all', 'every', 'each', 'overall'
    ]
    
    TREND_KEYWORDS = [
        'trend', 'change', 'evolution', 'growth', 'decline',
        'over time', 'progression', 'history', 'trajectory'
    ]
    
    TEMPORAL_KEYWORDS = [
        'when', 'date', 'year', 'quarter', 'month', 'period',
        'latest', 'recent', 'current', 'past', 'future'
    ]
    
    CALCULATION_KEYWORDS = [
        'calculate', 'compute', 'what is', 'how much',
        'percentage', 'ratio', 'rate', 'increase', 'decrease'
    ]
    
    MULTI_HOP_INDICATORS = [
        'and then', 'after that', 'based on', 'using',
        'first', 'second', 'finally', 'also'
    ]
    
    def __init__(self, llm=None):
        """
        Initialize planner
        
        Args:
            llm: Optional LLM for advanced query decomposition
        """
        self.llm = llm
    
    def analyze_query(self, query: str) -> ExecutionPlan:
        """
        Analyze query and create execution plan
        
        Args:
            query: User query string
            
        Returns:
            ExecutionPlan with decomposed sub-queries
        """
        query_lower = query.lower()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        # Detect complexity
        complexity = self._detect_complexity(query_lower, query_type)
        
        # Decompose if complex
        if complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]:
            sub_queries = self._decompose_query(query, query_type, complexity)
        else:
            # Simple query - single step
            sub_queries = [SubQuery(
                question=query,
                query_type=query_type,
                dependencies=[],
                priority=1,
                context_needed=False
            )]
        
        # Detect special requirements
        requires_calculation = any(kw in query_lower for kw in self.CALCULATION_KEYWORDS)
        requires_aggregation = any(kw in query_lower for kw in self.AGGREGATION_KEYWORDS)
        requires_temporal = any(kw in query_lower for kw in self.TEMPORAL_KEYWORDS)
        
        plan = ExecutionPlan(
            original_query=query,
            complexity=complexity,
            query_type=query_type,
            sub_queries=sub_queries,
            requires_calculation=requires_calculation,
            requires_aggregation=requires_aggregation,
            requires_temporal_reasoning=requires_temporal,
            estimated_steps=len(sub_queries)
        )
        
        logger.info(f"Created execution plan: {complexity.value} query with {len(sub_queries)} steps")
        
        return plan
    
    def _detect_query_type(self, query_lower: str) -> QueryType:
        """Detect the primary type of query"""
        
        # Check for multi-hop indicators first
        if any(indicator in query_lower for indicator in self.MULTI_HOP_INDICATORS):
            return QueryType.MULTI_HOP
        
        # Check for specific types
        if any(kw in query_lower for kw in self.COMPARISON_KEYWORDS):
            return QueryType.COMPARISON
        
        if any(kw in query_lower for kw in self.TREND_KEYWORDS):
            return QueryType.TREND
        
        if any(kw in query_lower for kw in self.AGGREGATION_KEYWORDS):
            return QueryType.AGGREGATION
        
        if any(kw in query_lower for kw in self.CALCULATION_KEYWORDS):
            return QueryType.CALCULATION
        
        if any(kw in query_lower for kw in self.TEMPORAL_KEYWORDS):
            return QueryType.TEMPORAL
        
        # Default to factual
        return QueryType.FACTUAL
    
    def _detect_complexity(self, query_lower: str, query_type: QueryType) -> QueryComplexity:
        """Detect query complexity level"""
        
        complexity_score = 0
        
        # Multi-hop queries are complex
        if query_type == QueryType.MULTI_HOP:
            complexity_score += 3
        
        # Comparison and aggregation add complexity
        if query_type in [QueryType.COMPARISON, QueryType.AGGREGATION]:
            complexity_score += 2
        
        # Trend and calculation add moderate complexity
        if query_type in [QueryType.TREND, QueryType.CALCULATION]:
            complexity_score += 1
        
        # Multiple entities/concepts increase complexity
        entity_indicators = ['and', ',', 'both', 'all', 'each']
        complexity_score += sum(1 for ind in entity_indicators if ind in query_lower)
        
        # Multiple time periods increase complexity
        time_indicators = ['q1', 'q2', 'q3', 'q4', '2024', '2025', 'year', 'quarter']
        time_count = sum(1 for ind in time_indicators if ind in query_lower)
        if time_count > 1:
            complexity_score += 2
        
        # Question length as proxy for complexity
        word_count = len(query_lower.split())
        if word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        # Classify based on score
        if complexity_score >= 6:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 4:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _decompose_query(
        self, 
        query: str, 
        query_type: QueryType,
        complexity: QueryComplexity
    ) -> List[SubQuery]:
        """
        Decompose complex query into sub-queries
        
        Uses rule-based decomposition with optional LLM enhancement
        """
        
        # Try LLM-based decomposition if available
        if self.llm:
            try:
                return self._llm_decompose(query, query_type)
            except Exception as e:
                logger.warning(f"LLM decomposition failed: {e}, falling back to rule-based")
        
        # Rule-based decomposition
        return self._rule_based_decompose(query, query_type)
    
    def _rule_based_decompose(self, query: str, query_type: QueryType) -> List[SubQuery]:
        """Rule-based query decomposition"""
        
        sub_queries = []
        query_lower = query.lower()
        
        if query_type == QueryType.COMPARISON:
            sub_queries = [
                SubQuery(
                    question=query,
                    query_type=QueryType.FACTUAL,
                    dependencies=[],
                    priority=1,
                    context_needed=False
                ),
                SubQuery(
                    question=f"Based on the retrieved information, compare and contrast: {query}",
                    query_type=QueryType.COMPARISON,
                    dependencies=[0],
                    priority=2,
                    context_needed=True
                )
            ]

        elif query_type == QueryType.TREND:
            sub_queries = [
                SubQuery(
                    question=query,
                    query_type=QueryType.FACTUAL,
                    dependencies=[],
                    priority=1,
                    context_needed=False
                ),
                SubQuery(
                    question=f"Analyze the trend and changes shown in the retrieved data for: {query}",
                    query_type=QueryType.TREND,
                    dependencies=[0],
                    priority=2,
                    context_needed=True
                )
            ]

        elif query_type == QueryType.AGGREGATION:
            sub_queries = [
                SubQuery(
                    question=query,
                    query_type=QueryType.FACTUAL,
                    dependencies=[],
                    priority=1,
                    context_needed=False
                ),
                SubQuery(
                    question=f"Summarize and aggregate the retrieved figures to answer: {query}",
                    query_type=QueryType.CALCULATION,
                    dependencies=[0],
                    priority=2,
                    context_needed=True
                )
            ]

        elif query_type == QueryType.MULTI_HOP:
            sub_queries = [
                SubQuery(
                    question=query,
                    query_type=QueryType.FACTUAL,
                    dependencies=[],
                    priority=1,
                    context_needed=False
                ),
                SubQuery(
                    question=f"Using the retrieved context, provide a complete answer to: {query}",
                    query_type=QueryType.FACTUAL,
                    dependencies=[0],
                    priority=2,
                    context_needed=True
                )
            ]

        else:
            sub_queries = [SubQuery(
                question=query,
                query_type=query_type,
                dependencies=[],
                priority=1,
                context_needed=False
            )]
        
        return sub_queries
    
    def _llm_decompose(self, query: str, query_type: QueryType) -> List[SubQuery]:
        """
        Use LLM to decompose query into sub-questions
        More flexible than rule-based approach
        """
        
        prompt = f"""Decompose this complex query into 2-4 simpler sub-questions that can be answered sequentially.

Original Query: {query}
Query Type: {query_type.value}

Requirements:
1. Each sub-question should be answerable independently or depend on previous answers
2. Order sub-questions logically (dependencies first)
3. Keep sub-questions clear and specific
4. Format as numbered list

Sub-questions:"""

        response, _ = self.llm.generate(
            prompt, 
            system_prompt="You are a query decomposition expert. Break complex questions into simpler sub-questions."
        )
        
        # Parse response into SubQuery objects
        lines = response.strip().split('\n')
        sub_queries = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 10:
                continue
            
            # Remove numbering
            question = line.lstrip('0123456789.-) ')
            
            if question:
                # Determine dependencies (simple heuristic)
                dependencies = []
                if i > 0 and any(word in question.lower() for word in ['compare', 'based on', 'using', 'from above']):
                    dependencies = list(range(i))
                
                sub_queries.append(SubQuery(
                    question=question,
                    query_type=QueryType.FACTUAL,  # Default type
                    dependencies=dependencies,
                    priority=i + 1,
                    context_needed=len(dependencies) > 0
                ))
        
        # Fallback to original query if parsing failed
        if not sub_queries:
            sub_queries = [SubQuery(
                question=query,
                query_type=query_type,
                dependencies=[],
                priority=1,
                context_needed=False
            )]
        
        return sub_queries
    
    def should_decompose(self, plan: ExecutionPlan) -> bool:
        """Determine if query should be decomposed"""
        return (
            plan.complexity != QueryComplexity.SIMPLE and
            len(plan.sub_queries) > 1
        )
    
    def get_next_sub_query(
        self, 
        plan: ExecutionPlan, 
        completed_indices: List[int]
    ) -> Optional[SubQuery]:
        """
        Get next sub-query to execute based on dependencies
        
        Args:
            plan: Execution plan
            completed_indices: Indices of completed sub-queries
            
        Returns:
            Next sub-query to execute, or None if all complete
        """
        for i, sub_query in enumerate(plan.sub_queries):
            if i in completed_indices:
                continue
            
            # Check if all dependencies are satisfied
            if all(dep in completed_indices for dep in sub_query.dependencies):
                return sub_query
        
        return None
