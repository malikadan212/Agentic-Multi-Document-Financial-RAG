# src/evaluation/evaluator.py
"""
Comprehensive Evaluation Framework for RAG System
Implements multiple metrics: Recall@K, Exact Match, F1, ROUGE, BERTScore, Citation Accuracy
"""

# Import necessary libraries
import numpy as np  # For numerical operations
from typing import List, Dict, Tuple, Optional  # For type hints
import pandas as pd  # For data manipulation and analysis
from dataclasses import dataclass, asdict  # For creating data classes
import json  # For JSON serialization
from collections import Counter  # For counting occurrences
import re  # For regular expressions
import logging  # For logging functionality

# Import evaluation metric libraries
from rouge_score import rouge_scorer  # For ROUGE scores (text similarity)
from bert_score import score as bert_score  # For BERTScore (semantic similarity)
import nltk  # Natural Language Toolkit for text processing
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # For BLEU score

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set default logging level to INFO
logger = logging.getLogger(__name__)  # Create logger instance for this module

# Flag to track if NLTK data has been checked/downloaded
_nltk_data_checked = False  # Global flag to avoid repeated downloads

# Function to ensure required NLTK data is available
def _ensure_nltk_data():
    """Lazily download required NLTK data if not present"""
    global _nltk_data_checked  # Access global flag
    if _nltk_data_checked:  # If already checked, return early
        return
    
    # Try to find and download punkt tokenizer if not present
    try:
        nltk.data.find('tokenizers/punkt')  # Check if punkt tokenizer exists
    except LookupError:  # If not found
        try:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)  # Download punkt tokenizer quietly
        except Exception as e:
            logger.warning(f"Could not download NLTK punkt tokenizer: {e}. BLEU scores may not work.")
    
    # Try to find and download punkt_tab tokenizer if not present (optional)
    try:
        nltk.data.find('tokenizers/punkt_tab')  # Check if punkt_tab tokenizer exists
    except LookupError:  # If not found
        try:
            nltk.download('punkt_tab', quiet=True)  # Download punkt_tab tokenizer quietly
        except Exception:
            pass  # Optional, not critical if download fails
    
    _nltk_data_checked = True  # Set flag to True after checking/downloading


# Data class representing a single test case for evaluation
@dataclass
class TestCase:
    """Represents a single test case for evaluation"""
    query: str  # The question/query being tested
    ground_truth_answer: str  # The correct answer
    ground_truth_sources: List[Dict]  # List of source documents that contain the answer [{'doc_name': str, 'page': int}]
    category: str  # Question category: 'factual', 'calculation', 'comparison', 'multi-doc'
    difficulty: str  # Difficulty level: 'easy', 'medium', 'hard'
    
    # Method to convert object to dictionary
    def to_dict(self):
        return asdict(self)  # Use dataclass asdict to convert all fields


# Data class for storing evaluation results for a single test case
@dataclass
class EvaluationResult:
    """Stores evaluation results for a single test case"""
    test_id: int  # Unique identifier for the test case
    query: str  # The question/query
    category: str  # Question category
    
    # Retrieval metrics
    recall_at_5: float  # Recall@K (how many relevant documents were retrieved in top K)
    precision_at_5: float  # Precision@K (how many of top K retrieved documents are relevant)
    mrr: float  # Mean Reciprocal Rank (rank of first relevant document)
    
    # Generation metrics (answer quality)
    exact_match: float  # Whether answer exactly matches ground truth (1.0 or 0.0)
    f1_score: float  # Token-level F1 score between prediction and reference
    rouge_1: float  # ROUGE-1 score (unigram overlap)
    rouge_2: float  # ROUGE-2 score (bigram overlap)
    rouge_l: float  # ROUGE-L score (longest common subsequence)
    bert_score_f1: float  # BERTScore F1 (semantic similarity)
    bleu_score: float  # BLEU score (machine translation metric)
    
    # Citation metrics
    citation_precision: float  # Percentage of citations that are correct
    citation_recall: float  # Percentage of required citations that were included
    false_citation_rate: float  # Percentage of citations that are incorrect
    
    # Performance metrics
    latency_ms: float  # Response time in milliseconds
    tokens_used: int  # Total tokens used (prompt + completion)
    cost_usd: float  # Cost in USD
    
    # Method to convert object to dictionary
    def to_dict(self):
        return asdict(self)  # Use dataclass asdict to convert all fields


# Class for evaluating retrieval quality
class RetrievalEvaluator:
    """Evaluates retrieval quality"""
    
    # Static method to calculate Recall@K
    @staticmethod
    def calculate_recall_at_k(retrieved_docs: List[Dict], 
                              relevant_docs: List[Dict], 
                              k: int = 5) -> float:
        """
        Calculate Recall@K
        
        Args:
            retrieved_docs: List of retrieved document dicts with 'doc_name' and 'page'
            relevant_docs: List of ground truth relevant documents
            k: Number of top results to consider
            
        Returns:
            Recall@K score (0-1)
        """
        if not relevant_docs:  # If no relevant documents, recall is 0
            return 0.0
        
        # Create sets of (doc_name, page) tuples for efficient comparison
        # Consider only top k retrieved documents
        retrieved_set = {(d['doc_name'], d['page']) for d in retrieved_docs[:k]}
        relevant_set = {(d['doc_name'], d['page']) for d in relevant_docs}
        
        # Find overlap between retrieved and relevant documents
        overlap = retrieved_set.intersection(relevant_set)
        
        # Recall = relevant documents retrieved / total relevant documents
        recall = len(overlap) / len(relevant_set)
        
        return recall
    
    # Static method to calculate Precision@K
    @staticmethod
    def calculate_precision_at_k(retrieved_docs: List[Dict],
                                 relevant_docs: List[Dict],
                                 k: int = 5) -> float:
        """Calculate Precision@K"""
        if not retrieved_docs:  # If no documents retrieved, precision is 0
            return 0.0
        
        # Create sets of (doc_name, page) tuples
        retrieved_set = {(d['doc_name'], d['page']) for d in retrieved_docs[:k]}
        relevant_set = {(d['doc_name'], d['page']) for d in relevant_docs}
        
        # Find overlap between retrieved and relevant documents
        overlap = retrieved_set.intersection(relevant_set)
        
        # Precision = relevant documents retrieved / total retrieved documents (up to k)
        precision = len(overlap) / min(k, len(retrieved_set))
        
        return precision
    
    # Static method to calculate Mean Reciprocal Rank (MRR)
    @staticmethod
    def calculate_mrr(retrieved_docs: List[Dict],
                     relevant_docs: List[Dict]) -> float:
        """
        Calculate Mean Reciprocal Rank
        
        Returns:
            MRR score (0-1)
        """
        # Create set of relevant documents for fast lookup
        relevant_set = {(d['doc_name'], d['page']) for d in relevant_docs}
        
        # Find the rank of the first relevant document (1-indexed)
        for rank, doc in enumerate(retrieved_docs, start=1):
            if (doc['doc_name'], doc['page']) in relevant_set:
                return 1.0 / rank  # Reciprocal of rank
        
        return 0.0  # No relevant documents found


# Class for evaluating generation quality
class GenerationEvaluator:
    """Evaluates generation quality"""
    
    # Initialize with ROUGE scorer
    def __init__(self):
        # Create ROUGE scorer for multiple ROUGE variants
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],  # ROUGE variants to compute
            use_stemmer=True  # Use stemming for better matching
        )
    
    # Method to calculate Exact Match score
    def calculate_exact_match(self, prediction: str, reference: str) -> float:
        """
        Calculate Exact Match score
        
        Args:
            prediction: Generated answer
            reference: Ground truth answer
            
        Returns:
            1.0 if exact match (case-insensitive), else 0.0
        """
        # Normalize both texts for case-insensitive comparison
        pred_normalized = self._normalize_text(prediction)
        ref_normalized = self._normalize_text(reference)
        
        # Return 1.0 if texts match exactly, 0.0 otherwise
        return 1.0 if pred_normalized == ref_normalized else 0.0
    
    # Method to calculate token-level F1 score
    def calculate_f1_score(self, prediction: str, reference: str) -> float:
        """
        Calculate token-level F1 score
        
        Args:
            prediction: Generated answer
            reference: Ground truth answer
            
        Returns:
            F1 score (0-1)
        """
        # Convert normalized texts to sets of tokens (words)
        pred_tokens = set(self._normalize_text(prediction).split())
        ref_tokens = set(self._normalize_text(reference).split())
        
        # Handle empty reference
        if not ref_tokens:
            return 0.0
        
        # Find common tokens between prediction and reference
        common = pred_tokens.intersection(ref_tokens)
        
        # If no common tokens, F1 is 0
        if not common:
            return 0.0
        
        # Calculate precision: common tokens / total prediction tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        
        # Calculate recall: common tokens / total reference tokens
        recall = len(common) / len(ref_tokens) if ref_tokens else 0.0
        
        # Handle division by zero
        if precision + recall == 0:
            return 0.0
        
        # Calculate F1 score: harmonic mean of precision and recall
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    # Method to calculate ROUGE scores
    def calculate_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores
        
        Returns:
            Dictionary with rouge1, rouge2, rougeL F1 scores
        """
        # Use ROUGE scorer to compute scores
        scores = self.rouge_scorer.score(reference, prediction)  # Note: reference first, then prediction
        
        # Extract F1 scores for each ROUGE variant
        return {
            'rouge1': scores['rouge1'].fmeasure,  # Unigram overlap F1
            'rouge2': scores['rouge2'].fmeasure,  # Bigram overlap F1
            'rougeL': scores['rougeL'].fmeasure  # Longest common subsequence F1
        }
    
    # Method to calculate BERTScore for semantic similarity
    def calculate_bert_score(self, predictions: List[str], 
                            references: List[str]) -> List[float]:
        """
        Calculate BERTScore for semantic similarity
        
        Args:
            predictions: List of generated answers
            references: List of ground truth answers
            
        Returns:
            List of BERTScore F1 values
        """
        try:
            # Calculate BERTScore using BERT model
            P, R, F1 = bert_score(
                predictions,  # List of predictions
                references,  # List of references
                lang='en',  # Language (English)
                verbose=False,  # Don't show progress bar
                device='cpu'  # Use CPU for computation
            )
            return F1.tolist()  # Convert tensor to list
        except Exception as e:
            # Log warning if BERTScore calculation fails
            logger.warning(f"BERTScore calculation failed: {str(e)}")
            return [0.0] * len(predictions)  # Return zeros as fallback
    
    # Method to calculate BLEU score
    def calculate_bleu_score(self, prediction: str, reference: str) -> float:
        """
        Calculate BLEU score
        
        Args:
            prediction: Generated answer
            reference: Ground truth answer
            
        Returns:
            BLEU score (0-1)
        """
        # Ensure NLTK data is available (download if needed)
        _ensure_nltk_data()
        
        try:
            # Tokenize both texts (convert to lowercase first)
            pred_tokens = nltk.word_tokenize(prediction.lower())
            ref_tokens = nltk.word_tokenize(reference.lower())
            
            # Create smoothing function to handle zero matches
            smoothing = SmoothingFunction().method1
            
            # Calculate BLEU score
            score = sentence_bleu(
                [ref_tokens],  # List of reference token lists (single reference)
                pred_tokens,  # Prediction tokens
                smoothing_function=smoothing  # Apply smoothing
            )
            return score
        except Exception as e:
            # Log warning and return 0.0 if calculation fails
            logger.warning(f"BLEU score calculation failed: {e}")
            return 0.0
    
    # Helper method to normalize text for comparison
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation except $, %, and . (important for financial data)
        text = re.sub(r'[^\w\s\$\%\.]', '', text)
        # Remove extra whitespace and normalize spacing
        text = ' '.join(text.split())
        return text


# Class for evaluating citation quality
class CitationEvaluator:
    """Evaluates citation quality and accuracy"""
    
    # Static method to calculate citation metrics
    @staticmethod
    def calculate_citation_metrics(generated_citations: List[Dict],
                                   ground_truth_sources: List[Dict]) -> Dict[str, float]:
        """
        Calculate citation precision, recall, and false citation rate
        
        Args:
            generated_citations: Citations from generated response
            ground_truth_sources: Ground truth source documents
            
        Returns:
            Dictionary with precision, recall, false_citation_rate
        """
        # Handle case with no generated citations
        if not generated_citations:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'false_citation_rate': 0.0
            }
        
        # Convert to sets of (doc_name, page) tuples for efficient comparison
        # Filter out invalid citations (citations marked as not valid)
        gen_set = {(c['doc_name'], c['page']) for c in generated_citations if c.get('valid', True)}
        gt_set = {(s['doc_name'], s['page']) for s in ground_truth_sources}
        
        # Calculate metrics
        correct_citations = gen_set.intersection(gt_set)  # Citations that match ground truth
        
        # Precision = correct citations / total generated citations
        precision = len(correct_citations) / len(gen_set) if gen_set else 0.0
        
        # Recall = correct citations / total ground truth sources
        recall = len(correct_citations) / len(gt_set) if gt_set else 0.0
        
        # False citation rate = incorrect citations / total generated citations
        false_citations = gen_set - gt_set  # Citations not in ground truth
        false_citation_rate = len(false_citations) / len(gen_set) if gen_set else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'false_citation_rate': false_citation_rate
        }


# Main evaluation class that orchestrates all evaluation metrics
class ComprehensiveEvaluator:
    """
    Main evaluation class that orchestrates all evaluation metrics
    """
    
    # Initialize with component evaluators
    def __init__(self):
        self.retrieval_eval = RetrievalEvaluator()  # For retrieval metrics
        self.generation_eval = GenerationEvaluator()  # For generation metrics
        self.citation_eval = CitationEvaluator()  # For citation metrics
        self.results = []  # List to store EvaluationResult objects
    
    # Method to evaluate a single test case
    def evaluate_test_case(self,
                          test_case: TestCase,
                          retrieved_docs: List[Dict],
                          generated_response: str,
                          generated_citations: List[Dict],
                          latency_ms: float,
                          tokens_used: int,
                          cost_usd: float) -> EvaluationResult:
        """
        Evaluate a single test case across all metrics
        
        Args:
            test_case: TestCase object
            retrieved_docs: Retrieved documents from retrieval system
            generated_response: Generated answer text
            generated_citations: Extracted citations
            latency_ms: Response time in milliseconds
            tokens_used: Total tokens used
            cost_usd: Cost in USD
            
        Returns:
            EvaluationResult object
        """
        # 1. Calculate retrieval metrics
        recall_5 = self.retrieval_eval.calculate_recall_at_k(
            retrieved_docs, test_case.ground_truth_sources, k=5  # Recall@5
        )
        precision_5 = self.retrieval_eval.calculate_precision_at_k(
            retrieved_docs, test_case.ground_truth_sources, k=5  # Precision@5
        )
        mrr = self.retrieval_eval.calculate_mrr(
            retrieved_docs, test_case.ground_truth_sources  # Mean Reciprocal Rank
        )
        
        # 2. Calculate generation metrics
        em = self.generation_eval.calculate_exact_match(
            generated_response, test_case.ground_truth_answer  # Exact Match
        )
        f1 = self.generation_eval.calculate_f1_score(
            generated_response, test_case.ground_truth_answer  # F1 Score
        )
        
        # ROUGE scores
        rouge_scores = self.generation_eval.calculate_rouge_scores(
            generated_response, test_case.ground_truth_answer  # ROUGE-1,2,L
        )
        
        # BLEU score
        bleu = self.generation_eval.calculate_bleu_score(
            generated_response, test_case.ground_truth_answer  # BLEU
        )
        
        # BERTScore (calculated in batch mode, but using single for consistency)
        bert_scores = self.generation_eval.calculate_bert_score(
            [generated_response], [test_case.ground_truth_answer]  # BERTScore
        )
        bert_f1 = bert_scores[0] if bert_scores else 0.0  # Extract single score
        
        # 3. Calculate citation metrics
        citation_metrics = self.citation_eval.calculate_citation_metrics(
            generated_citations, test_case.ground_truth_sources  # Citation precision/recall
        )
        
        # 4. Create EvaluationResult object with all metrics
        result = EvaluationResult(
            test_id=len(self.results),  # Use result count as ID
            query=test_case.query,  # Original query
            category=test_case.category,  # Question category
            
            # Retrieval metrics
            recall_at_5=recall_5,
            precision_at_5=precision_5,
            mrr=mrr,
            
            # Generation metrics
            exact_match=em,
            f1_score=f1,
            rouge_1=rouge_scores['rouge1'],
            rouge_2=rouge_scores['rouge2'],
            rouge_l=rouge_scores['rougeL'],
            bert_score_f1=bert_f1,
            bleu_score=bleu,
            
            # Citation metrics
            citation_precision=citation_metrics['precision'],
            citation_recall=citation_metrics['recall'],
            false_citation_rate=citation_metrics['false_citation_rate'],
            
            # Performance metrics
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd
        )
        
        # Store result
        self.results.append(result)
        return result
    
    # Method to calculate aggregate metrics across all test cases
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate metrics across all test cases"""
        if not self.results:  # If no results, return empty dict
            return {}
        
        # Convert results to DataFrame for easy aggregation
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Calculate mean values for each metric
        aggregates = {
            # Retrieval metrics averages
            'avg_recall@5': df['recall_at_5'].mean(),
            'avg_precision@5': df['precision_at_5'].mean(),
            'avg_mrr': df['mrr'].mean(),
            
            # Generation metrics averages
            'avg_exact_match': df['exact_match'].mean(),
            'avg_f1': df['f1_score'].mean(),
            'avg_rouge1': df['rouge_1'].mean(),
            'avg_rouge2': df['rouge_2'].mean(),
            'avg_rougeL': df['rouge_l'].mean(),
            'avg_bert_score': df['bert_score_f1'].mean(),
            'avg_bleu': df['bleu_score'].mean(),
            
            # Citation metrics averages
            'avg_citation_precision': df['citation_precision'].mean(),
            'avg_citation_recall': df['citation_recall'].mean(),
            'avg_false_citation_rate': df['false_citation_rate'].mean(),
            
            # Performance totals
            'avg_latency_ms': df['latency_ms'].mean(),
            'total_cost_usd': df['cost_usd'].sum(),
            'total_tokens': df['tokens_used'].sum()
        }
        
        # 5. Calculate category-wise breakdowns
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]  # Filter by category
            aggregates[f'{category}_f1'] = cat_df['f1_score'].mean()  # Category F1
            aggregates[f'{category}_recall@5'] = cat_df['recall_at_5'].mean()  # Category Recall@5
        
        return aggregates
    
    # Method to save evaluation results to JSON file
    def save_results(self, output_path: str):
        """Save evaluation results to JSON"""
        # Create dictionary with all results
        results_dict = {
            'individual_results': [r.to_dict() for r in self.results],  # All individual results
            'aggregate_metrics': self.get_aggregate_metrics()  # Aggregated metrics
        }
        
        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)  # Pretty print with indentation
        
        logger.info(f"✅ Saved evaluation results to {output_path}")  # Log success
    
    # Method to generate human-readable evaluation report
    def generate_report(self) -> str:
        """Generate a human-readable evaluation report"""
        aggregates = self.get_aggregate_metrics()  # Get aggregated metrics
        
        # Create formatted report string
        report = f"""
{'='*70}
EVALUATION REPORT
{'='*70}

RETRIEVAL METRICS:
  Recall@5:           {aggregates['avg_recall@5']:.3f}
  Precision@5:        {aggregates['avg_precision@5']:.3f}
  Mean Reciprocal Rank: {aggregates['avg_mrr']:.3f}

GENERATION METRICS:
  Exact Match:        {aggregates['avg_exact_match']:.3f}
  F1 Score:           {aggregates['avg_f1']:.3f}
  ROUGE-1:            {aggregates['avg_rouge1']:.3f}
  ROUGE-2:            {aggregates['avg_rouge2']:.3f}
  ROUGE-L:            {aggregates['avg_rougeL']:.3f}
  BERTScore F1:       {aggregates['avg_bert_score']:.3f}
  BLEU:               {aggregates['avg_bleu']:.3f}

CITATION METRICS:
  Precision:          {aggregates['avg_citation_precision']:.3f}
  Recall:             {aggregates['avg_citation_recall']:.3f}
  False Citation Rate: {aggregates['avg_false_citation_rate']:.3f}

PERFORMANCE:
  Avg Latency:        {aggregates['avg_latency_ms']:.2f} ms
  Total Tokens:       {aggregates['total_tokens']:.0f}
  Total Cost:         ${aggregates['total_cost_usd']:.4f}

TEST CASES:         {len(self.results)}
{'='*70}
"""
        
        return report


# Example Usage
if __name__ == "__main__":
    # Create a test case with sample data
    test_case = TestCase(
        query="What was Apple's revenue in Q3 2023?",  # Test query
        ground_truth_answer="Apple's revenue in Q3 2023 was $81.8 billion.",  # Expected answer
        ground_truth_sources=[
            {'doc_name': 'Apple_10Q_Q3_2023', 'page': 3}  # Expected source
        ],
        category='factual',  # Question category
        difficulty='easy'  # Difficulty level
    )
    
    # Simulate retrieval results (what the retriever found)
    retrieved_docs = [
        {'doc_name': 'Apple_10Q_Q3_2023', 'page': 3},  # Correct document/page
        {'doc_name': 'Apple_10Q_Q3_2023', 'page': 5},  # Wrong page but correct document
    ]
    
    # Simulate generated response with citation
    generated_response = "According to the quarterly report, Apple's revenue for Q3 2023 was $81.8 billion [Source: Apple_10Q_Q3_2023, Page 3]."
    
    # Simulate extracted citations
    generated_citations = [
        {'doc_name': 'Apple_10Q_Q3_2023', 'page': 3, 'valid': True}  # Citation from generated response
    ]
    
    # Create evaluator and run evaluation
    evaluator = ComprehensiveEvaluator()
    result = evaluator.evaluate_test_case(
        test_case=test_case,  # Test case definition
        retrieved_docs=retrieved_docs,  # Retrieved documents
        generated_response=generated_response,  # Generated answer
        generated_citations=generated_citations,  # Generated citations
        latency_ms=1500,  # Simulated latency (1.5 seconds)
        tokens_used=250,  # Simulated token usage
        cost_usd=0.005  # Simulated cost
    )
    
    # Print evaluation report
    print(evaluator.generate_report())