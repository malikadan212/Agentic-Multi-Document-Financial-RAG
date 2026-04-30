# Evaluation Prompts

## Overview

These prompts are used in `src/evaluation/evaluator.py` for assessing RAG system performance.

---

## Answer Quality Evaluation Prompt

```
Evaluate the quality of the generated answer compared to the reference answer.

GENERATED ANSWER:
{generated}

REFERENCE ANSWER:
{reference}

CONTEXT PROVIDED:
{context}

Evaluate on:
1. Factual Accuracy (0-1): Does the answer contain correct facts?
2. Completeness (0-1): Does it cover all key points from the reference?
3. Relevance (0-1): Is the answer focused on the question asked?
4. Citation Quality (0-1): Are citations properly formatted and accurate?
```

---

## Retrieval Evaluation Metrics

### Recall@K Prompt
```
Given the query, evaluate if the retrieved documents contain the answer.

Query: {query}
Retrieved Documents: {retrieved}
Ground Truth: {ground_truth}

Calculate:
- Recall@5: What percentage of relevant documents were retrieved in top 5?
- Recall@10: What percentage of relevant documents were retrieved in top 10?
```

---

## Citation Accuracy Evaluation

```
Verify that each citation in the answer can be traced back to the source documents.

ANSWER WITH CITATIONS:
{answer}

SOURCE DOCUMENTS:
{sources}

For each citation, verify:
1. Does the cited document exist in the sources?
2. Is the page number valid?
3. Does the cited content match the source?
```

---

## Metrics Implemented

| Metric | Description | Range |
|--------|-------------|-------|
| Recall@K | Retrieval completeness | 0-1 |
| Exact Match | Character-level match | 0-1 |
| F1 Score | Token overlap | 0-1 |
| ROUGE-L | Longest common subsequence | 0-1 |
| BERTScore | Semantic similarity | 0-1 |
| BLEU | N-gram precision | 0-1 |
| Citation Accuracy | Citation validation | 0-1 |

---

## Usage

```python
from evaluation.evaluator import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator()
results = evaluator.evaluate_response(
    query=query,
    generated_answer=answer,
    reference_answer=ground_truth,
    retrieved_chunks=chunks
)
```
