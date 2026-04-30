# RAG Prompt Templates

## 1. Standard Query Template

```python
STANDARD_RAG_TEMPLATE = """
Based on the following context from financial documents, answer the question.

CONTEXT:
{context}

QUESTION: {question}

Provide a detailed answer with citations using [Source: DocumentName, Page X] format.
If the answer cannot be found in the context, state "I cannot find this information in the provided documents."
"""
```

## 2. Financial Calculation Template

```python
CALCULATION_TEMPLATE = """
You are a financial analyst. Using the provided data:

FINANCIAL DATA:
{context}

CALCULATION REQUEST: {question}

Steps to follow:
1. Identify all relevant numerical values
2. Show your calculation steps
3. Provide the final result
4. Cite where each number came from [Source: DocName, Page X]
"""
```

## 3. Comparison Template

```python
COMPARISON_TEMPLATE = """
Compare the following aspects from the financial documents:

DOCUMENTS:
{context}

COMPARISON REQUEST: {question}

Structure your response as:
- Key metric 1: [Value A] vs [Value B] [Citations]
- Key metric 2: [Value A] vs [Value B] [Citations]
- Summary of differences
"""
```

## 4. Summary Template

```python
SUMMARY_TEMPLATE = """
Create a summary of the following financial document content:

CONTENT:
{context}

SUMMARY REQUEST: {question}

Provide:
1. Key highlights (bullet points)
2. Important figures
3. Notable trends or observations
All with proper citations.
"""
```

## 5. Multi-Document Synthesis Template

```python
SYNTHESIS_TEMPLATE = """
Synthesize information from multiple documents to answer:

DOCUMENTS:
{context}

SYNTHESIS QUESTION: {question}

Combine insights from all provided sources, noting any:
- Agreements between sources
- Contradictions or discrepancies
- Complementary information
Include citations for each source used.
"""
```

---

## Template Variables

| Variable | Description |
|----------|-------------|
| `{context}` | Retrieved document chunks with metadata |
| `{question}` | User's query |
| `{doc_name}` | Source document name |
| `{page_num}` | Page number in source |

## Usage in Code

These templates are used in `src/generation/generator.py`:

```python
prompt = STANDARD_RAG_TEMPLATE.format(
    context=formatted_context,
    question=user_query
)
```
