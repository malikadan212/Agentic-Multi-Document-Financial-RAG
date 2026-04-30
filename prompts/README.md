# Prompts Directory

This directory contains all prompts used in the Financial RAG System for document analysis and question answering.

## Prompt Files

| File | Purpose |
|------|---------|
| `system_prompts.md` | Core system prompts for RAG generation |
| `rag_prompt_templates.md` | Templates for context injection |
| `citation_prompts.md` | Prompts for citation generation |
| `evaluation_prompts.md` | Prompts used in evaluation metrics |

## Usage

These prompts are used by the `generation/generator.py` module to instruct LLMs on how to:
1. Analyze financial documents
2. Generate accurate answers from context
3. Provide proper citations
4. Maintain factual grounding

## Prompt Engineering Techniques Used

1. **Role-Based Prompting** - Assigning "financial analyst" role
2. **Context Injection** - Structured context from retrieved chunks
3. **Instruction Following** - Clear output format requirements
4. **Citation Enforcement** - Explicit citation format instructions
5. **Grounding Constraints** - "Only answer based on provided context"
