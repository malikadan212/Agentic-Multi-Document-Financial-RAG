# Agentic Multi-Document Financial RAG System with Temporal Reasoning, Knowledge Graph Integration, and Hallucination Grounding

**Course:** Generative AI  
**Instructor:** Sir Akhtar Jamil  
**Team Members:** Adan Malik, Shayan Ahmed, Bilal Raza  
**Institution:** FAST-NUCES  
**Date:** April 2026

---

## 1. Introduction

Sir, we are proposing to build an advanced Retrieval-Augmented Generation (RAG) system specifically designed for financial document analysis. While traditional RAG systems have shown promise in question-answering tasks, they face significant limitations when dealing with complex financial documents that require temporal understanding, cross-document reasoning, and factual accuracy. Our project aims to address these challenges by introducing four key innovations that go beyond standard RAG implementations.

The financial domain presents unique challenges: documents contain time-sensitive information, entities and metrics are interconnected across multiple reports, and accuracy is critical as hallucinations can lead to serious consequences. We believe that by incorporating agentic behavior, temporal reasoning, knowledge graph integration, and hallucination grounding mechanisms, we can create a system that not only retrieves and generates responses but does so with intelligence, context-awareness, and verifiable accuracy.

## 2. Problem Statement

Current RAG systems, while effective for basic question-answering, struggle with:

1. **Temporal Blindness**: They cannot understand time-based queries like "How has revenue changed over the last 3 quarters?" or compare metrics across different time periods.

2. **Isolated Reasoning**: Each query is processed independently without understanding relationships between entities, metrics, or documents.

3. **Hallucination Issues**: LLMs often generate plausible-sounding but factually incorrect information, especially when dealing with specific numbers and financial data.

4. **Passive Retrieval**: Traditional systems retrieve documents based on similarity alone, without intelligent query decomposition or multi-step reasoning.

Sir, we want to solve these problems by building a system that thinks, reasons temporally, understands relationships, and grounds every claim in verifiable sources.

## 3. Proposed Methodology

### 3.1 Agentic Architecture

We will implement an autonomous agent that can:
- **Decompose complex queries** into sub-questions (e.g., "Compare Q1 and Q2 revenue" → "What was Q1 revenue?" + "What was Q2 revenue?" + "Calculate difference")
- **Plan retrieval strategies** dynamically based on query type
- **Self-reflect** on generated responses to ensure completeness
- **Adapt context** by retrieving additional documents if initial results are insufficient

This agentic behavior will be implemented using a planning module that analyzes queries, determines the required steps, and orchestrates the retrieval and generation process intelligently.

### 3.2 Temporal Reasoning

Sir, we will add temporal awareness by:
- **Extracting temporal entities** (dates, quarters, years) from both queries and documents
- **Building a temporal index** that maps content to time periods
- **Implementing temporal operators** for comparison (before, after, during, trend analysis)
- **Time-aware retrieval** that prioritizes documents from relevant time periods

For example, when asked "What was the revenue trend in 2023?", the system will identify all Q1-Q4 2023 documents, extract revenue figures, and analyze the progression over time.

### 3.3 Knowledge Graph Integration

We will construct a knowledge graph where:
- **Nodes represent entities** (companies, products, metrics, time periods)
- **Edges represent relationships** (revenue_of, compared_to, increased_by)
- **Graph traversal enables reasoning** across documents

This allows the system to answer complex queries like "How does marketing spend correlate with revenue growth?" by traversing the graph to find relationships between these entities across multiple documents.

### 3.4 Hallucination Grounding

To ensure factual accuracy, we will implement:
- **Source verification**: Every claim must be traceable to a specific document and page
- **Confidence scoring**: Based on retrieval scores, citation density, and cross-document validation
- **Fact-checking module**: Compares generated numbers against retrieved context
- **Citation enforcement**: The system must provide `[Source: Document, Page X]` for all factual claims

If the system cannot find supporting evidence in the retrieved documents, it will explicitly state "I cannot find this information in the provided documents" rather than hallucinating an answer.

## 4. Technical Implementation

### 4.1 Current Progress (50% Complete)

Sir, we have already implemented:

1. **Multi-format document processing** with OCR support for PDFs, Excel, and CSV files
2. **Advanced embedding models** (MiniLM, MPNet, RoBERTa) for semantic search
3. **Hybrid vector stores** (FAISS and ChromaDB) for efficient retrieval
4. **Multi-LLM integration** supporting Groq, Google Gemini, OpenAI, Anthropic, and Cohere
5. **Citation extraction and validation** system
6. **Real-time streaming responses** with confidence scoring
7. **Web interface** with professional UI and analytics dashboard

The system is fully functional and can process documents, answer queries, and provide cited responses. We have tested it on a dataset of 27,283 banking document chunks.

### 4.2 Remaining Work (Task-2)

For the second phase, we will focus on implementing the four core innovations:

1. **Agentic Module**: Query decomposition, planning, and self-reflection mechanisms
2. **Temporal Engine**: Time-aware indexing, temporal entity extraction, and trend analysis
3. **Knowledge Graph**: Entity extraction, relationship mapping, and graph-based reasoning
4. **Hallucination Detector**: Enhanced fact-checking, cross-validation, and confidence thresholds

We will also conduct comprehensive experiments comparing our approach against baseline RAG systems and evaluate improvements in accuracy, temporal understanding, and hallucination reduction.

## 5. Expected Contributions

Sir, our project will contribute:

1. **Novel Architecture**: First RAG system combining all four innovations (agentic, temporal, KG, hallucination grounding) for financial analysis
2. **Practical Application**: Real-world system that can be used by financial analysts and researchers
3. **Comparative Analysis**: Detailed evaluation showing improvements over traditional RAG approaches
4. **Open Source**: Complete codebase with documentation for the research community

## 6. Evaluation Plan

We will evaluate our system on:

1. **Accuracy Metrics**: Exact match, F1 score, ROUGE, BERTScore
2. **Temporal Understanding**: Accuracy on time-based queries and trend analysis
3. **Hallucination Rate**: Percentage of factually incorrect statements
4. **Citation Quality**: Precision and recall of source attributions
5. **User Study**: Feedback from domain experts on response quality

We will compare against baseline systems including standard RAG, LangChain, and LlamaIndex implementations.

## 7. Related Work

Our work builds upon recent advances in:

- **RAG Systems**: Lewis et al. (2020) introduced retrieval-augmented generation, combining retrieval with generation for improved factuality
- **Temporal Reasoning**: Temporal knowledge graphs and time-aware NLP models for understanding temporal relationships
- **Knowledge Graphs**: Entity linking and graph neural networks for multi-hop reasoning
- **Hallucination Detection**: Recent work on factuality verification and attribution in LLMs
- **Agentic AI**: ReAct, AutoGPT, and other autonomous agent frameworks
- **Financial NLP**: Domain-specific models and datasets for financial document understanding

We will provide detailed citations to 8-10 papers in our final submission.

## 8. Timeline

- **April 30 (Task-1)**: Submit proposal + 50% working code ✓
- **May 1-5**: Implement agentic module and temporal reasoning
- **May 6-8**: Add knowledge graph and hallucination grounding
- **May 9-10**: Complete experiments and write full paper
- **May 10 (Task-2)**: Submit complete project + research paper

## 9. Conclusion

Sir, we believe this project addresses real challenges in financial document analysis and pushes the boundaries of what RAG systems can achieve. By making the system agentic, temporally aware, graph-enabled, and hallucination-resistant, we are creating something that is not just academically interesting but practically valuable. We are excited to complete this work and demonstrate how these innovations improve accuracy, intelligence, and trustworthiness in AI-powered financial analysis.

We look forward to your feedback and guidance as we move forward with this project.

---

**References** (to be expanded in final paper):

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.
2. Gao, L., et al. (2023). "Retrieval-Augmented Generation for Large Language Models: A Survey." arXiv.
3. Shuster, K., et al. (2021). "Retrieval Augmentation Reduces Hallucination in Conversation." EMNLP.
4. Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR.
5. Ji, Z., et al. (2023). "Survey of Hallucination in Natural Language Generation." ACM Computing Surveys.
6. Trivedi, H., et al. (2023). "Interleaving Retrieval with Chain-of-Thought Reasoning." ACL.
7. Yasunaga, M., et al. (2022). "Deep Bidirectional Language-Knowledge Graph Pretraining." NeurIPS.
8. Jiang, Z., et al. (2023). "Active Retrieval Augmented Generation." EMNLP.

---

**Contact Information:**  
Adan Malik - [email]  
Shayan Ahmed - [email]  
Bilal Raza - [email]
