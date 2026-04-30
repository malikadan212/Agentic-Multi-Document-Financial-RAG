# Agentic Multi-Document Financial RAG System

**With Temporal Reasoning, Knowledge Graph Integration, and Hallucination Grounding**

A state-of-the-art Retrieval-Augmented Generation (RAG) system that goes beyond traditional approaches by incorporating agentic behavior, temporal reasoning, knowledge graph integration, and hallucination grounding mechanisms for accurate financial document analysis.

---

## 🎯 Research Innovation

This system represents a significant advancement in RAG technology by addressing critical challenges in financial document analysis:

### Core Innovations

1. **Agentic Architecture**
   - Autonomous query decomposition and planning
   - Multi-step reasoning with self-reflection
   - Dynamic retrieval strategy selection
   - Adaptive context management

2. **Temporal Reasoning**
   - Time-aware document analysis
   - Historical trend detection and comparison
   - Temporal relationship extraction
   - Date-sensitive query handling

3. **Knowledge Graph Integration**
   - Entity relationship mapping across documents
   - Cross-document knowledge synthesis
   - Graph-based reasoning paths
   - Semantic relationship discovery

4. **Hallucination Grounding**
   - Source verification mechanisms
   - Confidence-based response filtering
   - Citation validation and tracking
   - Fact-checking against retrieved context

---

## System Overview

This system implements a complete RAG pipeline for analyzing financial documents such as quarterly reports, balance sheets, and financial statements. Users can upload documents, which are processed into semantically meaningful chunks, embedded into a vector space, and stored for efficient retrieval. When a query is submitted, the system retrieves the most relevant document chunks and uses a large language model to generate accurate, citation-backed responses.

### Core Capabilities

- **Document Ingestion**: Processes PDF documents (with OCR fallback for scanned files), Excel spreadsheets, and CSV files
- **Semantic Search**: Converts text into dense vector embeddings for similarity-based retrieval
- **Multi-Provider LLM Support**: Integrates with Groq (free tier), OpenAI, Anthropic, Google Gemini, and Cohere
- **Automatic Citation Generation**: Extracts and validates source citations in the format `[Source: DocumentName, Page X]`
- **Multimodal Analysis**: Supports image understanding through CLIP embeddings and vision-capable language models
- **Real-Time Streaming**: Displays responses token-by-token as they are generated
- **Comprehensive Evaluation**: Includes metrics for retrieval quality, generation accuracy, and citation correctness

---

## Architecture

### Data Flow

```
                                    User Query
                                         |
                                         v
+------------------+    +-----------------+    +------------------+
|    Documents     |    | Embedding Model |    |   Vector Store   |
| (PDF/Excel/CSV)  |--->| (MiniLM/MPNet/  |--->| (FAISS/ChromaDB) |
+------------------+    |  DistilBERT/    |    +------------------+
         |              |  RoBERTa)       |             |
         v              +-----------------+             |
+------------------+                                    |
| DocumentPipeline |                                    |
| - Text Extraction|                                    v
| - OCR Processing |                          +-----------------+
| - Table Parsing  |                          | HybridRetriever |
| - Chunking       |                          | - Query Encoding|
+------------------+                          | - Top-K Search  |
                                              +-----------------+
                                                       |
                                                       v
                                              +-----------------+
                                              |  RAGGenerator   |
                                              | - Context Build |
                                              | - LLM Call      |
                                              | - Citation      |
                                              |   Extraction    |
                                              +-----------------+
                                                       |
                                                       v
                                              Generated Response
                                              with Citations
```

### Component Overview

| Component | Description |
|-----------|-------------|
| Document Processing | Extracts text, tables, and images from input documents with OCR fallback |
| Embedding Model | Converts text chunks into dense vector representations |
| Vector Store | Stores and indexes embeddings for efficient similarity search |
| Retriever | Finds the most relevant document chunks for a given query |
| Generator | Constructs prompts and calls LLMs to produce cited answers |
| Evaluator | Measures system performance across retrieval, generation, and citation metrics |

---

## Features

### Document Processing

- **PDF Processing**: Uses PyMuPDF (fitz) for fast text extraction and pdfplumber for table extraction
- **OCR Fallback**: Automatically applies Tesseract OCR when extracted text is insufficient (fewer than 20 characters per page)
- **Table Extraction**: Converts tables to structured text format preserving row and column relationships
- **Image Extraction**: Extracts embedded images from PDFs for multimodal analysis (minimum size threshold: 100x100 pixels)
- **Excel and CSV Support**: Processes all sheets in Excel workbooks with automatic data type detection
- **Semantic Chunking**: Splits documents at sentence boundaries with configurable chunk size (default: 512 tokens) and overlap (default: 50 tokens)

### Embedding Models

The system supports four embedding models from the Sentence Transformers library:

| Model ID | Model Name | Dimension | Characteristics |
|----------|------------|-----------|-----------------|
| `minilm` | all-MiniLM-L6-v2 | 384 | Fastest inference, suitable for real-time applications |
| `mpnet` | all-mpnet-base-v2 | 768 | Balanced performance and quality |
| `distilbert` | msmarco-distilbert-base-v4 | 768 | Optimized for semantic search tasks |
| `roberta` | all-roberta-large-v1 | 1024 | Highest quality embeddings, slower inference |

### Vector Stores

- **FAISS (Facebook AI Similarity Search)**
  - Default vector store for fast in-memory similarity search
  - Supports both flat index (exact search) and IVF index (approximate search for datasets exceeding 10,000 documents)
  - Persistence through index serialization to disk
  
- **ChromaDB**
  - Alternative vector store with built-in persistence
  - Supports metadata filtering during retrieval
  - Automatic deduplication based on chunk IDs

### LLM Providers

| Provider | Models | API Key Variable | Cost |
|----------|--------|------------------|------|
| Groq | llama-3.1-8b-instant, llama-3.3-70b-versatile, mixtral-8x7b-32768 | `GROQ_API_KEY` | Free tier available |
| Groq Vision | meta-llama/llama-4-scout-17b-16e-instruct, meta-llama/llama-4-maverick-17b-128e-instruct | `GROQ_API_KEY` | Free tier available |
| OpenAI | gpt-3.5-turbo, gpt-4, gpt-4-turbo | `OPENAI_API_KEY` | Paid |
| Anthropic | claude-3-sonnet-20240229, claude-3-opus-20240229 | `ANTHROPIC_API_KEY` | Paid |
| Google | gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash | `GOOGLE_API_KEY` | Free tier available |
| Cohere | command | `COHERE_API_KEY` | Free tier available |

### Multimodal Capabilities

- **CLIP Embeddings**: Uses OpenCLIP to generate joint text-image embeddings for cross-modal retrieval
- **Supported CLIP Models**: ViT-B-32, ViT-B-16, ViT-L-14
- **Vision LLM Integration**: Groq Vision models can analyze images alongside text for document understanding
- **Text-to-Image Search**: Find relevant images based on text descriptions
- **Image-to-Image Search**: Find similar images in the document collection

### User Interface Features

- **Streaming Responses**: Real-time display of LLM output as tokens are generated (supported by Groq provider)
- **Confidence Scoring**: Visual indicator (High/Medium/Low) calculated from retrieval scores, source count, and citation density
- **Follow-up Questions**: LLM-generated suggested queries based on the current Q&A context
- **Semantic Highlighting**: Automatically highlights phrases in source documents that match the generated answer
- **PDF Export**: One-click download of Q&A sessions as formatted PDF reports with citations and metrics
- **Analytics Dashboard**: Displays query response times, source usage frequency, and session statistics
- **Summary Cards**: Shows document count, chunk count, queries processed, and average response time

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Tesseract OCR (required for scanned PDF support)

### Installing Tesseract OCR

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/anasbinrashid/Multi-Document-Financial-Analysis-System.git
cd Multi-Document-Financial-Analysis-System
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (required for evaluation metrics):
```bash
python -c "import nltk; nltk.download('punkt')"
```

5. Create environment file:
```bash
cp .env.example .env
# Edit .env with your API keys
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required (at least one LLM provider)
GROQ_API_KEY=your_groq_api_key_here

# Optional (for additional providers)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
COHERE_API_KEY=your_cohere_api_key_here

# Default configuration
DEFAULT_LLM_PROVIDER=groq
DEFAULT_EMBEDDING_MODEL=minilm
```

### Configuration File (config.yaml)

```yaml
document_processing:
  chunk_size: 512          # Maximum tokens per chunk
  chunk_overlap: 50        # Overlapping tokens between chunks
  supported_formats:
    - pdf
    - xlsx
    - csv
  enable_ocr: true         # Enable OCR for scanned documents
  extract_images: true     # Extract images for multimodal processing
  min_image_size: 100      # Minimum image dimension in pixels

retrieval:
  embedding_model: "all-MiniLM-L6-v2"
  vector_store: "faiss"    # Options: faiss, chroma
  top_k: 5                 # Number of chunks to retrieve
  similarity_threshold: 0.7
  use_ivf: false           # Enable IVF index for large datasets
  n_clusters: 100          # Number of clusters for IVF index

generation:
  llm_provider: "groq"
  model: "llama-3.3-70b-versatile"
  temperature: 0.1
  max_tokens: 1000
  top_p: 0.95
  enable_streaming: true

features:
  follow_up_questions: true
  confidence_scoring: true
  pdf_export: true
  semantic_highlighting: true
  analytics_dashboard: true
  summary_cards: true

evaluation:
  metrics:
    - recall@5
    - precision@5
    - mrr
    - exact_match
    - f1_score
    - rouge_l
    - bertscore
    - bleu
    - citation_precision
    - citation_recall
```

---

## Usage

### Starting the Application

**Local Development:**
```bash
streamlit run src/streamlit_app/app.py
```

The application will be available at `http://localhost:8501`.

**Production (with Gunicorn):**
```bash
gunicorn -w 4 -b 0.0.0.0:8000 src.streamlit_app.app:main
```

### Application Workflow

1. **Select Data Source**: Choose between uploading new documents or using pre-loaded indexed data
2. **Configure Settings**: Select embedding model, vector store, LLM provider, and generation parameters in the sidebar
3. **Process Documents**: Upload PDF, Excel, or CSV files and click "Process Documents" to extract, chunk, and index content
4. **Query Documents**: Enter natural language questions in the query interface
5. **Review Results**: Examine the generated answer, confidence score, citations, and source snippets
6. **Export**: Download the Q&A session as a PDF report

### Using Pre-loaded Data

The system supports loading pre-indexed document data from the `chunk_metadata/` directory:

- `chunk_metadata.json`: Contains chunk text, metadata, and vector IDs
- `rag_index.faiss`: Pre-computed FAISS index with embeddings

Select "Use Pre-loaded Data" in the sidebar to skip document processing and load existing indices.

### Programmatic Usage

```python
from src.document_processing.processor import DocumentPipeline
from src.retrieval.retriever import HybridRetriever
from src.generation.generator import RAGGenerator, GenerationConfig

# Process documents
pipeline = DocumentPipeline()
chunks = pipeline.process_directory("data/documents/")

# Index documents
retriever = HybridRetriever(
    embedding_model='minilm',
    vector_store_type='faiss',
    top_k=5
)
retriever.index_documents(chunks)

# Generate response
config = GenerationConfig(temperature=0.1, max_tokens=1000)
generator = RAGGenerator(provider='groq', model_name='llama-3.3-70b-versatile', config=config)

query = "What was the total revenue for Q3?"
results = retriever.retrieve(query)
response = generator.generate_with_citations(query, results)

print(f"Answer: {response.answer}")
print(f"Citations: {response.citations}")
```

---

## API Reference

### DocumentPipeline

```python
class DocumentPipeline:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50)
    def process_directory(self, directory: str) -> List[DocumentChunk]
    def process_directory_multimodal(self, directory: str, extract_images: bool = True) -> Tuple[List[DocumentChunk], List[ImageChunk]]
```

### HybridRetriever

```python
class HybridRetriever:
    def __init__(self, embedding_model: str = 'minilm', vector_store_type: str = 'faiss', top_k: int = 5)
    def index_documents(self, chunks: List[DocumentChunk]) -> None
    def retrieve(self, query: str, top_k: Optional[int] = None, metadata_filter: Optional[Dict] = None) -> List[RetrievalResult]
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

### RAGGenerator

```python
class RAGGenerator:
    def __init__(self, provider: str = 'groq', model_name: Optional[str] = None, config: Optional[GenerationConfig] = None)
    def generate_with_citations(self, query: str, retrieved_chunks: List[RetrievalResult], images: Optional[List] = None) -> GeneratedResponse
    def generate_stream(self, query: str, retrieved_chunks: List[RetrievalResult]) -> Generator[str, None, None]
```

### GeneratedResponse

```python
@dataclass
class GeneratedResponse:
    answer: str                    # Generated text response
    citations: List[Dict]          # Extracted citations with doc_name, page, text
    model: str                     # Model used for generation
    provider: str                  # LLM provider name
    tokens_used: int               # Total tokens consumed
    total_cost: float              # Cost in USD (0 for free providers)
    response_time: float           # Generation time in seconds
```

---

## Project Structure

```
Multi-Document-Financial-Analysis-System/
|
+-- src/
|   +-- document_processing/
|   |   +-- __init__.py
|   |   +-- processor.py              # PDFProcessor, ExcelProcessor, DocumentChunker, DocumentPipeline
|   |
|   +-- retrieval/
|   |   +-- __init__.py
|   |   +-- retriever.py              # EmbeddingModel, FAISSVectorStore, ChromaVectorStore, HybridRetriever
|   |   +-- multimodal_retriever.py   # CLIPEmbedding, MultimodalRetriever
|   |   +-- preloaded_retriever.py    # PreloadedRetriever for pre-indexed data
|   |
|   +-- generation/
|   |   +-- __init__.py
|   |   +-- generator.py              # BaseLLM, OpenAILLM, AnthropicLLM, GoogleLLM, CohereLLM, GroqLLM, GroqVisionLLM, RAGGenerator
|   |   +-- simple_generator.py       # Simplified Groq-only generator
|   |
|   +-- evaluation/
|   |   +-- __init__.py
|   |   +-- evaluator.py              # RetrievalEvaluator, GenerationEvaluator, CitationEvaluator, ComprehensiveEvaluator
|   |
|   +-- streamlit_app/
|   |   +-- __init__.py
|   |   +-- app.py                    # Main Streamlit web application
|   |
|   +-- utils/
|       +-- __init__.py
|       +-- chunk_loader.py           # ChunkMetadataLoader for pre-indexed data
|       +-- pdf_exporter.py           # RAGReportExporter for PDF generation
|
+-- data/
|   +-- raw/                          # Original uploaded documents
|   +-- processed/                    # Processed document outputs
|   +-- temp/                         # Temporary file storage
|
+-- chunk_metadata/
|   +-- chunk_metadata.json           # Pre-indexed chunk data
|   +-- rag_index.faiss               # Pre-computed FAISS index
|
+-- prompts/
|   +-- system_prompts.md             # Core system prompt templates
|   +-- rag_prompt_templates.md       # RAG-specific prompt formats
|   +-- citation_prompts.md           # Citation generation prompts
|   +-- evaluation_prompts.md         # Evaluation prompt templates
|   +-- development_prompts.txt       # Development and testing prompts
|
+-- config.yaml                       # Application configuration
+-- requirements.txt                  # Python dependencies
+-- Dockerfile                        # Container definition
+-- docker-compose.yml                # Docker orchestration
+-- .env                              # Environment variables (not tracked in git)
+-- .env.example                      # Example environment file
```

---

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.0 | Web application framework |
| groq | latest | Groq API client |
| openai | 1.3.0 | OpenAI API client |
| anthropic | 0.7.0 | Anthropic API client |
| google-generativeai | 0.3.0 | Google Gemini API client |
| cohere | 4.37 | Cohere API client |
| sentence-transformers | 2.2.2 | Embedding model library |
| transformers | 4.30.2 | Hugging Face transformers |
| faiss-cpu | 1.7.4 | Vector similarity search |
| chromadb | latest | Alternative vector store |

### Document Processing

| Package | Version | Purpose |
|---------|---------|---------|
| PyMuPDF | 1.23.0 | PDF text and image extraction |
| pdfplumber | 0.10.0 | PDF table extraction |
| pandas | 2.0.0 | Excel and CSV processing |
| openpyxl | latest | Excel file support |
| pytesseract | 0.3.10 | OCR for scanned documents |
| Pillow | 10.0.0 | Image processing |

### Multimodal

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.0.0 | Deep learning framework |
| open-clip-torch | latest | CLIP model implementation |

### Evaluation

| Package | Version | Purpose |
|---------|---------|---------|
| rouge-score | latest | ROUGE metric calculation |
| bert-score | latest | BERTScore metric |
| nltk | 3.8.0 | Text tokenization for BLEU |

### Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| fpdf2 | latest | PDF report generation |
| plotly | 5.17.0 | Interactive charts |
| python-dotenv | latest | Environment variable loading |
| PyYAML | latest | Configuration file parsing |

---

## Docker Deployment

### Building the Image

```bash
docker build -t financial-rag-system .
```

### Running with Docker Compose

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### Docker Compose Configuration

The `docker-compose.yml` file configures:

- **Port Mapping**: Maps container port 8501 to host port 8501
- **Environment Variables**: Injects API keys from `.env` file
- **Volume Mounts**: 
  - `./data:/app/data` for document persistence
  - `./chunk_metadata:/app/chunk_metadata` for index persistence
  - `huggingface_cache:/root/.cache/huggingface` for model caching
- **Resource Limits**: 4GB memory limit recommended
- **Health Check**: HTTP check on `/healthz` endpoint every 30 seconds

### Dockerfile Details

The Dockerfile:

1. Uses Python 3.11-slim as base image
2. Installs system dependencies including Tesseract OCR and required libraries
3. Installs PyTorch CPU version to reduce image size
4. Installs all Python dependencies from requirements.txt
5. Copies application source code
6. Exposes port 8501 for Streamlit
7. Sets the entrypoint to run the Streamlit application

---

## Evaluation Framework

### Retrieval Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| Recall@K | Proportion of relevant documents retrieved in top-K results | 0.0 - 1.0 |
| Precision@K | Proportion of retrieved documents that are relevant | 0.0 - 1.0 |
| MRR (Mean Reciprocal Rank) | Average reciprocal rank of first relevant result | 0.0 - 1.0 |

### Generation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| Exact Match | Binary match between generated and reference answer | 0 or 1 |
| F1 Score | Token-level overlap between generated and reference | 0.0 - 1.0 |
| ROUGE-1 | Unigram overlap with reference | 0.0 - 1.0 |
| ROUGE-2 | Bigram overlap with reference | 0.0 - 1.0 |
| ROUGE-L | Longest common subsequence overlap | 0.0 - 1.0 |
| BERTScore | Semantic similarity using BERT embeddings | 0.0 - 1.0 |
| BLEU | N-gram precision with brevity penalty | 0.0 - 1.0 |

### Citation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| Citation Precision | Proportion of generated citations that are correct | 0.0 - 1.0 |
| Citation Recall | Proportion of required citations that were generated | 0.0 - 1.0 |
| False Citation Rate | Proportion of citations that reference non-existent sources | 0.0 - 1.0 |

### Running Evaluations

```python
from src.evaluation.evaluator import ComprehensiveEvaluator, TestCase

# Define test cases
test_cases = [
    TestCase(
        query="What was Q3 revenue?",
        ground_truth_answer="Q3 revenue was $81.8 billion",
        ground_truth_sources=["Apple_10Q.pdf"],
        category="financial_metrics",
        difficulty="easy"
    )
]

# Run evaluation
evaluator = ComprehensiveEvaluator(retriever, generator)
results = evaluator.evaluate(test_cases)

# Generate report
evaluator.generate_report(results, output_path="evaluation_report.json")
```

---

## Troubleshooting

### Common Issues

**OCR not working:**
- Ensure Tesseract is installed and in system PATH
- On Windows, add Tesseract installation directory to PATH environment variable

**FAISS index loading fails:**
- Verify that `chunk_metadata/rag_index.faiss` exists and is not corrupted
- Ensure FAISS version matches the version used to create the index

**LLM API errors:**
- Verify API keys are correctly set in `.env` file
- Check API rate limits and quota
- Ensure network connectivity to API endpoints

**Memory issues with large documents:**
- Reduce chunk size in configuration
- Use IVF index for FAISS with large document collections
- Process documents in batches

**Embedding model download fails:**
- Ensure internet connectivity for first-time model download
- Set `HF_HOME` environment variable to specify cache directory
- Pre-download models using `sentence_transformers` CLI

---

## Contributing

Contributions are welcome. Please ensure that:

1. Code follows existing style conventions
2. New features include appropriate tests
3. Documentation is updated for any API changes
4. All tests pass before submitting pull requests

---

## 🎓 Academic Project

**Developed by:** Adan Malik, Shayan Ahmed, Bilal Raza  
**Course:** Generative AI  
**Instructor:** Sir Akhtar Jamil  
**Institution:** FAST-NUCES

This project represents advanced research in Retrieval-Augmented Generation systems with a focus on agentic behavior, temporal reasoning, knowledge graph integration, and hallucination grounding for financial document analysis.

---

## 📄 License

This project is developed for academic purposes as part of the Generative AI course at FAST-NUCES.
