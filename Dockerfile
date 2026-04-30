# ============================================
# DOCKERFILE FOR FINANCIAL RAG SYSTEM
# ============================================
# Production-ready Dockerfile for Multi-Document Financial Analysis System
# Features: Document processing, Multiple LLMs, Streaming, Analytics, PDF Export
# ============================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install PyTorch CPU version first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install all Python packages
RUN pip install --no-cache-dir \
    streamlit==1.28.0 \
    groq==0.13.0 \
    openai>=1.0.0 \
    anthropic>=0.7.0 \
    google-generativeai>=0.3.0 \
    cohere>=4.0.0 \
    sentence-transformers==2.2.2 \
    transformers==4.30.2 \
    huggingface-hub==0.16.4 \
    tokenizers==0.13.3 \
    faiss-cpu==1.7.4 \
    PyMuPDF>=1.23.0 \
    pdfplumber>=0.10.0 \
    pandas==2.0.0 \
    pytesseract>=0.3.10 \
    Pillow>=9.0.0 \
    open-clip-torch>=2.20.0 \
    plotly==5.17.0 \
    python-dotenv==1.0.0 \
    tqdm>=4.66.0 \
    numpy>=1.24.0 \
    chromadb>=0.4.0 \
    fpdf2>=2.7.0 \
    rouge-score \
    bert-score \
    nltk>=3.8.0

# Copy application code
COPY src/ ./src/
COPY chunk_metadata/ ./chunk_metadata/
COPY prompts/ ./prompts/
COPY config.yaml ./config.yaml

# Create necessary directories
RUN mkdir -p data/temp data/raw

# Skip pre-downloading models during build - they will download on first use
# This avoids timeout issues during Docker build
RUN echo "Models will be downloaded on first application start"

# Environment variables
ENV PYTHONPATH=/app:/app/src \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_OFFLINE=0 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PYTHONUNBUFFERED=1

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit application
CMD ["streamlit", "run", "src/streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
