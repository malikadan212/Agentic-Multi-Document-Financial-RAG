# src/streamlit_app/app.py
"""
Streamlit Web Application for Financial RAG System
Clean, professional UI with document upload, query interface, and result visualization
Enhanced with: Follow-up Questions, Confidence Score, PDF Export, Semantic Highlighting,
Analytics Dashboard, Streaming Responses, and Summary Cards
"""

# Import necessary libraries
import streamlit as st  # For building web interface
import sys  # For system path manipulation
from pathlib import Path  # For file path handling
import time  # For timing operations
import plotly.graph_objects as go  # For creating charts
import pandas as pd  # For data manipulation and display
import re  # For regex operations
from datetime import datetime  # For timestamps
from collections import Counter  # For analytics

# Add src to path (go up one level from streamlit_app to src)
sys.path.insert(0, str(Path(__file__).parent.parent))  # Enable importing modules from parent directory

# Import custom modules from the project
from document_processing.processor import DocumentPipeline  # For processing uploaded documents
from retrieval.retriever import HybridRetriever, RetrievalResult  # For retrieving relevant document chunks
from retrieval.preloaded_retriever import PreloadedRetriever  # For using pre-loaded data
from generation.generator import RAGGenerator, GenerationConfig  # For generating answers with LLM
import logging  # For logging functionality

# Try to import multimodal components (optional, requires extra dependencies)
try:
    from retrieval.multimodal_retriever import MultimodalRetriever, CLIPEmbedding
    MULTIMODAL_AVAILABLE = True
except Exception:  # Catch all errors (ImportError, NameError if torch missing, etc.)
    MULTIMODAL_AVAILABLE = False

# Try to import PDF exporter
try:
    from utils.pdf_exporter import RAGReportExporter
    PDF_EXPORT_AVAILABLE = True
except ImportError:
    PDF_EXPORT_AVAILABLE = False

# Try to import agentic module
try:
    from agentic import AgentPlanner, AgentExecutor
    AGENTIC_AVAILABLE = True
except ImportError:
    AGENTIC_AVAILABLE = False
    logger.warning("Agentic module not available")

# Try to import KG module — soft dependency on networkx/spacy. UI silently falls
# back to vector-only retrieval if either the module or the persisted graph is
# missing.
try:
    from kg import (
        FinancialEntityExtractor,
        FinancialKnowledgeGraph,
        KGAwareRetriever,
    )
    KG_MODULE_AVAILABLE = True
except ImportError as exc:  # noqa: BLE001
    KG_MODULE_AVAILABLE = False
    logger.warning(f"KG module not available: {exc}")

# Set up logging configuration
logging.basicConfig(level=logging.INFO)  # Set default logging level to INFO
logger = logging.getLogger(__name__)  # Create logger instance for this module


# Configure Streamlit page settings
st.set_page_config(
    page_title="Agentic Financial RAG | Temporal Reasoning & Knowledge Graphs",  # Browser tab title
    page_icon="🧠",  # Favicon
    layout="wide",  # Use full-width layout
    initial_sidebar_state="expanded"  # Sidebar starts expanded
)

# Custom CSS — Slate + Cyan modern theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-0: #07090f;
        --bg-1: #0b1220;
        --bg-2: #0f172a;
        --surface-1: rgba(30, 41, 59, 0.55);
        --surface-2: rgba(15, 23, 42, 0.75);
        --surface-3: rgba(51, 65, 85, 0.35);
        --border-subtle: rgba(148, 163, 184, 0.12);
        --border-strong: rgba(34, 211, 238, 0.35);
        --accent: #22d3ee;
        --accent-2: #06b6d4;
        --accent-3: #0ea5e9;
        --accent-glow: rgba(34, 211, 238, 0.25);
        --text-1: #f1f5f9;
        --text-2: #cbd5e1;
        --text-3: #94a3b8;
        --text-muted: #64748b;
        --success: #10b981;
        --warn: #f59e0b;
        --danger: #ef4444;
    }

    * { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    code, pre, kbd { font-family: 'JetBrains Mono', monospace !important; }

    /* App background — layered radial glows over slate */
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(34, 211, 238, 0.10), transparent 60%),
            radial-gradient(900px 500px at 90% 0%, rgba(14, 165, 233, 0.08), transparent 55%),
            radial-gradient(700px 400px at 50% 100%, rgba(6, 182, 212, 0.06), transparent 60%),
            linear-gradient(180deg, #07090f 0%, #0b1220 50%, #0f172a 100%);
        color: var(--text-1);
    }

    .block-container { padding-top: 2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11, 18, 32, 0.9) 0%, rgba(7, 9, 15, 0.95) 100%);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-right: 1px solid var(--border-subtle);
        box-shadow: inset -1px 0 0 rgba(34, 211, 238, 0.08);
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p { color: var(--text-2); }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: var(--text-1); letter-spacing: -0.01em; }

    /* Hero typography */
    .main-header {
        font-size: 3.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e0f2fe 0%, #22d3ee 45%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 0.5rem 0 0.25rem;
        letter-spacing: -0.035em;
        line-height: 1.05;
    }
    .sub-header {
        font-size: 1.05rem;
        color: var(--text-3);
        text-align: center;
        margin-bottom: 1.75rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    .brand-tagline {
        font-size: 0.78rem;
        color: var(--accent);
        text-align: center;
        margin-bottom: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.32em;
    }

    h1, h2, h3, h4 { color: var(--text-1); letter-spacing: -0.015em; }
    p, li, span, label, div { color: var(--text-2); }

    /* Answer box — glass + cyan rail */
    .answer-box {
        position: relative;
        background: var(--surface-2);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        color: var(--text-1);
        padding: 1.75rem 2rem;
        border-radius: 16px;
        border: 1px solid var(--border-subtle);
        margin: 1.5rem 0;
        font-size: 1.02rem;
        line-height: 1.75;
        box-shadow:
            0 1px 0 rgba(255, 255, 255, 0.04) inset,
            0 20px 40px -20px rgba(0, 0, 0, 0.6),
            0 0 0 1px rgba(34, 211, 238, 0.08);
    }
    .answer-box::before {
        content: "";
        position: absolute; left: 0; top: 12px; bottom: 12px;
        width: 3px; border-radius: 3px;
        background: linear-gradient(180deg, #22d3ee, #0ea5e9);
        box-shadow: 0 0 16px var(--accent-glow);
    }

    /* Source cards */
    .source-card {
        background: var(--surface-1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: var(--text-2);
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        margin: 0.85rem 0;
        border: 1px solid var(--border-subtle);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .source-card::before {
        content: "";
        position: absolute; left: 0; top: 0; bottom: 0;
        width: 2px;
        background: linear-gradient(180deg, var(--accent), transparent);
        opacity: 0.7;
    }
    .source-card:hover {
        transform: translateY(-2px);
        border-color: var(--border-strong);
        box-shadow: 0 12px 32px -16px rgba(34, 211, 238, 0.35);
    }

    /* Semantic highlight */
    .highlight {
        background: linear-gradient(180deg, transparent 55%, rgba(34, 211, 238, 0.35) 55%);
        padding: 0 3px;
        border-radius: 2px;
        color: #e0f7fa;
        font-weight: 600;
    }

    /* Confidence pills */
    .confidence-meter {
        padding: 0.55rem 1.1rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        margin: 0.5rem 0;
        font-size: 0.82rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border: 1px solid transparent;
    }
    .confidence-high {
        background: rgba(16, 185, 129, 0.12);
        color: #6ee7b7;
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 24px rgba(16, 185, 129, 0.15);
    }
    .confidence-medium {
        background: rgba(245, 158, 11, 0.12);
        color: #fcd34d;
        border-color: rgba(245, 158, 11, 0.4);
        box-shadow: 0 0 24px rgba(245, 158, 11, 0.12);
    }
    .confidence-low {
        background: rgba(148, 163, 184, 0.10);
        color: #cbd5e1;
        border-color: rgba(148, 163, 184, 0.3);
    }

    /* Follow-up chips (used for visual reference; stButton handles real buttons) */
    .followup-btn {
        background: var(--surface-1);
        border: 1px solid var(--border-strong);
        color: var(--accent);
        padding: 0.55rem 1.1rem;
        border-radius: 999px;
        margin: 0.4rem;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.25s ease;
    }
    .followup-btn:hover {
        background: rgba(34, 211, 238, 0.12);
        color: #ecfeff;
        transform: translateY(-1px);
        box-shadow: 0 8px 24px -10px var(--accent-glow);
    }

    /* Summary cards — modern glass */
    .summary-card {
        background: var(--surface-2);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--border-subtle);
        color: var(--text-1);
        padding: 1.4rem 1.25rem;
        border-radius: 16px;
        text-align: left;
        margin: 0.4rem 0;
        position: relative;
        overflow: hidden;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .summary-card::after {
        content: "";
        position: absolute; inset: 0;
        background: radial-gradient(400px 120px at 0% 0%, rgba(34, 211, 238, 0.10), transparent 60%);
        pointer-events: none;
    }
    .summary-card:hover {
        transform: translateY(-3px);
        border-color: var(--border-strong);
        box-shadow: 0 18px 40px -20px rgba(34, 211, 238, 0.4);
    }
    .summary-card h3 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text-1);
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #f1f5f9, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .summary-card p {
        margin: 0.4rem 0 0 0;
        font-size: 0.72rem;
        color: var(--text-3);
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-weight: 600;
    }

    /* Buttons — primary cyan */
    .stButton>button, .stDownloadButton>button {
        width: 100%;
        background: linear-gradient(135deg, #06b6d4 0%, #0ea5e9 100%);
        color: #042f2e;
        border: 1px solid rgba(34, 211, 238, 0.5);
        padding: 0.65rem 1.25rem;
        font-weight: 600;
        font-size: 0.9rem;
        border-radius: 10px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
        letter-spacing: 0.02em;
        text-transform: none;
        box-shadow: 0 4px 14px -4px var(--accent-glow);
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
        box-shadow: 0 10px 28px -8px var(--accent-glow);
    }
    .stButton>button:active { transform: translateY(0); }

    /* Secondary button variant when inside a form/help context — keep it readable */
    .stButton>button:focus { outline: none; box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.35); }

    /* Inputs */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stNumberInput>div>div>input {
        background-color: rgba(15, 23, 42, 0.7) !important;
        color: var(--text-1) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stNumberInput>div>div>input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.18) !important;
    }

    /* Selectbox / multiselect */
    .stSelectbox>div>div, .stMultiSelect>div>div {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        color: var(--text-1) !important;
    }

    /* Radio + checkbox labels */
    .stRadio label, .stCheckbox label { color: var(--text-2) !important; }

    /* Slider */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--accent) !important;
        border: 2px solid #ecfeff !important;
        box-shadow: 0 0 12px var(--accent-glow);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    [data-testid="stMetricLabel"] { color: var(--text-3) !important; }

    /* Expander */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        background: var(--surface-1) !important;
        color: var(--text-1) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        font-weight: 500;
    }
    [data-testid="stExpander"] summary:hover { border-color: var(--border-strong) !important; }
    [data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: var(--surface-3);
        padding: 0.35rem;
        border-radius: 12px;
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-3);
        border-radius: 8px;
        padding: 0.55rem 1rem;
        font-weight: 500;
        transition: color 0.2s ease, background 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--text-1); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.18), rgba(14, 165, 233, 0.12)) !important;
        color: var(--text-1) !important;
        box-shadow: 0 0 0 1px var(--border-strong);
    }

    /* Chat / text bubbles produced by st.info, st.success, st.warning */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid var(--border-subtle) !important;
        backdrop-filter: blur(8px);
    }

    /* Dataframe */
    .stDataFrame, [data-testid="stTable"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-subtle);
    }

    /* File uploader */
    [data-testid="stFileUploader"] section {
        background: var(--surface-1) !important;
        border: 1.5px dashed rgba(34, 211, 238, 0.35) !important;
        border-radius: 14px !important;
        color: var(--text-2) !important;
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--accent) !important;
        background: rgba(34, 211, 238, 0.05) !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-2), var(--accent)) !important;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-strong), transparent);
        opacity: 0.7;
        margin: 1.5rem 0;
    }

    /* Hide Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg-0); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #0ea5e9, #06b6d4);
        border-radius: 6px;
        border: 2px solid var(--bg-0);
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

    /* Subtle entrance fade */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .answer-box, .source-card, .summary-card {
        animation: fadeUp 0.45s ease both;
    }
</style>
""", unsafe_allow_html=True)


# === HELPER FUNCTIONS FOR NEW FEATURES ===

def calculate_confidence_score(retrieved_results, answer, citations, kg_coverage=None):
    """
    Calculate a confidence score (0-100) based on:
    - Average retrieval score        (0-40 points)
    - Number of sources used         (0-30 points)
    - Citation density               (0-30 points)
    - KG entity coverage (optional)  (0-15 points, additive bonus, capped at 100)

    Args:
        retrieved_results: list of RetrievalResult-like objects with `.score`.
        answer: generated answer text.
        citations: list of extracted citations.
        kg_coverage: float in [0, 1], the fraction of query entities that appeared
            in the retrieved chunks (computed by KGAwareRetriever). Pass None when
            KG retrieval is disabled — no bonus applied.

    Returns:
        (total_score: int, level: str)
    """
    if not retrieved_results:
        return 0, "Low"
    
    # Retrieval score component (0-40 points)
    avg_score = sum(r.score for r in retrieved_results) / len(retrieved_results)
    retrieval_points = min(40, avg_score * 50)  # Scale to 40 max
    
    # Sources component (0-30 points)
    num_sources = len(retrieved_results)
    source_points = min(30, num_sources * 6)  # 5 sources = 30 points
    
    # Citation density component (0-30 points)
    if answer:
        citation_count = len(citations) if citations else 0
        # More citations relative to answer length = higher confidence
        words_in_answer = len(answer.split())
        citation_density = citation_count / max(1, words_in_answer / 50)
        citation_points = min(30, citation_density * 15)
    else:
        citation_points = 0

    # KG bonus (0-15 points, additive on top of the 0-100 scale, capped at 100).
    # Only applied when KG retrieval is enabled AND the query had entities.
    kg_bonus_points = 0.0
    if kg_coverage is not None and kg_coverage > 0:
        kg_bonus_points = min(15.0, float(kg_coverage) * 15.0)

    total_score = int(min(100, retrieval_points + source_points + citation_points + kg_bonus_points))

    # Determine confidence level
    if total_score >= 70:
        level = "High"
    elif total_score >= 40:
        level = "Medium"
    else:
        level = "Low"
    
    return total_score, level


def highlight_matching_phrases(source_text, answer_text, min_phrase_length=4):
    """
    Find and highlight phrases from the source that appear in the answer.
    Returns HTML with highlighted matching phrases.
    """
    if not source_text or not answer_text:
        return source_text
    
    # Extract significant phrases from the answer (minimum 4 words)
    answer_words = answer_text.lower().split()
    highlighted_text = source_text
    
    # Find matching numeric values and short phrases
    # Look for numbers with context
    number_pattern = r'\$?[\d,]+\.?\d*\s*(?:billion|million|thousand|%|percent)?'
    answer_numbers = re.findall(number_pattern, answer_text.lower())
    
    for num in answer_numbers:
        if len(num.strip()) > 2:
            # Case-insensitive replacement with highlight
            pattern = re.escape(num.strip())
            highlighted_text = re.sub(
                f'({pattern})', 
                r'<mark class="highlight">\1</mark>', 
                highlighted_text, 
                flags=re.IGNORECASE
            )
    
    # Find matching multi-word phrases (3+ words)
    for i in range(len(answer_words) - 2):
        phrase = ' '.join(answer_words[i:i+3])
        if len(phrase) > 10:  # Only meaningful phrases
            pattern = re.escape(phrase)
            highlighted_text = re.sub(
                f'({pattern})', 
                r'<mark class="highlight">\1</mark>', 
                highlighted_text, 
                flags=re.IGNORECASE
            )
    
    return highlighted_text


def generate_follow_up_questions(query, answer, generator):
    """
    Generate 3-4 relevant follow-up questions based on the query and answer.
    Uses the LLM to suggest contextually relevant questions.
    """
    try:
        prompt = f"""Based on this Q&A about financial documents, suggest exactly 3 brief follow-up questions the user might want to ask next.

Original Question: {query}
Answer Summary: {answer[:500] if len(answer) > 500 else answer}

Requirements:
- Each question should be 10 words or less
- Questions should explore related topics
- Make them specific and actionable
- Format: Return ONLY the questions, one per line, numbered 1-3"""

        followup_response, _ = generator.llm.generate(prompt, system_prompt="You are a helpful assistant that generates concise follow-up questions.")
        
        # Parse the response to extract questions
        lines = followup_response.strip().split('\n')
        questions = []
        for line in lines:
            # Remove numbering and clean up
            cleaned = re.sub(r'^[\d\.\)\-\s]+', '', line).strip()
            if cleaned and len(cleaned) > 5 and '?' in cleaned or len(cleaned) > 10:
                if not cleaned.endswith('?'):
                    cleaned += '?'
                questions.append(cleaned)
        
        return questions[:3]  # Return max 3 questions
    except Exception as e:
        logger.warning(f"Could not generate follow-up questions: {e}")
        return []


def extract_summary_stats(retriever):
    """
    Extract summary statistics from the loaded data for summary cards.
    Works with both HybridRetriever and PreloadedRetriever.
    """
    try:
        total_chunks = 0
        embedding_dim = 'N/A'
        doc_names = set()
        chunks_list = []
        
        # Get chunks from different retriever types
        # HybridRetriever: chunks are in vector_store.chunks
        if hasattr(retriever, 'vector_store') and retriever.vector_store is not None:
            if hasattr(retriever.vector_store, 'chunks'):
                chunks_list = retriever.vector_store.chunks
                total_chunks = len(chunks_list)
            if hasattr(retriever, 'embedding_model'):
                embedding_dim = getattr(retriever.embedding_model, 'dimension', 'N/A')
        
        # PreloadedRetriever: chunks are in loader.chunks
        elif hasattr(retriever, 'loader'):
            if hasattr(retriever.loader, 'chunks'):
                chunks_list = retriever.loader.chunks
                total_chunks = len(chunks_list)
            if hasattr(retriever.loader, 'metadata'):
                meta = retriever.loader.metadata
                embedding_dim = meta.get('embedding_dimension', 'N/A')
        
        # Also try get_stats if available
        if hasattr(retriever, 'get_stats'):
            try:
                stats = retriever.get_stats()
                if total_chunks == 0:
                    total_chunks = stats.get('total_chunks', 0)
                if embedding_dim == 'N/A':
                    embedding_dim = stats.get('embedding_dimension', 'N/A')
            except:
                pass
        
        # Extract document names from ALL chunks (not just a sample) — chunks
        # belonging to a single document are usually stored contiguously, so
        # sampling biases the count to 1. This is just dict access, fast even
        # at 27k+ chunks.
        for chunk in chunks_list:
            if hasattr(chunk, 'metadata'):
                doc_names.add(chunk.metadata.get('doc_name', 'Unknown'))
            elif isinstance(chunk, dict):
                doc_names.add(chunk.get('metadata', {}).get('doc_name', 'Unknown'))
        
        # Remove 'Unknown' if we have real doc names
        if len(doc_names) > 1 and 'Unknown' in doc_names:
            doc_names.discard('Unknown')
        
        result = {
            'total_chunks': total_chunks if total_chunks > 0 else 'N/A',
            'embedding_dim': embedding_dim,
            'num_documents': len(doc_names) if doc_names else 'N/A',
            'doc_names': list(doc_names)[:5]
        }
        
        # Store in session state for persistence
        st.session_state.cached_stats = result
        
        return result
    except Exception as e:
        logger.warning(f"Could not extract summary stats: {e}")
        # Return cached stats if available
        if 'cached_stats' in st.session_state:
            return st.session_state.cached_stats
        return {'total_chunks': 'N/A', 'embedding_dim': 'N/A', 'num_documents': 'N/A', 'doc_names': []}


# Main UI class for the RAG system
class RAGSystemUI:
    """Main UI class for the RAG system with enhanced features"""
    
    # Initialize the UI class
    def __init__(self):
        self.initialize_session_state()  # Set up session state variables
    
    # Method to initialize all session state variables
    def initialize_session_state(self):
        """Initialize session state variables"""
        # Core state
        if 'documents_processed' not in st.session_state:
            st.session_state.documents_processed = False
        if 'system_loaded' not in st.session_state:
            st.session_state.system_loaded = False
        if 'retriever' not in st.session_state:
            st.session_state.retriever = None
        if 'generator' not in st.session_state:
            st.session_state.generator = None
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'total_cost' not in st.session_state:
            st.session_state.total_cost = 0.0
        if 'data_mode' not in st.session_state:
            st.session_state.data_mode = 'upload'
        
        # Multimodal state
        if 'multimodal_enabled' not in st.session_state:
            st.session_state.multimodal_enabled = False
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        if 'image_chunks' not in st.session_state:
            st.session_state.image_chunks = []
        if 'multimodal_retriever' not in st.session_state:
            st.session_state.multimodal_retriever = None
        
        # NEW: Analytics state
        if 'analytics' not in st.session_state:
            st.session_state.analytics = {
                'query_times': [],  # List of (timestamp, response_time)
                'topics': [],  # List of query topics/keywords
                'sources_used': [],  # Counter of source documents used
                'total_tokens': 0,
                'queries_by_hour': {}
            }
        
        # NEW: Follow-up questions state
        if 'follow_up_questions' not in st.session_state:
            st.session_state.follow_up_questions = []
        
        # NEW: Selected follow-up (to auto-populate query)
        if 'selected_followup' not in st.session_state:
            st.session_state.selected_followup = ""
        
        # NEW: Streaming preference
        if 'enable_streaming' not in st.session_state:
            st.session_state.enable_streaming = True
        
        # NEW: Last response for PDF export
        if 'last_response_data' not in st.session_state:
            st.session_state.last_response_data = None
    
    # Method to render the application header
    def render_header(self):
        """Render application header"""
        st.markdown('<div class="main-header">🧠 Agentic Financial RAG</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-tagline">Temporal Reasoning • Knowledge Graphs • Hallucination Grounding</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Advanced Multi-Document Analysis with Intelligent Agent Architecture</div>', unsafe_allow_html=True)
        
        with st.expander("🔬 About This Research System"):
            st.markdown("""
            <div style='color: #e0e0e0; line-height: 1.8;'>
            
            ## Agentic Multi-Document Financial RAG System
            
            **With Temporal Reasoning, Knowledge Graph Integration, and Hallucination Grounding**
            
            This is a cutting-edge research implementation that goes beyond traditional RAG systems by incorporating:
            
            ### 🎯 Core Innovations
            
            #### 1. **Agentic Architecture**
            - Autonomous decision-making for query decomposition
            - Multi-step reasoning with self-reflection
            - Dynamic retrieval strategy selection
            - Adaptive context management
            
            #### 2. **Temporal Reasoning**
            - Time-aware document analysis
            - Historical trend detection
            - Temporal relationship extraction
            - Date-sensitive query handling
            
            #### 3. **Knowledge Graph Integration**
            - Entity relationship mapping
            - Cross-document knowledge synthesis
            - Graph-based reasoning paths
            - Semantic relationship discovery
            
            #### 4. **Hallucination Grounding**
            - Source verification mechanisms
            - Confidence-based response filtering
            - Citation validation and tracking
            - Fact-checking against retrieved context
            
            ### 🚀 Technical Capabilities
            
            - **Multi-Format Processing**: PDF, Excel, CSV with OCR support
            - **Advanced Embeddings**: Multiple state-of-the-art models (MiniLM, MPNet, RoBERTa)
            - **Hybrid Retrieval**: FAISS and ChromaDB vector stores
            - **Multi-LLM Support**: Groq, Google Gemini, OpenAI, Anthropic, Cohere
            - **Real-Time Streaming**: Token-by-token response generation
            - **Semantic Highlighting**: Automatic source-answer matching
            - **Analytics Dashboard**: Performance tracking and insights
            - **PDF Export**: Complete session documentation
            
            ### 📊 Research Focus
            
            This system addresses key challenges in financial document analysis:
            - **Accuracy**: Hallucination detection and grounding
            - **Context**: Temporal awareness and historical reasoning
            - **Intelligence**: Agentic behavior and autonomous decision-making
            - **Transparency**: Full citation tracking and source attribution
            
            ### 🎓 Academic Project
            
            **Developed by:** Adan Malik, Shayan Ahmed, Bilal Raza  
            **Course:** Generative AI  
            **Instructor:** Sir Akhtar Jamil  
            **Institution:** FAST-NUCES
            
            ---
            
            *This system represents the state-of-the-art in retrieval-augmented generation for financial document analysis.*
            
            </div>
            """, unsafe_allow_html=True)
    
    # NEW: Render summary cards at top of interface
    def render_summary_cards(self):
        """Render quick summary cards showing loaded data statistics"""
        if not (st.session_state.documents_processed or st.session_state.system_loaded):
            return
        
        if st.session_state.retriever:
            stats = extract_summary_stats(st.session_state.retriever)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="summary-card">
                    <h3>{stats['num_documents']}</h3>
                    <p>Documents Loaded</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="summary-card">
                    <h3>{stats['total_chunks']}</h3>
                    <p>Text Chunks Indexed</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                queries_count = len(st.session_state.query_history)
                st.markdown(f"""
                <div class="summary-card">
                    <h3>{queries_count}</h3>
                    <p>Queries Processed</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                # Count temporal chunks
                temporal_count = 0
                try:
                    if hasattr(st.session_state.retriever, 'vector_store') and st.session_state.retriever.vector_store:
                        if hasattr(st.session_state.retriever.vector_store, 'chunks'):
                            chunks = st.session_state.retriever.vector_store.chunks
                            temporal_count = sum(1 for c in chunks 
                                               if hasattr(c, 'temporal_entities') and c.temporal_entities)
                except:
                    pass
                
                st.markdown(f"""
                <div class="summary-card">
                    <h3>{temporal_count}</h3>
                    <p>Temporal Chunks</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Refresh button to update stats
            col_refresh = st.columns([4, 1])[1]
            with col_refresh:
                if st.button("Refresh Stats", help="Refresh all statistics"):
                    # Clear cached stats to force recalculation
                    if 'cached_stats' in st.session_state:
                        del st.session_state.cached_stats
                    st.rerun()
    
    # Method to render the sidebar with configuration options
    def render_sidebar(self):
        """Render sidebar with configuration options"""
        with st.sidebar:
            # Branding
            st.markdown("""
            <div style='text-align: center; padding: 1rem 0 0.5rem; margin-bottom: 0.5rem;'>
                <h1 style='font-size: 1.7rem; margin: 0; line-height: 1.2; font-weight: 800; letter-spacing: -0.02em;
                           background: linear-gradient(135deg, #e0f2fe 0%, #22d3ee 60%, #06b6d4 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
                    Agentic RAG
                </h1>
                <p style='color: #22d3ee; font-size: 0.68rem; margin: 0.4rem 0 0 0; font-weight: 600;
                          text-transform: uppercase; letter-spacing: 0.28em;'>Temporal • Knowledge Graph</p>
                <p style='color: #64748b; font-size: 0.7rem; margin: 0.45rem 0 0 0;'>Configuration Panel</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Data Source Selection Section
            st.markdown("### 📁 Data Source")
            data_mode = st.radio(
                "Choose data source",
                options=['upload', 'preloaded'],
                format_func=lambda x: {
                    'upload': '📤 Upload Documents',
                    'preloaded': '⚡ Pre-loaded Data'
                }[x],
                help="Upload new documents or use existing pre-processed data",
                index=0 if st.session_state.data_mode == 'upload' else 1
            )
            
            if data_mode != st.session_state.data_mode:
                st.session_state.data_mode = data_mode
                st.session_state.documents_processed = False
                st.session_state.system_loaded = False
                st.session_state.retriever = None
                st.session_state.generator = None
                st.rerun()
            
            if data_mode == 'preloaded':
                # Note: 27,283 is the chunk count (each PDF gets split into many
                # 512-token chunks), not the document count. Once the index loads
                # the main panel shows the actual document count.
                st.success("✅ Pre-indexed banking corpus ready (27,283 chunks)")
            
            st.divider()
            
            # Embedding Model Selection
            st.markdown("### 🧠 Embedding Model")
            embedding_model = st.selectbox(
                "Select Model",
                options=['minilm', 'mpnet', 'distilbert', 'roberta'],
                format_func=lambda x: {
                    'minilm': '⚡ MiniLM (Fast, 384d)',
                    'mpnet': '⚖️ MPNet (Balanced, 768d)',
                    'distilbert': '🎯 DistilBERT (768d)',
                    'roberta': '🏆 RoBERTa (Best, 1024d)'
                }[x]
            )
            
            # Vector Store Selection
            st.markdown("### 💾 Vector Store")
            vector_store = st.selectbox(
                "Select Store",
                options=['faiss', 'chroma'],
                format_func=lambda x: '⚡ FAISS (Fast)' if x == 'faiss' else '💿 ChromaDB (Persistent)'
            )
            
            # LLM Provider Selection
            st.markdown("### 🤖 LLM Provider")
            llm_provider = st.selectbox(
                "Select Provider",
                options=['groq', 'groq_vision', 'openai', 'anthropic', 'google', 'cohere'],
                format_func=lambda x: {
                    'groq': '🚀 Groq (FREE)',
                    'groq_vision': '👁️ Groq Vision (FREE)',
                    'openai': '🟢 OpenAI GPT',
                    'anthropic': '🔵 Anthropic Claude',
                    'google': '🔴 Google Gemini',
                    'cohere': '🟣 Cohere Command'
                }[x]
            )
            
            # Model selection based on provider
            if llm_provider == 'openai':
                st.warning("⚠️ Requires OPENAI_API_KEY")
                model_name = st.selectbox("Model", options=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'])
            elif llm_provider == 'anthropic':
                st.warning("⚠️ Requires ANTHROPIC_API_KEY")
                model_name = st.selectbox("Model", options=['claude-3-sonnet-20240229', 'claude-3-opus-20240229'])
            elif llm_provider == 'google':
                st.warning("Requires GOOGLE_API_KEY")
                model_name = st.selectbox(
                    "Model",
                    options=['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash'],
                    format_func=lambda x: {
                        'gemini-1.5-flash': 'Gemini 1.5 Flash (Fast)',
                        'gemini-1.5-pro': 'Gemini 1.5 Pro (Best)',
                        'gemini-2.0-flash': 'Gemini 2.0 Flash'
                    }[x]
                )
            elif llm_provider == 'cohere':
                st.info("Requires COHERE_API_KEY")
                model_name = 'command'
            elif llm_provider == 'groq_vision':
                st.success("FREE Vision Model")
                model_name = st.selectbox(
                    "Model",
                    options=['meta-llama/llama-4-scout-17b-16e-instruct', 'meta-llama/llama-4-maverick-17b-128e-instruct'],
                    format_func=lambda x: {
                        'meta-llama/llama-4-scout-17b-16e-instruct': 'Llama 4 Scout',
                        'meta-llama/llama-4-maverick-17b-128e-instruct': 'Llama 4 Maverick'
                    }[x]
                )
            else:  # groq
                st.success("FREE - Requires GROQ_API_KEY")
                model_name = st.selectbox(
                    "Model",
                    options=['llama-3.1-8b-instant', 'llama-3.3-70b-versatile', 'mixtral-8x7b-32768']
                )
            
            # Generation Parameters
            st.subheader("⚙️ Parameters")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
            max_tokens = st.slider("Max Tokens", min_value=256, max_value=2048, value=1000, step=256)
            top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=5)
            
            st.divider()
            
            # NEW: Temporal Filtering
            st.markdown("### 📅 Temporal Filtering")
            enable_temporal = st.toggle(
                "Enable Date Filtering",
                value=False,
                help="Filter results by date range"
            )
            
            enable_temporal_scoring = st.toggle(
                "Enable Temporal Scoring",
                value=True,
                help="Boost recent/relevant documents in ranking"
            )
            
            temporal_config = {'enabled': False, 'scoring_enabled': enable_temporal_scoring}
            if enable_temporal:
                col1, col2 = st.columns(2)
                with col1:
                    from datetime import datetime
                    start_date = st.date_input("From Date", value=datetime(2024, 1, 1))
                with col2:
                    end_date = st.date_input("To Date", value=datetime.now())
                
                temporal_config = {
                    'enabled': True,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'scoring_enabled': enable_temporal_scoring
                }
            
            st.divider()
            
            # NEW: Agentic Capabilities
            st.markdown("### 🤖 Agentic Mode")
            enable_agentic = st.toggle(
                "Enable Agentic Processing",
                value=True,
                help="Autonomous query decomposition and multi-step reasoning"
            )
            
            agentic_config = {'enabled': enable_agentic}
            if enable_agentic:
                enable_reflection = st.toggle(
                    "Enable Self-Reflection",
                    value=True,
                    help="Agent evaluates and refines its own answers"
                )
                
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.5,
                    max_value=0.95,
                    value=0.7,
                    step=0.05,
                    help="Minimum confidence to accept answer without refinement"
                )
                
                agentic_config = {
                    'enabled': True,
                    'reflection': enable_reflection,
                    'confidence_threshold': confidence_threshold
                }
            
            st.divider()

            # NEW: Knowledge Graph toggle.
            # The persisted KG is built offline by `scripts/build_kg.py` and lives
            # at `chunk_metadata/kg.pkl`. If it's missing, we surface a hint instead
            # of a silent failure.
            st.markdown("### 🕸️ Knowledge Graph")
            kg_path = Path("chunk_metadata/kg.pkl")
            kg_present = KG_MODULE_AVAILABLE and kg_path.exists()
            kg_help = (
                "Use a pre-built financial entity KG to expand retrieval (related "
                "entities) and rerank by entity overlap. Improves recall on "
                "multi-entity / relationship questions and adds a confidence bonus."
            )
            enable_kg = st.toggle(
                "Enable Knowledge Graph Retrieval",
                value=kg_present,
                disabled=not kg_present,
                help=kg_help if kg_present else (
                    "KG file not found at chunk_metadata/kg.pkl. "
                    "Build it with: `python scripts/build_kg.py`."
                ),
            )
            if not KG_MODULE_AVAILABLE:
                st.caption("⚠️ KG module not installed (networkx missing).")
            elif not kg_path.exists():
                st.caption("⚠️ Run `python scripts/build_kg.py` to build the graph.")

            kg_config = {'enabled': bool(enable_kg and kg_present)}

            st.divider()

            # NEW: Streaming toggle
            st.subheader("Response Mode")
            enable_streaming = st.toggle(
                "Enable Streaming",
                value=st.session_state.enable_streaming,
                help="Show response word-by-word as it generates"
            )
            st.session_state.enable_streaming = enable_streaming
            
            st.divider()
            
            # Statistics
            if st.session_state.documents_processed or st.session_state.system_loaded:
                st.subheader("Session Stats")
                st.metric("Queries", len(st.session_state.query_history))
                st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
            
            return {
                'data_mode': data_mode,
                'embedding_model': embedding_model,
                'vector_store': vector_store,
                'llm_provider': llm_provider,
                'model_name': model_name,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_k': top_k,
                'temporal_filter': temporal_config,
                'agentic': agentic_config,
                'kg': kg_config
            }
    
    # Method to render document upload section
    def render_document_upload(self, data_mode):
        """Render document upload section"""
        if data_mode == 'preloaded':
            return None
        
        st.header("Step 1: Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload financial documents (PDF, Excel)",
            type=['pdf', 'xlsx', 'csv'],
            accept_multiple_files=True,
            help="Upload quarterly reports, balance sheets, or financial statements"
        )
        
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded")
            file_info = []
            for file in uploaded_files:
                file_info.append({
                    'Filename': file.name,
                    'Type': file.type,
                    'Size (KB)': f"{file.size / 1024:.2f}"
                })
            st.dataframe(pd.DataFrame(file_info), use_container_width=True)
            return uploaded_files
        
        return None
    
    # Method to load preloaded RAG system
    def load_preloaded_system(self, config):
        """Load pre-loaded RAG system"""
        st.header("Step 1: Load Pre-loaded System")
        
        if st.button("Load Pre-loaded System", type="primary"):
            with st.spinner("Loading pre-processed data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    progress_bar.progress(20)
                    status_text.text("Loading chunk metadata and FAISS index...")
                    
                    project_root = Path(__file__).parent.parent.parent
                    metadata_path = project_root / "chunk_metadata" / "chunk_metadata.json"
                    faiss_path = project_root / "chunk_metadata" / "rag_index.faiss"
                    
                    retriever = PreloadedRetriever(
                        metadata_path=str(metadata_path),
                        faiss_path=str(faiss_path),
                        embedding_model="all-MiniLM-L6-v2"
                    )
                    
                    progress_bar.progress(60)
                    status_text.text("Initializing LLM...")
                    
                    gen_config = GenerationConfig(
                        temperature=config['temperature'],
                        max_tokens=config['max_tokens']
                    )
                    
                    generator = RAGGenerator(
                        provider=config['llm_provider'],
                        model_name=config['model_name'],
                        config=gen_config
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("System loaded!")
                    
                    st.session_state.retriever = retriever
                    st.session_state.generator = generator
                    st.session_state.system_loaded = True

                    # Lazily load the KG once, if both the module and the persisted
                    # graph are available. The toggle in the sidebar guards usage,
                    # so the load is cheap insurance — graph fits in RAM and helps
                    # subsequent queries without a per-query load cost.
                    if KG_MODULE_AVAILABLE:
                        kg_path = project_root / "chunk_metadata" / "kg.pkl"
                        if kg_path.exists():
                            try:
                                kg_obj = FinancialKnowledgeGraph.load(str(kg_path))
                                if kg_obj is not None:
                                    st.session_state.kg_graph = kg_obj
                                    st.session_state.kg_extractor = (
                                        FinancialEntityExtractor(use_spacy=False)
                                    )
                                    logger.info(
                                        "✅ KG loaded: %s",
                                        kg_obj.stats(),
                                    )
                            except Exception as kg_exc:  # noqa: BLE001
                                logger.warning(f"KG load failed: {kg_exc}")

                    st.success("System ready for queries!")
                    
                except FileNotFoundError as e:
                    st.error("Pre-loaded data files not found. Ensure chunk_metadata/ directory exists.")
                    logger.error(f"Pre-loaded data not found: {e}")
                except Exception as e:
                    st.error(f"Error loading system: {str(e)}")
                    logger.error(f"Pre-loaded system error: {str(e)}")
    
    # Method to process uploaded documents
    def process_documents(self, files, config):
        """Process uploaded documents"""
        st.header("Step 2: Process Documents")
        
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                project_root = Path(__file__).parent.parent.parent
                temp_dir = project_root / "data" / "temp"
                
                try:
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    
                    for file in files:
                        file_path = temp_dir / file.name
                        with open(file_path, 'wb') as f:
                            f.write(file.getbuffer())
                    
                    progress_bar.progress(20)
                    status_text.text(
                        "Extracting text + tables in parallel… "
                        "(long PDFs run on multiple CPU cores; see container logs for per-page progress)"
                    )

                    pipeline = DocumentPipeline()

                    use_multimodal = config.get('llm_provider') == 'groq_vision' and MULTIMODAL_AVAILABLE

                    if use_multimodal:
                        status_text.text(
                            "Extracting text, tables, and images in parallel… "
                            "(see container logs for per-page progress)"
                        )
                        chunks, image_chunks = pipeline.process_directory_multimodal(str(temp_dir))
                        st.session_state.image_chunks = image_chunks
                    else:
                        chunks = pipeline.process_directory(str(temp_dir))
                        st.session_state.image_chunks = []
                    
                    progress_bar.progress(50)
                    status_text.text("Generating embeddings...")
                    
                    retriever = HybridRetriever(
                        embedding_model=config['embedding_model'],
                        vector_store_type=config['vector_store'],
                        top_k=config['top_k']
                    )
                    
                    retriever.index_documents(chunks)
                    
                    progress_bar.progress(80)
                    status_text.text("Initializing LLM...")
                    
                    gen_config = GenerationConfig(
                        temperature=config['temperature'],
                        max_tokens=config['max_tokens']
                    )
                    
                    generator = RAGGenerator(
                        provider=config['llm_provider'],
                        model_name=config['model_name'],
                        config=gen_config
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    
                    st.session_state.retriever = retriever
                    st.session_state.generator = generator
                    st.session_state.documents_processed = True
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Documents", len(files))
                    with col2:
                        st.metric("Chunks", len(chunks))
                    with col3:
                        st.metric("Model", config['model_name'])
                    
                    st.success("System ready for queries!")
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    logger.error(f"Document processing error: {str(e)}")
                finally:
                    import shutil
                    if temp_dir.exists():
                        try:
                            shutil.rmtree(temp_dir)
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup temp directory: {cleanup_error}")
    
    # Method to render query interface
    def render_query_interface(self, config):
        """Render query interface"""
        step_num = "2" if config['data_mode'] == 'preloaded' else "3"
        st.header(f"Step {step_num}: Ask Questions")
        
        system_ready = st.session_state.documents_processed or st.session_state.system_loaded
        
        if not system_ready:
            if config['data_mode'] == 'preloaded':
                st.warning("Please load the pre-loaded system first")
            else:
                st.warning("Please upload and process documents first")
            return
        
        # NEW: Render summary cards
        self.render_summary_cards()
        
        # Example queries
        with st.expander("💡 Example Queries - Showcasing Advanced Capabilities"):
            st.markdown("""
            **🔍 Factual Retrieval:**
            - What was the total revenue in Q3 2023?
            - What are the current credit card interest rates?
            
            **📊 Temporal Reasoning:**
            - How has revenue changed over the last 3 quarters?
            - What were the key financial metrics in 2022 vs 2023?
            - Show me the trend in operating expenses over time
            - What were the credit card rates in Q2 2024?
            - Show me documents valid for July-December 2025
            - Find information effective from July 1, 2025
            
            **🧮 Multi-Step Calculations:**
            - Calculate the P/E ratio from the financial statements
            - What is the year-over-year growth rate in net income?
            - Compute the debt-to-equity ratio and explain its implications
            
            **🔗 Knowledge Graph Queries:**
            - What is the relationship between revenue and marketing spend?
            - How are different product lines connected to overall profitability?
            - Map the dependencies between business segments
            
            **⚖️ Comparative Analysis:**
            - Compare gross margins across all quarters
            - How does operating income compare to previous year?
            - Contrast the performance of different business units
            
            **🎯 Hallucination-Grounded Responses:**
            - Provide evidence for the claim that revenue increased
            - What specific documents support the reported profit margin?
            - Verify the accuracy of the stated market share
            """)
        
        # Image upload section for vision models
        uploaded_image = None
        if config.get('llm_provider') == 'groq_vision':
            st.markdown("### Image Analysis (Optional)")
            st.caption("Upload an image to ask questions about charts, graphs, or visual content")
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=['png', 'jpg', 'jpeg', 'webp'],
                help="Upload a chart, graph, or any visual content to analyze"
            )
            
            if uploaded_image:
                # Display uploaded image
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
                st.session_state.uploaded_image = uploaded_image.getvalue()
            else:
                st.session_state.uploaded_image = None
        
        # Query input - check for selected follow-up
        default_query = st.session_state.selected_followup if st.session_state.selected_followup else ""
        if st.session_state.selected_followup:
            st.session_state.selected_followup = ""  # Clear after using
        
        query = st.text_area(
            "Enter your question:", 
            value=default_query,
            height=100, 
            placeholder="e.g., What was the total revenue?"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_button = st.button("Search", type="primary")
        with col2:
            clear_button = st.button("Clear History")
        
        if clear_button:
            st.session_state.query_history = []
            st.session_state.total_cost = 0.0
            st.session_state.analytics = {
                'query_times': [],
                'topics': [],
                'sources_used': [],
                'total_tokens': 0,
                'queries_by_hour': {}
            }
            st.session_state.follow_up_questions = []
            st.rerun()
        
        if search_button and query:
            self.process_query(query, config)
    
    # Method to process user query
    def process_query(self, query, config):
        """Process user query and display results"""
        
        # Check if agentic mode is enabled
        agentic_config = config.get('agentic', {})
        use_agentic = agentic_config.get('enabled', False) and AGENTIC_AVAILABLE
        
        if use_agentic:
            # Use agentic processing
            self.process_query_agentic(query, config)
        else:
            # Use standard processing
            self.process_query_standard(query, config)
    
    # Method for agentic query processing
    def process_query_agentic(self, query, config):
        """Process query using agentic module with multi-step reasoning"""
        start_time = time.time()
        
        try:
            # Initialize agentic components
            planner = AgentPlanner()
            
            # Prepare retriever
            from retrieval.temporal_retriever import TemporalAwareRetriever
            retriever = st.session_state.retriever
            if config.get('temporal_filter', {}).get('scoring_enabled', True):
                if not isinstance(retriever, TemporalAwareRetriever):
                    retriever = TemporalAwareRetriever(
                        base_retriever=retriever,
                        enable_temporal_scoring=True,
                        enable_query_expansion=True
                    )
            
            # Create a wrapper for the generator to work with AgentExecutor
            class GeneratorWrapper:
                def __init__(self, generator):
                    self.generator = generator
                    self.llm = self  # AgentExecutor expects generator.llm
                    self.model_name = generator.model_name
                
                def generate(self, prompt, system_prompt=None):
                    """Generate using the wrapped generator"""
                    return self.generator.llm.generate(prompt, system_prompt)
            
            wrapped_generator = GeneratorWrapper(st.session_state.generator)
            
            # Initialize executor
            agentic_config = config.get('agentic', {})
            executor = AgentExecutor(
                retriever=retriever,
                generator=wrapped_generator,
                max_iterations=agentic_config.get('max_iterations', 2),
                confidence_threshold=0.7
            )
            
            # Phase 1: Planning
            st.markdown("### 🧠 Agentic Processing")
            with st.expander("📋 Query Analysis & Planning", expanded=True):
                plan = planner.analyze_query(query)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Complexity", plan.complexity.value.upper())
                with col2:
                    st.metric("Query Type", plan.query_type.value)
                with col3:
                    st.metric("Steps", len(plan.sub_queries))
                
                if len(plan.sub_queries) > 1:
                    st.markdown("**Query Decomposition:**")
                    for i, sq in enumerate(plan.sub_queries, 1):
                        deps = f" *(depends on: {', '.join(map(str, [d+1 for d in sq.dependencies]))})*" if sq.dependencies else ""
                        st.markdown(f"{i}. {sq.question}{deps}")
                else:
                    st.info("Simple query - no decomposition needed")
            
            # Phase 2: Execution
            with st.spinner("🚀 Executing multi-step reasoning..."):
                result = executor.execute(
                    plan=plan,
                    top_k=config['top_k'],
                    enable_reflection=agentic_config.get('reflection', True)
                )
            
            # Display execution details
            if len(result.steps) > 1:
                with st.expander("🔍 Execution Steps", expanded=False):
                    for i, step in enumerate(result.steps, 1):
                        st.markdown(f"**Step {i}:** {step.sub_query.question}")
                        st.markdown(f"- Confidence: {step.confidence:.2f}")
                        st.markdown(f"- Time: {step.execution_time:.2f}s")
                        st.markdown(f"- Retrieved: {len(step.retrieved_chunks)} chunks")
                        
                        if step.reflection and step.reflection.issues:
                            st.warning(f"Issues: {', '.join(step.reflection.issues[:2])}")
                        
                        st.divider()
            
            # Display reflection info
            if result.iterations > 0:
                with st.expander("🔄 Self-Reflection & Refinement", expanded=False):
                    st.info(f"Answer was refined {result.iterations} time(s) to improve quality")
                    for step in result.steps:
                        if step.reflection and step.reflection.suggestions:
                            st.markdown("**Improvements made:**")
                            for suggestion in step.reflection.suggestions[:3]:
                                st.markdown(f"- {suggestion}")
            
            # Collect all retrieved chunks for display
            all_chunks = []
            for step in result.steps:
                all_chunks.extend(step.retrieved_chunks)
            
            # Remove duplicates based on chunk_id
            seen_ids = set()
            unique_chunks = []
            for chunk in all_chunks:
                chunk_id = getattr(chunk, 'chunk_id', id(chunk))
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    unique_chunks.append(chunk)
            
            # Extract citations from final answer
            citations = st.session_state.generator.llm.extract_citations(
                result.final_answer, 
                unique_chunks
            )
            
            # Create response object
            from generation.generator import GeneratedResponse
            response = GeneratedResponse(
                answer=result.final_answer,
                citations=citations,
                model_used=st.session_state.generator.model_name,
                prompt_tokens=0,  # Not tracked in agentic mode
                completion_tokens=0,
                total_cost=0.0
            )
            
            total_time = time.time() - start_time
            
            # Use agentic confidence score
            confidence_score = int(result.final_confidence * 100)
            if confidence_score >= 70:
                confidence_level = "High"
            elif confidence_score >= 40:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            
            # Update session state
            st.session_state.query_history.append({
                'query': query,
                'response': response,
                'retrieved': unique_chunks,
                'time': total_time,
                'confidence': confidence_score,
                'timestamp': datetime.now(),
                'agentic': True,
                'steps': len(result.steps),
                'iterations': result.iterations
            })
            
            # Update analytics
            st.session_state.analytics['query_times'].append((datetime.now(), total_time))
            st.session_state.analytics['topics'].extend(query.lower().split()[:5])
            for chunk in unique_chunks:
                if hasattr(chunk, 'metadata'):
                    st.session_state.analytics['sources_used'].append(
                        chunk.metadata.get('doc_name', 'Unknown')
                    )
            
            # Store for PDF export
            st.session_state.last_response_data = {
                'query': query,
                'answer': result.final_answer,
                'citations': citations,
                'sources': unique_chunks,
                'metrics': {
                    'response_time': round(total_time, 2),
                    'tokens': 0,
                    'cost': 0.0,
                    'confidence': confidence_score,
                    'steps': len(result.steps),
                    'iterations': result.iterations
                }
            }
            
            # Display results
            self.display_results(
                query, response, unique_chunks, total_time,
                confidence_score, confidence_level, already_streamed=False
            )
            
            # Generate follow-up questions
            try:
                follow_ups = generate_follow_up_questions(
                    query, result.final_answer, st.session_state.generator
                )
                st.session_state.follow_up_questions = follow_ups
            except Exception:
                st.session_state.follow_up_questions = []
                
        except Exception as e:
            st.error(f"Agentic processing error: {str(e)}")
            logger.error(f"Agentic error: {e}", exc_info=True)
            st.info("Falling back to standard processing...")
            self.process_query_standard(query, config)
    
    # Method for standard query processing
    def process_query_standard(self, query, config):
        """Process query using standard RAG pipeline"""
        with st.spinner("Searching documents..."):
            start_time = time.time()
            
            try:
                # Import temporal retriever
                from retrieval.temporal_retriever import TemporalAwareRetriever
                
                # Wrap retriever with temporal awareness if scoring enabled
                base_preloaded = st.session_state.retriever  # the PreloadedRetriever
                retriever = base_preloaded
                if config.get('temporal_filter', {}).get('scoring_enabled', True):
                    if not isinstance(retriever, TemporalAwareRetriever):
                        retriever = TemporalAwareRetriever(
                            base_retriever=retriever,
                            enable_temporal_scoring=True,
                            enable_query_expansion=True
                        )
                        logger.info("✅ Using temporal-aware retriever")

                # Wrap with KG-aware retriever ON TOP if user enabled it.
                # KG runs after temporal so it expands/reranks the temporally-relevant
                # candidate set. The original PreloadedRetriever is passed as the
                # chunk_loader so KG can resolve chunk_ids during expansion.
                kg_aware = None
                if config.get('kg', {}).get('enabled') and KG_MODULE_AVAILABLE:
                    kg_obj = st.session_state.get('kg_graph')
                    kg_extractor = st.session_state.get('kg_extractor')
                    if kg_obj is not None and kg_extractor is not None:
                        kg_aware = KGAwareRetriever(
                            base_retriever=retriever,
                            kg=kg_obj,
                            extractor=kg_extractor,
                            chunk_loader=base_preloaded,
                        )
                        retriever = kg_aware
                        logger.info("✅ Using KG-aware retriever")

                # Retrieve relevant chunks. We only pass temporal_filter to retrievers
                # that understand it (Temporal or KG-wrapping-Temporal). PreloadedRetriever
                # alone does not accept that kwarg.
                temporal_filter = config.get('temporal_filter', {})
                retrieve_kwargs = {'top_k': config['top_k']}
                if isinstance(retriever, (TemporalAwareRetriever,)) or kg_aware is not None:
                    retrieve_kwargs['temporal_filter'] = (
                        temporal_filter if temporal_filter.get('enabled') else None
                    )
                raw_results = retriever.retrieve(query, **retrieve_kwargs)
                
                # Convert to standard format (only if not already RetrievalResult)
                retrieved_results = raw_results
                if raw_results and not isinstance(raw_results[0], RetrievalResult):
                    if isinstance(st.session_state.retriever, PreloadedRetriever):
                        retrieved_results = []
                        for rank, result in enumerate(raw_results, start=1):
                            retrieved_results.append(RetrievalResult(
                                chunk_id=result.get('chunk_id', ''),
                                content=result.get('content', ''),
                                score=result.get('score', 0.0),
                                metadata=result.get('metadata', {}),
                                rank=rank
                            ))
                
                retrieval_time = time.time() - start_time

                generator = st.session_state.generator
                llm = generator.llm

                # ----- Hallucination / scope gate -----
                # Refuse out-of-scope or under-grounded queries BEFORE calling the LLM.
                # Triggered when the best retrieved chunk is below the similarity floor
                # (catches off-domain queries like "give me a python script").
                if not generator.is_query_in_scope(retrieved_results):
                    top_score = max(
                        (getattr(c, 'score', 0.0) or 0.0 for c in retrieved_results),
                        default=0.0,
                    )
                    logger.info(
                        f"🚫 Streamlit: refusing out-of-scope query "
                        f"(top score={top_score:.3f})"
                    )
                    refusal = generator.OUT_OF_SCOPE_REFUSAL

                    st.markdown("### Answer")
                    st.markdown(
                        f'<div class="answer-box">{refusal}</div>',
                        unsafe_allow_html=True,
                    )
                    st.info(
                        "This question doesn't appear to relate to the loaded financial "
                        "documents. Try rephrasing as a question about specific document "
                        "content (rates, fees, terms, dates, etc.)."
                    )

                    from generation.generator import GeneratedResponse
                    response = GeneratedResponse(
                        answer=refusal,
                        citations=[],
                        model_used=llm.model_name,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_cost=0.0,
                    )
                    total_time = time.time() - start_time

                    st.session_state.query_history.append({
                        'query': query,
                        'response': response,
                        'retrieved': retrieved_results,
                        'time': total_time,
                        'confidence': 0,
                        'timestamp': datetime.now(),
                        'refused': True,
                        'refusal_reason': f'low_top_score={top_score:.3f}',
                    })
                    st.session_state.analytics['query_times'].append(
                        (datetime.now(), total_time)
                    )
                    st.session_state.last_response_data = {
                        'query': query,
                        'answer': refusal,
                        'citations': [],
                        'sources': [],
                        'metrics': {
                            'response_time': round(total_time, 2),
                            'tokens': 0,
                            'cost': 0.0,
                            'confidence': 0,
                        },
                        'refused': True,
                    }
                    st.stop()

                # Build context and prompts
                context_parts = []
                for idx, chunk in enumerate(retrieved_results, start=1):
                    doc_name = chunk.metadata.get('doc_name', 'Unknown')
                    page = chunk.metadata.get('page', 0)
                    context_parts.append(f"[Document {idx}: {doc_name}, Page {page}]\n{chunk.content}\n")
                context = "\n".join(context_parts)
                
                system_prompt = generator._get_system_prompt()
                user_prompt = generator._format_prompt(query, context, retrieved_results)
                
                # Check if using vision model with images
                use_vision = (
                    config.get('llm_provider') == 'groq_vision' and 
                    st.session_state.uploaded_image is not None and
                    hasattr(llm, 'generate_with_images')
                )
                
                # Initialize already_streamed flag
                already_streamed = False
                
                if use_vision:
                    # Vision model with image analysis
                    st.markdown("### Answer (Vision Analysis)")
                    
                    # Collect images for analysis
                    images_to_analyze = [st.session_state.uploaded_image]
                    
                    # Also check for images from multimodal retriever
                    if st.session_state.multimodal_retriever:
                        try:
                            mm_results = st.session_state.multimodal_retriever.retrieve_by_text(
                                query, top_k=3, include_images=True
                            )
                            for result in mm_results:
                                if result.is_image and result.image_bytes:
                                    images_to_analyze.append(result.image_bytes)
                            if len(images_to_analyze) > 1:
                                logger.info(f"Including {len(images_to_analyze)} images in vision analysis")
                        except Exception as mm_error:
                            logger.warning(f"Multimodal retrieval failed: {mm_error}")
                    
                    # Create vision-enhanced prompt
                    vision_prompt = f"""CONTEXT DOCUMENTS:
{context}

USER QUESTION:
{query}

Analyze the provided images along with the context documents to give a comprehensive answer.
For claims from documents, cite using [Source: DocumentName, Page X] format.
For observations from images, describe what you see clearly."""
                    
                    with st.spinner("Analyzing images and generating answer..."):
                        answer, usage = llm.generate_with_images(
                            vision_prompt, 
                            images=images_to_analyze[:3],  # Limit to 3 images for API
                            system_prompt="You are a financial analysis assistant that analyzes both images and documents."
                        )
                    
                    # Display answer
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                    already_streamed = True
                
                # Check if streaming is enabled and supported (non-vision mode)
                elif st.session_state.enable_streaming and config['llm_provider'] == 'groq' and hasattr(llm, 'generate_stream'):
                    # Streaming response
                    st.markdown("### Answer")
                    answer_placeholder = st.empty()
                    full_answer = ""
                    
                    with st.spinner("Generating..."):
                        for token in llm.generate_stream(user_prompt, system_prompt):
                            full_answer += token
                            answer_placeholder.markdown(f'<div class="answer-box">{full_answer}</div>', unsafe_allow_html=True)
                    
                    # Get usage estimate (streaming doesn't return usage)
                    usage = {
                        'prompt_tokens': len(user_prompt.split()) * 1.3,  # Rough estimate
                        'completion_tokens': len(full_answer.split()) * 1.3,
                        'cost': 0.0
                    }
                    answer = full_answer
                    already_streamed = True
                else:
                    # Regular generation
                    with st.spinner("Generating answer..."):
                        answer, usage = llm.generate(user_prompt, system_prompt)
                    already_streamed = False
                
                # Extract citations
                citations = llm.extract_citations(answer, retrieved_results)
                
                # Create response object
                from generation.generator import GeneratedResponse
                response = GeneratedResponse(
                    answer=answer,
                    citations=citations,
                    model_used=llm.model_name,
                    prompt_tokens=int(usage.get('prompt_tokens', 0)),
                    completion_tokens=int(usage.get('completion_tokens', 0)),
                    total_cost=usage.get('cost', 0.0)
                )
                
                total_time = time.time() - start_time
                
                # Calculate confidence score (with optional KG bonus)
                kg_coverage = None
                if kg_aware is not None:
                    try:
                        kg_coverage = kg_aware.query_entity_coverage(retrieved_results)
                    except Exception as kg_exc:  # noqa: BLE001
                        logger.debug(f"KG coverage calc failed: {kg_exc}")
                confidence_score, confidence_level = calculate_confidence_score(
                    retrieved_results, answer, citations, kg_coverage=kg_coverage
                )
                
                # Update session state
                st.session_state.total_cost += response.total_cost
                st.session_state.query_history.append({
                    'query': query,
                    'response': response,
                    'retrieved': retrieved_results,
                    'time': total_time,
                    'confidence': confidence_score,
                    'timestamp': datetime.now()
                })
                
                # Update analytics
                st.session_state.analytics['query_times'].append((datetime.now(), total_time))
                st.session_state.analytics['topics'].extend(query.lower().split()[:5])
                st.session_state.analytics['total_tokens'] += response.prompt_tokens + response.completion_tokens
                for result in retrieved_results:
                    st.session_state.analytics['sources_used'].append(
                        result.metadata.get('doc_name', 'Unknown')
                    )
                
                # Store for PDF export
                st.session_state.last_response_data = {
                    'query': query,
                    'answer': answer,
                    'citations': citations,
                    'sources': retrieved_results,
                    'metrics': {
                        'response_time': round(total_time, 2),
                        'tokens': response.prompt_tokens + response.completion_tokens,
                        'cost': response.total_cost,
                        'confidence': confidence_score
                    }
                }
                
                # Display results
                self.display_results(query, response, retrieved_results, total_time, 
                                   confidence_score, confidence_level, already_streamed)
                
                # Generate follow-up questions (async-like, after main display)
                try:
                    follow_ups = generate_follow_up_questions(query, answer, generator)
                    st.session_state.follow_up_questions = follow_ups
                except Exception:
                    st.session_state.follow_up_questions = []
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"Error: {error_msg}")
                logger.error(f"Query processing error: {error_msg}", exc_info=True)
    
    # Method to display query results
    def display_results(self, query, response, retrieved_results, total_time, 
                       confidence_score, confidence_level, already_streamed=False):
        """Display query results with enhanced features"""
        st.divider()
        st.subheader("Results")
        
        # Display answer (if not already streamed)
        if not already_streamed:
            st.markdown("### Answer")
            st.markdown(f'<div class="answer-box">{response.answer}</div>', unsafe_allow_html=True)
        
        # NEW: Confidence meter
        confidence_class = f"confidence-{confidence_level.lower()}"
        st.markdown(f"""
        <div class="confidence-meter {confidence_class}">
            Confidence: {confidence_score}% ({confidence_level})
        </div>
        """, unsafe_allow_html=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Response Time", f"{total_time:.2f}s")
        with col2:
            st.metric("Citations", len(response.citations))
        with col3:
            st.metric("Tokens", response.completion_tokens)
        with col4:
            st.metric("Cost", f"${response.total_cost:.4f}")
        
        # NEW: PDF Export button
        if PDF_EXPORT_AVAILABLE and st.session_state.last_response_data:
            try:
                exporter = RAGReportExporter()
                data = st.session_state.last_response_data
                pdf_bytes = exporter.export_query_result(
                    query=data['query'],
                    answer=data['answer'],
                    citations=data['citations'],
                    retrieved_sources=data['sources'],
                    metrics=data['metrics']
                )
                st.download_button(
                    label="Export to PDF",
                    data=pdf_bytes,
                    file_name=f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                logger.warning(f"PDF export failed: {e}")
        
        # NEW: Follow-up questions
        if st.session_state.follow_up_questions:
            st.markdown("### Suggested Follow-ups")
            cols = st.columns(len(st.session_state.follow_up_questions))
            for i, (col, question) in enumerate(zip(cols, st.session_state.follow_up_questions)):
                with col:
                    if st.button(f"{question}", key=f"followup_{i}"):
                        st.session_state.selected_followup = question
                        st.rerun()
        
        # NEW: Retrieved Sources with semantic highlighting
        with st.expander(f"Retrieved Sources ({len(retrieved_results)} chunks)", expanded=False):
            for i, result in enumerate(retrieved_results, 1):
                doc_name = result.metadata.get('doc_name', 'Unknown')
                page = result.metadata.get('page', '?')
                
                st.markdown(f"**Source {i}** - {doc_name} (Page {page})")
                st.markdown(f"*Relevance Score: {result.score:.3f}*")
                
                # Apply semantic highlighting
                highlighted_content = highlight_matching_phrases(
                    result.content[:400], 
                    response.answer
                )
                st.markdown(f"{highlighted_content}...", unsafe_allow_html=True)
                st.divider()
        
        # NEW: Temporal Information Display
        with st.expander(f"📅 Temporal Information", expanded=False):
            temporal_chunks = []
            for result in retrieved_results:
                has_temporal = False
                if hasattr(result, 'temporal_entities') and result.temporal_entities:
                    has_temporal = True
                elif hasattr(result, 'valid_from') and result.valid_from:
                    has_temporal = True
                
                if has_temporal:
                    temporal_chunks.append(result)
            
            if temporal_chunks:
                st.markdown(f"**{len(temporal_chunks)} chunks with temporal information**")
                
                for i, result in enumerate(temporal_chunks, 1):
                    doc_name = result.metadata.get('doc_name', 'Unknown')
                    page = result.metadata.get('page', '?')
                    
                    st.markdown(f"**Source {i}** - {doc_name} (Page {page})")
                    
                    # Show validity period
                    if hasattr(result, 'valid_from') and result.valid_from:
                        st.markdown(f"📆 **Valid Period**: `{result.valid_from}` to `{result.valid_to}`")
                    
                    # Show extracted entities
                    if hasattr(result, 'temporal_entities') and result.temporal_entities:
                        st.markdown("**Temporal Entities Found:**")
                        for entity in result.temporal_entities[:5]:  # Show max 5
                            entity_type = entity.temporal_type.value if hasattr(entity, 'temporal_type') else 'UNKNOWN'
                            st.markdown(f"- `{entity.text}` ({entity_type}): {entity.start_date} → {entity.end_date}")
                    
                    st.divider()
            else:
                st.info("No temporal information found in retrieved sources")
        
        # Citations Analysis
        if response.citations:
            with st.expander(f"Citations Analysis ({len(response.citations)} citations)"):
                citation_df = pd.DataFrame(response.citations)
                st.dataframe(citation_df, use_container_width=True)
                
                valid_count = sum(1 for c in response.citations if c.get('valid', True))
                fig = go.Figure(data=[go.Pie(
                    labels=['Valid', 'Invalid'],
                    values=[valid_count, len(response.citations) - valid_count],
                    marker_colors=['#4CAF50', '#F44336']
                )])
                fig.update_layout(title="Citation Validity", height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # NEW: Analytics Dashboard
    def render_analytics_dashboard(self):
        """Render query analytics dashboard"""
        if not st.session_state.query_history:
            return
        
        with st.expander("Query Analytics Dashboard", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Response time trend
                if st.session_state.analytics['query_times']:
                    times = st.session_state.analytics['query_times']
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(times) + 1)),
                        y=[t[1] for t in times],
                        mode='lines+markers',
                        name='Response Time',
                        line=dict(color='#2196F3')
                    ))
                    fig.update_layout(
                        title="Response Time Trend",
                        xaxis_title="Query #",
                        yaxis_title="Time (s)",
                        height=250
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Most used sources
                if st.session_state.analytics['sources_used']:
                    source_counts = Counter(st.session_state.analytics['sources_used'])
                    top_sources = source_counts.most_common(5)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[s[1] for s in top_sources],
                            y=[s[0] for s in top_sources],
                            orientation='h',
                            marker_color='#4CAF50'
                        )
                    ])
                    fig.update_layout(
                        title="Most Referenced Sources",
                        xaxis_title="References",
                        height=250
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            st.markdown("---")
            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.metric("Total Queries", len(st.session_state.query_history))
            with stat_cols[1]:
                avg_time = sum(t[1] for t in st.session_state.analytics['query_times']) / max(1, len(st.session_state.analytics['query_times']))
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
            with stat_cols[2]:
                st.metric("Total Tokens", st.session_state.analytics['total_tokens'])
            with stat_cols[3]:
                st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    
    # Method to render query history
    def render_history(self):
        """Render query history"""
        if st.session_state.query_history:
            st.header("Query History")
            
            for i, item in enumerate(reversed(st.session_state.query_history), 1):
                idx = len(st.session_state.query_history) - i + 1
                confidence = item.get('confidence', 'N/A')
                with st.expander(f"Query {idx}: {item['query'][:50]}... (Confidence: {confidence}%)"):
                    st.markdown(f"**Question:** {item['query']}")
                    st.markdown(f"**Answer:** {item['response'].answer[:300]}...")
                    st.markdown(f"**Time:** {item['time']:.2f}s | **Cost:** ${item['response'].total_cost:.4f}")
    
    # Main application runner
    def run(self):
        """Main application loop"""
        self.render_header()
        config = self.render_sidebar()
        
        if config['data_mode'] == 'preloaded':
            if not st.session_state.system_loaded:
                self.load_preloaded_system(config)
            else:
                st.info("Pre-loaded system is ready. Start asking questions!")
        else:
            uploaded_files = self.render_document_upload(config['data_mode'])
            if uploaded_files:
                self.process_documents(uploaded_files, config)
        
        self.render_query_interface(config)
        self.render_analytics_dashboard()  # NEW
        self.render_history()


# Application entry point
if __name__ == "__main__":
    app = RAGSystemUI()
    app.run()