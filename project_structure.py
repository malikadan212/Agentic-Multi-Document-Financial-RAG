"""
Multi-Document Financial Analysis System - Project Structure
Team: Anas Bin Rashid (22I-0907), Rehan Tariq (22I-0965), Huzaifa Nasir (22I-1053),
      Adan Malik (22I-1000), Saad Mursaleen (22I-0835), Arshman Khawar (22I-2427)
Course: Natural Language Processing CS-7A
Instructor: Sir Owais Idrees
"""

import os
from pathlib import Path

class ProjectStructure:
    """
    Defines and creates the complete project directory structure
    following best practices for ML/AI projects
    """
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.structure = {
            "data": {
                "raw": ["pdfs", "excel"],
                "processed": ["chunks", "embeddings"],
                "temp": [],
                "test": ["questions", "ground_truth"]
            },
            "src": {
                "document_processing": [],
                "retrieval": [],
                "generation": [],
                "evaluation": [],
                "streamlit_app": [],
                "utils": []
            },
            "chunk_metadata": [],
            "models": ["embeddings", "vector_stores"],
            "notebooks": ["exploration", "experiments", "analysis"],
            "tests": ["unit", "integration"],
            "results": ["metrics", "visualizations", "reports"],
            "prompts": [],
            "docs": ["api", "user_guide"]
        }
    
    def create_structure(self):
        """Create all directories in the project structure"""
        for main_dir, subdirs in self.structure.items():
            main_path = self.base_path / main_dir
            main_path.mkdir(exist_ok=True)
            
            if isinstance(subdirs, dict):
                for subdir, subsubdirs in subdirs.items():
                    sub_path = main_path / subdir
                    sub_path.mkdir(exist_ok=True)
                    for subsubdir in subsubdirs:
                        (sub_path / subsubdir).mkdir(exist_ok=True)
            else:
                for subdir in subdirs:
                    (main_path / subdir).mkdir(exist_ok=True)
        
        print("Project structure created successfully!")
    
    def create_init_files(self):
        """Create __init__.py files for Python packages"""
        src_path = self.base_path / "src"
        for module in self.structure["src"].keys():
            init_file = src_path / module / "__init__.py"
            init_file.touch()
    
    def create_config_files(self):
        """Create essential configuration files"""
        configs = {
            ".gitignore": self._get_gitignore_content(),
            "requirements.txt": self._get_requirements(),
            "README.md": self._get_readme(),
            ".env.example": self._get_env_example(),
            "config.yaml": self._get_config_yaml()
        }
        
        for filename, content in configs.items():
            filepath = self.base_path / filename
            with open(filepath, 'w') as f:
                f.write(content)
    
    def _get_gitignore_content(self) -> str:
        return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Data
data/raw/
data/processed/
data/temp/
*.pdf
*.xlsx
*.csv

# Models
models/
*.pkl
*.h5
*.pt

# API Keys
.env
*.key

# IDE
.vscode/
.idea/
*.swp

# Results
results/
logs/

# Cache
.cache/
*.cache
"""

    def _get_requirements(self) -> str:
        return """# Core Dependencies
streamlit>=1.28.0
groq>=0.13.0

# LLM APIs
openai>=1.0.0
anthropic>=0.7.0
google-generativeai>=0.3.0
cohere>=4.0.0

# Embeddings and Retrieval
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
chromadb>=0.4.0
transformers>=4.30.0

# Document Processing
PyMuPDF>=1.23.0
pdfplumber>=0.10.0
pandas>=2.0.0
pytesseract>=0.3.10

# Multimodal
Pillow>=9.0.0
torch>=2.0.0
open-clip-torch>=2.20.0

# Visualization
plotly>=5.17.0

# PDF Export
fpdf2>=2.7.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
numpy>=1.24.0

# Testing
pytest>=7.4.0
"""

    def _get_readme(self) -> str:
        return """# Multi-Document Financial Analysis System Using RAG

## Overview
A Retrieval-Augmented Generation (RAG) system for intelligent financial document analysis.

## Team
- Anas Bin Rashid (22I-0907)
- Rehan Tariq (22I-0965)
- Huzaifa Nasir (22I-1053)
- Adan Malik (22I-1000)
- Saad Mursaleen (22I-0835)
- Arshman Khawar (22I-2427)

**Course:** Natural Language Processing CS-7A | **Instructor:** Sir Owais Idrees

## Features
- Multi-format document support (PDF, Excel)
- Semantic search with multiple embedding models
- Multiple LLM providers (Groq, OpenAI, Anthropic, Google, Cohere)
- Automatic citation generation
- Streaming responses
- Follow-up question suggestions
- Confidence scoring
- PDF export
- Analytics dashboard

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run src/streamlit_app/app.py
```

## Docker
```bash
docker-compose up --build
```
"""

    def _get_env_example(self) -> str:
        return """# API Keys (at least one required)
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
COHERE_API_KEY=your_cohere_key_here

# Model Configuration
DEFAULT_LLM=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE=faiss

# Application Settings
MAX_CHUNKS=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50
"""

    def _get_config_yaml(self) -> str:
        return """# Configuration for RAG System

document_processing:
  chunk_size: 512
  chunk_overlap: 50
  supported_formats: ["pdf", "xlsx", "csv"]
  enable_ocr: true
  extract_images: true
  
retrieval:
  embedding_model: "all-MiniLM-L6-v2"
  vector_store: "faiss"
  top_k: 5
  similarity_threshold: 0.7

generation:
  llm_provider: "groq"
  model: "llama-3.1-8b-instant"
  temperature: 0.1
  max_tokens: 1000
  enable_streaming: true

features:
  follow_up_questions: true
  confidence_scoring: true
  pdf_export: true
  semantic_highlighting: true
  analytics_dashboard: true
  summary_cards: true

evaluation:
  metrics: ["recall@5", "exact_match", "f1_score", "citation_accuracy"]
  test_set_size: 50
"""

if __name__ == "__main__":
    # Create project structure
    project = ProjectStructure()
    project.create_structure()
    project.create_init_files()
    project.create_config_files()
    print("Complete project structure initialized!")
