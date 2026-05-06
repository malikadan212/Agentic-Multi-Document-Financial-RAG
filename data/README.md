# Data Directory

## Overview

This directory contains the raw financial documents used for the RAG system.

## Structure

```
data/
└── datasetsforchatbot/
    ├── HBL_*.pdf          # Habib Bank Limited documents
    ├── Meezan_*.pdf       # Meezan Bank documents
    └── UBL_*.pdf          # United Bank Limited documents
```

## Documents Included

### HBL (Habib Bank Limited) - 13 documents
- Credit Card applications and terms
- Personal loan documents
- Car loan documents
- Debit card terms
- Account opening forms
- Glossary of terms

### Meezan Bank - 7 documents
- Account opening processes
- Car loan information
- Home loan information
- Premium account details
- Current account information

### UBL (United Bank Limited) - 7 documents
- Credit card FAQs and terms
- Personal loan information
- Debit card information
- Account opening processes
- NRP forms

## Total Dataset

- **27 PDF documents**
- **Banking domain**: Credit cards, loans, accounts
- **Languages**: English
- **Time period**: 2024-2025

## Pre-processed Data

The raw PDFs are processed into:
- **27,283 text chunks** (stored in `chunk_metadata/chunk_metadata.json`)
- **FAISS index** (stored in `chunk_metadata/rag_index.faiss`)

## Note on Raw Files

Raw PDF files are **not included in the Git repository** due to:
1. File size constraints
2. Privacy considerations
3. Copyright restrictions

The pre-processed data in `chunk_metadata/` is sufficient to run the system.

## Using Your Own Documents

To use your own documents:

1. Place PDF files in `data/datasetsforchatbot/`
2. Run the Streamlit app: `streamlit run src/streamlit_app/app.py`
3. Select "Upload Documents" mode
4. Upload your PDFs through the UI
5. Click "Process Documents"

The system will:
- Extract text from PDFs
- Apply OCR if needed
- Extract tables and images
- Create semantic chunks
- Build FAISS index
- Enable querying

## Supported Formats

- PDF (with OCR support)
- Excel (.xlsx)
- CSV (.csv)

## Document Processing Features

- **Text extraction** with PyMuPDF
- **OCR** with Tesseract for scanned documents
- **Table extraction** with pdfplumber
- **Image extraction** for multimodal analysis
- **Temporal entity extraction** from filenames and content
- **Semantic chunking** with configurable size and overlap

## Pre-loaded System

To use the pre-loaded system without uploading documents:

1. Run: `streamlit run src/streamlit_app/app.py`
2. Select "Use Pre-loaded Data" in sidebar
3. Start querying immediately

The pre-loaded system includes all 27,283 chunks from the banking documents.
