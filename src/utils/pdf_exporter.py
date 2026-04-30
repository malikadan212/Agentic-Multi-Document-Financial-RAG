# src/utils/pdf_exporter.py
"""
PDF Report Exporter for RAG System
Generates professional PDF reports from query results
"""

from fpdf import FPDF
from datetime import datetime
from typing import List, Dict, Any
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGReportPDF(FPDF):
    """Custom PDF class with header and footer"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        """Add header to each page"""
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(0, 102, 204)  # Blue
        self.cell(0, 10, 'Financial RAG Analysis Report', align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(0, 102, 204)
        self.line(10, 20, 200, 20)
        self.ln(5)
    
    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C')


class RAGReportExporter:
    """
    Exports RAG query results to PDF format
    """
    
    def __init__(self):
        self.pdf = None
    
    def export_query_result(
        self,
        query: str,
        answer: str,
        citations: List[Dict],
        retrieved_sources: List[Any],
        metrics: Dict[str, Any]
    ) -> bytes:
        """
        Export a single query result to PDF
        
        Args:
            query: The user's question
            answer: The generated answer
            citations: List of citation dictionaries
            retrieved_sources: List of retrieved source chunks
            metrics: Dictionary with response_time, tokens, cost, confidence
            
        Returns:
            PDF as bytes
        """
        self.pdf = RAGReportPDF()
        self.pdf.add_page()
        
        # Title section
        self.pdf.set_font('Helvetica', 'B', 16)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(0, 10, 'Query Analysis Report', new_x='LMARGIN', new_y='NEXT')
        self.pdf.ln(5)
        
        # Query section
        self._add_section_header('Question')
        self.pdf.set_font('Helvetica', '', 11)
        self.pdf.set_text_color(51, 51, 51)
        self.pdf.multi_cell(0, 6, query)
        self.pdf.ln(5)
        
        # Answer section
        self._add_section_header('Answer')
        self.pdf.set_font('Helvetica', '', 11)
        self.pdf.set_text_color(51, 51, 51)
        # Clean answer text for PDF (remove any HTML tags)
        clean_answer = self._clean_text(answer)
        self.pdf.multi_cell(0, 6, clean_answer)
        self.pdf.ln(5)
        
        # Metrics section
        self._add_section_header('Performance Metrics')
        self.pdf.set_font('Helvetica', '', 10)
        metrics_text = f"""
Response Time: {metrics.get('response_time', 'N/A')}s
Tokens Used: {metrics.get('tokens', 'N/A')}
Cost: ${metrics.get('cost', 0):.4f}
Confidence: {metrics.get('confidence', 'N/A')}%
Citations: {len(citations)}
Sources Retrieved: {len(retrieved_sources)}
        """.strip()
        self.pdf.multi_cell(0, 5, metrics_text)
        self.pdf.ln(5)
        
        # Citations section
        if citations:
            self._add_section_header(f'Citations ({len(citations)})')
            self.pdf.set_font('Helvetica', '', 10)
            for i, citation in enumerate(citations, 1):
                doc_name = citation.get('doc_name', 'Unknown')
                page = citation.get('page', 'N/A')
                valid = "Valid" if citation.get('valid', False) else "Unverified"
                self.pdf.set_text_color(0, 102, 204)
                self.pdf.cell(0, 5, f"{i}. {doc_name}, Page {page} [{valid}]", new_x='LMARGIN', new_y='NEXT')
            self.pdf.set_text_color(51, 51, 51)
            self.pdf.ln(5)
        
        # Sources section
        if retrieved_sources:
            self._add_section_header(f'Retrieved Sources ({len(retrieved_sources)})')
            for i, source in enumerate(retrieved_sources[:5], 1):  # Limit to 5 sources
                self.pdf.set_font('Helvetica', 'B', 10)
                doc_name = getattr(source, 'metadata', {}).get('doc_name', 'Unknown') if hasattr(source, 'metadata') else source.get('metadata', {}).get('doc_name', 'Unknown')
                page = getattr(source, 'metadata', {}).get('page', '?') if hasattr(source, 'metadata') else source.get('metadata', {}).get('page', '?')
                score = getattr(source, 'score', 0) if hasattr(source, 'score') else source.get('score', 0)
                
                self.pdf.cell(0, 5, f"Source {i}: {doc_name} (Page {page}) - Score: {score:.3f}", new_x='LMARGIN', new_y='NEXT')
                
                self.pdf.set_font('Helvetica', '', 9)
                content = getattr(source, 'content', '') if hasattr(source, 'content') else source.get('content', '')
                self.pdf.multi_cell(0, 4, self._clean_text(content[:300]) + "...")
                self.pdf.ln(3)
        
        # Return PDF as bytes
        return bytes(self.pdf.output())
    
    def _add_section_header(self, title: str):
        """Add a styled section header"""
        self.pdf.set_font('Helvetica', 'B', 12)
        self.pdf.set_text_color(0, 102, 204)
        self.pdf.cell(0, 8, title, new_x='LMARGIN', new_y='NEXT')
        self.pdf.set_draw_color(200, 200, 200)
        self.pdf.line(10, self.pdf.get_y(), 200, self.pdf.get_y())
        self.pdf.ln(3)
        self.pdf.set_text_color(51, 51, 51)
    
    def _clean_text(self, text: str) -> str:
        """Clean text for PDF output"""
        if not text:
            return ""
        # Remove HTML tags
        import re
        text = re.sub(r'<[^>]+>', '', text)
        # Replace special characters that might cause issues
        text = text.replace('\u2019', "'").replace('\u2018', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2014', '-').replace('\u2013', '-')
        # Encode to latin-1 compatible text (FPDF default)
        try:
            text = text.encode('latin-1', errors='replace').decode('latin-1')
        except:
            pass
        return text


# Example usage
if __name__ == "__main__":
    exporter = RAGReportExporter()
    
    # Test data
    pdf_bytes = exporter.export_query_result(
        query="What was Apple's revenue in Q3 2023?",
        answer="Apple's revenue in Q3 2023 was $81.8 billion [Source: Apple_10Q, Page 3].",
        citations=[{"doc_name": "Apple_10Q", "page": 3, "valid": True}],
        retrieved_sources=[],
        metrics={"response_time": 2.5, "tokens": 500, "cost": 0.0015, "confidence": 85}
    )
    
    # Save test PDF
    with open("test_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("Test PDF exported successfully!")
