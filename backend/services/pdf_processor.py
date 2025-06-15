from config import settings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import os
import pdfplumber
import PyPDF2
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    
    def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF and return page-wise content"""
        pages_content = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages_content.append({"page": i+1, "content": text})
        return pages_content
    
    def split_into_chunks(self, pages_content: List[Dict[str, Any]]) -> List[Document]:
        """Split page content into chunks"""
        docs = []
        for page in pages_content:
            chunks = self.splitter.split_text(page["content"])
            for c in chunks:
                docs.append(
                    Document(
                        page_content=c,
                        metadata={"page": page["page"]}
                    )
                )
        return docs
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF file and return list of Document objects"""
        pages = self.extract_text_from_pdf(file_path)
        return self.split_into_chunks(pages)