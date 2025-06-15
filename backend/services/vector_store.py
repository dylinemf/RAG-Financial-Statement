from config import settings
from langchain.schema import Document
from langchain.vectorstores import Chroma
import logging
from services.providers import get_embedding_provider
from typing import List, Tuple

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        self.persist_dir = settings.vector_db_path
        self.embeddings = get_embedding_provider()
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()
    
    def similarity_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if k is None:
            k = settings.retrieval_k
        return self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from vector store"""
        self.vectorstore.delete(document_ids)
        self.vectorstore.persist()
    
    def get_document_count(self) -> int:
        """Get total number of documents in vector store"""
        return len(self.vectorstore.get()["ids"])