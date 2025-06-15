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
        try:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
            logger.info(f"Vector Store initialized at {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to init vector store: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        if not documents:
            logger.warning("No documents provided to add to vector store.")
            return
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()
        logger.info(f"Vectorstore now contains {self.get_document_count()} chunks after add.")

    def similarity_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if k is None:
            k = settings.retrieval_k
        try:
            import time
            t0 = time.time()
            results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
            logger.info(f"Similarity search for '{query[:50]}...' returned {len(results)} chunks in {time.time()-t0:.3f}s")

            filtered = []
            for doc, score in results:
                if score >= settings.similarity_threshold:
                    filtered.append((doc, score))
            if not filtered:
                logger.warning(f"No chunk above similarity threshold {settings.similarity_threshold}")
            return filtered if filtered else results # fallback to raw results if none above threshold
        except Exception as e:
            logger.error(f"Vector similarity search error: {e}")
            raise RuntimeError("Vector store search error, possibly not indexed yet.")

    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from vector store"""
        self.vectorstore.delete(document_ids)
        self.vectorstore.persist()
        logger.info(f"Deleted {len(document_ids)} from vector store.")

    def get_document_count(self) -> int:
        """Get total number of documents in vector store"""
        try:
            return len(self.vectorstore.get()["ids"])
        except Exception:
            return 0