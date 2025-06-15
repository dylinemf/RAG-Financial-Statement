from config import settings
from langchain.schema import Document
import logging
from services.providers import get_llm_provider
from services.vector_store import VectorStoreService
from typing import List, Dict, Any
import time

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = get_llm_provider()

    def generate_answer(self, question, chat_history=None):
        """Generate answer using RAG pipeline"""
        start_time = time.time()
        sources = []
        try:
            ret_docs_scores = self.vector_store.similarity_search(question)
            if not ret_docs_scores:
                logger.warning("No relevant document chunk found; returning fallback message.")
                return {"answer": "Sorry, I could not find any relevant content in the documents.", "sources": [], "processing_time": 0.0}

            docs = [doc for doc, score in ret_docs_scores]
            for (doc, score) in ret_docs_scores:
                sources.append(
                    {
                        "content": doc.page_content,
                        "page": doc.metadata.get("page", "-"),
                        "score": float(score),
                        "metadata": doc.metadata,
                    }
                )
            context = self._generate_context(docs)
            answer = self._generate_llm_response(question, context, chat_history)
            proc_time = time.time() - start_time
            logger.info(f"Answered Q after vector search + LLM in {proc_time:.2f}s.")
            return {
                "answer": answer,
                "sources": sources,
                "processing_time": proc_time,
            }
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return {
                "answer": f"System error: {e}",
                "sources": [],
                "processing_time": 0.0,
            }

    def _generate_context(self, documents: List[Document]) -> str:
        """Generate context from retrieved documents"""
        # Optionally truncate context for LLM's input size performance
        context = "\n".join([doc.page_content for doc in documents])
        max_context_len = settings.max_tokens * 5  # approx char per token
        if len(context) > max_context_len:
            context = context[:max_context_len]
        return context

    def _generate_llm_response(self, question: str, context: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Generate response using LLM"""
        if not context or len(context.strip()) == 0:
            return "Sorry, no context found."
        try:
            if settings.llm_provider.lower() == "openai":
                # Modular prompt for financial statement RAG
                prompt = f"""
                            Given the context below, answer the following question.
                            If the answer is not explicitly stated, but enough data (such as numbers as the component of the formula that is mentioned)
                            is available in the context, please attempt to calculate the answer using the data provided.
                            If you perform a calculation, show your working briefly and with right format (latex but with '$$ ... $$' as delimiter or markdown).
                            Please provide the final answer at the beginning with the bold format and put it at the top alone (with newspace or enter for the following explanations).
                            Apply markdown to get the response to be neat.

                            Context:
                            {context}

                            Question:
                            {question}
                            """
                resp = self.llm.predict(prompt)
                return resp
            elif settings.llm_provider.lower() == "huggingface":
                # crop context for QA pipeline
                res = self.llm(question=question, context=context[:2000])
                return res.get("answer", "")
            else:
                return "[ERR: No llm_provider configured]"
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"LLM error: {e}"