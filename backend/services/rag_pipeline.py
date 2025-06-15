from config import settings
from langchain.schema import Document
import logging
from services.providers import get_llm_provider
from typing import List, Dict

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = get_llm_provider()

    def generate_answer(self, question, chat_history=None):
        ret_docs_scores = self.vector_store.similarity_search(question)
        sources = [
            {
                "content": doc.page_content,
                "page": doc.metadata.get("page", "-"),
                "score": float(score),
                "metadata": doc.metadata,
            }
            for (doc,score) in ret_docs_scores
        ]
        context = self._generate_context(docs)
        answer = self._generate_llm_response(question, context, chat_history)
        return {"answer": answer, "sources": sources}

    def _generate_context(self, documents: List[Document]) -> str:
        context = "\n".join([doc.page_content for doc in documents])
        return context

    def _generate_llm_response(self, question: str, context: str, chat_history: List[Dict[str, str]] = None) -> str:
        if not context or len(context.strip()) == 0:
            return "Sorry, no context found."
        try:
            if settings.llm_provider.lower() == "openai":
                prompt = prompt = f"""
                                    Given the context below, answer the following question.
                                    If the answer is not explicitly stated, but enough data (such as numbers as the component of the formula that is mentioned)
                                    is available in the context, please attempt to calculate the answer using the data provided.
                                    If you perform a calculation, show your working briefly.

                                    Context:
                                    {context}

                                    Question:
                                    {question}
                                    """
                return self.llm.predict(prompt)
            elif settings.llm_provider.lower() == "huggingface":
                # crop context for QA pipeline
                res = self.llm(question=question, context=context[:1000])
                return res.get("answer", "")
            else:
                return "[ERR: No llm_provider configured]"
        except Exception as e:
            return f"LLM error: {e}"