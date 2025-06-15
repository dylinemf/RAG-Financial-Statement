from config import settings
import logging

logger = logging.getLogger(__name__)

# HuggingFace Imports (based on selection)
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    from transformers import pipeline
except ImportError:
    HuggingFaceEmbeddings = None

# OpenAI Imports (based on selection)
try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.chat_models import ChatOpenAI
except ImportError:
    OpenAIEmbeddings = None

def get_embedding_provider():
    if settings.embedding_provider.lower() == "openai":
        if not OpenAIEmbeddings:
            raise ImportError("langchain OpenAIEmbeddings not installed")
        return OpenAIEmbeddings(api_key=settings.openai_api_key, model=settings.embedding_model)
    elif settings.embedding_provider.lower() == "huggingface":
        if not HuggingFaceEmbeddings:
            raise ImportError("langchain HuggingFaceEmbeddings not installed")
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    else:
        raise ValueError(f"Unknown embedding_provider: {settings.embedding_provider}")

def get_llm_provider():    
    if settings.llm_provider.lower() == "openai":
        return ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens
        )
    elif settings.llm_provider.lower() == "huggingface":
        # Can swap for any HF model, this is general
        return pipeline("question-answering", model=settings.llm_model)
    else:
        raise ValueError(f"Unknown llm_provider: {settings.llm_provider}")