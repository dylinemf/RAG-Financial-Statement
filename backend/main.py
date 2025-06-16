from config import settings
from datetime import datetime
from fastapi import BackgroundTasks
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import glob
import logging
from models.schemas import ChatRequest
import os
from services.pdf_processor import PDFProcessor
from services.vector_store import VectorStoreService
from services.rag_pipeline import RAGPipeline
import time

# Configure logging
logging.basicConfig(
    level=settings.log_level, 
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG-based Financial Statement Q&A System",
    description="AI-powered Q&A system for financial documents using RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pdf_processor = PDFProcessor()
vector_store = VectorStoreService()
rag_pipeline = RAGPipeline(vector_store)

def process_pdf_async(path):
    docs = pdf_processor.process_pdf(path)
    try:
        vector_store.target_chunk_count = len(docs)
    except Exception:
        pass
    vector_store.add_documents(docs, batch_size=32)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting RAG Q&A System...")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG-based Financial Statement Q&A System is running"}


@app.post("/api/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process PDF file"""
    try:
        filename = f"{int(time.time())}_{file.filename}"
        save_path = os.path.join(settings.pdf_upload_path, filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())
        # Process file in background
        background_tasks.add_task(process_pdf_async, save_path)
        return {"message": "PDF is received, processing in backend"}
    except Exception as e:
        logger.error(f"Upload endpoint error: {e}")
        return {"answer": "Internal error", "sources": [], "processing_time": 0.0}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process chat request and return AI response"""
    try:
        res = rag_pipeline.generate_answer(request.question)
        return res
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return {"answer": "Internal error", "sources": [], "processing_time": 0.0}


@app.get("/api/documents")
async def get_documents():
    """Get list of processed documents"""
    files = glob.glob(os.path.join(settings.pdf_upload_path, "*.pdf"))
    documents = []
    for f in files:
        filename = os.path.basename(f)
        upload_time = datetime.fromtimestamp(os.path.getmtime(f))
        # need to get the chunk count (like total chunk from vectorstore)
        try:
            chunk_count = vector_store.get_document_count()
        except:
            chunk_count = 0
        documents.append({
            "filename": filename,
            "upload_date": upload_time,
            "chunks_count": chunk_count,
            "status": "processed"
        })
    return {"documents": documents}


@app.get("/api/chunks")
async def get_chunks():
    result = vector_store.vectorstore.get()
    chunks = []
    for idx, content in enumerate(result["documents"]):
        chunks.append({
            "id": result["ids"][idx],
            "content": content,
            "page": result["metadatas"][idx].get("page", "-"),
            "metadata": result["metadatas"][idx],
        })
    # for numeric progress of pdf processing
    total_target_chunks = getattr(vector_store, "target_chunk_count", None)
    return {
        "chunks": chunks,
        "total_count": len(chunks),
        "total_target_count": total_target_chunks
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, reload=settings.debug) 