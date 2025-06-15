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
logging.basicConfig(level=settings.log_level)
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
# TODO: Initialize your services here
pdf_processor = PDFProcessor()
vector_store = VectorStoreService()
rag_pipeline = RAGPipeline(vector_store)

def process_pdf_async(path):
    docs = pdf_processor.process_pdf(path)
    vector_store.add_documents(docs)


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
    filename = f"{int(time.time())}_{file.filename}"
    save_path = os.path.join(settings.pdf_upload_path, filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    # Process file in background
    background_tasks.add_task(process_pdf_async, save_path)
    return {"message": "PDF is received, processing in backend"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process chat request and return AI response"""
    res = rag_pipeline.generate_answer(request.question)
    return res


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
    """Get document chunks (optional endpoint)"""
    result = vector_store.vectorstore.get()
    
    # result["documents"] usually is a list page_content
    # result["ids"] -> id chunk, result["metadatas"] -> meta info per chunk
    chunks = []
    for idx, content in enumerate(result["documents"]):
        chunks.append({
            "id": result["ids"][idx],
            "content": content,
            "page": result["metadatas"][idx].get("page", "-"),
            "metadata": result["metadatas"][idx],
        })
    return {"chunks": chunks, "total_count": len(chunks)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, reload=settings.debug) 