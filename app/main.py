"""
Main FastAPI application for the question-answering service.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from app.rag_qa import RAGQASystem
from app.extractor import DataExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Member Data QA Service",
    description="A question-answering system for member data",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
logger.info("Initializing data extractor and QA system...")
extractor = DataExtractor()

# Initialize QA system: loads messages on-demand based on questions (more efficient)
logger.info("Initializing QA system (messages loaded on-demand per question)...")
qa_system = RAGQASystem(extractor, use_embeddings=True, use_slm=True)

logger.info("✓ System ready: Messages will be loaded and embedded on-demand for each question")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Member Data QA Service"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/status")
async def status():
    """Get system status including message count and embedding status."""
    return {
        "status": "ready",
        "mode": "on-demand",  # Messages loaded on-demand per question
        "embeddings_ready": qa_system.embedding_model is not None,
        "slm_ready": qa_system.slm_pipeline is not None,
        "extractor_cache": len(extractor._cache) if extractor._cache else 0
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer a natural-language question about member data.
    
    The system retrieves top-k relevant messages using semantic embeddings,
    then generates an answer using a Small Language Model (SLM).
    
    Example questions:
    - "When is Layla planning her trip to London?"
    - "How many cars does Vikram Desai have?"
    - "What are Amira's favorite restaurants?"
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Received question: {request.question}")
        
        # Get answer from QA system
        answer, _ = qa_system.answer(request.question)
        
        logger.info(f"Generated answer: {answer}")
        
        return AnswerResponse(answer=answer)
    
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/refresh")
async def refresh_data():
    """
    Manually refresh messages and recompute embeddings.
    Useful when you want to force a refresh of the data cache.
    """
    try:
        logger.info("Manual refresh requested")
        qa_system._load_messages(force_refresh=True)
        return {
            "status": "success",
            "message": f"Refreshed {len(qa_system.messages)} messages and recomputed embeddings"
        }
    except Exception as e:
        logger.error(f"Error refreshing data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error refreshing data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

