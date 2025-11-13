"""
Main FastAPI application for the question-answering service.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

from app.rag_qa import RAGQASystem
from app.extractor import DataExtractor
from app.insights import DataInsights

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
extractor = DataExtractor()
# Initialize QA system with SLM for answer generation
qa_system = RAGQASystem(extractor, use_embeddings=True, use_slm=True)
insights_analyzer = DataInsights(extractor)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    confidence: float


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Member Data QA Service"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer a natural-language question about member data.
    
    Example questions:
    - "When is Layla planning her trip to London?"
    - "How many cars does Vikram Desai have?"
    - "What are Amira's favorite restaurants?"
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Received question: {request.question}")
        
        # Get answer and confidence from QA system
        answer, confidence = qa_system.answer(request.question)
        
        logger.info(f"Generated answer: {answer} (confidence: {confidence:.3f})")
        
        return AnswerResponse(answer=answer, confidence=confidence)
    
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/insights")
async def get_insights():
    """
    Get data insights and anomalies analysis.
    This endpoint analyzes the member data for quality issues and inconsistencies.
    """
    try:
        insights = insights_analyzer.analyze()
        return insights
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

