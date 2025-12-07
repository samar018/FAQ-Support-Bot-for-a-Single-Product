# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from rag_faq_api import rag_answer, SourceReference, FAQResponse  # import Pydantic models

# Request model
class FAQRequest(BaseModel):
    question: str

# FastAPI app
app = FastAPI(title="FinAI RAG FAQ API")

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/ask", response_model=FAQResponse)
def ask_faq(req: FAQRequest):
    """
    Receives a question from the user, calls rag_answer,
    and returns the structured FAQ response including confidence,
    source references, and escalation info.
    """
    return rag_answer(req.question)
