from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Escalation Server")

class EscalationMessage(BaseModel):
    question: str
    retrieved_docs: List[str]
    answer: str
    confidence: float

# Store escalated messages in memory
escalated_messages = []

@app.post("/escalate")
def escalate(msg: EscalationMessage):
    escalated_messages.append(msg)
    return {"status": "received", "total": len(escalated_messages)}

@app.get("/messages")
def get_messages():
    return {"escalated_messages": escalated_messages}
