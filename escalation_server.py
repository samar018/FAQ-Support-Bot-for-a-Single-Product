# escalation_server.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Escalation Server")

# Allow frontend to fetch escalations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store escalated messages in memory
escalated_messages = []

@app.post("/escalate")
async def receive_escalation(req: Request):
    data = await req.json()
    escalated_messages.append(data)
    print("⚠️ Escalation received:", data)
    return {"status": "received"}

@app.get("/escalations")
async def get_escalations():
    return {"escalated_messages": escalated_messages}
