from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from rag_faq_api import rag_answer

app = FastAPI(title="FinAI FAQ Bot Frontend")

# ---- FRONTEND HOME ----
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>FinAI FAQ Bot</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            input[type=text] { width: 400px; padding: 10px; font-size: 16px; }
            button { padding: 10px 20px; font-size: 16px; }
            .box { padding:20px; border:1px solid #ddd; margin-top:20px; border-radius:10px; }
        </style>
    </head>
    <body>
        <h2>FinAI FAQ Bot</h2>
        <form action="/ask_frontend" method="post">
            <input type="text" name="question" placeholder="Ask something..." required>
            <button type="submit">Ask</button>
        </form>
    </body>
    </html>
    """

# ---- FRONTEND ANSWER ----
@app.post("/ask_frontend", response_class=HTMLResponse)
def ask_frontend(question: str = Form(...)):
    result = rag_answer(question)

    return f"""
    <html>
    <head>
        <title>FinAI FAQ Bot</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            .box {{ padding:20px; border:1px solid #ddd; margin-top:20px; border-radius:10px; }}
        </style>
    </head>
    <body>
        <h2>FinAI FAQ Bot</h2>

        <div class="box">
            <h3>Your Question:</h3>
            <p>{question}</p>

            <h3>Answer:</h3>
            <p>{result.answer_text}</p>

            <h4>Confidence:</h4>
            <p>{result.confidence} ({result.confidence_label})</p>

            <h4>Escalated to Human:</h4>
            <p>{result.escalated_to_human}</p>

            <a href="/">Ask another</a>
        </div>
    </body>
    </html>
    """
