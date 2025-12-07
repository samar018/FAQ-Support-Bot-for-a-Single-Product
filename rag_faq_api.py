# rag_faq_api.py
import os
import uuid
import string
from pathlib import Path
from typing import List, Optional
import requests  # For dummy escalation POST

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv



# Load .env
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ESCALATION_ENDPOINT = os.getenv("ESCALATION_ENDPOINT", "http://localhost:8000/escalate")  # dummy endpoint

if not GITHUB_TOKEN:
    raise ValueError("❌ GITHUB_TOKEN missing in .env")



# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)


# -------------------------------
# Environment & LLM Setup
# -------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN missing in .env file")

PDF_PATH = Path(r"E:\workspace\AI job\FAQ single product RAG-LANGCHAIN-embedding\FinAI Comprehensive FAQ.pdf")
ESCALATION_ENDPOINT = "http://localhost:8000/escalate"  # Dummy endpoint

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=GITHUB_TOKEN
)
MODEL_NAME = "openai/gpt-4.1-nano"


# -------------------------------
# Load PDF & Create Vector DB
# -------------------------------
if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH.resolve()}")

loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()

all_chunks = []
for i, page in enumerate(pages, start=1):
    txt = page.page_content.strip()
    meta = {"page": i, "source": str(PDF_PATH.resolve())}
    # Assign category by page ranges
    if 1 <= i <= 20:
        category = "Platform Basics & Getting Started"
    elif 21 <= i <= 40:
        category = "Budgeting & Spending Analysis"
    elif 41 <= i <= 60:
        category = "Data Integrity & Security"
    elif 61 <= i <= 80:
        category = "Investment & Financial Planning"
    else:
        category = "Security, Compliance & Support"
    all_chunks.append(Document(page_content=txt, metadata={**meta, "category": category}))

vector_store = FAISS.from_documents(all_chunks, embedding_model)

# -------------------------------
# Pydantic Models
# -------------------------------
class SourceReference(BaseModel):
    page: int
    category: str
    source: str

class FAQResponse(BaseModel):
    answer_text: str
    confidence: float
    confidence_label: str
    source_reference: Optional[List[SourceReference]]
    escalated_to_human: bool
    escalation_request_id: Optional[str]

# -------------------------------
# Confidence Label Helper
# -------------------------------
def compute_confidence_label(score: float) -> str:
    if score >= 0.85:
        return "high"
    elif score >= 0.6:
        return "medium"
    else:
        return "low"

# -------------------------------
# RAG Answer with Escalation
# -------------------------------
CONFIDENCE_THRESHOLD = 0.6  # Below this triggers escalation

def rag_answer(question: str) -> FAQResponse:
    # Step 1: Retrieve top document
    results = vector_store.similarity_search_with_score(question, k=1)  # only top doc
    if not results:
        return FAQResponse(
            answer_text="I don't know",
            confidence=0.0,
            confidence_label="low",
            source_reference=None,
            escalated_to_human=True,
            escalation_request_id=str(uuid.uuid4())
        )

    doc, top_score = results[0]
    context = doc.page_content

    # Step 2: Prompt LLM
    system_prompt = (
        "Answer ONLY using the context. If answer not found, respond with 'I don't know'.\n\n"
        f"Context:\n{context}"
    )

    llm_resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        top_p=1.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    answer = llm_resp.choices[0].message.content.strip()

    # Step 3: Decide if escalation needed
    normalized_answer = answer.lower().strip().translate(str.maketrans('', '', string.punctuation))
    escalate = top_score < CONFIDENCE_THRESHOLD or normalized_answer in ["i dont know", "দুঃখিত এই বিষয়ে আমার জানা নেই"]

    source_refs = None if escalate else [
        SourceReference(
            page=doc.metadata["page"],
            category=doc.metadata["category"],
            source=doc.metadata["source"]
        )
    ]

    # Step 4: Call dummy escalation endpoint if needed
    escalation_id = str(uuid.uuid4()) if escalate else None
    if escalate:
        payload = {
            "question": question,
            "retrieved_snippets": context,
            "attempted_answer": answer,
            "confidence": float(top_score),
            "confidence_label": compute_confidence_label(top_score),
            "escalation_request_id": escalation_id
        }
        try:
            requests.post(ESCALATION_ENDPOINT, json=payload)
        except Exception as e:
            print(f"⚠️ Failed to call escalation endpoint: {e}")

    # Step 5: Return structured response
    return FAQResponse(
        answer_text=answer,
        confidence=top_score,
        confidence_label=compute_confidence_label(top_score),
        source_reference=source_refs,
        escalated_to_human=escalate,
        escalation_request_id=escalation_id
    )
