from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.agent import handle_chat

app = FastAPI(title="SHL Assessment Recommender", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]
    result = handle_chat(messages)
    return ChatResponse(**result)
