from fastapi import FastAPI
from pydantic import BaseModel

from graph import AnswerResponse, ask_question


app = FastAPI(
    title="Zepto Support Assistant",
    description="Offline RAG-based Zepto policy support assistant",
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant is running"
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest):
    return ask_question(request.query)