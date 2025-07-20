"""FastAPI service. One MCP session and one compiled graph are shared per process."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from recruiting_agent.graph import default_model, open_agent


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[dict]
    guard_events: list[str]
    blocked: bool
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with open_agent() as agent:
        app.state.agent = agent
        yield


app = FastAPI(title="MCP Recruiting Agent", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": default_model()._llm_type}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    result = await app.state.agent.chat(request.thread_id, request.message)
    return ChatResponse(
        answer=result.answer,
        tool_calls=result.tool_calls,
        guard_events=result.guard_events,
        blocked=result.blocked,
        latency_ms=round(result.latency_ms, 1),
    )
