import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from dotenv import load_dotenv

"""
Simple Azure OpenAI Agent — HTTP Server

Exposes the agent as a local FastAPI endpoint so that other services
(e.g. the Arena evaluator) can interact with it over HTTP.

Endpoints
---------
POST /chat   — stateless: send {"message": "...", "system_prompt": "..."}
                and receive {"reply": "..."}.
                Each request is independent (no conversation memory).

Run
---
    python main.py                  # starts on http://localhost:8000
    AGENT_PORT=9000 python main.py  # custom port
"""

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
AZURE_OPENAI_API_VERSION = os.environ.get(
    "AZURE_OPENAI_API_VERSION", "2025-04-01-preview"
)
AGENT_PORT = int(os.environ.get("AGENT_PORT", "8000"))

DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant.
""".strip()


# ── Agent ─────────────────────────────────────────────────────────────────────


class SimpleAgent:
    """Lightweight wrapper around Azure OpenAI chat completions."""

    def __init__(self, name: str, instructions: str):
        self.name = name
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        self.messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": instructions}]

    def run(self, user_message: str) -> str:
        """Send a message and return the assistant's reply."""
        self.messages.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=self.messages,
        )
        reply = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# ── Request / Response models ─────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    system_prompt: str | None = None  # optional override; falls back to default


class ChatResponse(BaseModel):
    reply: str


# ── FastAPI app ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Agent server listening on http://localhost:{AGENT_PORT}")
    yield


app = FastAPI(title="SimpleAgent Server", lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Stateless chat endpoint.

    Each request creates a fresh agent so conversations don't leak context
    between callers.  The caller can optionally supply a custom system_prompt;
    otherwise the server's DEFAULT_SYSTEM_PROMPT is used.
    """
    instructions = req.system_prompt or DEFAULT_SYSTEM_PROMPT
    agent = SimpleAgent(name="AgentEndpoint", instructions=instructions)
    reply = agent.run(req.message)
    return ChatResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=AGENT_PORT, reload=True)
