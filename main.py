import os
import pathlib
import re
from contextlib import asynccontextmanager

import anthropic
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from dotenv import load_dotenv


"""
Simple Azure OpenAI / Anthropic Agent — HTTP Server

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

Provider toggle
---------------
Set INFERENCE_PROVIDER=azure (default) or INFERENCE_PROVIDER=anthropic in .env.
"""

load_dotenv()

INFERENCE_PROVIDER = os.environ.get("INFERENCE_PROVIDER", "azure").lower()

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
AZURE_OPENAI_API_VERSION = os.environ.get(
    "AZURE_OPENAI_API_VERSION", "2025-04-01-preview"
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

AGENT_PORT = int(os.environ.get("AGENT_PORT", "8000"))


# ── Knowledge base ─────────────────────────────────────────────────────────────

_KB_PATH = pathlib.Path(__file__).parent / "knowledge_base.txt"
_KB_TEXT = _KB_PATH.read_text(encoding="utf-8") if _KB_PATH.exists() else ""


def _parse_kb_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (header, body) tuples split on [Header] markers."""
    parts = re.split(r"(\[[^\]]+\])", text)
    sections = []
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip("[]").lower()
        body = parts[i + 1].strip()
        sections.append((header, body))
    return sections


_KB_SECTIONS = _parse_kb_sections(_KB_TEXT)


def search_knowledge_base(query: str) -> str:
    """Return the top matching KB sections for a query (keyword overlap score)."""
    if not _KB_SECTIONS:
        return "Knowledge base is empty."
    query_words = set(query.lower().split())
    scored = []
    for header, body in _KB_SECTIONS:
        section_words = set((header + " " + body).lower().split())
        score = len(query_words & section_words)
        scored.append((score, header, body))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for s in scored if s[0] > 0][:3] or scored[:1]
    return "\n\n".join(f"[{h.title()}]\n{b}" for _, h, b in top)


## Takes default sytem prompt is nothing is provded in UI.
DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant.
""".strip()


# ── Agent ─────────────────────────────────────────────────────────────────────


class SimpleAgent:
    """Lightweight wrapper around Azure OpenAI or Anthropic chat completions.

    Internally stores messages in OpenAI format (list of role/content dicts).
    When using Anthropic, the system message is extracted and passed separately.
    """

    def __init__(self, name: str, instructions: str):
        self.name = name
        self.system_prompt = instructions
        self.messages: list[ChatCompletionMessageParam] = []

        if INFERENCE_PROVIDER == "anthropic":
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            self._client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
            )

    def run(self, user_message: str) -> str:
        """Send a message and return the assistant's reply."""
        self.messages.append({"role": "user", "content": user_message})

        if INFERENCE_PROVIDER == "anthropic":
            reply = self._run_anthropic()
        else:
            reply = self._run_azure()

        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _run_azure(self) -> str:
        full_messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.system_prompt},
            *self.messages,
        ]
        response = self._client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=full_messages,
        )
        return response.choices[0].message.content or ""

    def _run_anthropic(self) -> str:
        response = self._client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=8096,
            system=self.system_prompt,
            messages=self.messages,
        )
        return response.content[0].text


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
    print(f"Inference provider: {INFERENCE_PROVIDER}")
    yield


app = FastAPI(title="SimpleAgent Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


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


@app.get("/")
def ui():
    return FileResponse(pathlib.Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "provider": INFERENCE_PROVIDER}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=AGENT_PORT, reload=True)
