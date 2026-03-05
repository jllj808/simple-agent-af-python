# Simple Agent (Python)

A stateless conversational agent exposed as an HTTP API, with a built-in browser UI. Supports Azure OpenAI and Anthropic as inference providers. No Azure CLI, managed identity, or Agent Framework required.

## Setup

### 1. Clone the repository

### 2. Create a `.env` file inside `simple-agent-af-python/`

**Azure OpenAI** (default provider):

```
INFERENCE_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
```

**Anthropic:**

```
INFERENCE_PROVIDER=anthropic
ANTHROPIC_API_KEY=<your-api-key>
```

Optional overrides (defaults shown):

```
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2025-04-01-preview
ANTHROPIC_MODEL=claude-sonnet-4-6
AGENT_PORT=8000
```

### 3. Install dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Running

### Option A — start script (recommended)

From the repo root:

```bash
./start.sh
```

This starts the FastAPI backend and opens the UI at `http://localhost:8000` automatically.

### Option B — manual

```bash
# from simple-agent-af-python/
python main.py
# then open http://localhost:8000 in your browser
```

## UI

Open `http://localhost:8000` in your browser. The page has two sections:

- **POST /chat** — enter an optional system prompt and a user message, click Submit, see the reply.
- **GET /health** — click Check Health to verify the server and active provider.

## API

| Method | Path      | Description                     |
|--------|-----------|---------------------------------|
| GET    | `/`       | Serves the HTML UI              |
| GET    | `/health` | Health check                    |
| POST   | `/chat`   | Stateless chat                  |

### POST `/chat`

Each request is independent (no conversation memory between calls).

**Request body:**

```json
{
  "message": "What are the three laws of robotics?",
  "system_prompt": "You are a concise assistant."
}
```

**Response:**

```json
{
  "reply": "Protect humans, obey orders, self-preserve."
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello agent!"}'
```

### GET `/health`

```json
{"status": "ok", "provider": "anthropic"}
```

## Dependencies

- `openai` — Azure OpenAI SDK
- `anthropic` — Anthropic SDK
- `python-dotenv` — loads `.env` into environment variables
- `fastapi` + `uvicorn` — web framework and ASGI server
- `pandas`, `requests` — utility libraries
