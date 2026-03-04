# Simple Azure OpenAI Agent (Python)

A stateless conversational agent exposed as an HTTP API, powered by Azure OpenAI with API key authentication. No Azure CLI, managed identity, or Agent Framework required — just set two environment variables and run.

## Setup

### 1. Clone the repository

### 2. Create a `.env` file

Only two values are required — the model deployment and API version have sensible defaults and don't need to be changed:

```
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
```

Optional overrides (defaults shown):

```
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AGENT_PORT=8000
```

### 3. Install dependencies

#### Using uv (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

#### Using pip

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

This starts the FastAPI server on `http://localhost:8000` (override with `AGENT_PORT`).

## API

| Method | Path      | Description                         |
|--------|-----------|-------------------------------------|
| GET    | `/health` | Health check                        |
| POST   | `/chat`   | Stateless chat — send a message     |

### POST `/chat`

Each request is independent (no conversation memory between calls). You can optionally supply a `system_prompt` to override the default.

**Request body:**

```json
{
  "message": "what are the laws?",
  "system_prompt": "You are a concise assistant."
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello agent!"}'
```

**Response:**

```json
{
  "reply": "Protect humans, obey, self-preserve."
}
```

## Dependencies

- `openai` — Azure OpenAI SDK
- `python-dotenv` — loads `.env` into environment variables
- `fastapi` — web framework for the HTTP endpoint
- `uvicorn` — ASGI server
