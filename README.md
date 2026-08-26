# 🤖 Gen AI Chatbot

A full-stack AI chatbot built with **FastAPI**, **Streamlit**, **LangChain**, and the **Groq API**. It streams AI responses token-by-token, persists conversation history to a database, and supports multiple named conversations through a ChatGPT-style sidebar.

## Features

- 💬 **AI chat** powered by Groq (`openai/gpt-oss-20b` via `langchain-groq`)
- ⚡ **Streaming responses** — tokens appear as they're generated, not all at once
- 🧠 **Persistent conversation memory** — every message is stored in SQLite via SQLAlchemy, so context carries across turns
- 🗂️ **Multiple conversations** — a sidebar lists past chats, auto-titled from your first message, with the ability to switch between or delete them
- ➕ **New Chat** — start a fresh conversation at any time
- 🛡️ **Error handling** — backend validates input and surfaces clean errors; the UI reports connection/streaming failures instead of crashing
- 🧪 **Tested** — a pytest suite covers the API's chat, history, and conversation-management endpoints (with the LLM stubbed out, so tests don't burn real API calls)

## Project Status

✅ Core chatbot, chat history, and improved UI are complete and working end-to-end.

## Architecture

```
ui/streamlit_app.py   →  Streamlit frontend (chat UI + sidebar)
        │  HTTP (requests, streamed)
        ▼
app/api.py             →  FastAPI backend (routes)
app/services/chat_service.py → chat + conversation logic
app/llm.py              →  Groq LLM client (LangChain)
app/models.py           →  Pydantic request schema + SQLAlchemy models
app/database.py         →  SQLAlchemy engine/session (SQLite)
```

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/sakthi5/gen-ai-chatbot.git
cd gen-ai-chatbot
python -m venv .venv
```

Activate it:

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com/).

### 4. Run the backend

```bash
uvicorn app.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### 5. Run the frontend

In a second terminal (with the venv activated):

```bash
streamlit run ui/streamlit_app.py
```

The chat UI opens at `http://localhost:8501`.

## Running tests

```bash
pip install pytest
pytest test/
```

Tests stub out the Groq LLM call and use an isolated in-memory database, so they run fast and don't need a live API key or network access.

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/chat` | Send a message, stream back the AI's reply |
| `GET` | `/conversations` | List all conversations |
| `GET` | `/conversations/{id}/messages` | Get a conversation's full message history |
| `DELETE` | `/conversations/{id}` | Delete a conversation and its messages |

## Screenshots

_Add a screenshot of the chat UI here, e.g. `docs/screenshot.png` linked as `![Chat UI](docs/screenshot.png)`._

## Roadmap / possible next steps

- User authentication / multi-user support
- Editing or regenerating a previous message
- Deployment (Docker, cloud hosting)
- Rate limiting / usage tracking

## Tech Stack

- Python
- FastAPI
- Streamlit
- LangChain + langchain-groq
- Groq API
- SQLAlchemy + SQLite
- pytest
