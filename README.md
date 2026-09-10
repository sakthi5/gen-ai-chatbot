# 🤖 Gen AI Chatbot

A full-stack AI chatbot built with **FastAPI**, **Streamlit**, **LangChain**, and the **Groq API**. It streams AI responses token-by-token, persists conversation history to a database, and supports multiple named conversations through a ChatGPT-style sidebar.

## Features

- 💬 **AI chat** powered by Groq (`openai/gpt-oss-20b` via `langchain-groq`)
- ⚡ **Streaming responses** — tokens appear as they're generated, not all at once
- 🧠 **Persistent conversation memory** — every message is stored in SQLite via SQLAlchemy, so context carries across turns
- 🗂️ **Multiple conversations** — a sidebar lists past chats, auto-titled from your first message, with the ability to switch between or delete them
- ➕ **New Chat** — start a fresh conversation at any time
- 🛡️ **Error handling** — backend validates input and surfaces clean errors; the UI reports connection/streaming failures instead of crashing
- 📎 **Document Q&A** — attach a PDF, TXT, or DOCX file to a conversation and ask questions about it; the extracted text is added to that conversation's context
- 🖼️ **Image understanding (vision)** — attach a PNG/JPG/WEBP and ask about it; routes to a vision-capable Groq model, and the image stays in context for follow-up questions
- 🎨 **Image generation** — describe an image in a prompt and get one generated (via Pollinations.ai, free/keyless), with a download button
- ⬇️ **Document export** — download any conversation's transcript as DOCX or PDF
- 🧪 **Tested** — a pytest suite covers the API's chat, history, conversation-management, document, image, and export endpoints (with the LLM and image-gen API stubbed out, so tests don't burn real API calls or hit the network)

## Project Status

✅ Core chatbot, chat history, and improved UI are complete and working end-to-end.

## Architecture

```
ui/streamlit_app.py   →  Streamlit frontend (chat UI + sidebar)
        │  HTTP (requests, streamed)
        ▼
app/api.py             →  FastAPI backend (routes)
app/services/chat_service.py           → chat + conversation logic, vision routing
app/services/document_service.py       → PDF/TXT/DOCX text extraction
app/services/image_gen_service.py      → text-to-image via Pollinations.ai
app/services/document_export_service.py → DOCX/PDF transcript export
app/llm.py              →  Groq LLM clients (text + vision, via LangChain)
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
uvicorn app.api:app --reload --port 8020
```

The API will be available at `http://127.0.0.1:8020`.

> Port 8020 was picked to avoid colliding with the very common default of
> 8000 (used by plenty of other FastAPI/uvicorn projects). If 8020 is also
> taken on your machine, run on a different port and point the frontend at
> it: `set CHATBOT_API_URL=http://127.0.0.1:<port>` (Windows) or
> `export CHATBOT_API_URL=http://127.0.0.1:<port>` (macOS/Linux) before
> starting Streamlit.

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
| `POST` | `/conversations` | Create a new, empty conversation |
| `POST` | `/conversations/{id}/documents` | Upload a PDF/TXT/DOCX file to attach to a conversation |
| `GET` | `/conversations/{id}/documents` | List documents attached to a conversation |
| `POST` | `/generate-image` | Generate an image from a text prompt |
| `GET` | `/conversations/{id}/export` | Download the conversation transcript (`?format=docx` or `?format=pdf`) |
| `DELETE` | `/conversations/{id}` | Delete a conversation and its messages/documents/images |

### Document Q&A

Attach a file via the attach icon in the chat input. Its text is extracted
and added to that conversation's context — every question you ask afterward
can reference it, since the document is resent in full on *every* message
in that conversation, not just once. This is the "whole document in
context" approach (not chunked retrieval/RAG), so it works best for
short-to-medium documents; longer ones are truncated (see
`MAX_DOCUMENT_CHARACTERS` in `app/services/document_service.py`, currently
8,000 characters) to leave room for the system prompt, conversation
history, and the model's answer within a single request.

That cap is set conservatively because Groq's free/on-demand tier has a
fairly low per-minute token budget (e.g. 8,000 TPM for `openai/gpt-oss-20b`
at the time this was tuned) — and since the document resends on every
message, a large document eats into that budget repeatedly, not once. If
you're on a paid Groq tier with a higher limit, `MAX_DOCUMENT_CHARACTERS`
can safely be raised. A request that's still too large fails with a clear
in-chat message (rather than a raw API error) suggesting a shorter document
or a new conversation.

### Image understanding (vision)

Attach a PNG/JPG/WEBP the same way as a document, and ask about it. This
routes that turn to a separate vision-capable Groq model (`qwen/qwen3.6-27b`
in `app/llm.py`) instead of the regular text model — the main model
(`openai/gpt-oss-20b`) is text-only. The image is stored (base64, in SQLite)
tied to the specific message it was sent with, and gets replayed in that
turn whenever conversation history is rebuilt, so follow-up questions about
an image from a few messages back still work. Capped at 8MB per image
client-side to keep requests fast, since images are sent inline as base64
in the `/chat` JSON body rather than as a separate upload.

### Image generation

The "🎨 Generate an image" panel above the chat box calls
[Pollinations.ai](https://github.com/pollinations/pollinations) — a free
text-to-image API that needs no signup or API key for this project's usage
level. The prompt and resulting image are saved as a normal exchange in the
conversation (with a download button), so it shows up in history like any
other turn.

You don't have to use the dedicated panel — typing something like *"can you
generate a dog image?"* or *"draw me a sunset"* directly into the normal
chat box works too. `POST /chat` checks the message against a keyword
heuristic (`looks_like_image_request` in `app/services/image_gen_service.py`)
before deciding whether to reply normally or generate an image; if image
generation fails for any reason, it falls back to a normal chat reply
instead of erroring out. This is a simple heuristic, not real intent
understanding, so a genuinely informational question like *"how do
diffusion models generate images"* could occasionally misfire — an accepted
tradeoff for a free, instant check instead of spending an extra LLM call on
every message just to classify intent.

### Document export

The "⬇️ Export conversation" panel in the sidebar renders the full
transcript as a DOCX (via `python-docx`) or PDF (via `fpdf2`) and offers it
as a download. `<think>...</think>` reasoning blocks the vision model emits
are stripped from both the export and the conversation itself before saving.

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
- Groq API (text model + vision model)
- Pollinations.ai (image generation)
- SQLAlchemy + SQLite
- python-docx, pypdf, fpdf2 (document read/export)
- pytest
