# Conversational RAG Voice Agent

FastAPI + WebSocket voice agent that streams audio to/from Twilio and AssemblyAI, uses OpenAI for language/embeddings, and stores chunks in Qdrant for retrieval.

## Features
- Real-time Twilio-compatible audio bridge (8 kHz μ-law, 50 ms frames).
- TTS via Speechmatics; STT via AssemblyAI; LLM via OpenAI.
- PDF ingestion to Qdrant with embeddings.
- Dockerized deployment with ngrok tunneling for twilio.

## Requirements
- Python 3.11+ (project uses `pyproject.toml` and `uv.lock`).
- Qdrant running and reachable (default `http://localhost:6333`).
- The following API keys in environment:
  - `ASAI_KEY` (AssemblyAI streaming)
  - `OPENAI_API_KEY`
  - `SPEECHMATICS_API_KEY`
  - `TWILIO_ACCOUNT_SID` (for Twilio websocket auth/context)
  - `NGROK_AUTHTOKEN` 
  - Optional: `QDRANT_URL`, `QDRANT_COLLECTION` to override defaults

## Local Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
pip install -r <(python - <<'PY'
import tomllib, pathlib, sys
data = tomllib.loads(pathlib.Path("uv.lock").read_text())
pkgs = [f"{p['name']}=={p['version']}" for p in data.get("package",[])]
print("\\n".join(pkgs))
PY)
```
Set env vars (see above) in `.env`, then run:
```bash
python -m uvicorn RAG.server:app --host 0.0.0.0 --port 5000
```

In a seperate console, run:
```bash
ngrok http 5000
```

## Docker
Build and run (expects `.env` with keys, no quotes):
```bash
docker build -t conversationalrag .
docker run --env-file .env \
  -p 5000:5000 conversationalrag
```


## PDF Ingestion CLI
CLI lives in `RAG/load_and_upsert_data.py`.
```bash
python RAG/load_and_upsert_data.py /path/to/file.pdf \
  --collection docs3 \
  --qdrant-url http://localhost:6333 \
  --source-id my-doc
```
This loads the PDF, chunks it, embeds with OpenAI, and upserts into Qdrant.

## Runtime Behavior
- WebSocket endpoint: `/media` (Twilio connects here).
- Audio pipeline:
  - Twilio sends 8 kHz μ-law, 50 ms frames.
  - Frames are forwarded to AssemblyAI streaming STT.
  - Detected end-of-turn triggers RAG+LLM:
    - Query embedded chunks in Qdrant.
    - Build context and prompt OpenAI via `AGENTS.agent.Agent`.
    - TTS via Speechmatics; PCM is converted to μ-law and sent back to Twilio.
- Qdrant defaults: collection `docs3`, dim 3072 (OpenAI `text-embedding-3-large`).

## Configuration Notes
- Set `QDRANT_URL` to reach the right host (containerized vs host).
- Ensure `.env` values are unquoted when used with `docker --env-file`.
- Twilio/AAI expect 50–1000 ms audio frames; the code uses 50 ms frames.

## Troubleshooting
- Static/choppy audio: verify QDRANT_URL, network latency, and that you are not double-converting audio. Current pipeline sends 50 ms μ-law frames; avoid downsampling elsewhere.
- AssemblyAI “Input Duration Violation”: ensure frame size is 50 ms (default in code).
- Qdrant “connection refused”: set `QDRANT_URL=http://host.docker.internal:6333` when running in Docker Desktop, or use a shared Docker network if Qdrant is containerized.
- Ngrok auth errors: remove quotes around `NGROK_AUTHTOKEN` or pass via `-e`.

## Key Files
- `RAG/server.py` – FastAPI app, Twilio/AAI bridge.
- `RAG/data_loader.py` – PDF loading, chunking, embedding.
- `RAG/vector_db.py` – Qdrant client wrapper (env-overridable URL/collection).
- `RAG/load_and_upsert_data.py` – CLI for PDF ingestion.
- `AGENTS/agent.py` – LLM orchestration, TTS streaming.
- `AGENTS/tts.py` / `AGENTS/convert_audio.py` – Audio generation and μ-law conversion.
