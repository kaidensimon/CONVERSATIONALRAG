FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# System build tools for any optional native deps (kept minimal).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source into the image.
COPY . /app

# Install Python dependencies used across the project.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        websockets \
        httpx \
        qdrant-client \
        openai \
        python-dotenv \
        llama-index \
        llama-index-readers-file \
        langchain-openai \
        speechmatics-python \
        speechmatics-tts \
        inngest \
        flask \
        flask-sockets \
        markupsafe==2.0.0

# Install ngrok v3.
RUN curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz \
    | tar -xz -C /usr/local/bin

# Run uvicorn and expose it via ngrok. Requires NGROK_AUTHTOKEN in env.
EXPOSE 5000
CMD ["sh", "-c", "uvicorn RAG.server:app --host 0.0.0.0 --port 5000 & ngrok http 5000"]
