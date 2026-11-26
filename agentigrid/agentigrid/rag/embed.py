"""Embed text via a local Ollama embedding model (default: nomic-embed-text)."""
from __future__ import annotations

import requests


def embed(
    text: str,
    host: str = "http://localhost:11434",
    model: str = "nomic-embed-text",
    timeout: float = 60,
    connect_timeout: float = 5,
) -> list[float]:
    """Return the embedding vector for `text` from an Ollama server.

    Args:
        text:  The text to embed.
        host:  Ollama base URL (e.g. http://localhost:11434).
        model: Embedding model name (must be pulled in Ollama).
        timeout: Read timeout in seconds (the first call may load the model).
        connect_timeout: Connect timeout in seconds. Kept short so an unreachable
            server (e.g. a firewall that drops packets) fails fast instead of
            waiting the full read timeout.
    """
    r = requests.post(
        f"{host}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=(connect_timeout, timeout),
    )
    r.raise_for_status()
    return r.json()["embedding"]
