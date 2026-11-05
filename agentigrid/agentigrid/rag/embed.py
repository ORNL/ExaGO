"""Embed text via a local Ollama embedding model (default: nomic-embed-text)."""
from __future__ import annotations

import requests


def embed(
    text: str,
    host: str = "http://localhost:11434",
    model: str = "nomic-embed-text",
    timeout: int = 60,
) -> list[float]:
    """Return the embedding vector for `text` from an Ollama server.

    Args:
        text:  The text to embed.
        host:  Ollama base URL (e.g. http://localhost:11434).
        model: Embedding model name (must be pulled in Ollama).
        timeout: Request timeout in seconds.
    """
    r = requests.post(
        f"{host}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["embedding"]
