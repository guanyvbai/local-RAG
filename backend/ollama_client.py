"""Thread-safe Ollama client provider.

This module exposes a lazily initialised, module-level singleton for the
``ollama.Client`` so that the underlying HTTP session can be re-used across the
application. A thin wrapper is provided to guard each request with a lock to
avoid concurrency issues observed when multiple threads interacted with the
client simultaneously.
"""
from __future__ import annotations

import threading
from typing import Any, Optional, Iterable

import ollama

import config


class _LockReleasingIterator:
    """Iterator wrapper that releases the provided lock when exhausted."""

    def __init__(self, inner: Iterable, lock: threading.RLock) -> None:
        self._inner = iter(inner)
        self._lock = lock
        self._released = False

    def __iter__(self) -> "_LockReleasingIterator":
        return self

    def __next__(self) -> Any:
        try:
            return next(self._inner)
        except StopIteration:
            self._release()
            raise
        except Exception:
            self._release()
            raise

    def close(self) -> None:
        try:
            close = getattr(self._inner, "close", None)
            if callable(close):
                close()
        finally:
            self._release()

    def _release(self) -> None:
        if not self._released:
            self._released = True
            self._lock.release()

    def __del__(self) -> None:
        self._release()


class ThreadSafeOllamaClient:
    """Thread-safe facade for :class:`ollama.Client`.

    The Ollama Python client is backed by ``requests.Session`` which is not
    inherently thread-safe. This wrapper serialises calls to the underlying
    client to avoid race conditions while still exposing the same API surface
    used throughout the project.
    """

    def __init__(self, host: Optional[str] = None) -> None:
        self._client = ollama.Client(host=host)
        self._lock = threading.RLock()

    def embeddings(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._client.embeddings(*args, **kwargs)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        stream = kwargs.get("stream", False)
        self._lock.acquire()
        try:
            response = self._client.chat(*args, **kwargs)
        except Exception:
            self._lock.release()
            raise

        if stream and hasattr(response, "__iter__"):
            return _LockReleasingIterator(response, self._lock)

        self._lock.release()
        return response

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        stream = kwargs.get("stream", False)
        self._lock.acquire()
        try:
            response = self._client.generate(*args, **kwargs)
        except Exception:
            self._lock.release()
            raise

        if stream and hasattr(response, "__iter__"):
            return _LockReleasingIterator(response, self._lock)

        self._lock.release()
        return response

    def __getattr__(self, item: str) -> Any:
        return getattr(self._client, item)


_client_instance: Optional[ThreadSafeOllamaClient] = None
_instance_lock = threading.Lock()


def get_ollama_client() -> ThreadSafeOllamaClient:
    """Return the process-wide Ollama client instance."""
    global _client_instance
    if _client_instance is None:
        with _instance_lock:
            if _client_instance is None:
                _client_instance = ThreadSafeOllamaClient(host=config.OLLAMA_BASE_URL)
    return _client_instance

