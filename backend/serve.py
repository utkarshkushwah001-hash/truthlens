"""Run TruthLens behind Waitress, a Windows-compatible WSGI server."""
from __future__ import annotations

import os

from waitress import serve

if __package__:
    from .app import app
else:  # Support `python backend/serve.py` from the project root.
    from app import app


if __name__ == "__main__":
    host = os.getenv("TRUTHLENS_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    serve(app, host=host, port=port)

