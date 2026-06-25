"""
logging_utils.py

Quiet the chatty third-party libraries (HTTP clients, HF hub, tokenizers) so our
own INFO logs are readable. Every `HTTP Request: GET ...` line you see comes from
the ``httpx`` logger at INFO level; the "unauthenticated requests" notice comes
from ``huggingface_hub``. None of it is an error — it is just transport logging.

Call ``quiet_third_party_logs()`` once at the top of any entry-point script.
"""

from __future__ import annotations

import logging
import os

# Loggers that emit per-request / per-download chatter we don't want.
_NOISY = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "datasets",
    "urllib3",
    "filelock",
    "fsspec",
    "sentence_transformers",
    "openai",
    "groq",
)


def quiet_third_party_logs(level: int = logging.WARNING) -> None:
    """Raise the log level of noisy third-party libraries to ``level``."""
    for name in _NOISY:
        logging.getLogger(name).setLevel(level)
    # Silence the HF "set HF_TOKEN" advisory and tokenizers fork warnings.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
