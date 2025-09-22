# src/agentics/llm/clients.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI

Message = Dict[str, str]

@dataclass
class LLMResult:
    model: str
    content: str
    raw: Dict[str, Any]

def _first_nonempty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v:
            return v
    return None

# ---- OpenAI ----
class OpenAIClient:
    """
    Thin wrapper around OpenAI Python SDK (chat.completions).
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.client = OpenAI(
            api_key=_first_nonempty(api_key, os.getenv("OPENAI_API_KEY")),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")  # optional proxy
        )

    def chat(self, messages: List[Message], model: str = "gpt-4o-mini", **kwargs) -> LLMResult:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        content = resp.choices[0].message.content if resp.choices else ""
        return LLMResult(model=model, content=content, raw=resp.to_dict())


# ---- Grok (xAI) ----
class GrokClient:
    """
    Grok uses an OpenAI-compatible API surface.
    We just point OpenAI SDK to xAI base_url and pass the xAI key.
    Docs: https://docs.x.ai (chat completions).
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        grok_key = _first_nonempty(
            api_key,
            os.getenv("XAI_API_KEY"),
            os.getenv("GROK_API_KEY"),
        )
        self.client = OpenAI(
            api_key=grok_key,
            base_url=base_url or os.getenv("XAI_BASE_URL", "https://api.x.ai")
        )

    def chat(self, messages: List[Message], model: str = "grok-2", **kwargs) -> LLMResult:
        # Grok also supports /v1/chat/completions with OpenAI payload shape.
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        content = resp.choices[0].message.content if resp.choices else ""
        return LLMResult(model=model, content=content, raw=resp.to_dict())
