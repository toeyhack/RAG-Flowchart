import os
import httpx
import json
from typing import Any, Dict

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_API_URL", "https://api.openrouter.ai/v1/chat/completions")
MODEL_DEFAULT = os.getenv("MODEL_DEFAULT", "qwen3-vl-235b")

class OpenRouterClient:
    def __init__(self, api_key: str = OPENROUTER_KEY, base_url: str = OPENROUTER_URL):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)

    def call_model(self, model_id: str, messages: list, temperature: float = 0.0) -> Dict[str, Any]:
        payload = {
            "model": model_id or MODEL_DEFAULT,
            "messages": messages,
            "temperature": temperature
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = self.client.post(self.base_url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

    def analyze_image_strict_json(self, model_id: str, b64_image: str, prompt: str) -> Dict[str, Any]:
        """
        Send an image along with a prompt. The prompt MUST instruct the model to return strict JSON.
        Cursor should refine this to the exact schema.
        """
        messages = [
            {"role":"system","content":"You are a document analysis model. Return strict JSON only."},
            {"role":"user","content": prompt},
            {"role":"user","content": b64_image}
        ]
        return self.call_model(model_id, messages)
