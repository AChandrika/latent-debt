"""LLM agent that plays through a chat loop.

Works with any OpenAI-compatible endpoint, including a local Ollama server:

    export LLM_BASE_URL=http://localhost:11434/v1   # Ollama
    export LLM_MODEL=qwen2.5:7b
    export LLM_API_KEY=ollama                       # any string for Ollama

Requires `pip install openai`.
"""

from __future__ import annotations

import os
import random

from .text_wrapper import SYSTEM_PROMPT, parse_action, render


class LLMAgent:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, history_days: int = 4, temperature: float = 0.2,
                 max_tokens: int = 120):
        from openai import OpenAI   # imported here so baselines run without the package

        self.client = OpenAI(base_url=base_url or os.getenv("LLM_BASE_URL"),
                             api_key=api_key or os.getenv("LLM_API_KEY", "none"))
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:7b")
        self.history_days = history_days
        self.temperature = temperature
        self.max_tokens = max_tokens     # short replies are the biggest speed-up on a laptop
        self.reset()

    def reset(self) -> None:
        self.messages: list[dict] = []
        self.transcript: list[dict] = []   # full record for evals: obs, reply, action

    def __call__(self, obs: dict, rng: random.Random) -> str:
        self.messages.append({"role": "user", "content": render(obs)})
        window = self.messages[-2 * self.history_days:]   # keep context small and bounded
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *window],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        reply = resp.choices[0].message.content or ""
        action = parse_action(reply) or "ship_feature"     # unparseable reply -> count as default
        self.messages.append({"role": "assistant", "content": reply})
        self.transcript.append({"day": obs["day"], "reply": reply, "action": action,
                                "parsed": parse_action(reply) is not None})
        return action
