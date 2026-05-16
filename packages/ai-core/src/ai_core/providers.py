from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import httpx

from ai_core.device import choose_device


@dataclass(frozen=True)
class ReasoningRequest:
    prompt: str
    max_new_tokens: int = 180
    temperature: float = 0.2
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceProvider(Protocol):
    async def generate(self, request: ReasoningRequest) -> str:
        ...


class LocalTransformerProvider:
    def __init__(self, model_name: str, enable_gpu: bool = True) -> None:
        self.model_name = model_name
        self.enable_gpu = enable_gpu
        self.generator: Any | None = None
        self._lock = asyncio.Lock()

    async def generate(self, request: ReasoningRequest) -> str:
        async with self._lock:
            if self.generator is None:
                self.generator = await asyncio.to_thread(self._load_pipeline)
            generator = cast(Callable[..., list[dict[str, Any]]], self.generator)
            outputs = await asyncio.to_thread(
                generator,
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.temperature > 0,
                temperature=max(request.temperature, 1e-5),
            )
        generated = outputs[0].get("generated_text", "")
        return str(generated).strip()

    def _load_pipeline(self) -> Any:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = choose_device(self.enable_gpu)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)

        def generate(
            prompt: str,
            max_new_tokens: int,
            do_sample: bool,
            temperature: float,
        ) -> list[dict[str, str]]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generate_kwargs)
            generated_text = cast(str, tokenizer.decode(output_ids[0], skip_special_tokens=True))
            return [{"generated_text": generated_text}]

        return generate


class CloudInferenceProvider:
    def __init__(self, endpoint: str, api_key: str, timeout_seconds: float = 30.0) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=timeout_seconds)

    async def generate(self, request: ReasoningRequest) -> str:
        response = await self.client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "prompt": request.prompt,
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "metadata": request.metadata,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("text") or payload.get("output") or "").strip()

    async def aclose(self) -> None:
        await self.client.aclose()
