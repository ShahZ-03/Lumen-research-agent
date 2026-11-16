"""Lazy, serialized local cross-encoder inference for all pipeline entry points."""
from __future__ import annotations

import asyncio
import math
import os
import threading
from time import perf_counter

from .models import ScoredChunk

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")
CROSS_ENCODER_BATCH_SIZE = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "32"))
CROSS_ENCODER_MAX_CHARS = int(os.getenv("CROSS_ENCODER_MAX_CHARS", "1500"))
_CROSS_ENCODER = None
_CROSS_ENCODER_LOADED_MODEL: str | None = None
_MODEL_LOCK = threading.Lock()


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _cross_encoder_score_batch(model_name: str, topic: str, items: list[dict]):
    """Score logits explicitly, applying exactly one sigmoid and timing GPU work."""
    global _CROSS_ENCODER, _CROSS_ENCODER_LOADED_MODEL
    import torch
    from sentence_transformers import CrossEncoder

    if CROSS_ENCODER_BATCH_SIZE < 1 or CROSS_ENCODER_MAX_CHARS < 1:
        raise ValueError("Cross-encoder batch size and character limit must be positive")
    started = perf_counter()
    with _MODEL_LOCK:
        load_started = perf_counter()
        cold = _CROSS_ENCODER is None or _CROSS_ENCODER_LOADED_MODEL != model_name
        if cold:
            _CROSS_ENCODER = CrossEncoder(
                model_name, device="cuda" if torch.cuda.is_available() else "cpu",
            )
            _CROSS_ENCODER_LOADED_MODEL = model_name
        load_ms = (perf_counter() - load_started) * 1000 if cold else 0.0
        model = _CROSS_ENCODER
        device = str(next(model.model.parameters()).device)
        pairs_started = perf_counter()
        pairs = [(topic, str(item.get("text") or "")[:CROSS_ENCODER_MAX_CHARS]) for item in items]
        pair_ms = (perf_counter() - pairs_started) * 1000
        # Use the cached model tokenizer; diagnostic lengths exclude padding.
        token_started = perf_counter()
        encoded = model.tokenizer(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs],
            truncation=True, padding=False, max_length=model.max_length,
        )
        lengths = [len(ids) for ids in encoded["input_ids"]]
        tokenize_ms = (perf_counter() - token_started) * 1000
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        infer_started = perf_counter()
        # Identity avoids the model/library's optional default sigmoid.
        logits = model.predict(
            pairs, batch_size=CROSS_ENCODER_BATCH_SIZE,
            show_progress_bar=False, activation_fn=torch.nn.Identity(),
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        inference_ms = (perf_counter() - infer_started) * 1000
        scores = [_sigmoid(float(value)) for value in logits]
        if len(scores) != len(items) or not all(math.isfinite(value) for value in scores):
            raise RuntimeError("Cross encoder returned invalid scores")
        return scores, {
            "device": device, "cuda_available": torch.cuda.is_available(),
            "model_load_ms": round(load_ms, 2), "pair_creation_ms": round(pair_ms, 2),
            "tokenization_ms": round(tokenize_ms, 2), "inference_ms": round(inference_ms, 2),
            "total_ms": round((perf_counter() - started) * 1000, 2),
            "avg_tokens": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "max_tokens": max(lengths, default=0), "min_tokens": min(lengths, default=0),
            "batch_size": CROSS_ENCODER_BATCH_SIZE, "score_transform": "sigmoid_of_logit",
        }


async def score_chunks_with_cross_encoder(items: list[dict], topic: str):
    if not items:
        return [], CROSS_ENCODER_MODEL, {}
    scores, diagnostics = await asyncio.to_thread(
        _cross_encoder_score_batch, CROSS_ENCODER_MODEL, topic, items,
    )
    scored = [
        ScoredChunk(text=item["text"], score=score, url=item.get("url"), title=item.get("title"))
        for item, score in zip(items, scores)
    ]
    return scored, CROSS_ENCODER_MODEL, diagnostics
