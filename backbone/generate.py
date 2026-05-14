"""
backbone/generate.py
Higher-level generation helpers shared by all subsystems.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backbone.model import LLaMABackbone #replace with eventually LLaMA 2 Mistral GPT-4 
from typing import List
import torch


def generate_n_responses(
    backbone: LLaMABackbone,
    prompt: str,
    n: int = 4,
    temperature: float = 0.9,
    max_new_tokens: int = 256,
) -> List[str]:
    """Sample `n` independent responses from the backbone."""
    responses = []
    for _ in range(n):
        out = backbone.generate(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
        responses.append(out["text"])
    return responses


def greedy_response(backbone: LLaMABackbone, prompt: str,
                    max_new_tokens: int = 256) -> str:
    out = backbone.generate(
        prompt, max_new_tokens=max_new_tokens, do_sample=False
    )
    return out["text"]


def speculative_candidates(
    backbone: LLaMABackbone,
    context: str,
    k: int = 8,
) -> List[int]:
    """
    Return token-ids of top-k candidates by probability (used in PECCAVI
    tournament sampling).
    """
    dist = backbone.token_distribution(context)
    topk = torch.topk(dist, k)
    return topk.indices.tolist()