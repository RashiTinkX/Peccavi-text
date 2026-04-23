"""
peccavi/custos.py
Agent: Custos – Watermark Detection.
Computes S(x_1:T) and effective score S_eff across paraphrased variants.
"""

from __future__ import annotations
from peccavi.auctor import _watermark_score, _context_seed
from backbone.model import LLaMABackbone
from typing import List
import statistics


class Custos:
    def __init__(self, backbone: LLaMABackbone, secret_key: str = "AIISC-KEY"):
        self.backbone = backbone
        self.secret_key = secret_key

    def watermark_score(self, text: str) -> float:
        """
        S(x_1:T) = mean_t g(x_t, r_t)
        """
        tokenizer = self.backbone.tokenizer
        token_ids = tokenizer.encode(text)
        if not token_ids:
            return 0.0

        scores = []
        for i, tid in enumerate(token_ids):
            r_t = _context_seed(token_ids[:i], self.secret_key)
            scores.append(_watermark_score(tid, r_t, tokenizer.vocab_size))
        return statistics.mean(scores)

    def effective_score(self, paraphrases: List[str]) -> float:
        """
        S_eff = min_i S(x̃^(i)_1:T)  – worst-case across all paraphrases.
        """
        if not paraphrases:
            return 0.0
        return min(self.watermark_score(p) for p in paraphrases)

    def detect(self, text: str, threshold: float = 0.52) -> dict:
        score = self.watermark_score(text)
        return {
            "score": round(score, 4),
            "is_watermarked": score >= threshold,
            "threshold": threshold,
        }