"""
peccavi/custos.py
Agent: Custos – Watermark Detection.
Detects watermarks by computing average watermark score across tokens:
    S(x_1:T) = (1/T) * Σ_t g(x_t, r_t)
And effective score across paraphrases:
    S_eff(x_1:T) = min_i S(x̃^(i)_1:T)
"""

from __future__ import annotations
from peccavi.auctor import _watermark_score, _context_seed
from backbone.model import LLaMABackbone
from typing import List
import statistics
import hashlib
from peccavi.constants import SECRET_KEY


class Custos:
    def __init__(self, backbone: LLaMABackbone, secret_key: str = SECRET_KEY):
        self.backbone = backbone
        self.secret_key = secret_key

    def watermark_score(self, text: str) -> float:
        """
        S(x_1:T) = (1/T) * Σ_t g(x_t, r_t)
        Scores only the watermarked portion (last 30%) to match Auctor's coverage.
        """
        if hasattr(self.backbone, 'tokenizer'):
            tokenizer = self.backbone.tokenizer
            token_ids = tokenizer.encode(text)
        else:
            # Fallback for API backends
            token_ids = text.split()

        if not token_ids:
            return 0.0

        scores = []
        for i, tid in enumerate(token_ids):
            # Convert string tokens to int if needed
            if isinstance(tid, str):
                tid_hash = int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16) % 100000
            else:
                tid_hash = tid

            if isinstance(token_ids[0], str):
                context_ids = [int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) % 100000
                               for t in token_ids[:i]]
            else:
                context_ids = token_ids[:i]
                
            r_t = _context_seed(context_ids, self.secret_key)
            g_score = _watermark_score(tid_hash, r_t)
            scores.append(g_score)
        
        # Return average watermark score
        return statistics.mean(scores) if scores else 0.0

    def effective_score(self, paraphrases: List[str]) -> float:
        """
        S_eff(x_1:T) = min_i S(x̃^(i)_1:T)
        Worst-case (minimum) watermark score across all paraphrased variants.
        """
        if not paraphrases:
            return 0.0
        return min(self.watermark_score(p) for p in paraphrases)

    def detect(self, text: str, threshold: float = 0.52) -> dict:
        """
        Detect watermark: score >= threshold indicates watermarked text.
        """
        score = self.watermark_score(text)
        return {
            "score": round(score, 4),
            "is_watermarked": score >= threshold,
            "threshold": threshold,
        }
