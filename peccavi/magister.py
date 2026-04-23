"""
peccavi/magister.py
Agent: Magister – Policy Learning via REINFORCE.
Updates watermark parameter θ to maximise composite reward.
"""

from __future__ import annotations
import torch
from backbone.model import LLaMABackbone
from peccavi.auctor import Auctor, _watermark_score, _context_seed
from typing import List
import logging

logger = logging.getLogger(__name__)


def text_quality_score(text: str) -> float:
    """
    Lightweight proxy for fluency / coherence.
    Uses average token length as a simple heuristic.
    Replace with a proper metric (e.g., perplexity, BERTScore) in production.
    """
    words = text.split()
    if not words:
        return 0.0
    avg_len = sum(len(w) for w in words) / len(words)
    return min(avg_len / 8.0, 1.0)   # normalise


def composite_reward(
    effective_wm_score: float, quality: float,
    lam: float = 0.6, nu: float = 0.4
) -> float:
    """r = λ * S_eff + ν * Q"""
    return lam * effective_wm_score + nu * quality


class Magister:
    def __init__(
        self,
        backbone: LLaMABackbone,
        theta_init: float = 2.0,
        alpha: float = 0.05,
        gamma: float = 0.99,
        lam: float = 0.6,
        nu: float = 0.4,
        secret_key: str = "AIISC-KEY",
    ):
        self.backbone = backbone
        self.theta = theta_init
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.nu = nu
        self.secret_key = secret_key
        self.history: List[float] = []   # reward history for baseline

    def _policy_gradient(
        self, token_ids: List[int], reward: float
    ) -> float:
        """
        Approximate ∇_θ J(θ) = Σ_t ∇_θ log p_w(x_t) * (R - b)
        Here we use a scalar approximation: mean g(x_t, r_t).
        """
        grad = 0.0
        for i, tid in enumerate(token_ids):
            r_t = _context_seed(token_ids[:i], self.secret_key)
            g = _watermark_score(tid, r_t, self.backbone.tokenizer.vocab_size)
            grad += g * self.theta   # ∂/∂θ [θ * g] = g
        return grad / max(len(token_ids), 1)

    def update(
        self,
        generated_text: str,
        effective_wm_score: float,
    ) -> float:
        """
        One REINFORCE update step.
        Returns the new theta value.
        """
        quality = text_quality_score(generated_text)
        reward = composite_reward(
            effective_wm_score, quality, self.lam, self.nu
        )

        # Baseline: running mean of past rewards
        baseline = sum(self.history) / len(self.history) if self.history else 0.0
        self.history.append(reward)

        token_ids = self.backbone.tokenizer.encode(generated_text)
        grad = self._policy_gradient(token_ids, reward)
        advantage = reward - baseline

        self.theta += self.alpha * grad * advantage
        self.theta = max(0.1, min(self.theta, 10.0))   # clamp

        logger.debug(
            f"θ updated: {self.theta:.4f} | reward: {reward:.4f} | "
            f"advantage: {advantage:.4f}"
        )
        return self.theta