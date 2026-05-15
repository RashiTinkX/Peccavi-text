"""
peccavi/magister.py
Agent: Magister – Policy Learning via REINFORCE.
Updates watermark parameter θ to maximise composite reward.
"""

from __future__ import annotations
import torch
from backbone.model import LLaMABackbone #replace with eventually Base LM (same as Auctor)
from peccavi.auctor import Auctor, _watermark_score, _context_seed
from typing import List
import logging
from peccavi.constants import SECRET_KEY
from bert_score import score as bert_score

logger = logging.getLogger(__name__)


def text_quality_score(text: str, backbone=None, reference_text: str = None) -> float:
    if backbone is None or not text.strip():
        return 0.5
    
    ppl_score = 0.5  # Default
    
    if backbone.backend == "transformers" and hasattr(backbone, 'model'):
        try:
            import torch, math
            enc = backbone.tokenizer(text, return_tensors="pt").to(backbone.model.device)
            with torch.no_grad():
                loss = backbone.model(**enc, labels=enc["input_ids"]).loss
            ppl = math.exp(loss.item())
            ppl_score = max(0.0, 1.0 - (ppl - 1) / 99)
        except Exception as e:
            logger.warning(f"Failed to compute perplexity: {e}")
            ppl_score = 0.5
    
    if reference_text:
        try:
            P, R, F1 = bert_score([text], [reference_text], lang="en", rescale_with_baseline=True)
            bert_score_val = F1.mean().item()
            return (ppl_score + bert_score_val) / 2  # Average
        except Exception as e:
            logger.warning(f"Failed to compute BERTScore: {e}")
            return ppl_score
    return ppl_score
    
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
        secret_key: str = SECRET_KEY,
    ):
        self.backbone = backbone
        self.theta = theta_init
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.nu = nu
        self.secret_key = secret_key
        self.history: List[float] = []   # reward history for baseline

    def _policy_gradient(self, token_ids: List[int]) -> float:
        """
        Approximate ∇_θ log p_w(x) = Σ_t g(x_t, r_t)
        where g is the watermark signal for each chosen token.
        """
        grad = 0.0
        for i, tid in enumerate(token_ids):
            r_t = _context_seed(token_ids[:i], self.secret_key)
            g = _watermark_score(tid, r_t)
            grad += g
        return grad

    def update(
        self,
        generated_text: str,
        effective_wm_score: float,
        reference_text: str = None,
    ) -> float:
        """
        One REINFORCE-style update step.
        Returns the new theta value.
        reference_text: Original prompt for semantic fidelity check via BERTScore.
        """
        quality = text_quality_score(generated_text, self.backbone, reference_text)
        reward = composite_reward(
            effective_wm_score, quality, self.lam, self.nu
        )

        if hasattr(self.backbone, 'tokenizer'):
            token_ids = self.backbone.tokenizer.encode(generated_text)
        else:
            token_ids = generated_text.split()

        self.history.append(reward)

        grad = self._policy_gradient(token_ids)

        # Fixed baseline at chance level: reward > 0.5 → theta increases
        baseline = 0.5
        advantage = reward - baseline

        self.theta += self.alpha * grad * advantage
        self.theta = max(0.1, min(self.theta, 10.0))

        logger.debug(
            f"θ updated: {self.theta:.4f} | reward: {reward:.4f} | "
            f"advantage: {advantage:.4f}"
        )
        return self.theta