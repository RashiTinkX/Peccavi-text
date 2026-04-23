"""
peccavi/auctor.py
Agent: Auctor – Watermarked Token Generation via Tournament Sampling.
Implements the modified distribution:
    p_w(x_t | x_<t, θ) ∝ p_LM(x_t | x_<t) * exp(θ * g(x_t, r_t))
"""

from __future__ import annotations
import torch
import hashlib
from backbone.model import LLaMABackbone
from backbone.generate import speculative_candidates
from typing import List


def _watermark_score(token_id: int, random_seed: int, vocab_size: int) -> float:
    """
    g(x_t, r_t): deterministic score in [0,1] derived from token and seed.
    Uses a hash-based green/red list partition.
    """
    h = hashlib.sha256(f"{random_seed}:{token_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF  # normalise to [0,1]


def _context_seed(context_ids: List[int], secret_key: str = "AIISC-KEY") -> int:
    """Derive a random seed from rolling context window."""
    key_str = secret_key + "".join(str(x) for x in context_ids[-5:])
    return int(hashlib.sha256(key_str.encode()).hexdigest()[:8], 16)


class Auctor:
    """
    Generates watermarked text token-by-token using tournament sampling
    during speculative decoding.
    """

    def __init__(self, backbone: LLaMABackbone, theta: float = 2.0,
                 tournament_k: int = 8, secret_key: str = "AIISC-KEY"):
        self.backbone = backbone
        self.theta = theta
        self.tournament_k = tournament_k
        self.secret_key = secret_key

    def _tournament_sample(
        self, context: str, context_ids: List[int]
    ) -> int:
        """
        Sample top-k candidates, re-weight by watermark score, pick winner.
        Returns winning token_id.
        """
        # Step 1: get top-k token ids from backbone
        candidates = speculative_candidates(
            self.backbone, context, k=self.tournament_k
        )
        # Step 2: get base probabilities
        dist = self.backbone.token_distribution(context)
        r_t = _context_seed(context_ids, self.secret_key)

        # Step 3: apply watermark bias
        scores = []
        for tid in candidates:
            base_prob = dist[tid].item()
            g = _watermark_score(tid, r_t, dist.shape[0])
            biased = base_prob * torch.exp(torch.tensor(self.theta * g)).item()
            scores.append((tid, biased))

        # Step 4: normalise and sample
        total = sum(s for _, s in scores)
        probs = torch.tensor([s / total for _, s in scores])
        winner_idx = torch.multinomial(probs, 1).item()
        return scores[winner_idx][0]

    # 
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        tokenizer = self.backbone.tokenizer
        context = prompt
        context_ids = tokenizer.encode(prompt)
        generated_ids: List[int] = []

        for _ in range(max_tokens):
            token_id = self._tournament_sample(context, context_ids)
            token_str = tokenizer.decode([token_id])
            if token_id == tokenizer.eos_token_id:
                break
            generated_ids.append(token_id)
            context_ids.append(token_id)
            context += token_str

        return tokenizer.decode(generated_ids, skip_special_tokens=True)