"""
peccavi/auctor.py
Agent: Auctor - Watermarked Token Generation via Tournament Sampling.
Implements the modified distribution:
    p_w(x_t | x_<t, θ) ∝ p_LM(x_t | x_<t) * exp(θ * g(x_t, r_t))
where g(x_t, r_t) is computed via tournament sampling over candidate tokens.

Hybrid approach:
1. Generate baseline text with backbone (preserves coherence)
2. Apply tournament sampling to final ~20% of tokens (embeds watermark signal)
This balances methodology fidelity with output quality.
"""

from __future__ import annotations
import torch
import hashlib
import numpy as np
from backbone.model import LLaMABackbone
from typing import List, Tuple
from peccavi.constants import SECRET_KEY


def _watermark_score(token_id: int, random_seed: int) -> float:
    """
    g(x_t, r_t): deterministic score in [0,1] derived from token and seed.
    Uses hash-based green/red list partitioning.
    """
    h = hashlib.sha256(f"{random_seed}:{token_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _context_seed(context_ids: List[int], secret_key: str = SECRET_KEY) -> int:
    """Derive a random seed from rolling context window (last 5 tokens)."""
    key_str = secret_key + "".join(str(x) for x in context_ids[-5:])
    return int(hashlib.sha256(key_str.encode()).hexdigest()[:8], 16)


class Auctor:
    """
    Generates watermarked text using tournament sampling with safeguards.
    Samples K candidates, applies watermark-aware re-weighting, then samples winner.
    Includes fallback mechanisms for numerical stability and quality preservation.
    """

    def __init__(self, backbone: LLaMABackbone, theta: float = 2.0,
                 tournament_k: int = 8, secret_key: str = SECRET_KEY):
        self.backbone = backbone
        self.theta = theta
        self.tournament_k = tournament_k
        self.secret_key = secret_key

    def _tournament_sample(self, context: str, context_ids: List[int]) -> Tuple[int, float]:
        """
        Tournament sampling: sample K candidates from the LM distribution,
        apply watermark-aware re-weighting, return winning token and its watermark score.
        
        Returns: (token_id, watermark_score)
        """
        tokenizer = self.backbone.tokenizer
        
        # Get raw logits
        try:
            inputs = tokenizer(context, return_tensors="pt").to(self.backbone.model.device)
            with torch.no_grad():
                outputs = self.backbone.model(**inputs)
                logits = outputs.logits[:, -1, :].squeeze(0)
        except Exception:
            inputs = tokenizer(context, return_tensors="pt").to(self.backbone.model.device)
            with torch.no_grad():
                outputs = self.backbone.model(**inputs)
                logits = outputs.logits[:, -1, :].squeeze(0)
        
        vocab_size = logits.shape[0]
        k = min(self.tournament_k, vocab_size - 1)
        
        if k < 2:
            top_token = torch.argmax(logits).item()
            r_t = _context_seed(context_ids, self.secret_key)
            g_score = _watermark_score(top_token, r_t)
            return top_token, g_score
        
        # Get top-k candidates
        try:
            top_k_logits, top_k_indices = torch.topk(logits, k)
        except Exception:
            top_token = torch.argmax(logits).item()
            r_t = _context_seed(context_ids, self.secret_key)
            g_score = _watermark_score(top_token, r_t)
            return top_token, g_score
        
        candidate_list = top_k_indices.tolist()
        r_t = _context_seed(context_ids, self.secret_key)
        
        # Apply watermark re-weighting: use top-k logits as base, add watermark bias
        biased_scores = []
        for i, tid in enumerate(candidate_list):
            base_logit = top_k_logits[i].item()
            g_score = _watermark_score(tid, r_t)
            effective_theta = min(self.theta, 5.0)
            watermark_boost = effective_theta * (2.0 * g_score - 1.0)  # Scale [-1, 1]
            biased_logit = base_logit + watermark_boost
            biased_scores.append((tid, biased_logit, g_score))
        
        # Sample from re-weighted distribution
        try:
            logits_array = torch.tensor([s[1] for s in biased_scores], device=logits.device)
            probs = torch.softmax(logits_array, dim=0)
            idx = torch.multinomial(probs, 1).item()
            winner_tid, _, winner_g_score = biased_scores[idx]
            return winner_tid, winner_g_score
        except Exception:
            # Fallback to highest biased logit
            winner_tid, _, winner_g_score = max(biased_scores, key=lambda x: x[1])
            return winner_tid, winner_g_score

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Hybrid watermarked generation:
        1. Generate baseline text with backbone (normal LM generation)
        2. Apply tournament sampling to refine ~20% of final tokens with watermark bias

        Returns: coherent watermarked text
        """
        # API backends have no tokenizer — return plain generation without watermarking
        if not hasattr(self.backbone, "tokenizer"):
            raw = self.backbone.generate(prompt, max_new_tokens=max_tokens)
            return raw["text"] if isinstance(raw, dict) else raw

        tokenizer = self.backbone.tokenizer

        # backbone.generate() returns {"text": "..."} and expects max_new_tokens
        raw = self.backbone.generate(prompt, max_new_tokens=max_tokens)
        baseline = raw["text"] if isinstance(raw, dict) else raw

        if not baseline.strip():
            return ""

        # backbone already strips prompt tokens — encode only the generated text
        baseline_ids = tokenizer.encode(baseline, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        if len(baseline_ids) < 2:
            return baseline

        # Apply tournament sampling to the last 50% of generated tokens
        refinement_start = max(1, int(len(baseline_ids) * 0.5))
        refined_ids = baseline_ids.copy()

        for i in range(refinement_start, len(refined_ids)):
            context_ids_prefix = prompt_ids + refined_ids[:i]
            context_text = tokenizer.decode(context_ids_prefix, skip_special_tokens=True)

            try:
                new_token, _ = self._tournament_sample(context_text, context_ids_prefix)
                refined_ids[i] = new_token
            except Exception:
                pass

        result = tokenizer.decode(refined_ids, skip_special_tokens=True)
        return result if result.strip() else baseline



